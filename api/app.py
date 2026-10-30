"""
Backend REST API Application.
=============================

Flask application providing REST endpoints for 3D MRI Brain Tumor Segmentation:
  - GET  /api/health            : Health check and device status
  - POST /api/upload            : Upload and validate raw MRI NIfTI files
  - POST /api/segment           : Run segmentation with explicit tumor type
  - GET  /api/results/<req_id>  : Retrieve segmentation result and volumetrics
  - GET  /api/mask/<req_id>     : Download 3D NIfTI segmentation mask

Connects directly to the verified integration layer:
  - integration.inference_router.InferenceRouter
  - integration.result_schema.SegmentationResult
  - integration.result_schema.TumorType
"""

import datetime
import datetime
import json
import os
import re
import shutil
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import nibabel as nib
import SimpleITK as sitk
from flask import Flask, request, jsonify, send_file, send_from_directory, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename

from integration.inference_router import (
    InferenceRouter,
    run_inference,
    AutomaticClassificationUnavailableError,
    NoTumorDetectedError,
)
from integration.result_schema import TumorType, SegmentationResult
from utils.file_utils import is_segmentation_mask_filename

API_RUNS_DIR = PROJECT_ROOT / "outputs" / "api_runs"
API_RUNS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".nii", ".nii.gz"}
MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB per file limit
REQUEST_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")


def validate_request_id(request_id: Optional[str]) -> str:
    """
    Validates that a request_id adheres to strict alphanumeric/dash/underscore allowlist
    and does not escape the configured API_RUNS_DIR.
    Accepts and normalizes optional .nii.gz or .nii suffixes for direct volume streaming.

    Raises:
        ValueError: If request_id is empty, malformed, or attempts path traversal.
    """
    if not request_id or not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("Request ID cannot be empty.")

    clean_id = request_id.strip()
    for ext in [".nii.gz", ".nii", ".pdf", ".png", ".csv", ".txt", ".json"]:
        if clean_id.lower().endswith(ext):
            clean_id = clean_id[:-len(ext)]
            break

    if not REQUEST_ID_PATTERN.fullmatch(clean_id):
        raise ValueError(
            f"Invalid request ID '{request_id}'. Must be 1-64 characters and contain only "
            "alphanumeric characters, hyphens, and underscores."
        )

    # Path containment check against root directory
    resolved_path = (API_RUNS_DIR / clean_id).resolve()
    resolved_root = API_RUNS_DIR.resolve()
    if not str(resolved_path).startswith(str(resolved_root)) or resolved_path == resolved_root:
        raise ValueError(f"Path traversal detected in request ID '{request_id}'.")

    return clean_id


def sanitize_filename(filename: str) -> str:
    """Sanitizes uploaded filename to prevent directory traversal and path injection."""
    basename = Path(filename).name
    # Remove risky characters, preserve alphanumeric, underscores, hyphens, and dots
    clean_name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", basename)
    if not clean_name or clean_name.startswith("."):
        clean_name = f"scan_{uuid.uuid4().hex[:6]}.nii.gz"
    return clean_name


def validate_nifti_file(file_path: Path) -> None:
    """
    Validates that a file is a readable, non-empty NIfTI image with valid spatial geometry.
    Raises ValueError on invalid or corrupted files.
    """
    if not file_path.exists() or file_path.stat().st_size == 0:
        raise ValueError(f"Uploaded file '{file_path.name}' is empty or unreadable.")

    # Check extension
    suffix = "".join(file_path.suffixes).lower()
    if not any(suffix.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise ValueError(
            f"File '{file_path.name}' has unsupported format '{suffix}'. Supported formats: .nii, .nii.gz"
        )

    # Verify NIfTI header readability using SimpleITK
    try:
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(file_path))
        reader.ReadImageInformation()
        size = reader.GetSize()
        dim = reader.GetDimension()
        if dim != 3 or any(s <= 0 for s in size):
            raise ValueError(
                f"File '{file_path.name}' must be a 3D MRI volume, got dimension {dim} with shape {size}."
            )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"File '{file_path.name}' is corrupted or not a valid NIfTI image: {str(exc)}")


DEFAULT_AFFINE_TOLERANCE: float = 1e-3
DEFAULT_SPACING_TOLERANCE: float = 1e-3


class GeometryValidationError(ValueError):
    """Raised when uploaded multi-modal MRI scans exhibit incompatible spatial geometry."""
    pass


def validate_modality_geometry(
    modality_paths: Dict[str, Path],
    tumor_type: Optional[TumorType] = None,
    affine_tolerance: float = DEFAULT_AFFINE_TOLERANCE,
    spacing_tolerance: float = DEFAULT_SPACING_TOLERANCE,
) -> Dict[str, Any]:
    """
    Validates inter-modality spatial geometry across all required MRI acquisitions
    prior to invoking segmentation inference.

    Validation Rules:
      1. Every file exists, is non-empty, and is a readable NIfTI volume.
      2. Every volume is strictly 3-dimensional with positive spatial dimensions.
      3. All modalities share identical 3D matrix shapes (zero tolerance).
      4. Voxel spacings agree within `spacing_tolerance` (default 1e-3 mm).
      5. Coordinate affine transformation matrices agree within `affine_tolerance` (default 1e-3).

    Raises:
        GeometryValidationError: If any modality is missing, unreadable, non-3D, or spatially mismatched.

    Returns:
        Dict summarizing verified spatial geometry and reference affine.
    """
    if not modality_paths or not isinstance(modality_paths, dict):
        raise GeometryValidationError("Modality paths dictionary must be non-empty.")

    ref_mod: Optional[str] = None
    ref_file: Optional[str] = None
    ref_shape: Optional[Tuple[int, ...]] = None
    ref_zooms: Optional[Tuple[float, ...]] = None
    ref_aff: Optional[np.ndarray] = None
    verified_metadata: Dict[str, Any] = {}

    # Order keys deterministically by canonical sequence priority
    canonical_priority = ["T1", "t1n", "T1c", "t1c", "T2", "t2w", "FLAIR", "t2f"]
    sorted_keys = [k for k in canonical_priority if k in modality_paths] + [
        k for k in sorted(modality_paths.keys()) if k not in canonical_priority
    ]

    for mod_name in sorted_keys:
        file_path = modality_paths[mod_name]
        p = Path(file_path)
        if not p.exists():
            raise GeometryValidationError(f"Modality '{mod_name}' file not found: {p.name}")
        if p.stat().st_size == 0:
            raise GeometryValidationError(f"Modality '{mod_name}' file '{p.name}' is empty (0 bytes).")

        try:
            img = nib.load(str(p))
            hdr = img.header
            shape = tuple(int(x) for x in hdr.get_data_shape())
            zooms = tuple(float(round(z, 4)) for z in hdr.get_zooms()[:3])
            aff = np.array(img.affine, dtype=np.float64)
        except Exception as exc:
            raise GeometryValidationError(
                f"Modality '{mod_name}' file '{p.name}' is corrupted or not a valid NIfTI image: {str(exc)}"
            )

        # Validate 3D dimensionality
        if len(shape) != 3 or any(s <= 0 for s in shape):
            raise GeometryValidationError(
                f"Modality '{mod_name}' file '{p.name}' must be a 3D volume with positive dimensions, got shape {shape}."
            )

        if ref_shape is None:
            ref_mod = mod_name
            ref_file = p.name
            ref_shape = shape
            ref_zooms = zooms
            ref_aff = aff
        else:
            # 1. Shape compatibility
            if shape != ref_shape:
                raise GeometryValidationError(
                    f"Spatial shape mismatch for modality '{mod_name}' ('{p.name}'): "
                    f"got shape {shape}, expected {ref_shape} matching '{ref_mod}' ('{ref_file}'). "
                    "All input sequences must share identical voxel dimensions."
                )

            # 2. Voxel spacing compatibility
            if not np.allclose(zooms, ref_zooms, atol=spacing_tolerance):
                diff = float(np.max(np.abs(np.array(zooms) - np.array(ref_zooms))))
                raise GeometryValidationError(
                    f"Voxel spacing mismatch for modality '{mod_name}' ('{p.name}'): "
                    f"got spacing {zooms}, expected {ref_zooms} matching '{ref_mod}' ('{ref_file}') "
                    f"(max difference {diff:.6f} exceeds tolerance {spacing_tolerance}). "
                    "All input sequences must share identical voxel resolution."
                )

            # 3. Coordinate affine compatibility
            if not np.allclose(aff, ref_aff, atol=affine_tolerance):
                diff = float(np.max(np.abs(aff - ref_aff)))
                raise GeometryValidationError(
                    f"Spatial affine mismatch for modality '{mod_name}' ('{p.name}') relative to '{ref_mod}' ('{ref_file}'): "
                    f"max absolute difference {diff:.6f} exceeds tolerance {affine_tolerance}. "
                    "Scans are not aligned in the same physical coordinate frame."
                )

        verified_metadata[mod_name] = {
            "file": p.name,
            "shape": list(shape),
            "voxel_spacing": list(zooms),
            "affine": [[float(round(c, 6)) for c in row] for row in aff],
        }

    return {
        "reference_modality": ref_mod,
        "reference_shape": list(ref_shape) if ref_shape else None,
        "reference_spacing": list(ref_zooms) if ref_zooms else None,
        "modalities": verified_metadata,
        "affine_tolerance": affine_tolerance,
        "spacing_tolerance": spacing_tolerance,
    }


def detect_modality_from_filename(filename: str) -> Optional[str]:
    """Infers modality key from standard naming conventions (e.g. t1, t1c, t1ce, t2, flair).
    Returns None if the file is a segmentation mask or unidentifiable.
    """
    if is_segmentation_mask_filename(filename):
        return None
    fn_lower = filename.lower()
    if "t1c" in fn_lower or "t1ce" in fn_lower or "t1_ce" in fn_lower or "t1-ce" in fn_lower:
        return "T1c"
    elif "flair" in fn_lower or "t2f" in fn_lower:
        return "FLAIR"
    elif "t2" in fn_lower or "t2w" in fn_lower:
        return "T2"
    elif "t1" in fn_lower or "t1n" in fn_lower or "t1w" in fn_lower:
        return "T1"
    return None


def normalize_model_name(tumor_type: Union[TumorType, str, None]) -> Optional[str]:
    """Normalizes tumor type or model name into canonical key: 'glioma' or 'meningioma'.
    Returns None if unrecognized.
    """
    if not tumor_type:
        return None
    val = tumor_type.value.lower() if isinstance(tumor_type, TumorType) else str(tumor_type).lower().strip()
    if val in ("glioma", "glioma_model"):
        return "glioma"
    if val in ("meningioma", "meningioma_multimodal", "multimodal_meningioma", "meningioma_4channel", "meningioma_model"):
        return "meningioma"
    return None


def _atomic_write_json(file_path: Path, data: Dict[str, Any]) -> None:
    """Writes a dictionary as formatted JSON atomically using a temp file in the same directory."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temp_file = file_path.parent / f"{file_path.name}.tmp.{os.getpid()}_{uuid.uuid4().hex[:8]}"
    try:
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_file, file_path)
    except Exception:
        if temp_file.exists():
            try:
                temp_file.unlink()
            except Exception:
                pass
        raise


def create_app(router: Optional[InferenceRouter] = None) -> Flask:
    """Application factory for Flask backend."""
    app = Flask(__name__)
    app.json.default = str
    CORS(app)

    # Attach router instance
    router_inst = router or InferenceRouter()
    app.config["ROUTER"] = router_inst
    app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_BYTES

    @app.route("/api/health", methods=["GET"])
    def health_check():
        """Returns API health status, compute device, and supported tumor types."""
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        r = app.config.get("ROUTER")
        auto_avail = False
        if hasattr(r, "classifier_service"):
            try:
                st = r.classifier_service.get_status()
                if isinstance(st, dict):
                    auto_avail = bool(st.get("automatic_classification_available", False))
            except Exception:
                auto_avail = False

        return jsonify({
            "status": "healthy",
            "version": "1.0.0",
            "device": device,
            "device_name": device_name,
            "supported_tumor_types": ["glioma", "meningioma", "auto"],
            "supported_formats": [".nii", ".nii.gz"],
            "automatic_classification_available": auto_avail,
        }), 200

    @app.route("/api/classifier/status", methods=["GET"])
    def get_classifier_status():
        """Returns availability and status of 3D MRI automatic classifiers."""
        r = app.config.get("ROUTER")
        cs = getattr(r, "classifier_service", None)
        if cs is None or not hasattr(cs, "get_status"):
            cs = ClassifierService()
        st = cs.get_status()
        if isinstance(st, dict):
            return jsonify(st), 200
        return jsonify(ClassifierService().get_status()), 200

    @app.route("/api/classify", methods=["POST"])
    def classify_uploaded_scans():
        """
        Assesses tumor presence and tumor type for uploaded scans of a given request_id.
        Returns the classification assessment or an explicit 'automatic_classification_not_available'
        status when trained weights are missing.
        """
        data = request.get_json(silent=True) or {}
        raw_req_id = data.get("request_id")
        if not raw_req_id:
            return jsonify({"detail": "Missing required field 'request_id'."}), 400

        try:
            req_id = validate_request_id(raw_req_id)
        except ValueError as val_err:
            return jsonify({"detail": str(val_err)}), 400

        req_dir = API_RUNS_DIR / req_id
        upload_dir = req_dir / "uploads"
        if not upload_dir.exists():
            return jsonify({"detail": f"Request ID '{req_id}' not found. Please upload scans first."}), 404

        uploaded_files = list(upload_dir.glob("*.nii*"))
        if not uploaded_files:
            return jsonify({"detail": f"No NIfTI scans found for request '{req_id}'."}), 400

        mod_map: Dict[str, Path] = {}
        modality_map = data.get("modality_map")
        if not modality_map:
            saved_map_path = req_dir / "modality_map.json"
            if saved_map_path.exists():
                try:
                    with open(saved_map_path, "r", encoding="utf-8") as f:
                        modality_map = json.load(f)
                except Exception:
                    modality_map = None

        if modality_map and isinstance(modality_map, dict):
            for mod_key, fn in modality_map.items():
                if fn and isinstance(fn, str):
                    target = upload_dir / fn
                    if target.exists():
                        k_upper = str(mod_key).strip().upper()
                        if k_upper in ("T1", "T1N", "T1W"):
                            canon = "T1"
                        elif k_upper in ("T1C", "T1CE", "T1-CE", "T1_CE"):
                            canon = "T1c"
                        elif k_upper in ("T2", "T2W"):
                            canon = "T2"
                        elif k_upper in ("FLAIR", "T2F", "T2-FLAIR", "T2_FLAIR"):
                            canon = "FLAIR"
                        else:
                            canon = str(mod_key)
                        mod_map[canon] = target

        if not mod_map:
            for f in uploaded_files:
                m = detect_modality_from_filename(f.name)
                if m and m not in mod_map:
                    mod_map[m] = f

        r = app.config.get("ROUTER")
        cs = getattr(r, "classifier_service", None)
        if cs is None or not hasattr(cs, "classify"):
            cs = ClassifierService()
        assessment = cs.classify(mod_map)
        if not isinstance(assessment, dict):
            assessment = ClassifierService().classify(mod_map)
        assessment["request_id"] = req_id
        # Atomically persist classification.json for this request
        _atomic_write_json(req_dir / "classification.json", assessment)
        return jsonify(assessment), 200

    @app.route("/api/upload", methods=["POST"])
    def upload_scans():
        """
        Accepts and validates one or more 3D MRI NIfTI files (.nii, .nii.gz).
        Stores files in an isolated request directory under outputs/api_runs/<request_id>/uploads/.
        """
        uploaded_files_list = []
        file_to_slot_map = {}
        for key in request.files:
            flist = request.files.getlist(key)
            for f in flist:
                if f and f.filename:
                    uploaded_files_list.append(f)
                    k_lower = key.lower()
                    if k_lower in ("t1", "t1n", "slot_t1", "slot_t1n"):
                        file_to_slot_map[f.filename] = "T1"
                    elif k_lower in ("t1c", "t1ce", "slot_t1c", "slot_t1ce"):
                        file_to_slot_map[f.filename] = "T1c"
                    elif k_lower in ("t2", "t2w", "slot_t2", "slot_t2w"):
                        file_to_slot_map[f.filename] = "T2"
                    elif k_lower in ("flair", "t2f", "slot_flair", "slot_t2f"):
                        file_to_slot_map[f.filename] = "FLAIR"

        if not uploaded_files_list or all(f.filename == "" for f in uploaded_files_list):
            return jsonify({"detail": "No files were uploaded."}), 400

        raw_req_id = request.form.get("request_id")
        if raw_req_id:
            try:
                req_id = validate_request_id(raw_req_id)
            except ValueError as val_err:
                return jsonify({"detail": str(val_err)}), 400
        else:
            req_id = f"REQ-{uuid.uuid4().hex[:8].upper()}"

        req_upload_dir = API_RUNS_DIR / req_id / "uploads"
        req_upload_dir.mkdir(parents=True, exist_ok=True)

        uploaded_records = []
        detected_modalities = {}

        for upload_storage in uploaded_files_list:
            if not upload_storage.filename:
                continue

            safe_name = sanitize_filename(upload_storage.filename)
            dest_path = req_upload_dir / safe_name

            upload_storage.save(str(dest_path))

            # Validate NIfTI integrity
            try:
                validate_nifti_file(dest_path)
            except ValueError as val_err:
                if dest_path.exists():
                    dest_path.unlink()
                return jsonify({"detail": str(val_err)}), 400

            inferred_mod = file_to_slot_map.get(upload_storage.filename) or file_to_slot_map.get(safe_name) or detect_modality_from_filename(safe_name)
            if inferred_mod:
                detected_modalities[inferred_mod] = safe_name

            uploaded_records.append({
                "filename": safe_name,
                "size_bytes": dest_path.stat().st_size,
                "inferred_modality": inferred_mod,
            })

        explicit_slot_mapping = False
        explicit_mod_map = {}
        form_mod_map = request.form.get("modality_map")
        if form_mod_map:
            try:
                parsed_mm = json.loads(form_mod_map)
                if isinstance(parsed_mm, dict):
                    explicit_slot_mapping = True
                    for k, v in parsed_mm.items():
                        canon = k
                        k_upper = str(k).strip().upper()
                        if k_upper in ("T1", "T1N", "T1W"):
                            canon = "T1"
                        elif k_upper in ("T1C", "T1CE", "T1-CE", "T1_CE"):
                            canon = "T1c"
                        elif k_upper in ("T2", "T2W"):
                            canon = "T2"
                        elif k_upper in ("FLAIR", "T2F", "T2-FLAIR", "T2_FLAIR"):
                            canon = "FLAIR"
                        explicit_mod_map[canon] = sanitize_filename(v)
            except Exception:
                pass

        if file_to_slot_map:
            explicit_slot_mapping = True
            for fn, mod in file_to_slot_map.items():
                explicit_mod_map[mod] = sanitize_filename(fn)

        if explicit_slot_mapping and explicit_mod_map:
            _atomic_write_json(API_RUNS_DIR / req_id / "modality_map.json", explicit_mod_map)
            detected_modalities.update(explicit_mod_map)

        return jsonify({
            "request_id": req_id,
            "status": "uploaded",
            "files_count": len(uploaded_records),
            "files": uploaded_records,
            "detected_modalities": detected_modalities,
        }), 200
    def _execute_segmentation(data: dict):
        if not data or not isinstance(data, dict):
            return jsonify({"detail": "Request body must be a valid JSON object."}), 400

        raw_req_id = data.get("request_id")
        if raw_req_id is None or raw_req_id == "":
            return jsonify({"detail": "Missing required field 'request_id'."}), 400

        try:
            req_id = validate_request_id(raw_req_id)
        except ValueError as val_err:
            return jsonify({"detail": str(val_err)}), 400

        tumor_type_str = data.get("tumor_type")
        if not tumor_type_str:
            return jsonify({"request_id": req_id, "detail": "Missing required field 'tumor_type'."}), 400

        if "modality_map" in data and data["modality_map"] is not None and not isinstance(data["modality_map"], dict):
            return jsonify({
                "request_id": req_id,
                "detail": "Field 'modality_map' must be a JSON dictionary mapping modality names to filenames."
            }), 400

        req_dir = API_RUNS_DIR / req_id
        upload_dir = req_dir / "uploads"

        if not upload_dir.exists():
            return jsonify({
                "request_id": req_id,
                "detail": f"Request ID '{req_id}' not found. Please upload MRI scans first via /api/upload."
            }), 404

        # Validate tumor type
        try:
            router: InferenceRouter = app.config["ROUTER"]
            tumor_type = router.normalize_tumor_type(tumor_type_str)
        except (ValueError, TypeError) as exc:
            return jsonify({"request_id": req_id, "detail": str(exc)}), 400

        uploaded_files = list(upload_dir.glob("*.nii*"))
        if not uploaded_files:
            return jsonify({"request_id": req_id, "detail": f"No NIfTI scans found for request '{req_id}'."}), 400

        # Disallow segmentation masks / ground truth files from being used as input acquisitions
        mask_files = [f.name for f in uploaded_files if is_segmentation_mask_filename(f.name)]
        if mask_files:
            return jsonify({
                "request_id": req_id,
                "detail": (
                    f"Detected segmentation mask file(s): {mask_files}. "
                    "Segmentation masks (-seg.nii.gz / mask / gtv) cannot be used as input MRI acquisitions. "
                    "Please upload only raw structural MRI sequences (e.g. T1, T1c, T2, FLAIR)."
                )
            }), 400

        modality_map = data.get("modality_map")
        if not modality_map:
            saved_map_path = req_dir / "modality_map.json"
            if saved_map_path.exists():
                try:
                    with open(saved_map_path, "r", encoding="utf-8") as f:
                        modality_map = json.load(f)
                except Exception:
                    modality_map = None
        case_input: Union[Path, Dict[str, Path]]

        if modality_map and isinstance(modality_map, dict):
            for mod_key, fn in modality_map.items():
                if is_segmentation_mask_filename(str(fn)) or is_segmentation_mask_filename(str(mod_key)):
                    return jsonify({
                        "request_id": req_id,
                        "detail": (
                            f"File '{fn}' specified for modality '{mod_key}' is a segmentation mask, not an MRI acquisition sequence. "
                            "Please specify only raw structural acquisitions."
                        )
                    }), 400

        if tumor_type == TumorType.GLIOMA:
            mod_map: Dict[str, Path] = {}
            if modality_map and isinstance(modality_map, dict):
                for mod_key, fn in modality_map.items():
                    if not fn or not isinstance(fn, str):
                        return jsonify({
                            "request_id": req_id,
                            "detail": f"Filename for modality '{mod_key}' must be a non-empty string."
                        }), 400
                    target = upload_dir / fn
                    if not target.exists():
                        return jsonify({
                            "request_id": req_id,
                            "detail": f"File '{fn}' specified for modality '{mod_key}' was not found in uploads for request '{req_id}'."
                        }), 400

                    k_upper = str(mod_key).strip().upper()
                    if k_upper in ("T1", "T1N", "T1W"):
                        canon = "T1"
                    elif k_upper in ("T1C", "T1CE", "T1-CE", "T1_CE"):
                        canon = "T1c"
                    elif k_upper in ("T2", "T2W"):
                        canon = "T2"
                    elif k_upper in ("FLAIR", "T2F", "T2-FLAIR", "T2_FLAIR"):
                        canon = "FLAIR"
                    else:
                        return jsonify({
                            "request_id": req_id,
                            "detail": f"Unexpected modality '{mod_key}' for glioma segmentation. Required modalities: ['FLAIR', 'T1', 'T1c', 'T2']."
                        }), 400

                    if canon in mod_map:
                        return jsonify({
                            "request_id": req_id,
                            "detail": f"Duplicate modality '{canon}' specified in modality_map."
                        }), 400
                    mod_map[canon] = target
            else:
                for f in uploaded_files:
                    m = detect_modality_from_filename(f.name)
                    if m is None:
                        return jsonify({
                            "request_id": req_id,
                            "detail": f"Unrecognized file '{f.name}' for glioma segmentation. Could not infer modality from filename. Please use standard modality naming or provide 'modality_map'."
                        }), 400
                    if m in mod_map:
                        return jsonify({
                            "request_id": req_id,
                            "detail": f"Duplicate modality '{m}' detected: multiple uploaded files ('{mod_map[m].name}', '{f.name}') resolve to '{m}'."
                        }), 400
                    mod_map[m] = f

            missing = [m for m in ["T1", "T1c", "T2", "FLAIR"] if m not in mod_map]
            if missing:
                return jsonify({
                    "request_id": req_id,
                    "detail": f"Glioma segmentation requires 4 modalities (T1, T1c, T2, FLAIR). Missing: {missing}. Uploaded: {[f.name for f in uploaded_files]}"
                }), 400
            case_input = mod_map

        elif tumor_type in (TumorType.MENINGIOMA, TumorType.MENINGIOMA_MULTIMODAL):
            from integration.meningioma_multimodal_service import detect_modality_key, REQUIRED_MODALITIES
            mod_map: Dict[str, Path] = {}
            if modality_map and isinstance(modality_map, dict):
                for mod_key, fn in modality_map.items():
                    if not fn or not isinstance(fn, str):
                        return jsonify({
                            "request_id": req_id,
                            "detail": f"Filename for modality '{mod_key}' must be a non-empty string."
                        }), 400
                    target = upload_dir / fn
                    if not target.exists():
                        return jsonify({
                            "request_id": req_id,
                            "detail": f"File '{fn}' specified for modality '{mod_key}' was not found in uploads for request '{req_id}'."
                        }), 400

                    inferred = detect_modality_key(str(mod_key))
                    if not inferred and str(mod_key).lower().strip() in REQUIRED_MODALITIES:
                        inferred = str(mod_key).lower().strip()
                    if not inferred:
                        k_upper = str(mod_key).upper().strip()
                        if k_upper == "T1":
                            inferred = "t1n"
                        elif k_upper == "T1C":
                            inferred = "t1c"
                        elif k_upper == "T2":
                            inferred = "t2w"
                        elif k_upper == "FLAIR":
                            inferred = "t2f"

                    if not inferred:
                        return jsonify({
                            "request_id": req_id,
                            "detail": f"Unexpected modality '{mod_key}' for meningioma segmentation. Required modalities: ['t1c', 't1n', 't2f', 't2w']."
                        }), 400

                    if inferred in mod_map:
                        return jsonify({
                            "request_id": req_id,
                            "detail": f"Duplicate modality '{inferred}' specified in modality_map."
                        }), 400
                    mod_map[inferred] = target
            else:
                for f in uploaded_files:
                    m = detect_modality_key(f.name)
                    if not m:
                        m_old = detect_modality_from_filename(f.name)
                        if m_old == "T1":
                            m = "t1n"
                        elif m_old == "T1c":
                            m = "t1c"
                        elif m_old == "T2":
                            m = "t2w"
                        elif m_old == "FLAIR":
                            m = "t2f"

                    if m is None:
                        return jsonify({
                            "request_id": req_id,
                            "detail": f"Unrecognized file '{f.name}' for meningioma segmentation. Could not infer modality from filename. Please use standard modality naming or provide 'modality_map'."
                        }), 400

                    if m in mod_map:
                        return jsonify({
                            "request_id": req_id,
                            "detail": f"Duplicate modality '{m}' detected: multiple uploaded files ('{mod_map[m].name}', '{f.name}') resolve to '{m}'."
                        }), 400
                    mod_map[m] = f

            missing = [m for m in REQUIRED_MODALITIES if m not in mod_map]
            if missing:
                return jsonify({
                    "request_id": req_id,
                    "detail": f"Meningioma segmentation requires 4 modalities (t1n, t1c, t2w, t2f). Missing: {missing}. Uploaded: {[f.name for f in uploaded_files]}"
                }), 400
            case_input = mod_map

        elif tumor_type == TumorType.AUTO:
            mod_map: Dict[str, Path] = {}
            if modality_map and isinstance(modality_map, dict):
                for mod_key, fn in modality_map.items():
                    if not fn or not isinstance(fn, str):
                        return jsonify({"request_id": req_id, "detail": f"Filename for modality '{mod_key}' must be a non-empty string."}), 400
                    target = upload_dir / fn
                    if not target.exists():
                        return jsonify({"request_id": req_id, "detail": f"File '{fn}' specified for modality '{mod_key}' was not found in uploads."}), 400
                    k_upper = str(mod_key).strip().upper()
                    if k_upper in ("T1", "T1N", "T1W"):
                        canon = "T1"
                    elif k_upper in ("T1C", "T1CE", "T1-CE", "T1_CE"):
                        canon = "T1c"
                    elif k_upper in ("T2", "T2W"):
                        canon = "T2"
                    elif k_upper in ("FLAIR", "T2F", "T2-FLAIR", "T2_FLAIR"):
                        canon = "FLAIR"
                    else:
                        canon = str(mod_key)
                    mod_map[canon] = target
            else:
                for f in uploaded_files:
                    m = detect_modality_from_filename(f.name)
                    if m and m not in mod_map:
                        mod_map[m] = f
            case_input = mod_map

        # Validate spatial geometry across all required modalities before routing to inference
        try:
            validate_modality_geometry(
                modality_paths=case_input,
                tumor_type=tumor_type,
            )
        except GeometryValidationError as geom_err:
            return jsonify({
                "request_id": req_id,
                "detail": f"MRI geometry validation failed: {str(geom_err)}"
            }), 400

        # Destination mask path
        mask_output_dir = req_dir / "masks"
        mask_output_dir.mkdir(parents=True, exist_ok=True)
        mask_filename = f"{req_id}_{tumor_type.value.lower()}_seg.nii.gz"
        dest_mask_path = mask_output_dir / mask_filename

        try:
            router: InferenceRouter = app.config["ROUTER"]
            result: SegmentationResult = router.route(
                tumor_type=tumor_type,
                input_data=case_input,
                request_id=req_id,
                patient_id=data.get("patient_id"),
                session_id=data.get("session_id"),
                save_mask_path=dest_mask_path,
            )
        except NoTumorDetectedError as no_tumor_err:
            clean_model = normalize_model_name(tumor_type) or "auto"
            payload = {
                "request_id": req_id,
                "status": "no_tumor_detected",
                "tumor_presence": "healthy_control",
                "selected_model": clean_model,
                "confidence": no_tumor_err.details.get("confidence"),
                "presence_probability": no_tumor_err.details.get("presence_probability"),
                "recommendation": no_tumor_err.details.get(
                    "recommendation",
                    "No tumor detected by automated presence detector. Segmentation inference is not indicated for normative scans."
                ),
                "routing_mode": "automatic_classification",
                "disclaimer": "Research prototype only. Not cleared for primary clinical diagnosis or surgical planning.",
                "request": {
                    "request_id": req_id,
                    "tumor_type": "healthy_control",
                    "routing_mode": "automatic_classification",
                    "selected_model": "none",
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                },
                "request_metadata": {
                    "request_id": req_id,
                    "tumor_type": "healthy_control",
                    "routing_mode": "automatic_classification",
                    "selected_model": "none",
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                }
            }
            _atomic_write_json(req_dir / "result.json", payload)
            return jsonify(payload), 200
        except AutomaticClassificationUnavailableError as auto_err:
            auto_status = auto_err.details.get("status") or "automatic_classification_not_available"
            payload = {
                "request_id": req_id,
                "status": auto_status,
                "detail": str(auto_err),
                "blocker": auto_err.details.get("blocker", "Missing trained classifier checkpoints on disk."),
                "available_manual_routes": auto_err.details.get("available_manual_routes", ["glioma", "meningioma"]),
                "recommendation": auto_err.details.get("recommendation", "Please select a tumor model manually ('glioma' or 'meningioma')."),
                "next_steps": auto_err.details.get("next_steps"),
                "confidence": auto_err.details.get("confidence"),
                "routing_mode": "automatic_classification",
                "request": {
                    "request_id": req_id,
                    "tumor_type": "uncertain",
                    "routing_mode": "automatic_classification",
                    "selected_model": "none",
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                },
                "request_metadata": {
                    "request_id": req_id,
                    "tumor_type": "uncertain",
                    "routing_mode": "automatic_classification",
                    "selected_model": "none",
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                }
            }
            _atomic_write_json(req_dir / "result.json", payload)
            return jsonify(payload), 422
        except (ValueError, TypeError, FileNotFoundError) as client_err:
            return jsonify({
                "request_id": req_id,
                "detail": f"Input validation or processing error: {str(client_err)}"
            }), 400
        except Exception as exc:
            return jsonify({
                "request_id": req_id,
                "detail": f"Inference execution failed: {str(exc)}"
            }), 500

        # Determine model string for isolated persistence
        # Use actual executed model from result.request.tumor_type if available (handles auto-routing)
        executed_type = getattr(result.request, "tumor_type", tumor_type)
        canonical_model = normalize_model_name(executed_type) or (
            tumor_type.value.lower() if isinstance(tumor_type, TumorType) else str(tumor_type).lower().strip()
        )
        raw_model_str = executed_type.value.lower() if isinstance(executed_type, TumorType) else str(executed_type).lower().strip()

        res_data = result.to_dict()

        is_auto_requested = (tumor_type == TumorType.AUTO or str(tumor_type_str).lower().strip() == "auto")
        router_inst = app.config.get("ROUTER")
        cs = getattr(router_inst, "classifier_service", None)
        auto_avail = bool(cs and getattr(cs, "is_automatic_classification_available", False))

        if is_auto_requested:
            res_data["request_metadata"]["routing_mode"] = "automatic_classification"
            res_data["request_metadata"]["selected_model"] = canonical_model
            res_data["request_metadata"]["manual_override"] = False
            res_data["request_metadata"]["conflict_detected"] = False
            res_data["request_metadata"]["automatic_classification_available"] = True
            res_data["request_metadata"]["diagnostic_notice"] = (
                f"Segmentation was automatically routed to the {canonical_model.upper()} model based on multi-stage classifier prediction. "
                "Research prototype notice: automated routing does NOT replace histopathological or clinical verification."
            )
        else:
            res_data["request_metadata"]["routing_mode"] = "manual_selection"
            res_data["request_metadata"]["selected_model"] = canonical_model
            res_data["request_metadata"]["manual_override"] = True
            res_data["request_metadata"]["automatic_classification_available"] = auto_avail

            conflict_detected = False
            warning_msg = None
            diag_notice = None

            if auto_avail and isinstance(case_input, dict):
                try:
                    c_eval = cs.classify(case_input)
                    c_status = c_eval.get("status")
                    c_type = c_eval.get("predicted_tumor_type")
                    c_conf = c_eval.get("confidence")
                    if hasattr(c_status, "_mock_name") or not isinstance(c_status, (str, type(None))):
                        c_status = str(c_status)
                    if hasattr(c_type, "_mock_name") or not isinstance(c_type, (str, type(None))):
                        c_type = str(c_type)
                    if hasattr(c_conf, "_mock_name") or not isinstance(c_conf, (int, float, type(None))):
                        try:
                            c_conf = float(c_conf)
                        except Exception:
                            c_conf = str(c_conf)

                    res_data["request_metadata"]["classifier_assessment"] = {
                        "status": c_status,
                        "predicted_tumor_type": c_type,
                        "confidence": c_conf,
                    }

                    if c_status == "classified_healthy":
                        conflict_detected = True
                        warning_msg = (
                            f"MANUAL MODEL OVERRIDE CONFLICT: User manually selected {canonical_model.upper()} segmentation, "
                            f"but automated classifier assessed scan as HEALTHY_CONTROL (confidence: {c_conf if c_conf is not None else 'N/A'}). "
                            f"Segmentation quantifies hyperintensities matching {canonical_model.upper()} model assumptions only and does NOT confirm pathology."
                        )
                    elif c_status == "classified_tumor" and c_type and c_type.lower() != canonical_model.lower():
                        conflict_detected = True
                        warning_msg = (
                            f"MANUAL MODEL OVERRIDE CONFLICT: User manually selected {canonical_model.upper()} segmentation, "
                            f"but automated classifier predicted {c_type.upper()} (confidence: {c_conf if c_conf is not None else 'N/A'}). "
                            f"Segmentation was executed with the {canonical_model.upper()} model per manual override and does NOT confirm {canonical_model.upper()} pathology."
                        )
                    elif c_status == "classified_tumor" and c_type and c_type.lower() == canonical_model.lower():
                        diag_notice = (
                            f"Manual model selection ({canonical_model.upper()}) agrees with automated classifier prediction "
                            f"(confidence: {c_conf if c_conf is not None else 'N/A'}). "
                            "Research prototype notice: does not replace histopathological confirmation."
                        )
                except Exception:
                    pass

            if not warning_msg and not diag_notice:
                diag_notice = (
                    f"Segmentation was performed using the user-selected {canonical_model.upper()} model (manual override). "
                    f"This segmentation quantifies abnormalities matching {canonical_model.upper()} model assumptions, "
                    f"and does NOT constitute an independent histological or clinical diagnosis of {canonical_model}."
                )

            res_data["request_metadata"]["conflict_detected"] = conflict_detected
            if warning_msg:
                res_data["request_metadata"]["warning"] = warning_msg
                res_data["request_metadata"]["diagnostic_notice"] = warning_msg
            else:
                res_data["request_metadata"]["diagnostic_notice"] = diag_notice

        # Keep request and request_metadata synchronized
        res_data["request"] = res_data["request_metadata"]

        # Save model-specific result JSON atomically
        model_result_path = req_dir / f"result_{canonical_model}.json"
        _atomic_write_json(model_result_path, res_data)

        # If executed as meningioma multimodal, also write result_meningioma_multimodal.json
        if raw_model_str == "meningioma_multimodal" or str(tumor_type_str).lower().strip() in ("meningioma_multimodal", "multimodal_meningioma"):
            multimodal_result_path = req_dir / "result_meningioma_multimodal.json"
            _atomic_write_json(multimodal_result_path, res_data)

        # Save latest/active result JSON atomically
        result_json_path = req_dir / "result.json"
        _atomic_write_json(result_json_path, res_data)

        return jsonify(res_data), 200

    @app.route("/api/segment", methods=["POST"])
    def run_segmentation():
        """
        Executes segmentation on previously uploaded scans for an explicit tumor type or auto.
        Validates modality completeness, dispatches to InferenceRouter, saves output mask,
        and returns the common SegmentationResult JSON.
        """
        data = request.get_json(silent=True)
        if not data or not isinstance(data, dict):
            return jsonify({"detail": "Request body must be a valid JSON object."}), 400
        return _execute_segmentation(data)

    @app.route("/api/segment/glioma", methods=["POST"])
    def run_glioma_segmentation():
        """
        Isolated dedicated endpoint for manual Glioma segmentation.
        Dispatches request with tumor_type='glioma'.
        """
        data = request.get_json(silent=True)
        if not data or not isinstance(data, dict):
            return jsonify({"detail": "Request body must be a valid JSON object."}), 400
        data_copy = dict(data)
        data_copy["tumor_type"] = "glioma"
        return _execute_segmentation(data_copy)

    @app.route("/api/segment/meningioma", methods=["POST"])
    def run_meningioma_segmentation():
        """
        Isolated dedicated endpoint for manual Meningioma segmentation.
        Dispatches request with tumor_type='meningioma'.
        """
        data = request.get_json(silent=True)
        if not data or not isinstance(data, dict):
            return jsonify({"detail": "Request body must be a valid JSON object."}), 400
        data_copy = dict(data)
        data_copy["tumor_type"] = "meningioma"
        return _execute_segmentation(data_copy)

    @app.route("/api/segment/meningioma-multimodal", methods=["POST"])
    def segment_meningioma_multimodal():
        """
        Isolated dedicated endpoint for 4-channel Multimodal Meningioma segmentation.
        Dispatches request with tumor_type='meningioma_multimodal'.
        """
        data = request.get_json(silent=True)
        if not data or not isinstance(data, dict):
            return jsonify({"detail": "Request body must be a valid JSON object."}), 400

        data_copy = dict(data)
        data_copy["tumor_type"] = "meningioma_multimodal"
        return _execute_segmentation(data_copy)

    @app.route("/api/results/<request_id>", methods=["GET"])
    def get_results(request_id: str):
        """Retrieves previously generated segmentation result metadata by request ID."""
        try:
            clean_id = validate_request_id(request_id)
        except ValueError as val_err:
            return jsonify({"detail": str(val_err)}), 400

        req_dir = API_RUNS_DIR / clean_id
        if not req_dir.exists():
            return jsonify({"detail": f"Results not found for request ID '{clean_id}'."}), 404

        # Check if caller requested a specific model's result (e.g. ?tumor_type=glioma or ?model=meningioma)
        requested_type_raw = request.args.get("tumor_type") or request.args.get("model")
        if requested_type_raw:
            clean_req = requested_type_raw.lower().strip()
            canonical_key = normalize_model_name(clean_req)
            if not canonical_key and clean_req != "meningioma_multimodal":
                return jsonify({
                    "detail": f"Invalid tumor type or model '{requested_type_raw}'. Supported types: ['glioma', 'meningioma']."
                }), 400

            candidate_paths = [
                req_dir / f"result_{clean_req}.json",
                req_dir / f"result_{canonical_key}.json" if canonical_key else None,
            ]
            result_path = None
            for p in candidate_paths:
                if p and p.exists():
                    result_path = p
                    break

            if not result_path:
                return jsonify({
                    "detail": f"No {clean_req} segmentation results found for request ID '{clean_id}'."
                }), 404
        else:
            result_path = req_dir / "result.json"
            if not result_path.exists():
                return jsonify({"detail": f"Results not found for request ID '{clean_id}'."}), 404

        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "request" in data and "request_metadata" not in data:
            data["request_metadata"] = data["request"]
        if "request_metadata" in data and "request_id" not in data:
            data["request_id"] = data["request_metadata"].get("request_id")
        if "volumetrics" in data and "volumetric_summary" not in data:
            data["volumetric_summary"] = data["volumetrics"]

        return jsonify(data), 200

    @app.route("/api/report/<request_id>", methods=["GET"])
    def get_clinical_report(request_id: str):
        """
        Generates an understandable human-readable clinical research report
        for the given request ID, detailing tumor volumetrics, subregions,
        geometry, model selection, and research disclaimers.
        """
        try:
            clean_id = validate_request_id(request_id)
        except ValueError as val_err:
            return jsonify({"detail": str(val_err)}), 400

        req_dir = API_RUNS_DIR / clean_id
        if not req_dir.exists():
            return jsonify({"detail": f"Results not found for request ID '{clean_id}'."}), 404

        requested_type_raw = request.args.get("tumor_type") or request.args.get("model")
        if requested_type_raw:
            clean_req = requested_type_raw.lower().strip()
            canonical_key = normalize_model_name(clean_req)
            if not canonical_key and clean_req != "meningioma_multimodal":
                return jsonify({
                    "detail": f"Invalid tumor type or model '{requested_type_raw}'. Supported types: ['glioma', 'meningioma']."
                }), 400

            candidate_paths = [
                req_dir / f"result_{clean_req}.json",
                req_dir / f"result_{canonical_key}.json" if canonical_key else None,
            ]
            result_path = None
            for p in candidate_paths:
                if p and p.exists():
                    result_path = p
                    break

            if not result_path:
                return jsonify({
                    "detail": f"No {clean_req} report found for request ID '{clean_id}'."
                }), 404
        else:
            result_path = req_dir / "result.json"
            if not result_path.exists():
                return jsonify({"detail": f"Results not found for request ID '{clean_id}'."}), 404

        with open(result_path, "r", encoding="utf-8") as f:
            res_data = json.load(f)

        report_format = request.args.get("format", "").lower()
        if report_format == "pdf":
            cls_data = None
            cls_path = req_dir / "classification.json"
            if cls_path.exists():
                try:
                    with open(cls_path, "r", encoding="utf-8") as f_cls:
                        cls_data = json.load(f_cls)
                except Exception:
                    pass
            from api.pdf_report import generate_pdf_report
            pdf_bytes = generate_pdf_report(res_data, clean_id, cls_data)
            req_meta = res_data.get("request_metadata") or res_data.get("request", {})
            tumor_type_pdf = str(req_meta.get("tumor_type", res_data.get("status", "report"))).lower()
            return Response(
                pdf_bytes,
                mimetype="application/pdf",
                headers={"Content-Disposition": f"attachment; filename={clean_id}_{tumor_type_pdf}_report.pdf"},
            )
        elif report_format == "csv":
            cls_data = None
            cls_path = req_dir / "classification.json"
            if cls_path.exists():
                try:
                    with open(cls_path, "r", encoding="utf-8") as f_cls:
                        cls_data = json.load(f_cls)
                except Exception:
                    pass
            from api.export_utils import generate_csv_report
            csv_text = generate_csv_report(res_data, clean_id, cls_data)
            req_meta = res_data.get("request_metadata") or res_data.get("request", {})
            tumor_type_csv = str(req_meta.get("tumor_type", "tumor")).lower()
            return Response(
                csv_text,
                mimetype="text/csv; charset=utf-8",
                headers={"Content-Disposition": f"attachment; filename={clean_id}_{tumor_type_csv}_volumetrics.csv"},
            )
        elif report_format == "png":
            from api.export_utils import generate_segmentation_png
            png_bytes = generate_segmentation_png(res_data, clean_id, API_RUNS_DIR)
            req_meta = res_data.get("request_metadata") or res_data.get("request", {})
            tumor_type_png = str(req_meta.get("tumor_type", "tumor")).lower()
            return Response(
                png_bytes,
                mimetype="image/png",
                headers={"Content-Disposition": f"attachment; filename={clean_id}_{tumor_type_png}_segmentation_view.png"},
            )

        # Handle early-exit results (healthy scan or uncertain classification)
        res_status = res_data.get("status")
        if res_status == "no_tumor_detected":
            lines = [
                "=" * 68,
                "      3D BRAIN TUMOR MRI SEGMENTATION & QUANTIFICATION REPORT",
                "=" * 68,
                f"Request ID:       {clean_id}",
                f"Date / Time:      {res_data.get('timestamp') or datetime.datetime.now(datetime.timezone.utc).isoformat()}",
                "Selected Model:   None (Halted at Stage 1)",
                "Pipeline Target:  NORMATIVE SCAN (HEALTHY CONTROL)",
                "Routing Mode:     AUTOMATIC CLASSIFICATION (Normative Scan)",
                "Pipeline Status:  NO TUMOR DETECTED (200 OK)",
                "-" * 68,
                "DIAGNOSTIC & OPERATIONAL CLARIFICATION",
                f"NOTICE: Automated Stage 1 classifier assessed scan as normative / healthy control.",
                f"Presence probability: {res_data.get('presence_probability', 'N/A')} (Confidence: {res_data.get('confidence', 'N/A')}).",
                res_data.get("recommendation", "No tumor detected. Segmentation was safely halted to prevent phantom tumor contouring."),
                "Research prototype notice: does not replace clinical radiologist review.",
                "-" * 68,
                "CLINICAL SCOPE & REGULATORY NOTICE",
                "WARNING: RESEARCH PROTOTYPE ONLY. This software is an experimental prototype",
                "and has NOT been approved or cleared by the FDA or any medical regulatory authority.",
                "=" * 68,
            ]
            report_text = "\n".join(lines)
            as_download = request.args.get("download", "").lower() in ("1", "true")
            if as_download:
                return Response(
                    report_text,
                    mimetype="text/plain",
                    headers={"Content-Disposition": f"attachment; filename={clean_id}_healthy_report.txt"},
                )
            return Response(report_text, mimetype="text/plain; charset=utf-8")

        if res_status == "uncertain_classification":
            lines = [
                "=" * 68,
                "      3D BRAIN TUMOR MRI SEGMENTATION & QUANTIFICATION REPORT",
                "=" * 68,
                f"Request ID:       {clean_id}",
                f"Date / Time:      {res_data.get('timestamp') or datetime.datetime.now(datetime.timezone.utc).isoformat()}",
                "Selected Model:   None (Refused - Ambiguous)",
                "Pipeline Target:  UNCERTAIN PATHOLOGY",
                "Routing Mode:     AUTOMATIC CLASSIFICATION (Uncertainty Refusal)",
                "Pipeline Status:  UNCERTAIN CLASSIFICATION (422 UNPROCESSABLE ENTITY)",
                "-" * 68,
                "DIAGNOSTIC & OPERATIONAL CLARIFICATION",
                f"WARNING: Classification confidence ({res_data.get('confidence', 'N/A')}) is below safety threshold (0.55).",
                "Automated routing was refused to protect patient safety against potential misclassification.",
                res_data.get("recommendation", "Please select a segmentation model manually ('glioma' or 'meningioma')."),
                "Research prototype notice: does not replace clinical radiologist review.",
                "-" * 68,
                "CLINICAL SCOPE & REGULATORY NOTICE",
                "WARNING: RESEARCH PROTOTYPE ONLY. This software is an experimental prototype",
                "and has NOT been approved or cleared by the FDA or any medical regulatory authority.",
                "=" * 68,
            ]
            report_text = "\n".join(lines)
            as_download = request.args.get("download", "").lower() in ("1", "true")
            if as_download:
                return Response(
                    report_text,
                    mimetype="text/plain",
                    headers={"Content-Disposition": f"attachment; filename={clean_id}_uncertain_report.txt"},
                )
            return Response(report_text, mimetype="text/plain; charset=utf-8")

        req_meta = res_data.get("request_metadata") or res_data.get("request", {})
        inp_meta = res_data.get("input_metadata", {})
        vol_summary = res_data.get("volumetrics", res_data.get("volumetric_summary", {}))
        gl_meas = res_data.get("glioma", res_data.get("glioma_measurements"))
        men_meas = res_data.get("meningioma_multimodal", res_data.get("meningioma", res_data.get("meningioma_measurements")))

        raw_tumor_type = str(req_meta.get("tumor_type", "Unknown"))
        tumor_type = raw_tumor_type.lower()
        routing_mode = req_meta.get("routing_mode", "manual_selection")
        total_vol_mm3 = vol_summary.get("total_tumor_volume_mm3", 0.0)
        total_vol_cm3 = vol_summary.get("total_tumor_volume_cm3", 0.0)

        is_auto = (routing_mode == "automatic_classification")
        is_override = bool(req_meta.get("manual_override", False) and req_meta.get("conflict_detected", False))
        auto_avail = bool(req_meta.get("automatic_classification_available", True))
        c_assessment = req_meta.get("classifier_assessment")

        if is_auto:
            selected_model_str = f"{tumor_type.upper()} 3D U-Net (Automatic Routing)"
            routing_mode_str = "AUTOMATIC CLASSIFICATION (Hierarchical Gating)"
        elif is_override:
            selected_model_str = f"{tumor_type.upper()} 3D U-Net (Manual Override)"
            routing_mode_str = "MANUAL OVERRIDE (Classifier Disagreement)"
        elif not auto_avail:
            selected_model_str = f"{tumor_type.upper()} 3D U-Net (Manual Selection)"
            routing_mode_str = "MANUAL SELECTION (Classifier Unavailable)"
        else:
            selected_model_str = f"{tumor_type.upper()} 3D U-Net (Manual Selection)"
            routing_mode_str = "MANUAL SELECTION"

        lines = [
            "=" * 68,
            "      3D BRAIN TUMOR MRI SEGMENTATION & QUANTIFICATION REPORT",
            "=" * 68,
            f"Request ID:       {clean_id}",
            f"Date / Time:      {req_meta.get('timestamp', 'N/A')}",
            f"Selected Model:   {selected_model_str}",
            f"Pipeline Target:  {tumor_type.upper()}",
            f"Routing Mode:     {routing_mode_str}",
            f"Pipeline Status:  {str(res_data.get('quality_status', {}).get('status', res_data.get('pipeline_status', 'SUCCESS'))).upper()}",
            "-" * 68,
            "DIAGNOSTIC & OPERATIONAL CLARIFICATION",
        ]

        if is_auto:
            lines.extend([
                f"NOTICE: This segmentation was automatically routed to the {tumor_type.upper()} model",
                "based on multi-stage classifier prediction. Research prototype notice:",
                "automated routing does NOT replace histopathological or clinical verification.",
            ])
        elif is_override:
            warn = req_meta.get("warning") or req_meta.get("diagnostic_notice")
            if warn:
                lines.extend([
                    "WARNING: MANUAL OVERRIDE CONFLICT DETECTED",
                    warn,
                ])
            else:
                lines.extend([
                    f"WARNING: User manually selected {tumor_type.upper()} segmentation despite classifier",
                    "disagreement. This segmentation quantifies abnormalities matching the",
                    f"{tumor_type.upper()} model assumptions only and does NOT confirm pathology.",
                ])
        elif not auto_avail:
            lines.extend([
                f"NOTICE: This segmentation was generated using the user-selected {tumor_type.upper()}",
                "model. Automated tumor-presence and tumor-type classification are currently",
                "UNAVAILABLE due to missing trained classifier checkpoints. Therefore, this",
                f"volumetric segmentation does NOT constitute an independent histological or",
                f"clinical diagnosis of {tumor_type.lower()}.",
            ])
        else:
            notice = req_meta.get("diagnostic_notice")
            if notice:
                lines.append(f"NOTICE: {notice}")
            else:
                lines.extend([
                    f"NOTICE: This segmentation was performed using the user-selected {tumor_type.upper()}",
                    "model (manual selection). This volumetric segmentation quantifies abnormalities",
                    f"matching {tumor_type.upper()} model assumptions only, and does NOT constitute",
                    f"an independent histological or clinical diagnosis of {tumor_type.lower()}.",
                ])

        lines.extend([
            "-" * 68,
            "ACQUISITION & GEOMETRY SPECIFICATIONS",
            f"Matrix Dimensions: {inp_meta.get('original_shape', 'N/A')}",
            f"Voxel Spacing:     {inp_meta.get('original_spacing', 'N/A')} mm",
            f"Unit Voxel Volume: {inp_meta.get('voxel_volume_mm3', 0.0):.4f} mm³",
            f"Modalities Used:   {', '.join(inp_meta.get('modality_names', []))}",
            "-" * 68,
            "VOLUMETRIC QUANTIFICATION SUMMARY",
            f"Total Tumor Volume: {total_vol_mm3:,.1f} mm³  ({total_vol_cm3:.2f} cm³)",
        ])

        if gl_meas:
            sub = gl_meas.get("subregions") or gl_meas.get("subregion_volumes") or {}
            lines.extend([
                "Subregion Breakdown (Glioma):",
                f"  - Whole Tumor (WT):         {sub.get('wt_volume_mm3', 0.0):,.1f} mm³ ({sub.get('wt_volume_cm3', 0.0):.2f} cm³)",
                f"  - Tumor Core (TC):          {sub.get('tc_volume_mm3', 0.0):,.1f} mm³ ({sub.get('tc_volume_cm3', 0.0):.2f} cm³)",
                f"  - Enhancing Tumor (ET):      {sub.get('et_volume_mm3', 0.0):,.1f} mm³ ({sub.get('et_volume_cm3', 0.0):.2f} cm³)",
            ])
        elif men_meas:
            sub = men_meas.get("subregions") or men_meas.get("subregion_volumes") or {}
            lines.extend([
                "Subregion Breakdown (Meningioma Multimodal):",
                f"  - Whole Tumor (WT):         {sub.get('wt_volume_mm3', 0.0):,.1f} mm³ ({sub.get('wt_volume_cm3', 0.0):.2f} cm³)",
                f"  - Enhancing Tumor (ET):      {sub.get('et_volume_mm3', 0.0):,.1f} mm³ ({sub.get('et_volume_cm3', 0.0):.2f} cm³)",
                f"  - Non-Enhancing Core (NCR): {sub.get('ncr_volume_mm3', 0.0):,.1f} mm³ ({sub.get('ncr_volume_cm3', 0.0):.2f} cm³)",
                f"  - Edema (ED):                {sub.get('ed_volume_mm3', 0.0):,.1f} mm³ ({sub.get('ed_volume_cm3', 0.0):.2f} cm³)",
            ])

        lines.extend([
            "-" * 68,
            "AVAILABLE DOWNLOADABLE ARTIFACTS",
            f"Multiclass NIfTI Mask: /api/mask/{clean_id}?type=multiclass&tumor_type={tumor_type}",
            f"Binary GTV NIfTI Mask:  /api/mask/{clean_id}?type=gtv&tumor_type={tumor_type}",
            "-" * 68,
            "CLINICAL SCOPE & REGULATORY NOTICE",
            "WARNING: RESEARCH PROTOTYPE ONLY. This software is an experimental prototype",
            "and has NOT been approved or cleared by the FDA or any medical regulatory authority.",
            "Segmentations and volumetric calculations must NOT be used as the sole basis for",
            "primary diagnosis, biopsy targeting, radiation dosing, or surgical planning.",
            "=" * 68,
        ])

        report_text = "\n".join(lines)
        as_download = request.args.get("download", "").lower() in ("1", "true")
        if as_download:
            return Response(
                report_text,
                mimetype="text/plain",
                headers={"Content-Disposition": f"attachment; filename={clean_id}_{tumor_type}_report.txt"},
            )
        return Response(report_text, mimetype="text/plain; charset=utf-8")

    @app.route("/api/report/<request_id>/pdf", methods=["GET"])
    def get_clinical_report_pdf(request_id: str):
        """
        Direct dedicated endpoint to stream and download patient clinical report in PDF format.
        """
        try:
            clean_id = validate_request_id(request_id)
        except ValueError as val_err:
            return jsonify({"detail": str(val_err)}), 400

        req_dir = API_RUNS_DIR / clean_id
        if not req_dir.exists():
            return jsonify({"detail": f"Results not found for request ID '{clean_id}'."}), 404

        requested_type_raw = request.args.get("tumor_type") or request.args.get("model")
        if requested_type_raw:
            clean_req = requested_type_raw.lower().strip()
            canonical_key = normalize_model_name(clean_req)
            candidate_paths = [
                req_dir / f"result_{clean_req}.json",
                req_dir / f"result_{canonical_key}.json" if canonical_key else None,
            ]
            result_path = None
            for p in candidate_paths:
                if p and p.exists():
                    result_path = p
                    break
            if not result_path:
                return jsonify({"detail": f"No {clean_req} report found for request ID '{clean_id}'."}), 404
        else:
            result_path = req_dir / "result.json"
            if not result_path.exists():
                return jsonify({"detail": f"Results not found for request ID '{clean_id}'."}), 404

        with open(result_path, "r", encoding="utf-8") as f:
            res_data = json.load(f)

        cls_data = None
        cls_path = req_dir / "classification.json"
        if cls_path.exists():
            try:
                with open(cls_path, "r", encoding="utf-8") as f_cls:
                    cls_data = json.load(f_cls)
            except Exception:
                pass

        from api.pdf_report import generate_pdf_report
        pdf_bytes = generate_pdf_report(res_data, clean_id, cls_data)
        req_meta = res_data.get("request_metadata") or res_data.get("request", {})
        tumor_type_pdf = str(req_meta.get("tumor_type", res_data.get("status", "report"))).lower()
        return Response(
            pdf_bytes,
            mimetype="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={clean_id}_{tumor_type_pdf}_report.pdf"},
        )

    @app.route("/api/export/pdf/<request_id>", methods=["GET"])
    def export_report_pdf(request_id: str):
        """Dedicated export endpoint for patient PDF report."""
        return get_clinical_report_pdf(request_id)

    @app.route("/api/export/csv/<request_id>", methods=["GET"])
    def export_volumetrics_csv(request_id: str):
        """Dedicated endpoint to stream and download patient volume summary in CSV format."""
        try:
            clean_id = validate_request_id(request_id)
        except ValueError as val_err:
            return jsonify({"detail": str(val_err)}), 400

        req_dir = API_RUNS_DIR / clean_id
        if not req_dir.exists():
            return jsonify({"detail": f"Results not found for request ID '{clean_id}'."}), 404

        result_path = req_dir / "result.json"
        if not result_path.exists():
            return jsonify({"detail": f"Results not found for request ID '{clean_id}'."}), 404

        with open(result_path, "r", encoding="utf-8") as f:
            res_data = json.load(f)

        cls_data = None
        cls_path = req_dir / "classification.json"
        if cls_path.exists():
            try:
                with open(cls_path, "r", encoding="utf-8") as f_cls:
                    cls_data = json.load(f_cls)
            except Exception:
                pass

        from api.export_utils import generate_csv_report
        csv_text = generate_csv_report(res_data, clean_id, cls_data)
        req_meta = res_data.get("request_metadata") or res_data.get("request", {})
        tumor_type_csv = str(req_meta.get("tumor_type", "tumor")).lower()
        return Response(
            csv_text,
            mimetype="text/csv; charset=utf-8",
            headers={"Content-Disposition": f"attachment; filename={clean_id}_{tumor_type_csv}_volumetrics.csv"},
        )

    @app.route("/api/export/png/<request_id>", methods=["GET"])
    def export_segmentation_png(request_id: str):
        """Dedicated endpoint to stream and download representative 2D axial segmentation view in PNG format."""
        try:
            clean_id = validate_request_id(request_id)
        except ValueError as val_err:
            return jsonify({"detail": str(val_err)}), 400

        req_dir = API_RUNS_DIR / clean_id
        if not req_dir.exists():
            return jsonify({"detail": f"Results not found for request ID '{clean_id}'."}), 404

        result_path = req_dir / "result.json"
        if not result_path.exists():
            return jsonify({"detail": f"Results not found for request ID '{clean_id}'."}), 404

        with open(result_path, "r", encoding="utf-8") as f:
            res_data = json.load(f)

        from api.export_utils import generate_segmentation_png
        png_bytes = generate_segmentation_png(res_data, clean_id, API_RUNS_DIR)
        req_meta = res_data.get("request_metadata") or res_data.get("request", {})
        tumor_type_png = str(req_meta.get("tumor_type", "tumor")).lower()
        return Response(
            png_bytes,
            mimetype="image/png",
            headers={"Content-Disposition": f"attachment; filename={clean_id}_{tumor_type_png}_segmentation_view.png"},
        )

    @app.route("/api/mask/<request_id>", methods=["GET"])
    def download_mask(request_id: str):
        """
        Downloads the 3D NIfTI segmentation mask file (.nii.gz) for a given request ID.
        Optional query parameter 'type': 'multiclass' (default) or 'gtv'.
        Optional query parameter 'tumor_type' or 'model': 'glioma', 'meningioma'.
        """
        try:
            clean_id = validate_request_id(request_id)
        except ValueError as val_err:
            return jsonify({"detail": str(val_err)}), 400

        mask_dir = API_RUNS_DIR / clean_id / "masks"
        if not mask_dir.exists():
            return jsonify({"detail": f"Mask directory not found for request ID '{clean_id}'."}), 404

        mask_files = list(mask_dir.glob("*.nii.gz")) + list(mask_dir.glob("*.nii"))
        if not mask_files:
            return jsonify({"detail": f"Segmentation mask not found for request ID '{clean_id}'."}), 404

        # 1. Determine target tumor type from query param or latest result.json
        req_tumor_type = request.args.get("tumor_type") or request.args.get("model")
        if req_tumor_type:
            clean_req = req_tumor_type.lower().strip()
            canonical_key = normalize_model_name(clean_req)
            if not canonical_key and clean_req != "meningioma_multimodal":
                return jsonify({
                    "detail": f"Invalid tumor type or model '{req_tumor_type}'. Supported types: ['glioma', 'meningioma']."
                }), 400
            model_key = canonical_key or clean_req
            matching_masks = [f for f in mask_files if f"_{model_key}_" in f.name.lower()]
            if not matching_masks:
                return jsonify({
                    "detail": f"No {req_tumor_type} segmentation mask found for request ID '{clean_id}'."
                }), 404
        else:
            matching_masks = mask_files
            result_json_path = API_RUNS_DIR / clean_id / "result.json"
            if result_json_path.exists():
                try:
                    with open(result_json_path, "r", encoding="utf-8") as f:
                        res_json = json.load(f)
                    latest_model = str(res_json.get("request", res_json.get("request_metadata", {})).get("tumor_type", "")).lower().strip()
                    canonical_latest = normalize_model_name(latest_model)
                    if canonical_latest:
                        filtered = [f for f in mask_files if f"_{canonical_latest}_" in f.name.lower()]
                        if filtered:
                            matching_masks = filtered
                except Exception:
                    pass

        mask_type = request.args.get("type", "multiclass").lower().strip()
        if mask_type not in ("multiclass", "gtv"):
            return jsonify({
                "detail": f"Invalid mask type '{mask_type}'. Supported types: 'multiclass', 'gtv'."
            }), 400

        if mask_type == "gtv":
            gtv_candidates = [f for f in matching_masks if "_gtv" in f.name]
            gtv_candidates.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            if gtv_candidates:
                target_mask = gtv_candidates[0]
            else:
                # Dynamically generate binary GTV mask (labels 0 and 1) from multiclass mask if not yet present
                multi_candidates = [f for f in matching_masks if "_gtv" not in f.name]
                multi_candidates.sort(key=lambda f: f.stat().st_mtime, reverse=True)
                if not multi_candidates:
                    return jsonify({"detail": f"Segmentation mask not found for request ID '{clean_id}'."}), 404
                src_mask = multi_candidates[0]
                try:
                    src_img = sitk.ReadImage(str(src_mask))
                    src_arr = sitk.GetArrayFromImage(src_img)
                    gtv_arr = (src_arr > 0).astype(np.uint8)
                    gtv_img = sitk.GetImageFromArray(gtv_arr)
                    gtv_img.CopyInformation(src_img)

                    fname = src_mask.name
                    if fname.endswith("_seg.nii.gz"):
                        gtv_fname = fname[:-11] + "_gtv.nii.gz"
                    elif fname.endswith(".nii.gz"):
                        gtv_fname = fname[:-7] + "_gtv.nii.gz"
                    elif src_mask.suffix:
                        gtv_fname = f"{src_mask.stem}_gtv{src_mask.suffix}"
                    else:
                        gtv_fname = f"{fname}_gtv.nii.gz"

                    target_mask = mask_dir / gtv_fname
                    sitk.WriteImage(gtv_img, str(target_mask))
                except Exception as exc:
                    return jsonify({
                        "detail": f"Failed to generate binary GTV mask: {str(exc)}"
                    }), 500
        else:
            multi_candidates = [f for f in matching_masks if "_gtv" not in f.name]
            multi_candidates.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            target_mask = multi_candidates[0] if multi_candidates else matching_masks[0]

        return send_file(
            str(target_mask),
            mimetype="application/gzip",
            as_attachment=True,
            download_name=target_mask.name,
        )

    @app.route("/api/volume/<request_id>", methods=["GET"])
    def get_uploaded_volume(request_id: str):
        """
        Serves an uploaded structural MRI volume (.nii or .nii.gz) for interactive browser visualization.
        Optional query parameter 'modality': 't1c', 't1', 'flair', 't2' (case-insensitive).
        Defaults to priority order: T1c -> FLAIR -> T1 -> T2 -> first available volume.
        """
        try:
            clean_id = validate_request_id(request_id)
        except ValueError as val_err:
            return jsonify({"detail": str(val_err)}), 400

        upload_dir = API_RUNS_DIR / clean_id / "uploads"
        if not upload_dir.exists():
            return jsonify({"detail": f"Upload directory not found for request ID '{clean_id}'."}), 404

        vols = list(upload_dir.glob("*.nii.gz")) + list(upload_dir.glob("*.nii"))
        if not vols:
            return jsonify({"detail": f"No MRI volumes found for request ID '{clean_id}'."}), 404

        # Read modality_map.json if exists
        saved_map = {}
        map_path = API_RUNS_DIR / clean_id / "modality_map.json"
        if map_path.exists():
            try:
                with open(map_path, "r", encoding="utf-8") as f:
                    saved_map = json.load(f)
            except Exception:
                saved_map = {}

        req_mod = request.args.get("modality")
        target_vol = None

        if req_mod:
            rm_upper = req_mod.strip().upper()
            canon = (
                "T1" if rm_upper in ("T1", "T1N", "T1W")
                else "T1c" if rm_upper in ("T1C", "T1CE", "T1-CE", "T1_CE")
                else "T2" if rm_upper in ("T2", "T2W")
                else "FLAIR" if rm_upper in ("FLAIR", "T2F", "T2-FLAIR", "T2_FLAIR")
                else rm_upper
            )
            if canon in saved_map and (upload_dir / saved_map[canon]).exists():
                target_vol = upload_dir / saved_map[canon]
            else:
                for v in vols:
                    if detect_modality_from_filename(v.name) == canon or canon.lower() in v.name.lower():
                        target_vol = v
                        break

        if not target_vol:
            for pref in ["T1c", "FLAIR", "T1", "T2"]:
                if pref in saved_map and (upload_dir / saved_map[pref]).exists():
                    target_vol = upload_dir / saved_map[pref]
                    break
                for v in vols:
                    if detect_modality_from_filename(v.name) == pref:
                        target_vol = v
                        break
                if target_vol:
                    break

        if not target_vol:
            target_vol = vols[0]

        as_att = request.args.get("download", "0") in ("1", "true", "True")
        return send_file(
            str(target_vol),
            mimetype="application/gzip" if target_vol.name.endswith(".gz") else "application/octet-stream",
            as_attachment=as_att,
            download_name=target_vol.name,
        )

    FRONTEND_DIR = PROJECT_ROOT / "frontend"

    @app.route("/", methods=["GET"])
    def serve_frontend_index():
        if (FRONTEND_DIR / "index.html").exists():
            return send_from_directory(FRONTEND_DIR, "index.html")
        return jsonify({
            "message": "3D Brain Tumor MRI Segmentation Prototype API",
            "health": "/api/health",
            "classifier_status": "/api/classifier/status",
        }), 200

    @app.route("/<path:filename>", methods=["GET"])
    def serve_frontend_static(filename):
        if (FRONTEND_DIR / filename).exists():
            return send_from_directory(FRONTEND_DIR, filename)
        if filename.startswith("api/"):
            return jsonify({"detail": f"API endpoint '/{filename}' not found."}), 404
        if (FRONTEND_DIR / "index.html").exists():
            return send_from_directory(FRONTEND_DIR, "index.html")
        return jsonify({"detail": f"File '{filename}' not found."}), 404

    return app


app = create_app()

if __name__ == "__main__":
    print("Starting 3D Brain Tumor Segmentation Flask API on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)
