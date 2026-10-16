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
import SimpleITK as sitk
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename

from integration.inference_router import InferenceRouter, run_inference
from integration.result_schema import TumorType, SegmentationResult

API_RUNS_DIR = PROJECT_ROOT / "outputs" / "api_runs"
API_RUNS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".nii", ".nii.gz"}
MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB per file limit
REQUEST_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")


def validate_request_id(request_id: Optional[str]) -> str:
    """
    Validates that a request_id adheres to strict alphanumeric/dash/underscore allowlist
    and does not escape the configured API_RUNS_DIR.

    Raises:
        ValueError: If request_id is empty, malformed, or attempts path traversal.
    """
    if not request_id or not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("Request ID cannot be empty.")

    clean_id = request_id.strip()
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


def detect_modality_from_filename(filename: str) -> Optional[str]:
    """Infers modality key from standard naming conventions (e.g. t1, t1c, t1ce, t2, flair, gtv)."""
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


def create_app(router: Optional[InferenceRouter] = None) -> Flask:
    """Application factory for Flask backend."""
    app = Flask(__name__)
    CORS(app)

    # Attach router instance
    app.config["ROUTER"] = router or InferenceRouter()
    app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_BYTES

    @app.route("/api/health", methods=["GET"])
    def health_check():
        """Returns API health status, compute device, and supported tumor types."""
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        return jsonify({
            "status": "healthy",
            "version": "1.0.0",
            "device": device,
            "device_name": device_name,
            "supported_tumor_types": ["glioma", "meningioma"],
            "supported_formats": [".nii", ".nii.gz"],
        }), 200

    @app.route("/api/upload", methods=["POST"])
    def upload_scans():
        """
        Accepts and validates one or more 3D MRI NIfTI files (.nii, .nii.gz).
        Stores files in an isolated request directory under outputs/api_runs/<request_id>/uploads/.
        """
        uploaded_files_list = []
        if "files" in request.files:
            uploaded_files_list = request.files.getlist("files")
        elif len(request.files) > 0:
            uploaded_files_list = list(request.files.values())

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

            inferred_mod = detect_modality_from_filename(safe_name)
            if inferred_mod:
                detected_modalities[inferred_mod] = safe_name

            uploaded_records.append({
                "filename": safe_name,
                "size_bytes": dest_path.stat().st_size,
                "inferred_modality": inferred_mod,
            })

        return jsonify({
            "request_id": req_id,
            "status": "uploaded",
            "files_count": len(uploaded_records),
            "files": uploaded_records,
            "detected_modalities": detected_modalities,
        }), 200

    def _execute_segmentation(data: dict):
        raw_req_id = data.get("request_id")
        if not raw_req_id:
            return jsonify({"detail": "Missing required field 'request_id'."}), 400

        try:
            req_id = validate_request_id(raw_req_id)
        except ValueError as val_err:
            return jsonify({"detail": str(val_err)}), 400

        tumor_type_str = data.get("tumor_type")
        if not tumor_type_str:
            return jsonify({"detail": "Missing required field 'tumor_type'."}), 400

        req_dir = API_RUNS_DIR / req_id
        upload_dir = req_dir / "uploads"

        if not upload_dir.exists():
            return jsonify({
                "detail": f"Request ID '{req_id}' not found. Please upload MRI scans first via /api/upload."
            }), 404

        # Validate tumor type
        try:
            router: InferenceRouter = app.config["ROUTER"]
            tumor_type = router.normalize_tumor_type(tumor_type_str)
        except (ValueError, TypeError) as exc:
            return jsonify({"detail": str(exc)}), 400

        uploaded_files = list(upload_dir.glob("*.nii*"))
        if not uploaded_files:
            return jsonify({"detail": f"No NIfTI scans found for request '{req_id}'."}), 400

        modality_map = data.get("modality_map", {})
        case_input: Union[Path, Dict[str, Path]]

        if tumor_type == TumorType.GLIOMA:
            mod_map: Dict[str, Path] = {}
            if modality_map and isinstance(modality_map, dict):
                for mod_key, fn in modality_map.items():
                    target = upload_dir / fn
                    if target.exists():
                        mod_map[mod_key] = target
            else:
                for f in uploaded_files:
                    m = detect_modality_from_filename(f.name)
                    if m:
                        mod_map[m] = f

            missing = [m for m in ["T1", "T1c", "T2", "FLAIR"] if m not in mod_map]
            if missing:
                return jsonify({
                    "detail": f"Glioma segmentation requires 4 modalities (T1, T1c, T2, FLAIR). Missing: {missing}. Uploaded: {[f.name for f in uploaded_files]}"
                }), 400
            case_input = mod_map

        elif tumor_type in (TumorType.MENINGIOMA, TumorType.MENINGIOMA_MULTIMODAL):
            from integration.meningioma_multimodal_service import detect_modality_key, REQUIRED_MODALITIES
            mod_map: Dict[str, Path] = {}
            if modality_map and isinstance(modality_map, dict):
                for mod_key, fn in modality_map.items():
                    target = upload_dir / fn
                    if target.exists():
                        inferred = detect_modality_key(str(mod_key))
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
                        if inferred:
                            mod_map[inferred] = target
            else:
                for f in uploaded_files:
                    m = detect_modality_key(f.name)
                    if m and m not in mod_map:
                        mod_map[m] = f
                    elif not m:
                        m_old = detect_modality_from_filename(f.name)
                        if m_old == "T1" and "t1n" not in mod_map:
                            mod_map["t1n"] = f
                        elif m_old == "T1c" and "t1c" not in mod_map:
                            mod_map["t1c"] = f
                        elif m_old == "T2" and "t2w" not in mod_map:
                            mod_map["t2w"] = f
                        elif m_old == "FLAIR" and "t2f" not in mod_map:
                            mod_map["t2f"] = f

            missing = [m for m in REQUIRED_MODALITIES if m not in mod_map]
            if missing:
                return jsonify({
                    "detail": f"Meningioma segmentation requires 4 modalities (t1n, t1c, t2w, t2f). Missing: {missing}. Uploaded: {[f.name for f in uploaded_files]}"
                }), 400
            case_input = mod_map

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
        except (ValueError, TypeError, FileNotFoundError) as client_err:
            return jsonify({
                "detail": f"Input validation or processing error: {str(client_err)}"
            }), 400
        except Exception as exc:
            return jsonify({
                "detail": f"Inference execution failed: {str(exc)}"
            }), 500

        # Save result JSON for subsequent retrieval
        result_json_path = req_dir / "result.json"
        with open(result_json_path, "w", encoding="utf-8") as f:
            f.write(result.to_json(indent=2))

        return Response(result.to_json(), status=200, mimetype="application/json")

    @app.route("/api/segment", methods=["POST"])
    def run_segmentation():
        """
        Executes segmentation on previously uploaded scans for an explicit tumor type.
        Validates modality completeness, dispatches to InferenceRouter, saves output mask,
        and returns the common SegmentationResult JSON.
        """
        data = request.get_json(silent=True)
        if not data or not isinstance(data, dict):
            return jsonify({"detail": "Request body must be a valid JSON object."}), 400
        return _execute_segmentation(data)

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
        result_path = req_dir / "result.json"

        if not result_path.exists():
            return jsonify({"detail": f"Results not found for request ID '{clean_id}'."}), 404

        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data), 200

    @app.route("/api/mask/<request_id>", methods=["GET"])
    def download_mask(request_id: str):
        """
        Downloads the 3D NIfTI segmentation mask file (.nii.gz) for a given request ID.
        Optional query parameter 'type': 'multiclass' (default) or 'gtv'.
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

        mask_type = request.args.get("type", "multiclass").lower().strip()
        if mask_type not in ("multiclass", "gtv"):
            return jsonify({
                "detail": f"Invalid mask type '{mask_type}'. Supported types: 'multiclass', 'gtv'."
            }), 400

        if mask_type == "gtv":
            gtv_candidates = [f for f in mask_files if "_gtv" in f.name]
            if gtv_candidates:
                target_mask = gtv_candidates[0]
            else:
                # Dynamically generate binary GTV mask (labels 0 and 1) from multiclass mask if not yet present
                multi_candidates = [f for f in mask_files if "_gtv" not in f.name]
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
            multi_candidates = [f for f in mask_files if "_gtv" not in f.name]
            target_mask = multi_candidates[0] if multi_candidates else mask_files[0]

        return send_file(
            str(target_mask),
            mimetype="application/gzip",
            as_attachment=True,
            download_name=target_mask.name,
        )

    return app


app = create_app()

if __name__ == "__main__":
    print("Starting 3D Brain Tumor Segmentation Flask API on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)
