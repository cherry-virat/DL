"""
Meningioma Multimodal Inference Integration Service.
=====================================================

Isolated integration service wrapping the 4-channel Multi-Modal Meningioma 3D U-Net
segmentation pipeline (BraTS 2023 Meningioma). Connects 4-channel MRI inputs
(t1n, t1c, t2w, t2f), sliding-window inference with Gaussian blending, empty-air
masking on prediction, and subregion quantification (WT, TC, ET, NCR, ED) to the
standard SegmentationResult data contract.

Model & Pipeline Specifications:
---------------------------------
- Checkpoint: outputs/meningioma_multimodal_experiment/full800_brainaware_20261014/checkpoints/meningioma_multimodal_best.pth
- Architecture: 3D U-Net (4 input channels, 4 output classes, base_channels=16)
- Input Modalities: [t1n, t1c, t2w, t2f] (T1-native, T1-contrast, T2-weighted, T2-FLAIR)
- Normalization: Independent non-zero z-score per channel (empty-air voxels remain 0.0)
- Sliding Window: Patch size (64, 64, 64), Overlap 0.50, sw_batch_size 2, mode="gaussian"
- Empty-Air Masking: Zeroes predicted class for voxels where all 4 channels are 0.0
- Target Labels: 0: BG, 1: NCR (Necrotic Core), 2: ED (Edema), 3: ET (Enhancing Tumor)
- Derived Subregions:
    - Whole Tumor (WT): labels 1, 2, 3
    - Tumor Core (TC):  labels 1, 3
    - Enhancing Tumor (ET): label 3
- Performance Benchmark (Validation Split, 200 cases):
    - Macro WT Dice: 0.7991 (Median: 0.9006, Pooled: 0.9273)
- Clinical Scope Disclaimer:
    - Prototype research tool only; not for primary clinical diagnosis or standalone decision making.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import datetime
import uuid

import numpy as np
import torch
import nibabel as nib
from monai.inferers import sliding_window_inference

from integration.result_schema import (
    TumorType,
    PipelineStatus,
    RequestMetadata,
    InputMetadata,
    MaskReference,
    VolumetricSummary,
    GliomaSubregionMasks,
    MeningiomaMeasurements,
    MeningiomaMultimodalSubregionVolumes,
    MeningiomaMultimodalMeasurements,
    InferenceQualityStatus,
    OfflineEvaluationMetrics,
    SegmentationResult,
)
from models.meningioma_multimodal_unet import create_multimodal_unet
from training.meningioma_multimodal_dataset import normalize_nonzero_zscore
from training.train_meningioma_multimodal import apply_air_masking

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MULTIMODAL_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "meningioma_multimodal_experiment"
    / "full800_brainaware_20261014"
    / "checkpoints"
    / "meningioma_multimodal_best.pth"
)

PATCH_SIZE = (64, 64, 64)
ROI_SIZE = PATCH_SIZE
OVERLAP = 0.50
SW_BATCH_SIZE = 2
REQUIRED_MODALITIES = ["t1n", "t1c", "t2w", "t2f"]

LABEL_DEFINITIONS = {
    0: "Background",
    1: "Necrotic Tumor Core (NCR)",
    2: "Peritumoral Edema (ED)",
    3: "Enhancing Tumor (ET)",
}

QC_VERDICT_BENCHMARK = "PASS WITH VALIDATION BENCHMARK (0.7991 Macro WT Dice / 0.9006 Median WT Dice)"
QC_BENCHMARK_200_CASES = {
    "cohort_size": 200,
    "macro_wt_dice": 0.7991,
    "median_wt_dice": 0.9006,
    "pooled_wt_dice": 0.9273,
    "macro_tc_dice": 0.8141,
    "macro_et_dice": 0.8208,
}
TRAINING_25_CASES_WT_DICE = 0.8312

RESEARCH_DISCLAIMER = (
    "Research prototype only. Not approved for primary clinical diagnosis or medical decision making."
)


def is_segmentation_mask_filename(filename: str) -> bool:
    """
    Checks whether a filename represents a segmentation mask or ground-truth label volume
    rather than a raw structural MRI acquisition sequence.
    """
    fn_lower = filename.lower()
    base = fn_lower
    for ext in (".gz", ".nii", ".tar", ".zip"):
        if base.endswith(ext):
            base = base[:-len(ext)]

    if base in ("seg", "mask", "gtv", "label", "labels"):
        return True

    for tok in ("-seg", "_seg", ".seg", "-mask", "_mask", ".mask", "-gtv", "_gtv", ".gtv"):
        if base.endswith(tok):
            return True

    for tok in ("seg", "mask", "gtv"):
        if f"-{tok}-" in fn_lower or f"_{tok}_" in fn_lower or f".{tok}." in fn_lower:
            return True
        if fn_lower.endswith(f"-{tok}.nii.gz") or fn_lower.endswith(f"_{tok}.nii.gz"):
            return True
        if fn_lower.endswith(f"-{tok}.nii") or fn_lower.endswith(f"_{tok}.nii"):
            return True

    return False


def detect_modality_key(filename: str) -> Optional[str]:
    """Infers modality key from standard file naming conventions. Returns None for masks."""
    if is_segmentation_mask_filename(filename):
        return None
    fn = filename.lower()
    if "t1c" in fn or "t1ce" in fn or "t1_ce" in fn or "t1-ce" in fn:
        return "t1c"
    elif "t1n" in fn or "t1w" in fn or (("t1" in fn) and not any(k in fn for k in ["t1c", "t1ce", "t1_ce", "t1-ce"])):
        return "t1n"
    elif "flair" in fn or "t2f" in fn or "t2_flair" in fn:
        return "t2f"
    elif "t2w" in fn or (("t2" in fn) and not any(k in fn for k in ["flair", "t2f", "t2_flair"])):
        return "t2w"
    return None


class MeningiomaMultimodalInferenceService:
    """
    Isolated inference service for the 4-channel Multi-Modal Meningioma 3D U-Net pipeline.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
        model: Optional[torch.nn.Module] = None,
    ):
        self.checkpoint_path = Path(checkpoint_path or DEFAULT_MULTIMODAL_CHECKPOINT)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = model

    def get_model(self) -> torch.nn.Module:
        """Loads and caches the frozen 4-channel Multimodal Meningioma 3D U-Net model."""
        if self._model is not None:
            return self._model

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Multimodal Meningioma checkpoint not found at: {self.checkpoint_path}"
            )

        model = create_multimodal_unet(in_channels=4, num_classes=4, base_channels=16)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        else:
            raise TypeError(f"Unexpected checkpoint structure: {type(checkpoint)}")

        model.to(self.device)
        model.eval()
        self._model = model
        return self._model

    def preprocess_inputs(
        self,
        case_input: Union[str, Path, Dict[str, Union[str, Path, nib.Nifti1Image]]],
    ) -> Tuple[np.ndarray, InputMetadata, nib.Nifti1Image]:
        """
        Loads, validates, and normalizes the 4 MRI modalities in strict order [t1n, t1c, t2w, t2f].
        Verifies spatial dimension congruence and affine alignment across all 4 channels.
        
        Returns:
            processed_4d: 4D numpy array (4, D, H, W) of float32
            metadata: InputMetadata describing acquisition geometry
            ref_nii: Reference NIfTI object for affine preservation
        """
        modality_sources: Dict[str, Union[Path, nib.Nifti1Image]] = {}

        if isinstance(case_input, (str, Path)):
            case_path = Path(case_input)
            if not case_path.exists():
                raise FileNotFoundError(f"Input path does not exist: {case_path}")

            if case_path.is_dir():
                for mod in REQUIRED_MODALITIES:
                    matches = list(case_path.glob(f"*-{mod}.nii*")) + list(case_path.glob(f"*{mod}*.nii*"))
                    if not matches:
                        # Case insensitive fallback
                        all_files = list(case_path.glob("*.nii*"))
                        matches = [f for f in all_files if detect_modality_key(f.name) == mod]
                    if not matches:
                        raise FileNotFoundError(
                            f"Missing required modality '{mod}' in directory: {case_path}"
                        )
                    modality_sources[mod] = matches[0]
            else:
                raise ValueError(
                    f"Expected a directory containing 4 modality files, got single file: {case_path}"
                )

        elif isinstance(case_input, dict):
            # Normalize dictionary keys
            normalized_dict: Dict[str, Union[Path, nib.Nifti1Image]] = {}
            for k, v in case_input.items():
                inferred = detect_modality_key(str(k))
                if inferred:
                    target = Path(v) if isinstance(v, (str, Path)) else v
                    normalized_dict[inferred] = target
                else:
                    k_lower = str(k).lower().strip()
                    if k_lower in REQUIRED_MODALITIES:
                        target = Path(v) if isinstance(v, (str, Path)) else v
                        normalized_dict[k_lower] = target

            missing = [m for m in REQUIRED_MODALITIES if m not in normalized_dict]
            if missing:
                raise ValueError(
                    f"Missing required multimodal modalities: {missing}. "
                    f"Provided keys: {list(case_input.keys())}"
                )
            modality_sources = normalized_dict

        else:
            raise TypeError(
                f"Unsupported case_input type: {type(case_input)}. "
                "Expected directory path, dict of 4 modalities, or 4D numpy array."
            )

        # Load and validate all 4 modalities
        loaded_nii: Dict[str, nib.Nifti1Image] = {}
        for mod in REQUIRED_MODALITIES:
            src = modality_sources[mod]
            if isinstance(src, nib.Nifti1Image):
                loaded_nii[mod] = src
            else:
                p = Path(src)
                if not p.exists():
                    raise FileNotFoundError(f"Modality '{mod}' file does not exist: {p}")
                loaded_nii[mod] = nib.load(str(p))

        ref_mod = "t1c" if "t1c" in loaded_nii else REQUIRED_MODALITIES[0]
        ref_nii = loaded_nii[ref_mod]
        ref_arr = np.asanyarray(ref_nii.dataobj)
        ref_shape = ref_arr.shape
        ref_affine = ref_nii.affine
        ref_zooms = [float(z) for z in ref_nii.header.get_zooms()[:3]]
        voxel_vol_mm3 = float(np.prod(ref_zooms))

        if ref_arr.ndim != 3:
            raise ValueError(
                f"Reference scan '{ref_mod}' must be 3D, got dimension {ref_arr.ndim} with shape {ref_shape}"
            )

        channels: List[np.ndarray] = []
        for mod in REQUIRED_MODALITIES:
            nii_obj = loaded_nii[mod]
            vol_arr = np.asanyarray(nii_obj.dataobj).astype(np.float32)

            # Spatial consistency checks
            if vol_arr.shape != ref_shape:
                raise ValueError(
                    f"Spatial shape mismatch for modality '{mod}': "
                    f"got {vol_arr.shape}, expected {ref_shape} matching {ref_mod}."
                )

            if not np.allclose(nii_obj.affine, ref_affine, atol=1e-3):
                raise ValueError(
                    f"Spatial affine mismatch for modality '{mod}' relative to {ref_mod}."
                )

            if not np.isfinite(vol_arr).all():
                raise ValueError(f"Non-finite voxels (NaN or Inf) detected in modality '{mod}'.")

            # Independent non-zero z-score normalization
            norm_vol = normalize_nonzero_zscore(vol_arr)
            channels.append(norm_vol)

        # Stack into (4, D, H, W)
        processed_4d = np.stack(channels, axis=0).astype(np.float32)

        metadata = InputMetadata(
            modality_names=REQUIRED_MODALITIES,
            original_shape=[int(x) for x in ref_shape],
            original_spacing=ref_zooms,
            voxel_volume_mm3=voxel_vol_mm3,
            original_affine=[[float(c) for c in row] for row in ref_affine],
            resampled_shape=[int(x) for x in ref_shape],
            resampled_spacing=ref_zooms,
        )

        return processed_4d, metadata, ref_nii

    def predict(
        self,
        case_input: Union[str, Path, Dict[str, Any], np.ndarray],
        request_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        session_id: Optional[str] = None,
        save_mask_path: Optional[Union[str, Path]] = None,
        save_gtv_mask_path: Optional[Union[str, Path]] = None,
        ground_truth_mask: Optional[np.ndarray] = None,
        input_metadata: Optional[InputMetadata] = None,
        sw_batch_size: int = SW_BATCH_SIZE,
        overlap: float = OVERLAP,
        tumor_type: Optional[TumorType] = None,
    ) -> SegmentationResult:
        """
        Executes full-volume sliding-window inference, applies empty-air masking on prediction,
        derives multiclass and subregion masks and volumes (WT, TC, ET, NCR, ED), saves separate
        multiclass (0-3) and binary GTV (0-1) masks if requested, and returns the type-safe
        SegmentationResult data contract.
        """
        req_id = request_id or f"REQ-MEN-MULTI-{uuid.uuid4().hex[:8].upper()}"
        model = self.get_model()

        # Handle preprocessed numpy array or raw file/dict inputs
        ref_affine = np.eye(4)
        if isinstance(case_input, np.ndarray):
            if case_input.ndim != 4 or case_input.shape[0] != 4:
                raise ValueError(
                    f"Expected 4D numpy array with shape (4, D, H, W), got {case_input.shape}"
                )
            image_4d = case_input.astype(np.float32)
            d, h, w = image_4d.shape[1:]
            if input_metadata is None:
                input_metadata = InputMetadata(
                    modality_names=REQUIRED_MODALITIES,
                    original_shape=[d, h, w],
                    original_spacing=[1.0, 1.0, 1.0],
                    voxel_volume_mm3=1.0,
                    original_affine=[[float(c) for c in row] for row in ref_affine],
                    resampled_shape=[d, h, w],
                    resampled_spacing=[1.0, 1.0, 1.0],
                )
        else:
            image_4d, input_metadata, ref_nii = self.preprocess_inputs(case_input)
            ref_affine = ref_nii.affine

        d, h, w = image_4d.shape[1:]

        # Prepare tensor [1, 4, D, H, W]
        image_tensor = torch.from_numpy(image_4d).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
                logits = sliding_window_inference(
                    inputs=image_tensor,
                    roi_size=ROI_SIZE,
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                    overlap=overlap,
                    mode="gaussian",
                )

            # Argmax discrete class predictions [1, 4, D, H, W] -> [D, H, W]
            raw_pred = torch.argmax(logits, dim=1).squeeze(0).cpu()

            # Empty-air masking: Zero out predictions where all 4 channels are 0.0
            pred = apply_air_masking(raw_pred, image_tensor.squeeze(0).cpu())

        del image_tensor, logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        final_mask = pred.numpy().astype(np.uint8)

        # Derive BraTS subregions:
        # WT: labels 1, 2, 3
        # TC: labels 1, 3
        # ET: label 3
        # NCR: label 1
        # ED: label 2
        wt_mask = (final_mask >= 1) & (final_mask <= 3)
        tc_mask = (final_mask == 1) | (final_mask == 3)
        et_mask = (final_mask == 3)
        ncr_mask = (final_mask == 1)
        ed_mask = (final_mask == 2)

        wt_vox = int(wt_mask.sum())
        tc_vox = int(tc_mask.sum())
        et_vox = int(et_mask.sum())
        ncr_vox = int(ncr_mask.sum())
        ed_vox = int(ed_mask.sum())

        voxel_vol = float(input_metadata.voxel_volume_mm3)
        wt_mm3 = float(wt_vox * voxel_vol)
        tc_mm3 = float(tc_vox * voxel_vol)
        et_mm3 = float(et_vox * voxel_vol)
        ncr_mm3 = float(ncr_vox * voxel_vol)
        ed_mm3 = float(ed_vox * voxel_vol)

        # Optional mask file saving
        saved_mask_path = None
        saved_gtv_mask_path = None
        affine = np.asarray(input_metadata.original_affine) if input_metadata.original_affine else ref_affine

        if save_mask_path is not None:
            save_p = Path(save_mask_path)
            save_p.parent.mkdir(parents=True, exist_ok=True)
            out_nii = nib.Nifti1Image(final_mask, affine=affine)
            nib.save(out_nii, str(save_p))
            saved_mask_path = str(save_p)

            # Derive separate binary GTV mask path if not explicitly provided
            if save_gtv_mask_path is not None:
                gtv_p = Path(save_gtv_mask_path)
            else:
                fname = save_p.name
                if fname.endswith("_seg.nii.gz"):
                    gtv_fname = fname[:-11] + "_gtv.nii.gz"
                elif fname.endswith(".nii.gz"):
                    gtv_fname = fname[:-7] + "_gtv.nii.gz"
                elif save_p.suffix:
                    gtv_fname = f"{save_p.stem}_gtv{save_p.suffix}"
                else:
                    gtv_fname = f"{fname}_gtv.nii.gz"
                gtv_p = save_p.parent / gtv_fname

            gtv_p.parent.mkdir(parents=True, exist_ok=True)
            gtv_binary_arr = (final_mask > 0).astype(np.uint8)
            gtv_nii = nib.Nifti1Image(gtv_binary_arr, affine=affine)
            nib.save(gtv_nii, str(gtv_p))
            saved_gtv_mask_path = str(gtv_p)

        elif save_gtv_mask_path is not None:
            gtv_p = Path(save_gtv_mask_path)
            gtv_p.parent.mkdir(parents=True, exist_ok=True)
            gtv_binary_arr = (final_mask > 0).astype(np.uint8)
            gtv_nii = nib.Nifti1Image(gtv_binary_arr, affine=affine)
            nib.save(gtv_nii, str(gtv_p))
            saved_gtv_mask_path = str(gtv_p)

        multiclass_ref = MaskReference(
            mask_path=saved_mask_path,
            mask_format="nifti",
            is_original_space=True,
            shape=[d, h, w],
            non_zero_voxels=wt_vox,
        )

        gtv_ref = MaskReference(
            mask_path=saved_gtv_mask_path,
            mask_format="nifti",
            is_original_space=True,
            shape=[d, h, w],
            non_zero_voxels=wt_vox,
        )

        subregions = MeningiomaMultimodalSubregionVolumes(
            wt_voxels=wt_vox,
            wt_volume_mm3=wt_mm3,
            wt_volume_cm3=round(wt_mm3 / 1000.0, 4),
            tc_voxels=tc_vox,
            tc_volume_mm3=tc_mm3,
            tc_volume_cm3=round(tc_mm3 / 1000.0, 4),
            et_voxels=et_vox,
            et_volume_mm3=et_mm3,
            et_volume_cm3=round(et_mm3 / 1000.0, 4),
            ncr_net_voxels=ncr_vox,
            ncr_net_volume_mm3=ncr_mm3,
            ncr_net_volume_cm3=round(ncr_mm3 / 1000.0, 4),
            edema_voxels=ed_vox,
            edema_volume_mm3=ed_mm3,
            edema_volume_cm3=round(ed_mm3 / 1000.0, 4),
        )

        meningioma_multimodal = MeningiomaMultimodalMeasurements(
            subregions=subregions,
            masks=GliomaSubregionMasks(multiclass_mask=multiclass_ref),
            label_definitions=LABEL_DEFINITIONS,
            is_original_space_restored=True,
            disclaimer=RESEARCH_DISCLAIMER,
        )

        # Legacy backward-compatible GTV payload (GTV == Whole Tumor, binary labels 0 and 1)
        meningioma_legacy = MeningiomaMeasurements(
            gtv_voxels=wt_vox,
            gtv_volume_mm3=wt_mm3,
            gtv_volume_cm3=round(wt_mm3 / 1000.0, 4),
            gtv_mask=gtv_ref,
            is_original_space_restored=True,
            label_definitions={0: "Background", 1: "Whole Tumor (GTV composite)"},
        )

        volumetrics = VolumetricSummary(
            total_tumor_voxels=wt_vox,
            total_tumor_volume_mm3=wt_mm3,
            total_tumor_volume_cm3=round(wt_mm3 / 1000.0, 4),
            voxel_dimensions_mm=list(input_metadata.original_spacing),
        )

        quality_status = InferenceQualityStatus(
            status=PipelineStatus.SUCCESS if wt_vox > 0 else PipelineStatus.WARNING,
            inference_success=True,
            pipeline_frozen=True,
            checkpoint_used=str(self.checkpoint_path),
            threshold_applied=0.0,  # Multiclass argmax across logits (no probability threshold applied)
            min_component_size_voxels=0,  # Evaluated without connected-component noise filtering
            qc_verdict=QC_VERDICT_BENCHMARK,
            qc_notes=(
                "4-channel multimodal 3D U-Net evaluated on 200 validation cases "
                "(Macro WT Dice: 0.7991, Median WT Dice: 0.9006, Pooled WT Dice: 0.9273; "
                "note: in-training 25-case validation subset achieved 0.8312 Macro WT Dice). "
                "Multiclass argmax inference with empty-air masking; no connected-component filtering applied. "
                + RESEARCH_DISCLAIMER
            ),
            warnings=["Zero tumor voxels detected."] if wt_vox == 0 else [],
        )

        # Optional offline benchmarking if ground-truth mask is provided
        offline_eval = None
        if ground_truth_mask is not None:
            gt_arr = np.asanyarray(ground_truth_mask).astype(np.int64)
            gt_wt = (gt_arr >= 1) & (gt_arr <= 3)
            gt_tc = (gt_arr == 1) | (gt_arr == 3)
            gt_et = (gt_arr == 3)

            def _dice(p, g):
                tp = int((p & g).sum())
                denom = int(p.sum()) + int(g.sum())
                return float(2.0 * tp / denom) if denom > 0 else 1.0

            def _iou(p, g):
                tp = int((p & g).sum())
                denom = int((p | g).sum())
                return float(tp / denom) if denom > 0 else 1.0

            offline_eval = OfflineEvaluationMetrics(
                has_ground_truth=True,
                dice={"WT": _dice(wt_mask, gt_wt), "TC": _dice(tc_mask, gt_tc), "ET": _dice(et_mask, gt_et)},
                iou={"WT": _iou(wt_mask, gt_wt), "TC": _iou(tc_mask, gt_tc), "ET": _iou(et_mask, gt_et)},
                notes="Offline evaluation computed against provided ground-truth mask.",
            )

        res_tumor_type = tumor_type or TumorType.MENINGIOMA_MULTIMODAL
        return SegmentationResult(
            request=RequestMetadata(
                request_id=req_id,
                tumor_type=res_tumor_type,
                patient_id=patient_id,
                session_id=session_id,
            ),
            input_metadata=input_metadata,
            volumetrics=volumetrics,
            quality_status=quality_status,
            glioma=None,
            meningioma=meningioma_legacy,
            meningioma_multimodal=meningioma_multimodal,
            offline_evaluation=offline_eval,
        )


def run_meningioma_multimodal_inference(
    case_input: Union[str, Path, Dict[str, Any], np.ndarray],
    **kwargs: Any,
) -> SegmentationResult:
    """Convenience functional interface for Multimodal Meningioma inference."""
    service = MeningiomaMultimodalInferenceService()
    return service.predict(case_input, **kwargs)
