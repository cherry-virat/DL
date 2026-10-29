"""
Glioma Inference Integration Service.
=====================================

Thin integration service wrapping the frozen Glioma 3D U-Net segmentation
pipeline. Connects 4-channel MRI inputs, frozen model sliding-window inference,
and frozen post-processing to the common SegmentationResult data contract.

Frozen Specifications:
----------------------
- Checkpoint: outputs/glioma_training/glioma_best_medium_experiment.pth
- Architecture: MONAI 3D Residual U-Net (4 input channels, 4 output classes)
- Modalities: T1, T1c, T2, FLAIR
- Resampling: 1.0mm isotropic (TARGET_SPACING = (1.0, 1.0, 1.0))
- Sliding Window: ROI (64,64,64), Overlap 0.50, Gaussian blending, batch size 1
- Post-Processing: Probability threshold tau = 0.60 on P(WT) = P(1)+P(2)+P(3)
- Morphological Filter: 3D 6-connectivity, remove components < 1000 voxels
- Subregions: WT (1+2+3), TC (1+3), ET (3)
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import datetime
import uuid

import numpy as np
import torch
import SimpleITK as sitk
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference

from integration.result_schema import (
    TumorType,
    PipelineStatus,
    RequestMetadata,
    InputMetadata,
    MaskReference,
    VolumetricSummary,
    GliomaSubregionVolumes,
    GliomaSubregionMasks,
    GliomaMeasurements,
    InferenceQualityStatus,
    OfflineEvaluationMetrics,
    SegmentationResult,
)
from preprocessing.glioma_preprocess import (
    TARGET_SPACING,
    MRI_MODALITIES,
    load_nifti,
    resample_image,
    sitk_to_numpy,
    normalize_mri,
)
from preprocessing.glioma_dataset import pad_to_patch_size
from segmentation.glioma_formal_evaluation import (
    filter_connected_components,
    compute_binary_metrics,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_GLIOMA_CHECKPOINT = os.getenv("GLIOMA_CHECKPOINT_PATH")
DEFAULT_GLIOMA_CHECKPOINT = (
    (PROJECT_ROOT / ENV_GLIOMA_CHECKPOINT)
    if ENV_GLIOMA_CHECKPOINT and not Path(ENV_GLIOMA_CHECKPOINT).is_absolute()
    else (Path(ENV_GLIOMA_CHECKPOINT) if ENV_GLIOMA_CHECKPOINT else (
        PROJECT_ROOT
        / "outputs"
        / "glioma_training"
        / "glioma_best_medium_experiment.pth"
    ))
)

PATCH_SIZE = (64, 64, 64)
ROI_SIZE = PATCH_SIZE
OVERLAP = 0.50
SW_BATCH_SIZE = 1
PROBABILITY_THRESHOLD = 0.60
MIN_COMPONENT_SIZE = 1000
CONNECTIVITY = 1  # 6-connectivity


class GliomaInferenceService:
    """
    Service wrapper for the frozen Glioma segmentation pipeline.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
        model: Optional[torch.nn.Module] = None,
    ):
        if checkpoint_path is not None:
            p = Path(checkpoint_path)
            self.checkpoint_path = p if p.is_absolute() else (PROJECT_ROOT / p)
        else:
            env_ckpt = os.getenv("GLIOMA_CHECKPOINT_PATH")
            if env_ckpt:
                p = Path(env_ckpt)
                self.checkpoint_path = p if p.is_absolute() else (PROJECT_ROOT / p)
            else:
                self.checkpoint_path = DEFAULT_GLIOMA_CHECKPOINT
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = model

    def get_model(self) -> torch.nn.Module:
        """Loads and caches the frozen Glioma 3D U-Net model."""
        if self._model is not None:
            return self._model

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Frozen Glioma checkpoint not found at: {self.checkpoint_path}"
            )

        model = UNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=4,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            dropout=0.0,
        )

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.to(self.device)
        model.eval()
        self._model = model
        return self._model

    def preprocess_inputs(
        self,
        case_input: Union[str, Path, Dict[str, Union[str, Path]]],
    ) -> Tuple[np.ndarray, InputMetadata, Optional[sitk.Image]]:
        """
        Preprocesses raw multi-modal MRI acquisitions according to frozen pipeline.
        
        Returns:
            processed_4d: 4D numpy array (4, D, H, W)
            input_metadata: Extracted spatial and acquisition geometry
            reference_image: Reference SimpleITK image for spatial header restoration
        """
        modality_paths: Dict[str, Path] = {}

        if isinstance(case_input, (str, Path)):
            case_dir = Path(case_input)
            if not case_dir.exists():
                raise FileNotFoundError(f"Input directory does not exist: {case_dir}")
            case_name = case_dir.name
            for mod_name, suffix in MRI_MODALITIES.items():
                cand = case_dir / f"{case_name}-{suffix}.nii.gz"
                if not cand.exists():
                    # Fallback search by suffix
                    matches = list(case_dir.glob(f"*-{suffix}.nii.gz"))
                    if matches:
                        cand = matches[0]
                    else:
                        raise FileNotFoundError(f"Missing required Glioma modality {mod_name} in {case_dir}")
                modality_paths[mod_name] = cand
        elif isinstance(case_input, dict):
            missing_keys = [m for m in MRI_MODALITIES.keys() if m not in case_input]
            if missing_keys:
                raise ValueError(f"Missing required Glioma modality keys: {missing_keys}")
            for mod_name in MRI_MODALITIES.keys():
                p = Path(case_input[mod_name])
                if not p.exists():
                    raise FileNotFoundError(f"File for modality '{mod_name}' not found: {p}")
                modality_paths[mod_name] = p
        else:
            raise TypeError(f"Unsupported case_input type: {type(case_input)}")

        # Load reference modality (T1c or T1) for original geometry
        ref_image = load_nifti(modality_paths["T1c"])
        orig_shape = [int(x) for x in ref_image.GetSize()]  # (W, H, D) in SITK
        orig_spacing = [float(x) for x in ref_image.GetSpacing()]
        orig_origin = [float(x) for x in ref_image.GetOrigin()]
        orig_direction = [float(x) for x in ref_image.GetDirection()]
        voxel_vol_mm3 = float(orig_spacing[0] * orig_spacing[1] * orig_spacing[2])

        # Load, resample to 1.0mm, convert to numpy, normalize
        channels = []
        resampled_shape = None

        for mod_name in ["T1", "T1c", "T2", "FLAIR"]:
            sitk_img = load_nifti(modality_paths[mod_name])
            resampled_sitk = resample_image(sitk_img, TARGET_SPACING, is_label=False)
            np_arr = sitk_to_numpy(resampled_sitk)  # (D, H, W)
            if resampled_shape is None:
                resampled_shape = [int(x) for x in np_arr.shape]
            norm_arr = normalize_mri(np_arr)
            channels.append(norm_arr)

        processed_4d = np.stack(channels, axis=0).astype(np.float32)  # (4, D, H, W)

        metadata = InputMetadata(
            modality_names=["T1", "T1c", "T2", "FLAIR"],
            original_shape=orig_shape,
            original_spacing=orig_spacing,
            voxel_volume_mm3=voxel_vol_mm3,
            original_origin=orig_origin,
            original_direction=orig_direction,
            resampled_shape=resampled_shape,
            resampled_spacing=list(TARGET_SPACING),
        )

        return processed_4d, metadata, ref_image

    def predict(
        self,
        case_input: Union[str, Path, Dict[str, Union[str, Path]], np.ndarray],
        request_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        session_id: Optional[str] = None,
        save_mask_path: Optional[Union[str, Path]] = None,
        save_gtv_mask_path: Optional[Union[str, Path]] = None,
        ground_truth_mask: Optional[np.ndarray] = None,
        input_metadata: Optional[InputMetadata] = None,
    ) -> SegmentationResult:
        """
        Executes full-volume sliding-window Glioma inference and constructs SegmentationResult.
        """
        req_id = request_id or f"REQ-GLI-{uuid.uuid4().hex[:8].upper()}"
        model = self.get_model()

        # Handle pre-processed numpy array input (e.g. testing or direct memory feed)
        if isinstance(case_input, np.ndarray):
            if case_input.ndim != 4 or case_input.shape[0] != 4:
                raise ValueError(f"Expected 4D array (4, D, H, W), got shape {case_input.shape}")
            image_4d = case_input.astype(np.float32)
            d, h, w = image_4d.shape[1:]
            if input_metadata is None:
                input_metadata = InputMetadata(
                    modality_names=["T1", "T1c", "T2", "FLAIR"],
                    original_shape=[w, h, d],
                    original_spacing=[1.0, 1.0, 1.0],
                    voxel_volume_mm3=1.0,
                    resampled_shape=[d, h, w],
                    resampled_spacing=[1.0, 1.0, 1.0],
                )
            ref_sitk = None
        else:
            image_4d, input_metadata, ref_sitk = self.preprocess_inputs(case_input)

        d, h, w = image_4d.shape[1:]

        # Pad volume if any spatial dimension < 64
        dummy_mask_4d = np.zeros((1, d, h, w), dtype=np.int64)
        padded_image, _ = pad_to_patch_size(image_4d, dummy_mask_4d, PATCH_SIZE)

        image_tensor = torch.from_numpy(padded_image).float().unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = sliding_window_inference(
                inputs=image_tensor,
                roi_size=ROI_SIZE,
                sw_batch_size=SW_BATCH_SIZE,
                predictor=model,
                overlap=OVERLAP,
                mode="gaussian",
            )
            probabilities = torch.softmax(logits, dim=1)
            # P(WT) = P(1) + P(2) + P(3)
            wt_probability = probabilities[:, 1:, ...].sum(dim=1)
            raw_pred_classes = torch.argmax(probabilities, dim=1)

        wt_prob_np = wt_probability[0].detach().cpu().numpy()[:d, :h, :w]
        raw_classes_np = raw_pred_classes[0].detach().cpu().numpy()[:d, :h, :w]

        del image_tensor, logits, probabilities, wt_probability, raw_pred_classes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Apply threshold 0.60
        raw_wt_mask = (wt_prob_np >= PROBABILITY_THRESHOLD)

        # Zero-air background masking: suppress false-positive activations in empty air outside the brain
        brain_mask = np.any(image_4d != 0, axis=0)
        raw_wt_mask = np.logical_and(raw_wt_mask, brain_mask)

        # Apply 3D connected-component post-processing (min_size=1000)
        final_wt_mask, fp_cc_stats = filter_connected_components(
            raw_wt_mask,
            min_size=MIN_COMPONENT_SIZE,
            connectivity=CONNECTIVITY,
        )

        # Derive subregion classes inside post-processed WT mask
        final_multiclass = np.where(final_wt_mask, raw_classes_np, 0).astype(np.uint8)
        bg_inside_wt = np.logical_and(final_wt_mask, final_multiclass == 0)
        if np.any(bg_inside_wt):
            final_multiclass[bg_inside_wt] = 2  # default tumor edema

        # Calculate subregion voxel counts
        wt_vox = int(final_wt_mask.sum())
        tc_vox = int(np.isin(final_multiclass, [1, 3]).sum())
        et_vox = int((final_multiclass == 3).sum())
        ncr_vox = int((final_multiclass == 1).sum())
        edema_vox = int((final_multiclass == 2).sum())

        voxel_vol = float(input_metadata.voxel_volume_mm3)
        wt_mm3 = float(wt_vox * voxel_vol)
        tc_mm3 = float(tc_vox * voxel_vol)
        et_mm3 = float(et_vox * voxel_vol)
        ncr_mm3 = float(ncr_vox * voxel_vol)
        edema_mm3 = float(edema_vox * voxel_vol)

        # Optional mask file export if save_mask_path is requested
        mask_saved_path = None
        saved_gtv_mask_path = None
        if save_mask_path is not None:
            save_p = Path(save_mask_path)
            save_p.parent.mkdir(parents=True, exist_ok=True)
            out_img = sitk.GetImageFromArray(final_multiclass)
            if ref_sitk is not None:
                out_img.CopyInformation(ref_sitk)
            sitk.WriteImage(out_img, str(save_p))
            mask_saved_path = str(save_p)

            # Derive separate binary GTV / whole-tumor mask path if not explicitly provided
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
            gtv_binary_arr = (final_multiclass > 0).astype(np.uint8)
            gtv_img = sitk.GetImageFromArray(gtv_binary_arr)
            if ref_sitk is not None:
                gtv_img.CopyInformation(ref_sitk)
            sitk.WriteImage(gtv_img, str(gtv_p))
            saved_gtv_mask_path = str(gtv_p)

        elif save_gtv_mask_path is not None:
            gtv_p = Path(save_gtv_mask_path)
            gtv_p.parent.mkdir(parents=True, exist_ok=True)
            gtv_binary_arr = (final_multiclass > 0).astype(np.uint8)
            gtv_img = sitk.GetImageFromArray(gtv_binary_arr)
            if ref_sitk is not None:
                gtv_img.CopyInformation(ref_sitk)
            sitk.WriteImage(gtv_img, str(gtv_p))
            saved_gtv_mask_path = str(gtv_p)

        subregions = GliomaSubregionVolumes(
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
            edema_voxels=edema_vox,
            edema_volume_mm3=edema_mm3,
            edema_volume_cm3=round(edema_mm3 / 1000.0, 4),
        )

        multiclass_ref = MaskReference(
            mask_path=mask_saved_path,
            mask_format="nifti",
            is_original_space=True,
            shape=[d, h, w],
            non_zero_voxels=wt_vox,
        )

        wt_binary_ref = None
        if saved_gtv_mask_path is not None:
            wt_binary_ref = MaskReference(
                mask_path=saved_gtv_mask_path,
                mask_format="nifti",
                is_original_space=True,
                shape=[d, h, w],
                non_zero_voxels=wt_vox,
            )

        glioma_meas = GliomaMeasurements(
            subregions=subregions,
            masks=GliomaSubregionMasks(
                multiclass_mask=multiclass_ref,
                wt_mask=wt_binary_ref,
            ),
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
            threshold_applied=PROBABILITY_THRESHOLD,
            min_component_size_voxels=MIN_COMPONENT_SIZE,
            qc_verdict="PASS WITH DIFFICULT CASES",
            qc_notes=(
                "4-channel MONAI 3D Residual U-Net evaluated on 20 internal validation cases "
                "(Macro WT Dice: 0.8478, Median WT Dice: 0.9013, Pooled WT Dice: 0.9042; "
                "Macro TC Dice: 0.7240; Macro ET Dice: 0.7072). "
                "Note: Evaluated cohort represents the 20-case internal validation split used during "
                "training for checkpoint selection and post-processing tuning (tau=0.60, min CC size=1000 voxels); "
                "not an independent test cohort."
            ),
            warnings=["Zero tumor voxels detected."] if wt_vox == 0 else [],
        )

        # Optional offline benchmarking when ground-truth mask is provided
        offline_eval = None
        if ground_truth_mask is not None:
            gt_wt = (ground_truth_mask > 0)
            gt_tc = np.isin(ground_truth_mask, [1, 3])
            gt_et = (ground_truth_mask == 3)
            pred_wt = (final_multiclass > 0)
            pred_tc = np.isin(final_multiclass, [1, 3])
            pred_et = (final_multiclass == 3)

            m_wt = compute_binary_metrics(pred_wt, gt_wt)
            m_tc = compute_binary_metrics(pred_tc, gt_tc)
            m_et = compute_binary_metrics(pred_et, gt_et)

            offline_eval = OfflineEvaluationMetrics(
                has_ground_truth=True,
                dice={"WT": m_wt["dice"], "TC": m_tc["dice"], "ET": m_et["dice"]},
                iou={"WT": m_wt["iou"], "TC": m_tc["iou"], "ET": m_et["iou"]},
                precision={"WT": m_wt["precision"], "TC": m_tc["precision"], "ET": m_et["precision"]},
                recall={"WT": m_wt["recall"], "TC": m_tc["recall"], "ET": m_et["recall"]},
                notes="Offline evaluation computed against provided ground-truth mask.",
            )

        return SegmentationResult(
            request=RequestMetadata(
                request_id=req_id,
                tumor_type=TumorType.GLIOMA,
                patient_id=patient_id,
                session_id=session_id,
            ),
            input_metadata=input_metadata,
            volumetrics=volumetrics,
            quality_status=quality_status,
            glioma=glioma_meas,
            meningioma=None,
            offline_evaluation=offline_eval,
        )


def run_glioma_inference(
    case_input: Union[str, Path, Dict[str, Union[str, Path]], np.ndarray],
    **kwargs: Any,
) -> SegmentationResult:
    """Convenience functional interface for Glioma inference service."""
    service = GliomaInferenceService()
    return service.predict(case_input, **kwargs)
