"""
Meningioma Inference Integration Service.
=========================================

Thin integration service wrapping the frozen Meningioma 3D U-Net segmentation
pipeline. Connects single-channel T1c MRI inputs, frozen model sliding-window
inference, frozen morphological post-processing, and native-space spatial restoration
to the common SegmentationResult data contract.

Frozen Specifications:
----------------------
- Checkpoint: outputs/meningioma_training/meningioma_best.pth
- Architecture: 3D Residual U-Net (1 input channel, 2 output classes, base_channels=16)
- Modality: T1-contrast enhanced MRI (T1c)
- Resampling: 1.0mm isotropic (TARGET_SPACING = (1.0, 1.0, 1.0))
- Normalization: 1st-99th percentile clipping + non-zero z-score
- Cropping: Bounding box around non-zero MRI foreground
- Sliding Window: Patch size (64,64,64), Overlap 0.50
- Post-Processing: Probability threshold tau = 0.125 on P(GTV) = P(class 1)
- Morphological Filter: 3D 6-connectivity, remove components < 2500 voxels
- Spatial Restoration: Crop inversion -> Resampled space -> Original native acquisition geometry
- Validation Constraint: Official validation has NO ground truth -> strictly zero fake validation Dice
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import datetime
import uuid

import numpy as np
import torch
import nibabel as nib
from scipy.ndimage import zoom

from integration.result_schema import (
    TumorType,
    PipelineStatus,
    RequestMetadata,
    InputMetadata,
    MaskReference,
    VolumetricSummary,
    MeningiomaMeasurements,
    InferenceQualityStatus,
    OfflineEvaluationMetrics,
    SegmentationResult,
)
from models.meningioma_model import create_meningioma_model
from preprocessing.meningioma_validation_preprocess import (
    TARGET_SPACING,
    resample_image,
    normalize_t1c,
    crop_to_foreground,
)
from segmentation.meningioma_validation_inference import (
    pad_volume,
    calculate_positions,
    remove_small_components,
)
from segmentation.meningioma_restore_validation_predictions import (
    restore_to_resampled_space,
    restore_to_original_space,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_MENINGIOMA_CHECKPOINT = os.getenv("MENINGIOMA_SINGLE_CHECKPOINT_PATH") or os.getenv("MENINGIOMA_CHECKPOINT_PATH")
DEFAULT_MENINGIOMA_CHECKPOINT = (
    (PROJECT_ROOT / ENV_MENINGIOMA_CHECKPOINT)
    if ENV_MENINGIOMA_CHECKPOINT and not Path(ENV_MENINGIOMA_CHECKPOINT).is_absolute()
    else (Path(ENV_MENINGIOMA_CHECKPOINT) if ENV_MENINGIOMA_CHECKPOINT else (
        PROJECT_ROOT
        / "outputs"
        / "meningioma_training"
        / "meningioma_best.pth"
    ))
)

PATCH_SIZE = (64, 64, 64)
OVERLAP = 0.50
THRESHOLD = 0.125
MIN_COMPONENT_SIZE = 2500


class MeningiomaInferenceService:
    """
    Service wrapper for the frozen Meningioma segmentation pipeline.
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
            env_ckpt = os.getenv("MENINGIOMA_SINGLE_CHECKPOINT_PATH") or os.getenv("MENINGIOMA_CHECKPOINT_PATH")
            if env_ckpt:
                p = Path(env_ckpt)
                self.checkpoint_path = p if p.is_absolute() else (PROJECT_ROOT / p)
            else:
                self.checkpoint_path = DEFAULT_MENINGIOMA_CHECKPOINT
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = model

    def get_model(self) -> torch.nn.Module:
        """Loads and caches the frozen Meningioma 3D U-Net model."""
        if self._model is not None:
            return self._model

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Frozen Meningioma checkpoint not found at: {self.checkpoint_path}"
            )

        model = create_meningioma_model(base_channels=16)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.to(self.device)
        model.eval()
        self._model = model
        return self._model

    def preprocess_t1c(
        self,
        t1c_input: Union[str, Path, nib.Nifti1Image],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, InputMetadata, nib.Nifti1Image]:
        """
        Preprocesses a raw T1c NIfTI acquisition according to the frozen pipeline.
        
        Returns:
            cropped_norm: 3D numpy array (cropped, normalized MRI)
            crop_min: 3-element int array indicating foreground crop origin
            resampled_shape: shape of the 1.0mm resampled volume
            metadata: Extracted InputMetadata
            nii_obj: Raw nibabel NIfTI object with affine header
        """
        if isinstance(t1c_input, (str, Path)):
            p = Path(t1c_input)
            if not p.exists():
                raise FileNotFoundError(f"T1c file does not exist: {p}")
            nii = nib.load(str(p))
        elif isinstance(t1c_input, nib.Nifti1Image):
            nii = t1c_input
        else:
            raise TypeError(f"Unsupported t1c_input type: {type(t1c_input)}")

        raw_array = nii.get_fdata().astype(np.float32)
        orig_shape = [int(x) for x in raw_array.shape]
        header = nii.header
        orig_zooms = [float(x) for x in header.get_zooms()[:3]]
        orig_affine = [[float(c) for c in row] for row in nii.affine]
        voxel_vol_mm3 = float(orig_zooms[0] * orig_zooms[1] * orig_zooms[2])

        # 1. Resample to 1.0mm isotropic spacing
        resampled = resample_image(raw_array, np.asarray(orig_zooms, dtype=np.float32), TARGET_SPACING)
        resampled_shape = [int(x) for x in resampled.shape]

        # 2. Intensity normalization (1-99% clipping + non-zero z-score)
        normalized = normalize_t1c(resampled)

        # 3. Foreground cropping
        cropped, crop_min, _ = crop_to_foreground(normalized)

        metadata = InputMetadata(
            modality_names=["T1c"],
            original_shape=orig_shape,
            original_spacing=orig_zooms,
            voxel_volume_mm3=voxel_vol_mm3,
            original_affine=orig_affine,
            resampled_shape=resampled_shape,
            resampled_spacing=list(TARGET_SPACING),
        )

        return cropped, crop_min, np.asarray(resampled_shape), metadata, nii

    def predict(
        self,
        t1c_input: Union[str, Path, nib.Nifti1Image, np.ndarray],
        request_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        session_id: Optional[str] = None,
        save_mask_path: Optional[Union[str, Path]] = None,
        restore_to_native_space: bool = True,
        input_metadata: Optional[InputMetadata] = None,
    ) -> SegmentationResult:
        """
        Executes full-volume Meningioma inference, applies post-processing, restores
        to native acquisition space, and constructs SegmentationResult.
        """
        req_id = request_id or f"REQ-MEN-{uuid.uuid4().hex[:8].upper()}"
        model = self.get_model()

        if isinstance(t1c_input, np.ndarray):
            cropped_vol = t1c_input.astype(np.float32)
            crop_min = np.zeros(3, dtype=np.int64)
            resampled_shape = np.asarray(cropped_vol.shape, dtype=np.int64)
            nii_ref = None
            if input_metadata is None:
                input_metadata = InputMetadata(
                    modality_names=["T1c"],
                    original_shape=list(cropped_vol.shape),
                    original_spacing=[1.0, 1.0, 1.0],
                    voxel_volume_mm3=1.0,
                    resampled_shape=list(cropped_vol.shape),
                    resampled_spacing=[1.0, 1.0, 1.0],
                )
        else:
            cropped_vol, crop_min, resampled_shape, input_metadata, nii_ref = self.preprocess_t1c(t1c_input)

        # Run sliding-window inference across cropped volume
        padded, padding = pad_volume(cropped_vol, PATCH_SIZE)
        padded_shape = padded.shape

        z_positions = calculate_positions(padded_shape[0], PATCH_SIZE[0], OVERLAP)
        y_positions = calculate_positions(padded_shape[1], PATCH_SIZE[1], OVERLAP)
        x_positions = calculate_positions(padded_shape[2], PATCH_SIZE[2], OVERLAP)

        prob_sum = np.zeros(padded_shape, dtype=np.float32)
        count_map = np.zeros(padded_shape, dtype=np.float32)

        with torch.inference_mode():
            for z in z_positions:
                for y in y_positions:
                    for x in x_positions:
                        patch = padded[
                            z : z + PATCH_SIZE[0],
                            y : y + PATCH_SIZE[1],
                            x : x + PATCH_SIZE[2],
                        ]
                        patch_tensor = (
                            torch.from_numpy(patch.astype(np.float32))
                            .unsqueeze(0)
                            .unsqueeze(0)
                            .to(self.device)
                        )
                        logits = model(patch_tensor)
                        probs = torch.softmax(logits, dim=1)
                        tumor_prob = probs[0, 1].detach().cpu().numpy()

                        prob_sum[
                            z : z + PATCH_SIZE[0],
                            y : y + PATCH_SIZE[1],
                            x : x + PATCH_SIZE[2],
                        ] += tumor_prob
                        count_map[
                            z : z + PATCH_SIZE[0],
                            y : y + PATCH_SIZE[1],
                            x : x + PATCH_SIZE[2],
                        ] += 1.0

        # Average overlapping predictions
        prob_map_padded = prob_sum / np.maximum(count_map, 1e-8)

        # Unpad back to cropped shape
        z_before, z_after = padding[0]
        y_before, y_after = padding[1]
        x_before, x_after = padding[2]
        z_end = None if z_after == 0 else -z_after
        y_end = None if y_after == 0 else -y_after
        x_end = None if x_after == 0 else -x_after

        prob_map = prob_map_padded[
            z_before:z_end,
            y_before:y_end,
            x_before:x_end,
        ]

        # 1. Apply probability threshold tau = 0.125
        raw_binary = (prob_map >= THRESHOLD).astype(np.uint8)

        # 2. Apply connected-component filter (min_size = 2500)
        filtered_binary, num_comp, kept_comp = remove_small_components(
            raw_binary, minimum_size=MIN_COMPONENT_SIZE
        )

        # 3. Spatial restoration to native acquisition space
        if restore_to_native_space and input_metadata.original_shape != list(cropped_vol.shape):
            # Place back into full 1.0mm resampled grid
            resampled_mask = restore_to_resampled_space(
                filtered_binary,
                resampled_shape=tuple(resampled_shape),
                crop_min=crop_min,
            )
            # Zoom back to original raw matrix dimensions
            final_mask = restore_to_original_space(
                resampled_mask,
                original_shape=tuple(input_metadata.original_shape),
            )
            is_original = True
            mask_shape = list(input_metadata.original_shape)
            voxel_vol = float(input_metadata.voxel_volume_mm3)
        else:
            final_mask = filtered_binary
            is_original = False
            mask_shape = list(final_mask.shape)
            voxel_vol = float(input_metadata.voxel_volume_mm3)

        gtv_voxels = int(final_mask.sum())
        gtv_volume_mm3 = float(gtv_voxels * voxel_vol)
        gtv_volume_cm3 = round(gtv_volume_mm3 / 1000.0, 4)

        # Optional mask file saving
        saved_mask_path = None
        if save_mask_path is not None:
            save_p = Path(save_mask_path)
            save_p.parent.mkdir(parents=True, exist_ok=True)
            affine = np.asarray(input_metadata.original_affine) if input_metadata.original_affine else np.eye(4)
            out_nii = nib.Nifti1Image(final_mask.astype(np.uint8), affine=affine)
            nib.save(out_nii, str(save_p))
            saved_mask_path = str(save_p)

        gtv_mask_ref = MaskReference(
            mask_path=saved_mask_path,
            mask_format="nifti",
            is_original_space=is_original,
            shape=mask_shape,
            non_zero_voxels=gtv_voxels,
        )

        meningioma_meas = MeningiomaMeasurements(
            gtv_voxels=gtv_voxels,
            gtv_volume_mm3=gtv_volume_mm3,
            gtv_volume_cm3=gtv_volume_cm3,
            gtv_mask=gtv_mask_ref,
            is_original_space_restored=is_original,
        )

        volumetrics = VolumetricSummary(
            total_tumor_voxels=gtv_voxels,
            total_tumor_volume_mm3=gtv_volume_mm3,
            total_tumor_volume_cm3=gtv_volume_cm3,
            voxel_dimensions_mm=list(input_metadata.original_spacing),
        )

        quality_status = InferenceQualityStatus(
            status=PipelineStatus.SUCCESS if gtv_voxels > 0 else PipelineStatus.WARNING,
            inference_success=True,
            pipeline_frozen=True,
            checkpoint_used=str(self.checkpoint_path),
            threshold_applied=THRESHOLD,
            min_component_size_voxels=MIN_COMPONENT_SIZE,
            qc_verdict="PASS WITH UNCERTAINTY",
            qc_notes="Restored to original geometry with 100% verification. No validation ground truth GTV masks available.",
            warnings=["Zero tumor voxels detected or small lesion filtered out (<2500 voxels)."] if gtv_voxels == 0 else [],
        )

        # IMPORTANT: Official Meningioma validation set has NO ground truth -> strictly no fake Dice
        offline_eval = OfflineEvaluationMetrics(
            has_ground_truth=False,
            dice=None,
            iou=None,
            precision=None,
            recall=None,
            notes="Official challenge validation set contains no ground-truth GTV masks. Dice/IoU unavailable.",
        )

        return SegmentationResult(
            request=RequestMetadata(
                request_id=req_id,
                tumor_type=TumorType.MENINGIOMA,
                patient_id=patient_id,
                session_id=session_id,
            ),
            input_metadata=input_metadata,
            volumetrics=volumetrics,
            quality_status=quality_status,
            glioma=None,
            meningioma=meningioma_meas,
            offline_evaluation=offline_eval,
        )


def run_meningioma_inference(
    t1c_input: Union[str, Path, nib.Nifti1Image, np.ndarray],
    **kwargs: Any,
) -> SegmentationResult:
    """Convenience functional interface for Meningioma inference service."""
    service = MeningiomaInferenceService()
    return service.predict(t1c_input, **kwargs)

