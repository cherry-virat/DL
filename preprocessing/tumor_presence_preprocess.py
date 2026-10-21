"""
Standalone 2-Channel 3D Brain Tumor Presence Preprocessing Engine.
==================================================================

Independent spatial and intensity transformation module designed for 2-channel
structural brain MRI classification (Healthy Control vs. Tumor Present).

Design Principles & Policies:
-----------------------------
1. Two-Modality Standard: Uses exactly two structural sequences: T1-weighted (T1) and
   T2-weighted (T2). Unlike post-contrast (T1c) or FLAIR, T1 and T2 are universally available
   across both clinical tumor repositories (BraTS GLI & MEN) and normative healthy databases (IXI).
2. Zero Artificial Imputation: Strictly operates on genuine T1 and T2 sequences. Does NOT
   duplicate channels or zero-fill absent acquisitions.
3. Decoupled Pipeline: Completely independent of frozen segmentation pipelines and the 4-channel
   glioma-vs-meningioma classifier.
4. Consistent Geometry Policy:
   - BraTS scans are pre-aligned, resampled to 1mm isotropic space, and coregistered.
   - IXI scans possess divergent native acquisition planes (T1 sagittal, T2 axial) and slice counts.
   - When auto_align=True, the engine uses medical image registration (SimpleITK mutual-information
     rigid alignment or affine resampling) to map T2 into T1's physical coordinate space before
     standardization, preventing unaligned channel stacking.
   - Standardizes combined volumes into (2, 128, 128, 96) float32 tensors with isotropic 1.875mm spacing,
     symmetrically padding the Z-axis (155 -> 180) before trilinear scaling if in standard BraTS shape.
5. Intensity Standardization & Brain-Extraction Status:
   - Applies per-channel non-zero Z-score normalization on voxels > 0.
   - CRITICAL AUDIT NOTICE: This engine does NOT perform skull-stripping or brain extraction.
     BraTS scans are pre-skull-stripped, whereas raw IXI scans contain whole-head anatomy (skull, scalp,
     neck, eyes). Non-zero Z-score normalization alone does NOT eliminate skull confounding.
     Rigorous brain extraction (e.g. via HD-BET or ROBEX) remains a prerequisite before supervised training.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import tempfile

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False

from scipy import ndimage

# Canonical 2-channel ordering
CANONICAL_PRESENCE_MODALITIES: List[str] = ["t1", "t2"]

# Geometric target parameters
ORIGINAL_SHAPE: Tuple[int, int, int] = (240, 240, 155)
PADDED_SHAPE: Tuple[int, int, int] = (240, 240, 180)
TARGET_SHAPE: Tuple[int, int, int] = (128, 128, 96)
ORIGINAL_SPACING: Tuple[float, float, float] = (1.0, 1.0, 1.0)
TARGET_SPACING: Tuple[float, float, float] = (1.875, 1.875, 1.875)
PAD_Z_TOP: int = 12
PAD_Z_BOTTOM: int = 13

# Numerical tolerances for inter-modality alignment
DEFAULT_AFFINE_TOLERANCE: float = 1e-3
DEFAULT_SPACING_TOLERANCE: float = 1e-3


class SpatialAlignmentError(ValueError):
    """Raised when T1 and T2 modalities exhibit incompatible affines, spacings, or orientations."""
    pass


@dataclass(frozen=True)
class PresenceGeometricMetadata:
    """Audit metadata tracking physical coordinate and resolution transformations."""
    original_shape: Tuple[int, int, int]
    padded_shape: Tuple[int, int, int]
    target_shape: Tuple[int, int, int]
    original_spacing: Tuple[float, float, float]
    target_spacing: Tuple[float, float, float]
    scale_factor: float
    pad_z: Tuple[int, int]
    original_affine: Optional[List[List[float]]] = None
    target_affine: Optional[List[List[float]]] = None
    affine_tolerance: float = DEFAULT_AFFINE_TOLERANCE
    spacing_tolerance: float = DEFAULT_SPACING_TOLERANCE
    was_auto_aligned: bool = False
    nonzero_volume_cc: Optional[float] = None
    is_skull_stripped_plausible: Optional[bool] = None


def validate_brain_mask_plausibility(
    volume: np.ndarray,
    voxel_spacing: Tuple[float, float, float],
    min_volume_cc: float = 700.0,
    max_volume_cc: float = 2000.0,
    max_boundary_voxels: int = 500,
) -> Dict[str, Any]:
    """
    Evaluates whether a 3D volume displays characteristics consistent with an intracranial
    brain-extracted (skull-stripped) volume, vs a raw whole-head acquisition.

    Checks:
    1. Total non-zero volume: Normal adult human intracranial volume is ~1,100 to 1,600 cc.
       Raw whole-head acquisitions (scalp, neck, skull, face) typically exceed 3,500 - 6,000 cc.
    2. Peripheral field-of-view boundaries: Skull-stripped scans have background air (0.0)
       at outer bounding box boundaries, whereas unstripped scans contain substantial scalp and neck signal.
    """
    nz_mask = volume > 0
    voxel_vol_cc = float(np.prod(voxel_spacing)) / 1000.0
    nz_voxels = int(np.sum(nz_mask))
    nz_vol_cc = round(nz_voxels * voxel_vol_cc, 1)

    # Check outer 3 slices along each face
    outer_voxels = int(
        np.sum(nz_mask[:3, :, :]) + np.sum(nz_mask[-3:, :, :]) +
        np.sum(nz_mask[:, :3, :]) + np.sum(nz_mask[:, -3:, :]) +
        np.sum(nz_mask[:, :, :3]) + np.sum(nz_mask[:, :, -3:])
    )

    volume_valid = min_volume_cc <= nz_vol_cc <= max_volume_cc
    boundary_valid = outer_voxels <= max_boundary_voxels
    is_plausible = volume_valid and boundary_valid

    reasons = []
    if nz_vol_cc < min_volume_cc:
        reasons.append(f"Volume too small ({nz_vol_cc} cc < {min_volume_cc} cc; possible excessive brain tissue erosion)")
    elif nz_vol_cc > max_volume_cc:
        reasons.append(f"Volume too large ({nz_vol_cc} cc > {max_volume_cc} cc; typical of unstripped whole-head scans with skull/scalp/neck)")

    if not boundary_valid:
        reasons.append(f"Non-zero tissue signal detected on peripheral FOV boundary ({outer_voxels} voxels > {max_boundary_voxels} tolerance)")

    return {
        "nonzero_volume_cc": nz_vol_cc,
        "outer_boundary_voxels": outer_voxels,
        "is_skull_stripped_plausible": is_plausible,
        "failure_reasons": reasons,
    }


def canonicalize_modality_key(key: str) -> Optional[str]:
    """Maps variant sequence naming (e.g. t1n, t1w, t2w) to canonical 't1' or 't2'."""
    k = str(key).lower().strip()
    if k in ("t1", "t1n", "t1w", "t1-weighted", "t1_weighted"):
        return "t1"
    elif k in ("t2", "t2w", "t2-weighted", "t2_weighted"):
        return "t2"
    return None


def normalize_nonzero_zscore(volume: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Standardizes intensities of non-zero voxels to zero mean and unit variance.
    Empty-air background voxels (<= 0) remain strictly 0.0.
    NOTE: This does not skull-strip whole-head acquisitions.
    """
    norm_vol = volume.astype(np.float32).copy()
    mask = norm_vol > 0
    if np.any(mask):
        mean = float(norm_vol[mask].mean())
        std = float(norm_vol[mask].std())
        if std > eps:
            norm_vol[mask] = (norm_vol[mask] - mean) / std
        else:
            norm_vol[mask] = 0.0
    return norm_vol


def compute_target_affine(original_affine: np.ndarray) -> np.ndarray:
    """
    Computes physical affine matrix for the padded and isotropically resampled volume.
    Adjusts origin for the top Z-padding and scales voxel dimensions by 1.875x.
    """
    aff = original_affine.copy()
    direction_z = aff[:3, 2]
    aff[:3, 3] = aff[:3, 3] - (PAD_Z_TOP * direction_z)
    aff[:3, :3] = aff[:3, :3] * 1.875
    return aff


def align_and_resample_t2_to_t1(
    fixed_src: Union[str, Path, np.ndarray],
    moving_src: Union[str, Path, np.ndarray],
    fixed_affine: Optional[np.ndarray] = None,
    moving_affine: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Registers and resamples moving T2 acquisition into fixed T1 reference coordinate grid.
    Uses SimpleITK rigid registration with Mattes Mutual Information if paths are available,
    or falls back to scipy affine transform using header transformation matrices.
    """
    # 1. SimpleITK path-based registration if available
    if HAS_SITK and isinstance(fixed_src, (str, Path)) and isinstance(moving_src, (str, Path)):
        p_fix = Path(fixed_src)
        p_mov = Path(moving_src)
        if p_fix.exists() and p_mov.exists():
            fixed = sitk.ReadImage(str(p_fix), sitk.sitkFloat32)
            moving = sitk.ReadImage(str(p_mov), sitk.sitkFloat32)

            initial_transform = sitk.CenteredTransformInitializer(
                fixed,
                moving,
                sitk.Euler3DTransform(),
                sitk.CenteredTransformInitializerFilter.GEOMETRY,
            )

            registration_method = sitk.ImageRegistrationMethod()
            registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
            registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
            registration_method.SetMetricSamplingPercentage(0.02)
            registration_method.SetInterpolator(sitk.sitkLinear)
            registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=50)
            registration_method.SetOptimizerScalesFromPhysicalShift()
            registration_method.SetInitialTransform(initial_transform)

            final_transform = registration_method.Execute(fixed, moving)
            resampled = sitk.Resample(moving, fixed, final_transform, sitk.sitkLinear, 0.0, moving.GetPixelID())

            arr = sitk.GetArrayFromImage(resampled)
            # SimpleITK returns (Z, Y, X); transpose to (X, Y, Z) matching nibabel
            return np.transpose(arr, (2, 1, 0)).astype(np.float32)

    # 2. Scipy affine transform fallback using physical coordinate matrices
    if fixed_affine is not None and moving_affine is not None:
        if isinstance(moving_src, (str, Path)):
            nii_mov = nib.load(str(moving_src))
            arr_mov = np.asanyarray(nii_mov.dataobj, dtype=np.float32)
        else:
            arr_mov = moving_src.astype(np.float32)

        if isinstance(fixed_src, (str, Path)):
            nii_fix = nib.load(str(fixed_src))
            fix_shape = nii_fix.header.get_data_shape()[:3]
        else:
            fix_shape = fixed_src.shape

        t1_to_t2 = np.linalg.inv(moving_affine) @ fixed_affine
        mat = t1_to_t2[:3, :3]
        offset = t1_to_t2[:3, 3]

        resampled = ndimage.affine_transform(
            arr_mov,
            matrix=mat,
            offset=offset,
            output_shape=fix_shape,
            order=1,
            mode="constant",
            cval=0.0,
        )
        return resampled.astype(np.float32)

    raise ValueError("Cannot align modalities: insufficient spatial affine metadata provided.")


def preprocess_2channel_presence_tensor(
    modality_inputs: Dict[str, Union[np.ndarray, str, Path]],
    original_affine: Optional[np.ndarray] = None,
    affine_tolerance: float = DEFAULT_AFFINE_TOLERANCE,
    spacing_tolerance: float = DEFAULT_SPACING_TOLERANCE,
    auto_align: bool = False,
    enforce_skull_stripped: bool = False,
) -> Tuple[torch.Tensor, PresenceGeometricMetadata]:
    """
    Transforms 2 structural MRI modalities (T1 and T2) into a standardized 4D tensor (2, 128, 128, 96).

    Parameters:
        modality_inputs: Dict mapping 't1' (or 't1n') and 't2' (or 't2w') to arrays or NIfTI paths.
        original_affine: Optional (4, 4) reference affine matrix.
        affine_tolerance: Maximum allowable absolute difference between modality affine matrices.
        spacing_tolerance: Maximum allowable absolute difference between modality voxel spacings.
        auto_align: If True, uses medical image registration to resample T2 into T1's reference space
                    when grids or affines differ (e.g. for raw IXI). If False, strictly enforces alignment.

    Returns:
        tensor_4d: PyTorch FloatTensor of shape (2, 128, 128, 96).
        metadata: PresenceGeometricMetadata documenting geometric transformation.
    """
    if not isinstance(modality_inputs, dict):
        raise TypeError(f"modality_inputs must be a dict, got {type(modality_inputs).__name__}")

    # 1. Canonicalize input keys
    canon_inputs: Dict[str, Union[np.ndarray, str, Path]] = {}
    for raw_k, val in modality_inputs.items():
        ck = canonicalize_modality_key(raw_k)
        if ck:
            if ck in canon_inputs:
                raise ValueError(f"Duplicate modality detected: '{raw_k}' conflicts with '{ck}'.")
            canon_inputs[ck] = val

    missing = [m for m in CANONICAL_PRESENCE_MODALITIES if m not in canon_inputs]
    if missing:
        raise ValueError(
            f"Missing required modalities for tumor presence detection: {missing}. "
            f"Required: {CANONICAL_PRESENCE_MODALITIES} (T1 and T2)."
        )

    # 2. Ingest arrays and validate spatial coordinate alignment
    raw_arrays: Dict[str, np.ndarray] = {}
    ref_shape: Optional[Tuple[int, ...]] = None
    ref_aff: Optional[np.ndarray] = original_affine.copy() if original_affine is not None else None
    ref_spacing: Optional[Tuple[float, float, float]] = None
    ref_mod: Optional[str] = None
    was_auto_aligned: bool = False

    for mod in CANONICAL_PRESENCE_MODALITIES:
        src = canon_inputs[mod]
        mod_aff: Optional[np.ndarray] = None
        mod_zooms: Optional[Tuple[float, float, float]] = None

        if isinstance(src, (str, Path)):
            p = Path(src)
            if not p.exists():
                raise FileNotFoundError(f"Modality '{mod}' file not found: {p}")
            if p.stat().st_size == 0:
                raise ValueError(f"Modality '{mod}' file '{p.name}' is empty (0 bytes).")
            try:
                nii = nib.load(str(p))
                arr = np.asanyarray(nii.dataobj, dtype=np.float32)
                mod_aff = np.array(nii.affine, dtype=np.float64)
                mod_zooms = tuple(float(round(z, 4)) for z in nii.header.get_zooms()[:3])
            except Exception as exc:
                raise ValueError(f"Failed to load NIfTI modality '{mod}' from '{p}': {exc}")
        elif hasattr(src, "dataobj") and hasattr(src, "affine"):
            arr = np.asanyarray(src.dataobj, dtype=np.float32)
            mod_aff = np.array(src.affine, dtype=np.float64)
            if hasattr(src, "header") and hasattr(src.header, "get_zooms"):
                mod_zooms = tuple(float(round(z, 4)) for z in src.header.get_zooms()[:3])
            else:
                mod_zooms = ORIGINAL_SPACING
        elif isinstance(src, np.ndarray):
            arr = src.astype(np.float32)
            mod_aff = original_affine.copy() if original_affine is not None else None
            mod_zooms = ORIGINAL_SPACING
        else:
            raise TypeError(f"Unsupported source type for modality '{mod}': {type(src)}")

        if arr.ndim != 3:
            raise ValueError(f"Modality '{mod}' must be a 3D volume, got {arr.ndim}D with shape {arr.shape}.")

        if any(s <= 0 for s in arr.shape):
            raise ValueError(f"Modality '{mod}' has non-positive dimensions: {arr.shape}.")

        if not np.isfinite(arr).all():
            raise ValueError(f"Non-finite values (NaN or Inf) detected in modality '{mod}'.")

        # Handle reference setup or alignment check
        if ref_shape is None:
            ref_shape = arr.shape
            ref_mod = mod
            if mod_aff is not None and ref_aff is None:
                ref_aff = mod_aff.copy()
            if mod_zooms is not None and ref_spacing is None:
                ref_spacing = mod_zooms
        else:
            # Check if alignment is needed for moving modality (t2)
            needs_align = False
            if arr.shape != ref_shape:
                needs_align = True
            elif mod_aff is not None and ref_aff is not None:
                if not np.allclose(mod_aff, ref_aff, atol=affine_tolerance):
                    needs_align = True
            elif mod_zooms is not None and ref_spacing is not None:
                if not np.allclose(mod_zooms, ref_spacing, atol=spacing_tolerance):
                    needs_align = True

            if needs_align:
                if auto_align:
                    t1_src = canon_inputs["t1"]
                    arr = align_and_resample_t2_to_t1(
                        fixed_src=t1_src,
                        moving_src=src,
                        fixed_affine=ref_aff,
                        moving_affine=mod_aff,
                    )
                    mod_aff = ref_aff.copy() if ref_aff is not None else None
                    mod_zooms = ref_spacing
                    was_auto_aligned = True
                else:
                    if arr.shape != ref_shape:
                        raise ValueError(
                            f"Spatial shape mismatch for modality '{mod}': got {arr.shape}, expected {ref_shape} (matching {ref_mod})."
                        )
                    if mod_aff is not None and ref_aff is not None:
                        if not np.allclose(mod_aff, ref_aff, atol=affine_tolerance):
                            max_diff = float(np.max(np.abs(mod_aff - ref_aff)))
                            raise SpatialAlignmentError(
                                f"Spatial affine mismatch for modality '{mod}': max absolute difference {max_diff:.6f} "
                                f"exceeds tolerance {affine_tolerance}. T1 and T2 are not physically aligned."
                            )
                    if mod_zooms is not None and ref_spacing is not None:
                        if not np.allclose(mod_zooms, ref_spacing, atol=spacing_tolerance):
                            max_diff = float(np.max(np.abs(np.array(mod_zooms) - np.array(ref_spacing))))
                            raise SpatialAlignmentError(
                                f"Voxel spacing mismatch for modality '{mod}': max difference {max_diff:.6f} "
                                f"exceeds tolerance {spacing_tolerance} (got {mod_zooms}, expected {ref_spacing})."
                            )

        raw_arrays[mod] = arr

    # 3. Stack into (1, 2, D, H, W) PyTorch FloatTensor
    stacked_arr = np.stack([raw_arrays[m] for m in CANONICAL_PRESENCE_MODALITIES], axis=0)  # (2, D, H, W)
    t_input = torch.from_numpy(stacked_arr).unsqueeze(0)  # (1, 2, D, H, W)

    # 4. Symmetrically pad Z-axis from 155 to 180 if standard BraTS shape
    d, h, w = t_input.shape[2:]
    if (d, h, w) == ORIGINAL_SHAPE:
        t_padded = F.pad(t_input, (PAD_Z_TOP, PAD_Z_BOTTOM, 0, 0, 0, 0), mode="constant", value=0.0)
    else:
        t_padded = t_input

    # 5. Isotropic 1.875x Trilinear Resampling -> (1, 2, 128, 128, 96)
    t_resampled = F.interpolate(
        t_padded,
        size=TARGET_SHAPE,
        mode="trilinear",
        align_corners=False,
    ).squeeze(0)  # (2, 128, 128, 96)

    # 6. Per-channel non-zero Z-score normalization
    resampled_np = t_resampled.numpy()
    normalized_channels = []
    for c in range(2):
        norm_ch = normalize_nonzero_zscore(resampled_np[c])
        normalized_channels.append(norm_ch)

    final_tensor = torch.from_numpy(np.stack(normalized_channels, axis=0)).float()

    # 7. Document geometric and brain-mask transformation
    target_aff = compute_target_affine(ref_aff) if ref_aff is not None else None

    # Validate brain mask / skull-stripping plausibility
    plausibility = validate_brain_mask_plausibility(
        raw_arrays["t1"],
        ref_spacing if ref_spacing is not None else ORIGINAL_SPACING,
    )

    if enforce_skull_stripped and not plausibility["is_skull_stripped_plausible"]:
        raise ValueError(
            f"Input volume failed skull-stripping / brain mask validation: "
            f"{'; '.join(plausibility['failure_reasons'])}. "
            f"Ensure whole-head scans have undergone brain extraction before ingestion."
        )

    metadata = PresenceGeometricMetadata(
        original_shape=ref_shape,
        padded_shape=tuple(t_padded.shape[2:]),
        target_shape=TARGET_SHAPE,
        original_spacing=ref_spacing if ref_spacing is not None else ORIGINAL_SPACING,
        target_spacing=TARGET_SPACING,
        scale_factor=1.875,
        pad_z=(PAD_Z_TOP, PAD_Z_BOTTOM) if (d, h, w) == ORIGINAL_SHAPE else (0, 0),
        original_affine=ref_aff.tolist() if ref_aff is not None else None,
        target_affine=target_aff.tolist() if target_aff is not None else None,
        affine_tolerance=affine_tolerance,
        spacing_tolerance=spacing_tolerance,
        was_auto_aligned=was_auto_aligned,
        nonzero_volume_cc=plausibility["nonzero_volume_cc"],
        is_skull_stripped_plausible=plausibility["is_skull_stripped_plausible"],
    )

    return final_tensor, metadata
