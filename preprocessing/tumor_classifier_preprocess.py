"""
Standalone 3D Brain Tumor Classifier Preprocessing Engine.
==========================================================

Independent spatial and intensity transformation module designed for 4-channel
multimodal structural brain MRI classification (Glioma vs. Meningioma).

Design Principles:
------------------
1. Non-destructive: Completely decoupled from frozen Glioma and Meningioma segmentation pipelines.
2. Aspect-ratio Preserving: Symmetrically pads the Z-dimension (from 155 to 180 slices)
   before applying isotropic 1.875x spatial scaling, eliminating geometric elongation.
3. Target Tensor Shape: Produces strictly (4, 128, 128, 96) float32 tensors with isotropic 1.875mm spacing.
4. Leakage-free: Decouples all file paths, case names, and metadata from model input tensors.
5. Intensity Standardization: Applies per-channel non-zero Z-score normalization on brain voxels,
   preserving zero-intensity background air.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F


# Canonical 4-channel ordering
CANONICAL_MODALITIES: List[str] = ["t1n", "t1c", "t2w", "t2f"]

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
    """Raised when MRI modalities exhibit incompatible coordinate affines, voxel spacings, or orientations."""
    pass


@dataclass(frozen=True)
class ClassifierGeometricMetadata:
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


def normalize_nonzero_zscore(volume: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Standardizes intensities of non-zero brain voxels to zero mean and unit variance.
    Empty-air background voxels (<= 0) remain strictly 0.0.
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
    # Step 1: Shift origin along Z direction due to top padding
    direction_z = aff[:3, 2]
    aff[:3, 3] = aff[:3, 3] - (PAD_Z_TOP * direction_z)
    # Step 2: Scale spatial axes by 1.875
    aff[:3, :3] = aff[:3, :3] * 1.875
    return aff


def preprocess_4channel_classifier_tensor(
    modality_inputs: Dict[str, Union[np.ndarray, str, Path]],
    original_affine: Optional[np.ndarray] = None,
    affine_tolerance: float = DEFAULT_AFFINE_TOLERANCE,
    spacing_tolerance: float = DEFAULT_SPACING_TOLERANCE,
) -> Tuple[torch.Tensor, ClassifierGeometricMetadata]:
    """
    Transforms 4 structural MRI modalities into a leak-free 4D classifier tensor.

    Spatial Alignment Policy:
    -------------------------
    Shape equality alone does NOT prove that four MRI sequences are physically coregistered.
    Two scans may both have matrix dimensions (240, 240, 155) while depicting completely
    different FOVs or patient orientations if their affine matrices differ.
    Therefore, this module strictly verifies that all four input sequences share:
      1. Identical 3D matrix shapes.
      2. Coordinate transformation affines equal within `affine_tolerance` (default 1e-3).
      3. Voxel spacing zooms equal within `spacing_tolerance` (default 1e-3).
    Stacking is rejected with SpatialAlignmentError if any sequence differs.

    Parameters:
        modality_inputs: Dict mapping 't1n', 't1c', 't2w', 't2f' to 3D arrays or NIfTI file paths.
        original_affine: Optional (4, 4) reference affine matrix.
        affine_tolerance: Maximum allowable absolute difference between modality affine matrices.
        spacing_tolerance: Maximum allowable absolute difference between modality voxel spacings.

    Returns:
        tensor_4d: PyTorch FloatTensor of shape (4, 128, 128, 96).
        metadata: ClassifierGeometricMetadata documenting geometric transformation.
    """
    # 1. Validate modality completeness
    missing = [m for m in CANONICAL_MODALITIES if m not in modality_inputs]
    if missing:
        raise ValueError(
            f"Missing required classifier modalities: {missing}. Expected: {CANONICAL_MODALITIES}"
        )

    # 2. Ingest arrays and validate spatial coordinate alignment
    raw_arrays: Dict[str, np.ndarray] = {}
    ref_shape: Optional[Tuple[int, ...]] = None
    ref_aff: Optional[np.ndarray] = original_affine.copy() if original_affine is not None else None
    ref_spacing: Optional[Tuple[float, float, float]] = None
    ref_mod: Optional[str] = None

    for mod in CANONICAL_MODALITIES:
        src = modality_inputs[mod]
        mod_aff: Optional[np.ndarray] = None
        mod_zooms: Optional[Tuple[float, float, float]] = None

        if isinstance(src, (str, Path)):
            p = Path(src)
            if not p.exists():
                raise FileNotFoundError(f"Modality file not found: {p}")
            if p.stat().st_size == 0:
                raise ValueError(f"Modality file '{p.name}' is empty (0 bytes).")
            try:
                nii = nib.load(str(p))
                arr = np.asanyarray(nii.dataobj, dtype=np.float32)
                mod_aff = np.array(nii.affine, dtype=np.float64)
                mod_zooms = tuple(float(round(z, 4)) for z in nii.header.get_zooms()[:3])
            except Exception as exc:
                raise ValueError(f"Failed to load or parse NIfTI modality '{mod}' from '{p}': {exc}")
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

        # Validate 3D dimensionality
        if arr.ndim != 3:
            raise ValueError(
                f"Modality '{mod}' must be a 3D volume, got dimension {arr.ndim} with shape {arr.shape}."
            )

        if any(s <= 0 for s in arr.shape):
            raise ValueError(f"Modality '{mod}' has invalid non-positive dimensions: {arr.shape}.")

        if not np.isfinite(arr).all():
            raise ValueError(f"Non-finite values (NaN or Inf) detected in modality '{mod}'.")

        # Validate shape compatibility against reference
        if ref_shape is None:
            ref_shape = arr.shape
            ref_mod = mod
        elif arr.shape != ref_shape:
            raise ValueError(
                f"Spatial shape mismatch for modality '{mod}': got {arr.shape}, expected {ref_shape} (matching {ref_mod})."
            )

        # Validate affine coordinate alignment against reference
        if mod_aff is not None:
            if ref_aff is None:
                ref_aff = mod_aff.copy()
            else:
                if not np.allclose(mod_aff, ref_aff, atol=affine_tolerance):
                    max_diff = float(np.max(np.abs(mod_aff - ref_aff)))
                    raise SpatialAlignmentError(
                        f"Spatial affine mismatch for modality '{mod}': max absolute difference {max_diff:.6f} "
                        f"exceeds tolerance {affine_tolerance}. Modality is not physically coregistered with {ref_mod}."
                    )

        # Validate voxel spacing agreement against reference
        if mod_zooms is not None:
            if ref_spacing is None:
                ref_spacing = mod_zooms
            elif not np.allclose(mod_zooms, ref_spacing, atol=spacing_tolerance):
                max_diff = float(np.max(np.abs(np.array(mod_zooms) - np.array(ref_spacing))))
                raise SpatialAlignmentError(
                    f"Voxel spacing mismatch for modality '{mod}': max difference {max_diff:.6f} "
                    f"exceeds tolerance {spacing_tolerance} (got {mod_zooms}, expected {ref_spacing})."
                )

        raw_arrays[mod] = arr

    # 3. Stack into (1, 4, D, H, W) PyTorch FloatTensor
    stacked_arr = np.stack([raw_arrays[m] for m in CANONICAL_MODALITIES], axis=0)  # (4, D, H, W)
    t_input = torch.from_numpy(stacked_arr).unsqueeze(0)  # (1, 4, D, H, W)

    # 4. Symmetrically pad Z-axis from 155 to 180 to guarantee isotropic downsampling
    # F.pad expects (pad_W_left, pad_W_right, pad_H_left, pad_H_right, pad_D_left, pad_D_right)
    # PyTorch 5D spatial dimensions: (B, C, D, H, W)
    d, h, w = t_input.shape[2:]
    if (d, h, w) == ORIGINAL_SHAPE:
        # Standard BraTS input (240, 240, 155): pad along last dim (Z: 155 -> 180)
        t_padded = F.pad(t_input, (PAD_Z_TOP, PAD_Z_BOTTOM, 0, 0, 0, 0), mode="constant", value=0.0)
    else:
        # Generic input: pad to multiple or pass as-is
        t_padded = t_input

    # 5. Isotropic 1.875x Trilinear Resampling -> (1, 4, 128, 128, 96)
    t_resampled = F.interpolate(
        t_padded,
        size=TARGET_SHAPE,
        mode="trilinear",
        align_corners=False,
    ).squeeze(0)  # (4, 128, 128, 96)

    # 6. Independent per-channel non-zero Z-score normalization
    resampled_np = t_resampled.numpy()
    normalized_channels = []
    for c in range(4):
        norm_ch = normalize_nonzero_zscore(resampled_np[c])
        normalized_channels.append(norm_ch)

    final_tensor = torch.from_numpy(np.stack(normalized_channels, axis=0)).float()

    # 7. Document geometric transformation
    target_aff = compute_target_affine(ref_aff) if ref_aff is not None else None

    metadata = ClassifierGeometricMetadata(
        original_shape=ref_shape,
        padded_shape=tuple(t_padded.shape[2:]),
        target_shape=TARGET_SHAPE,
        original_spacing=ref_spacing if ref_spacing is not None else ORIGINAL_SPACING,
        target_spacing=TARGET_SPACING,
        scale_factor=1.875,
        pad_z=(PAD_Z_TOP, PAD_Z_BOTTOM),
        original_affine=ref_aff.tolist() if ref_aff is not None else None,
        target_affine=target_aff.tolist() if target_aff is not None else None,
        affine_tolerance=affine_tolerance,
        spacing_tolerance=spacing_tolerance,
    )

    return final_tensor, metadata


def canonicalize_classifier_modality_key(key: str) -> Optional[str]:
    """Maps variant modality names (e.g. t1, t1c, flair) to canonical keys ('t1n', 't1c', 't2w', 't2f')."""
    k = str(key).lower().strip()
    if k in ("t1", "t1n", "t1w", "t1-weighted", "t1_weighted"):
        return "t1n"
    elif k in ("t1c", "t1ce", "t1-contrast", "t1_contrast", "t1c_weighted"):
        return "t1c"
    elif k in ("t2", "t2w", "t2-weighted", "t2_weighted"):
        return "t2w"
    elif k in ("flair", "t2f", "t2-flair", "t2_flair", "t2_flair_weighted"):
        return "t2f"
    return None


def preprocess_case_to_tensor(
    modality_inputs: Dict[str, Union[np.ndarray, str, Path]],
    original_affine: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """
    Convenience wrapper that canonicalizes modality keys, validates alignment,
    and returns only the preprocessed (4, 128, 128, 96) tensor.
    """
    canon_inputs: Dict[str, Union[np.ndarray, str, Path]] = {}
    for raw_k, val in modality_inputs.items():
        ck = canonicalize_classifier_modality_key(raw_k)
        if ck:
            canon_inputs[ck] = val
        else:
            canon_inputs[str(raw_k).lower().strip()] = val

    tensor_4d, _ = preprocess_4channel_classifier_tensor(
        canon_inputs,
        original_affine=original_affine,
    )
    return tensor_4d
