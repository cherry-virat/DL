from pathlib import Path

import numpy as np
import SimpleITK as sitk


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_SPACING = (1.0, 1.0, 1.0)

MRI_MODALITIES = {
    "T1": "t1n",
    "T1c": "t1c",
    "T2": "t2w",
    "FLAIR": "t2f",
}

SEGMENTATION_NAME = "seg"


# ============================================================
# LOAD NIFTI
# ============================================================

def load_nifti(path):
    """
    Load a NIfTI image using SimpleITK.
    """
    return sitk.ReadImage(str(path))


# ============================================================
# RESAMPLE IMAGE
# ============================================================

def resample_image(
    image,
    target_spacing=TARGET_SPACING,
    is_label=False
):
    """
    Resample an MRI image or segmentation mask.

    MRI:
        Linear interpolation

    Segmentation:
        Nearest-neighbor interpolation
    """

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(
            round(
                original_size[i]
                * original_spacing[i]
                / target_spacing[i]
            )
        )
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()

    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)

    resampler.SetOutputDirection(
        image.GetDirection()
    )

    resampler.SetOutputOrigin(
        image.GetOrigin()
    )

    resampler.SetTransform(
        sitk.Transform()
    )

    if is_label:
        resampler.SetInterpolator(
            sitk.sitkNearestNeighbor
        )
    else:
        resampler.SetInterpolator(
            sitk.sitkLinear
        )

    return resampler.Execute(image)


# ============================================================
# SIMPLEITK → NUMPY
# ============================================================

def sitk_to_numpy(image):
    return sitk.GetArrayFromImage(image)


# ============================================================
# MRI NORMALIZATION
# ============================================================

def normalize_mri(volume):
    """
    Z-score normalization using non-zero voxels.

    Background remains zero.
    """

    volume = volume.astype(np.float32)

    mask = volume != 0

    if not np.any(mask):
        return volume

    mean = volume[mask].mean()
    std = volume[mask].std()

    if std < 1e-8:
        return volume

    normalized = np.zeros_like(
        volume,
        dtype=np.float32
    )

    normalized[mask] = (
        volume[mask] - mean
    ) / std

    return normalized


# ============================================================
# FIND TUMOR BOUNDING BOX
# ============================================================

def get_tumor_bbox(mask):
    """
    Return tumor bounding box.

    Coordinates are in:
        Z, Y, X
    """

    tumor = mask > 0

    if not np.any(tumor):
        return None

    coords = np.where(tumor)

    min_coords = np.array(
        [
            coords[0].min(),
            coords[1].min(),
            coords[2].min()
        ]
    )

    max_coords = np.array(
        [
            coords[0].max(),
            coords[1].max(),
            coords[2].max()
        ]
    )

    return min_coords, max_coords


# ============================================================
# CROP AROUND TUMOR
# ============================================================

def crop_around_tumor(
    modalities,
    mask,
    margin=16
):
    """
    Apply the SAME spatial crop to every MRI modality
    and to the segmentation mask.
    """

    bbox = get_tumor_bbox(mask)

    if bbox is None:
        raise ValueError(
            "No tumor found in segmentation mask."
        )

    min_coords, max_coords = bbox

    shape = np.array(mask.shape)

    start = np.maximum(
        min_coords - margin,
        0
    )

    end = np.minimum(
        max_coords + margin + 1,
        shape
    )

    slices = tuple(
        slice(
            int(start[i]),
            int(end[i])
        )
        for i in range(3)
    )

    cropped_modalities = {
        name: volume[slices]
        for name, volume in modalities.items()
    }

    cropped_mask = mask[slices]

    return (
        cropped_modalities,
        cropped_mask
    )


# ============================================================
# PROCESS ONE GLIOMA CASE
# ============================================================

def process_case(
    case_dir,
    crop_margin=16
):
    """
    Complete preprocessing for one Glioma case.

    Returns:

        image:
            shape = (4, D, H, W)

        mask:
            shape = (D, H, W)
    """

    case_dir = Path(case_dir)

    case_name = case_dir.name

    # --------------------------------------------------------
    # Load MRI modalities
    # --------------------------------------------------------

    images = {}

    for modality, suffix in MRI_MODALITIES.items():

        path = (
            case_dir
            / f"{case_name}-{suffix}.nii.gz"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing {modality}: {path}"
            )

        images[modality] = load_nifti(
            path
        )

    # --------------------------------------------------------
    # Load segmentation
    # --------------------------------------------------------

    seg_path = (
        case_dir
        / f"{case_name}-{SEGMENTATION_NAME}.nii.gz"
    )

    if not seg_path.exists():
        raise FileNotFoundError(
            f"Missing segmentation: {seg_path}"
        )

    segmentation = load_nifti(
        seg_path
    )

    # --------------------------------------------------------
    # Resample MRI
    # --------------------------------------------------------

    resampled_images = {}

    for modality, image in images.items():

        resampled_images[modality] = (
            resample_image(
                image,
                TARGET_SPACING,
                is_label=False
            )
        )

    # --------------------------------------------------------
    # Resample segmentation
    # --------------------------------------------------------

    resampled_segmentation = (
        resample_image(
            segmentation,
            TARGET_SPACING,
            is_label=True
        )
    )

    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    arrays = {
        name: sitk_to_numpy(image)
        for name, image
        in resampled_images.items()
    }

    mask = sitk_to_numpy(
        resampled_segmentation
    )

    # --------------------------------------------------------
    # Normalize MRI
    # --------------------------------------------------------

    normalized = {
        name: normalize_mri(volume)
        for name, volume
        in arrays.items()
    }

    # --------------------------------------------------------
    # Convert segmentation to integer
    # --------------------------------------------------------

    mask = mask.astype(
        np.int64
    )

    # --------------------------------------------------------
    # Tumor-aware crop
    # --------------------------------------------------------

    normalized, mask = crop_around_tumor(
        normalized,
        mask,
        margin=crop_margin
    )

    # --------------------------------------------------------
    # Create 4-channel tensor
    # --------------------------------------------------------

    image = np.stack(
        [
            normalized["T1"],
            normalized["T1c"],
            normalized["T2"],
            normalized["FLAIR"]
        ],
        axis=0
    )

    image = image.astype(
        np.float32
    )

    return image, mask



if __name__ == "__main__":

    dataset_root = Path(
        r"C:\3D_DATASET\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    )

    nested = (
        dataset_root
        / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    )

    if nested.exists():
        dataset_root = nested

    cases = sorted(
        [
            p
            for p in dataset_root.iterdir()
            if p.is_dir()
        ]
    )

    print("=" * 70)
    print("GLIOMA PREPROCESSING TEST")
    print("=" * 70)

    print("Dataset:", dataset_root)
    print("Cases:", len(cases))

    image, mask = process_case(
        cases[0]
    )

    print(
        "\nMRI shape:",
        image.shape
    )

    print(
        "MRI dtype:",
        image.dtype
    )

    print(
        "Mask shape:",
        mask.shape
    )

    print(
        "Mask dtype:",
        mask.dtype
    )

    print(
        "MRI min:",
        image.min()
    )

    print(
        "MRI max:",
        image.max()
    )

    print(
        "Tumor voxels:",
        np.count_nonzero(mask > 0)
    )