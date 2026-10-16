from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


# ============================================================
# PATHS
# ============================================================

TRAIN_ROOT = Path(
    r"C:\3D_DATASET\BraTS2024-MEN-RT-TrainingData\BraTS-MEN-RT-Train-v2"
)

OUTPUT_ROOT = Path(
    r"C:\Users\nchan\Desktop\DL\data\meningioma_preprocessed"
)


# ============================================================
# PREPROCESSING SETTINGS
# ============================================================

TARGET_SPACING = np.array(
    [1.0, 1.0, 1.0],
    dtype=np.float32
)

# Small epsilon for numerical safety
EPSILON = 1e-8


# ============================================================
# FILE DISCOVERY
# ============================================================

def find_case_files(case_dir):
    """
    Find T1c MRI and GTV segmentation files.
    """

    t1c_file = None
    gtv_file = None

    for file in case_dir.glob("*.nii.gz"):

        name = file.name.lower()

        if name.endswith("_t1c.nii.gz"):
            t1c_file = file

        elif name.endswith("_gtv.nii.gz"):
            gtv_file = file

    return t1c_file, gtv_file


# ============================================================
# RESAMPLING
# ============================================================

def resample_image(image, original_spacing, target_spacing):
    """
    Resample MRI image using linear interpolation.

    scipy zoom works using:

        new_shape / old_shape

    """

    zoom_factors = (
        original_spacing / target_spacing
    )

    resampled = zoom(
        image,
        zoom=zoom_factors,
        order=1,
        mode="constant",
        cval=0,
        prefilter=True
    )

    return resampled.astype(np.float32)


def resample_mask(mask, original_spacing, target_spacing):
    """
    Resample segmentation mask using nearest-neighbor
    interpolation so that labels remain binary.
    """

    zoom_factors = (
        original_spacing / target_spacing
    )

    resampled = zoom(
        mask,
        zoom=zoom_factors,
        order=0,
        mode="constant",
        cval=0
    )

    # Force binary mask
    resampled = (
        resampled > 0.5
    ).astype(np.uint8)

    return resampled


# ============================================================
# INTENSITY NORMALIZATION
# ============================================================

def normalize_t1c(image):
    """
    Robust z-score normalization using non-zero voxels.

    Background remains zero.
    """

    image = image.astype(np.float32)

    foreground = image[image > 0]

    if foreground.size == 0:
        return np.zeros_like(image)

    # Robust percentile clipping
    lower = np.percentile(
        foreground,
        1
    )

    upper = np.percentile(
        foreground,
        99
    )

    if upper <= lower:
        return np.zeros_like(image)

    image = np.clip(
        image,
        lower,
        upper
    )

    foreground = image[image > 0]

    mean = foreground.mean()
    std = foreground.std()

    if std < EPSILON:
        std = 1.0

    normalized = np.zeros_like(
        image,
        dtype=np.float32
    )

    nonzero = image > 0

    normalized[nonzero] = (
        image[nonzero] - mean
    ) / std

    return normalized


# ============================================================
# FOREGROUND CROPPING
# ============================================================

def get_foreground_bbox(image):
    """
    Calculate bounding box around non-zero MRI region.
    """

    foreground = image > 0

    coordinates = np.argwhere(
        foreground
    )

    if coordinates.size == 0:
        return None

    min_coords = coordinates.min(
        axis=0
    )

    max_coords = coordinates.max(
        axis=0
    )

    return (
        min_coords,
        max_coords
    )


def crop_to_foreground(image, mask):
    """
    Crop MRI and mask using MRI foreground bounding box.
    """

    bbox = get_foreground_bbox(
        image
    )

    if bbox is None:
        return image, mask

    min_coords, max_coords = bbox

    slices = tuple(
        slice(
            int(min_coords[i]),
            int(max_coords[i]) + 1
        )
        for i in range(3)
    )

    cropped_image = image[slices]
    cropped_mask = mask[slices]

    return (
        cropped_image,
        cropped_mask
    )


# ============================================================
# PROCESS ONE CASE
# ============================================================

def process_case(case_dir, output_dir):
    """
    Complete preprocessing pipeline for one case.
    """

    t1c_file, gtv_file = find_case_files(
        case_dir
    )

    if t1c_file is None:
        raise FileNotFoundError(
            f"T1c missing: {case_dir.name}"
        )

    if gtv_file is None:
        raise FileNotFoundError(
            f"GTV missing: {case_dir.name}"
        )

    # --------------------------------------------------------
    # Load NIfTI
    # --------------------------------------------------------

    t1c_img = nib.load(
        str(t1c_file)
    )

    gtv_img = nib.load(
        str(gtv_file)
    )

    t1c = t1c_img.get_fdata(
        dtype=np.float32
    )

    gtv = gtv_img.get_fdata()

    # --------------------------------------------------------
    # Original spacing
    # --------------------------------------------------------

    original_spacing = np.array(
        t1c_img.header.get_zooms()[:3],
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Resampling
    # --------------------------------------------------------

    t1c_resampled = resample_image(
        t1c,
        original_spacing,
        TARGET_SPACING
    )

    gtv_resampled = resample_mask(
        gtv,
        original_spacing,
        TARGET_SPACING
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    t1c_normalized = normalize_t1c(
        t1c_resampled
    )

    # --------------------------------------------------------
    # Crop foreground
    # --------------------------------------------------------

    t1c_cropped, gtv_cropped = (
        crop_to_foreground(
            t1c_normalized,
            gtv_resampled
        )
    )

    # --------------------------------------------------------
    # Verify mask
    # --------------------------------------------------------

    gtv_cropped = (
        gtv_cropped > 0
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    case_output = (
        output_dir
        / case_dir.name
    )

    case_output.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save arrays
    # --------------------------------------------------------

    image_output = (
        case_output
        / "t1c.npy"
    )

    mask_output = (
        case_output
        / "gtv.npy"
    )

    metadata_output = (
        case_output
        / "metadata.npz"
    )

    np.save(
        image_output,
        t1c_cropped.astype(
            np.float32
        )
    )

    np.save(
        mask_output,
        gtv_cropped.astype(
            np.uint8
        )
    )

    np.savez(
        metadata_output,
        original_shape=np.array(
            t1c.shape
        ),
        resampled_shape=np.array(
            t1c_resampled.shape
        ),
        cropped_shape=np.array(
            t1c_cropped.shape
        ),
        original_spacing=original_spacing,
        target_spacing=TARGET_SPACING
    )

    return {
        "case": case_dir.name,
        "original_shape": tuple(
            t1c.shape
        ),
        "resampled_shape": tuple(
            t1c_resampled.shape
        ),
        "cropped_shape": tuple(
            t1c_cropped.shape
        ),
        "original_spacing": tuple(
            original_spacing
        ),
        "tumor_voxels": int(
            np.sum(
                gtv_cropped > 0
            )
        )
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("MENINGIOMA PREPROCESSING")
    print("=" * 100)

    print("\nTraining root:")
    print(TRAIN_ROOT)

    print("\nOutput root:")
    print(OUTPUT_ROOT)

    print("\nTarget spacing:")
    print(
        TARGET_SPACING,
        "mm"
    )

    if not TRAIN_ROOT.exists():

        print("\nERROR:")
        print(
            "Training directory does not exist."
        )

        return

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    case_dirs = sorted(
        [
            path
            for path in TRAIN_ROOT.iterdir()
            if path.is_dir()
        ]
    )

    print(
        f"\nTraining cases found: "
        f"{len(case_dirs)}"
    )

    successful = 0
    failed = []

    for index, case_dir in enumerate(
        case_dirs,
        start=1
    ):

        try:

            result = process_case(
                case_dir,
                OUTPUT_ROOT
            )

            successful += 1

            print(
                f"\rProcessed "
                f"{index}/{len(case_dirs)}: "
                f"{case_dir.name}",
                end=""
            )

        except Exception as e:

            failed.append(
                (
                    case_dir.name,
                    str(e)
                )
            )

    print("\n")

    # ========================================================
    # SUMMARY
    # ========================================================

    print("=" * 100)
    print("PREPROCESSING SUMMARY")
    print("=" * 100)

    print(
        f"\nTotal cases:      {len(case_dirs)}"
    )

    print(
        f"Successful:       {successful}"
    )

    print(
        f"Failed:           {len(failed)}"
    )

    if failed:

        print("\nFailed cases:")

        for case_name, error in failed[:50]:

            print(
                f"- {case_name}: {error}"
            )

    else:

        print(
            "\nAll cases processed successfully."
        )

    print(
        "\nOutput directory:"
    )

    print(
        OUTPUT_ROOT.resolve()
    )

    print(
        "\n" + "=" * 100
    )


if __name__ == "__main__":
    main()