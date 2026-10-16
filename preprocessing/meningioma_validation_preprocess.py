from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


# ============================================================
# PATHS
# ============================================================

VALIDATION_ROOT = Path(
    r"C:\3D_DATASET\BraTS2024-MEN-RT-ValidationData\BraTS-MEN-RT-Val-v1"
)

OUTPUT_ROOT = Path(
    r"C:\Users\nchan\Desktop\DL\data\meningioma_validation_preprocessed"
)


# ============================================================
# PREPROCESSING SETTINGS
# ============================================================

TARGET_SPACING = np.array(
    [1.0, 1.0, 1.0],
    dtype=np.float32
)

EPSILON = 1e-8


# ============================================================
# FILE DISCOVERY
# ============================================================

def find_t1c_file(case_dir):
    """
    Find the T1c MRI file for a validation case.
    """

    for file in case_dir.glob("*.nii.gz"):

        name = file.name.lower()

        if name.endswith("_t1c.nii.gz"):
            return file

    return None


# ============================================================
# RESAMPLING
# ============================================================

def resample_image(
    image,
    original_spacing,
    target_spacing
):
    """
    Resample MRI image using linear interpolation.

    Same method as the training preprocessing pipeline.
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

    return resampled.astype(
        np.float32
    )


# ============================================================
# INTENSITY NORMALIZATION
# ============================================================

def normalize_t1c(image):
    """
    Same normalization used during training:

    1. Select non-zero voxels
    2. Clip to 1st-99th percentile
    3. Calculate mean/std on foreground
    4. Z-score normalize foreground
    5. Keep background zero
    """

    image = image.astype(
        np.float32
    )

    foreground = image[
        image > 0
    ]

    if foreground.size == 0:

        return np.zeros_like(
            image
        )

    lower = np.percentile(
        foreground,
        1
    )

    upper = np.percentile(
        foreground,
        99
    )

    if upper <= lower:

        return np.zeros_like(
            image
        )

    image = np.clip(
        image,
        lower,
        upper
    )

    foreground = image[
        image > 0
    ]

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


def crop_to_foreground(image):
    """
    Crop MRI using the same foreground
    bounding-box logic used during training.
    """

    bbox = get_foreground_bbox(
        image
    )

    if bbox is None:

        return (
            image,
            np.zeros(
                3,
                dtype=np.int64
            ),
            np.array(
                image.shape,
                dtype=np.int64
            )
        )

    min_coords, max_coords = bbox

    slices = tuple(
        slice(
            int(min_coords[i]),
            int(max_coords[i]) + 1
        )
        for i in range(3)
    )

    cropped_image = image[
        slices
    ]

    return (
        cropped_image,
        min_coords.astype(
            np.int64
        ),
        max_coords.astype(
            np.int64
        )
    )


# ============================================================
# PROCESS ONE CASE
# ============================================================

def process_case(
    case_dir,
    output_dir
):
    """
    Preprocess one validation T1c case.

    No GTV is required because this is an
    inference-only validation set.
    """

    t1c_file = find_t1c_file(
        case_dir
    )

    if t1c_file is None:

        raise FileNotFoundError(
            f"T1c missing: {case_dir.name}"
        )

    # --------------------------------------------------------
    # Load NIfTI
    # --------------------------------------------------------

    t1c_img = nib.load(
        str(t1c_file)
    )

    t1c = t1c_img.get_fdata(
        dtype=np.float32
    )

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

    # --------------------------------------------------------
    # Normalization
    # --------------------------------------------------------

    t1c_normalized = normalize_t1c(
        t1c_resampled
    )

    # --------------------------------------------------------
    # Foreground crop
    # --------------------------------------------------------

    (
        t1c_cropped,
        crop_min,
        crop_max
    ) = crop_to_foreground(
        t1c_normalized
    )

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
    # Save T1c
    # --------------------------------------------------------

    np.save(
        case_output / "t1c.npy",
        t1c_cropped.astype(
            np.float32
        )
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    np.savez(
    case_output / "metadata.npz",

    original_shape=np.array(
        t1c.shape
    ),

    original_affine=t1c_img.affine,

    resampled_shape=np.array(
        t1c_resampled.shape
    ),

    cropped_shape=np.array(
        t1c_cropped.shape
    ),

    original_spacing=original_spacing,

    target_spacing=TARGET_SPACING,

    crop_min=crop_min,

    crop_max=crop_max
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("MENINGIOMA VALIDATION PREPROCESSING")
    print("=" * 100)

    print("\nValidation root:")
    print(
        VALIDATION_ROOT
    )

    print("\nOutput root:")
    print(
        OUTPUT_ROOT
    )

    print("\nTarget spacing:")
    print(
        TARGET_SPACING,
        "mm"
    )

    # --------------------------------------------------------
    # Validate root
    # --------------------------------------------------------

    if not VALIDATION_ROOT.exists():

        print("\nERROR:")
        print(
            "Validation directory does not exist."
        )

        return

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Find cases
    # --------------------------------------------------------

    case_dirs = sorted(
        [
            path
            for path in VALIDATION_ROOT.iterdir()
            if path.is_dir()
        ]
    )

    print(
        f"\nValidation cases found: "
        f"{len(case_dirs)}"
    )

    successful = 0
    failed = []

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

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
    print("VALIDATION PREPROCESSING SUMMARY")
    print("=" * 100)

    print(
        f"\nTotal cases:      "
        f"{len(case_dirs)}"
    )

    print(
        f"Successful:       "
        f"{successful}"
    )

    print(
        f"Failed:           "
        f"{len(failed)}"
    )

    if failed:

        print(
            "\nFailed cases:"
        )

        for case_name, error in failed[:50]:

            print(
                f"- {case_name}: {error}"
            )

    else:

        print(
            "\nAll validation cases "
            "processed successfully."
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