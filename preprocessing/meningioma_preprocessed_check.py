from pathlib import Path

import numpy as np


PREPROCESSED_ROOT = Path(
    r"C:\Users\nchan\Desktop\DL\data\meningioma_preprocessed"
)


TEST_CASES = [
    "BraTS-MEN-RT-0002-1",
    "BraTS-MEN-RT-0314-1",
    "BraTS-MEN-RT-0625-1",
]


def check_case(case_name):

    case_dir = PREPROCESSED_ROOT / case_name

    image_file = case_dir / "t1c.npy"
    mask_file = case_dir / "gtv.npy"
    metadata_file = case_dir / "metadata.npz"

    print("\n" + "=" * 90)
    print(f"CASE: {case_name}")
    print("=" * 90)

    if not image_file.exists():
        print("ERROR: t1c.npy missing")
        return

    if not mask_file.exists():
        print("ERROR: gtv.npy missing")
        return

    if not metadata_file.exists():
        print("ERROR: metadata.npz missing")
        return

    image = np.load(image_file)
    mask = np.load(mask_file)
    metadata = np.load(metadata_file)

    print("\nImage:")
    print("  Shape:", image.shape)
    print("  Dtype:", image.dtype)
    print("  Min:", float(image.min()))
    print("  Max:", float(image.max()))
    print("  Mean:", float(image.mean()))
    print("  Std:", float(image.std()))

    print("\nMask:")
    print("  Shape:", mask.shape)
    print("  Dtype:", mask.dtype)
    print("  Unique:", np.unique(mask))
    print("  Tumor voxels:", int(np.sum(mask > 0)))

    print("\nMetadata:")

    print(
        "  Original shape:",
        tuple(metadata["original_shape"])
    )

    print(
        "  Resampled shape:",
        tuple(metadata["resampled_shape"])
    )

    print(
        "  Cropped shape:",
        tuple(metadata["cropped_shape"])
    )

    print(
        "  Original spacing:",
        tuple(metadata["original_spacing"])
    )

    print(
        "  Target spacing:",
        tuple(metadata["target_spacing"])
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print("\nValidation:")

    if image.shape != mask.shape:
        print("  ❌ Image/mask shape mismatch")
    else:
        print("  ✅ Image/mask shapes match")

    if not np.isfinite(image).all():
        print("  ❌ Image contains NaN/Inf")
    else:
        print("  ✅ Image contains no NaN/Inf")

    unique_values = np.unique(mask)

    if np.all(np.isin(unique_values, [0, 1])):
        print("  ✅ Mask is binary")
    else:
        print("  ❌ Mask is not binary")

    if np.sum(mask > 0) > 0:
        print("  ✅ Tumor is present")
    else:
        print("  ❌ Tumor mask is empty")

    # Check normalization approximately
    foreground = image[image != 0]

    if foreground.size > 0:

        foreground_mean = float(
            foreground.mean()
        )

        foreground_std = float(
            foreground.std()
        )

        print(
            "\nForeground normalized statistics:"
        )

        print(
            "  Mean:",
            foreground_mean
        )

        print(
            "  Std:",
            foreground_std
        )


def main():

    print("=" * 90)
    print("MENINGIOMA PREPROCESSED DATASET CHECK")
    print("=" * 90)

    if not PREPROCESSED_ROOT.exists():

        print(
            "\nERROR: Preprocessed directory does not exist."
        )

        print(PREPROCESSED_ROOT)

        return

    case_dirs = sorted(
        [
            p
            for p in PREPROCESSED_ROOT.iterdir()
            if p.is_dir()
        ]
    )

    print(
        f"\nPreprocessed case directories: "
        f"{len(case_dirs)}"
    )

    # --------------------------------------------------------
    # Check total number of cases
    # --------------------------------------------------------

    if len(case_dirs) == 500:
        print("✅ All 500 cases are present")
    else:
        print(
            f"⚠️ Expected 500 cases, found {len(case_dirs)}"
        )

    # --------------------------------------------------------
    # Check representative cases
    # --------------------------------------------------------

    for case_name in TEST_CASES:

        check_case(case_name)

    # --------------------------------------------------------
    # Check all cases for missing files
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("ALL-CASE FILE CHECK")
    print("=" * 90)

    complete = 0
    incomplete = []

    for case_dir in case_dirs:

        image_file = case_dir / "t1c.npy"
        mask_file = case_dir / "gtv.npy"
        metadata_file = case_dir / "metadata.npz"

        if (
            image_file.exists()
            and mask_file.exists()
            and metadata_file.exists()
        ):

            complete += 1

        else:

            incomplete.append(
                case_dir.name
            )

    print(
        f"\nComplete cases: "
        f"{complete}"
    )

    print(
        f"Incomplete cases: "
        f"{len(incomplete)}"
    )

    if incomplete:

        print("\nIncomplete cases:")

        for case in incomplete[:50]:

            print(
                " -",
                case
            )

    else:

        print(
            "\n✅ Every case contains:"
        )

        print(
            "   t1c.npy"
        )

        print(
            "   gtv.npy"
        )

        print(
            "   metadata.npz"
        )

    print("\n" + "=" * 90)
    print("PREPROCESSED DATASET CHECK COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()