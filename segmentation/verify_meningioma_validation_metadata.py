from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(
    r"C:\Users\nchan\Desktop\DL"
)

VALIDATION_PREPROCESSED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "meningioma_validation_preprocessed"
)


def main():

    print("=" * 100)
    print("VERIFY MENINGIOMA VALIDATION METADATA")
    print("=" * 100)

    print(
        "\nValidation preprocessing root:"
    )
    print(
        VALIDATION_PREPROCESSED_ROOT
    )

    if not VALIDATION_PREPROCESSED_ROOT.exists():
        print(
            "\nERROR: Validation preprocessing directory does not exist."
        )
        return

    case_dirs = sorted(
        [
            path
            for path in VALIDATION_PREPROCESSED_ROOT.iterdir()
            if path.is_dir()
        ]
    )

    print(
        f"\nCases found: {len(case_dirs)}"
    )

    if not case_dirs:
        print(
            "\nERROR: No validation cases found."
        )
        return

    success = 0
    failed = []

    for case_dir in case_dirs:

        metadata_file = (
            case_dir
            / "metadata.npz"
        )

        if not metadata_file.exists():
            failed.append(
                (
                    case_dir.name,
                    "metadata.npz missing"
                )
            )
            continue

        try:

            metadata = np.load(
                metadata_file
            )

            required_fields = [
                "original_shape",
                "original_affine",
                "resampled_shape",
                "cropped_shape",
                "original_spacing",
                "target_spacing",
                "crop_min",
                "crop_max"
            ]

            missing = [
                field
                for field in required_fields
                if field not in metadata
            ]

            if missing:
                failed.append(
                    (
                        case_dir.name,
                        f"Missing fields: {missing}"
                    )
                )
                continue

            original_shape = metadata[
                "original_shape"
            ]

            original_affine = metadata[
                "original_affine"
            ]

            resampled_shape = metadata[
                "resampled_shape"
            ]

            cropped_shape = metadata[
                "cropped_shape"
            ]

            crop_min = metadata[
                "crop_min"
            ]

            crop_max = metadata[
                "crop_max"
            ]

            if tuple(original_affine.shape) != (
                4,
                4
            ):
                failed.append(
                    (
                        case_dir.name,
                        "Invalid affine shape"
                    )
                )
                continue

            if len(original_shape) != 3:
                failed.append(
                    (
                        case_dir.name,
                        "Invalid original shape"
                    )
                )
                continue

            if len(resampled_shape) != 3:
                failed.append(
                    (
                        case_dir.name,
                        "Invalid resampled shape"
                    )
                )
                continue

            if len(cropped_shape) != 3:
                failed.append(
                    (
                        case_dir.name,
                        "Invalid cropped shape"
                    )
                )
                continue

            if len(crop_min) != 3:
                failed.append(
                    (
                        case_dir.name,
                        "Invalid crop_min"
                    )
                )
                continue

            if len(crop_max) != 3:
                failed.append(
                    (
                        case_dir.name,
                        "Invalid crop_max"
                    )
                )
                continue

            success += 1

            if success == 1:

                print(
                    "\nExample case:"
                )

                print(
                    f"Case: {case_dir.name}"
                )

                print(
                    f"Original shape: "
                    f"{tuple(original_shape)}"
                )

                print(
                    f"Resampled shape: "
                    f"{tuple(resampled_shape)}"
                )

                print(
                    f"Cropped shape: "
                    f"{tuple(cropped_shape)}"
                )

                print(
                    "\nOriginal affine:"
                )

                print(
                    original_affine
                )

                print(
                    f"\nCrop min: "
                    f"{crop_min}"
                )

                print(
                    f"Crop max: "
                    f"{crop_max}"
                )

            metadata.close()

        except Exception as e:

            failed.append(
                (
                    case_dir.name,
                    str(e)
                )
            )

    print("\n")
    print("=" * 100)
    print("METADATA VERIFICATION SUMMARY")
    print("=" * 100)

    print(
        f"\nTotal cases:       {len(case_dirs)}"
    )

    print(
        f"Valid metadata:    {success}"
    )

    print(
        f"Failed:            {len(failed)}"
    )

    if failed:

        print(
            "\nFailed cases:"
        )

        for case_name, error in failed[:20]:

            print(
                f"- {case_name}: {error}"
            )

    else:

        print(
            "\nAll validation metadata "
            "contains original affine and "
            "required restoration information."
        )

    print(
        "\n" + "=" * 100
    )


if __name__ == "__main__":
    main()