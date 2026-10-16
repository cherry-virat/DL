from pathlib import Path

import nibabel as nib
import numpy as np


PROJECT_ROOT = Path(
    r"C:\Users\nchan\Desktop\DL"
)

RAW_VALIDATION_ROOT = Path(
    r"C:\3D_DATASET\BraTS2024-MEN-RT-ValidationData\BraTS-MEN-RT-Val-v1"
)

RESTORED_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "meningioma_validation_original_space"
)


def find_t1c(case_dir):

    for file in case_dir.glob(
        "*_t1c.nii.gz"
    ):

        return file

    return None


def affine_close(
    a,
    b,
    tolerance=1e-4
):

    return np.allclose(
        a,
        b,
        atol=tolerance,
        rtol=tolerance
    )


def main():

    print("=" * 100)
    print("VERIFY MENINGIOMA ORIGINAL-SPACE NIFTI GEOMETRY")
    print("=" * 100)

    cases = sorted(
        [
            path
            for path in RAW_VALIDATION_ROOT.iterdir()
            if path.is_dir()
        ]
    )

    print(
        f"\nRaw validation cases: "
        f"{len(cases)}"
    )

    successful = 0
    failed = []

    for case_dir in cases:

        case_name = case_dir.name

        t1c_file = find_t1c(
            case_dir
        )

        prediction_file = (
            RESTORED_ROOT
            / case_name
            / f"{case_name}_meningioma_prediction.nii.gz"
        )

        if t1c_file is None:

            failed.append(
                (
                    case_name,
                    "Original T1c missing"
                )
            )

            continue

        if not prediction_file.exists():

            failed.append(
                (
                    case_name,
                    "Restored prediction missing"
                )
            )

            continue

        try:

            original = nib.load(
                str(t1c_file)
            )

            prediction = nib.load(
                str(prediction_file)
            )

            original_shape = (
                original.shape[:3]
            )

            prediction_shape = (
                prediction.shape[:3]
            )

            if original_shape != prediction_shape:

                failed.append(
                    (
                        case_name,
                        f"Shape mismatch: "
                        f"original={original_shape}, "
                        f"prediction={prediction_shape}"
                    )
                )

                continue

            if not affine_close(
                original.affine,
                prediction.affine
            ):

                failed.append(
                    (
                        case_name,
                        "Affine mismatch"
                    )
                )

                continue

            original_spacing = (
                nib.affines.voxel_sizes(
                    original.affine
                )
            )

            prediction_spacing = (
                nib.affines.voxel_sizes(
                    prediction.affine
                )
            )

            if not np.allclose(
                original_spacing,
                prediction_spacing,
                atol=1e-4,
                rtol=1e-4
            ):

                failed.append(
                    (
                        case_name,
                        "Voxel spacing mismatch"
                    )
                )

                continue

            data = prediction.get_fdata()

            unique_values = np.unique(
                data
            )

            if not np.all(
                np.isin(
                    unique_values,
                    [0, 1]
                )
            ):

                failed.append(
                    (
                        case_name,
                        f"Prediction is not binary: "
                        f"{unique_values}"
                    )
                )

                continue

            successful += 1

            if successful <= 3:

                print(
                    f"\nCase: {case_name}"
                )

                print(
                    f"Shape: "
                    f"{original_shape}"
                )

                print(
                    f"Spacing: "
                    f"{original_spacing}"
                )

                print(
                    f"Prediction voxels: "
                    f"{int(data.sum())}"
                )

        except Exception as e:

            failed.append(
                (
                    case_name,
                    str(e)
                )
            )

    print("\n")
    print("=" * 100)
    print("ORIGINAL-SPACE VERIFICATION SUMMARY")
    print("=" * 100)

    print(
        f"\nTotal cases:       "
        f"{len(cases)}"
    )

    print(
        f"Verified:          "
        f"{successful}"
    )

    print(
        f"Failed:            "
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
            "\nALL 70 CASES PASSED:"
        )

        print(
            "Shape              ✅"
        )

        print(
            "Affine             ✅"
        )

        print(
            "Voxel spacing      ✅"
        )

        print(
            "Binary mask        ✅"
        )

    print(
        "\n" + "=" * 100
    )


if __name__ == "__main__":
    main()