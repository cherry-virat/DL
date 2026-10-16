from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


PROJECT_ROOT = Path(
    r"C:\Users\nchan\Desktop\DL"
)

PREDICTION_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "meningioma_validation_inference"
)

PREPROCESSED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "meningioma_validation_preprocessed"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "meningioma_validation_original_space"
)


def restore_to_resampled_space(
    prediction,
    resampled_shape,
    crop_min
):

    resampled_shape = tuple(
        int(x)
        for x in resampled_shape
    )

    crop_min = np.asarray(
        crop_min,
        dtype=np.int64
    )

    cropped_shape = prediction.shape

    restored = np.zeros(
        resampled_shape,
        dtype=np.uint8
    )

    start = crop_min

    end = (
        crop_min
        + np.asarray(
            cropped_shape,
            dtype=np.int64
        )
    )

    for axis in range(3):

        if start[axis] < 0:

            raise ValueError(
                "Crop start is negative."
            )

        if end[axis] > resampled_shape[axis]:

            raise ValueError(
                "Crop exceeds resampled volume."
            )

    slices = tuple(
        slice(
            int(start[axis]),
            int(end[axis])
        )
        for axis in range(3)
    )

    restored[slices] = (
        prediction > 0
    ).astype(
        np.uint8
    )

    return restored


def restore_to_original_space(
    prediction_resampled,
    original_shape
):

    original_shape = tuple(
        int(x)
        for x in original_shape
    )

    zoom_factors = (
        np.asarray(
            original_shape,
            dtype=np.float32
        )
        /
        np.asarray(
            prediction_resampled.shape,
            dtype=np.float32
        )
    )

    restored = zoom(
        prediction_resampled.astype(
            np.uint8
        ),
        zoom=zoom_factors,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False
    )

    if restored.shape != original_shape:

        corrected = np.zeros(
            original_shape,
            dtype=np.uint8
        )

        copy_shape = tuple(
            min(
                restored.shape[i],
                original_shape[i]
            )
            for i in range(3)
        )

        slices = tuple(
            slice(
                0,
                copy_shape[i]
            )
            for i in range(3)
        )

        corrected[slices] = (
            restored[slices]
        )

        restored = corrected

    return (
        restored > 0
    ).astype(
        np.uint8
    )


def process_case(
    case_dir
):

    case_name = case_dir.name

    preprocessed_case = (
        PREPROCESSED_ROOT
        / case_name
    )

    prediction_file = (
        case_dir
        / "prediction.npy"
    )

    probability_file = (
        case_dir
        / "tumor_probability.npy"
    )

    metadata_file = (
        preprocessed_case
        / "metadata.npz"
    )

    if not prediction_file.exists():

        raise FileNotFoundError(
            f"Prediction missing: "
            f"{prediction_file}"
        )

    if not metadata_file.exists():

        raise FileNotFoundError(
            f"Metadata missing: "
            f"{metadata_file}"
        )

    prediction = np.load(
        prediction_file
    )

    metadata = np.load(
        metadata_file
    )

    original_shape = tuple(
        int(x)
        for x in metadata[
            "original_shape"
        ]
    )

    original_affine = metadata[
        "original_affine"
    ]

    resampled_shape = tuple(
        int(x)
        for x in metadata[
            "resampled_shape"
        ]
    )

    cropped_shape = tuple(
        int(x)
        for x in metadata[
            "cropped_shape"
        ]
    )

    crop_min = metadata[
        "crop_min"
    ]

    crop_max = metadata[
        "crop_max"
    ]

    metadata.close()

    prediction = (
        prediction > 0
    ).astype(
        np.uint8
    )

    if prediction.shape != cropped_shape:

        raise ValueError(
            f"Prediction shape mismatch "
            f"for {case_name}: "
            f"prediction={prediction.shape}, "
            f"metadata={cropped_shape}"
        )

    prediction_resampled = (
        restore_to_resampled_space(
            prediction,
            resampled_shape,
            crop_min
        )
    )

    prediction_original = (
        restore_to_original_space(
            prediction_resampled,
            original_shape
        )
    )

    if prediction_original.shape != original_shape:

        raise ValueError(
            f"Original shape mismatch "
            f"for {case_name}: "
            f"{prediction_original.shape} "
            f"!= {original_shape}"
        )

    output_case = (
        OUTPUT_ROOT
        / case_name
    )

    output_case.mkdir(
        parents=True,
        exist_ok=True
    )

    prediction_output = (
        output_case
        / f"{case_name}_meningioma_prediction.nii.gz"
    )

    nifti = nib.Nifti1Image(
        prediction_original.astype(
            np.uint8
        ),
        original_affine
    )

    nifti.set_data_dtype(
        np.uint8
    )

    nib.save(
        nifti,
        str(prediction_output)
    )

    if probability_file.exists():

        probability = np.load(
            probability_file
        )

        probability_resampled = (
            restore_to_resampled_space(
                (
                    probability
                    >= 0.125
                ).astype(
                    np.uint8
                ),
                resampled_shape,
                crop_min
            )
        )

        probability_original = (
            restore_to_original_space(
                probability_resampled,
                original_shape
            )
        )

        threshold_output = (
            output_case
            / f"{case_name}_threshold_prediction.nii.gz"
        )

        threshold_nifti = nib.Nifti1Image(
            probability_original.astype(
                np.uint8
            ),
            original_affine
        )

        threshold_nifti.set_data_dtype(
            np.uint8
        )

        nib.save(
            threshold_nifti,
            str(threshold_output)
        )

    return {
        "case": case_name,
        "original_shape": original_shape,
        "resampled_shape": resampled_shape,
        "cropped_shape": cropped_shape,
        "prediction_voxels": int(
            prediction_original.sum()
        ),
        "prediction_volume_ml": float(
            prediction_original.sum()
            * np.prod(
                nib.affines.voxel_sizes(
                    original_affine
                )
            )
            / 1000.0
        ),
        "prediction_file": str(
            prediction_output
        )
    }


def main():

    print("=" * 100)
    print("MENINGIOMA ORIGINAL-SPACE VALIDATION RESTORATION")
    print("=" * 100)

    print(
        "\nPrediction root:"
    )
    print(
        PREDICTION_ROOT
    )

    print(
        "\nPreprocessed root:"
    )
    print(
        PREPROCESSED_ROOT
    )

    print(
        "\nOutput root:"
    )
    print(
        OUTPUT_ROOT
    )

    if not PREDICTION_ROOT.exists():

        print(
            "\nERROR: Prediction directory "
            "does not exist."
        )
        return

    if not PREPROCESSED_ROOT.exists():

        print(
            "\nERROR: Preprocessed directory "
            "does not exist."
        )
        return

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    case_dirs = sorted(
        [
            path
            for path in PREDICTION_ROOT.iterdir()
            if path.is_dir()
        ]
    )

    print(
        f"\nPrediction cases found: "
        f"{len(case_dirs)}"
    )

    successful = 0
    failed = []

    volumes = []

    for index, case_dir in enumerate(
        case_dirs,
        start=1
    ):

        try:

            result = process_case(
                case_dir
            )

            successful += 1

            volumes.append(
                result[
                    "prediction_volume_ml"
                ]
            )

            print(
                f"\rRestored "
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

    print("=" * 100)
    print("RESTORATION SUMMARY")
    print("=" * 100)

    print(
        f"\nTotal cases:       "
        f"{len(case_dirs)}"
    )

    print(
        f"Successful:        "
        f"{successful}"
    )

    print(
        f"Failed:            "
        f"{len(failed)}"
    )

    if volumes:

        print(
            f"\nMean predicted volume: "
            f"{np.mean(volumes):.3f} mL"
        )

        print(
            f"Median predicted volume: "
            f"{np.median(volumes):.3f} mL"
        )

        print(
            f"Minimum predicted volume: "
            f"{np.min(volumes):.3f} mL"
        )

        print(
            f"Maximum predicted volume: "
            f"{np.max(volumes):.3f} mL"
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
            "\nAll validation predictions "
            "restored successfully."
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