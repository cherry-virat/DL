from pathlib import Path
import sys
import json

import numpy as np
import torch
from scipy.ndimage import label


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\nchan\Desktop\DL"
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


# ============================================================
# IMPORT MODEL
# ============================================================

from models.meningioma_model import (
    create_meningioma_model
)


# ============================================================
# PATHS
# ============================================================

INPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "meningioma_validation_preprocessed"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "meningioma_validation_inference"
)

CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "meningioma_training"
    / "meningioma_best.pth"
)


# ============================================================
# INFERENCE SETTINGS
# ============================================================

PATCH_SIZE = (
    64,
    64,
    64
)

OVERLAP = 0.5

THRESHOLD = 0.125

MIN_COMPONENT_SIZE = 2500

TARGET_SPACING_MM = 1.0

VOXEL_VOLUME_MM3 = (
    TARGET_SPACING_MM ** 3
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# CONNECTED COMPONENT STRUCTURE
# ============================================================

STRUCTURE = np.ones(
    (3, 3, 3),
    dtype=np.uint8
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("\nLoading model...")

    model = create_meningioma_model(
        base_channels=16
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )

    if isinstance(
        checkpoint,
        dict
    ) and "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

    model.to(
        DEVICE
    )

    model.eval()

    print(
        f"Model loaded from:\n"
        f"{CHECKPOINT}"
    )

    print(
        f"Device: {DEVICE}"
    )

    return model


# ============================================================
# PAD VOLUME
# ============================================================

def pad_volume(
    volume,
    patch_size
):

    padding = []

    for size, patch in zip(
        volume.shape,
        patch_size
    ):

        if size < patch:

            total = patch - size

        else:

            total = 0

        before = total // 2
        after = total - before

        padding.append(
            (
                before,
                after
            )
        )

    if any(
        before > 0 or after > 0
        for before, after in padding
    ):

        padded = np.pad(
            volume,
            padding,
            mode="constant",
            constant_values=0
        )

    else:

        padded = volume

    return (
        padded,
        padding
    )


# ============================================================
# CALCULATE START POSITIONS
# ============================================================

def calculate_positions(
    size,
    patch,
    overlap
):

    if size <= patch:

        return [0]

    stride = int(
        patch * (
            1.0 - overlap
        )
    )

    if stride <= 0:

        raise ValueError(
            "Invalid overlap."
        )

    positions = list(
        range(
            0,
            size - patch + 1,
            stride
        )
    )

    last_position = (
        size - patch
    )

    if positions[-1] != last_position:

        positions.append(
            last_position
        )

    return positions


# ============================================================
# FULL-VOLUME INFERENCE
# ============================================================

@torch.no_grad()
def predict_volume(
    model,
    volume
):

    original_shape = (
        volume.shape
    )

    padded, padding = pad_volume(
        volume,
        PATCH_SIZE
    )

    padded_shape = (
        padded.shape
    )

    z_positions = calculate_positions(
        padded_shape[0],
        PATCH_SIZE[0],
        OVERLAP
    )

    y_positions = calculate_positions(
        padded_shape[1],
        PATCH_SIZE[1],
        OVERLAP
    )

    x_positions = calculate_positions(
        padded_shape[2],
        PATCH_SIZE[2],
        OVERLAP
    )

    probability_sum = np.zeros(
        padded_shape,
        dtype=np.float32
    )

    prediction_count = np.zeros(
        padded_shape,
        dtype=np.float32
    )

    total_patches = (
        len(z_positions)
        * len(y_positions)
        * len(x_positions)
    )

    processed_patches = 0

    for z in z_positions:

        for y in y_positions:

            for x in x_positions:

                patch = padded[
                    z:z + PATCH_SIZE[0],
                    y:y + PATCH_SIZE[1],
                    x:x + PATCH_SIZE[2]
                ]

                patch_tensor = torch.from_numpy(
                    patch.astype(
                        np.float32
                    )
                )

                patch_tensor = (
                    patch_tensor
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(DEVICE)
                )

                logits = model(
                    patch_tensor
                )

                probabilities = torch.softmax(
                    logits,
                    dim=1
                )

                tumor_probability = (
                    probabilities[
                        0,
                        1
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                )

                probability_sum[
                    z:z + PATCH_SIZE[0],
                    y:y + PATCH_SIZE[1],
                    x:x + PATCH_SIZE[2]
                ] += tumor_probability

                prediction_count[
                    z:z + PATCH_SIZE[0],
                    y:y + PATCH_SIZE[1],
                    x:x + PATCH_SIZE[2]
                ] += 1.0

                processed_patches += 1

    probability_map = (
        probability_sum
        / np.maximum(
            prediction_count,
            1e-8
        )
    )

    # --------------------------------------------------------
    # Remove padding
    # --------------------------------------------------------

    z_before, z_after = padding[0]
    y_before, y_after = padding[1]
    x_before, x_after = padding[2]

    z_end = (
        None
        if z_after == 0
        else -z_after
    )

    y_end = (
        None
        if y_after == 0
        else -y_after
    )

    x_end = (
        None
        if x_after == 0
        else -x_after
    )

    probability_map = probability_map[
        z_before:z_end,
        y_before:y_end,
        x_before:x_end
    ]

    if probability_map.shape != original_shape:

        raise RuntimeError(
            "Probability map shape mismatch: "
            f"{probability_map.shape} "
            f"!= {original_shape}"
        )

    return (
        probability_map,
        total_patches
    )


# ============================================================
# CONNECTED COMPONENT FILTER
# ============================================================

def remove_small_components(
    binary_mask,
    minimum_size
):

    labeled, number_of_components = label(
        binary_mask,
        structure=STRUCTURE
    )

    if number_of_components == 0:

        return (
            np.zeros_like(
                binary_mask,
                dtype=np.uint8
            ),
            0,
            0
        )

    component_sizes = np.bincount(
        labeled.ravel()
    )

    keep = (
        component_sizes
        >= minimum_size
    )

    keep[0] = False

    filtered = keep[
        labeled
    ]

    filtered = filtered.astype(
        np.uint8
    )

    kept_components = int(
        np.sum(
            keep
        )
    )

    return (
        filtered,
        int(number_of_components),
        kept_components
    )


# ============================================================
# CALCULATE CONFIDENCE
# ============================================================

def calculate_confidence(
    probability_map,
    prediction
):

    tumor_pixels = (
        prediction > 0
    )

    if not np.any(
        tumor_pixels
    ):

        return {
            "mean_tumor_probability": 0.0,
            "max_tumor_probability": float(
                np.max(
                    probability_map
                )
            ),
            "tumor_voxel_count": 0
        }

    tumor_probabilities = (
        probability_map[
            tumor_pixels
        ]
    )

    return {
        "mean_tumor_probability": float(
            np.mean(
                tumor_probabilities
            )
        ),
        "max_tumor_probability": float(
            np.max(
                tumor_probabilities
            )
        ),
        "tumor_voxel_count": int(
            np.sum(
                tumor_pixels
            )
        )
    }


# ============================================================
# PROCESS ONE CASE
# ============================================================

def process_case(
    model,
    case_dir,
    output_root
):

    case_name = (
        case_dir.name
    )

    image_path = (
        case_dir
        / "t1c.npy"
    )

    metadata_path = (
        case_dir
        / "metadata.npz"
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Missing t1c.npy: "
            f"{case_name}"
        )

    if not metadata_path.exists():

        raise FileNotFoundError(
            f"Missing metadata.npz: "
            f"{case_name}"
        )

    # --------------------------------------------------------
    # Load input
    # --------------------------------------------------------

    volume = np.load(
        image_path
    ).astype(
        np.float32
    )

    metadata = np.load(
        metadata_path
    )

    original_shape = tuple(
        metadata[
            "original_shape"
        ].tolist()
    )

    resampled_shape = tuple(
        metadata[
            "resampled_shape"
        ].tolist()
    )

    cropped_shape = tuple(
        metadata[
            "cropped_shape"
        ].tolist()
    )

    crop_min = tuple(
        metadata[
            "crop_min"
        ].tolist()
    )

    crop_max = tuple(
        metadata[
            "crop_max"
        ].tolist()
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    probability_map, total_patches = (
        predict_volume(
            model,
            volume
        )
    )

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    raw_prediction = (
        probability_map
        >= THRESHOLD
    ).astype(
        np.uint8
    )

    # --------------------------------------------------------
    # Connected components
    # --------------------------------------------------------

    (
        prediction,
        total_components,
        kept_components
    ) = remove_small_components(
        raw_prediction,
        MIN_COMPONENT_SIZE
    )

    # --------------------------------------------------------
    # Tumor volume
    # --------------------------------------------------------

    tumor_voxels = int(
        np.sum(
            prediction > 0
        )
    )

    tumor_volume_mm3 = (
        tumor_voxels
        * VOXEL_VOLUME_MM3
    )

    tumor_volume_cm3 = (
        tumor_volume_mm3
        / 1000.0
    )

    tumor_present = (
        tumor_voxels > 0
    )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = calculate_confidence(
        probability_map,
        prediction
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    case_output = (
        output_root
        / case_name
    )

    case_output.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save probability
    # --------------------------------------------------------

    np.save(
        case_output
        / "tumor_probability.npy",
        probability_map.astype(
            np.float32
        )
    )

    # --------------------------------------------------------
    # Save raw threshold prediction
    # --------------------------------------------------------

    np.save(
        case_output
        / "raw_prediction.npy",
        raw_prediction.astype(
            np.uint8
        )
    )

    # --------------------------------------------------------
    # Save final prediction
    # --------------------------------------------------------

    np.save(
        case_output
        / "prediction.npy",
        prediction.astype(
            np.uint8
        )
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    case_report = {

        "case": case_name,

        "tumor_type": "meningioma",

        "model": "MeningiomaUNet3D",

        "checkpoint": str(
            CHECKPOINT
        ),

        "input_shape": list(
            volume.shape
        ),

        "original_shape": list(
            original_shape
        ),

        "resampled_shape": list(
            resampled_shape
        ),

        "cropped_shape": list(
            cropped_shape
        ),

        "crop_min": list(
            crop_min
        ),

        "crop_max": list(
            crop_max
        ),

        "patch_size": list(
            PATCH_SIZE
        ),

        "overlap": OVERLAP,

        "total_patches": int(
            total_patches
        ),

        "threshold": THRESHOLD,

        "minimum_component_size": (
            MIN_COMPONENT_SIZE
        ),

        "total_components_before_filter": (
            total_components
        ),

        "components_after_filter": (
            kept_components
        ),

        "tumor_present": bool(
            tumor_present
        ),

        "tumor_voxel_count": tumor_voxels,

        "tumor_volume_mm3": float(
            tumor_volume_mm3
        ),

        "tumor_volume_cm3": float(
            tumor_volume_cm3
        ),

        "mean_tumor_probability": (
            confidence[
                "mean_tumor_probability"
            ]
        ),

        "max_tumor_probability": (
            confidence[
                "max_tumor_probability"
            ]
        )
    }

    with open(
        case_output / "metrics.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            case_report,
            file,
            indent=4
        )

    return case_report


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print(
        "MENINGIOMA 70-CASE VALIDATION "
        "FULL-VOLUME INFERENCE"
    )
    print("=" * 100)

    print("\nInput:")
    print(
        INPUT_ROOT
    )

    print("\nCheckpoint:")
    print(
        CHECKPOINT
    )

    print("\nOutput:")
    print(
        OUTPUT_ROOT
    )

    print("\nInference settings:")
    print(
        f"Patch size              : {PATCH_SIZE}"
    )
    print(
        f"Overlap                 : {OVERLAP}"
    )
    print(
        f"Threshold               : {THRESHOLD}"
    )
    print(
        f"Minimum component size  : "
        f"{MIN_COMPONENT_SIZE}"
    )

    print("\nDevice:")
    print(
        DEVICE
    )

    # --------------------------------------------------------
    # Validate paths
    # --------------------------------------------------------

    if not INPUT_ROOT.exists():

        print(
            "\nERROR: Input directory does not exist."
        )

        return

    if not CHECKPOINT.exists():

        print(
            "\nERROR: Model checkpoint does not exist."
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
            for path in INPUT_ROOT.iterdir()
            if path.is_dir()
        ]
    )

    print(
        f"\nValidation cases found: "
        f"{len(case_dirs)}"
    )

    if len(case_dirs) == 0:

        print(
            "\nERROR: No validation cases found."
        )

        return

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Process cases
    # --------------------------------------------------------

    results = []
    failures = []

    for index, case_dir in enumerate(
        case_dirs,
        start=1
    ):

        print(
            "\n" + "-" * 100
        )

        print(
            f"Case {index}/{len(case_dirs)}: "
            f"{case_dir.name}"
        )

        try:

            result = process_case(
                model,
                case_dir,
                OUTPUT_ROOT
            )

            results.append(
                result
            )

            print(
                f"Tumor present       : "
                f"{result['tumor_present']}"
            )

            print(
                f"Tumor voxels        : "
                f"{result['tumor_voxel_count']}"
            )

            print(
                f"Tumor volume        : "
                f"{result['tumor_volume_cm3']:.3f} cm³"
            )

            print(
                f"Mean tumor prob.    : "
                f"{result['mean_tumor_probability']:.4f}"
            )

            print(
                f"Max tumor prob.     : "
                f"{result['max_tumor_probability']:.4f}"
            )

            print(
                f"Components kept     : "
                f"{result['components_after_filter']}"
            )

        except Exception as e:

            failures.append(
                {
                    "case": case_dir.name,
                    "error": str(e)
                }
            )

            print(
                f"ERROR: {e}"
            )

    # ========================================================
    # AGGREGATE SUMMARY
    # ========================================================

    tumor_positive_cases = sum(
        1
        for result in results
        if result["tumor_present"]
    )

    tumor_negative_cases = (
        len(results)
        - tumor_positive_cases
    )

    volumes = [
        result["tumor_volume_cm3"]
        for result in results
        if result["tumor_present"]
    ]

    mean_volume = (
        float(
            np.mean(volumes)
        )
        if volumes
        else 0.0
    )

    median_volume = (
        float(
            np.median(volumes)
        )
        if volumes
        else 0.0
    )

    mean_confidence = (
        float(
            np.mean(
                [
                    result[
                        "mean_tumor_probability"
                    ]
                    for result in results
                    if result["tumor_present"]
                ]
            )
        )
        if volumes
        else 0.0
    )

    mean_components = (
        float(
            np.mean(
                [
                    result[
                        "components_after_filter"
                    ]
                    for result in results
                ]
            )
        )
        if results
        else 0.0
    )

    summary = {

        "dataset": (
            "BraTS2024-MEN-RT "
            "Validation"
        ),

        "total_cases": len(
            case_dirs
        ),

        "successful_cases": len(
            results
        ),

        "failed_cases": len(
            failures
        ),

        "tumor_positive_predictions": (
            tumor_positive_cases
        ),

        "tumor_negative_predictions": (
            tumor_negative_cases
        ),

        "threshold": THRESHOLD,

        "minimum_component_size": (
            MIN_COMPONENT_SIZE
        ),

        "patch_size": list(
            PATCH_SIZE
        ),

        "overlap": OVERLAP,

        "mean_predicted_tumor_volume_cm3": (
            mean_volume
        ),

        "median_predicted_tumor_volume_cm3": (
            median_volume
        ),

        "mean_tumor_probability_positive_cases": (
            mean_confidence
        ),

        "mean_components_after_filter": (
            mean_components
        ),

        "voxel_spacing_mm": [
            1.0,
            1.0,
            1.0
        ],

        "note": (
            "Validation dataset does not contain "
            "ground-truth GTV masks locally. "
            "Dice, IoU, precision and recall "
            "are therefore not calculated."
        ),

        "failures": failures
    }

    # --------------------------------------------------------
    # Save detailed results
    # --------------------------------------------------------

    detailed_report = (
        OUTPUT_ROOT
        / "meningioma_validation_report.json"
    )

    with open(
        detailed_report,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "summary": summary,
                "cases": results
            },
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary_file = (
        OUTPUT_ROOT
        / "meningioma_validation_summary.json"
    )

    with open(
        summary_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n" + "=" * 100)
    print(
        "MENINGIOMA VALIDATION INFERENCE COMPLETE"
    )
    print("=" * 100)

    print(
        f"\nTotal cases:              "
        f"{len(case_dirs)}"
    )

    print(
        f"Successful:               "
        f"{len(results)}"
    )

    print(
        f"Failed:                   "
        f"{len(failures)}"
    )

    print(
        f"Tumor-positive predicted: "
        f"{tumor_positive_cases}"
    )

    print(
        f"Tumor-negative predicted: "
        f"{tumor_negative_cases}"
    )

    print(
        f"\nMean tumor volume:         "
        f"{mean_volume:.3f} cm³"
    )

    print(
        f"Median tumor volume:       "
        f"{median_volume:.3f} cm³"
    )

    print(
        f"Mean components/case:      "
        f"{mean_components:.2f}"
    )

    print(
        "\nDice / IoU / Precision / Recall:"
    )

    print(
        "NOT AVAILABLE — no GTV ground truth "
        "in validation dataset."
    )

    print(
        "\nDetailed report:"
    )

    print(
        detailed_report
    )

    print(
        "\nSummary:"
    )

    print(
        summary_file
    )

    print(
        "\n" + "=" * 100
    )


if __name__ == "__main__":
    main()