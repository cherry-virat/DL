from pathlib import Path
import json
import sys

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

import numpy as np
from scipy import ndimage


# ============================================================
# PATHS
# ============================================================

INFERENCE_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "meningioma_full_volume_inference"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "meningioma_joint_postprocessing_analysis"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# JOINT EXPERIMENT
# ============================================================
THRESHOLDS = [
    0.10,
    0.125,
    0.15,
    0.175,
    0.20,
]

MIN_COMPONENT_SIZES = [
    1000,
    1250,
    1500,
    1750,
    2000,
    2250,
    2500,
    3000,
]


# ============================================================
# 3D CONNECTIVITY
# ============================================================

STRUCTURE = np.ones(
    (3, 3, 3),
    dtype=np.uint8
)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    prediction,
    ground_truth
):

    prediction = prediction.astype(
        bool
    )

    ground_truth = ground_truth.astype(
        bool
    )

    tp = np.logical_and(
        prediction,
        ground_truth
    ).sum()

    fp = np.logical_and(
        prediction,
        np.logical_not(ground_truth)
    ).sum()

    fn = np.logical_and(
        np.logical_not(prediction),
        ground_truth
    ).sum()

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 1.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 1.0
    )

    dice = (
        (2.0 * tp)
        /
        (2.0 * tp + fp + fn)
        if (2.0 * tp + fp + fn) > 0
        else 1.0
    )

    iou = (
        tp / (tp + fp + fn)
        if (tp + fp + fn) > 0
        else 1.0
    )

    return {
        "precision": float(
            precision
        ),

        "recall": float(
            recall
        ),

        "dice": float(
            dice
        ),

        "iou": float(
            iou
        ),

        "tp": int(tp),

        "fp": int(fp),

        "fn": int(fn),

        "predicted_tumor_voxels": int(
            prediction.sum()
        ),

        "ground_truth_tumor_voxels": int(
            ground_truth.sum()
        ),
    }


# ============================================================
# CONNECTED COMPONENT FILTER
# ============================================================

def remove_small_components(
    binary_mask,
    minimum_size
):

    binary_mask = binary_mask.astype(
        bool
    )

    if minimum_size <= 0:
        return binary_mask.copy()

    labeled_mask, number_of_components = (
        ndimage.label(
            binary_mask,
            structure=STRUCTURE
        )
    )

    if number_of_components == 0:

        return np.zeros_like(
            binary_mask,
            dtype=bool
        )

    component_sizes = np.bincount(
        labeled_mask.ravel()
    )

    keep = (
        component_sizes
        >= minimum_size
    )

    keep[0] = False

    filtered_mask = keep[
        labeled_mask
    ]

    return filtered_mask.astype(
        bool
    )


# ============================================================
# COMPONENT STATISTICS
# ============================================================

def component_statistics(
    binary_mask
):

    binary_mask = binary_mask.astype(
        bool
    )

    labeled_mask, number_of_components = (
        ndimage.label(
            binary_mask,
            structure=STRUCTURE
        )
    )

    if number_of_components == 0:

        return {
            "component_count": 0,
            "largest_component_voxels": 0,
        }

    sizes = np.bincount(
        labeled_mask.ravel()
    )[1:]

    return {
        "component_count": int(
            number_of_components
        ),

        "largest_component_voxels": int(
            sizes.max()
        ),
    }


# ============================================================
# FIND CASES
# ============================================================

case_dirs = sorted(
    [
        path
        for path in INFERENCE_ROOT.iterdir()
        if path.is_dir()
    ]
)

case_dirs = [
    path
    for path in case_dirs
    if (
        (
            path
            / "tumor_probability.npy"
        ).exists()
        and
        (
            path
            / "ground_truth.npy"
        ).exists()
    )
]

if not case_dirs:

    raise RuntimeError(
        "No valid inference cases found."
    )


# ============================================================
# HEADER
# ============================================================

print("=" * 110)

print(
    "MENINGIOMA JOINT THRESHOLD + "
    "CONNECTED-COMPONENT ANALYSIS"
)

print("=" * 110)

print()

print(
    f"Cases: {len(case_dirs)}"
)

print()

print(
    "Thresholds:"
)

print(
    THRESHOLDS
)

print()

print(
    "Minimum component sizes:"
)

print(
    MIN_COMPONENT_SIZES
)

print()

print(
    f"Total configurations: "
    f"{len(THRESHOLDS) * len(MIN_COMPONENT_SIZES)}"
)

print()


# ============================================================
# LOAD DATA
# ============================================================

cases = []

for index, case_dir in enumerate(
    case_dirs,
    start=1
):

    case_id = case_dir.name

    probability = np.load(
        case_dir
        / "tumor_probability.npy"
    )

    ground_truth = np.load(
        case_dir
        / "ground_truth.npy"
    )

    if probability.shape != ground_truth.shape:

        raise ValueError(
            f"Shape mismatch: "
            f"{case_id}"
        )

    ground_truth = (
        ground_truth > 0
    )

    cases.append(
        {
            "case_id": case_id,

            "probability": (
                probability.astype(
                    np.float32
                )
            ),

            "ground_truth": ground_truth,
        }
    )

    print(
        f"Loaded {index:3d}/"
        f"{len(case_dirs):3d}: "
        f"{case_id}"
    )


# ============================================================
# JOINT SEARCH
# ============================================================

all_results = []

configuration_number = 0

total_configurations = (
    len(THRESHOLDS)
    *
    len(MIN_COMPONENT_SIZES)
)


for threshold in THRESHOLDS:

    for minimum_size in MIN_COMPONENT_SIZES:

        configuration_number += 1

        print()
        print(
            "-" * 110
        )

        print(
            f"Configuration "
            f"{configuration_number}/"
            f"{total_configurations}"
        )

        print(
            f"Threshold = "
            f"{threshold:.3f} | "
            f"Minimum component = "
            f"{minimum_size}"
        )

        print(
            "-" * 110
        )

        case_metrics = []

        component_counts = []

        for case in cases:

            probability = case[
                "probability"
            ]

            ground_truth = case[
                "ground_truth"
            ]

            raw_prediction = (
                probability
                >= threshold
            )

            filtered_prediction = (
                remove_small_components(
                    raw_prediction,
                    minimum_size
                )
            )

            metrics = calculate_metrics(
                filtered_prediction,
                ground_truth
            )

            component_stats = (
                component_statistics(
                    filtered_prediction
                )
            )

            metrics[
                "case_id"
            ] = case[
                "case_id"
            ]

            metrics[
                "threshold"
            ] = float(
                threshold
            )

            metrics[
                "minimum_component_size"
            ] = int(
                minimum_size
            )

            metrics[
                "component_count"
            ] = (
                component_stats[
                    "component_count"
                ]
            )

            case_metrics.append(
                metrics
            )

            component_counts.append(
                component_stats[
                    "component_count"
                ]
            )

        precisions = np.array(
            [
                m["precision"]
                for m in case_metrics
            ],
            dtype=np.float64
        )

        recalls = np.array(
            [
                m["recall"]
                for m in case_metrics
            ],
            dtype=np.float64
        )

        dices = np.array(
            [
                m["dice"]
                for m in case_metrics
            ],
            dtype=np.float64
        )

        ious = np.array(
            [
                m["iou"]
                for m in case_metrics
            ],
            dtype=np.float64
        )

        result = {

            "threshold": float(
                threshold
            ),

            "minimum_component_size": int(
                minimum_size
            ),

            "cases": len(
                case_metrics
            ),

            "mean_precision": float(
                np.mean(
                    precisions
                )
            ),

            "mean_recall": float(
                np.mean(
                    recalls
                )
            ),

            "mean_dice": float(
                np.mean(
                    dices
                )
            ),

            "median_dice": float(
                np.median(
                    dices
                )
            ),

            "mean_iou": float(
                np.mean(
                    ious
                )
            ),

            "median_iou": float(
                np.median(
                    ious
                )
            ),

            "mean_component_count": float(
                np.mean(
                    component_counts
                )
            ),

            "cases_dice_ge_0_50": int(
                np.sum(
                    dices >= 0.50
                )
            ),

            "cases_dice_ge_0_70": int(
                np.sum(
                    dices >= 0.70
                )
            ),

            "cases_dice_ge_0_80": int(
                np.sum(
                    dices >= 0.80
                )
            ),

            "case_metrics": case_metrics,
        }

        all_results.append(
            result
        )

        print(
            f"Mean Dice = "
            f"{result['mean_dice']:.6f} | "
            f"Precision = "
            f"{result['mean_precision']:.6f} | "
            f"Recall = "
            f"{result['mean_recall']:.6f} | "
            f"IoU = "
            f"{result['mean_iou']:.6f} | "
            f"Components = "
            f"{result['mean_component_count']:.2f}"
        )


# ============================================================
# SORT RESULTS BY MEAN DICE
# ============================================================

ranked_results = sorted(
    all_results,
    key=lambda result: result[
        "mean_dice"
    ],
    reverse=True
)


# ============================================================
# TOP CONFIGURATIONS
# ============================================================

print()
print("=" * 110)

print(
    "TOP JOINT CONFIGURATIONS BY MEAN DICE"
)

print("=" * 110)

print()

print(
    f"{'Rank':>6} "
    f"{'Threshold':>12} "
    f"{'MinSize':>10} "
    f"{'Precision':>12} "
    f"{'Recall':>12} "
    f"{'Dice':>12} "
    f"{'IoU':>12} "
    f"{'Components':>12}"
)

print(
    "-" * 110
)


for rank, result in enumerate(
    ranked_results[:15],
    start=1
):

    print(
        f"{rank:>6d} "
        f"{result['threshold']:>12.3f} "
        f"{result['minimum_component_size']:>10d} "
        f"{result['mean_precision']:>12.6f} "
        f"{result['mean_recall']:>12.6f} "
        f"{result['mean_dice']:>12.6f} "
        f"{result['mean_iou']:>12.6f} "
        f"{result['mean_component_count']:>12.2f}"
    )


# ============================================================
# BEST CONFIGURATION
# ============================================================

best_result = ranked_results[0]

best_threshold = (
    best_result[
        "threshold"
    ]
)

best_component_size = (
    best_result[
        "minimum_component_size"
    ]
)


# ============================================================
# SAVE COMPLETE REPORT
# ============================================================

report = {

    "experiment": (
        "meningioma_joint_postprocessing_analysis"
    ),

    "description": (
        "Joint search over probability "
        "threshold and 3D connected-component "
        "minimum size using saved full-volume "
        "development probability maps."
    ),

    "cases_evaluated": len(
        cases
    ),

    "thresholds": THRESHOLDS,

    "minimum_component_sizes": (
        MIN_COMPONENT_SIZES
    ),

    "best_threshold_by_mean_dice": (
        float(
            best_threshold
        )
    ),

    "best_component_size_by_mean_dice": (
        int(
            best_component_size
        )
    ),

    "best_result": best_result,

    "ranked_results": ranked_results,
}


report_path = (
    OUTPUT_ROOT
    / "meningioma_joint_postprocessing_report.json"
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        report,
        file,
        indent=2
    )


# ============================================================
# SAVE COMPACT SUMMARY
# ============================================================

summary = []

for result in ranked_results:

    summary.append(
        {
            "threshold": (
                result[
                    "threshold"
                ]
            ),

            "minimum_component_size": (
                result[
                    "minimum_component_size"
                ]
            ),

            "mean_precision": (
                result[
                    "mean_precision"
                ]
            ),

            "mean_recall": (
                result[
                    "mean_recall"
                ]
            ),

            "mean_dice": (
                result[
                    "mean_dice"
                ]
            ),

            "median_dice": (
                result[
                    "median_dice"
                ]
            ),

            "mean_iou": (
                result[
                    "mean_iou"
                ]
            ),

            "median_iou": (
                result[
                    "median_iou"
                ]
            ),

            "mean_component_count": (
                result[
                    "mean_component_count"
                ]
            ),

            "cases_dice_ge_0_50": (
                result[
                    "cases_dice_ge_0_50"
                ]
            ),

            "cases_dice_ge_0_70": (
                result[
                    "cases_dice_ge_0_70"
                ]
            ),

            "cases_dice_ge_0_80": (
                result[
                    "cases_dice_ge_0_80"
                ]
            ),
        }
    )


summary_path = (
    OUTPUT_ROOT
    / "meningioma_joint_postprocessing_summary.json"
)


with open(
    summary_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=2
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 110)

print(
    "JOINT THRESHOLD + "
    "CONNECTED-COMPONENT ANALYSIS COMPLETE"
)

print("=" * 110)

print()

print(
    f"Cases evaluated: "
    f"{len(cases)}"
)

print()

print(
    f"Best threshold: "
    f"{best_threshold:.3f}"
)

print(
    f"Best minimum component size: "
    f"{best_component_size} voxels"
)

print()

print(
    f"Best mean Precision: "
    f"{best_result['mean_precision']:.6f}"
)

print(
    f"Best mean Recall: "
    f"{best_result['mean_recall']:.6f}"
)

print(
    f"Best mean Dice: "
    f"{best_result['mean_dice']:.6f}"
)

print(
    f"Best mean IoU: "
    f"{best_result['mean_iou']:.6f}"
)

print(
    f"Mean components/case: "
    f"{best_result['mean_component_count']:.2f}"
)

print()

print(
    "Detailed report:"
)

print(
    report_path
)

print()

print(
    "Summary:"
)

print(
    summary_path
)

print()

print("=" * 110)