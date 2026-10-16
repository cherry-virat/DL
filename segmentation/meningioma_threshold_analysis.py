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
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np


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
    / "meningioma_threshold_analysis"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# THRESHOLDS
# ============================================================

THRESHOLDS = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
]


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    prediction,
    ground_truth
):
    prediction = prediction.astype(bool)
    ground_truth = ground_truth.astype(bool)

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

    predicted_voxels = prediction.sum()
    ground_truth_voxels = ground_truth.sum()

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
        / (2.0 * tp + fp + fn)
        if (2.0 * tp + fp + fn) > 0
        else 1.0
    )

    iou = (
        tp / (tp + fp + fn)
        if (tp + fp + fn) > 0
        else 1.0
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "dice": float(dice),
        "iou": float(iou),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "predicted_tumor_voxels": int(predicted_voxels),
        "ground_truth_tumor_voxels": int(ground_truth_voxels),
    }


# ============================================================
# LOAD CASES
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
        (path / "tumor_probability.npy").exists()
        and
        (path / "ground_truth.npy").exists()
    )
]

if not case_dirs:
    raise RuntimeError(
        "No inference cases with tumor_probability.npy "
        "and ground_truth.npy were found."
    )


print("=" * 100)
print("MENINGIOMA THRESHOLD ANALYSIS")
print("=" * 100)

print()
print("Inference root:")
print(INFERENCE_ROOT)

print()
print("Output root:")
print(OUTPUT_ROOT)

print()
print(f"Cases found: {len(case_dirs)}")

print()
print("Thresholds:")
print(THRESHOLDS)

print()


# ============================================================
# LOAD ALL PROBABILITIES / GROUND TRUTH
# ============================================================

cases = []

for index, case_dir in enumerate(case_dirs, start=1):

    case_id = case_dir.name

    probability_path = (
        case_dir
        / "tumor_probability.npy"
    )

    ground_truth_path = (
        case_dir
        / "ground_truth.npy"
    )

    probability = np.load(
        probability_path
    )

    ground_truth = np.load(
        ground_truth_path
    )

    if probability.shape != ground_truth.shape:
        raise ValueError(
            f"Shape mismatch for {case_id}: "
            f"probability={probability.shape}, "
            f"ground_truth={ground_truth.shape}"
        )

    probability = probability.astype(
        np.float32
    )

    ground_truth = (
        ground_truth > 0
    )

    cases.append(
        {
            "case_id": case_id,
            "probability": probability,
            "ground_truth": ground_truth,
        }
    )

    print(
        f"Loaded {index:3d}/{len(case_dirs):3d}: "
        f"{case_id} | "
        f"shape={probability.shape} | "
        f"GT voxels={int(ground_truth.sum())}"
    )


# ============================================================
# THRESHOLD EVALUATION
# ============================================================

all_results = []


for threshold in THRESHOLDS:

    print()
    print("-" * 100)
    print(
        f"THRESHOLD = {threshold:.2f}"
    )
    print("-" * 100)

    case_metrics = []

    for case in cases:

        case_id = case["case_id"]
        probability = case["probability"]
        ground_truth = case["ground_truth"]

        prediction = (
            probability >= threshold
        )

        metrics = calculate_metrics(
            prediction,
            ground_truth
        )

        metrics["case_id"] = case_id
        metrics["threshold"] = threshold

        case_metrics.append(metrics)

    precisions = np.array(
        [m["precision"] for m in case_metrics],
        dtype=np.float64
    )

    recalls = np.array(
        [m["recall"] for m in case_metrics],
        dtype=np.float64
    )

    dices = np.array(
        [m["dice"] for m in case_metrics],
        dtype=np.float64
    )

    ious = np.array(
        [m["iou"] for m in case_metrics],
        dtype=np.float64
    )

    predicted_voxels = np.array(
        [
            m["predicted_tumor_voxels"]
            for m in case_metrics
        ],
        dtype=np.float64
    )

    positive_cases = int(
        np.sum(dices > 0)
    )

    dice_50_cases = int(
        np.sum(dices >= 0.50)
    )

    dice_70_cases = int(
        np.sum(dices >= 0.70)
    )

    dice_80_cases = int(
        np.sum(dices >= 0.80)
    )

    summary = {
        "threshold": float(threshold),

        "cases": len(case_metrics),

        "mean_precision": float(
            np.mean(precisions)
        ),

        "mean_recall": float(
            np.mean(recalls)
        ),

        "mean_dice": float(
            np.mean(dices)
        ),

        "median_dice": float(
            np.median(dices)
        ),

        "mean_iou": float(
            np.mean(ious)
        ),

        "median_iou": float(
            np.median(ious)
        ),

        "mean_predicted_tumor_voxels": float(
            np.mean(predicted_voxels)
        ),

        "cases_with_dice_above_zero": positive_cases,

        "cases_with_dice_at_least_0.50": dice_50_cases,

        "cases_with_dice_at_least_0.70": dice_70_cases,

        "cases_with_dice_at_least_0.80": dice_80_cases,

        "case_metrics": case_metrics,
    }

    all_results.append(
        summary
    )

    print(
        f"Mean Precision : {summary['mean_precision']:.6f}"
    )

    print(
        f"Mean Recall    : {summary['mean_recall']:.6f}"
    )

    print(
        f"Mean Dice      : {summary['mean_dice']:.6f}"
    )

    print(
        f"Median Dice    : {summary['median_dice']:.6f}"
    )

    print(
        f"Mean IoU       : {summary['mean_iou']:.6f}"
    )

    print(
        f"Dice >= 0.50   : "
        f"{summary['cases_with_dice_at_least_0.50']}/"
        f"{len(case_metrics)}"
    )

    print(
        f"Dice >= 0.70   : "
        f"{summary['cases_with_dice_at_least_0.70']}/"
        f"{len(case_metrics)}"
    )


# ============================================================
# FIND BEST THRESHOLD BY MEAN DICE
# ============================================================

best_result = max(
    all_results,
    key=lambda result: result["mean_dice"]
)

best_threshold = best_result[
    "threshold"
]


# ============================================================
# COMPACT TABLE
# ============================================================

print()
print("=" * 100)
print("THRESHOLD SUMMARY")
print("=" * 100)

print()

print(
    f"{'Threshold':>10} "
    f"{'Precision':>12} "
    f"{'Recall':>12} "
    f"{'Dice':>12} "
    f"{'IoU':>12} "
    f"{'Dice>=0.50':>12}"
)

print("-" * 100)

for result in all_results:

    print(
        f"{result['threshold']:>10.2f} "
        f"{result['mean_precision']:>12.6f} "
        f"{result['mean_recall']:>12.6f} "
        f"{result['mean_dice']:>12.6f} "
        f"{result['mean_iou']:>12.6f} "
        f"{result['cases_with_dice_at_least_0.50']:>12d}"
    )


# ============================================================
# SAVE JSON
# ============================================================

report = {
    "experiment": "meningioma_threshold_analysis",

    "description": (
        "Threshold sweep over saved full-volume tumor "
        "probability maps from the 100-case development set."
    ),

    "inference_root": str(
        INFERENCE_ROOT
    ),

    "cases_evaluated": len(
        cases
    ),

    "thresholds": THRESHOLDS,

    "best_threshold_by_mean_dice": (
        float(best_threshold)
    ),

    "best_result": best_result,

    "results": all_results,
}


report_path = (
    OUTPUT_ROOT
    / "meningioma_threshold_report.json"
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
# SAVE COMPACT CSV-LIKE JSON
# ============================================================

summary_path = (
    OUTPUT_ROOT
    / "meningioma_threshold_summary.json"
)

summary_only = []

for result in all_results:

    summary_only.append(
        {
            "threshold": result["threshold"],
            "mean_precision": result["mean_precision"],
            "mean_recall": result["mean_recall"],
            "mean_dice": result["mean_dice"],
            "median_dice": result["median_dice"],
            "mean_iou": result["mean_iou"],
            "median_iou": result["median_iou"],
            "mean_predicted_tumor_voxels": (
                result[
                    "mean_predicted_tumor_voxels"
                ]
            ),
            "cases_with_dice_above_zero": (
                result[
                    "cases_with_dice_above_zero"
                ]
            ),
            "cases_with_dice_at_least_0.50": (
                result[
                    "cases_with_dice_at_least_0.50"
                ]
            ),
            "cases_with_dice_at_least_0.70": (
                result[
                    "cases_with_dice_at_least_0.70"
                ]
            ),
            "cases_with_dice_at_least_0.80": (
                result[
                    "cases_with_dice_at_least_0.80"
                ]
            ),
        }
    )


with open(
    summary_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary_only,
        file,
        indent=2
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 100)
print("MENINGIOMA THRESHOLD ANALYSIS COMPLETE")
print("=" * 100)

print()
print(
    f"Cases evaluated: {len(cases)}"
)

print()
print(
    f"Best threshold by mean Dice: "
    f"{best_threshold:.2f}"
)

print(
    f"Best mean Dice: "
    f"{best_result['mean_dice']:.6f}"
)

print(
    f"Best mean Precision: "
    f"{best_result['mean_precision']:.6f}"
)

print(
    f"Best mean Recall: "
    f"{best_result['mean_recall']:.6f}"
)

print(
    f"Best mean IoU: "
    f"{best_result['mean_iou']:.6f}"
)

print()
print("Detailed report:")
print(report_path)

print()
print("Summary:")
print(summary_path)

print()
print("=" * 100)