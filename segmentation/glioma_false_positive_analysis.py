"""
Glioma False-Positive Analysis
==============================

Analyzes already-generated full-volume predictions.

Purpose:
    Determine how much the model is over-segmenting.

Input:
    outputs/glioma_full_volume/<case>/
        full_volume_prediction.npy
        full_volume_ground_truth.npy

Output:
    outputs/glioma_false_positive_analysis/
        false_positive_analysis.txt
"""


from pathlib import Path

import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = (
    Path("outputs")
    / "glioma_full_volume"
)

OUTPUT_DIR = (
    Path("outputs")
    / "glioma_false_positive_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


CASES = [
    "BraTS-GLI-01666-000",
    "BraTS-GLI-01518-000",
    "BraTS-GLI-01497-000",
    "BraTS-GLI-01397-000",
    "BraTS-GLI-00291-000",
]


# ============================================================
# ANALYSIS
# ============================================================

def analyze_case(case_name):

    case_dir = (
        INPUT_DIR
        / case_name
    )

    prediction_path = (
        case_dir
        / "full_volume_prediction.npy"
    )

    ground_truth_path = (
        case_dir
        / "full_volume_ground_truth.npy"
    )

    prediction = np.load(
        prediction_path
    )

    ground_truth = np.load(
        ground_truth_path
    )

    if prediction.shape != ground_truth.shape:

        raise RuntimeError(
            f"Shape mismatch for {case_name}: "
            f"{prediction.shape} vs "
            f"{ground_truth.shape}"
        )

    predicted_tumor = (
        prediction > 0
    )

    actual_tumor = (
        ground_truth > 0
    )

    true_positive = np.logical_and(
        predicted_tumor,
        actual_tumor,
    ).sum()

    false_positive = np.logical_and(
        predicted_tumor,
        ~actual_tumor,
    ).sum()

    false_negative = np.logical_and(
        ~predicted_tumor,
        actual_tumor,
    ).sum()

    ground_truth_voxels = (
        actual_tumor.sum()
    )

    predicted_voxels = (
        predicted_tumor.sum()
    )

    precision = (
        true_positive
        /
        max(
            true_positive + false_positive,
            1,
        )
    )

    recall = (
        true_positive
        /
        max(
            true_positive + false_negative,
            1,
        )
    )

    dice = (
        2 * true_positive
        /
        max(
            predicted_voxels
            +
            ground_truth_voxels,
            1,
        )
    )

    iou = (
        true_positive
        /
        max(
            true_positive
            +
            false_positive
            +
            false_negative,
            1,
        )
    )

    volume_ratio = (
        predicted_voxels
        /
        max(
            ground_truth_voxels,
            1,
        )
    )

    return {
        "case": case_name,
        "ground_truth": int(
            ground_truth_voxels
        ),
        "predicted": int(
            predicted_voxels
        ),
        "tp": int(
            true_positive
        ),
        "fp": int(
            false_positive
        ),
        "fn": int(
            false_negative
        ),
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
        "volume_ratio": float(
            volume_ratio
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("GLIOMA FALSE-POSITIVE ANALYSIS")
    print("=" * 70)

    results = []

    for case_name in CASES:

        print(
            "\n" + "-" * 70
        )

        print(
            "Case:",
            case_name,
        )

        result = analyze_case(
            case_name
        )

        results.append(
            result
        )

        print(
            "Ground truth:",
            result["ground_truth"]
        )

        print(
            "Predicted:",
            result["predicted"]
        )

        print(
            "True positive:",
            result["tp"]
        )

        print(
            "False positive:",
            result["fp"]
        )

        print(
            "False negative:",
            result["fn"]
        )

        print(
            "Precision:",
            f"{result['precision']:.6f}"
        )

        print(
            "Recall:",
            f"{result['recall']:.6f}"
        )

        print(
            "Dice:",
            f"{result['dice']:.6f}"
        )

        print(
            "IoU:",
            f"{result['iou']:.6f}"
        )

        print(
            "Prediction / GT volume:",
            f"{result['volume_ratio']:.4f}x"
        )


    # ========================================================
    # AGGREGATE
    # ========================================================

    total_tp = sum(
        r["tp"]
        for r in results
    )

    total_fp = sum(
        r["fp"]
        for r in results
    )

    total_fn = sum(
        r["fn"]
        for r in results
    )

    total_gt = sum(
        r["ground_truth"]
        for r in results
    )

    total_predicted = sum(
        r["predicted"]
        for r in results
    )


    global_precision = (
        total_tp
        /
        max(
            total_tp + total_fp,
            1,
        )
    )

    global_recall = (
        total_tp
        /
        max(
            total_tp + total_fn,
            1,
        )
    )

    global_dice = (
        2 * total_tp
        /
        max(
            total_predicted
            +
            total_gt,
            1,
        )
    )

    global_iou = (
        total_tp
        /
        max(
            total_tp
            +
            total_fp
            +
            total_fn,
            1,
        )
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "AGGREGATE FALSE-POSITIVE ANALYSIS"
    )

    print(
        "=" * 70
    )

    print(
        "Total ground-truth tumor voxels:",
        total_gt,
    )

    print(
        "Total predicted tumor voxels:",
        total_predicted,
    )

    print(
        "Total true positives:",
        total_tp,
    )

    print(
        "Total false positives:",
        total_fp,
    )

    print(
        "Total false negatives:",
        total_fn,
    )

    print(
        "Global precision:",
        f"{global_precision:.6f}",
    )

    print(
        "Global recall:",
        f"{global_recall:.6f}",
    )

    print(
        "Global Dice:",
        f"{global_dice:.6f}",
    )

    print(
        "Global IoU:",
        f"{global_iou:.6f}",
    )


    # ========================================================
    # REPORT
    # ========================================================

    report_path = (
        OUTPUT_DIR
        /
        "false_positive_analysis.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "GLIOMA FALSE-POSITIVE ANALYSIS\n"
        )

        file.write(
            "=" * 70
            + "\n\n"
        )

        for result in results:

            file.write(
                f"Case: {result['case']}\n"
            )

            file.write(
                f"Ground truth: "
                f"{result['ground_truth']}\n"
            )

            file.write(
                f"Predicted: "
                f"{result['predicted']}\n"
            )

            file.write(
                f"TP: {result['tp']}\n"
            )

            file.write(
                f"FP: {result['fp']}\n"
            )

            file.write(
                f"FN: {result['fn']}\n"
            )

            file.write(
                f"Precision: "
                f"{result['precision']:.6f}\n"
            )

            file.write(
                f"Recall: "
                f"{result['recall']:.6f}\n"
            )

            file.write(
                f"Dice: "
                f"{result['dice']:.6f}\n"
            )

            file.write(
                f"IoU: "
                f"{result['iou']:.6f}\n"
            )

            file.write(
                f"Volume ratio: "
                f"{result['volume_ratio']:.6f}\n\n"
            )


        file.write(
            "=" * 70
            + "\n"
        )

        file.write(
            "GLOBAL RESULTS\n"
        )

        file.write(
            "=" * 70
            + "\n"
        )

        file.write(
            f"Precision: "
            f"{global_precision:.6f}\n"
        )

        file.write(
            f"Recall: "
            f"{global_recall:.6f}\n"
        )

        file.write(
            f"Dice: "
            f"{global_dice:.6f}\n"
        )

        file.write(
            f"IoU: "
            f"{global_iou:.6f}\n"
        )

        file.write(
            f"False positives: "
            f"{total_fp}\n"
        )

        file.write(
            f"False negatives: "
            f"{total_fn}\n"
        )


    print(
        "\nReport saved:"
    )

    print(
        report_path
    )


    print(
        "\n✓ FALSE-POSITIVE ANALYSIS COMPLETE"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()