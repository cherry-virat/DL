"""
Glioma Tumor Threshold Analysis
===============================

Purpose
-------
Evaluate different tumor-confidence thresholds using the
trained medium 3D U-Net.

Goal
----
Reduce false-positive tumor voxels while preserving recall
and Dice as much as possible.

Important
---------
This script does NOT retrain the model.

It uses the same:

    process_case()
    padding
    64 x 64 x 64 ROI
    0.5 sliding-window overlap
    medium 3D U-Net

as the successful full-volume inference.

Tumor definition:

    class 1 + class 2 + class 3

Tumor probability:

    P(tumor) = P(class 1) + P(class 2) + P(class 3)

A threshold is then applied:

    P(tumor) >= threshold
        -> tumor

    P(tumor) < threshold
        -> background
"""


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import sys

# Allow direct execution from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference

from preprocessing.glioma_dataset import (
    DATASET_ROOT,
    get_cases,
    pad_to_patch_size,
)

from preprocessing.glioma_preprocess import (
    process_case,
)

from preprocessing.glioma_split import (
    create_train_val_split,
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

TRAIN_CASES_FOR_TEST = 20

VAL_CASES_FOR_TEST = 5

INPUT_CHANNELS = 4

OUTPUT_CLASSES = 4


# ------------------------------------------------------------
# Patch / sliding-window configuration
# ------------------------------------------------------------

PATCH_SIZE = (
    64,
    64,
    64,
)

ROI_SIZE = PATCH_SIZE

OVERLAP = 0.5

SW_BATCH_SIZE = 1


# ------------------------------------------------------------
# Same architecture as medium training
# ------------------------------------------------------------

MODEL_CHANNELS = (
    16,
    32,
    64,
    128,
    256,
)

DROPOUT = 0.0


# ============================================================
# THRESHOLDS
# ============================================================

THRESHOLDS = [
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
    0.85,
    0.90,
]


# ============================================================
# CHECKPOINT
# ============================================================

CHECKPOINT_PATH = (
    Path("outputs")
    / "glioma_training"
    / "glioma_best_medium_experiment.pth"
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = (
    Path("outputs")
    / "glioma_threshold_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
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
# RANDOM SEED
# ============================================================

torch.manual_seed(
    SEED
)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        SEED
    )


# ============================================================
# CREATE MODEL
# ============================================================

def create_model():

    model = UNet(
        spatial_dims=3,

        in_channels=INPUT_CHANNELS,

        out_channels=OUTPUT_CLASSES,

        channels=MODEL_CHANNELS,

        strides=(
            2,
            2,
            2,
            2,
        ),

        num_res_units=2,

        dropout=DROPOUT,
    )

    return model.to(
        DEVICE
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print(
        "\nLoading model..."
    )

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            "\nCheckpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    model = create_model()

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    if "model_state_dict" not in checkpoint:

        raise RuntimeError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    print(
        "✓ Model loaded."
    )

    print(
        "Checkpoint:",
        CHECKPOINT_PATH
    )

    if "epoch" in checkpoint:

        print(
            "Checkpoint epoch:",
            checkpoint["epoch"]
        )

    return model


# ============================================================
# CALCULATE BINARY TUMOR METRICS
# ============================================================

def calculate_metrics(
    prediction,
    ground_truth,
):
    """
    Parameters
    ----------
    prediction:
        Boolean tensor, shape (D,H,W)

    ground_truth:
        Boolean tensor, shape (D,H,W)

    Returns
    -------
    Dictionary containing:

        TP
        FP
        FN
        precision
        recall
        dice
        iou
    """

    # --------------------------------------------------------
    # Force both tensors onto CPU.
    #
    # This prevents CUDA/CPU mismatch.
    # --------------------------------------------------------

    prediction = (
        prediction
        .detach()
        .cpu()
        .bool()
    )

    ground_truth = (
        ground_truth
        .detach()
        .cpu()
        .bool()
    )

    # --------------------------------------------------------
    # Shape check
    # --------------------------------------------------------

    if prediction.shape != ground_truth.shape:

        raise RuntimeError(
            "\nMETRIC SHAPE MISMATCH\n"
            f"Prediction: {tuple(prediction.shape)}\n"
            f"Ground truth: {tuple(ground_truth.shape)}"
        )

    # --------------------------------------------------------
    # True positives
    # --------------------------------------------------------

    true_positive = int(
        (
            prediction
            &
            ground_truth
        )
        .sum()
        .item()
    )

    # --------------------------------------------------------
    # False positives
    # --------------------------------------------------------

    false_positive = int(
        (
            prediction
            &
            ~ground_truth
        )
        .sum()
        .item()
    )

    # --------------------------------------------------------
    # False negatives
    # --------------------------------------------------------

    false_negative = int(
        (
            ~prediction
            &
            ground_truth
        )
        .sum()
        .item()
    )

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    precision_denominator = (
        true_positive
        +
        false_positive
    )

    if precision_denominator > 0:

        precision = (
            true_positive
            /
            precision_denominator
        )

    else:

        precision = 0.0

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    recall_denominator = (
        true_positive
        +
        false_negative
    )

    if recall_denominator > 0:

        recall = (
            true_positive
            /
            recall_denominator
        )

    else:

        recall = 0.0

    # --------------------------------------------------------
    # Dice
    # --------------------------------------------------------

    dice_denominator = (
        2 * true_positive
        +
        false_positive
        +
        false_negative
    )

    if dice_denominator > 0:

        dice = (
            2
            *
            true_positive
            /
            dice_denominator
        )

    else:

        dice = 0.0

    # --------------------------------------------------------
    # IoU
    # --------------------------------------------------------

    iou_denominator = (
        true_positive
        +
        false_positive
        +
        false_negative
    )

    if iou_denominator > 0:

        iou = (
            true_positive
            /
            iou_denominator
        )

    else:

        iou = 0.0

    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "precision": precision,
        "recall": recall,
        "dice": dice,
        "iou": iou,
    }


# ============================================================
# GET TUMOR PROBABILITY FOR ONE CASE
# ============================================================

def get_case_probability(
    model,
    case_dir,
):
    """
    Run full-volume inference and return:

        tumor_probability
        ground_truth

    tumor_probability:
        shape (D,H,W)

    ground_truth:
        shape (D,H,W)
    """

    # ========================================================
    # PREPROCESS USING THE EXACT TRAINING PIPELINE
    # ========================================================

    image, mask = process_case(
        case_dir
    )

    image = np.asarray(
        image,
        dtype=np.float32,
    )

    mask = np.asarray(
        mask,
        dtype=np.int64,
    )

    # ========================================================
    # VALIDATE
    # ========================================================

    if image.ndim != 4:

        raise RuntimeError(
            "Expected image shape (C,D,H,W).\n"
            f"Received: {image.shape}"
        )

    if mask.ndim != 3:

        raise RuntimeError(
            "Expected mask shape (D,H,W).\n"
            f"Received: {mask.shape}"
        )

    if image.shape[1:] != mask.shape:

        raise RuntimeError(
            "\nIMAGE/MASK SHAPE MISMATCH\n"
            f"Image: {image.shape}\n"
            f"Mask: {mask.shape}"
        )

    # ========================================================
    # SAVE ORIGINAL PREPROCESSED SHAPE
    # ========================================================

    original_shape = tuple(
        image.shape[1:]
    )

    # ========================================================
    # ADD MASK CHANNEL FOR PADDING
    # ========================================================

    mask_4d = np.expand_dims(
        mask,
        axis=0,
    )

    # ========================================================
    # PAD IF NECESSARY
    # ========================================================

    padded_image, padded_mask = (
        pad_to_patch_size(
            image,
            mask_4d,
            PATCH_SIZE,
        )
    )

    # ========================================================
    # TORCH IMAGE
    # ========================================================

    image_tensor = torch.from_numpy(
        padded_image
    ).float()

    # ========================================================
    # ADD BATCH DIMENSION
    # ========================================================

    input_tensor = (
        image_tensor
        .unsqueeze(0)
        .to(
            DEVICE,
            non_blocking=True,
        )
    )

    # ========================================================
    # SLIDING-WINDOW INFERENCE
    # ========================================================

    with torch.inference_mode():

        logits = sliding_window_inference(
            inputs=input_tensor,

            roi_size=ROI_SIZE,

            sw_batch_size=SW_BATCH_SIZE,

            predictor=model,

            overlap=OVERLAP,

            mode="gaussian",
        )

        # ----------------------------------------------------
        # Convert logits to probabilities
        # ----------------------------------------------------

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        # ----------------------------------------------------
        # Tumor probability:
        #
        # class 1 + class 2 + class 3
        # ----------------------------------------------------

        tumor_probability = (
            probabilities[:, 1:, ...]
            .sum(dim=1)
        )

    # ========================================================
    # REMOVE BATCH DIMENSION
    #
    # Current:
    #
    # (1,D,H,W)
    #
    # becomes:
    #
    # (D,H,W)
    # ========================================================

    tumor_probability = (
        tumor_probability[
            0
        ]
        .detach()
        .cpu()
    )

    # ========================================================
    # GROUND TRUTH
    # ========================================================

    ground_truth = torch.from_numpy(
        padded_mask
    ).long()

    # padded_mask shape:
    #
    # (1,D,H,W)
    #
    # Remove channel dimension:
    #
    # (D,H,W)

    ground_truth = (
        ground_truth[
            0
        ]
        .detach()
        .cpu()
    )

    # ========================================================
    # REMOVE PADDING
    # ========================================================

    d, h, w = original_shape

    tumor_probability = (
        tumor_probability[
            :d,
            :h,
            :w,
        ]
    )

    ground_truth = (
        ground_truth[
            :d,
            :h,
            :w,
        ]
    )

    # ========================================================
    # FINAL SHAPE CHECK
    # ========================================================

    if tumor_probability.shape != ground_truth.shape:

        raise RuntimeError(
            "\nFINAL PROBABILITY/GROUND-TRUTH "
            "SHAPE MISMATCH\n"
            f"Probability: "
            f"{tuple(tumor_probability.shape)}\n"
            f"Ground truth: "
            f"{tuple(ground_truth.shape)}"
        )

    return (
        tumor_probability,
        ground_truth,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "GLIOMA TUMOR THRESHOLD ANALYSIS"
    )

    print(
        "=" * 70
    )

    print(
        "\nDataset:",
        DATASET_ROOT
    )

    print(
        "Device:",
        DEVICE
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print(
        "\nCheckpoint:",
        CHECKPOINT_PATH
    )

    # ========================================================
    # LOAD DATASET
    # ========================================================

    cases = get_cases(
        DATASET_ROOT
    )

    print(
        "\nTotal cases:",
        len(cases)
    )

    # ========================================================
    # TRAIN / VALIDATION SPLIT
    # ========================================================

    train_cases, validation_cases = (
        create_train_val_split(
            cases,
            train_ratio=0.90,
            seed=SEED,
        )
    )

    # ========================================================
    # SAME TEST CONFIGURATION AS FULL-VOLUME INFERENCE
    # ========================================================

    train_cases = train_cases[
        :TRAIN_CASES_FOR_TEST
    ]

    validation_cases = validation_cases[
        :VAL_CASES_FOR_TEST
    ]

    # ========================================================
    # OVERLAP CHECK
    # ========================================================

    train_names = {
        case.name
        for case in train_cases
    }

    validation_names = {
        case.name
        for case in validation_cases
    }

    overlap = (
        train_names
        &
        validation_names
    )

    if overlap:

        raise RuntimeError(
            "Training/validation overlap detected:\n"
            f"{overlap}"
        )

    print(
        "\n✓ No training/validation overlap."
    )

    # ========================================================
    # VALIDATION CASES
    # ========================================================

    print(
        "\nValidation cases:"
    )

    for index, case in enumerate(
        validation_cases,
        start=1,
    ):

        print(
            f"{index}. {case.name}"
        )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    model = load_model()

    # ========================================================
    # RUN INFERENCE ON ALL CASES ONCE
    #
    # We save the probability maps in memory so thresholds
    # can be evaluated without repeatedly running the model.
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "RUNNING FULL-VOLUME INFERENCE"
    )

    print(
        "=" * 70
    )

    case_data = []

    for index, case_dir in enumerate(
        validation_cases,
        start=1,
    ):

        print(
            "\n" + "-" * 70
        )

        print(
            f"CASE {index}/{len(validation_cases)}"
        )

        print(
            "-" * 70
        )

        print(
            "Case:",
            case_dir.name
        )

        probability, ground_truth = (
            get_case_probability(
                model,
                case_dir,
            )
        )

        print(
            "Tumor probability shape:",
            tuple(probability.shape)
        )

        print(
            "Ground-truth shape:",
            tuple(ground_truth.shape)
        )

        ground_truth_tumor_voxels = int(
            (
                ground_truth > 0
            )
            .sum()
            .item()
        )

        print(
            "Ground-truth tumor voxels:",
            ground_truth_tumor_voxels
        )

        print(
            "Probability minimum:",
            f"{probability.min().item():.6f}"
        )

        print(
            "Probability maximum:",
            f"{probability.max().item():.6f}"
        )

        print(
            "Probability mean:",
            f"{probability.mean().item():.6f}"
        )

        case_data.append(
            {
                "case":
                    case_dir.name,

                "probability":
                    probability,

                "ground_truth":
                    ground_truth,
            }
        )

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    # ========================================================
    # THRESHOLD ANALYSIS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "THRESHOLD RESULTS"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "Threshold | Precision | Recall | Dice | IoU | False Positives"
    )

    print(
        "-" * 70
    )

    results = []

    # ========================================================
    # TEST EVERY THRESHOLD
    # ========================================================

    for threshold in THRESHOLDS:

        total_tp = 0

        total_fp = 0

        total_fn = 0

        # ----------------------------------------------------
        # Evaluate all validation cases
        # ----------------------------------------------------

        for case in case_data:

            probability = (
                case["probability"]
            )

            ground_truth = (
                case["ground_truth"]
                > 0
            )

            # ------------------------------------------------
            # Apply threshold
            # ------------------------------------------------

            prediction = (
                probability
                >= threshold
            )

            metrics = calculate_metrics(
                prediction,
                ground_truth,
            )

            total_tp += (
                metrics["tp"]
            )

            total_fp += (
                metrics["fp"]
            )

            total_fn += (
                metrics["fn"]
            )

        # ----------------------------------------------------
        # Aggregate precision
        # ----------------------------------------------------

        precision_denominator = (
            total_tp
            +
            total_fp
        )

        if precision_denominator > 0:

            precision = (
                total_tp
                /
                precision_denominator
            )

        else:

            precision = 0.0

        # ----------------------------------------------------
        # Aggregate recall
        # ----------------------------------------------------

        recall_denominator = (
            total_tp
            +
            total_fn
        )

        if recall_denominator > 0:

            recall = (
                total_tp
                /
                recall_denominator
            )

        else:

            recall = 0.0

        # ----------------------------------------------------
        # Aggregate Dice
        # ----------------------------------------------------

        dice_denominator = (
            2 * total_tp
            +
            total_fp
            +
            total_fn
        )

        if dice_denominator > 0:

            dice = (
                2
                *
                total_tp
                /
                dice_denominator
            )

        else:

            dice = 0.0

        # ----------------------------------------------------
        # Aggregate IoU
        # ----------------------------------------------------

        iou_denominator = (
            total_tp
            +
            total_fp
            +
            total_fn
        )

        if iou_denominator > 0:

            iou = (
                total_tp
                /
                iou_denominator
            )

        else:

            iou = 0.0

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        result = {
            "threshold":
                threshold,

            "precision":
                precision,

            "recall":
                recall,

            "dice":
                dice,

            "iou":
                iou,

            "tp":
                total_tp,

            "fp":
                total_fp,

            "fn":
                total_fn,
        }

        results.append(
            result
        )

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print(
            f"{threshold:9.2f} | "
            f"{precision:9.6f} | "
            f"{recall:6.6f} | "
            f"{dice:6.6f} | "
            f"{iou:6.6f} | "
            f"{total_fp}"
        )

    # ========================================================
    # BEST PRECISION
    # ========================================================

    best_precision = max(
        results,
        key=lambda x: x["precision"]
    )

    # ========================================================
    # BEST DICE
    # ========================================================

    best_dice = max(
        results,
        key=lambda x: x["dice"]
    )

    # ========================================================
    # BEST IOU
    # ========================================================

    best_iou = max(
        results,
        key=lambda x: x["iou"]
    )

    # ========================================================
    # BALANCED THRESHOLD
    #
    # We require precision >= 0.90.
    #
    # Among those thresholds, choose the highest Dice.
    # ========================================================

    balanced_candidates = [
        result
        for result in results
        if result["precision"] >= 0.90
    ]

    if balanced_candidates:

        best_balanced = max(
            balanced_candidates,
            key=lambda x: x["dice"]
        )

    else:

        best_balanced = best_dice

    # ========================================================
    # LOW-FP THRESHOLD
    #
    # Choose the threshold with the smallest FP while
    # keeping recall >= 0.85.
    # ========================================================

    low_fp_candidates = [
        result
        for result in results
        if result["recall"] >= 0.85
    ]

    if low_fp_candidates:

        best_low_fp = min(
            low_fp_candidates,
            key=lambda x: x["fp"]
        )

    else:

        best_low_fp = min(
            results,
            key=lambda x: x["fp"]
        )

    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "THRESHOLD RECOMMENDATIONS"
    )

    print(
        "=" * 70
    )

    print(
        "\nBest precision:"
    )

    print(
        f"Threshold: "
        f"{best_precision['threshold']:.2f}"
    )

    print(
        f"Precision: "
        f"{best_precision['precision']:.6f}"
    )

    print(
        f"Recall: "
        f"{best_precision['recall']:.6f}"
    )

    print(
        f"Dice: "
        f"{best_precision['dice']:.6f}"
    )

    print(
        f"False positives: "
        f"{best_precision['fp']}"
    )

    # --------------------------------------------------------

    print(
        "\nBest Dice:"
    )

    print(
        f"Threshold: "
        f"{best_dice['threshold']:.2f}"
    )

    print(
        f"Precision: "
        f"{best_dice['precision']:.6f}"
    )

    print(
        f"Recall: "
        f"{best_dice['recall']:.6f}"
    )

    print(
        f"Dice: "
        f"{best_dice['dice']:.6f}"
    )

    print(
        f"False positives: "
        f"{best_dice['fp']}"
    )

    # --------------------------------------------------------

    print(
        "\nBest balanced threshold "
        "(precision >= 0.90):"
    )

    print(
        f"Threshold: "
        f"{best_balanced['threshold']:.2f}"
    )

    print(
        f"Precision: "
        f"{best_balanced['precision']:.6f}"
    )

    print(
        f"Recall: "
        f"{best_balanced['recall']:.6f}"
    )

    print(
        f"Dice: "
        f"{best_balanced['dice']:.6f}"
    )

    print(
        f"False positives: "
        f"{best_balanced['fp']}"
    )

    # --------------------------------------------------------

    print(
        "\nLowest-FP threshold "
        "(recall >= 0.85):"
    )

    print(
        f"Threshold: "
        f"{best_low_fp['threshold']:.2f}"
    )

    print(
        f"Precision: "
        f"{best_low_fp['precision']:.6f}"
    )

    print(
        f"Recall: "
        f"{best_low_fp['recall']:.6f}"
    )

    print(
        f"Dice: "
        f"{best_low_fp['dice']:.6f}"
    )

    print(
        f"False positives: "
        f"{best_low_fp['fp']}"
    )

    # ========================================================
    # SAVE REPORT
    # ========================================================

    report_path = (
        OUTPUT_DIR
        /
        "threshold_results.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "GLIOMA TUMOR THRESHOLD ANALYSIS\n"
        )

        file.write(
            "=" * 70
            +
            "\n\n"
        )

        file.write(
            f"Checkpoint: "
            f"{CHECKPOINT_PATH}\n"
        )

        file.write(
            f"ROI size: "
            f"{ROI_SIZE}\n"
        )

        file.write(
            f"Overlap: "
            f"{OVERLAP}\n\n"
        )

        file.write(
            "THRESHOLD RESULTS\n"
        )

        file.write(
            "-" * 70
            +
            "\n"
        )

        file.write(
            "Threshold,Precision,Recall,Dice,IoU,TP,FP,FN\n"
        )

        for result in results:

            file.write(
                f"{result['threshold']:.2f},"
                f"{result['precision']:.6f},"
                f"{result['recall']:.6f},"
                f"{result['dice']:.6f},"
                f"{result['iou']:.6f},"
                f"{result['tp']},"
                f"{result['fp']},"
                f"{result['fn']}\n"
            )

        file.write(
            "\n\nBEST PRECISION\n"
        )

        file.write(
            str(best_precision)
            +
            "\n"
        )

        file.write(
            "\nBEST DICE\n"
        )

        file.write(
            str(best_dice)
            +
            "\n"
        )

        file.write(
            "\nBEST IOU\n"
        )

        file.write(
            str(best_iou)
            +
            "\n"
        )

        file.write(
            "\nBEST BALANCED THRESHOLD\n"
        )

        file.write(
            str(best_balanced)
            +
            "\n"
        )

        file.write(
            "\nLOWEST-FP THRESHOLD\n"
        )

        file.write(
            str(best_low_fp)
            +
            "\n"
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "REPORT SAVED"
    )

    print(
        "=" * 70
    )

    print(
        report_path
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "THRESHOLD ANALYSIS COMPLETE"
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
