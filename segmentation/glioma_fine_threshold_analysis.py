"""
Glioma Fine Threshold Analysis
==============================

Purpose
-------
Find the best tumor probability threshold around the current
best threshold of 0.60.

Current baseline:
    Precision = 0.919470
    Recall    = 0.914667
    Dice      = 0.917062
    FP        = 40474

Current best threshold:
    0.60

Current result:
    Precision = 0.932302
    Recall    = 0.907461
    Dice      = 0.919714
    FP        = 33292

This script performs full-volume inference ONCE per case and
then evaluates thresholds from 0.50 to 0.70 in increments
of 0.01.
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

PATCH_SIZE = (
    64,
    64,
    64,
)

ROI_SIZE = PATCH_SIZE

OVERLAP = 0.5

SW_BATCH_SIZE = 1

MODEL_CHANNELS = (
    16,
    32,
    64,
    128,
    256,
)

DROPOUT = 0.0


# ============================================================
# FINE THRESHOLDS
# ============================================================

THRESHOLDS = [
    round(
        value,
        2,
    )
    for value in np.arange(
        0.50,
        0.701,
        0.01,
    )
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
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    Path("outputs")
    / "glioma_fine_threshold_analysis"
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
# SEED
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
            "Checkpoint not found:\n"
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

    return model


# ============================================================
# GET CASE PROBABILITY
# ============================================================

def get_case_probability(
    model,
    case_dir,
):

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

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

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
            "Image/mask mismatch.\n"
            f"Image: {image.shape}\n"
            f"Mask : {mask.shape}"
        )

    original_shape = tuple(
        image.shape[1:]
    )

    # --------------------------------------------------------
    # Add mask channel for padding
    # --------------------------------------------------------

    mask_4d = np.expand_dims(
        mask,
        axis=0,
    )

    # --------------------------------------------------------
    # Pad
    # --------------------------------------------------------

    padded_image, padded_mask = (
        pad_to_patch_size(
            image,
            mask_4d,
            PATCH_SIZE,
        )
    )

    # --------------------------------------------------------
    # Image tensor
    # --------------------------------------------------------

    image_tensor = torch.from_numpy(
        padded_image
    ).float()

    input_tensor = (
        image_tensor
        .unsqueeze(0)
        .to(
            DEVICE,
            non_blocking=True,
        )
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    with torch.inference_mode():

        logits = sliding_window_inference(
            inputs=input_tensor,
            roi_size=ROI_SIZE,
            sw_batch_size=SW_BATCH_SIZE,
            predictor=model,
            overlap=OVERLAP,
            mode="gaussian",
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        tumor_probability = (
            probabilities[:, 1:, ...]
            .sum(dim=1)
        )

    # --------------------------------------------------------
    # Remove batch dimension
    # --------------------------------------------------------

    tumor_probability = (
        tumor_probability[0]
        .detach()
        .cpu()
    )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    ground_truth = torch.from_numpy(
        padded_mask
    ).long()

    ground_truth = (
        ground_truth[0]
        .detach()
        .cpu()
    )

    # --------------------------------------------------------
    # Remove padding
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Final check
    # --------------------------------------------------------

    if (
        tumor_probability.shape
        !=
        ground_truth.shape
    ):

        raise RuntimeError(
            "Probability/ground-truth "
            "shape mismatch.\n"
            f"Probability: "
            f"{tumor_probability.shape}\n"
            f"Ground truth: "
            f"{ground_truth.shape}"
        )

    return (
        tumor_probability,
        ground_truth,
    )


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    prediction,
    ground_truth,
):

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

    if prediction.shape != ground_truth.shape:

        raise RuntimeError(
            "Prediction/ground-truth "
            "shape mismatch."
        )

    tp = int(
        (
            prediction
            &
            ground_truth
        )
        .sum()
        .item()
    )

    fp = int(
        (
            prediction
            &
            ~ground_truth
        )
        .sum()
        .item()
    )

    fn = int(
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

    if (
        tp + fp
    ) > 0:

        precision = (
            tp
            /
            (tp + fp)
        )

    else:

        precision = 0.0

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    if (
        tp + fn
    ) > 0:

        recall = (
            tp
            /
            (tp + fn)
        )

    else:

        recall = 0.0

    # --------------------------------------------------------
    # Dice
    # --------------------------------------------------------

    dice_denominator = (
        2 * tp
        +
        fp
        +
        fn
    )

    if dice_denominator > 0:

        dice = (
            2
            *
            tp
            /
            dice_denominator
        )

    else:

        dice = 0.0

    # --------------------------------------------------------
    # IoU
    # --------------------------------------------------------

    iou_denominator = (
        tp
        +
        fp
        +
        fn
    )

    if iou_denominator > 0:

        iou = (
            tp
            /
            iou_denominator
        )

    else:

        iou = 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "dice": dice,
        "iou": iou,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "GLIOMA FINE THRESHOLD ANALYSIS"
    )

    print(
        "=" * 70
    )

    print(
        "\nDevice:",
        DEVICE
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print(
        "\nThreshold range:"
    )

    print(
        "0.50 → 0.70"
    )

    print(
        "Step: 0.01"
    )

    # ========================================================
    # GET CASES
    # ========================================================

    cases = get_cases(
        DATASET_ROOT
    )

    # ========================================================
    # SPLIT
    # ========================================================

    train_cases, validation_cases = (
        create_train_val_split(
            cases,
            train_ratio=0.90,
            seed=SEED,
        )
    )

    train_cases = train_cases[
        :TRAIN_CASES_FOR_TEST
    ]

    validation_cases = validation_cases[
        :VAL_CASES_FOR_TEST
    ]

    # ========================================================
    # CHECK OVERLAP
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
    # MODEL
    # ========================================================

    model = load_model()

    # ========================================================
    # FULL-VOLUME INFERENCE
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "RUNNING INFERENCE ON VALIDATION CASES"
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

        print(
            "Probability shape:",
            tuple(
                probability.shape
            )
        )

        print(
            "Ground truth shape:",
            tuple(
                ground_truth.shape
            )
        )

    # ========================================================
    # THRESHOLD SEARCH
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "FINE THRESHOLD RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        "\nThreshold | Precision | Recall | Dice | IoU | FP"
    )

    print(
        "-" * 70
    )

    results = []

    for threshold in THRESHOLDS:

        total_tp = 0

        total_fp = 0

        total_fn = 0

        # ----------------------------------------------------
        # Each validation case
        # ----------------------------------------------------

        for case in case_data:

            probability = (
                case["probability"]
            )

            ground_truth = (
                case["ground_truth"]
                > 0
            )

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
        # Aggregate metrics
        # ----------------------------------------------------

        precision = (
            total_tp
            /
            (total_tp + total_fp)
            if (
                total_tp + total_fp
            ) > 0
            else 0.0
        )

        recall = (
            total_tp
            /
            (total_tp + total_fn)
            if (
                total_tp + total_fn
            ) > 0
            else 0.0
        )

        dice_denominator = (
            2 * total_tp
            +
            total_fp
            +
            total_fn
        )

        dice = (
            2 * total_tp
            /
            dice_denominator
            if dice_denominator > 0
            else 0.0
        )

        iou_denominator = (
            total_tp
            +
            total_fp
            +
            total_fn
        )

        iou = (
            total_tp
            /
            iou_denominator
            if iou_denominator > 0
            else 0.0
        )

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

        print(
            f"{threshold:9.2f} | "
            f"{precision:9.6f} | "
            f"{recall:6.6f} | "
            f"{dice:6.6f} | "
            f"{iou:6.6f} | "
            f"{total_fp}"
        )

    # ========================================================
    # FIND BEST DICE
    # ========================================================

    best_dice = max(
        results,
        key=lambda x: x["dice"]
    )

    # ========================================================
    # FIND BEST PRECISION
    # ========================================================

    best_precision = max(
        results,
        key=lambda x: x["precision"]
    )

    # ========================================================
    # PRECISION >= 0.90
    # ========================================================

    precision_90 = [
        result
        for result in results
        if result["precision"] >= 0.90
    ]

    if precision_90:

        best_precision_90 = max(
            precision_90,
            key=lambda x: x["dice"]
        )

    else:

        best_precision_90 = None

    # ========================================================
    # RECALL >= 0.90
    # ========================================================

    recall_90 = [
        result
        for result in results
        if result["recall"] >= 0.90
    ]

    if recall_90:

        best_recall_90 = max(
            recall_90,
            key=lambda x: x["dice"]
        )

    else:

        best_recall_90 = None

    # ========================================================
    # LOWEST FP WITH RECALL >= 0.85
    # ========================================================

    recall_85 = [
        result
        for result in results
        if result["recall"] >= 0.85
    ]

    if recall_85:

        lowest_fp = min(
            recall_85,
            key=lambda x: x["fp"]
        )

    else:

        lowest_fp = None

    # ========================================================
    # PRINT RECOMMENDATIONS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "FINE THRESHOLD RECOMMENDATIONS"
    )

    print(
        "=" * 70
    )

    print(
        "\nBEST DICE"
    )

    print(
        f"Threshold : "
        f"{best_dice['threshold']:.2f}"
    )

    print(
        f"Precision : "
        f"{best_dice['precision']:.6f}"
    )

    print(
        f"Recall    : "
        f"{best_dice['recall']:.6f}"
    )

    print(
        f"Dice      : "
        f"{best_dice['dice']:.6f}"
    )

    print(
        f"IoU       : "
        f"{best_dice['iou']:.6f}"
    )

    print(
        f"FP        : "
        f"{best_dice['fp']}"
    )

    # --------------------------------------------------------

    print(
        "\nBEST PRECISION"
    )

    print(
        f"Threshold : "
        f"{best_precision['threshold']:.2f}"
    )

    print(
        f"Precision : "
        f"{best_precision['precision']:.6f}"
    )

    print(
        f"Recall    : "
        f"{best_precision['recall']:.6f}"
    )

    print(
        f"Dice      : "
        f"{best_precision['dice']:.6f}"
    )

    print(
        f"FP        : "
        f"{best_precision['fp']}"
    )

    # --------------------------------------------------------

    if best_precision_90 is not None:

        print(
            "\nBEST DICE WITH PRECISION >= 0.90"
        )

        print(
            f"Threshold : "
            f"{best_precision_90['threshold']:.2f}"
        )

        print(
            f"Precision : "
            f"{best_precision_90['precision']:.6f}"
        )

        print(
            f"Recall    : "
            f"{best_precision_90['recall']:.6f}"
        )

        print(
            f"Dice      : "
            f"{best_precision_90['dice']:.6f}"
        )

        print(
            f"FP        : "
            f"{best_precision_90['fp']}"
        )

    # --------------------------------------------------------

    if best_recall_90 is not None:

        print(
            "\nBEST DICE WITH RECALL >= 0.90"
        )

        print(
            f"Threshold : "
            f"{best_recall_90['threshold']:.2f}"
        )

        print(
            f"Precision : "
            f"{best_recall_90['precision']:.6f}"
        )

        print(
            f"Recall    : "
            f"{best_recall_90['recall']:.6f}"
        )

        print(
            f"Dice      : "
            f"{best_recall_90['dice']:.6f}"
        )

        print(
            f"FP        : "
            f"{best_recall_90['fp']}"
        )

    # --------------------------------------------------------

    if lowest_fp is not None:

        print(
            "\nLOWEST FP WITH RECALL >= 0.85"
        )

        print(
            f"Threshold : "
            f"{lowest_fp['threshold']:.2f}"
        )

        print(
            f"Precision : "
            f"{lowest_fp['precision']:.6f}"
        )

        print(
            f"Recall    : "
            f"{lowest_fp['recall']:.6f}"
        )

        print(
            f"Dice      : "
            f"{lowest_fp['dice']:.6f}"
        )

        print(
            f"FP        : "
            f"{lowest_fp['fp']}"
        )

    # ========================================================
    # SAVE REPORT
    # ========================================================

    report_path = (
        OUTPUT_DIR
        /
        "fine_threshold_results.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "GLIOMA FINE THRESHOLD ANALYSIS\n"
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
            f"ROI: "
            f"{ROI_SIZE}\n"
        )

        file.write(
            f"Overlap: "
            f"{OVERLAP}\n\n"
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
            "\n\nBEST DICE\n"
        )

        file.write(
            str(best_dice)
            +
            "\n"
        )

        file.write(
            "\nBEST PRECISION\n"
        )

        file.write(
            str(best_precision)
            +
            "\n"
        )

        if best_precision_90 is not None:

            file.write(
                "\nBEST DICE WITH PRECISION >= 0.90\n"
            )

            file.write(
                str(best_precision_90)
                +
                "\n"
            )

        if best_recall_90 is not None:

            file.write(
                "\nBEST DICE WITH RECALL >= 0.90\n"
            )

            file.write(
                str(best_recall_90)
                +
                "\n"
            )

        if lowest_fp is not None:

            file.write(
                "\nLOWEST FP WITH RECALL >= 0.85\n"
            )

            file.write(
                str(lowest_fp)
                +
                "\n"
            )

    # ========================================================
    # FINISH
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "FINE THRESHOLD ANALYSIS COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nReport saved:"
    )

    print(
        report_path
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
    
