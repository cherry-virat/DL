"""
Glioma 3D U-Net Full-Volume Inference
=====================================

Purpose
-------
Run the trained medium Glioma 3D U-Net on complete validation
volumes using MONAI sliding-window inference.

IMPORTANT
---------
Training uses:

    glioma_preprocess.process_case()

Therefore full-volume inference uses the EXACT same preprocessing.

Pipeline
--------
BraTS NIfTI
    |
    v
process_case()
    |
    v
4-channel preprocessed MRI
    |
    v
Pad if necessary
    |
    v
MONAI sliding-window inference
    |
    v
4-class prediction
    |
    v
Crop padding
    |
    v
CPU metric calculation
    |
    v
Precision / Recall / Dice / IoU
"""


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import sys

# Allow `python segmentation\\glioma_full_volume_inference.py` from the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference

from preprocessing.glioma_dataset import (
    DATASET_ROOT,
    PATCH_SIZE,
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

# Same experiment split used during training/evaluation.
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

# Sliding-window overlap.
OVERLAP = 0.5

# Keep this at 1 for the RTX 3050 6 GB.
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
    / "glioma_full_volume_inference"
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
# REPRODUCIBILITY
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
        "\n" + "=" * 70
    )

    print(
        "LOADING TRAINED MEDIUM MODEL"
    )

    print(
        "=" * 70
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
        "Checkpoint loaded successfully."
    )

    print(
        "Checkpoint:",
        CHECKPOINT_PATH,
    )

    if "epoch" in checkpoint:

        print(
            "Checkpoint epoch:",
            checkpoint["epoch"],
        )

    if "validation_dice" in checkpoint:

        history = checkpoint[
            "validation_dice"
        ]

        if len(history) > 0:

            print(
                "Best validation Dice:",
                f"{max(history):.6f}",
            )

    return model


# ============================================================
# TUMOR METRICS
# ============================================================

def calculate_tumor_metrics(
    prediction,
    ground_truth,
):
    """
    Calculate binary tumor metrics.

    Tumor = labels 1, 2, 3
    Background = label 0

    IMPORTANT
    ---------
    Metric tensors are explicitly moved to CPU.

    This prevents:

        CUDA tensor
        +
        CPU tensor

    device mismatch errors.
    """

    # --------------------------------------------------------
    # CRITICAL DEVICE FIX
    # --------------------------------------------------------

    prediction = prediction.detach().cpu()

    ground_truth = ground_truth.detach().cpu()

    # --------------------------------------------------------
    # Validate shapes
    # --------------------------------------------------------

    if prediction.shape != ground_truth.shape:

        raise RuntimeError(
            "Prediction and ground truth shapes differ "
            "inside metric calculation.\n"
            f"Prediction: {prediction.shape}\n"
            f"Ground truth: {ground_truth.shape}"
        )

    # --------------------------------------------------------
    # Tumor masks
    # --------------------------------------------------------

    prediction_tumor = (
        prediction > 0
    )

    ground_truth_tumor = (
        ground_truth > 0
    )

    # --------------------------------------------------------
    # True positive
    # --------------------------------------------------------

    true_positive = (
        prediction_tumor
        &
        ground_truth_tumor
    ).sum().item()

    # --------------------------------------------------------
    # False positive
    # --------------------------------------------------------

    false_positive = (
        prediction_tumor
        &
        ~ground_truth_tumor
    ).sum().item()

    # --------------------------------------------------------
    # False negative
    # --------------------------------------------------------

    false_negative = (
        ~prediction_tumor
        &
        ground_truth_tumor
    ).sum().item()

    true_positive = int(
        true_positive
    )

    false_positive = int(
        false_positive
    )

    false_negative = int(
        false_negative
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
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "dice": dice,
        "iou": iou,
    }


# ============================================================
# PER-CLASS DICE
# ============================================================

def calculate_per_class_dice(
    prediction,
    ground_truth,
):
    """
    Calculate Dice independently for classes 0,1,2,3.
    """

    # --------------------------------------------------------
    # CRITICAL DEVICE FIX
    # --------------------------------------------------------

    prediction = prediction.detach().cpu()

    ground_truth = ground_truth.detach().cpu()

    # --------------------------------------------------------
    # Validate shape
    # --------------------------------------------------------

    if prediction.shape != ground_truth.shape:

        raise RuntimeError(
            "Prediction and ground truth shapes differ.\n"
            f"Prediction: {prediction.shape}\n"
            f"Ground truth: {ground_truth.shape}"
        )

    results = {}

    for class_id in range(
        OUTPUT_CLASSES
    ):

        prediction_class = (
            prediction == class_id
        )

        ground_truth_class = (
            ground_truth == class_id
        )

        intersection = (
            prediction_class
            &
            ground_truth_class
        ).sum().item()

        prediction_count = (
            prediction_class.sum().item()
        )

        ground_truth_count = (
            ground_truth_class.sum().item()
        )

        denominator = (
            prediction_count
            +
            ground_truth_count
        )

        if denominator > 0:

            dice = (
                2.0
                *
                intersection
                /
                denominator
            )

        else:

            dice = 1.0

        results[class_id] = dice

    return results


# ============================================================
# PROCESS ONE CASE
# ============================================================

def process_one_case(
    model,
    case_dir,
    case_number,
):

    print(
        "\n" + "-" * 70
    )

    print(
        f"FULL-VOLUME CASE {case_number}"
    )

    print(
        "-" * 70
    )

    print(
        "Case:",
        case_dir.name
    )

    # ========================================================
    # EXACT TRAINING PREPROCESSING
    # ========================================================

    image, mask = process_case(
        case_dir
    )

    # ========================================================
    # CONVERT TO NUMPY
    # ========================================================

    image = np.asarray(
        image,
        dtype=np.float32,
    )

    mask = np.asarray(
        mask,
        dtype=np.int64,
    )

    print(
        "\nPreprocessed volume:"
    )

    print(
        "Image:",
        image.shape
    )

    print(
        "Mask:",
        mask.shape
    )

    # ========================================================
    # IMAGE VALIDATION
    # ========================================================

    if image.ndim != 4:

        raise RuntimeError(
            "Expected image shape "
            "(C,D,H,W).\n"
            f"Received: {image.shape}"
        )

    if image.shape[0] != INPUT_CHANNELS:

        raise RuntimeError(
            "Invalid number of MRI channels.\n"
            f"Expected: {INPUT_CHANNELS}\n"
            f"Received: {image.shape[0]}"
        )

    # ========================================================
    # MASK VALIDATION
    # ========================================================

    if mask.ndim != 3:

        raise RuntimeError(
            "Expected mask shape "
            "(D,H,W).\n"
            f"Received: {mask.shape}"
        )

    # ========================================================
    # IMAGE / MASK ALIGNMENT
    # ========================================================

    if image.shape[1:] != mask.shape:

        raise RuntimeError(
            "\nIMAGE/MASK MISMATCH\n"
            f"Image: {image.shape}\n"
            f"Mask : {mask.shape}"
        )

    # ========================================================
    # ORIGINAL PREPROCESSED SHAPE
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
    # PAD SMALL VOLUME
    # ========================================================

    padded_image, padded_mask = (
        pad_to_patch_size(
            image,
            mask_4d,
            PATCH_SIZE,
        )
    )

    print(
        "\nInference volume after padding:"
    )

    print(
        "Image:",
        padded_image.shape
    )

    print(
        "Mask:",
        padded_mask.shape
    )

    # ========================================================
    # CONVERT TO TORCH
    # ========================================================

    image_tensor = torch.from_numpy(
        padded_image
    ).float()

    mask_tensor = torch.from_numpy(
        padded_mask
    ).long()

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
    # GROUND TRUTH
    #
    # Keep it on CPU.
    #
    # Metrics are calculated on CPU.
    # ========================================================

    ground_truth = (
        mask_tensor
        .squeeze(0)
        .cpu()
    )

    print(
        "\nInput tensor:",
        tuple(input_tensor.shape)
    )

    print(
        "Input device:",
        input_tensor.device
    )

    print(
        "Ground truth device:",
        ground_truth.device
    )

    # ========================================================
    # SLIDING WINDOW INFERENCE
    # ========================================================

    print(
        "\nRunning sliding-window inference..."
    )

    print(
        "ROI:",
        ROI_SIZE
    )

    print(
        "Overlap:",
        OVERLAP
    )

    print(
        "Window batch size:",
        SW_BATCH_SIZE
    )

    with torch.inference_mode():

        output = sliding_window_inference(

            inputs=input_tensor,

            roi_size=ROI_SIZE,

            sw_batch_size=SW_BATCH_SIZE,

            predictor=model,

            overlap=OVERLAP,

            mode="gaussian",
        )

    print(
        "Sliding-window inference completed."
    )

    print(
        "Raw output:",
        tuple(output.shape)
    )

    # ========================================================
    # ARGMAX
    # ========================================================

    prediction = torch.argmax(
        output,
        dim=1,
    )

    # Remove batch dimension.
    prediction = prediction[
        0
    ]

    print(
        "Prediction before cropping:",
        tuple(prediction.shape)
    )

    # ========================================================
    # MOVE PREDICTION TO CPU
    # ========================================================

    prediction = (
        prediction
        .detach()
        .cpu()
    )

    print(
        "Prediction device:",
        prediction.device
    )

    # ========================================================
    # REMOVE PADDING
    # ========================================================

    d, h, w = original_shape

    prediction = prediction[
        :d,
        :h,
        :w,
    ]

    ground_truth = ground_truth[
        :d,
        :h,
        :w,
    ]

    print(
        "\nPrediction after removing padding:",
        tuple(prediction.shape)
    )

    print(
        "Ground truth:",
        tuple(ground_truth.shape)
    )

    # ========================================================
    # FINAL SHAPE CHECK
    # ========================================================

    if tuple(
        prediction.shape
    ) != tuple(
        ground_truth.shape
    ):

        raise RuntimeError(
            "\nFINAL PREDICTION/GROUND-TRUTH "
            "SHAPE MISMATCH\n"
            f"Prediction: {tuple(prediction.shape)}\n"
            f"Ground truth: {tuple(ground_truth.shape)}"
        )

    print(
        "[OK] Prediction and ground-truth shapes match."
    )

    # ========================================================
    # LABEL VALIDATION
    # ========================================================

    prediction_labels = torch.unique(
        prediction
    ).tolist()

    ground_truth_labels = torch.unique(
        ground_truth
    ).tolist()

    print(
        "\nPrediction labels:",
        prediction_labels
    )

    print(
        "Ground-truth labels:",
        ground_truth_labels
    )

    # ========================================================
    # CHECK INVALID PREDICTIONS
    # ========================================================

    if any(
        label < 0
        or label >= OUTPUT_CLASSES
        for label in prediction_labels
    ):

        raise RuntimeError(
            "Invalid prediction label detected:\n"
            f"{prediction_labels}"
        )

    # ========================================================
    # METRICS
    # ========================================================

    metrics = calculate_tumor_metrics(
        prediction,
        ground_truth,
    )

    per_class_dice = (
        calculate_per_class_dice(
            prediction,
            ground_truth,
        )
    )

    # ========================================================
    # VOXEL STATISTICS
    # ========================================================

    ground_truth_tumor = int(
        (
            ground_truth > 0
        )
        .sum()
        .item()
    )

    predicted_tumor = int(
        (
            prediction > 0
        )
        .sum()
        .item()
    )

    # ========================================================
    # PREDICTION / GT RATIO
    # ========================================================

    if ground_truth_tumor > 0:

        volume_ratio = (
            predicted_tumor
            /
            ground_truth_tumor
        )

    else:

        volume_ratio = 0.0

    # ========================================================
    # CASE RESULTS
    # ========================================================

    print(
        "\nCase results:"
    )

    print(
        "Ground-truth tumor voxels:",
        ground_truth_tumor
    )

    print(
        "Predicted tumor voxels:",
        predicted_tumor
    )

    print(
        "Prediction / GT volume:",
        f"{volume_ratio:.4f}x"
    )

    print(
        "True positives:",
        metrics["true_positive"]
    )

    print(
        "False positives:",
        metrics["false_positive"]
    )

    print(
        "False negatives:",
        metrics["false_negative"]
    )

    print(
        "Precision:",
        f"{metrics['precision']:.6f}"
    )

    print(
        "Recall:",
        f"{metrics['recall']:.6f}"
    )

    print(
        "Dice:",
        f"{metrics['dice']:.6f}"
    )

    print(
        "IoU:",
        f"{metrics['iou']:.6f}"
    )

    # ========================================================
    # PER-CLASS DICE
    # ========================================================

    print(
        "\nPer-class Dice:"
    )

    for class_id, dice in (
        per_class_dice.items()
    ):

        print(
            f"  Class {class_id}: "
            f"{dice:.6f}"
        )

    # ========================================================
    # SAVE NUMPY PREDICTION
    # ========================================================

    prediction_path = (
        OUTPUT_DIR
        /
        f"{case_dir.name}_prediction.npy"
    )

    np.save(
        prediction_path,
        prediction.numpy().astype(
            np.uint8
        ),
    )

    print(
        "\nPrediction saved:"
    )

    print(
        prediction_path
    )

    # ========================================================
    # FREE GPU OUTPUT
    # ========================================================

    del output
    del input_tensor

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    # ========================================================
    # RETURN
    # ========================================================

    return {
        "case": case_dir.name,

        "ground_truth":
            ground_truth_tumor,

        "predicted":
            predicted_tumor,

        "volume_ratio":
            volume_ratio,

        "true_positive":
            metrics["true_positive"],

        "false_positive":
            metrics["false_positive"],

        "false_negative":
            metrics["false_negative"],

        "precision":
            metrics["precision"],

        "recall":
            metrics["recall"],

        "dice":
            metrics["dice"],

        "iou":
            metrics["iou"],

        "per_class_dice":
            per_class_dice,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "GLIOMA 3D U-NET FULL-VOLUME INFERENCE"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # ENVIRONMENT
    # ========================================================

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

    else:

        print(
            "GPU: CPU"
        )

    # ========================================================
    # CHECKPOINT
    # ========================================================

    print(
        "\nCheckpoint:",
        CHECKPOINT_PATH
    )

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            "\nMedium model checkpoint does not exist:\n"
            f"{CHECKPOINT_PATH}"
        )

    # ========================================================
    # GET CASES
    # ========================================================

    cases = get_cases(
        DATASET_ROOT
    )

    print(
        "\nTotal cases:",
        len(cases)
    )

    # ========================================================
    # CREATE SAME SPLIT
    # ========================================================

    train_cases, validation_cases = (
        create_train_val_split(
            cases,
            train_ratio=0.90,
            seed=SEED,
        )
    )

    # ========================================================
    # SAME SMALL EXPERIMENT
    # ========================================================

    train_cases = train_cases[
        :TRAIN_CASES_FOR_TEST
    ]

    validation_cases = validation_cases[
        :VAL_CASES_FOR_TEST
    ]

    # ========================================================
    # VERIFY NO OVERLAP
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
            "Training/validation case overlap detected:\n"
            f"{overlap}"
        )

    print(
        "[OK] No training/validation overlap."
    )

    # ========================================================
    # PRINT VALIDATION CASES
    # ========================================================

    print(
        "\nValidation cases:"
    )

    for index, case in enumerate(
        validation_cases,
        start=1,
    ):

        print(
            f"  {index}. {case.name}"
        )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    model = load_model()

    # ========================================================
    # START
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "STARTING FULL-VOLUME INFERENCE"
    )

    print(
        "=" * 70
    )

    results = []

    # ========================================================
    # PROCESS CASES
    # ========================================================

    for case_number, case_dir in enumerate(
        validation_cases,
        start=1,
    ):

        result = process_one_case(
            model=model,
            case_dir=case_dir,
            case_number=case_number,
        )

        results.append(
            result
        )

    # ========================================================
    # AGGREGATE METRICS
    # ========================================================

    total_tp = sum(
        result["true_positive"]
        for result in results
    )

    total_fp = sum(
        result["false_positive"]
        for result in results
    )

    total_fn = sum(
        result["false_negative"]
        for result in results
    )

    total_gt = sum(
        result["ground_truth"]
        for result in results
    )

    total_predicted = sum(
        result["predicted"]
        for result in results
    )

    # ========================================================
    # GLOBAL PRECISION
    # ========================================================

    precision_denominator = (
        total_tp
        +
        total_fp
    )

    if precision_denominator > 0:

        global_precision = (
            total_tp
            /
            precision_denominator
        )

    else:

        global_precision = 0.0

    # ========================================================
    # GLOBAL RECALL
    # ========================================================

    recall_denominator = (
        total_tp
        +
        total_fn
    )

    if recall_denominator > 0:

        global_recall = (
            total_tp
            /
            recall_denominator
        )

    else:

        global_recall = 0.0

    # ========================================================
    # GLOBAL DICE
    # ========================================================

    dice_denominator = (
        2 * total_tp
        +
        total_fp
        +
        total_fn
    )

    if dice_denominator > 0:

        global_dice = (
            2
            *
            total_tp
            /
            dice_denominator
        )

    else:

        global_dice = 0.0

    # ========================================================
    # GLOBAL IOU
    # ========================================================

    iou_denominator = (
        total_tp
        +
        total_fp
        +
        total_fn
    )

    if iou_denominator > 0:

        global_iou = (
            total_tp
            /
            iou_denominator
        )

    else:

        global_iou = 0.0

    # ========================================================
    # VOLUME RATIO
    # ========================================================

    if total_gt > 0:

        global_volume_ratio = (
            total_predicted
            /
            total_gt
        )

    else:

        global_volume_ratio = 0.0

    # ========================================================
    # RESULTS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "AGGREGATE FULL-VOLUME RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        "Cases processed:",
        len(results)
    )

    print(
        "\nTotal ground-truth tumor voxels:",
        total_gt
    )

    print(
        "Total predicted tumor voxels:",
        total_predicted
    )

    print(
        "Prediction / GT volume:",
        f"{global_volume_ratio:.4f}x"
    )

    print(
        "\nTotal true positives:",
        total_tp
    )

    print(
        "Total false positives:",
        total_fp
    )

    print(
        "Total false negatives:",
        total_fn
    )

    print(
        "\nGlobal precision:",
        f"{global_precision:.6f}"
    )

    print(
        "Global recall:",
        f"{global_recall:.6f}"
    )

    print(
        "Global Dice:",
        f"{global_dice:.6f}"
    )

    print(
        "Global IoU:",
        f"{global_iou:.6f}"
    )

    # ========================================================
    # SAVE REPORT
    # ========================================================

    report_path = (
        OUTPUT_DIR
        /
        "full_volume_inference_results.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "GLIOMA FULL-VOLUME INFERENCE RESULTS\n"
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

        # ----------------------------------------------------
        # Individual cases
        # ----------------------------------------------------

        for result in results:

            file.write(
                "-" * 70
                +
                "\n"
            )

            file.write(
                f"Case: "
                f"{result['case']}\n"
            )

            file.write(
                f"Ground truth tumor voxels: "
                f"{result['ground_truth']}\n"
            )

            file.write(
                f"Predicted tumor voxels: "
                f"{result['predicted']}\n"
            )

            file.write(
                f"Prediction / GT volume: "
                f"{result['volume_ratio']:.4f}x\n"
            )

            file.write(
                f"True positive: "
                f"{result['true_positive']}\n"
            )

            file.write(
                f"False positive: "
                f"{result['false_positive']}\n"
            )

            file.write(
                f"False negative: "
                f"{result['false_negative']}\n"
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
                "\nPer-class Dice:\n"
            )

            for class_id, dice in (
                result[
                    "per_class_dice"
                ].items()
            ):

                file.write(
                    f"  Class {class_id}: "
                    f"{dice:.6f}\n"
                )

            file.write(
                "\n"
            )

        # ----------------------------------------------------
        # Aggregate
        # ----------------------------------------------------

        file.write(
            "\n"
            +
            "=" * 70
            +
            "\n"
        )

        file.write(
            "AGGREGATE RESULTS\n"
        )

        file.write(
            "=" * 70
            +
            "\n"
        )

        file.write(
            f"Cases processed: "
            f"{len(results)}\n"
        )

        file.write(
            f"Ground truth tumor voxels: "
            f"{total_gt}\n"
        )

        file.write(
            f"Predicted tumor voxels: "
            f"{total_predicted}\n"
        )

        file.write(
            f"Prediction / GT volume: "
            f"{global_volume_ratio:.4f}x\n"
        )

        file.write(
            f"True positives: "
            f"{total_tp}\n"
        )

        file.write(
            f"False positives: "
            f"{total_fp}\n"
        )

        file.write(
            f"False negatives: "
            f"{total_fn}\n"
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

    print(
        "\nReport saved:"
    )

    print(
        report_path
    )

    # ========================================================
    # GPU MEMORY
    # ========================================================

    if torch.cuda.is_available():

        print(
            "\nGPU memory:"
        )

        print(
            "Allocated:",
            f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB"
        )

        print(
            "Reserved:",
            f"{torch.cuda.memory_reserved() / 1024**2:.2f} MB"
        )

        print(
            "Peak:",
            f"{torch.cuda.max_memory_allocated() / 1024**2:.2f} MB"
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "FULL-VOLUME INFERENCE COMPLETE"
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
