"""
Glioma 3D U-Net Final Tumor Inference
=====================================

Purpose
-------
Run the trained medium Glioma 3D U-Net on validation cases
using the experimentally selected post-processing configuration.

Selected configuration
----------------------
Tumor probability threshold:
    0.60

Minimum connected-component size:
    1000 voxels

Model
-----
4 MRI channels
4 output classes
3D U-Net

Pipeline
--------
MRI
 ↓
GliomaDataset preprocessing
 ↓
64x64x64 patches
 ↓
3D U-Net
 ↓
Softmax probabilities
 ↓
Whole-tumor probability
    P(class 1) + P(class 2) + P(class 3)
 ↓
Threshold 0.60
 ↓
3D connected-component filtering
 ↓
Remove components < 1000 voxels
 ↓
Final tumor mask
 ↓
Metrics

This file does NOT retrain the model.
"""


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from scipy import ndimage

from torch.utils.data import DataLoader

from monai.networks.nets import UNet

from preprocessing.glioma_dataset import (
    GliomaDataset,
    DATASET_ROOT,
    get_cases,
)

from preprocessing.glioma_split import (
    create_train_val_split,
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

VAL_CASES = 20

BATCH_SIZE = 1

INPUT_CHANNELS = 4

OUTPUT_CLASSES = 4

MODEL_CHANNELS = (
    16,
    32,
    64,
    128,
    256,
)

DROPOUT = 0.0

PATCH_SIZE = (
    64,
    64,
    64,
)

# ------------------------------------------------------------
# Selected experimentally
# ------------------------------------------------------------

TUMOR_THRESHOLD = 0.60

MIN_COMPONENT_SIZE = 1000


# ============================================================
# CHECKPOINT
# ============================================================

CHECKPOINT = (
    Path("outputs")
    / "glioma_training"
    / "glioma_best_medium_experiment.pth"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    Path("outputs")
    / "glioma_final_inference"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


REPORT_PATH = (
    OUTPUT_DIR
    /
    "final_inference_report.txt"
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

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)


# ============================================================
# COLLATE FUNCTION
# ============================================================

def glioma_collate_fn(batch):
    """
    Convert:

        [
            [
                patch1,
                patch2
            ]
        ]

    into:

        {
            "image": (N,4,64,64,64),
            "label": (N,1,64,64,64)
        }
    """

    patches = []

    for case_output in batch:

        if not isinstance(
            case_output,
            list,
        ):

            raise TypeError(
                "Expected GliomaDataset to return "
                "a list of patch dictionaries."
            )

        patches.extend(
            case_output
        )


    if len(patches) == 0:

        raise RuntimeError(
            "Dataset returned zero patches."
        )


    images = torch.stack(
        [
            patch["image"]
            for patch in patches
        ],
        dim=0,
    )


    labels = torch.stack(
        [
            patch["label"]
            for patch in patches
        ],
        dim=0,
    )


    return {
        "image": images,
        "label": labels,
    }


# ============================================================
# MODEL
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


    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )


    model.load_state_dict(
        checkpoint["model_state_dict"]
    )


    model = model.to(
        DEVICE
    )


    model.eval()


    return model, checkpoint


# ============================================================
# CONNECTED COMPONENT FILTER
# ============================================================

def remove_small_components(
    binary_mask,
    minimum_size,
):
    """
    Remove 3D connected components smaller than
    minimum_size voxels.

    Uses 26-connectivity.
    """

    binary_mask = np.asarray(
        binary_mask,
        dtype=bool,
    )


    if minimum_size <= 0:

        return binary_mask.copy()


    structure = np.ones(
        (
            3,
            3,
            3,
        ),
        dtype=np.uint8,
    )


    labeled_mask, number_of_components = (
        ndimage.label(
            binary_mask,
            structure=structure,
        )
    )


    if number_of_components == 0:

        return np.zeros_like(
            binary_mask,
            dtype=bool,
        )


    component_sizes = np.bincount(
        labeled_mask.ravel()
    )


    keep = (
        component_sizes
        >= minimum_size
    )


    # Background must never be retained.
    keep[0] = False


    filtered_mask = keep[
        labeled_mask
    ]


    return filtered_mask.astype(
        bool
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    prediction,
    ground_truth,
):

    prediction = np.asarray(
        prediction,
        dtype=bool,
    )

    ground_truth = np.asarray(
        ground_truth,
        dtype=bool,
    )


    if prediction.shape != ground_truth.shape:

        raise RuntimeError(
            "Prediction/ground-truth shape mismatch.\n"
            f"Prediction: {prediction.shape}\n"
            f"Ground truth: {ground_truth.shape}"
        )


    tp = np.logical_and(
        prediction,
        ground_truth,
    ).sum()


    fp = np.logical_and(
        prediction,
        np.logical_not(
            ground_truth
        ),
    ).sum()


    fn = np.logical_and(
        np.logical_not(
            prediction
        ),
        ground_truth,
    ).sum()


    precision = (
        tp
        /
        max(
            tp + fp,
            1,
        )
    )


    recall = (
        tp
        /
        max(
            tp + fn,
            1,
        )
    )


    dice = (
        2 * tp
        /
        max(
            2 * tp
            + fp
            + fn,
            1,
        )
    )


    iou = (
        tp
        /
        max(
            tp
            + fp
            + fn,
            1,
        )
    )


    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "dice": float(dice),
        "iou": float(iou),
    }


# ============================================================
# PROCESS ONE CASE
# ============================================================

def process_case(
    model,
    case,
):

    print(
        "\n" + "-" * 70
    )

    print(
        "CASE:",
        case.name,
    )

    print(
        "-" * 70
    )


    # --------------------------------------------------------
    # Create a one-case dataset
    # --------------------------------------------------------

    dataset = GliomaDataset(
        [case]
    )


    loader = DataLoader(
        dataset,

        batch_size=1,

        shuffle=False,

        num_workers=0,

        pin_memory=torch.cuda.is_available(),

        collate_fn=glioma_collate_fn,
    )


    # --------------------------------------------------------
    # Patch-level prediction collection
    # --------------------------------------------------------

    all_predictions = []

    all_ground_truth = []


    total_tp = 0

    total_fp = 0

    total_fn = 0


    with torch.no_grad():

        for batch_index, batch in enumerate(
            loader,
            start=1,
        ):

            images = batch[
                "image"
            ]

            labels = batch[
                "label"
            ]


            # ------------------------------------------------
            # Shape checks BEFORE GPU
            # ------------------------------------------------

            if images.ndim != 5:

                raise RuntimeError(
                    f"Invalid image shape: "
                    f"{images.shape}"
                )


            if labels.ndim != 5:

                raise RuntimeError(
                    f"Invalid label shape: "
                    f"{labels.shape}"
                )


            # ------------------------------------------------
            # GPU
            # ------------------------------------------------

            images = images.to(
                DEVICE,
                non_blocking=True,
            )


            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            outputs = model(
                images
            )


            if tuple(
                outputs.shape[1:]
            ) != (
                OUTPUT_CLASSES,
                *PATCH_SIZE,
            ):

                raise RuntimeError(
                    "Unexpected model output shape.\n"
                    f"Output: {outputs.shape}"
                )


            # ------------------------------------------------
            # Softmax
            # ------------------------------------------------

            probabilities = torch.softmax(
                outputs,
                dim=1,
            )


            # ------------------------------------------------
            # Whole-tumor probability
            #
            # Class 0 = background
            # Classes 1,2,3 = tumor
            #
            # Therefore:
            #
            # P(tumor) =
            # P(1)+P(2)+P(3)
            # ------------------------------------------------

            tumor_probability = (
                probabilities[:, 1:]
                .sum(
                    dim=1
                )
            )


            # ------------------------------------------------
            # Threshold
            # ------------------------------------------------

            raw_prediction = (
                tumor_probability
                >= TUMOR_THRESHOLD
            )


            raw_prediction = (
                raw_prediction
                .cpu()
                .numpy()
            )


            ground_truth = (
                labels
                .squeeze(1)
                .cpu()
                .numpy()
                > 0
            )


            # ------------------------------------------------
            # Connected components
            # ------------------------------------------------

            for patch_index in range(
                raw_prediction.shape[0]
            ):

                filtered_prediction = (
                    remove_small_components(
                        raw_prediction[
                            patch_index
                        ],
                        MIN_COMPONENT_SIZE,
                    )
                )


                gt = ground_truth[
                    patch_index
                ]


                metrics = calculate_metrics(
                    filtered_prediction,
                    gt,
                )


                total_tp += metrics[
                    "tp"
                ]

                total_fp += metrics[
                    "fp"
                ]

                total_fn += metrics[
                    "fn"
                ]


                all_predictions.append(
                    filtered_prediction
                )

                all_ground_truth.append(
                    gt
                )


            if batch_index == 1:

                print(
                    "\nFirst batch:"
                )

                print(
                    "Input:",
                    tuple(images.shape),
                )

                print(
                    "Output:",
                    tuple(outputs.shape),
                )

                print(
                    "Threshold:",
                    TUMOR_THRESHOLD,
                )

                print(
                    "Minimum component:",
                    MIN_COMPONENT_SIZE,
                )


    # --------------------------------------------------------
    # Aggregate metrics
    # --------------------------------------------------------

    precision = (
        total_tp
        /
        max(
            total_tp + total_fp,
            1,
        )
    )


    recall = (
        total_tp
        /
        max(
            total_tp + total_fn,
            1,
        )
    )


    dice = (
        2 * total_tp
        /
        max(
            2 * total_tp
            + total_fp
            + total_fn,
            1,
        )
    )


    iou = (
        total_tp
        /
        max(
            total_tp
            + total_fp
            + total_fn,
            1,
        )
    )


    print(
        "\nCase results:"
    )

    print(
        "True positives:",
        total_tp,
    )

    print(
        "False positives:",
        total_fp,
    )

    print(
        "False negatives:",
        total_fn,
    )

    print(
        "Precision:",
        f"{precision:.6f}",
    )

    print(
        "Recall:",
        f"{recall:.6f}",
    )

    print(
        "Dice:",
        f"{dice:.6f}",
    )

    print(
        "IoU:",
        f"{iou:.6f}",
    )


    return {
        "case": case.name,

        "tp": total_tp,

        "fp": total_fp,

        "fn": total_fn,

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
        "GLIOMA FINAL PRECISION-OPTIMIZED INFERENCE"
    )

    print(
        "=" * 70
    )


    print(
        "\nDataset:",
        DATASET_ROOT,
    )

    print(
        "Device:",
        DEVICE,
    )

    print(
        "Checkpoint:",
        CHECKPOINT,
    )


    print(
        "\nSelected post-processing:"
    )

    print(
        "Tumor probability threshold:",
        TUMOR_THRESHOLD,
    )

    print(
        "Minimum component size:",
        MIN_COMPONENT_SIZE,
        "voxels",
    )


    # ========================================================
    # CHECKPOINT
    # ========================================================

    if not CHECKPOINT.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT}"
        )


    # ========================================================
    # CASES
    # ========================================================

    cases = get_cases(
        DATASET_ROOT
    )


    print(
        "\nTotal cases:",
        len(cases),
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


    validation_cases = validation_cases[
        :VAL_CASES
    ]


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
            f"Training/validation overlap: "
            f"{overlap}"
        )


    print(
        "Validation cases:",
        len(validation_cases),
    )


    # ========================================================
    # MODEL
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "LOADING TRAINED MODEL"
    )

    print(
        "=" * 70
    )


    model, checkpoint = create_model()


    print(
        "[OK] Model loaded successfully."
    )


    print(
        "Checkpoint:",
        CHECKPOINT,
    )


    if "epoch" in checkpoint:

        print(
            "Checkpoint epoch:",
            checkpoint["epoch"],
        )


    # ========================================================
    # EVALUATION
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "RUNNING FINAL INFERENCE"
    )

    print(
        "=" * 70
    )


    results = []


    for index, case in enumerate(
        validation_cases,
        start=1,
    ):

        print(
            f"\nCASE {index}/{len(validation_cases)}"
        )


        result = process_case(
            model,
            case,
        )


        results.append(
            result
        )


    # ========================================================
    # GLOBAL METRICS
    # ========================================================

    total_tp = sum(
        item["tp"]
        for item in results
    )


    total_fp = sum(
        item["fp"]
        for item in results
    )


    total_fn = sum(
        item["fn"]
        for item in results
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
            2 * total_tp
            + total_fp
            + total_fn,
            1,
        )
    )


    global_iou = (
        total_tp
        /
        max(
            total_tp
            + total_fp
            + total_fn,
            1,
        )
    )


    # ========================================================
    # REPORT
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL INFERENCE RESULTS"
    )

    print(
        "=" * 70
    )


    report = []


    report.append(
        "GLIOMA FINAL PRECISION-OPTIMIZED INFERENCE"
    )

    report.append(
        "=" * 70
    )


    report.append(
        f"Probability threshold: "
        f"{TUMOR_THRESHOLD}"
    )

    report.append(
        f"Minimum component size: "
        f"{MIN_COMPONENT_SIZE}"
    )


    for result in results:

        print(
            f"\n{result['case']}"
        )

        print(
            f"  Precision: "
            f"{result['precision']:.6f}"
        )

        print(
            f"  Recall: "
            f"{result['recall']:.6f}"
        )

        print(
            f"  Dice: "
            f"{result['dice']:.6f}"
        )

        print(
            f"  IoU: "
            f"{result['iou']:.6f}"
        )

        print(
            f"  TP: "
            f"{result['tp']}"
        )

        print(
            f"  FP: "
            f"{result['fp']}"
        )

        print(
            f"  FN: "
            f"{result['fn']}"
        )


        report.append(
            f"\n{result['case']}"
        )

        report.append(
            f"Precision: "
            f"{result['precision']:.6f}"
        )

        report.append(
            f"Recall: "
            f"{result['recall']:.6f}"
        )

        report.append(
            f"Dice: "
            f"{result['dice']:.6f}"
        )

        report.append(
            f"IoU: "
            f"{result['iou']:.6f}"
        )

        report.append(
            f"TP: {result['tp']}"
        )

        report.append(
            f"FP: {result['fp']}"
        )

        report.append(
            f"FN: {result['fn']}"
        )


    # ========================================================
    # GLOBAL
    # ========================================================

    print(
        "\n" + "-" * 70
    )

    print(
        "GLOBAL METRICS"
    )

    print(
        "-" * 70
    )


    print(
        "True positives:",
        total_tp,
    )

    print(
        "False positives:",
        total_fp,
    )

    print(
        "False negatives:",
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


    report.append(
        "\nGLOBAL METRICS"
    )

    report.append(
        f"TP: {total_tp}"
    )

    report.append(
        f"FP: {total_fp}"
    )

    report.append(
        f"FN: {total_fn}"
    )

    report.append(
        f"Precision: {global_precision:.6f}"
    )

    report.append(
        f"Recall: {global_recall:.6f}"
    )

    report.append(
        f"Dice: {global_dice:.6f}"
    )

    report.append(
        f"IoU: {global_iou:.6f}"
    )


    # ========================================================
    # SAVE REPORT
    # ========================================================

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "\n".join(report)
        )


    print(
        "\nReport saved:"
    )

    print(
        REPORT_PATH
    )


    # ========================================================
    # GPU
    # ========================================================

    if torch.cuda.is_available():

        allocated = (
            torch.cuda.memory_allocated()
            /
            1024 ** 2
        )

        reserved = (
            torch.cuda.memory_reserved()
            /
            1024 ** 2
        )


        print(
            "\nGPU memory:"
        )

        print(
            "Allocated:",
            f"{allocated:.2f} MB",
        )

        print(
            "Reserved:",
            f"{reserved:.2f} MB",
        )


    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL INFERENCE COMPLETE"
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()