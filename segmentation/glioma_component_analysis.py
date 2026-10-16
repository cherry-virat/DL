"""
Glioma 3D Connected-Component False-Positive Analysis
=====================================================

Purpose
-------
Evaluate whether removing small isolated tumor components
reduces false-positive tumor voxels without damaging recall.

Pipeline
--------
MRI
  ↓
3D U-Net
  ↓
Tumor probability
  ↓
Probability threshold
  ↓
Binary tumor mask
  ↓
Connected components
  ↓
Remove components smaller than minimum size
  ↓
Final tumor mask
  ↓
Precision / Recall / Dice / IoU

Validation:
    20 validation cases

Probability thresholds:
    0.50
    0.55
    0.60

Minimum component sizes:
    0
    10
    25
    50
    100
    250
    500
    1000

Important
---------
This script evaluates the EXISTING trained medium model.

It does NOT retrain the model.
"""


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import random

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

CHECKPOINT = (
    Path("outputs")
    / "glioma_training"
    / "glioma_best_medium_experiment.pth"
)

OUTPUT_DIR = (
    Path("outputs")
    / "glioma_component_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_PATH = (
    OUTPUT_DIR
    /
    "component_analysis.txt"
)


# ============================================================
# EXPERIMENT VALUES
# ============================================================

PROBABILITY_THRESHOLDS = (
    0.50,
    0.55,
    0.60,
)

MIN_COMPONENT_SIZES = (
    0,
    10,
    25,
    50,
    100,
    250,
    500,
    1000,
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

    patches = []

    for case_output in batch:

        if not isinstance(
            case_output,
            list,
        ):

            raise TypeError(
                "Expected GliomaDataset output "
                "to be a list of patch dictionaries."
            )

        patches.extend(
            case_output
        )


    if not patches:

        raise RuntimeError(
            "No patches returned by dataset."
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


    model = model.to(
        DEVICE
    )


    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )


    model.load_state_dict(
        checkpoint["model_state_dict"]
    )


    model.eval()


    return model


# ============================================================
# CONNECTED COMPONENT FILTER
# ============================================================

def remove_small_components(
    mask,
    minimum_size,
):
    """
    Remove connected 3D tumor components smaller than
    minimum_size voxels.

    Parameters
    ----------
    mask:
        Boolean 3D NumPy array.

    minimum_size:
        Minimum number of connected voxels required
        to keep a component.

    Returns
    -------
    Boolean 3D NumPy array.
    """

    if minimum_size <= 0:

        return mask.copy()


    # --------------------------------------------------------
    # 26-connectivity
    #
    # This allows diagonal voxel connections.
    # --------------------------------------------------------

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
            mask,
            structure=structure,
        )
    )


    if number_of_components == 0:

        return np.zeros_like(
            mask,
            dtype=bool,
        )


    component_sizes = np.bincount(
        labeled_mask.ravel()
    )


    keep_components = (
        component_sizes
        >= minimum_size
    )


    keep_components[0] = False


    filtered_mask = keep_components[
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
            "Prediction / ground-truth shape mismatch.\n"
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
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "GLIOMA CONNECTED-COMPONENT ANALYSIS"
    )

    print(
        "=" * 70
    )


    print(
        "\nDataset:",
        DATASET_ROOT,
    )

    print(
        "Checkpoint:",
        CHECKPOINT,
    )

    print(
        "Device:",
        DEVICE,
    )


    if not CHECKPOINT.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT}"
        )


    # ========================================================
    # FIND CASES
    # ========================================================

    cases = get_cases(
        DATASET_ROOT
    )


    print(
        "\nTotal cases:",
        len(cases),
    )


    # ========================================================
    # CASE-LEVEL SPLIT
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


    if train_names & validation_names:

        raise RuntimeError(
            "Training/validation overlap detected."
        )


    print(
        "Validation cases:",
        len(validation_cases),
    )


    # ========================================================
    # DATASET
    # ========================================================

    dataset = GliomaDataset(
        validation_cases
    )


    loader = DataLoader(
        dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0,

        pin_memory=torch.cuda.is_available(),

        collate_fn=glioma_collate_fn,
    )


    # ========================================================
    # MODEL
    # ========================================================

    print(
        "\nLoading medium model..."
    )


    model = create_model()


    print(
        "✓ Model loaded successfully."
    )


    # ========================================================
    # STATISTICS
    # ========================================================

    results = {}


    for threshold in PROBABILITY_THRESHOLDS:

        results[threshold] = {}


        for minimum_size in MIN_COMPONENT_SIZES:

            results[
                threshold
            ][minimum_size] = {

                "tp": 0,

                "fp": 0,

                "fn": 0,
            }


    # ========================================================
    # INFERENCE
    # ========================================================

    print(
        "\nRunning connected-component experiment..."
    )


    with torch.no_grad():

        for batch_index, batch in enumerate(
            loader,
            start=1,
        ):

            images = batch["image"].to(
                DEVICE,
                non_blocking=True,
            )


            labels = batch["label"]


            outputs = model(
                images
            )


            probabilities = torch.softmax(
                outputs,
                dim=1,
            )


            # ------------------------------------------------
            # Whole tumor probability
            # ------------------------------------------------

            tumor_probability = (
                probabilities[:, 1:]
                .sum(
                    dim=1
                )
            )


            tumor_probability = (
                tumor_probability
                .detach()
                .cpu()
                .numpy()
            )


            ground_truth = (
                labels
                .squeeze(1)
                .numpy()
                > 0
            )


            # ------------------------------------------------
            # Process each patch
            # ------------------------------------------------

            for patch_index in range(
                tumor_probability.shape[0]
            ):

                probability = (
                    tumor_probability[
                        patch_index
                    ]
                )


                gt = (
                    ground_truth[
                        patch_index
                    ]
                )


                for threshold in (
                    PROBABILITY_THRESHOLDS
                ):

                    raw_prediction = (
                        probability
                        >= threshold
                    )


                    for minimum_size in (
                        MIN_COMPONENT_SIZES
                    ):

                        filtered_prediction = (
                            remove_small_components(
                                raw_prediction,
                                minimum_size,
                            )
                        )


                        metrics = calculate_metrics(
                            filtered_prediction,
                            gt,
                        )


                        results[
                            threshold
                        ][
                            minimum_size
                        ]["tp"] += metrics["tp"]


                        results[
                            threshold
                        ][
                            minimum_size
                        ]["fp"] += metrics["fp"]


                        results[
                            threshold
                        ][
                            minimum_size
                        ]["fn"] += metrics["fn"]


            if batch_index == 1:

                print(
                    "\nFirst batch processed."
                )

                print(
                    "Input:",
                    tuple(images.shape),
                )

                print(
                    "Output:",
                    tuple(outputs.shape),
                )


    # ========================================================
    # RESULTS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "CONNECTED-COMPONENT RESULTS"
    )

    print(
        "=" * 70
    )


    report_lines = []


    report_lines.append(
        "GLIOMA CONNECTED-COMPONENT ANALYSIS"
    )

    report_lines.append(
        "=" * 70
    )


    for threshold in PROBABILITY_THRESHOLDS:

        print(
            f"\nPROBABILITY THRESHOLD = "
            f"{threshold:.2f}"
        )

        print(
            "-" * 70
        )


        report_lines.append(
            f"\nProbability threshold = "
            f"{threshold:.2f}"
        )


        report_lines.append(
            "-" * 70
        )


        for minimum_size in (
            MIN_COMPONENT_SIZES
        ):

            stats = results[
                threshold
            ][
                minimum_size
            ]


            tp = stats["tp"]

            fp = stats["fp"]

            fn = stats["fn"]


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


            print(
                f"Minimum component: "
                f"{minimum_size:4d} | "
                f"Precision: {precision:.6f} | "
                f"Recall: {recall:.6f} | "
                f"Dice: {dice:.6f} | "
                f"IoU: {iou:.6f} | "
                f"FP: {fp} | "
                f"FN: {fn}"
            )


            report_lines.append(
                f"Minimum component: "
                f"{minimum_size:4d} | "
                f"Precision: {precision:.6f} | "
                f"Recall: {recall:.6f} | "
                f"Dice: {dice:.6f} | "
                f"IoU: {iou:.6f} | "
                f"FP: {fp} | "
                f"FN: {fn}"
            )


    # ========================================================
    # FIND BEST DICE
    # ========================================================

    all_results = []


    for threshold in PROBABILITY_THRESHOLDS:

        for minimum_size in MIN_COMPONENT_SIZES:

            stats = results[
                threshold
            ][
                minimum_size
            ]


            tp = stats["tp"]

            fp = stats["fp"]

            fn = stats["fn"]


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


            all_results.append(
                {
                    "threshold": threshold,
                    "minimum_size": minimum_size,
                    "precision": precision,
                    "recall": recall,
                    "dice": dice,
                    "iou": iou,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                }
            )


    # --------------------------------------------------------
    # Best Dice
    # --------------------------------------------------------

    best_dice = max(
        all_results,
        key=lambda item: item["dice"],
    )


    # --------------------------------------------------------
    # Best precision subject to recall >= 0.85
    # --------------------------------------------------------

    acceptable_results = [
        item
        for item in all_results
        if item["recall"] >= 0.85
    ]


    if acceptable_results:

        best_precision = max(
            acceptable_results,
            key=lambda item: item["precision"],
        )

    else:

        best_precision = None


    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "BEST CONFIGURATIONS"
    )

    print(
        "=" * 70
    )


    print(
        "\nBest Dice configuration:"
    )

    print(
        "Probability threshold:",
        f"{best_dice['threshold']:.2f}",
    )

    print(
        "Minimum component:",
        best_dice["minimum_size"],
    )

    print(
        "Precision:",
        f"{best_dice['precision']:.6f}",
    )

    print(
        "Recall:",
        f"{best_dice['recall']:.6f}",
    )

    print(
        "Dice:",
        f"{best_dice['dice']:.6f}",
    )

    print(
        "IoU:",
        f"{best_dice['iou']:.6f}",
    )

    print(
        "False positives:",
        best_dice["fp"],
    )


    report_lines.append(
        "\nBEST DICE CONFIGURATION"
    )

    report_lines.append(
        str(best_dice)
    )


    if best_precision is not None:

        print(
            "\nBest precision configuration "
            "(recall >= 0.85):"
        )

        print(
            "Probability threshold:",
            f"{best_precision['threshold']:.2f}",
        )

        print(
            "Minimum component:",
            best_precision["minimum_size"],
        )

        print(
            "Precision:",
            f"{best_precision['precision']:.6f}",
        )

        print(
            "Recall:",
            f"{best_precision['recall']:.6f}",
        )

        print(
            "Dice:",
            f"{best_precision['dice']:.6f}",
        )

        print(
            "IoU:",
            f"{best_precision['iou']:.6f}",
        )

        print(
            "False positives:",
            best_precision["fp"],
        )


        report_lines.append(
            "\nBEST PRECISION CONFIGURATION "
            "(RECALL >= 0.85)"
        )

        report_lines.append(
            str(best_precision)
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
            "\n".join(
                report_lines
            )
        )


    print(
        "\nReport saved:"
    )

    print(
        REPORT_PATH
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "CONNECTED-COMPONENT ANALYSIS COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()