"""
Glioma 3D U-Net Evaluation
===========================

Purpose
-------
Evaluate the best small-scale Glioma segmentation model on
unseen validation cases.

Experiment
----------
- Same 20/5 case-level split used during training
- Best checkpoint from small-scale training
- 4 MRI input channels
- 4 segmentation classes
- 64 x 64 x 64 patches
- 2 patches per case

Metrics
-------
1. Validation loss
2. Overall Dice
3. Per-class Dice
4. Tumor Dice
5. Overall IoU
6. Tumor IoU
7. Ground-truth tumor voxels
8. Predicted tumor voxels

Important
---------
GliomaDataset returns a LIST of patch dictionaries for each case.

Therefore a custom collate function is required.
"""


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import random

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from monai.networks.nets import UNet
from monai.losses import DiceCELoss

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

TRAIN_CASES_FOR_TEST = 20

VAL_CASES_FOR_TEST = 5

BATCH_SIZE = 1

NUM_SAMPLES_PER_CASE = 2

PATCH_SIZE = (
    64,
    64,
    64,
)

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

LEARNING_RATE = 1e-4

NUM_WORKERS = 0


# ============================================================
# CHECKPOINT
# ============================================================

CHECKPOINT_PATH = (
    Path("outputs")
    / "glioma_training"
    / "glioma_best_small_experiment.pth"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    Path("outputs")
    / "glioma_evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


RESULTS_PATH = (
    OUTPUT_DIR
    / "glioma_evaluation_results.txt"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# CUSTOM COLLATE FUNCTION
# ============================================================

def glioma_collate_fn(batch):
    """
    Convert case-level dataset output into a patch batch.

    Input:

        [
            [
                {"image": ..., "label": ...},
                {"image": ..., "label": ...}
            ]
        ]

    Output:

        {
            "image": (N,4,64,64,64),
            "label": (N,1,64,64,64)
        }
    """

    if len(batch) == 0:

        raise RuntimeError(
            "Empty DataLoader batch."
        )


    patches = []


    for case_output in batch:

        if not isinstance(
            case_output,
            list,
        ):

            raise TypeError(
                "Expected GliomaDataset output "
                "to be a list."
            )


        patches.extend(
            case_output
        )


    if len(patches) == 0:

        raise RuntimeError(
            "No patches found in batch."
        )


    images = []

    labels = []


    for patch in patches:

        if not isinstance(
            patch,
            dict,
        ):

            raise TypeError(
                "Patch must be a dictionary."
            )


        if "image" not in patch:

            raise KeyError(
                "Missing 'image' in patch."
            )


        if "label" not in patch:

            raise KeyError(
                "Missing 'label' in patch."
            )


        images.append(
            patch["image"]
        )

        labels.append(
            patch["label"]
        )


    images = torch.stack(
        images,
        dim=0,
    )

    labels = torch.stack(
        labels,
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


    return model.to(
        DEVICE
    )


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint(
    model,
):

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            "Checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )


    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )


    if "model_state_dict" not in checkpoint:

        raise KeyError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )


    model.load_state_dict(
        checkpoint["model_state_dict"]
    )


    print(
        "\nCheckpoint loaded successfully."
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

        validation_dice = (
            checkpoint["validation_dice"]
        )


        if len(validation_dice) > 0:

            print(
                "Checkpoint best/last Dice:",
                f"{validation_dice[-1]:.6f}",
            )


    return checkpoint


# ============================================================
# DICE CALCULATION
# ============================================================

def calculate_dice(
    prediction,
    target,
):

    """
    Calculate Dice for one class.

    prediction:
        Boolean tensor

    target:
        Boolean tensor
    """

    prediction = prediction.bool()

    target = target.bool()


    intersection = (
        prediction
        &
        target
    ).sum().float()


    prediction_volume = (
        prediction.sum().float()
    )

    target_volume = (
        target.sum().float()
    )


    denominator = (
        prediction_volume
        +
        target_volume
    )


    if denominator.item() == 0:

        return None


    dice = (
        2.0
        *
        intersection
        /
        denominator
    )


    return dice.item()


# ============================================================
# IOU CALCULATION
# ============================================================

def calculate_iou(
    prediction,
    target,
):

    """
    Calculate IoU for one class.
    """

    prediction = prediction.bool()

    target = target.bool()


    intersection = (
        prediction
        &
        target
    ).sum().float()


    union = (
        prediction
        |
        target
    ).sum().float()


    if union.item() == 0:

        return None


    iou = (
        intersection
        /
        union
    )


    return iou.item()


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate(
    model,
    loader,
    loss_function,
):

    model.eval()


    total_loss = 0.0

    batch_count = 0


    # --------------------------------------------------------
    # Store per-class values
    # --------------------------------------------------------

    dice_values = {
        0: [],
        1: [],
        2: [],
        3: [],
    }


    iou_values = {
        0: [],
        1: [],
        2: [],
        3: [],
    }


    # --------------------------------------------------------
    # Tumor-level statistics
    # --------------------------------------------------------

    tumor_dice_values = []

    tumor_iou_values = []


    total_ground_truth_tumor = 0

    total_predicted_tumor = 0

    total_correct_tumor = 0

    total_correct_voxels = 0

    total_voxels = 0


    # --------------------------------------------------------
    # Case counter
    # --------------------------------------------------------

    processed_patches = 0


    print(
        "\n" + "=" * 70
    )

    print(
        "RUNNING EVALUATION"
    )

    print(
        "=" * 70
    )


    with torch.no_grad():

        for batch_index, batch in enumerate(
            loader,
            start=1,
        ):

            images = batch["image"]

            labels = batch["label"]


            # ------------------------------------------------
            # Move to GPU
            # ------------------------------------------------

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True,
            )


            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            outputs = model(
                images
            )


            # ------------------------------------------------
            # Validate output shape
            # ------------------------------------------------

            expected_shape = (
                images.shape[0],
                OUTPUT_CLASSES,
                *PATCH_SIZE,
            )


            if tuple(outputs.shape) != expected_shape:

                raise RuntimeError(
                    "Unexpected model output shape.\n"
                    f"Expected: {expected_shape}\n"
                    f"Received: {tuple(outputs.shape)}"
                )


            # ------------------------------------------------
            # Loss
            # ------------------------------------------------

            loss = loss_function(
                outputs,
                labels,
            )


            if not torch.isfinite(loss):

                raise RuntimeError(
                    "Non-finite evaluation loss."
                )


            total_loss += loss.item()

            batch_count += 1


            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            predictions = torch.argmax(
                outputs,
                dim=1,
            )


            target = labels.squeeze(
                1
            )


            # ------------------------------------------------
            # Process each patch
            # ------------------------------------------------

            for patch_index in range(
                predictions.shape[0]
            ):

                patch_prediction = (
                    predictions[patch_index]
                )

                patch_target = (
                    target[patch_index]
                )


                processed_patches += 1


                # =================================================
                # PER CLASS METRICS
                # =================================================

                for class_index in range(
                    OUTPUT_CLASSES
                ):

                    prediction_class = (
                        patch_prediction
                        ==
                        class_index
                    )


                    target_class = (
                        patch_target
                        ==
                        class_index
                    )


                    dice = calculate_dice(
                        prediction_class,
                        target_class,
                    )


                    iou = calculate_iou(
                        prediction_class,
                        target_class,
                    )


                    if dice is not None:

                        dice_values[
                            class_index
                        ].append(
                            dice
                        )


                    if iou is not None:

                        iou_values[
                            class_index
                        ].append(
                            iou
                        )


                # =================================================
                # TUMOR VS BACKGROUND
                # =================================================

                prediction_tumor = (
                    patch_prediction > 0
                )


                target_tumor = (
                    patch_target > 0
                )


                tumor_dice = calculate_dice(
                    prediction_tumor,
                    target_tumor,
                )


                tumor_iou = calculate_iou(
                    prediction_tumor,
                    target_tumor,
                )


                if tumor_dice is not None:

                    tumor_dice_values.append(
                        tumor_dice
                    )


                if tumor_iou is not None:

                    tumor_iou_values.append(
                        tumor_iou
                    )


                # ------------------------------------------------
                # Tumor voxel statistics
                # ------------------------------------------------

                ground_truth_tumor = (
                    target_tumor.sum().item()
                )


                predicted_tumor = (
                    prediction_tumor.sum().item()
                )


                correct_tumor = (
                    (
                        prediction_tumor
                        &
                        target_tumor
                    )
                    .sum()
                    .item()
                )

                correct_voxels = (
                    patch_prediction
                    ==
                    patch_target
                ).sum().item()

                total_correct_voxels += correct_voxels

                total_voxels += patch_target.numel()


                total_ground_truth_tumor += (
                    ground_truth_tumor
                )


                total_predicted_tumor += (
                    predicted_tumor
                )


                total_correct_tumor += (
                    correct_tumor
                )


            # ----------------------------------------------------
            # First batch information
            # ----------------------------------------------------

            if batch_index == 1:

                print(
                    "\nFirst evaluation batch:"
                )

                print(
                    "MRI:",
                    tuple(images.shape),
                )

                print(
                    "Mask:",
                    tuple(labels.shape),
                )

                print(
                    "Output:",
                    tuple(outputs.shape),
                )

                print(
                    "Loss:",
                    f"{loss.item():.6f}",
                )

                print(
                    "Prediction labels:",
                    torch.unique(
                        predictions
                    ).tolist(),
                )


    # ========================================================
    # FINAL METRICS
    # ========================================================

    average_loss = (
        total_loss
        /
        max(batch_count, 1)
    )


    # --------------------------------------------------------
    # Mean class Dice
    # --------------------------------------------------------

    mean_class_dice = {}


    for class_index in range(
        OUTPUT_CLASSES
    ):

        values = dice_values[
            class_index
        ]


        if values:

            mean_class_dice[
                class_index
            ] = (
                sum(values)
                /
                len(values)
            )

        else:

            mean_class_dice[
                class_index
            ] = None


    # --------------------------------------------------------
    # Mean class IoU
    # --------------------------------------------------------

    mean_class_iou = {}


    for class_index in range(
        OUTPUT_CLASSES
    ):

        values = iou_values[
            class_index
        ]


        if values:

            mean_class_iou[
                class_index
            ] = (
                sum(values)
                /
                len(values)
            )

        else:

            mean_class_iou[
                class_index
            ] = None


    # --------------------------------------------------------
    # Tumor Dice
    # --------------------------------------------------------

    if tumor_dice_values:

        tumor_dice = (
            sum(tumor_dice_values)
            /
            len(tumor_dice_values)
        )

    else:

        tumor_dice = 0.0


    # --------------------------------------------------------
    # Tumor IoU
    # --------------------------------------------------------

    if tumor_iou_values:

        tumor_iou = (
            sum(tumor_iou_values)
            /
            len(tumor_iou_values)
        )

    else:

        tumor_iou = 0.0


    # --------------------------------------------------------
    # Tumor precision / recall
    # --------------------------------------------------------

    tumor_precision_denominator = (
        total_predicted_tumor
    )


    tumor_recall_denominator = (
        total_ground_truth_tumor
    )


    if tumor_precision_denominator > 0:

        tumor_precision = (
            total_correct_tumor
            /
            tumor_precision_denominator
        )

    else:

        tumor_precision = 0.0


    if tumor_recall_denominator > 0:

        tumor_recall = (
            total_correct_tumor
            /
            tumor_recall_denominator
        )

    else:

        tumor_recall = 0.0


    voxel_accuracy = (
        total_correct_voxels
        /
        total_voxels
        if total_voxels > 0
        else 0.0
    )


    return {
        "average_loss":
            average_loss,

        "class_dice":
            mean_class_dice,

        "class_iou":
            mean_class_iou,

        "tumor_dice":
            tumor_dice,

        "tumor_iou":
            tumor_iou,

        "tumor_precision":
            tumor_precision,

        "tumor_recall":
            tumor_recall,

        "voxel_accuracy":
            voxel_accuracy,

        "ground_truth_tumor_voxels":
            total_ground_truth_tumor,

        "predicted_tumor_voxels":
            total_predicted_tumor,

        "correct_tumor_voxels":
            total_correct_tumor,

        "processed_patches":
            processed_patches,
    }


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results,
):

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "GLIOMA 3D U-NET EVALUATION RESULTS\n"
        )

        file.write(
            "=" * 70
            +
            "\n\n"
        )


        file.write(
            f"Dataset: {DATASET_ROOT}\n"
        )

        file.write(
            f"Checkpoint: {CHECKPOINT_PATH}\n"
        )

        file.write(
            f"Device: {DEVICE}\n"
        )

        file.write(
            f"Validation cases: {VAL_CASES_FOR_TEST}\n"
        )

        file.write(
            f"Patches per case: "
            f"{NUM_SAMPLES_PER_CASE}\n"
        )

        file.write(
            f"Processed patches: "
            f"{results['processed_patches']}\n\n"
        )


        file.write(
            "-" * 70
            +
            "\n"
        )

        file.write(
            "OVERALL RESULTS\n"
        )

        file.write(
            "-" * 70
            +
            "\n"
        )


        file.write(
            f"Validation loss: "
            f"{results['average_loss']:.6f}\n"
        )

        file.write(
            f"Tumor Dice: "
            f"{results['tumor_dice']:.6f}\n"
        )

        file.write(
            f"Tumor IoU: "
            f"{results['tumor_iou']:.6f}\n"
        )

        file.write(
            f"Tumor Precision: "
            f"{results['tumor_precision']:.6f}\n"
        )

        file.write(
            f"Tumor Recall: "
            f"{results['tumor_recall']:.6f}\n"
        )


        file.write(
            "\n"
        )


        file.write(
            "-" * 70
            +
            "\n"
        )

        file.write(
            "PER-CLASS DICE\n"
        )

        file.write(
            "-" * 70
            +
            "\n"
        )


        for class_index in range(
            OUTPUT_CLASSES
        ):

            value = results[
                "class_dice"
            ][
                class_index
            ]


            if value is None:

                file.write(
                    f"Class {class_index}: N/A\n"
                )

            else:

                file.write(
                    f"Class {class_index}: "
                    f"{value:.6f}\n"
                )


        file.write(
            "\n"
        )


        file.write(
            "-" * 70
            +
            "\n"
        )

        file.write(
            "PER-CLASS IoU\n"
        )

        file.write(
            "-" * 70
            +
            "\n"
        )


        for class_index in range(
            OUTPUT_CLASSES
        ):

            value = results[
                "class_iou"
            ][
                class_index
            ]


            if value is None:

                file.write(
                    f"Class {class_index}: N/A\n"
                )

            else:

                file.write(
                    f"Class {class_index}: "
                    f"{value:.6f}\n"
                )


        file.write(
            "\n"
        )


        file.write(
            "-" * 70
            +
            "\n"
        )

        file.write(
            "TUMOR VOXEL STATISTICS\n"
        )

        file.write(
            "-" * 70
            +
            "\n"
        )


        file.write(
            f"Ground truth tumor voxels: "
            f"{results['ground_truth_tumor_voxels']}\n"
        )

        file.write(
            f"Predicted tumor voxels: "
            f"{results['predicted_tumor_voxels']}\n"
        )

        file.write(
            f"Correct tumor voxels: "
            f"{results['correct_tumor_voxels']}\n"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "GLIOMA 3D U-NET MODEL EVALUATION"
    )

    print(
        "=" * 70
    )


    # ========================================================
    # ENVIRONMENT
    # ========================================================

    print(
        "\nDataset:",
        DATASET_ROOT,
    )

    print(
        "Device:",
        DEVICE,
    )


    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    else:

        print(
            "GPU: CPU",
        )


    # ========================================================
    # CHECKPOINT
    # ========================================================

    print(
        "\nCheckpoint:"
    )

    print(
        CHECKPOINT_PATH
    )


    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            "Best model checkpoint does not exist."
        )


    # ========================================================
    # FIND CASES
    # ========================================================

    cases = get_cases(
        DATASET_ROOT
    )


    print(
        "\nTotal Glioma cases:",
        len(cases),
    )


    # ========================================================
    # SAME SPLIT AS TRAINING
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


    print(
        "\nEvaluation validation cases:"
    )


    for index, case in enumerate(
        validation_cases,
        start=1,
    ):

        print(
            f"  {index}. {case.name}"
        )


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
            "Training/validation overlap detected: "
            f"{overlap}"
        )


    print(
        "\n✓ No training/validation overlap."
    )


    # ========================================================
    # DATASET
    # ========================================================

    validation_dataset = GliomaDataset(
        validation_cases
    )


    print(
        "\nValidation dataset size:",
        len(validation_dataset),
    )


    # ========================================================
    # DATALOADER
    # ========================================================

    validation_loader = DataLoader(
        validation_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=torch.cuda.is_available(),

        collate_fn=glioma_collate_fn,
    )


    print(
        "Validation DataLoader created."
    )


    print(
        "Batch size:",
        BATCH_SIZE,
    )

    print(
        "Patches per case:",
        NUM_SAMPLES_PER_CASE,
    )

    print(
        "Patch size:",
        PATCH_SIZE,
    )


    # ========================================================
    # MODEL
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "CREATING MODEL"
    )

    print(
        "=" * 70
    )


    model = create_model()


    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )


    print(
        "Total parameters:",
        f"{total_parameters:,}",
    )


    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    load_checkpoint(
        model
    )


    # ========================================================
    # LOSS
    # ========================================================

    loss_function = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
    )


    # ========================================================
    # EVALUATE
    # ========================================================

    results = evaluate(
        model=model,

        loader=validation_loader,

        loss_function=loss_function,
    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "EVALUATION RESULTS"
    )

    print(
        "=" * 70
    )


    print(
        "\nValidation loss:",
        f"{results['average_loss']:.6f}",
    )


    print(
        "\nTumor Dice:",
        f"{results['tumor_dice']:.6f}",
    )


    print(
        "Tumor IoU:",
        f"{results['tumor_iou']:.6f}",
    )


    print(
        "Tumor Precision:",
        f"{results['tumor_precision']:.6f}",
    )


    print(
        "Voxel Accuracy:",
        f"{results['voxel_accuracy']:.6f}",
    )


    print(
        "\nPer-class Dice:"
    )


    for class_index in range(
        OUTPUT_CLASSES
    ):

        value = results[
            "class_dice"
        ][
            class_index
        ]


        if value is None:

            print(
                f"  Class {class_index}: N/A"
            )

        else:

            print(
                f"  Class {class_index}: "
                f"{value:.6f}"
            )


    print(
        "\nPer-class IoU:"
    )


    for class_index in range(
        OUTPUT_CLASSES
    ):

        value = results[
            "class_iou"
        ][
            class_index
        ]


        if value is None:

            print(
                f"  Class {class_index}: N/A"
            )

        else:

            print(
                f"  Class {class_index}: "
                f"{value:.6f}"
            )


    print(
        "\nTumor voxel statistics:"
    )


    print(
        "  Ground truth:",
        results[
            "ground_truth_tumor_voxels"
        ],
    )


    print(
        "  Predicted:",
        results[
            "predicted_tumor_voxels"
        ],
    )


    print(
        "  Correct:",
        results[
            "correct_tumor_voxels"
        ],
    )


    print(
        "\nProcessed patches:",
        results[
            "processed_patches"
        ],
    )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    save_results(
        results
    )


    print(
        "\nResults saved to:"
    )

    print(
        RESULTS_PATH
    )


    # ========================================================
    # GPU MEMORY
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

        peak = (
            torch.cuda.max_memory_allocated()
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
            "Peak:",
            f"{peak:.2f} MB",
        )


    # ========================================================
    # COMPLETE
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "GLIOMA MODEL EVALUATION COMPLETE"
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()