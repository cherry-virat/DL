"""
Glioma 3D U-Net Medium-Scale Training
======================================

Purpose
-------
Improved training experiment intended to reduce false-positive
tumor predictions while maintaining high tumor recall.

Experiment
----------
- 100 training cases
- 20 validation cases
- 2 patches per case
- 64 x 64 x 64 patches
- Batch size = 1 case
- 20 epochs
- 4 MRI channels
- 4 segmentation classes
- Dice + Cross Entropy loss
- Adam optimizer
- ReduceLROnPlateau scheduler
- CUDA GPU when available

Important
---------
GliomaDataset returns a LIST of patch dictionaries for every case.

Therefore a custom collate function is required.

This file does NOT modify:
    preprocessing/glioma_dataset.py

The current small experiment is preserved.
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
from monai.metrics import DiceMetric

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

# ------------------------------------------------------------
# Medium experiment
# ------------------------------------------------------------

TRAIN_CASES_FOR_TEST = 100
VAL_CASES_FOR_TEST = 20

# One case per DataLoader item.
# Each case produces 2 patches.
BATCH_SIZE = 1

NUM_SAMPLES_PER_CASE = 2

PATCH_SIZE = (
    64,
    64,
    64,
)

NUM_EPOCHS = 20

LEARNING_RATE = 1e-4

MIN_LEARNING_RATE = 1e-6

NUM_WORKERS = 0

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# Early stopping
# ------------------------------------------------------------

EARLY_STOPPING_PATIENCE = 6


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

OUTPUT_DIR = (
    Path("outputs")
    / "glioma_training"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CHECKPOINT PATHS
# ============================================================

BEST_MODEL_PATH = (
    OUTPUT_DIR
    /
    "glioma_best_medium_experiment.pth"
)

FINAL_MODEL_PATH = (
    OUTPUT_DIR
    /
    "glioma_final_medium_experiment.pth"
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
    Convert:

        [
            [
                patch1,
                patch2
            ]
        ]

    into:

        {
            "image": Tensor(N,4,64,64,64),
            "label": Tensor(N,1,64,64,64)
        }

    Since BATCH_SIZE = 1 and every case produces
    NUM_SAMPLES_PER_CASE = 2 patches, N is normally 2.
    """

    if len(batch) == 0:

        raise RuntimeError(
            "DataLoader returned an empty batch."
        )


    patches = []


    # --------------------------------------------------------
    # Flatten case -> patches
    # --------------------------------------------------------

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
            "No patches were returned by the dataset."
        )


    images = []

    labels = []


    # --------------------------------------------------------
    # Extract patch tensors
    # --------------------------------------------------------

    for patch in patches:

        if not isinstance(
            patch,
            dict,
        ):

            raise TypeError(
                "Each patch must be a dictionary."
            )


        if "image" not in patch:

            raise KeyError(
                "Patch dictionary does not contain 'image'."
            )


        if "label" not in patch:

            raise KeyError(
                "Patch dictionary does not contain 'label'."
            )


        image = patch["image"]

        label = patch["label"]


        if not torch.is_tensor(image):

            image = torch.as_tensor(
                image
            )


        if not torch.is_tensor(label):

            label = torch.as_tensor(
                label
            )


        images.append(
            image
        )

        labels.append(
            label
        )


    # --------------------------------------------------------
    # Stack
    # --------------------------------------------------------

    images = torch.stack(
        images,
        dim=0,
    )

    labels = torch.stack(
        labels,
        dim=0,
    )


    # --------------------------------------------------------
    # Shape validation
    # --------------------------------------------------------

    expected_image_shape = (
        images.shape[0],
        INPUT_CHANNELS,
        *PATCH_SIZE,
    )

    expected_label_shape = (
        labels.shape[0],
        1,
        *PATCH_SIZE,
    )


    if tuple(images.shape) != expected_image_shape:

        raise RuntimeError(
            "Invalid image batch shape.\n"
            f"Expected: {expected_image_shape}\n"
            f"Received: {tuple(images.shape)}"
        )


    if tuple(labels.shape) != expected_label_shape:

        raise RuntimeError(
            "Invalid label batch shape.\n"
            f"Expected: {expected_label_shape}\n"
            f"Received: {tuple(labels.shape)}"
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
# MODEL INFORMATION
# ============================================================

def print_model_information(
    model,
):

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )


    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


    model_size_mb = (
        total_parameters
        * 4
        /
        (1024 ** 2)
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "MODEL INFORMATION"
    )

    print(
        "=" * 70
    )

    print(
        "Input channels:",
        INPUT_CHANNELS
    )

    print(
        "Output classes:",
        OUTPUT_CLASSES
    )

    print(
        "Model channels:",
        MODEL_CHANNELS
    )

    print(
        "Total parameters:",
        f"{total_parameters:,}"
    )

    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}"
    )

    print(
        "Approximate model size:",
        f"{model_size_mb:.2f} MB"
    )


# ============================================================
# VALIDATE TRAINING BATCH
# ============================================================

def validate_batch(
    batch,
):

    images = batch["image"]

    labels = batch["label"]


    print(
        "\n" + "-" * 70
    )

    print(
        "BATCH VALIDATION"
    )

    print(
        "-" * 70
    )


    print(
        "MRI batch:",
        tuple(images.shape)
    )

    print(
        "Mask batch:",
        tuple(labels.shape)
    )

    print(
        "MRI dtype:",
        images.dtype
    )

    print(
        "Mask dtype:",
        labels.dtype
    )


    # --------------------------------------------------------
    # Shape checks
    # --------------------------------------------------------

    if images.ndim != 5:

        raise RuntimeError(
            f"Expected image ndim=5, "
            f"got {images.ndim}"
        )


    if labels.ndim != 5:

        raise RuntimeError(
            f"Expected label ndim=5, "
            f"got {labels.ndim}"
        )


    if images.shape[1] != INPUT_CHANNELS:

        raise RuntimeError(
            f"Expected {INPUT_CHANNELS} "
            f"MRI channels, "
            f"got {images.shape[1]}"
        )


    if labels.shape[1] != 1:

        raise RuntimeError(
            "Expected one segmentation label channel."
        )


    if tuple(images.shape[2:]) != PATCH_SIZE:

        raise RuntimeError(
            f"Unexpected patch shape: "
            f"{images.shape}"
        )


    if tuple(labels.shape[2:]) != PATCH_SIZE:

        raise RuntimeError(
            f"Unexpected label patch shape: "
            f"{labels.shape}"
        )


    # --------------------------------------------------------
    # Dtype checks
    # --------------------------------------------------------

    if images.dtype != torch.float32:

        raise RuntimeError(
            f"Expected MRI float32, "
            f"got {images.dtype}"
        )


    if labels.dtype != torch.int64:

        raise RuntimeError(
            f"Expected labels int64, "
            f"got {labels.dtype}"
        )


    # --------------------------------------------------------
    # Label range
    # --------------------------------------------------------

    unique_labels = torch.unique(
        labels
    )


    print(
        "Unique labels:",
        unique_labels.tolist()
    )


    if torch.any(
        unique_labels < 0
    ):

        raise RuntimeError(
            "Negative labels detected."
        )


    if torch.any(
        unique_labels >= OUTPUT_CLASSES
    ):

        raise RuntimeError(
            f"Label >= {OUTPUT_CLASSES} detected."
        )


    # --------------------------------------------------------
    # Tumor statistics
    # --------------------------------------------------------

    tumor_voxels = (
        labels > 0
    ).sum(
        dim=(1, 2, 3, 4)
    )


    print(
        "Tumor voxels per patch:",
        tumor_voxels.tolist()
    )


    print(
        "Patches containing tumor:",
        f"{(tumor_voxels > 0).sum().item()} / "
        f"{len(tumor_voxels)}"
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    loss_function,
    optimizer,
    epoch,
):

    model.train()


    running_loss = 0.0

    number_of_batches = 0


    first_batch_printed = False


    for batch_index, batch in enumerate(
        loader,
        start=1,
    ):

        images = batch["image"]

        labels = batch["label"]


        # ----------------------------------------------------
        # Move to GPU
        # ----------------------------------------------------

        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )


        # ----------------------------------------------------
        # Clear gradients
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )


        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        outputs = model(
            images
        )


        # ----------------------------------------------------
        # Output validation
        # ----------------------------------------------------

        if outputs.shape[1] != OUTPUT_CLASSES:

            raise RuntimeError(
                f"Expected {OUTPUT_CLASSES} output "
                f"classes, got {outputs.shape}"
            )


        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = loss_function(
            outputs,
            labels,
        )


        if not torch.isfinite(
            loss
        ):

            raise RuntimeError(
                f"Non-finite loss: {loss.item()}"
            )


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()


        # ----------------------------------------------------
        # Gradient monitoring
        # ----------------------------------------------------

        maximum_gradient = 0.0

        gradients_found = False


        for parameter in model.parameters():

            if parameter.grad is not None:

                gradients_found = True

                current_gradient = (
                    parameter.grad.detach()
                    .abs()
                    .max()
                    .item()
                )


                maximum_gradient = max(
                    maximum_gradient,
                    current_gradient,
                )


        if not gradients_found:

            raise RuntimeError(
                "No gradients generated."
            )


        # ----------------------------------------------------
        # Gradient clipping
        # ----------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )


        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------

        optimizer.step()


        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        running_loss += loss.item()

        number_of_batches += 1


        # ----------------------------------------------------
        # First batch
        # ----------------------------------------------------

        if not first_batch_printed:

            print(
                "\nFirst training batch:"
            )

            print(
                "MRI:",
                tuple(images.shape)
            )

            print(
                "Mask:",
                tuple(labels.shape)
            )

            print(
                "Output:",
                tuple(outputs.shape)
            )

            print(
                "Loss:",
                f"{loss.item():.6f}"
            )

            print(
                "Maximum gradient:",
                f"{maximum_gradient:.6f}"
            )


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
                    "GPU allocated:",
                    f"{allocated:.2f} MB"
                )

                print(
                    "GPU reserved:",
                    f"{reserved:.2f} MB"
                )


            first_batch_printed = True


    average_loss = (
        running_loss
        /
        max(
            number_of_batches,
            1,
        )
    )


    return (
        average_loss,
        number_of_batches,
    )


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    loader,
    loss_function,
):

    model.eval()


    validation_loss = 0.0

    number_of_batches = 0


    dice_metric = DiceMetric(
        include_background=False,
        reduction="mean",
    )


    # Global tumor statistics
    total_tp = 0
    total_fp = 0
    total_fn = 0


    with torch.no_grad():

        for batch in loader:

            images = batch["image"]

            labels = batch["label"]


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
            # Loss
            # ------------------------------------------------

            loss = loss_function(
                outputs,
                labels,
            )


            if not torch.isfinite(
                loss
            ):

                raise RuntimeError(
                    "Non-finite validation loss."
                )


            validation_loss += (
                loss.item()
            )

            number_of_batches += 1


            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            predictions = torch.argmax(
                outputs,
                dim=1,
            )


            # ------------------------------------------------
            # Tumor binary masks
            # ------------------------------------------------

            predicted_tumor = (
                predictions > 0
            )

            ground_truth_tumor = (
                labels.squeeze(1) > 0
            )


            total_tp += torch.logical_and(
                predicted_tumor,
                ground_truth_tumor,
            ).sum().item()


            total_fp += torch.logical_and(
                predicted_tumor,
                ~ground_truth_tumor,
            ).sum().item()


            total_fn += torch.logical_and(
                ~predicted_tumor,
                ground_truth_tumor,
            ).sum().item()


            # ------------------------------------------------
            # One-hot prediction
            # ------------------------------------------------

            predictions_one_hot = (
                F.one_hot(
                    predictions,
                    num_classes=OUTPUT_CLASSES,
                )
                .permute(
                    0,
                    4,
                    1,
                    2,
                    3,
                )
                .float()
            )


            # ------------------------------------------------
            # One-hot labels
            # ------------------------------------------------

            labels_one_hot = (
                F.one_hot(
                    labels.squeeze(1),
                    num_classes=OUTPUT_CLASSES,
                )
                .permute(
                    0,
                    4,
                    1,
                    2,
                    3,
                )
                .float()
            )


            # ------------------------------------------------
            # Dice
            # ------------------------------------------------

            dice_metric(
                predictions_one_hot,
                labels_one_hot,
            )


    average_loss = (
        validation_loss
        /
        max(
            number_of_batches,
            1,
        )
    )


    dice_value = (
        dice_metric
        .aggregate()
        .item()
    )


    dice_metric.reset()


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


    tumor_dice = (
        2 * total_tp
        /
        max(
            2 * total_tp
            +
            total_fp
            +
            total_fn,
            1,
        )
    )


    return {
        "loss": average_loss,
        "dice": dice_value,
        "tumor_dice": tumor_dice,
        "precision": precision,
        "recall": recall,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "batches": number_of_batches,
    }


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    train_losses,
    validation_losses,
    validation_dice,
    validation_precision,
    validation_recall,
    validation_tumor_dice,
    path,
):

    torch.save(
        {
            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

            "train_losses":
                train_losses,

            "validation_losses":
                validation_losses,

            "validation_dice":
                validation_dice,

            "validation_precision":
                validation_precision,

            "validation_recall":
                validation_recall,

            "validation_tumor_dice":
                validation_tumor_dice,

            "patch_size":
                PATCH_SIZE,

            "input_channels":
                INPUT_CHANNELS,

            "output_classes":
                OUTPUT_CLASSES,

            "model_channels":
                MODEL_CHANNELS,

            "learning_rate":
                LEARNING_RATE,

            "seed":
                SEED,

            "train_cases":
                TRAIN_CASES_FOR_TEST,

            "validation_cases":
                VAL_CASES_FOR_TEST,
        },
        path,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "GLIOMA 3D U-NET MEDIUM-SCALE TRAINING"
    )

    print("=" * 70)


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


    # ========================================================
    # CASES
    # ========================================================

    cases = get_cases(
        DATASET_ROOT
    )


    print(
        "\nTotal Glioma cases:",
        len(cases)
    )


    required_cases = (
        TRAIN_CASES_FOR_TEST
        +
        VAL_CASES_FOR_TEST
    )


    if len(cases) < required_cases:

        raise RuntimeError(
            f"Need at least {required_cases} cases."
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


    # --------------------------------------------------------
    # Select experiment subset
    # --------------------------------------------------------

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
            "Training/validation overlap detected: "
            f"{sorted(overlap)}"
        )


    print(
        "\nSmall-scale medium experiment:"
    )

    print(
        "Training cases:",
        len(train_cases)
    )

    print(
        "Validation cases:",
        len(validation_cases)
    )

    print(
        "✓ No training/validation overlap."
    )


    # ========================================================
    # DATASETS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "CREATING DATASETS"
    )

    print("=" * 70)


    train_dataset = GliomaDataset(
        train_cases
    )


    validation_dataset = GliomaDataset(
        validation_cases
    )


    print(
        "Training dataset:",
        len(train_dataset)
    )

    print(
        "Validation dataset:",
        len(validation_dataset)
    )


    # ========================================================
    # DATALOADERS
    # ========================================================

    train_loader = DataLoader(
        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=NUM_WORKERS,

        pin_memory=torch.cuda.is_available(),

        collate_fn=glioma_collate_fn,
    )


    validation_loader = DataLoader(
        validation_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=torch.cuda.is_available(),

        collate_fn=glioma_collate_fn,
    )


    print(
        "\nDataLoaders created."
    )

    print(
        "Batch size:",
        BATCH_SIZE
    )

    print(
        "Patches per case:",
        NUM_SAMPLES_PER_CASE
    )

    print(
        "Patch size:",
        PATCH_SIZE
    )


    # ========================================================
    # MODEL
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "CREATING 3D U-NET"
    )

    print("=" * 70)


    model = create_model()


    print_model_information(
        model
    )


    # ========================================================
    # LOSS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "LOSS FUNCTION"
    )

    print("=" * 70)


    loss_function = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
    )


    print(
        "DiceCELoss"
    )


    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )


    print(
        "\nOptimizer: Adam"
    )

    print(
        "Learning rate:",
        LEARNING_RATE
    )


    # ========================================================
    # SCHEDULER
    # ========================================================

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,

            mode="max",

            factor=0.5,

            patience=3,

            min_lr=MIN_LEARNING_RATE,
        )
    )


    print(
        "Scheduler: ReduceLROnPlateau"
    )

    print(
        "Scheduler monitors: Tumor Dice"
    )


    # ========================================================
    # HISTORY
    # ========================================================

    train_losses = []

    validation_losses = []

    validation_dice_scores = []

    validation_precision_scores = []

    validation_recall_scores = []

    validation_tumor_dice_scores = []


    # ========================================================
    # BEST MODEL
    # ========================================================

    best_tumor_dice = -1.0

    best_precision = -1.0

    best_epoch = 0

    epochs_without_improvement = 0


    # ========================================================
    # TRAINING
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "STARTING MEDIUM TRAINING"
    )

    print("=" * 70)


    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):

        print(
            "\n" + "=" * 70
        )

        print(
            f"EPOCH {epoch}/{NUM_EPOCHS}"
        )

        print("=" * 70)


        # ----------------------------------------------------
        # Reset CUDA peak memory
        # ----------------------------------------------------

        if torch.cuda.is_available():

            torch.cuda.reset_peak_memory_stats()


        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        training_loss, training_batches = (
            train_one_epoch(
                model=model,

                loader=train_loader,

                loss_function=loss_function,

                optimizer=optimizer,

                epoch=epoch,
            )
        )


        train_losses.append(
            training_loss
        )


        print(
            "\nTraining batches:",
            training_batches
        )

        print(
            "Training loss:",
            f"{training_loss:.6f}"
        )


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        print(
            "\n" + "-" * 70
        )

        print(
            "VALIDATION"
        )

        print("-" * 70)


        validation_result = validate(
            model=model,

            loader=validation_loader,

            loss_function=loss_function,
        )


        validation_loss = (
            validation_result["loss"]
        )

        validation_dice = (
            validation_result["dice"]
        )

        tumor_dice = (
            validation_result["tumor_dice"]
        )

        precision = (
            validation_result["precision"]
        )

        recall = (
            validation_result["recall"]
        )

        tp = (
            validation_result["tp"]
        )

        fp = (
            validation_result["fp"]
        )

        fn = (
            validation_result["fn"]
        )


        validation_losses.append(
            validation_loss
        )

        validation_dice_scores.append(
            validation_dice
        )

        validation_precision_scores.append(
            precision
        )

        validation_recall_scores.append(
            recall
        )

        validation_tumor_dice_scores.append(
            tumor_dice
        )


        # ----------------------------------------------------
        # Print metrics
        # ----------------------------------------------------

        print(
            "\nValidation loss:",
            f"{validation_loss:.6f}"
        )

        print(
            "Validation multiclass Dice:",
            f"{validation_dice:.6f}"
        )

        print(
            "Tumor Dice:",
            f"{tumor_dice:.6f}"
        )

        print(
            "Tumor Precision:",
            f"{precision:.6f}"
        )

        print(
            "Tumor Recall:",
            f"{recall:.6f}"
        )

        print(
            "True positives:",
            tp
        )

        print(
            "False positives:",
            fp
        )

        print(
            "False negatives:",
            fn
        )


        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step(
            tumor_dice
        )


        current_lr = (
            optimizer.param_groups[0]["lr"]
        )


        print(
            "Current learning rate:",
            f"{current_lr:.8f}"
        )


        # ----------------------------------------------------
        # Best model criterion
        #
        # Tumor Dice is primary.
        # Precision is reported separately.
        # ----------------------------------------------------

        improved = (
            tumor_dice
            >
            best_tumor_dice
            +
            1e-6
        )


        if improved:

            best_tumor_dice = (
                tumor_dice
            )

            best_precision = (
                precision
            )

            best_epoch = (
                epoch
            )

            epochs_without_improvement = 0


            save_checkpoint(
                model=model,

                optimizer=optimizer,

                scheduler=scheduler,

                epoch=epoch,

                train_losses=train_losses,

                validation_losses=validation_losses,

                validation_dice=validation_dice_scores,

                validation_precision=validation_precision_scores,

                validation_recall=validation_recall_scores,

                validation_tumor_dice=validation_tumor_dice_scores,

                path=BEST_MODEL_PATH,
            )


            print(
                "\n✓ NEW BEST MODEL"
            )

            print(
                "Best tumor Dice:",
                f"{best_tumor_dice:.6f}"
            )

            print(
                "Corresponding precision:",
                f"{best_precision:.6f}"
            )

            print(
                "Saved:",
                BEST_MODEL_PATH
            )


        else:

            epochs_without_improvement += 1


            print(
                "\nNo tumor-Dice improvement."
            )

            print(
                "Epochs without improvement:",
                epochs_without_improvement
            )


        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >=
            EARLY_STOPPING_PATIENCE
        ):

            print(
                "\nEarly stopping activated."
            )

            print(
                "Patience:",
                EARLY_STOPPING_PATIENCE
            )

            break


        # ----------------------------------------------------
        # GPU memory
        # ----------------------------------------------------

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
                f"{allocated:.2f} MB"
            )

            print(
                "Reserved:",
                f"{reserved:.2f} MB"
            )

            print(
                "Peak:",
                f"{peak:.2f} MB"
            )


    # ========================================================
    # SAVE FINAL MODEL
    # ========================================================

    final_epoch = len(
        train_losses
    )


    save_checkpoint(
        model=model,

        optimizer=optimizer,

        scheduler=scheduler,

        epoch=final_epoch,

        train_losses=train_losses,

        validation_losses=validation_losses,

        validation_dice=validation_dice_scores,

        validation_precision=validation_precision_scores,

        validation_recall=validation_recall_scores,

        validation_tumor_dice=validation_tumor_dice_scores,

        path=FINAL_MODEL_PATH,
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "MEDIUM TRAINING COMPLETE"
    )

    print("=" * 70)


    print(
        "\nEpoch results:"
    )


    for index in range(
        final_epoch
    ):

        epoch_number = index + 1


        print(
            f"\nEpoch {epoch_number}"
        )

        print(
            "  Train loss:",
            f"{train_losses[index]:.6f}"
        )

        print(
            "  Validation loss:",
            f"{validation_losses[index]:.6f}"
        )

        print(
            "  Multiclass Dice:",
            f"{validation_dice_scores[index]:.6f}"
        )

        print(
            "  Tumor Dice:",
            f"{validation_tumor_dice_scores[index]:.6f}"
        )

        print(
            "  Tumor Precision:",
            f"{validation_precision_scores[index]:.6f}"
        )

        print(
            "  Tumor Recall:",
            f"{validation_recall_scores[index]:.6f}"
        )


    print(
        "\n" + "-" * 70
    )


    print(
        "Best epoch:",
        best_epoch
    )

    print(
        "Best tumor Dice:",
        f"{best_tumor_dice:.6f}"
    )

    print(
        "Precision at best epoch:",
        f"{best_precision:.6f}"
    )


    print(
        "\nBest model saved:"
    )

    print(
        BEST_MODEL_PATH
    )


    print(
        "\nFinal model saved:"
    )

    print(
        FINAL_MODEL_PATH
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "MEDIUM TRAINING EXPERIMENT COMPLETE"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()