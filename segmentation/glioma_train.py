"""
Glioma 3D U-Net Small-Scale Training
=====================================

Purpose
-------
First real training experiment for Glioma segmentation.

Configuration
-------------
Training cases:
    20

Validation cases:
    5

Patches per case:
    2

Patch size:
    64 x 64 x 64

Batch size:
    1 case

Effective patch batch:
    2 patches

Input:
    4 MRI channels

Output:
    4 segmentation classes

Loss:
    Dice + Cross Entropy

Optimizer:
    Adam

Device:
    CUDA when available


Important
---------
GliomaDataset returns a LIST of patch dictionaries.

Therefore a custom collate function is used to convert:

    [
        [
            patch1,
            patch2
        ]
    ]

into:

    {
        "image": Tensor(2,4,64,64,64),
        "label": Tensor(2,1,64,64,64)
    }
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
    PATCH_SIZE,
    NUM_SAMPLES_PER_CASE,
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


NUM_EPOCHS = 5


LEARNING_RATE = 1e-4


NUM_WORKERS = 0


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


# ============================================================
# OUTPUT DIRECTORY
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
# REPRODUCIBILITY
# ============================================================

random.seed(
    SEED
)


torch.manual_seed(
    SEED
)


if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        SEED
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
# CUSTOM COLLATE FUNCTION
# ============================================================

def glioma_collate_fn(
    batch,
):
    """
    Flatten case-level patch lists and stack patches.

    Input:

        [
            [
                {"image": ..., "label": ...},
                {"image": ..., "label": ...}
            ]
        ]

    Output:

        {
            "image": Tensor(
                2, 4, 64, 64, 64
            ),

            "label": Tensor(
                2, 1, 64, 64, 64
            )
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
                "GliomaDataset must return "
                "a list of patch dictionaries."
            )


        patches.extend(
            case_output
        )


    if len(patches) == 0:

        raise RuntimeError(
            "No patches returned."
        )


    images = []

    labels = []


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
                "Patch missing 'image'."
            )


        if "label" not in patch:

            raise KeyError(
                "Patch missing 'label'."
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


    # --------------------------------------------------------
    # Final batch validation
    # --------------------------------------------------------

    expected_image_shape = (
        len(images),
        INPUT_CHANNELS,
        *PATCH_SIZE,
    )


    expected_label_shape = (
        len(labels),
        1,
        *PATCH_SIZE,
    )


    if tuple(
        images.shape
    ) != expected_image_shape:

        raise RuntimeError(
            "Invalid image batch.\n"
            f"Expected: {expected_image_shape}\n"
            f"Received: {tuple(images.shape)}"
        )


    if tuple(
        labels.shape
    ) != expected_label_shape:

        raise RuntimeError(
            "Invalid label batch.\n"
            f"Expected: {expected_label_shape}\n"
            f"Received: {tuple(labels.shape)}"
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
        INPUT_CHANNELS,
    )

    print(
        "Output classes:",
        OUTPUT_CLASSES,
    )

    print(
        "Model channels:",
        MODEL_CHANNELS,
    )

    print(
        "Total parameters:",
        f"{total_parameters:,}",
    )

    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}",
    )

    print(
        "Approximate model size:",
        f"{model_size_mb:.2f} MB",
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


    total_loss = 0.0

    batch_count = 0


    print(
        "\n" + "=" * 70
    )

    print(
        f"TRAINING — EPOCH {epoch}/{NUM_EPOCHS}"
    )

    print(
        "=" * 70
    )


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
        # Validate labels
        # ----------------------------------------------------

        unique_labels = torch.unique(
            labels
        )


        if torch.any(
            unique_labels < 0
        ):

            raise RuntimeError(
                "Negative labels found."
            )


        if torch.any(
            unique_labels >= OUTPUT_CLASSES
        ):

            raise RuntimeError(
                "Invalid label detected:\n"
                f"{unique_labels.tolist()}"
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

        if tuple(
            outputs.shape
        ) != (
            images.shape[0],
            OUTPUT_CLASSES,
            *PATCH_SIZE,
        ):

            raise RuntimeError(
                "Unexpected model output shape.\n"
                f"Input : {tuple(images.shape)}\n"
                f"Output: {tuple(outputs.shape)}"
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
                f"Non-finite training loss: "
                f"{loss.item()}"
            )


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()


        # ----------------------------------------------------
        # Gradient validation
        # ----------------------------------------------------

        gradient_found = False

        maximum_gradient = 0.0


        for parameter in model.parameters():

            if parameter.grad is not None:

                gradient_found = True


                current_gradient = (
                    parameter.grad
                    .detach()
                    .abs()
                    .max()
                    .item()
                )


                maximum_gradient = max(
                    maximum_gradient,
                    current_gradient,
                )


        if not gradient_found:

            raise RuntimeError(
                "No gradients generated."
            )


        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------

        optimizer.step()


        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        total_loss += (
            loss.item()
        )


        batch_count += 1


        # ----------------------------------------------------
        # Print first batch
        # ----------------------------------------------------

        if batch_index == 1:

            print(
                "\nFirst training batch:"
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
                "Unique labels:",
                unique_labels.tolist(),
            )

            print(
                "Loss:",
                f"{loss.item():.6f}",
            )

            print(
                "Maximum gradient:",
                f"{maximum_gradient:.6f}",
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
                    f"{allocated:.2f} MB",
                )


                print(
                    "GPU reserved:",
                    f"{reserved:.2f} MB",
                )


    average_loss = (
        total_loss
        /
        max(
            batch_count,
            1,
        )
    )


    print(
        "\nTraining batches:",
        batch_count,
    )


    print(
        "Average training loss:",
        f"{average_loss:.6f}",
    )


    return average_loss


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    loader,
    loss_function,
):

    model.eval()


    total_loss = 0.0

    batch_count = 0


    dice_metric = DiceMetric(
        include_background=False,
        reduction="mean",
    )


    with torch.no_grad():

        for batch in loader:

            images = batch[
                "image"
            ]


            labels = batch[
                "label"
            ]


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


            total_loss += (
                loss.item()
            )


            batch_count += 1


            # ------------------------------------------------
            # Predictions
            # ------------------------------------------------

            predictions = torch.argmax(
                outputs,
                dim=1,
            )


            # ------------------------------------------------
            # One-hot predictions
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
            # One-hot ground truth
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
        total_loss
        /
        max(
            batch_count,
            1,
        )
    )


    dice_score = (
        dice_metric
        .aggregate()
        .item()
    )


    dice_metric.reset()


    return (
        average_loss,
        dice_score,
    )


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    train_losses,
    validation_losses,
    validation_dice,
    path,
):

    torch.save(

        {

            "epoch":
                epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "train_losses":
                train_losses,

            "validation_losses":
                validation_losses,

            "validation_dice":
                validation_dice,

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

        },

        path,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "GLIOMA 3D U-NET SMALL-SCALE TRAINING"
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
    # SPLIT
    # ========================================================

    train_cases, validation_cases = (
        create_train_val_split(
            cases,
            train_ratio=0.90,
            seed=SEED,
        )
    )


    # ========================================================
    # SMALL EXPERIMENT
    # ========================================================

    train_cases = train_cases[
        :TRAIN_CASES_FOR_TEST
    ]


    validation_cases = validation_cases[
        :VAL_CASES_FOR_TEST
    ]


    print(
        "\nSmall-scale experiment:"
    )


    print(
        "Training cases:",
        len(train_cases),
    )


    print(
        "Validation cases:",
        len(validation_cases),
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
            "Training/validation overlap:\n"
            f"{sorted(overlap)}"
        )


    print(
        "✓ No train/validation overlap."
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

    print(
        "=" * 70
    )


    train_dataset = GliomaDataset(
        train_cases
    )


    validation_dataset = GliomaDataset(
        validation_cases
    )


    print(
        "Training dataset size:",
        len(train_dataset),
    )


    print(
        "Validation dataset size:",
        len(validation_dataset),
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
        "\nTraining DataLoader created."
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
        "CREATING 3D U-NET"
    )

    print(
        "=" * 70
    )


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

    print(
        "=" * 70
    )


    loss_function = DiceCELoss(

        to_onehot_y=True,

        softmax=True,
    )


    print(
        "Loss: DiceCELoss"
    )


    print(
        "Components: Dice + Cross Entropy"
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
        LEARNING_RATE,
    )


    # ========================================================
    # HISTORY
    # ========================================================

    train_losses = []

    validation_losses = []

    validation_dice_scores = []


    best_validation_dice = -1.0


    best_model_path = (
        OUTPUT_DIR
        /
        "glioma_best_small_experiment.pth"
    )


    final_model_path = (
        OUTPUT_DIR
        /
        "glioma_final_small_experiment.pth"
    )


    # ========================================================
    # TRAINING
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "STARTING TRAINING"
    )

    print(
        "=" * 70
    )


    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):

        print(
            f"\nEPOCH {epoch}/{NUM_EPOCHS}"
        )


        print(
            "-" * 70
        )


        # ----------------------------------------------------
        # Reset CUDA statistics
        # ----------------------------------------------------

        if torch.cuda.is_available():

            torch.cuda.reset_peak_memory_stats()


        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        training_loss = train_one_epoch(

            model=model,

            loader=train_loader,

            loss_function=loss_function,

            optimizer=optimizer,

            epoch=epoch,
        )


        train_losses.append(
            training_loss
        )


        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        print(
            "\n" + "-" * 70
        )

        print(
            "VALIDATION"
        )

        print(
            "-" * 70
        )


        validation_loss, dice_score = validate(

            model=model,

            loader=validation_loader,

            loss_function=loss_function,
        )


        validation_losses.append(
            validation_loss
        )


        validation_dice_scores.append(
            dice_score
        )


        print(
            "Validation loss:",
            f"{validation_loss:.6f}",
        )


        print(
            "Validation Dice:",
            f"{dice_score:.6f}",
        )


        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if dice_score > best_validation_dice:

            best_validation_dice = (
                dice_score
            )


            save_checkpoint(

                model=model,

                optimizer=optimizer,

                epoch=epoch,

                train_losses=train_losses,

                validation_losses=validation_losses,

                validation_dice=validation_dice_scores,

                path=best_model_path,
            )


            print(
                "✓ New best model saved."
            )


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
    # SAVE FINAL MODEL
    # ========================================================

    save_checkpoint(

        model=model,

        optimizer=optimizer,

        epoch=NUM_EPOCHS,

        train_losses=train_losses,

        validation_losses=validation_losses,

        validation_dice=validation_dice_scores,

        path=final_model_path,
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "SMALL-SCALE TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )


    print(
        "\nEpoch results:"
    )


    for index in range(
        NUM_EPOCHS
    ):

        epoch_number = (
            index + 1
        )


        print(
            f"\nEpoch {epoch_number}"
        )


        print(
            "  Train loss:",
            f"{train_losses[index]:.6f}",
        )


        print(
            "  Validation loss:",
            f"{validation_losses[index]:.6f}",
        )


        print(
            "  Validation Dice:",
            f"{validation_dice_scores[index]:.6f}",
        )


    print(
        "\n" + "-" * 70
    )


    print(
        "Best validation Dice:",
        f"{best_validation_dice:.6f}",
    )


    print(
        "\nBest model:"
    )


    print(
        best_model_path
    )


    print(
        "\nFinal model:"
    )


    print(
        final_model_path
    )


    print(
        "\n" + "=" * 70
    )


    print(
        "TRAINING EXPERIMENT PASSED"
    )


    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()