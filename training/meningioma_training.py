# ============================================================
# MENINGIOMA 3D U-NET TRAINING
# ============================================================
#
# Dataset:
#   BraTS2024-MEN-RT
#
# Input:
#   Preprocessed T1c MRI
#
# Task:
#   Binary meningioma GTV segmentation
#
# Classes:
#   0 = Background
#   1 = Tumor
#
# ============================================================


# ============================================================
# PYTHON PATH
# ============================================================

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from preprocessing.meningioma_dataset import (
    MeningiomaPatchDataset
)

from models.meningioma_model import (
    create_meningioma_model
)


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 2

NUM_WORKERS = 0

PIN_MEMORY = True

NUM_EPOCHS = 20

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-5

DICE_WEIGHT = 0.5

CE_WEIGHT = 0.5

GRADIENT_CLIP_NORM = 5.0

SEED = 42

OUTPUT_DIR = (
    PROJECT_ROOT /
    "outputs" /
    "meningioma_training"
)

BEST_CHECKPOINT = (
    OUTPUT_DIR /
    "meningioma_best.pth"
)

LAST_CHECKPOINT = (
    OUTPUT_DIR /
    "meningioma_last.pth"
)

HISTORY_FILE = (
    OUTPUT_DIR /
    "training_history.json"
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

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DICE LOSS
# ============================================================

class DiceLoss(nn.Module):

    def __init__(
        self,
        smooth=1e-5
    ):
        super().__init__()

        self.smooth = smooth

    def forward(
        self,
        logits,
        target
    ):

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        num_classes = logits.shape[1]

        target_one_hot = F.one_hot(
            target.long(),
            num_classes=num_classes
        )

        target_one_hot = (
            target_one_hot
            .permute(0, 4, 1, 2, 3)
            .float()
        )

        dimensions = (
            0,
            2,
            3,
            4
        )

        intersection = torch.sum(
            probabilities *
            target_one_hot,
            dim=dimensions
        )

        denominator = (
            torch.sum(
                probabilities,
                dim=dimensions
            )
            +
            torch.sum(
                target_one_hot,
                dim=dimensions
            )
        )

        dice = (
            2.0 * intersection
            +
            self.smooth
        ) / (
            denominator
            +
            self.smooth
        )

        return 1.0 - dice.mean()


# ============================================================
# DICE + CROSS ENTROPY
# ============================================================

class DiceCELoss(nn.Module):

    def __init__(
        self,
        dice_weight=0.5,
        ce_weight=0.5
    ):
        super().__init__()

        self.dice_weight = dice_weight

        self.ce_weight = ce_weight

        self.dice_loss = DiceLoss()

        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(
        self,
        logits,
        target
    ):

        dice = self.dice_loss(
            logits,
            target
        )

        ce = self.cross_entropy(
            logits,
            target
        )

        return (
            self.dice_weight * dice
            +
            self.ce_weight * ce
        )


# ============================================================
# BATCH DICE
# ============================================================

def calculate_tumor_dice(
    logits,
    target,
    smooth=1e-5
):
    """
    Calculate Dice only for the tumor class.

    Class 0 = background
    Class 1 = tumor
    """

    prediction = torch.argmax(
        logits,
        dim=1
    )

    prediction_tumor = (
        prediction == 1
    )

    target_tumor = (
        target == 1
    )

    intersection = torch.sum(
        prediction_tumor &
        target_tumor
    ).float()

    prediction_volume = torch.sum(
        prediction_tumor
    ).float()

    target_volume = torch.sum(
        target_tumor
    ).float()

    denominator = (
        prediction_volume
        +
        target_volume
    )

    # --------------------------------------------------------
    # If both prediction and target contain no tumor
    # --------------------------------------------------------

    if denominator == 0:

        return torch.tensor(
            1.0,
            device=logits.device
        )

    dice = (
        2.0 * intersection
        +
        smooth
    ) / (
        denominator
        +
        smooth
    )

    return dice


# ============================================================
# TRAINING EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0

    running_dice = 0.0

    batch_count = 0

    positive_batches = 0

    start_time = time.time()

    for batch_index, batch in enumerate(
        loader,
        start=1
    ):

        images = batch["image"]

        masks = batch["mask"]

        # ----------------------------------------------------
        # Move to GPU
        # ----------------------------------------------------

        images = images.to(
            device,
            non_blocking=True
        )

        masks = masks.to(
            device,
            non_blocking=True
        )

        # ----------------------------------------------------
        # Remove mask channel
        # ----------------------------------------------------

        target = masks.squeeze(1)

        # ----------------------------------------------------
        # Count tumor voxels
        # ----------------------------------------------------

        tumor_voxels = int(
            torch.sum(
                target > 0
            )
        )

        if tumor_voxels > 0:

            positive_batches += 1

        # ----------------------------------------------------
        # Clear gradients
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        logits = model(
            images
        )

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = criterion(
            logits,
            target
        )

        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        loss.backward()

        # ----------------------------------------------------
        # Gradient clipping
        # ----------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP_NORM
        )

        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------

        optimizer.step()

        # ----------------------------------------------------
        # Dice
        # ----------------------------------------------------

        with torch.no_grad():

            dice = calculate_tumor_dice(
                logits,
                target
            )

        # ----------------------------------------------------
        # Accumulate
        # ----------------------------------------------------

        running_loss += (
            loss.detach().item()
        )

        running_dice += (
            dice.detach().item()
        )

        batch_count += 1

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            batch_index == 1
            or
            batch_index % 50 == 0
            or
            batch_index == len(loader)
        ):

            print(
                f"    Batch "
                f"{batch_index:03d}/"
                f"{len(loader):03d} "
                f"| Loss: "
                f"{loss.detach().item():.4f} "
                f"| Dice: "
                f"{dice.detach().item():.4f}"
            )

        # ----------------------------------------------------
        # Cleanup
        # ----------------------------------------------------

        del images
        del masks
        del target
        del logits
        del loss
        del dice

    elapsed = (
        time.time()
        -
        start_time
    )

    return {
        "loss": running_loss / batch_count,
        "dice": running_dice / batch_count,
        "positive_batches": positive_batches,
        "total_batches": batch_count,
        "time_seconds": elapsed
    }


# ============================================================
# VALIDATION EPOCH
# ============================================================

def validate_one_epoch(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0

    running_dice = 0.0

    batch_count = 0

    positive_batches = 0

    start_time = time.time()

    with torch.no_grad():

        for batch_index, batch in enumerate(
            loader,
            start=1
        ):

            images = batch["image"]

            masks = batch["mask"]

            images = images.to(
                device,
                non_blocking=True
            )

            masks = masks.to(
                device,
                non_blocking=True
            )

            target = masks.squeeze(1)

            tumor_voxels = int(
                torch.sum(
                    target > 0
                )
            )

            if tumor_voxels > 0:

                positive_batches += 1

            logits = model(
                images
            )

            loss = criterion(
                logits,
                target
            )

            dice = calculate_tumor_dice(
                logits,
                target
            )

            running_loss += (
                loss.item()
            )

            running_dice += (
                dice.item()
            )

            batch_count += 1

            del images
            del masks
            del target
            del logits
            del loss
            del dice

    elapsed = (
        time.time()
        -
        start_time
    )

    return {
        "loss": running_loss / batch_count,
        "dice": running_dice / batch_count,
        "positive_batches": positive_batches,
        "total_batches": batch_count,
        "time_seconds": elapsed
    }


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    best_dice,
    history
):

    checkpoint = {

        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "best_dice":
            best_dice,

        "history":
            history,

        "configuration": {

            "batch_size":
                BATCH_SIZE,

            "num_epochs":
                NUM_EPOCHS,

            "learning_rate":
                LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY,

            "dice_weight":
                DICE_WEIGHT,

            "ce_weight":
                CE_WEIGHT,

            "patch_size":
                [64, 64, 64],

            "input_channels":
                1,

            "num_classes":
                2,

            "base_channels":
                16
        }
    }

    torch.save(
        checkpoint,
        path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("MENINGIOMA 3D U-NET TRAINING")
    print("=" * 100)

    print(
        "\nProject root:"
    )

    print(
        PROJECT_ROOT
    )

    print(
        "\nDevice:"
    )

    print(
        DEVICE
    )

    if DEVICE.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "CUDA:",
            torch.version.cuda
        )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "\nOutput directory:"
    )

    print(
        OUTPUT_DIR
    )

    # ========================================================
    # DATASETS
    # ========================================================

    print("\n" + "=" * 100)
    print("DATASETS")
    print("=" * 100)

    train_dataset = MeningiomaPatchDataset(
        split="train"
    )

    dev_dataset = MeningiomaPatchDataset(
        split="dev"
    )

    print(
        "\nTraining samples:",
        len(train_dataset)
    )

    print(
        "Development samples:",
        len(dev_dataset)
    )

    # ========================================================
    # DATALOADERS
    # ========================================================

    print("\n" + "=" * 100)
    print("DATALOADERS")
    print("=" * 100)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    print(
        "\nTraining batches:",
        len(train_loader)
    )

    print(
        "Development batches:",
        len(dev_loader)
    )

    # ========================================================
    # MODEL
    # ========================================================

    print("\n" + "=" * 100)
    print("MODEL")
    print("=" * 100)

    model = create_meningioma_model(
        base_channels=16
    )

    model = model.to(
        DEVICE
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        "\nTotal parameters:",
        f"{total_parameters:,}"
    )

    # ========================================================
    # LOSS
    # ========================================================

    print("\n" + "=" * 100)
    print("LOSS")
    print("=" * 100)

    criterion = DiceCELoss(
        dice_weight=DICE_WEIGHT,
        ce_weight=CE_WEIGHT
    )

    print(
        "\nDice weight:",
        DICE_WEIGHT
    )

    print(
        "Cross Entropy weight:",
        CE_WEIGHT
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    print("\n" + "=" * 100)
    print("OPTIMIZER")
    print("=" * 100)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    print(
        "\nOptimizer: AdamW"
    )

    print(
        "Learning rate:",
        LEARNING_RATE
    )

    print(
        "Weight decay:",
        WEIGHT_DECAY
    )

    # ========================================================
    # TRAINING HISTORY
    # ========================================================

    history = {

        "train_loss": [],

        "train_dice": [],

        "train_positive_batches": [],

        "train_total_batches": [],

        "dev_loss": [],

        "dev_dice": [],

        "dev_positive_batches": [],

        "dev_total_batches": [],

        "epoch_time_seconds": []
    }

    best_dev_dice = -1.0

    best_epoch = 0

    # ========================================================
    # TRAINING
    # ========================================================

    print("\n" + "=" * 100)
    print("TRAINING")
    print("=" * 100)

    print(
        f"\nTotal epochs: {NUM_EPOCHS}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    print()

    training_start = time.time()

    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):

        print("\n" + "-" * 100)

        print(
            f"EPOCH {epoch}/{NUM_EPOCHS}"
        )

        print("-" * 100)

        epoch_start = time.time()

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=DEVICE
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        print(
            "\n  Validation:"
        )

        dev_metrics = validate_one_epoch(
            model=model,
            loader=dev_loader,
            criterion=criterion,
            device=DEVICE
        )

        epoch_time = (
            time.time()
            -
            epoch_start
        )

        # ----------------------------------------------------
        # Store history
        # ----------------------------------------------------

        history["train_loss"].append(
            train_metrics["loss"]
        )

        history["train_dice"].append(
            train_metrics["dice"]
        )

        history["train_positive_batches"].append(
            train_metrics["positive_batches"]
        )

        history["train_total_batches"].append(
            train_metrics["total_batches"]
        )

        history["dev_loss"].append(
            dev_metrics["loss"]
        )

        history["dev_dice"].append(
            dev_metrics["dice"]
        )

        history["dev_positive_batches"].append(
            dev_metrics["positive_batches"]
        )

        history["dev_total_batches"].append(
            dev_metrics["total_batches"]
        )

        history["epoch_time_seconds"].append(
            epoch_time
        )

        # ----------------------------------------------------
        # Print summary
        # ----------------------------------------------------

        print(
            "\n  Epoch summary:"
        )

        print(
            f"    Train Loss : "
            f"{train_metrics['loss']:.6f}"
        )

        print(
            f"    Train Dice : "
            f"{train_metrics['dice']:.6f}"
        )

        print(
            f"    Dev Loss   : "
            f"{dev_metrics['loss']:.6f}"
        )

        print(
            f"    Dev Dice   : "
            f"{dev_metrics['dice']:.6f}"
        )

        print(
            f"    Train positive batches: "
            f"{train_metrics['positive_batches']}/"
            f"{train_metrics['total_batches']}"
        )

        print(
            f"    Dev positive batches: "
            f"{dev_metrics['positive_batches']}/"
            f"{dev_metrics['total_batches']}"
        )

        print(
            f"    Epoch time: "
            f"{epoch_time:.2f} seconds"
        )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        if dev_metrics["dice"] > best_dev_dice:

            best_dev_dice = (
                dev_metrics["dice"]
            )

            best_epoch = epoch

            save_checkpoint(
                path=BEST_CHECKPOINT,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_dice=best_dev_dice,
                history=history
            )

            print(
                "\n  ★ NEW BEST MODEL"
            )

            print(
                f"    Best dev Dice: "
                f"{best_dev_dice:.6f}"
            )

            print(
                f"    Saved to: "
                f"{BEST_CHECKPOINT}"
            )

        # ----------------------------------------------------
        # Last checkpoint
        # ----------------------------------------------------

        save_checkpoint(
            path=LAST_CHECKPOINT,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_dice=best_dev_dice,
            history=history
        )

        # ----------------------------------------------------
        # Save history
        # ----------------------------------------------------

        with open(
            HISTORY_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                history,
                file,
                indent=2
            )

        # ----------------------------------------------------
        # GPU memory
        # ----------------------------------------------------

        if DEVICE.type == "cuda":

            torch.cuda.synchronize()

            peak_memory_mb = (
                torch.cuda.max_memory_allocated(
                    DEVICE
                )
                /
                (1024 ** 2)
            )

            print(
                f"    Peak GPU memory: "
                f"{peak_memory_mb:.2f} MB"
            )

            torch.cuda.reset_peak_memory_stats(
                DEVICE
            )

    # ========================================================
    # TRAINING COMPLETE
    # ========================================================

    total_training_time = (
        time.time()
        -
        training_start
    )

    print("\n" + "=" * 100)
    print("MENINGIOMA TRAINING COMPLETE")
    print("=" * 100)

    print(
        "\nBest epoch:",
        best_epoch
    )

    print(
        "Best development Dice:",
        f"{best_dev_dice:.6f}"
    )

    print(
        "\nBest checkpoint:"
    )

    print(
        BEST_CHECKPOINT
    )

    print(
        "\nLast checkpoint:"
    )

    print(
        LAST_CHECKPOINT
    )

    print(
        "\nTraining history:"
    )

    print(
        HISTORY_FILE
    )

    print(
        "\nTotal training time:",
        f"{total_training_time / 60:.2f} minutes"
    )

    print("\n" + "=" * 100)
    print("NEXT STAGE")
    print("=" * 100)

    print(
        "\nMeningioma full-volume inference"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()