# ============================================================
# MENINGIOMA TRAINING PIPELINE TEST
# ============================================================
#
# Purpose:
#   1. Verify positive and negative patches
#   2. Verify DataLoader sampling
#   3. Verify model forward pass
#   4. Verify Dice + Cross Entropy loss
#   5. Verify backward pass
#   6. Verify optimizer step
#   7. Verify GPU memory usage
#
# IMPORTANT:
#   This is NOT full training.
#
#   It performs only a small number of tests to make sure
#   the complete training pipeline is technically correct.
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

LEARNING_RATE = 1e-4

POSITIVE_PATCH_TARGET = 10

TOTAL_PATCHES_TO_CHECK = 20


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# DICE LOSS
# ============================================================

class DiceLoss(nn.Module):
    """
    Multi-class Dice loss.

    Expected:

        prediction:
            (B, C, D, H, W)

        target:
            (B, D, H, W)

    where:

        C = number of segmentation classes
        class 0 = background
        class 1 = tumor
    """

    def __init__(
        self,
        smooth: float = 1e-5
    ):
        super().__init__()

        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:

        # ----------------------------------------------------
        # Convert logits to probabilities
        # ----------------------------------------------------

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        # ----------------------------------------------------
        # One-hot encode target
        # ----------------------------------------------------

        num_classes = logits.shape[1]

        target_one_hot = F.one_hot(
            target.long(),
            num_classes=num_classes
        )

        # Current shape:
        #
        # (B, D, H, W, C)
        #
        # Convert to:
        #
        # (B, C, D, H, W)

        target_one_hot = (
            target_one_hot
            .permute(0, 4, 1, 2, 3)
            .float()
        )

        # ----------------------------------------------------
        # Calculate Dice
        # ----------------------------------------------------

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
            (
                2.0 * intersection
                +
                self.smooth
            )
            /
            (
                denominator
                +
                self.smooth
            )
        )

        # ----------------------------------------------------
        # Average over classes
        # ----------------------------------------------------

        dice_loss = 1.0 - dice.mean()

        return dice_loss


# ============================================================
# DICE + CROSS ENTROPY LOSS
# ============================================================

class DiceCELoss(nn.Module):
    """
    Combined Dice + Cross Entropy loss.

    This is the initial loss function for the Meningioma
    binary segmentation model.
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5
    ):
        super().__init__()

        self.dice_weight = dice_weight

        self.ce_weight = ce_weight

        self.dice_loss = DiceLoss()

        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:

        dice = self.dice_loss(
            logits,
            target
        )

        ce = self.cross_entropy(
            logits,
            target
        )

        total_loss = (
            self.dice_weight * dice
            +
            self.ce_weight * ce
        )

        return total_loss


# ============================================================
# POSITIVE PATCH CHECK
# ============================================================

def check_dataset_sampling(
    dataset,
    name: str
):

    print("\n" + "-" * 100)
    print(f"{name.upper()} POSITIVE/NEGATIVE PATCH CHECK")
    print("-" * 100)

    positive_count = 0
    negative_count = 0

    tumor_voxel_counts = []

    for index in range(
        min(
            TOTAL_PATCHES_TO_CHECK,
            len(dataset)
        )
    ):

        sample = dataset[index]

        image = sample["image"]

        mask = sample["mask"]

        tumor_voxels = int(
            torch.sum(
                mask > 0
            )
        )

        tumor_voxel_counts.append(
            tumor_voxels
        )

        if tumor_voxels > 0:

            positive_count += 1

        else:

            negative_count += 1

    print(
        "\nPatches checked:",
        len(tumor_voxel_counts)
    )

    print(
        "Positive patches:",
        positive_count
    )

    print(
        "Negative patches:",
        negative_count
    )

    print(
        "Tumor voxels per patch:"
    )

    print(
        tumor_voxel_counts
    )

    # --------------------------------------------------------
    # Verify that at least one positive patch exists
    # --------------------------------------------------------

    if positive_count == 0:

        raise RuntimeError(
            f"No positive tumor patches were found in "
            f"the first {TOTAL_PATCHES_TO_CHECK} "
            f"{name} samples."
        )

    print(
        "\nPositive patch check: PASS"
    )

    # --------------------------------------------------------
    # Verify that at least one negative patch exists
    # --------------------------------------------------------

    if negative_count == 0:

        print(
            "\nWARNING:"
        )

        print(
            "No negative patches found in this sample."
        )

    else:

        print(
            "Negative patch check: PASS"
        )

    return positive_count, negative_count


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("MENINGIOMA TRAINING PIPELINE TEST")
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

    # ========================================================
    # DATASET
    # ========================================================

    print("\n" + "=" * 100)
    print("1. DATASET")
    print("=" * 100)

    train_dataset = MeningiomaPatchDataset(
        split="train"
    )

    print(
        "\nTraining samples:",
        len(train_dataset)
    )

    # ========================================================
    # POSITIVE / NEGATIVE CHECK
    # ========================================================

    positive_count, negative_count = (
        check_dataset_sampling(
            train_dataset,
            "training"
        )
    )

    # ========================================================
    # DATALOADER
    # ========================================================

    print("\n" + "=" * 100)
    print("2. DATALOADER")
    print("=" * 100)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    print(
        "\nTraining batches:",
        len(train_loader)
    )

    # ========================================================
    # SEARCH FOR A BATCH WITH TUMOR
    # ========================================================

    print("\n" + "=" * 100)
    print("3. SEARCHING FOR POSITIVE TRAINING BATCH")
    print("=" * 100)

    positive_batch = None

    batches_checked = 0

    max_batches_to_search = min(
        20,
        len(train_loader)
    )

    for batch in train_loader:

        batches_checked += 1

        batch_tumor_voxels = int(
            torch.sum(
                batch["mask"] > 0
            )
        )

        print(
            f"Batch {batches_checked:02d}: "
            f"tumor voxels = "
            f"{batch_tumor_voxels}"
        )

        if batch_tumor_voxels > 0:

            positive_batch = batch

            break

        if batches_checked >= max_batches_to_search:

            break

    if positive_batch is None:

        raise RuntimeError(
            "Could not find a training batch containing "
            "tumor voxels."
        )

    print(
        "\nPositive training batch found."
    )

    # ========================================================
    # BATCH INFORMATION
    # ========================================================

    train_images = positive_batch["image"]

    train_masks = positive_batch["mask"]

    print(
        "\nImages shape:",
        tuple(train_images.shape)
    )

    print(
        "Masks shape:",
        tuple(train_masks.shape)
    )

    print(
        "Images dtype:",
        train_images.dtype
    )

    print(
        "Masks dtype:",
        train_masks.dtype
    )

    batch_tumor_voxels = int(
        torch.sum(
            train_masks > 0
        )
    )

    print(
        "Tumor voxels:",
        batch_tumor_voxels
    )

    # ========================================================
    # MODEL
    # ========================================================

    print("\n" + "=" * 100)
    print("4. MODEL")
    print("=" * 100)

    model = create_meningioma_model(
        base_channels=16
    )

    model = model.to(
        DEVICE
    )

    print(
        "\nModel moved to:",
        DEVICE
    )

    # ========================================================
    # LOSS
    # ========================================================

    print("\n" + "=" * 100)
    print("5. LOSS FUNCTION")
    print("=" * 100)

    criterion = DiceCELoss(
        dice_weight=0.5,
        ce_weight=0.5
    )

    print(
        "\nLoss:",
        criterion.__class__.__name__
    )

    print(
        "Dice weight:",
        criterion.dice_weight
    )

    print(
        "Cross Entropy weight:",
        criterion.ce_weight
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    print("\n" + "=" * 100)
    print("6. OPTIMIZER")
    print("=" * 100)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-5
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
        1e-5
    )

    # ========================================================
    # MOVE BATCH TO GPU
    # ========================================================

    print("\n" + "=" * 100)
    print("7. MOVING BATCH TO DEVICE")
    print("=" * 100)

    train_images = train_images.to(
        DEVICE,
        non_blocking=True
    )

    train_masks = train_masks.to(
        DEVICE,
        non_blocking=True
    )

    # --------------------------------------------------------
    # Remove channel from target
    # --------------------------------------------------------
    #
    # Current:
    #
    # (B, 1, D, H, W)
    #
    # CrossEntropyLoss requires:
    #
    # (B, D, H, W)
    #
    # --------------------------------------------------------

    train_masks_target = (
        train_masks.squeeze(1)
    )

    print(
        "\nInput shape:",
        tuple(train_images.shape)
    )

    print(
        "Target shape:",
        tuple(train_masks_target.shape)
    )

    print(
        "Input device:",
        train_images.device
    )

    print(
        "Target device:",
        train_masks_target.device
    )

    # ========================================================
    # FORWARD PASS
    # ========================================================

    print("\n" + "=" * 100)
    print("8. FORWARD PASS")
    print("=" * 100)

    model.train()

    torch.cuda.empty_cache()

    if DEVICE.type == "cuda":

        torch.cuda.reset_peak_memory_stats(
            DEVICE
        )

    logits = model(
        train_images
    )

    print(
        "\nLogits shape:",
        tuple(logits.shape)
    )

    expected_logits_shape = (
        BATCH_SIZE,
        2,
        64,
        64,
        64
    )

    if tuple(logits.shape) != expected_logits_shape:

        raise RuntimeError(
            "Unexpected logits shape.\n"
            f"Expected: {expected_logits_shape}\n"
            f"Found: {tuple(logits.shape)}"
        )

    print(
        "Forward shape check: PASS"
    )

    if torch.isnan(logits).any():

        raise RuntimeError(
            "NaN detected in model logits."
        )

    if torch.isinf(logits).any():

        raise RuntimeError(
            "Inf detected in model logits."
        )

    print(
        "Logits finite check: PASS"
    )

    # ========================================================
    # LOSS
    # ========================================================

    print("\n" + "=" * 100)
    print("9. LOSS CALCULATION")
    print("=" * 100)

    loss = criterion(
        logits,
        train_masks_target
    )

    print(
        "\nLoss:",
        float(loss)
    )

    if torch.isnan(loss):

        raise RuntimeError(
            "Loss is NaN."
        )

    if torch.isinf(loss):

        raise RuntimeError(
            "Loss is Inf."
        )

    if not loss.requires_grad:

        raise RuntimeError(
            "Loss does not require gradients."
        )

    print(
        "\nLoss finite check: PASS"
    )

    print(
        "Gradient tracking check: PASS"
    )

    # ========================================================
    # BACKWARD
    # ========================================================

    print("\n" + "=" * 100)
    print("10. BACKWARD PASS")
    print("=" * 100)

    optimizer.zero_grad(
        set_to_none=True
    )

    loss.backward()

    print(
        "\nBackward pass completed."
    )

    # --------------------------------------------------------
    # Check gradients
    # --------------------------------------------------------

    parameters_with_gradients = 0

    parameters_with_invalid_gradients = 0

    total_gradient_norm = 0.0

    for parameter in model.parameters():

        if parameter.grad is None:

            continue

        parameters_with_gradients += 1

        if torch.isnan(
            parameter.grad
        ).any():

            parameters_with_invalid_gradients += 1

        if torch.isinf(
            parameter.grad
        ).any():

            parameters_with_invalid_gradients += 1

        gradient_norm = (
            parameter.grad.detach()
            .norm(2)
            .item()
        )

        total_gradient_norm += (
            gradient_norm ** 2
        )

    total_gradient_norm = (
        total_gradient_norm ** 0.5
    )

    print(
        "\nParameters with gradients:",
        parameters_with_gradients
    )

    print(
        "Parameters with invalid gradients:",
        parameters_with_invalid_gradients
    )

    print(
        "Total gradient norm:",
        total_gradient_norm
    )

    if parameters_with_gradients == 0:

        raise RuntimeError(
            "No model parameters received gradients."
        )

    if parameters_with_invalid_gradients > 0:

        raise RuntimeError(
            "Invalid NaN/Inf gradients detected."
        )

    print(
        "\nBackward gradient check: PASS"
    )

    # ========================================================
    # OPTIMIZER STEP
    # ========================================================

    print("\n" + "=" * 100)
    print("11. OPTIMIZER STEP")
    print("=" * 100)

    optimizer.step()

    print(
        "\nOptimizer step completed."
    )

    # ========================================================
    # GPU MEMORY
    # ========================================================

    if DEVICE.type == "cuda":

        torch.cuda.synchronize()

        allocated_mb = (
            torch.cuda.memory_allocated(
                DEVICE
            )
            /
            (1024 ** 2)
        )

        reserved_mb = (
            torch.cuda.memory_reserved(
                DEVICE
            )
            /
            (1024 ** 2)
        )

        peak_allocated_mb = (
            torch.cuda.max_memory_allocated(
                DEVICE
            )
            /
            (1024 ** 2)
        )

        peak_reserved_mb = (
            torch.cuda.max_memory_reserved(
                DEVICE
            )
            /
            (1024 ** 2)
        )

        print(
            "\nGPU memory after training step:"
        )

        print(
            "  Current allocated:",
            f"{allocated_mb:.2f} MB"
        )

        print(
            "  Current reserved:",
            f"{reserved_mb:.2f} MB"
        )

        print(
            "  Peak allocated:",
            f"{peak_allocated_mb:.2f} MB"
        )

        print(
            "  Peak reserved:",
            f"{peak_reserved_mb:.2f} MB"
        )

    # ========================================================
    # CLEANUP
    # ========================================================

    del logits
    del loss

    del train_images
    del train_masks
    del train_masks_target

    del optimizer
    del criterion
    del model

    if DEVICE.type == "cuda":

        torch.cuda.empty_cache()

        print(
            "\nGPU cache cleared."
        )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print("\n" + "=" * 100)
    print("MENINGIOMA TRAINING PIPELINE TEST COMPLETE")
    print("=" * 100)

    print("\nRESULT:")

    print(
        "  Positive patch sampling : PASS"
    )

    print(
        "  DataLoader              : PASS"
    )

    print(
        "  Model forward           : PASS"
    )

    print(
        "  Loss calculation        : PASS"
    )

    print(
        "  Backward pass           : PASS"
    )

    print(
        "  Gradient validation     : PASS"
    )

    print(
        "  Optimizer step          : PASS"
    )

    if DEVICE.type == "cuda":

        print(
            "  CUDA training step      : PASS"
        )

    print("\nNext stage:")

    print(
        "  Full Meningioma training"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()