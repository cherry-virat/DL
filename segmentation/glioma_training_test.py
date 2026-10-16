"""
Glioma 3D U-Net Training-Step Test
===================================

Purpose
-------
Validate ONE complete training step using a real Glioma
MRI batch.

This is NOT full training.

It verifies:

    DataLoader
        ↓
    MRI batch
        ↓
    3D U-Net
        ↓
    segmentation logits
        ↓
    Dice + Cross Entropy loss
        ↓
    backpropagation
        ↓
    optimizer.step()
        ↓
    updated model weights

Expected input:

    (B, 4, 64, 64, 64)

Expected target:

    (B, 1, 64, 64, 64)

Expected model output:

    (B, 4, 64, 64, 64)
"""

# ============================================================
# IMPORTS
# ============================================================

import torch

from monai.losses import DiceCELoss
from monai.networks.nets import UNet
from torch.optim import Adam

from preprocessing.glioma_loader import (
    create_glioma_dataloader,
    get_cases,
    DATASET_ROOT,
)


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# Test only a small number of cases.
TEST_CASES = 2

# Only one batch for this test.
MAX_BATCHES = 1

# MRI:
# T1 + T1c + T2 + FLAIR
IN_CHANNELS = 4

# BraTS:
# 0 = background
# 1 = NCR/NET
# 2 = edema
# 3 = enhancing tumor
OUT_CHANNELS = 4

# Same architecture that already passed
# the forward-pass test.
MODEL_CHANNELS = (
    16,
    32,
    64,
    128,
    256,
)

# Learning rate for the test.
LEARNING_RATE = 1e-4


# ============================================================
# CREATE MODEL
# ============================================================

def create_model():

    print("=" * 70)
    print("CREATING 3D U-NET")
    print("=" * 70)

    model = UNet(
        spatial_dims=3,

        in_channels=IN_CHANNELS,

        out_channels=OUT_CHANNELS,

        channels=MODEL_CHANNELS,

        strides=(
            2,
            2,
            2,
            2,
        ),

        num_res_units=2,
    )

    model = model.to(DEVICE)

    print(
        "\nModel:",
        model.__class__.__name__
    )

    print(
        "Input channels:",
        IN_CHANNELS
    )

    print(
        "Output channels:",
        OUT_CHANNELS
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

    return model


# ============================================================
# MODEL INFORMATION
# ============================================================

def print_model_information(model):

    print("\n" + "=" * 70)
    print("MODEL INFORMATION")
    print("=" * 70)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        "\nTotal parameters:",
        f"{total_parameters:,}"
    )

    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}"
    )


# ============================================================
# CHECK TARGET LABELS
# ============================================================

def check_target_labels(labels):

    print("\n" + "=" * 70)
    print("TARGET LABEL VALIDATION")
    print("=" * 70)

    unique_labels = torch.unique(
        labels
    )

    print(
        "\nUnique labels:",
        unique_labels.tolist()
    )

    # --------------------------------------------------------
    # Verify labels are valid BraTS classes.
    # --------------------------------------------------------

    invalid = unique_labels[
        (unique_labels < 0)
        |
        (unique_labels >= OUT_CHANNELS)
    ]

    if invalid.numel() > 0:

        raise ValueError(
            "Invalid segmentation labels found: "
            f"{invalid.tolist()}"
        )

    print(
        "Target labels are valid."
    )


# ============================================================
# TRAINING STEP
# ============================================================

def run_training_step(
    model,
    optimizer,
    loss_function,
    batch,
):

    print("\n" + "=" * 70)
    print("SINGLE TRAINING STEP")
    print("=" * 70)

    # --------------------------------------------------------
    # Extract batch
    # --------------------------------------------------------

    images = batch["image"]

    labels = batch["label"]

    print(
        "\nOriginal MRI shape:",
        tuple(images.shape)
    )

    print(
        "Original mask shape:",
        tuple(labels.shape)
    )

    # --------------------------------------------------------
    # Move tensors to GPU
    # --------------------------------------------------------

    images = images.to(
        DEVICE,
        non_blocking=True
    )

    labels = labels.to(
        DEVICE,
        non_blocking=True
    )

    # --------------------------------------------------------
    # Target format
    #
    # DiceCELoss expects the segmentation target to have
    # a channel dimension.
    #
    # Current:
    #
    # (B, 1, D, H, W)
    #
    # which is correct.
    # --------------------------------------------------------

    print(
        "\nGPU MRI shape:",
        tuple(images.shape)
    )

    print(
        "GPU mask shape:",
        tuple(labels.shape)
    )

    print(
        "MRI device:",
        images.device
    )

    print(
        "Mask device:",
        labels.device
    )

    # --------------------------------------------------------
    # Validate target labels
    # --------------------------------------------------------

    check_target_labels(
        labels
    )

    # --------------------------------------------------------
    # Clear previous gradients
    # --------------------------------------------------------

    optimizer.zero_grad(
        set_to_none=True
    )

    # --------------------------------------------------------
    # Training mode
    # --------------------------------------------------------

    model.train()

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    print("\nRunning forward pass...")

    outputs = model(
        images
    )

    print(
        "Model output shape:",
        tuple(outputs.shape)
    )

    # --------------------------------------------------------
    # Validate output shape
    # --------------------------------------------------------

    expected_shape = (
        images.shape[0],
        OUT_CHANNELS,
        images.shape[2],
        images.shape[3],
        images.shape[4],
    )

    if tuple(outputs.shape) != expected_shape:

        raise ValueError(
            "\nIncorrect model output shape.\n"
            f"Expected: {expected_shape}\n"
            f"Received: {tuple(outputs.shape)}"
        )

    # --------------------------------------------------------
    # Check NaN
    # --------------------------------------------------------

    if torch.isnan(outputs).any():

        raise ValueError(
            "Model output contains NaN."
        )

    # --------------------------------------------------------
    # Calculate loss
    # --------------------------------------------------------

    print(
        "\nCalculating Dice + Cross Entropy loss..."
    )

    loss = loss_function(
        outputs,
        labels,
    )

    print(
        "Loss:",
        float(loss.item())
    )

    # --------------------------------------------------------
    # Validate loss
    # --------------------------------------------------------

    if torch.isnan(loss):

        raise ValueError(
            "Loss is NaN."
        )

    if torch.isinf(loss):

        raise ValueError(
            "Loss is infinite."
        )

    if loss.item() <= 0:

        raise ValueError(
            "Unexpected non-positive loss."
        )

    # --------------------------------------------------------
    # Backpropagation
    # --------------------------------------------------------

    print(
        "\nRunning backpropagation..."
    )

    loss.backward()

    print(
        "Backpropagation completed."
    )

    # --------------------------------------------------------
    # Check gradients
    # --------------------------------------------------------

    gradient_found = False

    maximum_gradient = 0.0

    for parameter in model.parameters():

        if parameter.grad is not None:

            gradient_found = True

            gradient_value = (
                parameter.grad.detach().abs().max().item()
            )

            maximum_gradient = max(
                maximum_gradient,
                gradient_value,
            )

    if not gradient_found:

        raise RuntimeError(
            "No gradients were generated."
        )

    print(
        "Gradients detected: YES"
    )

    print(
        "Maximum gradient:",
        maximum_gradient
    )

    # --------------------------------------------------------
    # Optimizer update
    # --------------------------------------------------------

    print(
        "\nRunning optimizer.step()..."
    )

    optimizer.step()

    print(
        "Optimizer step completed."
    )

    # --------------------------------------------------------
    # GPU memory
    # --------------------------------------------------------

    if torch.cuda.is_available():

        torch.cuda.synchronize()

        allocated_mb = (
            torch.cuda.memory_allocated()
            / 1024**2
        )

        reserved_mb = (
            torch.cuda.memory_reserved()
            / 1024**2
        )

        peak_mb = (
            torch.cuda.max_memory_allocated()
            / 1024**2
        )

        print(
            "\nGPU memory:"
        )

        print(
            f"Allocated: {allocated_mb:.2f} MB"
        )

        print(
            f"Reserved: {reserved_mb:.2f} MB"
        )

        print(
            f"Peak: {peak_mb:.2f} MB"
        )

    return loss.item()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("GLIOMA 3D U-NET TRAINING-STEP TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print(
        "\nDataset:",
        DATASET_ROOT
    )

    cases = get_cases(
        DATASET_ROOT
    )

    print(
        "Total Glioma cases:",
        len(cases)
    )

    # --------------------------------------------------------
    # Select test cases
    # --------------------------------------------------------

    test_cases = cases[
        :TEST_CASES
    ]

    print(
        "\nTesting cases:"
    )

    for index, case in enumerate(
        test_cases
    ):

        print(
            f"  {index + 1}. {case.name}"
        )

    # --------------------------------------------------------
    # Create DataLoader
    # --------------------------------------------------------

    loader = create_glioma_dataloader(
        cases=test_cases,

        batch_size=1,

        shuffle=False,
    )

    print(
        "\nDataLoader created successfully."
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = create_model()

    print_model_information(
        model
    )

    # --------------------------------------------------------
    # Loss function
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOSS FUNCTION")
    print("=" * 70)

    loss_function = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
    )

    print(
        "\nLoss:",
        "DiceCELoss"
    )

    print(
        "Dice + Cross Entropy"
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print(
        "\nOptimizer:",
        "Adam"
    )

    print(
        "Learning rate:",
        LEARNING_RATE
    )

    # --------------------------------------------------------
    # CUDA memory reset
    # --------------------------------------------------------

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats()

    # --------------------------------------------------------
    # Run one batch
    # --------------------------------------------------------

    training_steps = 0

    for batch_number, batch in enumerate(
        loader,
        start=1
    ):

        training_steps += 1

        print(
            "\nProcessing batch:",
            batch_number
        )

        loss_value = run_training_step(
            model=model,
            optimizer=optimizer,
            loss_function=loss_function,
            batch=batch,
        )

        if training_steps >= MAX_BATCHES:

            break

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING-STEP TEST COMPLETE")
    print("=" * 70)

    print(
        "\nSuccessfully verified:"
    )

    print(
        "1. Real Glioma MRI batch"
    )

    print(
        "2. 4-channel MRI input"
    )

    print(
        "3. 3D U-Net forward pass"
    )

    print(
        "4. Multi-class segmentation output"
    )

    print(
        "5. Dice + Cross Entropy loss"
    )

    print(
        "6. Loss calculation"
    )

    print(
        "7. Backpropagation"
    )

    print(
        "8. Gradient generation"
    )

    print(
        "9. Adam optimizer"
    )

    print(
        "10. GPU training-step execution"
    )

    print(
        "\nFinal loss:",
        loss_value
    )

    print(
        "\nNext stage:"
    )

    print(
        "Small-scale training experiment."
    )