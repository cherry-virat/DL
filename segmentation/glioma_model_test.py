"""
Glioma 3D U-Net Forward-Pass Test
=================================

Purpose
-------
Test whether a real Glioma MRI batch from the existing
DataLoader can successfully pass through a 3D U-Net
on the RTX 3050 GPU.

This is NOT training.

Pipeline:

    MRI
      ↓
    Preprocessing
      ↓
    64x64x64 patches
      ↓
    DataLoader
      ↓
    3D U-Net
      ↓
    Segmentation logits

Expected input:

    (B, 4, 64, 64, 64)

Expected output:

    (B, 4, 64, 64, 64)

The four output channels correspond to the four
BraTS glioma segmentation classes.

"""

from pathlib import Path

import torch

from monai.networks.nets import UNet

from preprocessing.glioma_loader import (
    create_glioma_dataloader,
    get_cases,
    DATASET_ROOT,
)


# ============================================================
# CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ------------------------------------------------------------
# Test settings
# ------------------------------------------------------------

TEST_CASES = 2

MAX_BATCHES = 1


# ------------------------------------------------------------
# MRI configuration
# ------------------------------------------------------------

# Glioma uses:
#
# T1
# T1c
# T2
# FLAIR
#
# Therefore:
#
# input channels = 4
# ------------------------------------------------------------

IN_CHANNELS = 4


# ------------------------------------------------------------
# BraTS segmentation classes
# ------------------------------------------------------------

# BraTS Glioma segmentation labels:
#
# 0 = background
# 1 = NCR/NET
# 2 = ED
# 3 = ET
#
# Therefore the model predicts 4 classes.
# ------------------------------------------------------------

OUT_CHANNELS = 4


# ------------------------------------------------------------
# U-Net architecture
# ------------------------------------------------------------

MODEL_CHANNELS = (
    16,
    32,
    64,
    128,
    256,
)


# ============================================================
# CREATE 3D U-NET
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

    model = model.to(
        DEVICE
    )

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
        "Model channels:",
        MODEL_CHANNELS
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
# MODEL PARAMETER INFORMATION
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

    print(
        "Model size:",
        f"{total_parameters * 4 / 1024**2:.2f} MB"
    )


# ============================================================
# FORWARD PASS
# ============================================================

def run_forward_pass(
    model,
    batch,
):

    print("\n" + "=" * 70)
    print("3D U-NET FORWARD PASS")
    print("=" * 70)

    # --------------------------------------------------------
    # Extract MRI and label
    # --------------------------------------------------------

    images = batch["image"]

    labels = batch["label"]

    print(
        "\nInput MRI shape:",
        tuple(images.shape)
    )

    print(
        "Ground-truth mask shape:",
        tuple(labels.shape)
    )

    # --------------------------------------------------------
    # Move to GPU
    # --------------------------------------------------------

    images = images.to(
        DEVICE,
        non_blocking=True
    )

    labels = labels.to(
        DEVICE,
        non_blocking=True
    )

    print(
        "\nInput device:",
        images.device
    )

    # --------------------------------------------------------
    # Clear CUDA cache
    # --------------------------------------------------------

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats()

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # This is inference-only testing.
    #
    # No gradients are calculated.
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        output = model(
            images
        )

    # --------------------------------------------------------
    # Synchronize CUDA
    # --------------------------------------------------------

    if torch.cuda.is_available():

        torch.cuda.synchronize()

    # --------------------------------------------------------
    # Output information
    # --------------------------------------------------------

    print(
        "\nModel output shape:",
        tuple(output.shape)
    )

    print(
        "Model output dtype:",
        output.dtype
    )

    print(
        "Model output device:",
        output.device
    )

    # --------------------------------------------------------
    # Expected output
    # --------------------------------------------------------

    expected_shape = (
        images.shape[0],
        OUT_CHANNELS,
        images.shape[2],
        images.shape[3],
        images.shape[4],
    )

    if tuple(output.shape) != expected_shape:

        raise ValueError(
            "\nUnexpected model output shape.\n"
            f"Expected: {expected_shape}\n"
            f"Received: {tuple(output.shape)}"
        )

    # --------------------------------------------------------
    # Check NaN
    # --------------------------------------------------------

    if torch.isnan(output).any():

        raise ValueError(
            "Model output contains NaN values."
        )

    # --------------------------------------------------------
    # Check infinity
    # --------------------------------------------------------

    if torch.isinf(output).any():

        raise ValueError(
            "Model output contains infinite values."
        )

    # --------------------------------------------------------
    # Output statistics
    # --------------------------------------------------------

    print(
        "\nOutput minimum:",
        float(output.min())
    )

    print(
        "Output maximum:",
        float(output.max())
    )

    print(
        "Output mean:",
        float(output.mean())
    )

    # --------------------------------------------------------
    # Convert logits to predicted class
    # --------------------------------------------------------

    prediction = torch.argmax(
        output,
        dim=1,
    )

    print(
        "\nPrediction shape:",
        tuple(prediction.shape)
    )

    print(
        "Prediction dtype:",
        prediction.dtype
    )

    # --------------------------------------------------------
    # Predicted classes
    # --------------------------------------------------------

    unique_classes = torch.unique(
        prediction
    )

    print(
        "Predicted classes:",
        unique_classes.tolist()
    )

    # --------------------------------------------------------
    # GPU memory
    # --------------------------------------------------------

    if torch.cuda.is_available():

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

    # --------------------------------------------------------
    # Success
    # --------------------------------------------------------

    print(
        "\n3D U-Net forward pass PASSED."
    )

    return output


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("GLIOMA 3D U-NET FORWARD-PASS TEST")
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
    # Select small number of cases
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
        "\nDataLoader created."
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = create_model()

    # --------------------------------------------------------
    # Model information
    # --------------------------------------------------------

    print_model_information(
        model
    )

    # --------------------------------------------------------
    # Get one batch
    # --------------------------------------------------------

    for batch_number, batch in enumerate(
        loader,
        start=1
    ):

        print(
            "\nProcessing batch:",
            batch_number
        )

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        output = run_forward_pass(
            model,
            batch,
        )

        # ----------------------------------------------------
        # Stop after one batch
        # ----------------------------------------------------

        if batch_number >= MAX_BATCHES:

            break

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "GLIOMA 3D U-NET TEST COMPLETE"
    )
    print("=" * 70)

    print(
        "\nSuccessfully verified:"
    )

    print(
        "1. Glioma DataLoader"
    )

    print(
        "2. 4-channel MRI input"
    )

    print(
        "3. 64 x 64 x 64 3D patches"
    )

    print(
        "4. PyTorch batch creation"
    )

    print(
        "5. CUDA transfer"
    )

    print(
        "6. 3D U-Net creation"
    )

    print(
        "7. 3D U-Net forward pass"
    )

    print(
        "8. Segmentation output shape"
    )

    print(
        "9. GPU memory usage"
    )

    print(
        "\nNext stage: loss-function and training-step test."
    )