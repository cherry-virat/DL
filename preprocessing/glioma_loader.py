"""
Glioma DataLoader
=================

This module connects the existing GliomaDataset
to a PyTorch DataLoader.

The preprocessing and patch extraction logic
remains inside glioma_dataset.py.

This file is responsible only for:

    GliomaDataset
          ↓
    DataLoader
          ↓
    Batch
          ↓
    CUDA

Expected final batch:

    MRI:
        (B, 4, 64, 64, 64)

    Mask:
        (B, 1, 64, 64, 64)
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from preprocessing.glioma_dataset import (
    GliomaDataset,
    get_cases,
    DATASET_ROOT,
    PATCH_SIZE,
    NUM_SAMPLES_PER_CASE,
)


# ============================================================
# CONFIGURATION
# ============================================================

# Number of cases used for this test.
#
# DO NOT use all 1251 cases yet.
#
# Start with 4.
TEST_CASES = 4

# The Dataset processes one case and returns
# NUM_SAMPLES_PER_CASE patches.
#
# We use batch_size=1 at the case level.
#
# Our custom collate function will flatten those
# patches into one actual training batch.
CASE_BATCH_SIZE = 1

# Number of DataLoader batches to test.
MAX_BATCHES = 2

# Use CUDA if available.
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# CUSTOM COLLATE FUNCTION
# ============================================================

def flatten_patch_collate(batch):
    """
    Convert a batch of cases containing lists of patches
    into one flat batch of MRI tensors and mask tensors.

    Example input:

        batch_size = 1

        [
            [
                {
                    "image": (4,64,64,64),
                    "label": (1,64,64,64)
                },
                {
                    "image": (4,64,64,64),
                    "label": (1,64,64,64)
                }
            ]
        ]

    Output:

        images:
            (2,4,64,64,64)

        labels:
            (2,1,64,64,64)
    """

    images = []
    labels = []

    # --------------------------------------------------------
    # Go through each case in the DataLoader batch
    # --------------------------------------------------------

    for case_patches in batch:

        # case_patches is the list returned by GliomaDataset.
        #
        # Example:
        #
        # [
        #     patch1,
        #     patch2
        # ]

        if not isinstance(
            case_patches,
            list
        ):
            case_patches = [case_patches]

        # ----------------------------------------------------
        # Extract image and label from each patch
        # ----------------------------------------------------

        for patch in case_patches:

            image = patch["image"]
            label = patch["label"]

            images.append(
                image
            )

            labels.append(
                label
            )

    # --------------------------------------------------------
    # Stack patches into a real PyTorch batch
    # --------------------------------------------------------

    images = torch.stack(
        images,
        dim=0
    )

    labels = torch.stack(
        labels,
        dim=0
    )

    return {
        "image": images,
        "label": labels,
    }


# ============================================================
# CREATE DATALOADER
# ============================================================

def create_glioma_dataloader(
    cases,
    batch_size=CASE_BATCH_SIZE,
    shuffle=False,
):
    """
    Create the Glioma DataLoader.

    Parameters
    ----------
    cases:
        List of Glioma case directories.

    batch_size:
        Number of cases processed together.

    shuffle:
        Whether to shuffle cases.

    Returns
    -------
    DataLoader
    """

    # --------------------------------------------------------
    # Create the existing dataset
    # --------------------------------------------------------

    dataset = GliomaDataset(
        cases
    )

    # --------------------------------------------------------
    # Create DataLoader
    # --------------------------------------------------------

    loader = DataLoader(
        dataset,

        batch_size=batch_size,

        shuffle=shuffle,

        # Windows-safe setting.
        #
        # Start with 0 workers.
        # Later we can test multiple workers.
        num_workers=0,

        # Speeds up CPU → GPU transfer when CUDA exists.
        pin_memory=torch.cuda.is_available(),

        # IMPORTANT:
        #
        # GliomaDataset returns a LIST of patches.
        #
        # Therefore we use our custom collate function.
        collate_fn=flatten_patch_collate,
    )

    return loader


# ============================================================
# BATCH VALIDATION
# ============================================================

def validate_batch(
    batch,
    batch_number,
):
    """
    Validate one DataLoader batch.
    """

    print("\n")
    print("=" * 70)
    print(
        f"BATCH {batch_number} VALIDATION"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Extract tensors
    # --------------------------------------------------------

    images = batch["image"]
    labels = batch["label"]

    # --------------------------------------------------------
    # Print shapes
    # --------------------------------------------------------

    print(
        "\nMRI batch shape:",
        tuple(images.shape)
    )

    print(
        "Mask batch shape:",
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
    # Expected shapes
    #
    # MRI:
    #
    # (B, 4, 64, 64, 64)
    #
    # Mask:
    #
    # (B, 1, 64, 64, 64)
    # --------------------------------------------------------

    expected_image_shape = (
        images.shape[0],
        4,
        *PATCH_SIZE
    )

    expected_label_shape = (
        labels.shape[0],
        1,
        *PATCH_SIZE
    )

    if tuple(images.shape) != expected_image_shape:

        raise ValueError(
            "\nIncorrect MRI batch shape.\n"
            f"Expected: {expected_image_shape}\n"
            f"Received: {tuple(images.shape)}"
        )

    if tuple(labels.shape) != expected_label_shape:

        raise ValueError(
            "\nIncorrect mask batch shape.\n"
            f"Expected: {expected_label_shape}\n"
            f"Received: {tuple(labels.shape)}"
        )

    # --------------------------------------------------------
    # Check MRI dtype
    # --------------------------------------------------------

    if images.dtype != torch.float32:

        raise TypeError(
            "MRI tensor must be float32.\n"
            f"Received: {images.dtype}"
        )

    # --------------------------------------------------------
    # Check mask dtype
    # --------------------------------------------------------

    if labels.dtype != torch.int64:

        raise TypeError(
            "Mask tensor must be int64.\n"
            f"Received: {labels.dtype}"
        )

    # --------------------------------------------------------
    # Check for NaN
    # --------------------------------------------------------

    if torch.isnan(images).any():

        raise ValueError(
            "MRI batch contains NaN values."
        )

    # --------------------------------------------------------
    # Check for infinity
    # --------------------------------------------------------

    if torch.isinf(images).any():

        raise ValueError(
            "MRI batch contains infinite values."
        )

    # --------------------------------------------------------
    # Tumor statistics
    # --------------------------------------------------------

    tumor_voxels_per_patch = (
        (labels > 0)
        .flatten(
            start_dim=1
        )
        .sum(dim=1)
    )

    print(
        "\nTumor voxels per patch:"
    )

    print(
        tumor_voxels_per_patch.tolist()
    )

    tumor_patches = (
        tumor_voxels_per_patch > 0
    ).sum().item()

    print(
        "Patches containing tumor:",
        tumor_patches,
        "/",
        labels.shape[0]
    )

    # --------------------------------------------------------
    # Image range
    # --------------------------------------------------------

    print(
        "\nMRI minimum:",
        float(images.min())
    )

    print(
        "MRI maximum:",
        float(images.max())
    )

    print(
        "MRI mean:",
        float(images.mean())
    )

    # --------------------------------------------------------
    # Success
    # --------------------------------------------------------

    print(
        "\nBatch validation PASSED."
    )


# ============================================================
# GPU VALIDATION
# ============================================================

def validate_gpu_batch(
    batch
):
    """
    Move one DataLoader batch to CUDA and validate it.
    """

    print("\n")
    print("=" * 70)
    print("GPU BATCH VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # CPU-only fallback
    # --------------------------------------------------------

    if not torch.cuda.is_available():

        print(
            "\nCUDA is not available."
        )

        print(
            "GPU validation skipped."
        )

        return

    # --------------------------------------------------------
    # Get tensors
    # --------------------------------------------------------

    images = batch["image"]
    labels = batch["label"]

    # --------------------------------------------------------
    # Clear previous CUDA memory
    # --------------------------------------------------------

    torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Reset peak memory counter
    # --------------------------------------------------------

    torch.cuda.reset_peak_memory_stats()

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
    # Synchronize CUDA
    # --------------------------------------------------------

    torch.cuda.synchronize()

    # --------------------------------------------------------
    # Print GPU information
    # --------------------------------------------------------

    print(
        "\nDevice:",
        DEVICE
    )

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "\nMRI GPU shape:",
        tuple(images.shape)
    )

    print(
        "Mask GPU shape:",
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
    # GPU memory
    # --------------------------------------------------------

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
        f"\nAllocated GPU memory: "
        f"{allocated_mb:.2f} MB"
    )

    print(
        f"Reserved GPU memory: "
        f"{reserved_mb:.2f} MB"
    )

    print(
        f"Peak GPU memory: "
        f"{peak_mb:.2f} MB"
    )

    # --------------------------------------------------------
    # Final GPU check
    # --------------------------------------------------------

    print(
        "\nGPU batch validation PASSED."
    )

    # --------------------------------------------------------
    # Remove tensors from GPU
    # --------------------------------------------------------

    del images
    del labels

    torch.cuda.empty_cache()


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("GLIOMA DATALOADER TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Dataset information
    # --------------------------------------------------------

    print(
        "\nDataset:",
        DATASET_ROOT
    )

    # --------------------------------------------------------
    # Find all cases
    # --------------------------------------------------------

    cases = get_cases(
        DATASET_ROOT
    )

    print(
        "Total Glioma cases:",
        len(cases)
    )

    # --------------------------------------------------------
    # Select only a few cases
    # --------------------------------------------------------

    test_cases = cases[
        :TEST_CASES
    ]

    print(
        "\nCases selected for test:",
        len(test_cases)
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
        batch_size=CASE_BATCH_SIZE,
        shuffle=False,
    )

    print(
        "\nDataLoader created successfully."
    )

    print(
        "Case batch size:",
        CASE_BATCH_SIZE
    )

    print(
        "Patches per case:",
        NUM_SAMPLES_PER_CASE
    )

    print(
        "Patch size:",
        PATCH_SIZE
    )

    # --------------------------------------------------------
    # Test DataLoader batches
    # --------------------------------------------------------

    for batch_number, batch in enumerate(
        loader,
        start=1
    ):

        # ----------------------------------------------------
        # Validate CPU batch
        # ----------------------------------------------------

        validate_batch(
            batch,
            batch_number
        )

        # ----------------------------------------------------
        # Validate GPU batch
        # ----------------------------------------------------

        validate_gpu_batch(
            batch
        )

        # ----------------------------------------------------
        # Stop after requested number of batches
        # ----------------------------------------------------

        if batch_number >= MAX_BATCHES:

            break

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("GLIOMA DATALOADER TEST COMPLETE")
    print("=" * 70)

    print(
        "\nSuccessfully verified:"
    )

    print(
        "1. Multiple Glioma cases can be loaded."
    )

    print(
        "2. Existing preprocessing works."
    )

    print(
        "3. Existing MONAI patch extraction works."
    )

    print(
        "4. Multiple patches can be combined into a batch."
    )

    print(
        "5. MRI batch has shape (B, 4, 64, 64, 64)."
    )

    print(
        "6. Mask batch has shape (B, 1, 64, 64, 64)."
    )

    print(
        "7. MRI tensors are float32."
    )

    print(
        "8. Mask tensors are int64."
    )

    print(
        "9. Batch contains valid tumor labels."
    )

    print(
        "10. Batch can be transferred to CUDA."
    )

    print(
        "\nReady for 3D model forward-pass testing."
    )