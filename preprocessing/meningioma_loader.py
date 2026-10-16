# ============================================================
# MENINGIOMA DATALOADER TEST
# ============================================================
#
# Purpose:
#   1. Load the Meningioma training dataset
#   2. Load the Meningioma development dataset
#   3. Create PyTorch DataLoaders
#   4. Verify batch shapes and dtypes
#   5. Verify tumor voxels are present
#   6. Verify CUDA/GPU transfer
#
# Run from project root:
#
#   python preprocessing\meningioma_loader.py
#
# ============================================================

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


# ============================================================
# PROJECT ROOT
# ============================================================

# This file is:
#
# DL/
# └── preprocessing/
#     └── meningioma_loader.py
#
# Therefore:
# parent      = preprocessing
# parent.parent = DL
#
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Make sure Python can find:
#
# preprocessing/
# data/
# models/
# segmentation/
# etc.
#
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORT
# ============================================================

from preprocessing.meningioma_dataset import (
    MeningiomaPatchDataset
)


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 2

NUM_WORKERS = 0

PIN_MEMORY = True


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("MENINGIOMA DATALOADER TEST")
    print("=" * 100)

    print("\nProject root:")
    print(PROJECT_ROOT)

    # --------------------------------------------------------
    # Check project root
    # --------------------------------------------------------

    if not PROJECT_ROOT.exists():

        raise FileNotFoundError(
            f"Project root does not exist:\n{PROJECT_ROOT}"
        )

    print("\nProject root check: OK")

    # --------------------------------------------------------
    # Check preprocessing package
    # --------------------------------------------------------

    preprocessing_dir = PROJECT_ROOT / "preprocessing"

    if not preprocessing_dir.exists():

        raise FileNotFoundError(
            f"Preprocessing directory not found:\n"
            f"{preprocessing_dir}"
        )

    print("Preprocessing directory check: OK")

    # --------------------------------------------------------
    # Check dataset file
    # --------------------------------------------------------

    dataset_file = (
        preprocessing_dir /
        "meningioma_dataset.py"
    )

    if not dataset_file.exists():

        raise FileNotFoundError(
            f"Meningioma dataset file not found:\n"
            f"{dataset_file}"
        )

    print("Meningioma dataset file check: OK")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print("\n" + "-" * 100)
    print("CREATING DATASETS")
    print("-" * 100)

    train_dataset = MeningiomaPatchDataset(
        split="train"
    )

    dev_dataset = MeningiomaPatchDataset(
        split="dev"
    )

    print("\nTraining dataset created.")

    print(
        "Training samples:",
        len(train_dataset)
    )

    print("\nDevelopment dataset created.")

    print(
        "Development samples:",
        len(dev_dataset)
    )

    # --------------------------------------------------------
    # Expected sizes
    # --------------------------------------------------------

    expected_train_samples = 400 * 2
    expected_dev_samples = 100 * 2

    print("\nExpected training samples:")
    print(expected_train_samples)

    print("Expected development samples:")
    print(expected_dev_samples)

    if len(train_dataset) != expected_train_samples:

        raise RuntimeError(
            "Unexpected training dataset size.\n"
            f"Expected: {expected_train_samples}\n"
            f"Found: {len(train_dataset)}"
        )

    if len(dev_dataset) != expected_dev_samples:

        raise RuntimeError(
            "Unexpected development dataset size.\n"
            f"Expected: {expected_dev_samples}\n"
            f"Found: {len(dev_dataset)}"
        )

    print("\nDataset size check: OK")

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    print("\n" + "-" * 100)
    print("CREATING DATALOADERS")
    print("-" * 100)

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

    print("\nDataLoaders created successfully.")

    print(
        "Training batches:",
        len(train_loader)
    )

    print(
        "Development batches:",
        len(dev_loader)
    )

    # --------------------------------------------------------
    # Expected number of batches
    # --------------------------------------------------------

    expected_train_batches = (
        expected_train_samples //
        BATCH_SIZE
    )

    expected_dev_batches = (
        expected_dev_samples //
        BATCH_SIZE
    )

    if len(train_loader) != expected_train_batches:

        raise RuntimeError(
            "Unexpected number of training batches.\n"
            f"Expected: {expected_train_batches}\n"
            f"Found: {len(train_loader)}"
        )

    if len(dev_loader) != expected_dev_batches:

        raise RuntimeError(
            "Unexpected number of development batches.\n"
            f"Expected: {expected_dev_batches}\n"
            f"Found: {len(dev_loader)}"
        )

    print("\nDataLoader batch-count check: OK")

    # --------------------------------------------------------
    # Get one training batch
    # --------------------------------------------------------

    print("\n" + "-" * 100)
    print("TRAINING BATCH TEST")
    print("-" * 100)

    train_batch = next(
        iter(train_loader)
    )

    # --------------------------------------------------------
    # Verify dictionary structure
    # --------------------------------------------------------

    if "image" not in train_batch:

        raise RuntimeError(
            "Training batch does not contain 'image'."
        )

    if "mask" not in train_batch:

        raise RuntimeError(
            "Training batch does not contain 'mask'."
        )

    train_images = train_batch["image"]

    train_masks = train_batch["mask"]

    print("\nTraining batch obtained successfully.")

    print(
        "Images shape:",
        tuple(train_images.shape)
    )

    print(
        "Masks shape:",
        tuple(train_masks.shape)
    )

    print(
        "Image dtype:",
        train_images.dtype
    )

    print(
        "Mask dtype:",
        train_masks.dtype
    )

    # --------------------------------------------------------
    # Expected shapes
    # --------------------------------------------------------

    expected_image_shape = (
        BATCH_SIZE,
        1,
        64,
        64,
        64
    )

    expected_mask_shape = (
        BATCH_SIZE,
        1,
        64,
        64,
        64
    )

    if tuple(train_images.shape) != expected_image_shape:

        raise RuntimeError(
            "Unexpected training image shape.\n"
            f"Expected: {expected_image_shape}\n"
            f"Found: {tuple(train_images.shape)}"
        )

    if tuple(train_masks.shape) != expected_mask_shape:

        raise RuntimeError(
            "Unexpected training mask shape.\n"
            f"Expected: {expected_mask_shape}\n"
            f"Found: {tuple(train_masks.shape)}"
        )

    # --------------------------------------------------------
    # Dtype checks
    # --------------------------------------------------------

    if train_images.dtype != torch.float32:

        raise RuntimeError(
            "Training image tensor must be float32.\n"
            f"Found: {train_images.dtype}"
        )

    if train_masks.dtype != torch.int64:

        raise RuntimeError(
            "Training mask tensor must be int64.\n"
            f"Found: {train_masks.dtype}"
        )

    print("\nShape check: OK")

    print("Image dtype check: OK")

    print("Mask dtype check: OK")

    # --------------------------------------------------------
    # Image statistics
    # --------------------------------------------------------

    image_min = float(
        train_images.min()
    )

    image_max = float(
        train_images.max()
    )

    image_mean = float(
        train_images.mean()
    )

    image_std = float(
        train_images.std()
    )

    print("\nTraining image statistics:")

    print(
        "  Min:",
        image_min
    )

    print(
        "  Max:",
        image_max
    )

    print(
        "  Mean:",
        image_mean
    )

    print(
        "  Std:",
        image_std
    )

    # --------------------------------------------------------
    # Check NaN / Inf
    # --------------------------------------------------------

    if torch.isnan(train_images).any():

        raise RuntimeError(
            "NaN detected in training images."
        )

    if torch.isinf(train_images).any():

        raise RuntimeError(
            "Inf detected in training images."
        )

    print("\nNaN check: OK")

    print("Inf check: OK")

    # --------------------------------------------------------
    # Mask statistics
    # --------------------------------------------------------

    tumor_voxels = int(
        torch.sum(
            train_masks > 0
        )
    )

    unique_mask_values = torch.unique(
        train_masks
    )

    print("\nTraining mask statistics:")

    print(
        "  Unique values:",
        unique_mask_values.tolist()
    )

    print(
        "  Tumor voxels:",
        tumor_voxels
    )

    # --------------------------------------------------------
    # Binary mask validation
    # --------------------------------------------------------

    valid_mask_values = set(
        unique_mask_values.tolist()
    )

    if not valid_mask_values.issubset(
        {0, 1}
    ):

        raise RuntimeError(
            "Mask contains values other than 0 and 1.\n"
            f"Found: {sorted(valid_mask_values)}"
        )

    print("\nBinary mask check: OK")

    # --------------------------------------------------------
    # GPU TEST
    # --------------------------------------------------------

    print("\n" + "-" * 100)
    print("GPU TEST")
    print("-" * 100)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\nDevice:")
    print(device)

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "CUDA version:",
            torch.version.cuda
        )

        # ----------------------------------------------------
        # Transfer image to GPU
        # ----------------------------------------------------

        train_images_gpu = train_images.to(
            device,
            non_blocking=True
        )

        # ----------------------------------------------------
        # Transfer mask to GPU
        # ----------------------------------------------------

        train_masks_gpu = train_masks.to(
            device,
            non_blocking=True
        )

        print(
            "\nGPU image shape:",
            tuple(train_images_gpu.shape)
        )

        print(
            "GPU mask shape:",
            tuple(train_masks_gpu.shape)
        )

        print(
            "GPU image device:",
            train_images_gpu.device
        )

        print(
            "GPU mask device:",
            train_masks_gpu.device
        )

        # ----------------------------------------------------
        # Verify device
        # ----------------------------------------------------

        if train_images_gpu.device.type != "cuda":

            raise RuntimeError(
                "Image was not transferred to CUDA."
            )

        if train_masks_gpu.device.type != "cuda":

            raise RuntimeError(
                "Mask was not transferred to CUDA."
            )

        print(
            "\nGPU transfer successful."
        )

        # ----------------------------------------------------
        # GPU tensor operation
        # ----------------------------------------------------

        gpu_test = (
            train_images_gpu.mean()
        )

        print(
            "GPU tensor operation successful."
        )

        print(
            "GPU mean:",
            float(gpu_test)
        )

        # ----------------------------------------------------
        # GPU memory
        # ----------------------------------------------------

        allocated_mb = (
            torch.cuda.memory_allocated(0)
            / (1024 ** 2)
        )

        reserved_mb = (
            torch.cuda.memory_reserved(0)
            / (1024 ** 2)
        )

        print(
            "\nGPU memory allocated:",
            f"{allocated_mb:.2f} MB"
        )

        print(
            "GPU memory reserved:",
            f"{reserved_mb:.2f} MB"
        )

        del train_images_gpu
        del train_masks_gpu
        del gpu_test

        torch.cuda.empty_cache()

        print(
            "\nGPU cache cleared."

        )

    else:

        print(
            "\nWARNING: CUDA is not available."
        )

    # --------------------------------------------------------
    # Development batch
    # --------------------------------------------------------

    print("\n" + "-" * 100)
    print("DEVELOPMENT BATCH TEST")
    print("-" * 100)

    dev_batch = next(
        iter(dev_loader)
    )

    if "image" not in dev_batch:

        raise RuntimeError(
            "Development batch does not contain 'image'."
        )

    if "mask" not in dev_batch:

        raise RuntimeError(
            "Development batch does not contain 'mask'."
        )

    dev_images = dev_batch["image"]

    dev_masks = dev_batch["mask"]

    print("\nDevelopment batch obtained successfully.")

    print(
        "Images shape:",
        tuple(dev_images.shape)
    )

    print(
        "Masks shape:",
        tuple(dev_masks.shape)
    )

    print(
        "Image dtype:",
        dev_images.dtype
    )

    print(
        "Mask dtype:",
        dev_masks.dtype
    )

    if tuple(dev_images.shape) != expected_image_shape:

        raise RuntimeError(
            "Unexpected development image shape.\n"
            f"Expected: {expected_image_shape}\n"
            f"Found: {tuple(dev_images.shape)}"
        )

    if tuple(dev_masks.shape) != expected_mask_shape:

        raise RuntimeError(
            "Unexpected development mask shape.\n"
            f"Expected: {expected_mask_shape}\n"
            f"Found: {tuple(dev_masks.shape)}"
        )

    print("\nDevelopment shape check: OK")

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print("\n" + "=" * 100)
    print("MENINGIOMA DATALOADER TEST COMPLETE")
    print("=" * 100)

    print("\nRESULT:")
    print("  Dataset loading      : PASS")
    print("  Train DataLoader     : PASS")
    print("  Dev DataLoader       : PASS")
    print("  Patch shape          : PASS")
    print("  Tensor dtype         : PASS")
    print("  Binary mask          : PASS")
    print("  NaN / Inf check      : PASS")

    if device.type == "cuda":
        print("  CUDA transfer        : PASS")
        print("  GPU tensor operation : PASS")
    else:
        print("  CUDA transfer        : NOT TESTED")

    print("\nNext stage:")
    print("  Meningioma 3D U-Net model + forward-pass test")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()