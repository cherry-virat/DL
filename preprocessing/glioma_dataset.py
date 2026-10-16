"""
Glioma MONAI Dataset
====================

Purpose
-------
Prepare preprocessed Glioma MRI cases for 3D patch-based training
and full-volume inference.

Actual BraTS-Glioma filenames
-----------------------------
Each case contains:

    <case>-t1n.nii.gz
    <case>-t1c.nii.gz
    <case>-t2w.nii.gz
    <case>-t2f.nii.gz
    <case>-seg.nii.gz

Channel mapping
---------------
    T1n  -> T1
    T1c  -> T1 contrast-enhanced
    T2w  -> T2
    T2f  -> FLAIR

Pipeline
--------
Raw NIfTI case
    |
    v
glioma_preprocess.process_case()
    |
    v
4-channel MRI + segmentation
    |
    v
Validate image/mask alignment
    |
    v
Pad small volumes
    |
    v
MONAI tumor-aware random patch extraction
    |
    v
64 x 64 x 64 patches
    |
    v
PyTorch tensors

Full-volume support
-------------------
prepare_case() uses the SAME MRI files and normalization
logic as the training pipeline, but returns the complete
preprocessed volume instead of extracting patches.
"""


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset

from monai.transforms import (
    Compose,
    EnsureTyped,
    RandCropByPosNegLabeld,
)


# ============================================================
# DATASET PATH
# ============================================================

DATASET_ROOT = Path(
    r"C:\3D_DATASET\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
)


NESTED_DATASET_ROOT = (
    DATASET_ROOT
    / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
)


if NESTED_DATASET_ROOT.exists():
    DATASET_ROOT = NESTED_DATASET_ROOT


# ============================================================
# PATCH CONFIGURATION
# ============================================================

PATCH_SIZE = (
    64,
    64,
    64,
)

NUM_SAMPLES_PER_CASE = 2


# ============================================================
# FIND CASES
# ============================================================

def get_cases(dataset_root):
    """
    Find all Glioma case directories.
    """

    dataset_root = Path(dataset_root)

    if not dataset_root.exists():

        raise FileNotFoundError(
            "\nGlioma dataset directory does not exist:\n"
            f"{dataset_root}"
        )

    cases = sorted(
        [
            path
            for path in dataset_root.iterdir()
            if path.is_dir()
        ]
    )

    if len(cases) == 0:

        raise RuntimeError(
            "\nNo Glioma case directories found in:\n"
            f"{dataset_root}"
        )

    return cases


# ============================================================
# PAD SMALL VOLUMES
# ============================================================

def pad_to_patch_size(
    image,
    mask,
    patch_size=PATCH_SIZE,
):
    """
    Ensure spatial dimensions are at least PATCH_SIZE.

    image:
        (C, D, H, W)

    mask:
        (1, D, H, W)
    """

    if image.ndim != 4:

        raise ValueError(
            "Image must have shape (C,D,H,W).\n"
            f"Received: {image.shape}"
        )

    if mask.ndim != 4:

        raise ValueError(
            "Mask must have shape (1,D,H,W).\n"
            f"Received: {mask.shape}"
        )

    if image.shape[1:] != mask.shape[1:]:

        raise ValueError(
            "Image and mask spatial dimensions do not match.\n"
            f"Image: {image.shape}\n"
            f"Mask : {mask.shape}"
        )

    spatial_shape = image.shape[1:]

    padding = []

    for current_size, target_size in zip(
        spatial_shape,
        patch_size,
    ):

        total_padding = max(
            target_size - current_size,
            0,
        )

        before = total_padding // 2

        after = total_padding - before

        padding.append(
            (
                before,
                after,
            )
        )

    # --------------------------------------------------------
    # No padding required
    # --------------------------------------------------------

    if all(
        before == 0 and after == 0
        for before, after in padding
    ):

        return (
            image,
            mask,
        )

    # --------------------------------------------------------
    # Convert to tensors
    # --------------------------------------------------------

    image_tensor = torch.as_tensor(
        image,
        dtype=torch.float32,
    )

    mask_tensor = torch.as_tensor(
        mask,
        dtype=torch.int64,
    )

    # --------------------------------------------------------
    # F.pad order:
    #
    # W before, W after,
    # H before, H after,
    # D before, D after
    #
    # Therefore reverse D,H,W.
    # --------------------------------------------------------

    pad_values = []

    for before, after in reversed(
        padding
    ):

        pad_values.extend(
            [
                before,
                after,
            ]
        )

    # --------------------------------------------------------
    # Pad image
    # --------------------------------------------------------

    image_tensor = F.pad(
        image_tensor,
        pad_values,
        mode="constant",
        value=0,
    )

    # --------------------------------------------------------
    # Pad mask
    # --------------------------------------------------------

    mask_tensor = F.pad(
        mask_tensor,
        pad_values,
        mode="constant",
        value=0,
    )

    return (
        image_tensor.numpy(),
        mask_tensor.numpy(),
    )


# ============================================================
# MONAI PATCH TRANSFORM
# ============================================================

patch_transform = Compose(
    [

        RandCropByPosNegLabeld(

            keys=[
                "image",
                "label",
            ],

            label_key="label",

            spatial_size=PATCH_SIZE,

            pos=1,

            neg=1,

            num_samples=NUM_SAMPLES_PER_CASE,

            image_key="image",

            image_threshold=0,

            allow_smaller=False,
        ),

        EnsureTyped(

            keys=[
                "image",
                "label",
            ],

            dtype=(
                torch.float32,
                torch.int64,
            ),
        ),
    ]
)


# ============================================================
# FULL-VOLUME CASE PREPARATION
# ============================================================

def prepare_case(case_dir):
    """
    Prepare one complete Glioma case.

    This function is intended for full-volume inference.

    It does NOT perform random patch extraction.

    Returns
    -------
    image:
        torch.Tensor
        Shape:
            (4, D, H, W)

    mask:
        torch.Tensor
        Shape:
            (1, D, H, W)

    Actual BraTS filenames used:

        -t1n.nii.gz
        -t1c.nii.gz
        -t2w.nii.gz
        -t2f.nii.gz
        -seg.nii.gz
    """

    import nibabel as nib

    case_dir = Path(case_dir)

    if not case_dir.exists():

        raise FileNotFoundError(
            f"Case directory does not exist:\n{case_dir}"
        )

    print(
        "\n" + "-" * 70
    )

    print(
        "FULL-VOLUME CASE PREPARATION"
    )

    print(
        "-" * 70
    )

    print(
        "Case:",
        case_dir.name,
    )

    # ========================================================
    # FIND NIFTI FILES
    # ========================================================

    nifti_files = sorted(
        case_dir.glob("*.nii.gz")
    )

    if not nifti_files:

        raise FileNotFoundError(
            "No .nii.gz files found.\n"
            f"Case: {case_dir}"
        )

    print(
        "\nAvailable NIfTI files:"
    )

    for file_path in nifti_files:

        print(
            " ",
            file_path.name
        )

    # ========================================================
    # RESOLVE ACTUAL BRATS FILE NAMES
    # ========================================================

    prefix = case_dir.name

    modality_files = {

        "t1": case_dir / (
            f"{prefix}-t1n.nii.gz"
        ),

        "t1c": case_dir / (
            f"{prefix}-t1c.nii.gz"
        ),

        "t2": case_dir / (
            f"{prefix}-t2w.nii.gz"
        ),

        "flair": case_dir / (
            f"{prefix}-t2f.nii.gz"
        ),
    }

    segmentation_file = (
        case_dir
        /
        f"{prefix}-seg.nii.gz"
    )

    # ========================================================
    # VERIFY FILES
    # ========================================================

    missing_files = []

    for modality, file_path in modality_files.items():

        if not file_path.exists():

            missing_files.append(
                f"{modality}: {file_path.name}"
            )

    if not segmentation_file.exists():

        missing_files.append(
            f"seg: {segmentation_file.name}"
        )

    if missing_files:

        raise FileNotFoundError(
            "\nMissing required BraTS-Glioma files.\n"
            f"Case: {case_dir.name}\n\n"
            "Missing:\n"
            +
            "\n".join(
                f"  {item}"
                for item in missing_files
            )
            +
            "\n\nAvailable files:\n"
            +
            "\n".join(
                f"  {file.name}"
                for file in nifti_files
            )
        )

    # ========================================================
    # PRINT RESOLVED FILES
    # ========================================================

    print(
        "\nResolved MRI files:"
    )

    print(
        "  t1   :",
        modality_files["t1"].name,
    )

    print(
        "  t1c  :",
        modality_files["t1c"].name,
    )

    print(
        "  t2   :",
        modality_files["t2"].name,
    )

    print(
        "  flair:",
        modality_files["flair"].name,
    )

    print(
        "  seg  :",
        segmentation_file.name,
    )

    # ========================================================
    # LOAD MRI
    # ========================================================

    images = []

    for modality in (
        "t1",
        "t1c",
        "t2",
        "flair",
    ):

        nii = nib.load(
            str(
                modality_files[modality]
            )
        )

        image_data = nii.get_fdata(
            dtype=np.float32
        )

        images.append(
            image_data
        )

    # ========================================================
    # CHECK MRI SHAPES
    # ========================================================

    reference_shape = images[0].shape

    for modality, image_data in zip(
        (
            "t1",
            "t1c",
            "t2",
            "flair",
        ),
        images,
    ):

        if image_data.shape != reference_shape:

            raise RuntimeError(
                "\nMRI MODALITY SHAPE MISMATCH\n"
                f"Reference: {reference_shape}\n"
                f"{modality}: {image_data.shape}"
            )

    # ========================================================
    # STACK FOUR MRI CHANNELS
    # ========================================================

    image = np.stack(
        images,
        axis=0,
    ).astype(
        np.float32
    )

    # ========================================================
    # LOAD SEGMENTATION
    # ========================================================

    mask = nib.load(
        str(
            segmentation_file
        )
    ).get_fdata()

    mask = np.asarray(
        mask,
        dtype=np.int64,
    )

    # ========================================================
    # MASK SHAPE CHECK
    # ========================================================

    if mask.shape != reference_shape:

        raise RuntimeError(
            "\nSEGMENTATION SHAPE MISMATCH\n"
            f"MRI: {reference_shape}\n"
            f"Mask: {mask.shape}"
        )

    # ========================================================
    # MRI NORMALIZATION
    #
    # Same non-zero z-score normalization used by the
    # preprocessing pipeline.
    # ========================================================

    for channel_index in range(
        image.shape[0]
    ):

        channel_data = image[
            channel_index
        ]

        nonzero = (
            channel_data != 0
        )

        if np.any(nonzero):

            values = channel_data[
                nonzero
            ]

            mean = values.mean()

            std = values.std()

            if std > 0:

                channel_data[
                    nonzero
                ] = (
                    channel_data[
                        nonzero
                    ]
                    - mean
                ) / std

            else:

                channel_data[
                    nonzero
                ] = (
                    channel_data[
                        nonzero
                    ]
                    - mean
                )

        image[
            channel_index
        ] = channel_data

    # ========================================================
    # CLEAN SEGMENTATION LABELS
    # ========================================================

    mask = np.rint(
        mask
    ).astype(
        np.int64
    )

    # ========================================================
    # VALIDATE LABELS
    # ========================================================

    unique_labels = np.unique(
        mask
    )

    print(
        "\nSegmentation labels:",
        unique_labels.tolist()
    )

    if np.any(
        unique_labels < 0
    ):

        raise RuntimeError(
            "Negative segmentation labels detected:\n"
            f"{unique_labels}"
        )

    if np.any(
        unique_labels > 3
    ):

        raise RuntimeError(
            "Invalid Glioma segmentation labels detected.\n"
            f"Expected labels: [0, 1, 2, 3]\n"
            f"Found: {unique_labels.tolist()}"
        )

    # ========================================================
    # ADD MASK CHANNEL
    # ========================================================

    mask = np.expand_dims(
        mask,
        axis=0,
    )

    # ========================================================
    # CONVERT TO TORCH
    # ========================================================

    image_tensor = torch.from_numpy(
        image
    ).float()

    mask_tensor = torch.from_numpy(
        mask
    ).long()

    # ========================================================
    # VALIDATE TENSOR SHAPES
    # ========================================================

    if image_tensor.ndim != 4:

        raise RuntimeError(
            "MRI must have shape (4,D,H,W).\n"
            f"Received: {image_tensor.shape}"
        )

    if image_tensor.shape[0] != 4:

        raise RuntimeError(
            "MRI must contain exactly 4 channels.\n"
            f"Received: {image_tensor.shape}"
        )

    if mask_tensor.ndim != 4:

        raise RuntimeError(
            "Mask must have shape (1,D,H,W).\n"
            f"Received: {mask_tensor.shape}"
        )

    if mask_tensor.shape[0] != 1:

        raise RuntimeError(
            "Mask must have exactly one channel.\n"
            f"Received: {mask_tensor.shape}"
        )

    if image_tensor.shape[1:] != mask_tensor.shape[1:]:

        raise RuntimeError(
            "\nMRI/MASK SPATIAL MISMATCH\n"
            f"MRI : {image_tensor.shape}\n"
            f"Mask: {mask_tensor.shape}"
        )

    # ========================================================
    # PRINT ORIGINAL VOLUME
    # ========================================================

    print(
        "\nOriginal preprocessed volume:"
    )

    print(
        "Image:",
        tuple(image_tensor.shape)
    )

    print(
        "Label:",
        tuple(mask_tensor.shape)
    )

    print(
        "MRI dtype:",
        image_tensor.dtype
    )

    print(
        "Mask dtype:",
        mask_tensor.dtype
    )

    tumor_voxels = int(
        (
            mask_tensor > 0
        ).sum().item()
    )

    print(
        "Tumor voxels:",
        tumor_voxels
    )

    # ========================================================
    # PAD SMALL VOLUME
    # ========================================================

    original_spatial_shape = (
        tuple(
            image_tensor.shape[1:]
        )
    )

    image_tensor, mask_tensor = (
        pad_to_patch_size(
            image_tensor,
            mask_tensor,
            PATCH_SIZE,
        )
    )

    padded_spatial_shape = (
        tuple(
            image_tensor.shape[1:]
        )
    )

    # ========================================================
    # REPORT PADDING
    # ========================================================

    if (
        padded_spatial_shape
        !=
        original_spatial_shape
    ):

        print(
            "\nPadding applied:"
        )

        print(
            "Before:",
            original_spatial_shape
        )

        print(
            "After :",
            padded_spatial_shape
        )

    else:

        print(
            "Padding:",
            "Not required"
        )

    # ========================================================
    # FINAL SIZE CHECK
    # ========================================================

    for current_size, required_size in zip(
        image_tensor.shape[1:],
        PATCH_SIZE,
    ):

        if current_size < required_size:

            raise RuntimeError(
                "\nPADDING FAILED\n"
                f"Volume: {image_tensor.shape}\n"
                f"Patch : {PATCH_SIZE}"
            )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    if image_tensor.shape[1:] != mask_tensor.shape[1:]:

        raise RuntimeError(
            "\nFINAL MRI/MASK MISMATCH\n"
            f"MRI : {image_tensor.shape}\n"
            f"Mask: {mask_tensor.shape}"
        )

    return (
        image_tensor,
        mask_tensor,
    )


# ============================================================
# GLIOMA DATASET
# ============================================================

class GliomaDataset(
    Dataset
):

    def __init__(
        self,
        cases,
    ):

        self.cases = list(
            cases
        )

    # --------------------------------------------------------
    # Dataset length
    # --------------------------------------------------------

    def __len__(
        self,
    ):

        return len(
            self.cases
        )

    # --------------------------------------------------------
    # Get one case
    # --------------------------------------------------------

    def __getitem__(
        self,
        index,
    ):

        case_dir = self.cases[
            index
        ]

        # ----------------------------------------------------
        # Existing preprocessing
        # ----------------------------------------------------

        from preprocessing.glioma_preprocess import (
            process_case,
        )

        image, mask = process_case(
            case_dir
        )

        # ----------------------------------------------------
        # NumPy conversion
        # ----------------------------------------------------

        image = np.asarray(
            image,
            dtype=np.float32,
        )

        mask = np.asarray(
            mask,
            dtype=np.int64,
        )

        # ----------------------------------------------------
        # Add mask channel
        # ----------------------------------------------------

        if mask.ndim == 3:

            mask = np.expand_dims(
                mask,
                axis=0,
            )

        # ----------------------------------------------------
        # Image validation
        # ----------------------------------------------------

        if image.ndim != 4:

            raise ValueError(
                "Expected MRI shape (C,D,H,W).\n"
                f"Received: {image.shape}"
            )

        if image.shape[0] != 4:

            raise ValueError(
                "Glioma MRI must contain exactly 4 channels.\n"
                f"Received: {image.shape}"
            )

        # ----------------------------------------------------
        # Mask validation
        # ----------------------------------------------------

        if mask.ndim != 4:

            raise ValueError(
                "Expected mask shape (1,D,H,W).\n"
                f"Received: {mask.shape}"
            )

        if mask.shape[0] != 1:

            raise ValueError(
                "Glioma segmentation mask must have "
                "exactly one channel.\n"
                f"Received: {mask.shape}"
            )

        # ----------------------------------------------------
        # Spatial validation
        # ----------------------------------------------------

        if image.shape[1:] != mask.shape[1:]:

            raise ValueError(
                "\nIMAGE/MASK SPATIAL MISMATCH\n"
                f"Image: {image.shape}\n"
                f"Mask : {mask.shape}"
            )

        # ----------------------------------------------------
        # Label validation
        # ----------------------------------------------------

        unique_labels = np.unique(
            mask
        )

        if np.any(
            unique_labels < 0
        ):

            raise ValueError(
                "Negative segmentation labels found:\n"
                f"{unique_labels}"
            )

        if np.any(
            unique_labels > 3
        ):

            raise ValueError(
                "Unexpected segmentation labels:\n"
                f"{unique_labels}"
            )

        # ----------------------------------------------------
        # Tumor statistics
        # ----------------------------------------------------

        tumor_voxels = int(
            np.sum(
                mask > 0
            )
        )

        # ----------------------------------------------------
        # Case information
        # ----------------------------------------------------

        print(
            "\n" + "-" * 70
        )

        print(
            "CASE PREPARATION"
        )

        print(
            "-" * 70
        )

        print(
            "Case:",
            case_dir.name,
        )

        print(
            "MRI shape:",
            image.shape,
        )

        print(
            "Mask shape:",
            mask.shape,
        )

        print(
            "MRI dtype:",
            image.dtype,
        )

        print(
            "Mask dtype:",
            mask.dtype,
        )

        print(
            "Tumor voxels:",
            tumor_voxels,
        )

        # ----------------------------------------------------
        # Save original shape
        # ----------------------------------------------------

        original_shape = (
            image.shape[1:]
        )

        # ----------------------------------------------------
        # Pad if required
        # ----------------------------------------------------

        image, mask = pad_to_patch_size(
            image,
            mask,
            PATCH_SIZE,
        )

        padded_shape = (
            image.shape[1:]
        )

        if padded_shape != original_shape:

            print(
                "\nPadding applied:"
            )

            print(
                "Before:",
                original_shape,
            )

            print(
                "After :",
                padded_shape,
            )

        else:

            print(
                "Padding:",
                "Not required",
            )

        # ----------------------------------------------------
        # Final size check
        # ----------------------------------------------------

        for current_size, required_size in zip(
            image.shape[1:],
            PATCH_SIZE,
        ):

            if current_size < required_size:

                raise RuntimeError(
                    "\nPADDING FAILED\n"
                    f"Volume: {image.shape}\n"
                    f"Patch : {PATCH_SIZE}"
                )

        # ----------------------------------------------------
        # MONAI dictionary
        # ----------------------------------------------------

        data = {
            "image": image,
            "label": mask,
        }

        # ----------------------------------------------------
        # Tumor-aware random patches
        # ----------------------------------------------------

        patches = patch_transform(
            data
        )

        # ----------------------------------------------------
        # Validate patches
        # ----------------------------------------------------

        validated_patches = []

        for patch_index, patch in enumerate(
            patches,
            start=1,
        ):

            patch_image = patch[
                "image"
            ]

            patch_label = patch[
                "label"
            ]

            expected_image_shape = (
                4,
                *PATCH_SIZE,
            )

            expected_label_shape = (
                1,
                *PATCH_SIZE,
            )

            if tuple(
                patch_image.shape
            ) != expected_image_shape:

                raise RuntimeError(
                    "\nINVALID IMAGE PATCH\n"
                    f"Expected: {expected_image_shape}\n"
                    f"Received: {tuple(patch_image.shape)}"
                )

            if tuple(
                patch_label.shape
            ) != expected_label_shape:

                raise RuntimeError(
                    "\nINVALID LABEL PATCH\n"
                    f"Expected: {expected_label_shape}\n"
                    f"Received: {tuple(patch_label.shape)}"
                )

            if patch_image.dtype != torch.float32:

                raise RuntimeError(
                    "MRI patch must be torch.float32.\n"
                    f"Received: {patch_image.dtype}"
                )

            if patch_label.dtype != torch.int64:

                raise RuntimeError(
                    "Label patch must be torch.int64.\n"
                    f"Received: {patch_label.dtype}"
                )

            patch_labels = torch.unique(
                patch_label
            )

            if torch.any(
                patch_labels < 0
            ):

                raise RuntimeError(
                    "Negative labels found in patch."
                )

            if torch.any(
                patch_labels > 3
            ):

                raise RuntimeError(
                    "Invalid labels found in patch:\n"
                    f"{patch_labels.tolist()}"
                )

            validated_patches.append(
                {
                    "image": patch_image,
                    "label": patch_label,
                }
            )

        # ----------------------------------------------------
        # Verify patch count
        # ----------------------------------------------------

        if len(
            validated_patches
        ) != NUM_SAMPLES_PER_CASE:

            raise RuntimeError(
                "Unexpected number of patches.\n"
                f"Expected: {NUM_SAMPLES_PER_CASE}\n"
                f"Received: {len(validated_patches)}"
            )

        return validated_patches


# ============================================================
# DIRECT DATASET TEST
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 70
    )

    print(
        "GLIOMA MONAI DATASET TEST"
    )

    print(
        "=" * 70
    )

    cases = get_cases(
        DATASET_ROOT
    )

    print(
        "\nDataset:",
        DATASET_ROOT,
    )

    print(
        "Total cases:",
        len(cases),
    )

    # --------------------------------------------------------
    # Test first case
    # --------------------------------------------------------

    dataset = GliomaDataset(
        cases[:1]
    )

    print(
        "\nDataset length:",
        len(dataset),
    )

    patches = dataset[0]

    print(
        "\n" + "=" * 70
    )

    print(
        "PATCH VALIDATION"
    )

    print(
        "=" * 70
    )

    print(
        "Number of patches:",
        len(patches),
    )

    for index, patch in enumerate(
        patches,
        start=1,
    ):

        image = patch[
            "image"
        ]

        label = patch[
            "label"
        ]

        print(
            "\nPatch",
            index,
        )

        print(
            "Image shape:",
            tuple(image.shape),
        )

        print(
            "Label shape:",
            tuple(label.shape),
        )

        print(
            "Image dtype:",
            image.dtype,
        )

        print(
            "Label dtype:",
            label.dtype,
        )

        print(
            "Tumor voxels:",
            int(
                (label > 0)
                .sum()
                .item()
            ),
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "GLIOMA DATASET TEST PASSED"
    )

    print(
        "=" * 70
    )