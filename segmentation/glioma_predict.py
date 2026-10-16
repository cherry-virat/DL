"""
Glioma 3D U-Net Prediction and Visualization
==============================================

Purpose
-------
Load the trained Glioma 3D U-Net and generate segmentation
predictions on unseen validation cases.

Outputs
-------
For selected validation cases:

1. MRI slice
2. Ground-truth segmentation
3. Predicted segmentation
4. Overlay visualization
5. NumPy prediction mask

The trained model is NOT modified.
"""


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import random

import numpy as np

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from monai.networks.nets import UNet

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


# ============================================================
# CHECKPOINT
# ============================================================

CHECKPOINT_PATH = (
    Path("outputs")
    / "glioma_training"
    / "glioma_best_small_experiment.pth"
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = (
    Path("outputs")
    / "glioma_predictions"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

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
# COLLATE FUNCTION
# ============================================================

def glioma_collate_fn(batch):

    """
    GliomaDataset returns:

        [
            [
                patch1,
                patch2
            ]
        ]

    Convert to:

        {
            "image": Tensor(N,C,D,H,W),
            "label": Tensor(N,1,D,H,W)
        }
    """

    if len(batch) == 0:

        raise RuntimeError(
            "Empty batch."
        )


    patches = []


    for case_output in batch:

        if not isinstance(
            case_output,
            list,
        ):

            raise TypeError(
                "Expected list of patches."
            )


        patches.extend(
            case_output
        )


    if len(patches) == 0:

        raise RuntimeError(
            "No patches found."
        )


    images = []

    labels = []


    for patch in patches:

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
# LOAD MODEL
# ============================================================

def load_model():

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            "Checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )


    model = create_model()


    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


    model.eval()


    print(
        "\nModel loaded successfully."
    )

    print(
        "Checkpoint:",
        CHECKPOINT_PATH,
    )


    return model


# ============================================================
# SAVE NUMPY MASK
# ============================================================

def save_numpy_mask(
    prediction,
    target,
    output_directory,
):

    prediction_path = (
        output_directory
        /
        "prediction.npy"
    )


    target_path = (
        output_directory
        /
        "ground_truth.npy"
    )


    np.save(
        prediction_path,
        prediction,
    )


    np.save(
        target_path,
        target,
    )


    print(
        "Prediction saved:",
        prediction_path,
    )

    print(
        "Ground truth saved:",
        target_path,
    )


# ============================================================
# CREATE VISUALIZATION
# ============================================================

def create_visualization(
    image,
    prediction,
    target,
    output_directory,
    case_name,
    patch_index,
):

    """
    Create a central axial slice visualization.

    image:
        (4,D,H,W)

    prediction:
        (D,H,W)

    target:
        (D,H,W)
    """


    # --------------------------------------------------------
    # Select slice with maximum ground-truth tumor
    # --------------------------------------------------------

    tumor_counts = (
        (target > 0)
        .sum(
            axis=(1, 2)
        )
    )


    if tumor_counts.max() > 0:

        slice_index = int(
            np.argmax(
                tumor_counts
            )
        )

    else:

        slice_index = (
            target.shape[0] // 2
        )


    # --------------------------------------------------------
    # MRI modality
    #
    # Channel 0 is used as the visualization background.
    # --------------------------------------------------------

    mri_slice = (
        image[0, slice_index]
    )


    target_slice = (
        target[slice_index]
    )


    prediction_slice = (
        prediction[slice_index]
    )


    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    figure = plt.figure(
        figsize=(16, 4)
    )


    # ========================================================
    # MRI
    # ========================================================

    axis1 = figure.add_subplot(
        1,
        4,
        1,
    )


    axis1.imshow(
        mri_slice,
        cmap="gray",
    )


    axis1.set_title(
        f"MRI\nSlice {slice_index}"
    )


    axis1.axis(
        "off"
    )


    # ========================================================
    # GROUND TRUTH
    # ========================================================

    axis2 = figure.add_subplot(
        1,
        4,
        2,
    )


    axis2.imshow(
        target_slice,
        cmap="nipy_spectral",
        vmin=0,
        vmax=3,
    )


    axis2.set_title(
        "Ground Truth"
    )


    axis2.axis(
        "off"
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    axis3 = figure.add_subplot(
        1,
        4,
        3,
    )


    axis3.imshow(
        prediction_slice,
        cmap="nipy_spectral",
        vmin=0,
        vmax=3,
    )


    axis3.set_title(
        "Prediction"
    )


    axis3.axis(
        "off"
    )


    # ========================================================
    # OVERLAY
    # ========================================================

    axis4 = figure.add_subplot(
        1,
        4,
        4,
    )


    axis4.imshow(
        mri_slice,
        cmap="gray",
    )


    # Ground truth contour
    axis4.contour(
        target_slice > 0,
        levels=[0.5],
        linewidths=1.5,
    )


    # Prediction contour
    axis4.contour(
        prediction_slice > 0,
        levels=[0.5],
        linewidths=1.5,
        linestyles="dashed",
    )


    axis4.set_title(
        "Tumor Overlay"
    )


    axis4.axis(
        "off"
    )


    # --------------------------------------------------------
    # Figure title
    # --------------------------------------------------------

    figure.suptitle(
        f"{case_name} | Patch {patch_index}",
        fontsize=14,
    )


    figure.tight_layout()


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = (
        output_directory
        /
        f"patch_{patch_index:02d}.png"
    )


    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )


    plt.close(
        figure
    )


    print(
        "Visualization saved:",
        output_path,
    )


# ============================================================
# PREDICT CASE
# ============================================================

def predict_case(
    model,
    case,
    case_index,
):

    print(
        "\n" + "-" * 70
    )

    print(
        f"PREDICTING CASE {case_index}"
    )

    print(
        "-" * 70
    )


    print(
        "Case:",
        case.name,
    )


    # --------------------------------------------------------
    # Dataset containing only this case
    # --------------------------------------------------------

    dataset = GliomaDataset(
        [case]
    )


    loader = DataLoader(
        dataset,

        batch_size=1,

        shuffle=False,

        num_workers=0,

        collate_fn=glioma_collate_fn,
    )


    case_output_directory = (
        OUTPUT_DIR
        /
        case.name
    )


    case_output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )


    # --------------------------------------------------------
    # Process patches
    # --------------------------------------------------------

    with torch.no_grad():

        for batch in loader:

            images = batch["image"]

            labels = batch["label"]


            images_gpu = images.to(
                DEVICE,
                non_blocking=True,
            )


            outputs = model(
                images_gpu
            )


            predictions = torch.argmax(
                outputs,
                dim=1,
            )


            # Move to CPU
            images_cpu = (
                images.cpu()
                .numpy()
            )


            labels_cpu = (
                labels.squeeze(1)
                .cpu()
                .numpy()
            )


            predictions_cpu = (
                predictions.cpu()
                .numpy()
            )


            # ------------------------------------------------
            # Each patch
            # ------------------------------------------------

            for patch_index in range(
                predictions_cpu.shape[0]
            ):

                image = (
                    images_cpu[
                        patch_index
                    ]
                )


                prediction = (
                    predictions_cpu[
                        patch_index
                    ]
                )


                target = (
                    labels_cpu[
                        patch_index
                    ]
                )


                # ------------------------------------------------
                # Statistics
                # ------------------------------------------------

                ground_truth_tumor = (
                    target > 0
                ).sum()


                predicted_tumor = (
                    prediction > 0
                ).sum()


                intersection = (
                    (
                        (prediction > 0)
                        &
                        (target > 0)
                    )
                    .sum()
                )


                if (
                    predicted_tumor
                    +
                    ground_truth_tumor
                ) > 0:

                    dice = (
                        2.0
                        *
                        intersection
                        /
                        (
                            predicted_tumor
                            +
                            ground_truth_tumor
                        )
                    )

                else:

                    dice = 1.0


                print(
                    f"\nPatch {patch_index + 1}"
                )

                print(
                    "Ground-truth tumor voxels:",
                    int(ground_truth_tumor),
                )

                print(
                    "Predicted tumor voxels:",
                    int(predicted_tumor),
                )

                print(
                    "Tumor Dice:",
                    f"{dice:.6f}",
                )


                # ------------------------------------------------
                # Save masks
                # ------------------------------------------------

                save_numpy_mask(
                    prediction=prediction,

                    target=target,

                    output_directory=(
                        case_output_directory
                    ),
                )


                # ------------------------------------------------
                # Visualization
                # ------------------------------------------------

                create_visualization(
                    image=image,

                    prediction=prediction,

                    target=target,

                    output_directory=(
                        case_output_directory
                    ),

                    case_name=case.name,

                    patch_index=(
                        patch_index + 1
                    ),
                )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "GLIOMA 3D U-NET PREDICTION"
    )

    print(
        "=" * 70
    )


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
            "Best model checkpoint not found."
        )


    # ========================================================
    # FIND CASES
    # ========================================================

    cases = get_cases(
        DATASET_ROOT
    )


    print(
        "\nTotal cases:",
        len(cases),
    )


    # ========================================================
    # CREATE SAME SPLIT
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


    # ========================================================
    # VERIFY NO OVERLAP
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
            "Training/validation overlap detected."
        )


    print(
        "\n✓ Validation cases are unseen during training."
    )


    # ========================================================
    # PRINT VALIDATION CASES
    # ========================================================

    print(
        "\nValidation cases:"
    )


    for index, case in enumerate(
        validation_cases,
        start=1,
    ):

        print(
            f"  {index}. {case.name}"
        )


    # ========================================================
    # LOAD MODEL
    # ========================================================

    model = load_model()


    # ========================================================
    # PREDICT
    # ========================================================

    for case_index, case in enumerate(
        validation_cases,
        start=1,
    ):

        predict_case(
            model=model,

            case=case,

            case_index=case_index,
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


    # ========================================================
    # COMPLETE
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "PREDICTION GENERATION COMPLETE"
    )

    print(
        "=" * 70
    )


    print(
        "\nPredictions saved to:"
    )

    print(
        OUTPUT_DIR
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()