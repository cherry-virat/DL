"""
Glioma True Full-Volume Sliding-Window Inference
================================================

Purpose
-------
Run the trained medium Glioma 3D U-Net over the complete
preprocessed MRI volume using overlapping sliding windows.

Pipeline
--------
Full MRI volume
    ↓
Pad if necessary
    ↓
Sliding-window inference
    ↓
Probability volume
    ↓
Whole-tumor probability
    ↓
Threshold = 0.60
    ↓
Connected components
    ↓
Remove components < 1000 voxels
    ↓
Final full-volume tumor mask
"""


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F

from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference

from scipy import ndimage

from preprocessing.glioma_dataset import (
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

VAL_CASES = 5

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

ROI_SIZE = (
    64,
    64,
    64,
)

SW_BATCH_SIZE = 1

OVERLAP = 0.50

TUMOR_THRESHOLD = 0.60

MIN_COMPONENT_SIZE = 1000


# ============================================================
# CHECKPOINT
# ============================================================

CHECKPOINT = (
    Path("outputs")
    / "glioma_training"
    / "glioma_best_medium_experiment.pth"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    Path("outputs")
    / "glioma_true_full_volume"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
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
# SEED
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)


# ============================================================
# MODEL
# ============================================================

def create_model():

    model = UNet(
        spatial_dims=3,
        in_channels=INPUT_CHANNELS,
        out_channels=OUTPUT_CLASSES,
        channels=MODEL_CHANNELS,
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=DROPOUT,
    )


    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )


    model.load_state_dict(
        checkpoint["model_state_dict"]
    )


    model = model.to(
        DEVICE
    )

    model.eval()

    return model, checkpoint


# ============================================================
# PAD VOLUME
# ============================================================

def pad_volume(
    image,
    label,
):
    """
    image:
        (C,D,H,W)

    label:
        (D,H,W)
    """

    _, depth, height, width = image.shape


    target_depth = max(
        depth,
        ROI_SIZE[0],
    )

    target_height = max(
        height,
        ROI_SIZE[1],
    )

    target_width = max(
        width,
        ROI_SIZE[2],
    )


    pad_d = target_depth - depth
    pad_h = target_height - height
    pad_w = target_width - width


    if (
        pad_d == 0
        and pad_h == 0
        and pad_w == 0
    ):

        return image, label, (
            depth,
            height,
            width,
        )


    # --------------------------------------------------------
    # Symmetric padding
    # --------------------------------------------------------

    before_d = pad_d // 2
    after_d = pad_d - before_d

    before_h = pad_h // 2
    after_h = pad_h - before_h

    before_w = pad_w // 2
    after_w = pad_w - before_w


    image = F.pad(
        image,
        (
            before_w,
            after_w,
            before_h,
            after_h,
            before_d,
            after_d,
        ),
        mode="constant",
        value=0,
    )


    label = F.pad(
        label,
        (
            before_w,
            after_w,
            before_h,
            after_h,
            before_d,
            after_d,
        ),
        mode="constant",
        value=0,
    )


    return image, label, (
        depth,
        height,
        width,
    )


# ============================================================
# REMOVE SMALL COMPONENTS
# ============================================================

def remove_small_components(
    mask,
    minimum_size,
):

    mask = np.asarray(
        mask,
        dtype=bool,
    )


    structure = np.ones(
        (3, 3, 3),
        dtype=np.uint8,
    )


    labeled, number = ndimage.label(
        mask,
        structure=structure,
    )


    if number == 0:

        return np.zeros_like(
            mask,
            dtype=bool,
        )


    sizes = np.bincount(
        labeled.ravel()
    )


    keep = sizes >= minimum_size

    keep[0] = False


    return keep[
        labeled
    ].astype(bool)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    prediction,
    ground_truth,
):

    prediction = prediction.astype(
        bool
    )

    ground_truth = ground_truth.astype(
        bool
    )


    if prediction.shape != ground_truth.shape:

        raise RuntimeError(
            "Shape mismatch:\n"
            f"Prediction: {prediction.shape}\n"
            f"Ground truth: {ground_truth.shape}"
        )


    tp = np.logical_and(
        prediction,
        ground_truth,
    ).sum()


    fp = np.logical_and(
        prediction,
        ~ground_truth,
    ).sum()


    fn = np.logical_and(
        ~prediction,
        ground_truth,
    ).sum()


    precision = tp / max(
        tp + fp,
        1,
    )


    recall = tp / max(
        tp + fn,
        1,
    )


    dice = (
        2 * tp
        /
        max(
            2 * tp + fp + fn,
            1,
        )
    )


    iou = (
        tp
        /
        max(
            tp + fp + fn,
            1,
        )
    )


    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "dice": float(dice),
        "iou": float(iou),
    }


# ============================================================
# PROCESS CASE
# ============================================================

def process_case(
    model,
    case,
):

    print(
        "\n" + "-" * 70
    )

    print(
        "CASE:",
        case.name,
    )

    print(
        "-" * 70
    )


    # --------------------------------------------------------
    # Load through existing preprocessing
    # --------------------------------------------------------

    from preprocessing.glioma_dataset import (
        prepare_case,
    )


    image, label = prepare_case(
        case
    )


    # image:
    # (4,D,H,W)

    # label:
    # (1,D,H,W)


    if label.ndim == 4:

        label = label.squeeze(
            0
        )


    original_shape = tuple(
        label.shape
    )


    print(
        "Original image:",
        tuple(image.shape),
    )

    print(
        "Original label:",
        tuple(label.shape),
    )


    # --------------------------------------------------------
    # Pad
    # --------------------------------------------------------

    image, label, original_shape = (
        pad_volume(
            image,
            label,
        )
    )


    print(
        "Inference image:",
        tuple(image.shape),
    )


    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    image_tensor = image.unsqueeze(
        0
    ).to(
        DEVICE
    )


    # --------------------------------------------------------
    # Sliding-window inference
    # --------------------------------------------------------

    print(
        "Running sliding-window inference..."
    )


    with torch.no_grad():

        output = sliding_window_inference(
            inputs=image_tensor,
            roi_size=ROI_SIZE,
            sw_batch_size=SW_BATCH_SIZE,
            predictor=model,
            overlap=OVERLAP,
            mode="gaussian",
        )


    print(
        "Raw output:",
        tuple(output.shape),
    )


    # --------------------------------------------------------
    # Softmax
    # --------------------------------------------------------

    probabilities = torch.softmax(
        output,
        dim=1,
    )


    # --------------------------------------------------------
    # Whole tumor probability
    # --------------------------------------------------------

    tumor_probability = (
        probabilities[:, 1:]
        .sum(
            dim=1
        )
    )


    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    prediction = (
        tumor_probability
        >= TUMOR_THRESHOLD
    )


    prediction = (
        prediction
        .squeeze(0)
        .cpu()
        .numpy()
    )


    label_np = (
        label
        .cpu()
        .numpy()
        > 0
    )


    # --------------------------------------------------------
    # Remove padding
    # --------------------------------------------------------

    depth, height, width = (
        original_shape
    )


    prediction = prediction[
        :depth,
        :height,
        :width,
    ]


    label_np = label_np[
        :depth,
        :height,
        :width,
    ]


    # --------------------------------------------------------
    # Connected components
    # --------------------------------------------------------

    before_components = (
        prediction.sum()
    )


    prediction = (
        remove_small_components(
            prediction,
            MIN_COMPONENT_SIZE,
        )
    )


    after_components = (
        prediction.sum()
    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        prediction,
        label_np,
    )


    print(
        "\nResults:"
    )

    print(
        "TP:",
        metrics["tp"],
    )

    print(
        "FP:",
        metrics["fp"],
    )

    print(
        "FN:",
        metrics["fn"],
    )

    print(
        "Precision:",
        f"{metrics['precision']:.6f}",
    )

    print(
        "Recall:",
        f"{metrics['recall']:.6f}",
    )

    print(
        "Dice:",
        f"{metrics['dice']:.6f}",
    )

    print(
        "IoU:",
        f"{metrics['iou']:.6f}",
    )

    print(
        "Tumor voxels before filtering:",
        int(before_components),
    )

    print(
        "Tumor voxels after filtering:",
        int(after_components),
    )


    # --------------------------------------------------------
    # Save numpy mask
    # --------------------------------------------------------

    output_path = (
        OUTPUT_DIR
        /
        f"{case.name}_tumor_mask.npy"
    )


    np.save(
        output_path,
        prediction.astype(
            np.uint8
        ),
    )


    print(
        "Mask saved:",
        output_path,
    )


    return metrics


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "GLIOMA TRUE FULL-VOLUME INFERENCE"
    )

    print(
        "=" * 70
    )


    print(
        "Dataset:",
        DATASET_ROOT,
    )

    print(
        "Checkpoint:",
        CHECKPOINT,
    )

    print(
        "Device:",
        DEVICE,
    )


    print(
        "\nConfiguration:"
    )

    print(
        "ROI:",
        ROI_SIZE,
    )

    print(
        "Overlap:",
        OVERLAP,
    )

    print(
        "Probability threshold:",
        TUMOR_THRESHOLD,
    )

    print(
        "Minimum component:",
        MIN_COMPONENT_SIZE,
    )


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model, checkpoint = create_model()


    print(
        "\n✓ Model loaded."
    )


    if "epoch" in checkpoint:

        print(
            "Checkpoint epoch:",
            checkpoint["epoch"],
        )


    # --------------------------------------------------------
    # Cases
    # --------------------------------------------------------

    cases = get_cases(
        DATASET_ROOT
    )


    train_cases, validation_cases = (
        create_train_val_split(
            cases,
            train_ratio=0.90,
            seed=SEED,
        )
    )


    validation_cases = validation_cases[
        :VAL_CASES
    ]


    train_names = {
        case.name
        for case in train_cases
    }


    validation_names = {
        case.name
        for case in validation_cases
    }


    if train_names & validation_names:

        raise RuntimeError(
            "Training/validation overlap detected."
        )


    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    results = []


    for index, case in enumerate(
        validation_cases,
        start=1,
    ):

        print(
            f"\nCASE {index}/{len(validation_cases)}"
        )


        result = process_case(
            model,
            case,
        )


        results.append(
            result
        )


    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    tp = sum(
        r["tp"]
        for r in results
    )

    fp = sum(
        r["fp"]
        for r in results
    )

    fn = sum(
        r["fn"]
        for r in results
    )


    precision = tp / max(
        tp + fp,
        1,
    )


    recall = tp / max(
        tp + fn,
        1,
    )


    dice = (
        2 * tp
        /
        max(
            2 * tp + fp + fn,
            1,
        )
    )


    iou = tp / max(
        tp + fp + fn,
        1,
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "FULL-VOLUME RESULTS"
    )

    print(
        "=" * 70
    )


    print(
        "TP:",
        tp,
    )

    print(
        "FP:",
        fp,
    )

    print(
        "FN:",
        fn,
    )

    print(
        "Precision:",
        f"{precision:.6f}",
    )

    print(
        "Recall:",
        f"{recall:.6f}",
    )

    print(
        "Dice:",
        f"{dice:.6f}",
    )

    print(
        "IoU:",
        f"{iou:.6f}",
    )


    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    report_path = (
        OUTPUT_DIR
        /
        "full_volume_report.txt"
    )


    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "GLIOMA TRUE FULL-VOLUME INFERENCE\n"
        )

        file.write(
            "=" * 70
            + "\n"
        )

        file.write(
            f"Threshold: {TUMOR_THRESHOLD}\n"
        )

        file.write(
            f"Minimum component: "
            f"{MIN_COMPONENT_SIZE}\n"
        )

        file.write(
            f"Precision: {precision:.6f}\n"
        )

        file.write(
            f"Recall: {recall:.6f}\n"
        )

        file.write(
            f"Dice: {dice:.6f}\n"
        )

        file.write(
            f"IoU: {iou:.6f}\n"
        )

        file.write(
            f"TP: {tp}\n"
        )

        file.write(
            f"FP: {fp}\n"
        )

        file.write(
            f"FN: {fn}\n"
        )


    print(
        "\nReport saved:"
    )

    print(
        report_path
    )


    print(
        "\n✓ TRUE FULL-VOLUME INFERENCE COMPLETE"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()