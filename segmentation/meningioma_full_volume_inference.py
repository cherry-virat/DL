from pathlib import Path
import json
import time
import sys


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\nchan\Desktop\DL"
)

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


import numpy as np
import torch
import torch.nn.functional as F

from models.meningioma_model import create_meningioma_model


# ============================================================
# PATHS
# ============================================================



PREPROCESSED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "meningioma_preprocessed"
)

SPLIT_FILE = (
    PROJECT_ROOT
    / "data"
    / "meningioma_split.json"
)

CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "meningioma_training"
    / "meningioma_best.pth"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "meningioma_full_volume_inference"
)


# ============================================================
# INFERENCE SETTINGS
# ============================================================

PATCH_SIZE = (
    64,
    64,
    64
)

# Sliding-window overlap.
#
# 0.5 means adjacent windows overlap by 50%.
#
# This is the same general strategy used by the
# Glioma full-volume pipeline.
OVERLAP = 0.5

# Probability threshold.
#
# IMPORTANT:
# This is only the initial baseline.
#
# Threshold tuning will be performed later by
# meningioma_threshold_analysis.py.
INITIAL_THRESHOLD = 0.50

# Model configuration must match training.
BASE_CHANNELS = 16

# Batch size for inference windows.
#
# RTX 3050 6GB:
# Start conservatively with 1.
WINDOW_BATCH_SIZE = 1


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_model():

    print("\nLoading model...")

    if not CHECKPOINT.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT}"
        )

    model = create_meningioma_model(
        base_channels=BASE_CHANNELS
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )

    # --------------------------------------------------------
    # Support both:
    #
    # 1. raw state_dict
    # 2. checkpoint dictionary containing state_dict
    # --------------------------------------------------------

    if isinstance(
        checkpoint,
        dict
    ) and "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif isinstance(
        checkpoint,
        dict
    ) and "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    print(
        "Checkpoint:",
        CHECKPOINT
    )

    print(
        "Device:",
        DEVICE
    )

    if DEVICE.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    return model


# ============================================================
# LOAD SPLIT
# ============================================================

def load_dev_cases():

    if not SPLIT_FILE.exists():

        raise FileNotFoundError(
            f"Split file not found:\n{SPLIT_FILE}"
        )

    with open(
        SPLIT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        split_data = json.load(f)

    dev_cases = list(
        split_data["dev_cases"]
    )

    return dev_cases


# ============================================================
# PAD VOLUME
# ============================================================

def pad_volume(
    image,
    patch_size
):

    pads = []

    for dimension, patch_dimension in zip(
        image.shape,
        patch_size
    ):

        if dimension < patch_dimension:

            total_pad = (
                patch_dimension
                - dimension
            )

            before = (
                total_pad // 2
            )

            after = (
                total_pad
                - before
            )

        else:

            before = 0
            after = 0

        pads.append(
            (before, after)
        )

    if any(
        before > 0 or after > 0
        for before, after in pads
    ):

        padded = np.pad(
            image,
            pads,
            mode="constant",
            constant_values=0
        )

    else:

        padded = image

    return padded, pads


# ============================================================
# SLIDING WINDOW STARTS
# ============================================================

def compute_starts(
    dimension,
    patch_dimension,
    overlap
):

    if dimension <= patch_dimension:

        return [0]

    stride = int(
        patch_dimension
        * (1.0 - overlap)
    )

    if stride <= 0:

        raise ValueError(
            "Invalid overlap. "
            "Overlap must be < 1.0."
        )

    starts = list(
        range(
            0,
            dimension - patch_dimension + 1,
            stride
        )
    )

    final_start = (
        dimension
        - patch_dimension
    )

    if starts[-1] != final_start:

        starts.append(
            final_start
        )

    return starts


# ============================================================
# SLIDING WINDOW INFERENCE
# ============================================================

@torch.no_grad()
def sliding_window_inference(
    model,
    image
):

    original_shape = image.shape

    # --------------------------------------------------------
    # Pad if necessary
    # --------------------------------------------------------

    padded_image, pads = pad_volume(
        image,
        PATCH_SIZE
    )

    padded_shape = padded_image.shape

    # --------------------------------------------------------
    # Compute window starts
    # --------------------------------------------------------

    starts_d = compute_starts(
        padded_shape[0],
        PATCH_SIZE[0],
        OVERLAP
    )

    starts_h = compute_starts(
        padded_shape[1],
        PATCH_SIZE[1],
        OVERLAP
    )

    starts_w = compute_starts(
        padded_shape[2],
        PATCH_SIZE[2],
        OVERLAP
    )

    total_windows = (
        len(starts_d)
        * len(starts_h)
        * len(starts_w)
    )

    # --------------------------------------------------------
    # Accumulation arrays
    #
    # We accumulate tumor probabilities and the number
    # of predictions contributing to each voxel.
    # --------------------------------------------------------

    probability_sum = np.zeros(
        padded_shape,
        dtype=np.float32
    )

    prediction_count = np.zeros(
        padded_shape,
        dtype=np.float32
    )

    window_index = 0

    for start_d in starts_d:

        for start_h in starts_h:

            for start_w in starts_w:

                end_d = (
                    start_d
                    + PATCH_SIZE[0]
                )

                end_h = (
                    start_h
                    + PATCH_SIZE[1]
                )

                end_w = (
                    start_w
                    + PATCH_SIZE[2]
                )

                patch = padded_image[
                    start_d:end_d,
                    start_h:end_h,
                    start_w:end_w
                ]

                # ------------------------------------------------
                # Convert:
                #
                # (D,H,W)
                #     ↓
                # (1,1,D,H,W)
                # ------------------------------------------------

                patch_tensor = torch.from_numpy(
                    patch.astype(
                        np.float32,
                        copy=False
                    )
                )

                patch_tensor = (
                    patch_tensor
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(DEVICE)
                )

                # ------------------------------------------------
                # Model prediction
                # ------------------------------------------------

                logits = model(
                    patch_tensor
                )

                # ------------------------------------------------
                # Softmax
                # ------------------------------------------------

                probabilities = F.softmax(
                    logits,
                    dim=1
                )

                # Class 1 = tumor
                tumor_probability = (
                    probabilities[
                        0,
                        1
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                )

                # ------------------------------------------------
                # Accumulate
                # ------------------------------------------------

                probability_sum[
                    start_d:end_d,
                    start_h:end_h,
                    start_w:end_w
                ] += tumor_probability

                prediction_count[
                    start_d:end_d,
                    start_h:end_h,
                    start_w:end_w
                ] += 1.0

                window_index += 1

    # --------------------------------------------------------
    # Average overlapping predictions
    # --------------------------------------------------------

    tumor_probability = (
        probability_sum
        / np.maximum(
            prediction_count,
            1.0
        )
    )

    # --------------------------------------------------------
    # Remove padding
    # --------------------------------------------------------

    crop_slices = []

    for axis, (
        before,
        after
    ) in enumerate(pads):

        start = before

        if after == 0:

            stop = None

        else:

            stop = (
                tumor_probability.shape[axis]
                - after
            )

        crop_slices.append(
            slice(
                start,
                stop
            )
        )

    tumor_probability = (
        tumor_probability[
            tuple(crop_slices)
        ]
    )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if tumor_probability.shape != original_shape:

        raise RuntimeError(
            "Inference output shape mismatch.\n"
            f"Expected: {original_shape}\n"
            f"Found: {tumor_probability.shape}"
        )

    return (
        tumor_probability.astype(
            np.float32
        ),
        total_windows
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    prediction,
    target
):

    prediction = (
        prediction > 0
    )

    target = (
        target > 0
    )

    true_positive = np.logical_and(
        prediction,
        target
    ).sum()

    false_positive = np.logical_and(
        prediction,
        np.logical_not(target)
    ).sum()

    false_negative = np.logical_and(
        np.logical_not(prediction),
        target
    ).sum()

    intersection = true_positive

    prediction_sum = prediction.sum()

    target_sum = target.sum()

    union = np.logical_or(
        prediction,
        target
    ).sum()

    # --------------------------------------------------------
    # Dice
    # --------------------------------------------------------

    dice_denominator = (
        prediction_sum
        + target_sum
    )

    if dice_denominator == 0:

        dice = 1.0

    else:

        dice = (
            2.0
            * intersection
            / dice_denominator
        )

    # --------------------------------------------------------
    # IoU
    # --------------------------------------------------------

    if union == 0:

        iou = 1.0

    else:

        iou = (
            intersection
            / union
        )

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    precision_denominator = (
        true_positive
        + false_positive
    )

    if precision_denominator == 0:

        precision = 1.0

    else:

        precision = (
            true_positive
            / precision_denominator
        )

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    recall_denominator = (
        true_positive
        + false_negative
    )

    if recall_denominator == 0:

        recall = 1.0

    else:

        recall = (
            true_positive
            / recall_denominator
        )

    return {
        "precision": float(
            precision
        ),
        "recall": float(
            recall
        ),
        "dice": float(
            dice
        ),
        "iou": float(
            iou
        ),
        "true_positive": int(
            true_positive
        ),
        "false_positive": int(
            false_positive
        ),
        "false_negative": int(
            false_negative
        ),
        "predicted_tumor_voxels": int(
            prediction_sum
        ),
        "ground_truth_tumor_voxels": int(
            target_sum
        )
    }


# ============================================================
# PROCESS ONE CASE
# ============================================================

def process_case(
    model,
    case_name
):

    case_dir = (
        PREPROCESSED_ROOT
        / case_name
    )

    image_file = (
        case_dir
        / "t1c.npy"
    )

    mask_file = (
        case_dir
        / "gtv.npy"
    )

    if not image_file.exists():

        raise FileNotFoundError(
            f"Missing image:\n{image_file}"
        )

    if not mask_file.exists():

        raise FileNotFoundError(
            f"Missing ground truth:\n{mask_file}"
        )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    image = np.load(
        image_file
    ).astype(
        np.float32
    )

    ground_truth = np.load(
        mask_file
    ).astype(
        np.uint8
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if image.ndim != 3:

        raise RuntimeError(
            f"Expected 3D image. "
            f"Found shape {image.shape}"
        )

    if ground_truth.ndim != 3:

        raise RuntimeError(
            f"Expected 3D mask. "
            f"Found shape {ground_truth.shape}"
        )

    if image.shape != ground_truth.shape:

        raise RuntimeError(
            "Image/mask shape mismatch.\n"
            f"Image: {image.shape}\n"
            f"Mask:  {ground_truth.shape}"
        )

    if not np.isfinite(image).all():

        raise RuntimeError(
            "Image contains NaN or Inf."
        )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    start_time = time.time()

    probability, total_windows = (
        sliding_window_inference(
            model,
            image
        )
    )

    elapsed = (
        time.time()
        - start_time
    )

    # --------------------------------------------------------
    # Initial threshold
    # --------------------------------------------------------

    prediction = (
        probability
        >= INITIAL_THRESHOLD
    ).astype(
        np.uint8
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        prediction,
        ground_truth
    )

    metrics[
        "case_name"
    ] = case_name

    metrics[
        "image_shape"
    ] = list(
        image.shape
    )

    metrics[
        "windows"
    ] = int(
        total_windows
    )

    metrics[
        "inference_seconds"
    ] = float(
        elapsed
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    case_output = (
        OUTPUT_ROOT
        / case_name
    )

    case_output.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save probability
    # --------------------------------------------------------

    np.save(
        case_output
        / "tumor_probability.npy",
        probability.astype(
            np.float32
        )
    )

    # --------------------------------------------------------
    # Save binary prediction
    # --------------------------------------------------------

    np.save(
        case_output
        / "prediction.npy",
        prediction.astype(
            np.uint8
        )
    )

    # --------------------------------------------------------
    # Save ground truth
    # --------------------------------------------------------

    np.save(
        case_output
        / "ground_truth.npy",
        ground_truth.astype(
            np.uint8
        )
    )

    # --------------------------------------------------------
    # Save case metrics
    # --------------------------------------------------------

    with open(
        case_output / "metrics.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("MENINGIOMA FULL-VOLUME INFERENCE")
    print("=" * 100)

    print(
        "\nProject root:"
    )

    print(
        PROJECT_ROOT
    )

    print(
        "\nPreprocessed root:"
    )

    print(
        PREPROCESSED_ROOT
    )

    print(
        "\nCheckpoint:"
    )

    print(
        CHECKPOINT
    )

    print(
        "\nOutput root:"
    )

    print(
        OUTPUT_ROOT
    )

    print(
        "\nPatch size:"
    )

    print(
        PATCH_SIZE
    )

    print(
        "\nOverlap:"
    )

    print(
        OVERLAP
    )

    print(
        "\nInitial threshold:"
    )

    print(
        INITIAL_THRESHOLD
    )

    # --------------------------------------------------------
    # Validate paths
    # --------------------------------------------------------

    if not PREPROCESSED_ROOT.exists():

        raise FileNotFoundError(
            f"Preprocessed root not found:\n"
            f"{PREPROCESSED_ROOT}"
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

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
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Load development cases
    # --------------------------------------------------------

    dev_cases = load_dev_cases()

    print(
        "\nDevelopment cases:",
        len(dev_cases)
    )

    # --------------------------------------------------------
    # Process cases
    # --------------------------------------------------------

    all_metrics = []

    total_start = time.time()

    for index, case_name in enumerate(
        dev_cases,
        start=1
    ):

        print("\n" + "-" * 100)

        print(
            f"CASE {index}/{len(dev_cases)}"
        )

        print(
            case_name
        )

        try:

            metrics = process_case(
                model,
                case_name
            )

            all_metrics.append(
                metrics
            )

            print(
                "\nShape:",
                tuple(
                    metrics["image_shape"]
                )
            )

            print(
                "Sliding windows:",
                metrics["windows"]
            )

            print(
                "Inference time:",
                f"{metrics['inference_seconds']:.2f}",
                "seconds"
            )

            print(
                "Ground-truth tumor voxels:",
                metrics[
                    "ground_truth_tumor_voxels"
                ]
            )

            print(
                "Predicted tumor voxels:",
                metrics[
                    "predicted_tumor_voxels"
                ]
            )

            print(
                "Precision:",
                f"{metrics['precision']:.6f}"
            )

            print(
                "Recall:",
                f"{metrics['recall']:.6f}"
            )

            print(
                "Dice:",
                f"{metrics['dice']:.6f}"
            )

            print(
                "IoU:",
                f"{metrics['iou']:.6f}"
            )

        except Exception as e:

            print(
                "\nERROR:",
                str(e)
            )

            raise

    # --------------------------------------------------------
    # Aggregate metrics
    # --------------------------------------------------------

    if not all_metrics:

        raise RuntimeError(
            "No cases were successfully processed."
        )

    metric_names = [
        "precision",
        "recall",
        "dice",
        "iou"
    ]

    aggregate = {}

    for metric_name in metric_names:

        values = [
            item[metric_name]
            for item in all_metrics
        ]

        aggregate[
            metric_name
        ] = float(
            np.mean(values)
        )

    aggregate[
        "cases"
    ] = len(
        all_metrics
    )

    aggregate[
        "threshold"
    ] = INITIAL_THRESHOLD

    aggregate[
        "overlap"
    ] = OVERLAP

    aggregate[
        "patch_size"
    ] = list(
        PATCH_SIZE
    )

    aggregate[
        "total_time_seconds"
    ] = float(
        time.time()
        - total_start
    )

    # --------------------------------------------------------
    # Save complete report
    # --------------------------------------------------------

    report = {
        "configuration": {
            "checkpoint": str(
                CHECKPOINT
            ),
            "patch_size": list(
                PATCH_SIZE
            ),
            "overlap": OVERLAP,
            "threshold": INITIAL_THRESHOLD,
            "base_channels": BASE_CHANNELS
        },
        "aggregate": aggregate,
        "cases": all_metrics
    }

    report_file = (
        OUTPUT_ROOT
        / "meningioma_full_volume_report.json"
    )

    with open(
        report_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 100)

    print(
        "MENINGIOMA FULL-VOLUME INFERENCE COMPLETE"
    )

    print("=" * 100)

    print(
        "\nCases evaluated:",
        aggregate["cases"]
    )

    print(
        "\nMean Precision:",
        f"{aggregate['precision']:.6f}"
    )

    print(
        "Mean Recall:",
        f"{aggregate['recall']:.6f}"
    )

    print(
        "Mean Dice:",
        f"{aggregate['dice']:.6f}"
    )

    print(
        "Mean IoU:",
        f"{aggregate['iou']:.6f}"
    )

    print(
        "\nReport:"
    )

    print(
        report_file
    )

    print(
        "\nPredictions:"
    )

    print(
        OUTPUT_ROOT
    )

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()