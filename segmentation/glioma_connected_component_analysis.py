"""
Glioma Connected Component False-Positive Analysis
===================================================

Purpose
-------
Reduce false-positive tumor voxels after probability thresholding.

Pipeline:

    3D U-Net
        |
        v
    Tumor probability
        |
        v
    Threshold = 0.60
        |
        v
    Binary tumor mask
        |
        v
    Connected components
        |
        v
    Remove small components
        |
        v
    Final segmentation
        |
        v
    Evaluate Precision / Recall / Dice / IoU

IMPORTANT
---------
This script does NOT retrain the model.

It evaluates post-processing only.

Current operating threshold:

    0.60
"""


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import sys

# Allow direct execution from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from scipy import ndimage

from monai.networks.nets import UNet

from monai.inferers import sliding_window_inference

from preprocessing.glioma_dataset import (
    DATASET_ROOT,
    get_cases,
    pad_to_patch_size,
)

from preprocessing.glioma_preprocess import (
    process_case,
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

INPUT_CHANNELS = 4

OUTPUT_CLASSES = 4

PATCH_SIZE = (
    64,
    64,
    64,
)

ROI_SIZE = PATCH_SIZE

OVERLAP = 0.5

SW_BATCH_SIZE = 1

MODEL_CHANNELS = (
    16,
    32,
    64,
    128,
    256,
)

DROPOUT = 0.0


# ============================================================
# CURRENT BEST THRESHOLD
# ============================================================

PROBABILITY_THRESHOLD = 0.60


# ============================================================
# CONNECTIVITY
# ============================================================

"""
3D connectivity:

6-connectivity:

A voxel is connected to its:

    up
    down
    left
    right
    front
    back

This is conservative and appropriate for removing isolated
small regions.
"""

CONNECTIVITY = 1


# ============================================================
# COMPONENT SIZES TO TEST
# ============================================================

"""
We do NOT immediately choose one size.

We test several minimum connected-component sizes.

Example:

    0       = no component filtering
    10      = remove components < 10 voxels
    25      = remove components < 25 voxels
    50      = remove components < 50 voxels
    100     = remove components < 100 voxels
    250     = remove components < 250 voxels
    500     = remove components < 500 voxels
    1000    = remove components < 1000 voxels

The best setting will be selected from validation results.
"""

MIN_COMPONENT_SIZES = [
    0,
    10,
    25,
    50,
    100,
    250,
    500,
    1000,
]


# ============================================================
# CHECKPOINT
# ============================================================

CHECKPOINT_PATH = (
    Path("outputs")
    / "glioma_training"
    / "glioma_best_medium_experiment.pth"
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = (
    Path("outputs")
    / "glioma_connected_component_analysis"
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

torch.manual_seed(
    SEED
)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        SEED
    )


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

    print(
        "\n" + "=" * 70
    )

    print(
        "LOADING MEDIUM 3D U-NET"
    )

    print(
        "=" * 70
    )

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            "\nCheckpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    model = create_model()

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    if "model_state_dict" not in checkpoint:

        raise RuntimeError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    print(
        "✓ Model loaded successfully."
    )

    print(
        "Checkpoint:",
        CHECKPOINT_PATH
    )

    if "epoch" in checkpoint:

        print(
            "Checkpoint epoch:",
            checkpoint["epoch"]
        )

    return model


# ============================================================
# FULL-VOLUME PROBABILITY
# ============================================================

def get_case_probability(
    model,
    case_dir,
):

    # --------------------------------------------------------
    # Exact training preprocessing
    # --------------------------------------------------------

    image, mask = process_case(
        case_dir
    )

    image = np.asarray(
        image,
        dtype=np.float32,
    )

    mask = np.asarray(
        mask,
        dtype=np.int64,
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if image.ndim != 4:

        raise RuntimeError(
            "Expected image shape "
            "(C,D,H,W).\n"
            f"Received: {image.shape}"
        )

    if mask.ndim != 3:

        raise RuntimeError(
            "Expected mask shape "
            "(D,H,W).\n"
            f"Received: {mask.shape}"
        )

    if image.shape[1:] != mask.shape:

        raise RuntimeError(
            "Image/mask shape mismatch.\n"
            f"Image: {image.shape}\n"
            f"Mask : {mask.shape}"
        )

    original_shape = tuple(
        image.shape[1:]
    )

    # --------------------------------------------------------
    # Add channel to mask for padding
    # --------------------------------------------------------

    mask_4d = np.expand_dims(
        mask,
        axis=0,
    )

    # --------------------------------------------------------
    # Pad
    # --------------------------------------------------------

    padded_image, padded_mask = (
        pad_to_patch_size(
            image,
            mask_4d,
            PATCH_SIZE,
        )
    )

    # --------------------------------------------------------
    # Input tensor
    # --------------------------------------------------------

    image_tensor = torch.from_numpy(
        padded_image
    ).float()

    input_tensor = (
        image_tensor
        .unsqueeze(0)
        .to(
            DEVICE,
            non_blocking=True,
        )
    )

    # --------------------------------------------------------
    # Sliding-window inference
    # --------------------------------------------------------

    with torch.inference_mode():

        logits = sliding_window_inference(

            inputs=input_tensor,

            roi_size=ROI_SIZE,

            sw_batch_size=SW_BATCH_SIZE,

            predictor=model,

            overlap=OVERLAP,

            mode="gaussian",
        )

        # ----------------------------------------------------
        # Convert logits to probabilities
        # ----------------------------------------------------

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        # ----------------------------------------------------
        # Tumor probability
        #
        # Classes:
        #
        # 0 = background
        # 1 = tumor class
        # 2 = tumor class
        # 3 = tumor class
        #
        # Tumor probability = P(1)+P(2)+P(3)
        # ----------------------------------------------------

        tumor_probability = (
            probabilities[:, 1:, ...]
            .sum(
                dim=1
            )
        )

    # --------------------------------------------------------
    # CPU
    # --------------------------------------------------------

    tumor_probability = (
        tumor_probability[
            0
        ]
        .detach()
        .cpu()
    )

    ground_truth = torch.from_numpy(
        padded_mask
    ).long()

    ground_truth = (
        ground_truth[
            0
        ]
        .detach()
        .cpu()
    )

    # --------------------------------------------------------
    # Remove padding
    # --------------------------------------------------------

    d, h, w = original_shape

    tumor_probability = (
        tumor_probability[
            :d,
            :h,
            :w,
        ]
    )

    ground_truth = (
        ground_truth[
            :d,
            :h,
            :w,
        ]
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if (
        tumor_probability.shape
        !=
        ground_truth.shape
    ):

        raise RuntimeError(
            "Probability/ground-truth "
            "shape mismatch.\n"
            f"Probability: "
            f"{tumor_probability.shape}\n"
            f"Ground truth: "
            f"{ground_truth.shape}"
        )

    return (
        tumor_probability.numpy(),
        ground_truth.numpy(),
    )


# ============================================================
# CONNECTED COMPONENT FILTER
# ============================================================

def remove_small_components(
    binary_mask,
    minimum_size,
):
    """
    Remove connected components smaller than
    minimum_size voxels.

    Parameters
    ----------
    binary_mask:
        Boolean 3D NumPy array.

    minimum_size:
        Minimum number of voxels a component must contain.

    Returns
    -------
    filtered_mask:
        Boolean 3D NumPy array.

    component_count:
        Number of components before filtering.

    removed_components:
        Number of components removed.

    removed_voxels:
        Number of voxels removed.
    """

    binary_mask = np.asarray(
        binary_mask,
        dtype=bool,
    )

    # --------------------------------------------------------
    # No filtering
    # --------------------------------------------------------

    if minimum_size <= 0:

        component_count = (
            int(
                ndimage.label(
                    binary_mask,
                    structure=ndimage.generate_binary_structure(
                        3,
                        CONNECTIVITY,
                    ),
                )[1]
            )
        )

        return (
            binary_mask.copy(),
            component_count,
            0,
            0,
        )

    # --------------------------------------------------------
    # 3D connectivity structure
    # --------------------------------------------------------

    structure = (
        ndimage.generate_binary_structure(
            3,
            CONNECTIVITY,
        )
    )

    # --------------------------------------------------------
    # Label components
    # --------------------------------------------------------

    labeled_mask, component_count = (
        ndimage.label(
            binary_mask,
            structure=structure,
        )
    )

    component_count = int(
        component_count
    )

    if component_count == 0:

        return (
            binary_mask.copy(),
            0,
            0,
            0,
        )

    # --------------------------------------------------------
    # Count component sizes
    #
    # bincount:
    #
    # index = component ID
    # value = number of voxels
    # --------------------------------------------------------

    component_sizes = np.bincount(
        labeled_mask.ravel()
    )

    # --------------------------------------------------------
    # Components that survive
    # --------------------------------------------------------

    keep_components = (
        component_sizes
        >=
        minimum_size
    )

    # Background must always be removed
    # from the component list.

    keep_components[0] = False

    # --------------------------------------------------------
    # Create filtered mask
    # --------------------------------------------------------

    filtered_mask = (
        keep_components[
            labeled_mask
        ]
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    kept_components = int(
        keep_components.sum()
    )

    removed_components = (
        component_count
        -
        kept_components
    )

    original_voxels = int(
        binary_mask.sum()
    )

    filtered_voxels = int(
        filtered_mask.sum()
    )

    removed_voxels = (
        original_voxels
        -
        filtered_voxels
    )

    return (
        filtered_mask,
        component_count,
        removed_components,
        removed_voxels,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    prediction,
    ground_truth,
):

    prediction = np.asarray(
        prediction,
        dtype=bool,
    )

    ground_truth = np.asarray(
        ground_truth,
        dtype=bool,
    )

    if prediction.shape != ground_truth.shape:

        raise RuntimeError(
            "Prediction and ground truth "
            "shapes do not match."
        )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    true_positive = int(
        (
            prediction
            &
            ground_truth
        ).sum()
    )

    false_positive = int(
        (
            prediction
            &
            ~ground_truth
        ).sum()
    )

    false_negative = int(
        (
            ~prediction
            &
            ground_truth
        ).sum()
    )

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    precision_denominator = (
        true_positive
        +
        false_positive
    )

    if precision_denominator > 0:

        precision = (
            true_positive
            /
            precision_denominator
        )

    else:

        precision = 0.0

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    recall_denominator = (
        true_positive
        +
        false_negative
    )

    if recall_denominator > 0:

        recall = (
            true_positive
            /
            recall_denominator
        )

    else:

        recall = 0.0

    # --------------------------------------------------------
    # Dice
    # --------------------------------------------------------

    dice_denominator = (
        2 * true_positive
        +
        false_positive
        +
        false_negative
    )

    if dice_denominator > 0:

        dice = (
            2
            *
            true_positive
            /
            dice_denominator
        )

    else:

        dice = 0.0

    # --------------------------------------------------------
    # IoU
    # --------------------------------------------------------

    iou_denominator = (
        true_positive
        +
        false_positive
        +
        false_negative
    )

    if iou_denominator > 0:

        iou = (
            true_positive
            /
            iou_denominator
        )

    else:

        iou = 0.0

    return {
        "true_positive":
            true_positive,

        "false_positive":
            false_positive,

        "false_negative":
            false_negative,

        "precision":
            precision,

        "recall":
            recall,

        "dice":
            dice,

        "iou":
            iou,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "GLIOMA CONNECTED-COMPONENT "
        "FALSE-POSITIVE ANALYSIS"
    )

    print(
        "=" * 70
    )

    print(
        "\nDataset:",
        DATASET_ROOT
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

    print(
        "\nProbability threshold:",
        PROBABILITY_THRESHOLD
    )

    print(
        "\nMinimum component sizes:"
    )

    print(
        MIN_COMPONENT_SIZES
    )

    # ========================================================
    # GET CASES
    # ========================================================

    cases = get_cases(
        DATASET_ROOT
    )

    print(
        "\nTotal cases:",
        len(cases)
    )

    # ========================================================
    # TRAIN / VALIDATION SPLIT
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
    # CHECK OVERLAP
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
            "Training/validation overlap detected:\n"
            f"{overlap}"
        )

    print(
        "\n✓ No training/validation overlap."
    )

    print(
        "\nValidation cases:"
    )

    for index, case in enumerate(
        validation_cases,
        start=1,
    ):

        print(
            f"{index}. {case.name}"
        )

    # ========================================================
    # MODEL
    # ========================================================

    model = load_model()

    # ========================================================
    # STORE CASE DATA
    # ========================================================

    all_case_data = []

    # ========================================================
    # RUN INFERENCE
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "RUNNING FULL-VOLUME INFERENCE"
    )

    print(
        "=" * 70
    )

    for index, case_dir in enumerate(
        validation_cases,
        start=1,
    ):

        print(
            "\n" + "-" * 70
        )

        print(
            f"CASE {index}/{len(validation_cases)}"
        )

        print(
            "-" * 70
        )

        print(
            "Case:",
            case_dir.name
        )

        probability, ground_truth = (
            get_case_probability(
                model,
                case_dir,
            )
        )

        # ----------------------------------------------------
        # Initial threshold mask
        # ----------------------------------------------------

        binary_prediction = (
            probability
            >=
            PROBABILITY_THRESHOLD
        )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        print(
            "Probability shape:",
            probability.shape
        )

        print(
            "Ground-truth shape:",
            ground_truth.shape
        )

        print(
            "Ground-truth tumor voxels:",
            int(
                (
                    ground_truth > 0
                ).sum()
            )
        )

        print(
            "Threshold tumor voxels:",
            int(
                binary_prediction.sum()
            )
        )

        all_case_data.append(
            {
                "case":
                    case_dir.name,

                "probability":
                    probability,

                "ground_truth":
                    ground_truth,
            }
        )

    # ========================================================
    # COMPONENT SEARCH
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "CONNECTED-COMPONENT SEARCH"
    )

    print(
        "=" * 70
    )

    print(
        "\nMinimum Size | Precision | Recall | Dice | IoU | FP"
    )

    print(
        "-" * 70
    )

    results = []

    for minimum_size in (
        MIN_COMPONENT_SIZES
    ):

        total_tp = 0

        total_fp = 0

        total_fn = 0

        total_removed_voxels = 0

        total_components = 0

        total_removed_components = 0

        # ----------------------------------------------------
        # Process each validation case
        # ----------------------------------------------------

        for case in all_case_data:

            probability = (
                case["probability"]
            )

            ground_truth = (
                case["ground_truth"]
                > 0
            )

            # ------------------------------------------------
            # Threshold
            # ------------------------------------------------

            binary_prediction = (
                probability
                >=
                PROBABILITY_THRESHOLD
            )

            # ------------------------------------------------
            # Connected components
            # ------------------------------------------------

            (
                filtered_prediction,
                component_count,
                removed_components,
                removed_voxels,
            ) = remove_small_components(
                binary_prediction,
                minimum_size,
            )

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            metrics = calculate_metrics(
                filtered_prediction,
                ground_truth,
            )

            total_tp += (
                metrics[
                    "true_positive"
                ]
            )

            total_fp += (
                metrics[
                    "false_positive"
                ]
            )

            total_fn += (
                metrics[
                    "false_negative"
                ]
            )

            total_removed_voxels += (
                removed_voxels
            )

            total_components += (
                component_count
            )

            total_removed_components += (
                removed_components
            )

        # ----------------------------------------------------
        # Aggregate precision
        # ----------------------------------------------------

        precision_denominator = (
            total_tp
            +
            total_fp
        )

        precision = (
            total_tp
            /
            precision_denominator
            if precision_denominator > 0
            else 0.0
        )

        # ----------------------------------------------------
        # Aggregate recall
        # ----------------------------------------------------

        recall_denominator = (
            total_tp
            +
            total_fn
        )

        recall = (
            total_tp
            /
            recall_denominator
            if recall_denominator > 0
            else 0.0
        )

        # ----------------------------------------------------
        # Aggregate Dice
        # ----------------------------------------------------

        dice_denominator = (
            2 * total_tp
            +
            total_fp
            +
            total_fn
        )

        dice = (
            2 * total_tp
            /
            dice_denominator
            if dice_denominator > 0
            else 0.0
        )

        # ----------------------------------------------------
        # Aggregate IoU
        # ----------------------------------------------------

        iou_denominator = (
            total_tp
            +
            total_fp
            +
            total_fn
        )

        iou = (
            total_tp
            /
            iou_denominator
            if iou_denominator > 0
            else 0.0
        )

        result = {
            "minimum_size":
                minimum_size,

            "precision":
                precision,

            "recall":
                recall,

            "dice":
                dice,

            "iou":
                iou,

            "tp":
                total_tp,

            "fp":
                total_fp,

            "fn":
                total_fn,

            "removed_voxels":
                total_removed_voxels,

            "components":
                total_components,

            "removed_components":
                total_removed_components,
        }

        results.append(
            result
        )

        print(
            f"{minimum_size:13d} | "
            f"{precision:9.6f} | "
            f"{recall:6.6f} | "
            f"{dice:6.6f} | "
            f"{iou:6.6f} | "
            f"{total_fp}"
        )

    # ========================================================
    # BEST DICE
    # ========================================================

    best_dice = max(
        results,
        key=lambda x: x["dice"]
    )

    # ========================================================
    # BEST PRECISION
    # ========================================================

    best_precision = max(
        results,
        key=lambda x: x["precision"]
    )

    # ========================================================
    # BEST DICE WITH PRECISION >= 0.93
    # ========================================================

    precision_93 = [
        result
        for result in results
        if result["precision"] >= 0.93
    ]

    if precision_93:

        best_precision_93 = max(
            precision_93,
            key=lambda x: x["dice"]
        )

    else:

        best_precision_93 = None

    # ========================================================
    # BEST DICE WITH RECALL >= 0.90
    # ========================================================

    recall_90 = [
        result
        for result in results
        if result["recall"] >= 0.90
    ]

    if recall_90:

        best_recall_90 = max(
            recall_90,
            key=lambda x: x["dice"]
        )

    else:

        best_recall_90 = None

    # ========================================================
    # LOWEST FP WITH RECALL >= 0.85
    # ========================================================

    recall_85 = [
        result
        for result in results
        if result["recall"] >= 0.85
    ]

    if recall_85:

        lowest_fp = min(
            recall_85,
            key=lambda x: x["fp"]
        )

    else:

        lowest_fp = None

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "CONNECTED-COMPONENT RECOMMENDATIONS"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Best Dice
    # --------------------------------------------------------

    print(
        "\nBEST DICE"
    )

    print(
        f"Minimum component size: "
        f"{best_dice['minimum_size']}"
    )

    print(
        f"Precision: "
        f"{best_dice['precision']:.6f}"
    )

    print(
        f"Recall: "
        f"{best_dice['recall']:.6f}"
    )

    print(
        f"Dice: "
        f"{best_dice['dice']:.6f}"
    )

    print(
        f"IoU: "
        f"{best_dice['iou']:.6f}"
    )

    print(
        f"False positives: "
        f"{best_dice['fp']}"
    )

    # --------------------------------------------------------
    # Best Precision
    # --------------------------------------------------------

    print(
        "\nBEST PRECISION"
    )

    print(
        f"Minimum component size: "
        f"{best_precision['minimum_size']}"
    )

    print(
        f"Precision: "
        f"{best_precision['precision']:.6f}"
    )

    print(
        f"Recall: "
        f"{best_precision['recall']:.6f}"
    )

    print(
        f"Dice: "
        f"{best_precision['dice']:.6f}"
    )

    print(
        f"False positives: "
        f"{best_precision['fp']}"
    )

    # --------------------------------------------------------
    # Precision >= 0.93
    # --------------------------------------------------------

    if best_precision_93 is not None:

        print(
            "\nBEST DICE WITH PRECISION >= 0.93"
        )

        print(
            f"Minimum component size: "
            f"{best_precision_93['minimum_size']}"
        )

        print(
            f"Precision: "
            f"{best_precision_93['precision']:.6f}"
        )

        print(
            f"Recall: "
            f"{best_precision_93['recall']:.6f}"
        )

        print(
            f"Dice: "
            f"{best_precision_93['dice']:.6f}"
        )

        print(
            f"False positives: "
            f"{best_precision_93['fp']}"
        )

    # --------------------------------------------------------
    # Recall >= 0.90
    # --------------------------------------------------------

    if best_recall_90 is not None:

        print(
            "\nBEST DICE WITH RECALL >= 0.90"
        )

        print(
            f"Minimum component size: "
            f"{best_recall_90['minimum_size']}"
        )

        print(
            f"Precision: "
            f"{best_recall_90['precision']:.6f}"
        )

        print(
            f"Recall: "
            f"{best_recall_90['recall']:.6f}"
        )

        print(
            f"Dice: "
            f"{best_recall_90['dice']:.6f}"
        )

        print(
            f"False positives: "
            f"{best_recall_90['fp']}"
        )

    # --------------------------------------------------------
    # Lowest FP
    # --------------------------------------------------------

    if lowest_fp is not None:

        print(
            "\nLOWEST FP WITH RECALL >= 0.85"
        )

        print(
            f"Minimum component size: "
            f"{lowest_fp['minimum_size']}"
        )

        print(
            f"Precision: "
            f"{lowest_fp['precision']:.6f}"
        )

        print(
            f"Recall: "
            f"{lowest_fp['recall']:.6f}"
        )

        print(
            f"Dice: "
            f"{lowest_fp['dice']:.6f}"
        )

        print(
            f"False positives: "
            f"{lowest_fp['fp']}"
        )

    # ========================================================
    # SAVE REPORT
    # ========================================================

    report_path = (
        OUTPUT_DIR
        /
        "connected_component_results.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "GLIOMA CONNECTED-COMPONENT "
            "FALSE-POSITIVE ANALYSIS\n"
        )

        file.write(
            "=" * 70
            +
            "\n\n"
        )

        file.write(
            f"Probability threshold: "
            f"{PROBABILITY_THRESHOLD}\n"
        )

        file.write(
            f"Connectivity: "
            f"{CONNECTIVITY}\n\n"
        )

        file.write(
            "MinimumSize,Precision,Recall,Dice,IoU,"
            "TP,FP,FN,RemovedVoxels,"
            "Components,RemovedComponents\n"
        )

        for result in results:

            file.write(
                f"{result['minimum_size']},"
                f"{result['precision']:.6f},"
                f"{result['recall']:.6f},"
                f"{result['dice']:.6f},"
                f"{result['iou']:.6f},"
                f"{result['tp']},"
                f"{result['fp']},"
                f"{result['fn']},"
                f"{result['removed_voxels']},"
                f"{result['components']},"
                f"{result['removed_components']}\n"
            )

        file.write(
            "\n\nBEST DICE\n"
        )

        file.write(
            str(best_dice)
            +
            "\n"
        )

        file.write(
            "\nBEST PRECISION\n"
        )

        file.write(
            str(best_precision)
            +
            "\n"
        )

        if best_precision_93 is not None:

            file.write(
                "\nBEST DICE WITH PRECISION >= 0.93\n"
            )

            file.write(
                str(best_precision_93)
                +
                "\n"
            )

        if best_recall_90 is not None:

            file.write(
                "\nBEST DICE WITH RECALL >= 0.90\n"
            )

            file.write(
                str(best_recall_90)
                +
                "\n"
            )

        if lowest_fp is not None:

            file.write(
                "\nLOWEST FP WITH RECALL >= 0.85\n"
            )

            file.write(
                str(lowest_fp)
                +
                "\n"
            )

    # ========================================================
    # SAVE BEST MASKS
    # ========================================================

    best_component_size = (
        best_dice[
            "minimum_size"
        ]
    )

    print(
        "\nSaving masks using best Dice "
        "component size:"
    )

    print(
        best_component_size
    )

    for case in all_case_data:

        probability = (
            case["probability"]
        )

        binary_prediction = (
            probability
            >=
            PROBABILITY_THRESHOLD
        )

        (
            filtered_prediction,
            _,
            _,
            _,
        ) = remove_small_components(
            binary_prediction,
            best_component_size,
        )

        mask_path = (
            OUTPUT_DIR
            /
            f"{case['case']}_"
            f"threshold_{PROBABILITY_THRESHOLD:.2f}_"
            f"cc_{best_component_size}_mask.npy"
        )

        np.save(
            mask_path,
            filtered_prediction.astype(
                np.uint8
            ),
        )

        print(
            "Saved:",
            mask_path
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "CONNECTED-COMPONENT ANALYSIS COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nReport:"
    )

    print(
        report_path
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
    
