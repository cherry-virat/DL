"""
Glioma 3D U-Net Medium Model Evaluation
========================================

Purpose
-------
Evaluate the trained medium-scale Glioma model on the
20-case validation subset.

Checkpoint
----------
outputs/glioma_training/glioma_best_medium_experiment.pth

Metrics
-------
1. Validation loss
2. Multiclass Dice
3. Tumor Dice
4. Tumor IoU
5. Tumor precision
6. Tumor recall
7. True positives
8. False positives
9. False negatives
10. Per-case metrics

This file does NOT modify the dataset or training code.
"""


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import random

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric

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

TRAIN_CASES_FOR_TEST = 100
VAL_CASES_FOR_TEST = 20

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

CHECKPOINT_PATH = (
    Path("outputs")
    / "glioma_training"
    / "glioma_best_medium_experiment.pth"
)

OUTPUT_DIR = (
    Path("outputs")
    / "glioma_evaluation_medium"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_PATH = (
    OUTPUT_DIR
    /
    "glioma_medium_evaluation_results.txt"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)

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

    if len(batch) == 0:

        raise RuntimeError(
            "Empty DataLoader batch."
        )


    patches = []


    for case_output in batch:

        if not isinstance(
            case_output,
            list,
        ):

            raise TypeError(
                "Expected each dataset item "
                "to be a list of patches."
            )


        patches.extend(
            case_output
        )


    if len(patches) == 0:

        raise RuntimeError(
            "No patches returned."
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
# MODEL
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
# CHECKPOINT LOADING
# ============================================================

def load_checkpoint(
    model,
):

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            "Checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )


    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )


    model.load_state_dict(
        checkpoint["model_state_dict"]
    )


    model.eval()


    print(
        "Checkpoint loaded successfully."
    )

    print(
        "Checkpoint:",
        CHECKPOINT_PATH,
    )


    print(
        "Checkpoint epoch:",
        checkpoint.get(
            "epoch",
            "unknown",
        ),
    )


    if "validation_tumor_dice" in checkpoint:

        dice_history = (
            checkpoint["validation_tumor_dice"]
        )

        if len(dice_history) > 0:

            print(
                "Best/last tumor Dice:",
                f"{dice_history[-1]:.6f}",
            )


    return checkpoint


# ============================================================
# CASE METRICS
# ============================================================

def calculate_case_metrics(
    predictions,
    labels,
):

    predicted_tumor = (
        predictions > 0
    )


    ground_truth_tumor = (
        labels > 0
    )


    tp = torch.logical_and(
        predicted_tumor,
        ground_truth_tumor,
    ).sum().item()


    fp = torch.logical_and(
        predicted_tumor,
        ~ground_truth_tumor,
    ).sum().item()


    fn = torch.logical_and(
        ~predicted_tumor,
        ground_truth_tumor,
    ).sum().item()


    precision = (
        tp
        /
        max(
            tp + fp,
            1,
        )
    )


    recall = (
        tp
        /
        max(
            tp + fn,
            1,
        )
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


    ground_truth_voxels = (
        ground_truth_tumor
        .sum()
        .item()
    )


    predicted_voxels = (
        predicted_tumor
        .sum()
        .item()
    )


    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "dice": dice,
        "iou": iou,
        "ground_truth": ground_truth_voxels,
        "predicted": predicted_voxels,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "GLIOMA MEDIUM MODEL EVALUATION"
    )

    print(
        "=" * 70
    )


    # ========================================================
    # ENVIRONMENT
    # ========================================================

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


    print(
        "\nCheckpoint:",
        CHECKPOINT_PATH,
    )


    # ========================================================
    # CASES
    # ========================================================

    cases = get_cases(
        DATASET_ROOT
    )


    print(
        "\nTotal Glioma cases:",
        len(cases),
    )


    # ========================================================
    # SPLIT
    # ========================================================

    train_cases, validation_cases = (
        create_train_val_split(
            cases,
            train_ratio=0.90,
            seed=SEED,
        )
    )


    validation_cases = validation_cases[
        :VAL_CASES_FOR_TEST
    ]


    train_subset = train_cases[
        :TRAIN_CASES_FOR_TEST
    ]


    # ========================================================
    # OVERLAP CHECK
    # ========================================================

    train_names = {
        case.name
        for case in train_subset
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
            f"{sorted(overlap)}"
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
            f"  {index}. {case.name}"
        )


    # ========================================================
    # DATASET
    # ========================================================

    validation_dataset = GliomaDataset(
        validation_cases
    )


    print(
        "\nValidation dataset size:",
        len(validation_dataset),
    )


    # ========================================================
    # DATALOADER
    # ========================================================

    validation_loader = DataLoader(
        validation_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0,

        pin_memory=torch.cuda.is_available(),

        collate_fn=glioma_collate_fn,
    )


    print(
        "Validation DataLoader created."
    )

    print(
        "Batch size:",
        BATCH_SIZE,
    )

    print(
        "Patches per case:",
        NUM_SAMPLES_PER_CASE,
    )

    print(
        "Patch size:",
        PATCH_SIZE,
    )


    # ========================================================
    # MODEL
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "CREATING MODEL"
    )

    print("=" * 70)


    model = create_model()


    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )


    print(
        "Total parameters:",
        f"{total_parameters:,}",
    )


    # ========================================================
    # LOAD MODEL
    # ========================================================

    checkpoint = load_checkpoint(
        model
    )


    # ========================================================
    # LOSS
    # ========================================================

    loss_function = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
    )


    dice_metric = DiceMetric(
        include_background=False,
        reduction="mean",
    )


    # ========================================================
    # AGGREGATE STATISTICS
    # ========================================================

    total_validation_loss = 0.0

    number_of_batches = 0

    total_tp = 0

    total_fp = 0

    total_fn = 0

    total_ground_truth = 0

    total_predicted = 0

    processed_patches = 0


    # --------------------------------------------------------
    # Per-class confusion statistics
    # --------------------------------------------------------

    class_tp = [0] * OUTPUT_CLASSES

    class_fp = [0] * OUTPUT_CLASSES

    class_fn = [0] * OUTPUT_CLASSES


    # --------------------------------------------------------
    # Per-case results
    # --------------------------------------------------------

    case_results = []


    # ========================================================
    # EVALUATION
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "RUNNING MEDIUM MODEL EVALUATION"
    )

    print("=" * 70)


    with torch.no_grad():

        for case_index, case in enumerate(
            validation_cases,
            start=1,
        ):

            print(
                "\n" + "-" * 70
            )

            print(
                f"CASE {case_index}/{len(validation_cases)}"
            )

            print(
                "Case:",
                case.name,
            )

            print(
                "-" * 70
            )


            case_dataset = GliomaDataset(
                [case]
            )


            case_loader = DataLoader(
                case_dataset,

                batch_size=1,

                shuffle=False,

                num_workers=0,

                pin_memory=torch.cuda.is_available(),

                collate_fn=glioma_collate_fn,
            )


            case_tp = 0

            case_fp = 0

            case_fn = 0

            case_gt = 0

            case_pred = 0


            for batch in case_loader:

                images = batch["image"]

                labels = batch["label"]


                images = images.to(
                    DEVICE,
                    non_blocking=True,
                )


                labels = labels.to(
                    DEVICE,
                    non_blocking=True,
                )


                # ------------------------------------------------
                # Forward
                # ------------------------------------------------

                outputs = model(
                    images
                )


                # ------------------------------------------------
                # Loss
                # ------------------------------------------------

                loss = loss_function(
                    outputs,
                    labels,
                )


                total_validation_loss += (
                    loss.item()
                )


                number_of_batches += 1


                # ------------------------------------------------
                # Prediction
                # ------------------------------------------------

                predictions = torch.argmax(
                    outputs,
                    dim=1,
                )


                # ------------------------------------------------
                # Multiclass Dice
                # ------------------------------------------------

                predictions_one_hot = (
                    F.one_hot(
                        predictions,
                        num_classes=OUTPUT_CLASSES,
                    )
                    .permute(
                        0,
                        4,
                        1,
                        2,
                        3,
                    )
                    .float()
                )


                labels_one_hot = (
                    F.one_hot(
                        labels.squeeze(1),
                        num_classes=OUTPUT_CLASSES,
                    )
                    .permute(
                        0,
                        4,
                        1,
                        2,
                        3,
                    )
                    .float()
                )


                dice_metric(
                    predictions_one_hot,
                    labels_one_hot,
                )


                # ------------------------------------------------
                # Tumor statistics
                # ------------------------------------------------

                result = calculate_case_metrics(
                    predictions,
                    labels.squeeze(1),
                )


                case_tp += result["tp"]

                case_fp += result["fp"]

                case_fn += result["fn"]

                case_gt += result["ground_truth"]

                case_pred += result["predicted"]


                total_tp += result["tp"]

                total_fp += result["fp"]

                total_fn += result["fn"]

                total_ground_truth += (
                    result["ground_truth"]
                )

                total_predicted += (
                    result["predicted"]
                )


                # ------------------------------------------------
                # Per-class statistics
                # ------------------------------------------------

                for class_id in range(
                    OUTPUT_CLASSES
                ):

                    prediction_class = (
                        predictions
                        ==
                        class_id
                    )


                    ground_truth_class = (
                        labels.squeeze(1)
                        ==
                        class_id
                    )


                    class_tp[class_id] += (
                        torch.logical_and(
                            prediction_class,
                            ground_truth_class,
                        )
                        .sum()
                        .item()
                    )


                    class_fp[class_id] += (
                        torch.logical_and(
                            prediction_class,
                            ~ground_truth_class,
                        )
                        .sum()
                        .item()
                    )


                    class_fn[class_id] += (
                        torch.logical_and(
                            ~prediction_class,
                            ground_truth_class,
                        )
                        .sum()
                        .item()
                    )


                processed_patches += (
                    images.shape[0]
                )


            # ----------------------------------------------------
            # Case metrics
            # ----------------------------------------------------

            case_precision = (
                case_tp
                /
                max(
                    case_tp + case_fp,
                    1,
                )
            )


            case_recall = (
                case_tp
                /
                max(
                    case_tp + case_fn,
                    1,
                )
            )


            case_dice = (
                2 * case_tp
                /
                max(
                    2 * case_tp
                    +
                    case_fp
                    +
                    case_fn,
                    1,
                )
            )


            case_iou = (
                case_tp
                /
                max(
                    case_tp
                    +
                    case_fp
                    +
                    case_fn,
                    1,
                )
            )


            volume_ratio = (
                case_pred
                /
                max(
                    case_gt,
                    1,
                )
            )


            print(
                "Ground truth tumor:",
                case_gt,
            )

            print(
                "Predicted tumor:",
                case_pred,
            )

            print(
                "True positive:",
                case_tp,
            )

            print(
                "False positive:",
                case_fp,
            )

            print(
                "False negative:",
                case_fn,
            )

            print(
                "Precision:",
                f"{case_precision:.6f}",
            )

            print(
                "Recall:",
                f"{case_recall:.6f}",
            )

            print(
                "Dice:",
                f"{case_dice:.6f}",
            )

            print(
                "IoU:",
                f"{case_iou:.6f}",
            )

            print(
                "Prediction / GT volume:",
                f"{volume_ratio:.4f}x",
            )


            case_results.append(
                {
                    "case": case.name,

                    "ground_truth":
                        case_gt,

                    "predicted":
                        case_pred,

                    "tp":
                        case_tp,

                    "fp":
                        case_fp,

                    "fn":
                        case_fn,

                    "precision":
                        case_precision,

                    "recall":
                        case_recall,

                    "dice":
                        case_dice,

                    "iou":
                        case_iou,

                    "volume_ratio":
                        volume_ratio,
                }
            )


    # ========================================================
    # AGGREGATE METRICS
    # ========================================================

    validation_loss = (
        total_validation_loss
        /
        max(
            number_of_batches,
            1,
        )
    )


    global_precision = (
        total_tp
        /
        max(
            total_tp + total_fp,
            1,
        )
    )


    global_recall = (
        total_tp
        /
        max(
            total_tp + total_fn,
            1,
        )
    )


    global_dice = (
        2 * total_tp
        /
        max(
            2 * total_tp
            +
            total_fp
            +
            total_fn,
            1,
        )
    )


    global_iou = (
        total_tp
        /
        max(
            total_tp
            +
            total_fp
            +
            total_fn,
            1,
        )
    )


    multiclass_dice = (
        dice_metric
        .aggregate()
        .item()
    )


    dice_metric.reset()


    # ========================================================
    # PER-CLASS METRICS
    # ========================================================

    per_class_dice = []

    per_class_iou = []

    per_class_precision = []

    per_class_recall = []


    for class_id in range(
        OUTPUT_CLASSES
    ):

        tp = class_tp[class_id]

        fp = class_fp[class_id]

        fn = class_fn[class_id]


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


        precision = (
            tp
            /
            max(
                tp + fp,
                1,
            )
        )


        recall = (
            tp
            /
            max(
                tp + fn,
                1,
            )
        )


        per_class_dice.append(
            dice
        )

        per_class_iou.append(
            iou
        )

        per_class_precision.append(
            precision
        )

        per_class_recall.append(
            recall
        )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "MEDIUM MODEL EVALUATION RESULTS"
    )

    print("=" * 70)


    print(
        "\nValidation loss:",
        f"{validation_loss:.6f}",
    )


    print(
        "\nMulticlass Dice:",
        f"{multiclass_dice:.6f}",
    )


    print(
        "\nTumor Dice:",
        f"{global_dice:.6f}",
    )


    print(
        "Tumor IoU:",
        f"{global_iou:.6f}",
    )


    print(
        "Tumor Precision:",
        f"{global_precision:.6f}",
    )


    print(
        "Tumor Recall:",
        f"{global_recall:.6f}",
    )


    print(
        "\nTumor voxel statistics:"
    )


    print(
        "Ground truth:",
        total_ground_truth,
    )


    print(
        "Predicted:",
        total_predicted,
    )


    print(
        "True positives:",
        total_tp,
    )


    print(
        "False positives:",
        total_fp,
    )


    print(
        "False negatives:",
        total_fn,
    )


    print(
        "\nPer-class metrics:"
    )


    for class_id in range(
        OUTPUT_CLASSES
    ):

        print(
            f"\nClass {class_id}:"
        )

        print(
            "  Dice:",
            f"{per_class_dice[class_id]:.6f}",
        )

        print(
            "  IoU:",
            f"{per_class_iou[class_id]:.6f}",
        )

        print(
            "  Precision:",
            f"{per_class_precision[class_id]:.6f}",
        )

        print(
            "  Recall:",
            f"{per_class_recall[class_id]:.6f}",
        )


    # ========================================================
    # WORST CASES
    # ========================================================

    worst_cases = sorted(
        case_results,
        key=lambda x: x["precision"],
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "LOWEST-PRECISION CASES"
    )

    print("=" * 70)


    for result in worst_cases[:5]:

        print(
            f"{result['case']}: "
            f"precision={result['precision']:.6f}, "
            f"Dice={result['dice']:.6f}, "
            f"FP={result['fp']}, "
            f"volume={result['volume_ratio']:.3f}x"
        )


    # ========================================================
    # SAVE REPORT
    # ========================================================

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "GLIOMA MEDIUM MODEL EVALUATION\n"
        )

        file.write(
            "=" * 70
            +
            "\n\n"
        )


        file.write(
            f"Checkpoint: {CHECKPOINT_PATH}\n"
        )

        file.write(
            f"Validation cases: "
            f"{len(validation_cases)}\n\n"
        )


        file.write(
            f"Validation loss: "
            f"{validation_loss:.6f}\n"
        )

        file.write(
            f"Multiclass Dice: "
            f"{multiclass_dice:.6f}\n"
        )

        file.write(
            f"Tumor Dice: "
            f"{global_dice:.6f}\n"
        )

        file.write(
            f"Tumor IoU: "
            f"{global_iou:.6f}\n"
        )

        file.write(
            f"Tumor Precision: "
            f"{global_precision:.6f}\n"
        )

        file.write(
            f"Tumor Recall: "
            f"{global_recall:.6f}\n\n"
        )


        file.write(
            "Tumor voxel statistics\n"
        )

        file.write(
            "-" * 50
            +
            "\n"
        )


        file.write(
            f"Ground truth: "
            f"{total_ground_truth}\n"
        )

        file.write(
            f"Predicted: "
            f"{total_predicted}\n"
        )

        file.write(
            f"True positives: "
            f"{total_tp}\n"
        )

        file.write(
            f"False positives: "
            f"{total_fp}\n"
        )

        file.write(
            f"False negatives: "
            f"{total_fn}\n\n"
        )


        file.write(
            "Per-class metrics\n"
        )

        file.write(
            "-" * 50
            +
            "\n"
        )


        for class_id in range(
            OUTPUT_CLASSES
        ):

            file.write(
                f"\nClass {class_id}\n"
            )

            file.write(
                f"Dice: "
                f"{per_class_dice[class_id]:.6f}\n"
            )

            file.write(
                f"IoU: "
                f"{per_class_iou[class_id]:.6f}\n"
            )

            file.write(
                f"Precision: "
                f"{per_class_precision[class_id]:.6f}\n"
            )

            file.write(
                f"Recall: "
                f"{per_class_recall[class_id]:.6f}\n"
            )


        file.write(
            "\nPer-case results\n"
        )

        file.write(
            "-" * 50
            +
            "\n"
        )


        for result in case_results:

            file.write(
                f"\n{result['case']}\n"
            )

            file.write(
                f"Ground truth: "
                f"{result['ground_truth']}\n"
            )

            file.write(
                f"Predicted: "
                f"{result['predicted']}\n"
            )

            file.write(
                f"TP: {result['tp']}\n"
            )

            file.write(
                f"FP: {result['fp']}\n"
            )

            file.write(
                f"FN: {result['fn']}\n"
            )

            file.write(
                f"Precision: "
                f"{result['precision']:.6f}\n"
            )

            file.write(
                f"Recall: "
                f"{result['recall']:.6f}\n"
            )

            file.write(
                f"Dice: "
                f"{result['dice']:.6f}\n"
            )

            file.write(
                f"IoU: "
                f"{result['iou']:.6f}\n"
            )

            file.write(
                f"Volume ratio: "
                f"{result['volume_ratio']:.6f}x\n"
            )


    print(
        "\nReport saved:"
    )

    print(
        REPORT_PATH
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
            f"{allocated:.2f} MB"
        )

        print(
            "Reserved:",
            f"{reserved:.2f} MB"
        )


    print(
        "\n" + "=" * 70
    )

    print(
        "MEDIUM MODEL EVALUATION COMPLETE"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()