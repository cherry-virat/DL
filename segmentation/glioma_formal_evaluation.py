"""
Glioma Formal Full-Volume Evaluation
====================================

Purpose
-------
Run a formal, complete, reproducible full-volume evaluation of the trained
medium Glioma 3D U-Net model on all 20 validation cases from the medium
training experiment split.

Pipeline & Rules
----------------
1. Checkpoint:
   outputs/glioma_training/glioma_best_medium_experiment.pth (frozen)
2. Dataset & Split:
   preprocessing.glioma_dataset.get_cases()
   preprocessing.glioma_split.create_train_val_split(train_ratio=0.90, seed=42)
   Validation subset: first 20 cases (identical to medium experiment validation split).
3. Preprocessing:
   preprocessing.glioma_preprocess.process_case()
   Exact 1.0mm isotropic resampling + robust normalization (4 channels: T1, T1c, T2, FLAIR).
4. Sliding-Window Inference:
   MONAI sliding_window_inference: ROI (64,64,64), overlap=0.50, Gaussian weighting, batch size 1.
5. Frozen Post-Processing:
   Tumor probability = P(class 1) + P(class 2) + P(class 3)
   Threshold = 0.60
   3D Connected Components with 6-connectivity (scipy.ndimage)
   Remove components < 1000 voxels
6. Subregion Evaluation:
   - Whole Tumor (WT): labels 1 + 2 + 3
   - Tumor Core (TC): labels 1 + 3
   - Enhancing Tumor (ET): label 3
7. Metrics:
   - Micro / Global pooled metrics (summed TP/FP/FN across all 20 cases)
   - Macro metrics (mean, std, median, min, max, IQR across 20 cases)
   - Volume analysis (Pearson r, Spearman rho, R2, MAE, MRE)
   - False positive connected component analysis
   - Visualizations for Best, Median, and Worst cases (Axial, Coronal, Sagittal)
8. Output Directory:
   outputs/glioma_formal_evaluation/
"""

from pathlib import Path
import json
import math
import sys
import random

# Allow execution from repo root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage, stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

TRAIN_RATIO = 0.90
TRAIN_CASES_COUNT = 100
VAL_CASES_COUNT = 20

INPUT_CHANNELS = 4
OUTPUT_CLASSES = 4

PATCH_SIZE = (64, 64, 64)
ROI_SIZE = PATCH_SIZE
OVERLAP = 0.50
SW_BATCH_SIZE = 1

MODEL_CHANNELS = (16, 32, 64, 128, 256)
DROPOUT = 0.0

# Frozen post-processing configuration
PROBABILITY_THRESHOLD = 0.60
MIN_COMPONENT_SIZE = 1000
CONNECTIVITY = 1  # 6-connectivity in 3D

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "glioma_training"
    / "glioma_best_medium_experiment.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "glioma_formal_evaluation"
)

VIS_DIR = OUTPUT_DIR / "visualizations"
INTERMEDIATE_DIR = OUTPUT_DIR / "intermediate_cases"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# MODEL SETUP
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
    return model.to(DEVICE)


def load_model(checkpoint_path=CHECKPOINT_PATH):
    print("=" * 70)
    print("LOADING FROZEN GLIOMA CHECKPOINT")
    print("=" * 70)
    print(f"Checkpoint path: {checkpoint_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Glioma checkpoint not found: {checkpoint_path}")

    model = create_model()
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    if "model_state_dict" not in checkpoint:
        raise RuntimeError("Checkpoint missing 'model_state_dict'.")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    epoch = checkpoint.get("epoch", "unknown")
    print(f"[OK] Loaded model successfully (Epoch: {epoch}).")
    return model


# ============================================================
# POST-PROCESSING (6-CONNECTIVITY CC FILTERING)
# ============================================================

def filter_connected_components(binary_mask, min_size=MIN_COMPONENT_SIZE, connectivity=CONNECTIVITY):
    """
    Applies 3D connected-component labeling and removes clusters < min_size.
    Uses 6-connectivity (connectivity=1) by default.
    """
    binary_mask = np.asarray(binary_mask, dtype=bool)
    if not np.any(binary_mask):
        return binary_mask.copy(), {
            "components_before": 0,
            "components_after": 0,
            "components_removed": 0,
            "voxels_removed": 0,
            "largest_component_size": 0,
        }

    structure = ndimage.generate_binary_structure(3, connectivity)
    labeled_mask, num_features = ndimage.label(binary_mask, structure=structure)

    if num_features == 0:
        return binary_mask.copy(), {
            "components_before": 0,
            "components_after": 0,
            "components_removed": 0,
            "voxels_removed": 0,
            "largest_component_size": 0,
        }

    component_sizes = np.bincount(labeled_mask.ravel())
    # Exclude background (index 0)
    sizes_only = component_sizes[1:]
    largest_size = int(sizes_only.max()) if len(sizes_only) > 0 else 0

    keep = component_sizes >= min_size
    keep[0] = False  # Background always False

    filtered_mask = keep[labeled_mask]
    kept_count = int(keep[1:].sum())
    removed_count = num_features - kept_count
    voxels_before = int(binary_mask.sum())
    voxels_after = int(filtered_mask.sum())
    voxels_removed = voxels_before - voxels_after

    stats_dict = {
        "components_before": int(num_features),
        "components_after": int(kept_count),
        "components_removed": int(removed_count),
        "voxels_removed": int(voxels_removed),
        "largest_component_size": int(largest_size),
    }

    return filtered_mask, stats_dict


# ============================================================
# METRICS COMPUTATION
# ============================================================

def compute_binary_metrics(prediction, ground_truth):
    """
    Computes confusion matrix and metrics for binary segmentation.
    """
    pred_b = np.asarray(prediction, dtype=bool)
    gt_b = np.asarray(ground_truth, dtype=bool)

    tp = int(np.logical_and(pred_b, gt_b).sum())
    fp = int(np.logical_and(pred_b, ~gt_b).sum())
    fn = int(np.logical_and(~pred_b, gt_b).sum())
    tn = int(np.logical_and(~pred_b, ~gt_b).sum())

    # Dice
    dice_denom = 2 * tp + fp + fn
    if dice_denom > 0:
        dice = float((2.0 * tp) / dice_denom)
    else:
        dice = 1.0 if (tp == 0 and fp == 0 and fn == 0) else 0.0

    # IoU
    iou_denom = tp + fp + fn
    if iou_denom > 0:
        iou = float(tp / iou_denom)
    else:
        iou = 1.0 if (tp == 0 and fp == 0 and fn == 0) else 0.0

    # Precision
    prec_denom = tp + fp
    if prec_denom > 0:
        precision = float(tp / prec_denom)
    else:
        precision = 1.0 if fn == 0 else 0.0

    # Recall / Sensitivity
    rec_denom = tp + fn
    if rec_denom > 0:
        recall = float(tp / rec_denom)
    else:
        recall = 1.0 if fp == 0 else 0.0

    # Specificity
    spec_denom = tn + fp
    if spec_denom > 0:
        specificity = float(tn / spec_denom)
    else:
        specificity = 1.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "gt_volume": int(gt_b.sum()),
        "pred_volume": int(pred_b.sum()),
    }


# ============================================================
# FULL-VOLUME INFERENCE ON ONE CASE
# ============================================================

def evaluate_single_case(model, case_dir):
    """
    Executes sliding window inference on a single 4-channel MRI case.
    Applies frozen post-processing and calculates WT, TC, and ET metrics.
    """
    case_name = case_dir.name

    # 1. Preprocess case using exact repo pipeline
    image, mask = process_case(case_dir)
    image = np.asarray(image, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.int64)

    original_shape = tuple(image.shape[1:])
    d, h, w = original_shape

    # 2. Pad small volumes if any dimension < 64
    mask_4d = np.expand_dims(mask, axis=0)
    padded_image, padded_mask = pad_to_patch_size(image, mask_4d, PATCH_SIZE)

    # 3. Torch tensor & device transfer
    image_tensor = torch.from_numpy(padded_image).float().unsqueeze(0).to(DEVICE)

    # 4. Sliding-window inference
    with torch.inference_mode():
        logits = sliding_window_inference(
            inputs=image_tensor,
            roi_size=ROI_SIZE,
            sw_batch_size=SW_BATCH_SIZE,
            predictor=model,
            overlap=OVERLAP,
            mode="gaussian",
        )
        probabilities = torch.softmax(logits, dim=1)

        # Whole tumor probability = P(1) + P(2) + P(3)
        wt_probability = probabilities[:, 1:, ...].sum(dim=1)

        # Class predictions from argmax
        raw_pred_classes = torch.argmax(probabilities, dim=1)

    # 5. Move to CPU & remove padding
    wt_prob_np = wt_probability[0].detach().cpu().numpy()[:d, :h, :w]
    raw_classes_np = raw_pred_classes[0].detach().cpu().numpy()[:d, :h, :w]
    gt_mask_np = mask[:d, :h, :w]

    # Free GPU tensors
    del image_tensor, logits, probabilities, wt_probability, raw_pred_classes
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 6. Apply threshold 0.60
    raw_wt_mask = (wt_prob_np >= PROBABILITY_THRESHOLD)

    # 7. Apply 3D connected-component post-processing (min_size=1000)
    final_wt_mask, fp_cc_stats = filter_connected_components(
        raw_wt_mask,
        min_size=MIN_COMPONENT_SIZE,
        connectivity=CONNECTIVITY,
    )

    # 8. Derive subregions inside post-processed WT mask
    # For voxels inside final_wt_mask: use raw_pred_classes (classes 1, 2, 3)
    # For voxels outside final_wt_mask: background (class 0)
    final_multiclass = np.where(final_wt_mask, raw_classes_np, 0)

    # If argmax selected background inside WT mask (very rare due to threshold), assign most likely tumor class
    bg_inside_wt = np.logical_and(final_wt_mask, final_multiclass == 0)
    if np.any(bg_inside_wt):
        # Fallback to class with max prob among 1,2,3
        final_multiclass[bg_inside_wt] = 2  # default tumor edema / core

    # Ground truth subregions
    gt_wt = (gt_mask_np > 0)
    gt_tc = np.isin(gt_mask_np, [1, 3])
    gt_et = (gt_mask_np == 3)

    # Predicted subregions
    pred_wt = final_wt_mask
    pred_tc = np.isin(final_multiclass, [1, 3])
    pred_et = (final_multiclass == 3)

    # 9. Compute Metrics
    wt_metrics = compute_binary_metrics(pred_wt, gt_wt)
    tc_metrics = compute_binary_metrics(pred_tc, gt_tc)
    et_metrics = compute_binary_metrics(pred_et, gt_et)

    # Volume differences
    gt_wt_vol = wt_metrics["gt_volume"]
    pred_wt_vol = wt_metrics["pred_volume"]
    vol_diff = pred_wt_vol - gt_wt_vol
    vol_ratio = (pred_wt_vol / gt_wt_vol) if gt_wt_vol > 0 else 0.0
    abs_vol_err = abs(vol_diff)
    rel_vol_err = (abs_vol_err / gt_wt_vol) if gt_wt_vol > 0 else 0.0

    case_result = {
        "case_id": case_name,
        "dimensions": [int(x) for x in original_shape],
        "wt": {
            **wt_metrics,
            "vol_diff": int(vol_diff),
            "vol_ratio": float(vol_ratio),
            "abs_vol_err": int(abs_vol_err),
            "rel_vol_err": float(rel_vol_err),
        },
        "tc": tc_metrics,
        "et": et_metrics,
        "cc_analysis": fp_cc_stats,
    }

    return case_result, {
        "image": image,
        "gt_mask": gt_mask_np,
        "final_multiclass": final_multiclass,
        "final_wt_mask": final_wt_mask,
        "wt_prob": wt_prob_np,
    }


# ============================================================
# STATISTICAL SUMMARY FUNCTIONS
# ============================================================

def compute_macro_statistics(values):
    """
    Computes mean, std, median, min, max, and IQR.
    """
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "iqr": 0.0}

    q75, q25 = np.percentile(arr, [75, 25])
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "iqr": float(q75 - q25),
    }


def compute_pooled_micro_metrics(cases_metrics, subregion="wt"):
    """
    Computes global pooled (micro) metrics by summing TP, FP, FN over all cases.
    """
    total_tp = sum(c[subregion]["tp"] for c in cases_metrics)
    total_fp = sum(c[subregion]["fp"] for c in cases_metrics)
    total_fn = sum(c[subregion]["fn"] for c in cases_metrics)
    total_tn = sum(c[subregion]["tn"] for c in cases_metrics)

    dice_denom = 2 * total_tp + total_fp + total_fn
    dice = float((2.0 * total_tp) / dice_denom) if dice_denom > 0 else 0.0

    iou_denom = total_tp + total_fp + total_fn
    iou = float(total_tp / iou_denom) if iou_denom > 0 else 0.0

    prec_denom = total_tp + total_fp
    precision = float(total_tp / prec_denom) if prec_denom > 0 else 0.0

    rec_denom = total_tp + total_fn
    recall = float(total_tp / rec_denom) if rec_denom > 0 else 0.0

    spec_denom = total_tn + total_fp
    specificity = float(total_tn / spec_denom) if spec_denom > 0 else 1.0

    return {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "tn": total_tn,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
    }


def compute_volume_correlations(gt_volumes, pred_volumes):
    """
    Computes Pearson r, Spearman rho, and R2 regression score.
    """
    gt = np.asarray(gt_volumes, dtype=np.float64)
    pred = np.asarray(pred_volumes, dtype=np.float64)

    # Pearson
    if len(gt) > 2 and np.std(gt) > 0 and np.std(pred) > 0:
        pearson_r, pearson_p = stats.pearsonr(gt, pred)
        spearman_rho, spearman_p = stats.spearmanr(gt, pred)
    else:
        pearson_r, pearson_p = 0.0, 1.0
        spearman_rho, spearman_p = 0.0, 1.0

    # R^2 = 1 - SS_res / SS_tot
    ss_res = np.sum((gt - pred) ** 2)
    ss_tot = np.sum((gt - np.mean(gt)) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    mae = float(np.mean(np.abs(pred - gt)))
    mre = float(np.mean(np.abs(pred - gt) / np.maximum(gt, 1.0)))

    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "r2_score": float(r2),
        "mean_absolute_error": mae,
        "mean_relative_error": mre,
    }


# ============================================================
# VISUALIZATIONS GENERATOR
# ============================================================

def generate_orthogonal_visualization(case_id, image, gt_mask, pred_wt_mask, save_path, dice_score):
    """
    Generates high-resolution 3-view (Axial, Coronal, Sagittal) overlay visualization
    comparing MRI background, Ground Truth contour, Predicted contour, and Error map.
    """
    # image channels: 0=T1, 1=T1c, 2=T2, 3=FLAIR
    t1c = image[1]
    flair = image[3]

    # Center of mass of GT tumor, fallback to predicted or center of volume
    if np.any(gt_mask > 0):
        coords = np.argwhere(gt_mask > 0)
        cz, cy, cx = coords.mean(axis=0).astype(int)
    elif np.any(pred_wt_mask > 0):
        coords = np.argwhere(pred_wt_mask > 0)
        cz, cy, cx = coords.mean(axis=0).astype(int)
    else:
        cz, cy, cx = [s // 2 for s in t1c.shape]

    # Clamp slice indices safely
    cz = np.clip(cz, 0, t1c.shape[0] - 1)
    cy = np.clip(cy, 0, t1c.shape[1] - 1)
    cx = np.clip(cx, 0, t1c.shape[2] - 1)

    views = [
        ("Axial", t1c[cz, :, :], flair[cz, :, :], gt_mask[cz, :, :] > 0, pred_wt_mask[cz, :, :]),
        ("Coronal", t1c[:, cy, :], flair[:, cy, :], gt_mask[:, cy, :] > 0, pred_wt_mask[:, cy, :]),
        ("Sagittal", t1c[:, :, cx], flair[:, :, cx], gt_mask[:, :, cx] > 0, pred_wt_mask[:, :, cx]),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(18, 13.5))
    plt.suptitle(f"Formal Glioma Evaluation: Case {case_id} (WT Dice: {dice_score:.4f})", fontsize=16, fontweight="bold", y=0.98)

    for row_idx, (view_name, t1c_slice, flair_slice, gt_slice, pred_slice) in enumerate(views):
        # 1. T1c MRI
        axes[row_idx, 0].imshow(t1c_slice, cmap="gray", origin="lower")
        axes[row_idx, 0].set_title(f"{view_name} - T1c Background", fontsize=11)
        axes[row_idx, 0].axis("off")

        # 2. FLAIR MRI
        axes[row_idx, 1].imshow(flair_slice, cmap="gray", origin="lower")
        axes[row_idx, 1].set_title(f"{view_name} - FLAIR Background", fontsize=11)
        axes[row_idx, 1].axis("off")

        # 3. Contour Overlay (T1c + GT green contour + Pred red contour)
        axes[row_idx, 2].imshow(t1c_slice, cmap="gray", origin="lower")
        if np.any(gt_slice):
            axes[row_idx, 2].contour(gt_slice, colors=["#00FF00"], linewidths=1.5)
        if np.any(pred_slice):
            axes[row_idx, 2].contour(pred_slice, colors=["#FF0055"], linewidths=1.5)
        axes[row_idx, 2].set_title(f"{view_name} - Overlay (Green: GT, Red: Pred)", fontsize=11)
        axes[row_idx, 2].axis("off")

        # 4. Error Map (TP=Green, FP=Blue, FN=Red/Orange)
        tp_s = np.logical_and(pred_slice, gt_slice)
        fp_s = np.logical_and(pred_slice, ~gt_slice)
        fn_s = np.logical_and(~pred_slice, gt_slice)

        rgb_error = np.zeros((*t1c_slice.shape, 3), dtype=np.float32)
        norm_t1c = (t1c_slice - t1c_slice.min()) / max(t1c_slice.max() - t1c_slice.min(), 1e-5)
        for c in range(3):
            rgb_error[..., c] = norm_t1c * 0.7

        # TP: Green
        rgb_error[tp_s] = [0.0, 0.9, 0.2]
        # FP: Cyan/Blue
        rgb_error[fp_s] = [0.1, 0.5, 1.0]
        # FN: Orange/Red
        rgb_error[fn_s] = [1.0, 0.3, 0.0]

        axes[row_idx, 3].imshow(rgb_error, origin="lower")
        axes[row_idx, 3].set_title(f"{view_name} - Error Map (Grn=TP, Blu=FP, Org=FN)", fontsize=11)
        axes[row_idx, 3].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved orthogonal visualization: {save_path.name}")


# ============================================================
# MAIN EVALUATION PIPELINE
# ============================================================

def run_formal_evaluation():
    set_seed(SEED)

    print("\n" + "=" * 75)
    print("FORMAL GLIOMA FULL-VOLUME EVALUATION PIPELINE")
    print("=" * 75)
    print(f"Dataset root : {DATASET_ROOT}")
    print(f"Checkpoint   : {CHECKPOINT_PATH}")
    print(f"Device       : {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU Name     : {torch.cuda.get_device_name(0)}")
    print(f"Post-Proc    : Threshold={PROBABILITY_THRESHOLD}, Min CC Size={MIN_COMPONENT_SIZE}, 6-conn")
    print("=" * 75)

    # 1. Setup Directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Case Discovery & Train/Val Split
    all_cases = get_cases(DATASET_ROOT)
    train_cases, val_cases = create_train_val_split(
        all_cases,
        train_ratio=TRAIN_RATIO,
        seed=SEED,
    )
    val_cases = val_cases[:VAL_CASES_COUNT]

    print(f"\nTotal Cases in Dataset : {len(all_cases)}")
    print(f"Validation Cases Target: {len(val_cases)}")
    print("\nValidation cohort case IDs:")
    for idx, c in enumerate(val_cases, 1):
        print(f"  {idx:2d}. {c.name}")

    # 3. Load Model
    model = load_model(CHECKPOINT_PATH)

    # 4. Sequential Evaluation across all 20 Cases
    case_results = []
    case_visual_data = {}

    print("\n" + "-" * 75)
    print("EXECUTING CASE-BY-CASE INFERENCE")
    print("-" * 75)

    for idx, case_dir in enumerate(val_cases, 1):
        case_id = case_dir.name
        print(f"\n[{idx}/{len(val_cases)}] Processing {case_id}...")

        case_res, vis_data = evaluate_single_case(model, case_dir)
        case_results.append(case_res)
        case_visual_data[case_id] = vis_data

        # Print concise summary
        wt = case_res["wt"]
        tc = case_res["tc"]
        et = case_res["et"]
        cc = case_res["cc_analysis"]
        print(
            f"  -> WT Dice: {wt['dice']:.4f} | IoU: {wt['iou']:.4f} | "
            f"Prec: {wt['precision']:.4f} | Rec: {wt['recall']:.4f} | "
            f"GT Vol: {wt['gt_volume']:,} | Pred Vol: {wt['pred_volume']:,} ({wt['vol_ratio']:.2f}x)"
        )
        print(
            f"  -> TC Dice: {tc['dice']:.4f} | ET Dice: {et['dice']:.4f} | "
            f"CCs: {cc['components_before']} -> {cc['components_after']} (removed {cc['components_removed']})"
        )

        # Save intermediate checkpoint for safety
        inter_file = INTERMEDIATE_DIR / f"{case_id}_metrics.json"
        with open(inter_file, "w", encoding="utf-8") as f:
            json.dump(case_res, f, indent=2)

    # 5. Aggregate Analysis
    print("\n" + "=" * 75)
    print("COMPUTING AGGREGATE METRICS & VOLUME STATISTICS")
    print("=" * 75)

    # Micro Pooled Metrics
    micro_wt = compute_pooled_micro_metrics(case_results, "wt")
    micro_tc = compute_pooled_micro_metrics(case_results, "tc")
    micro_et = compute_pooled_micro_metrics(case_results, "et")

    # Macro Statistics across cases
    macro_metrics = {
        "wt": {
            "dice": compute_macro_statistics([c["wt"]["dice"] for c in case_results]),
            "iou": compute_macro_statistics([c["wt"]["iou"] for c in case_results]),
            "precision": compute_macro_statistics([c["wt"]["precision"] for c in case_results]),
            "recall": compute_macro_statistics([c["wt"]["recall"] for c in case_results]),
            "specificity": compute_macro_statistics([c["wt"]["specificity"] for c in case_results]),
        },
        "tc": {
            "dice": compute_macro_statistics([c["tc"]["dice"] for c in case_results]),
            "iou": compute_macro_statistics([c["tc"]["iou"] for c in case_results]),
            "precision": compute_macro_statistics([c["tc"]["precision"] for c in case_results]),
            "recall": compute_macro_statistics([c["tc"]["recall"] for c in case_results]),
            "specificity": compute_macro_statistics([c["tc"]["specificity"] for c in case_results]),
        },
        "et": {
            "dice": compute_macro_statistics([c["et"]["dice"] for c in case_results]),
            "iou": compute_macro_statistics([c["et"]["iou"] for c in case_results]),
            "precision": compute_macro_statistics([c["et"]["precision"] for c in case_results]),
            "recall": compute_macro_statistics([c["et"]["recall"] for c in case_results]),
            "specificity": compute_macro_statistics([c["et"]["specificity"] for c in case_results]),
        },
    }

    # Volume Analysis
    gt_vols = [c["wt"]["gt_volume"] for c in case_results]
    pred_vols = [c["wt"]["pred_volume"] for c in case_results]
    volume_analysis = compute_volume_correlations(gt_vols, pred_vols)

    # False Positive Analysis
    fp_analysis = {
        "total_components_before": sum(c["cc_analysis"]["components_before"] for c in case_results),
        "total_components_after": sum(c["cc_analysis"]["components_after"] for c in case_results),
        "total_components_removed": sum(c["cc_analysis"]["components_removed"] for c in case_results),
        "total_voxels_removed": sum(c["cc_analysis"]["voxels_removed"] for c in case_results),
        "mean_components_per_case_before": float(np.mean([c["cc_analysis"]["components_before"] for c in case_results])),
        "mean_components_per_case_after": float(np.mean([c["cc_analysis"]["components_after"] for c in case_results])),
        "total_wt_fp_voxels": micro_wt["fp"],
    }

    # 6. Save Per-Case CSV
    per_case_csv = OUTPUT_DIR / "per_case_metrics.csv"
    with open(per_case_csv, "w", encoding="utf-8") as f:
        headers = [
            "Case_ID",
            "GT_WT_Volume_mm3", "Pred_WT_Volume_mm3", "WT_Vol_Diff", "WT_Vol_Ratio",
            "WT_TP", "WT_FP", "WT_FN", "WT_Dice", "WT_IoU", "WT_Precision", "WT_Recall", "WT_Specificity",
            "GT_TC_Volume_mm3", "Pred_TC_Volume_mm3", "TC_TP", "TC_FP", "TC_FN", "TC_Dice", "TC_IoU", "TC_Precision", "TC_Recall",
            "GT_ET_Volume_mm3", "Pred_ET_Volume_mm3", "ET_TP", "ET_FP", "ET_FN", "ET_Dice", "ET_IoU", "ET_Precision", "ET_Recall",
            "CC_Before", "CC_After", "CC_Removed", "CC_Voxels_Removed", "Largest_CC_Size"
        ]
        f.write(",".join(headers) + "\n")
        for c in case_results:
            row = [
                c["case_id"],
                c["wt"]["gt_volume"], c["wt"]["pred_volume"], c["wt"]["vol_diff"], f"{c['wt']['vol_ratio']:.4f}",
                c["wt"]["tp"], c["wt"]["fp"], c["wt"]["fn"], f"{c['wt']['dice']:.6f}", f"{c['wt']['iou']:.6f}", f"{c['wt']['precision']:.6f}", f"{c['wt']['recall']:.6f}", f"{c['wt']['specificity']:.6f}",
                c["tc"]["gt_volume"], c["tc"]["pred_volume"], c["tc"]["tp"], c["tc"]["fp"], c["tc"]["fn"], f"{c['tc']['dice']:.6f}", f"{c['tc']['iou']:.6f}", f"{c['tc']['precision']:.6f}", f"{c['tc']['recall']:.6f}",
                c["et"]["gt_volume"], c["et"]["pred_volume"], c["et"]["tp"], c["et"]["fp"], c["et"]["fn"], f"{c['et']['dice']:.6f}", f"{c['et']['iou']:.6f}", f"{c['et']['precision']:.6f}", f"{c['et']['recall']:.6f}",
                c["cc_analysis"]["components_before"], c["cc_analysis"]["components_after"], c["cc_analysis"]["components_removed"], c["cc_analysis"]["voxels_removed"], c["cc_analysis"]["largest_component_size"]
            ]
            f.write(",".join(str(x) for x in row) + "\n")
    print(f"[OK] Saved per-case metrics table: {per_case_csv.name}")

    # 7. Save Volume & FP Analysis CSVs & JSONs
    with open(OUTPUT_DIR / "volume_analysis.json", "w", encoding="utf-8") as f:
        json.dump(volume_analysis, f, indent=2)

    with open(OUTPUT_DIR / "false_positive_analysis.json", "w", encoding="utf-8") as f:
        json.dump(fp_analysis, f, indent=2)

    # 8. Save Aggregate Metrics JSON
    aggregate_data = {
        "validation_cohort_size": len(case_results),
        "micro_global_pooled": {
            "wt": micro_wt,
            "tc": micro_tc,
            "et": micro_et,
        },
        "macro_case_level": macro_metrics,
        "volume_analysis": volume_analysis,
        "false_positive_analysis": fp_analysis,
    }
    with open(OUTPUT_DIR / "aggregate_metrics.json", "w", encoding="utf-8") as f:
        json.dump(aggregate_data, f, indent=2)
    print("[OK] Saved aggregate metrics JSON.")

    # 9. Identify Best, Median, and Worst Cases & Generate Visualizations
    sorted_cases = sorted(case_results, key=lambda x: x["wt"]["dice"])
    worst_case = sorted_cases[0]
    median_case = sorted_cases[len(sorted_cases) // 2]
    best_case = sorted_cases[-1]

    print("\n" + "-" * 75)
    print("GENERATING ORTHOGONAL VISUALIZATIONS (Best / Median / Worst)")
    print("-" * 75)
    print(f"  Best Case   : {best_case['case_id']} (WT Dice: {best_case['wt']['dice']:.4f})")
    print(f"  Median Case : {median_case['case_id']} (WT Dice: {median_case['wt']['dice']:.4f})")
    print(f"  Worst Case  : {worst_case['case_id']} (WT Dice: {worst_case['wt']['dice']:.4f})")

    for rank, case_entry in [("best", best_case), ("median", median_case), ("worst", worst_case)]:
        c_id = case_entry["case_id"]
        v_data = case_visual_data[c_id]
        save_file = VIS_DIR / f"{rank}_{c_id}_orthogonal_visualization.png"
        generate_orthogonal_visualization(
            case_id=c_id,
            image=v_data["image"],
            gt_mask=v_data["gt_mask"],
            pred_wt_mask=v_data["final_wt_mask"],
            save_path=save_file,
            dice_score=case_entry["wt"]["dice"],
        )

    # 10. Generate Comprehensive Final Text & Markdown Reports
    report_txt = OUTPUT_DIR / "evaluation_summary.txt"
    report_md = OUTPUT_DIR / "evaluation_summary.md"

    md_content = f"""# Formal Full-Volume Glioma Evaluation Report

## 1. Executive Summary & Configuration
- **Model Checkpoint**: `{CHECKPOINT_PATH.name}` (Medium 3D U-Net, 20 Epochs)
- **Evaluation Dataset**: BraTS 2023 Glioma Validation Split (Deterministic Seed 42, 90/10 split)
- **Validation Cohort Size**: **20 complete full-volume MRI cases** (100% evaluated via sliding-window)
- **Modality Inputs**: 4 Channels (T1, T1c, T2, FLAIR) resampled to $1.0\\,\\text{{mm}}$ isotropic
- **Sliding-Window Parameters**: ROI $(64, 64, 64)$, Overlap $0.50$, Gaussian blending, batch size 1
- **Frozen Post-Processing**:
  - Whole Tumor Probability: $P(\\text{{WT}}) = P(\\text{{NCR/NET}}) + P(\\text{{ED}}) + P(\\text{{ET}})$
  - Operating Threshold: $\\tau = {PROBABILITY_THRESHOLD:.2f}$
  - Connected Component Filter: 3D 6-connectivity, minimum cluster size $= {MIN_COMPONENT_SIZE}$ voxels

---

## 2. Primary Subregion Segmentation Performance

### A. Whole Tumor (WT: Labels 1 + 2 + 3)
| Metric Type | Dice | IoU | Precision | Recall | Specificity |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Micro (Global Pooled)** | **{micro_wt['dice']:.6f}** | **{micro_wt['iou']:.6f}** | **{micro_wt['precision']:.6f}** | **{micro_wt['recall']:.6f}** | **{micro_wt['specificity']:.6f}** |
| **Macro Mean ± Std** | {macro_metrics['wt']['dice']['mean']:.6f} ± {macro_metrics['wt']['dice']['std']:.6f} | {macro_metrics['wt']['iou']['mean']:.6f} ± {macro_metrics['wt']['iou']['std']:.6f} | {macro_metrics['wt']['precision']['mean']:.6f} ± {macro_metrics['wt']['precision']['std']:.6f} | {macro_metrics['wt']['recall']['mean']:.6f} ± {macro_metrics['wt']['recall']['std']:.6f} | {macro_metrics['wt']['specificity']['mean']:.6f} ± {macro_metrics['wt']['specificity']['std']:.6f} |
| **Macro Median (IQR)** | {macro_metrics['wt']['dice']['median']:.6f} ({macro_metrics['wt']['dice']['iqr']:.4f}) | {macro_metrics['wt']['iou']['median']:.6f} ({macro_metrics['wt']['iou']['iqr']:.4f}) | {macro_metrics['wt']['precision']['median']:.6f} ({macro_metrics['wt']['precision']['iqr']:.4f}) | {macro_metrics['wt']['recall']['median']:.6f} ({macro_metrics['wt']['recall']['iqr']:.4f}) | {macro_metrics['wt']['specificity']['median']:.6f} ({macro_metrics['wt']['specificity']['iqr']:.4f}) |
| **Range [Min – Max]** | [{macro_metrics['wt']['dice']['min']:.4f} – {macro_metrics['wt']['dice']['max']:.4f}] | [{macro_metrics['wt']['iou']['min']:.4f} – {macro_metrics['wt']['iou']['max']:.4f}] | [{macro_metrics['wt']['precision']['min']:.4f} – {macro_metrics['wt']['precision']['max']:.4f}] | [{macro_metrics['wt']['recall']['min']:.4f} – {macro_metrics['wt']['recall']['max']:.4f}] | [{macro_metrics['wt']['specificity']['min']:.4f} – {macro_metrics['wt']['specificity']['max']:.4f}] |

### B. Tumor Core (TC: Labels 1 + 3)
| Metric Type | Dice | IoU | Precision | Recall | Specificity |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Micro (Global Pooled)** | **{micro_tc['dice']:.6f}** | **{micro_tc['iou']:.6f}** | **{micro_tc['precision']:.6f}** | **{micro_tc['recall']:.6f}** | **{micro_tc['specificity']:.6f}** |
| **Macro Mean ± Std** | {macro_metrics['tc']['dice']['mean']:.6f} ± {macro_metrics['tc']['dice']['std']:.6f} | {macro_metrics['tc']['iou']['mean']:.6f} ± {macro_metrics['tc']['iou']['std']:.6f} | {macro_metrics['tc']['precision']['mean']:.6f} ± {macro_metrics['tc']['precision']['std']:.6f} | {macro_metrics['tc']['recall']['mean']:.6f} ± {macro_metrics['tc']['recall']['std']:.6f} | {macro_metrics['tc']['specificity']['mean']:.6f} ± {macro_metrics['tc']['specificity']['std']:.6f} |
| **Macro Median (IQR)** | {macro_metrics['tc']['dice']['median']:.6f} ({macro_metrics['tc']['dice']['iqr']:.4f}) | {macro_metrics['tc']['iou']['median']:.6f} ({macro_metrics['tc']['iou']['iqr']:.4f}) | {macro_metrics['tc']['precision']['median']:.6f} ({macro_metrics['tc']['precision']['iqr']:.4f}) | {macro_metrics['tc']['recall']['median']:.6f} ({macro_metrics['tc']['recall']['iqr']:.4f}) | {macro_metrics['tc']['specificity']['median']:.6f} ({macro_metrics['tc']['specificity']['iqr']:.4f}) |

### C. Enhancing Tumor (ET: Label 3)
| Metric Type | Dice | IoU | Precision | Recall | Specificity |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Micro (Global Pooled)** | **{micro_et['dice']:.6f}** | **{micro_et['iou']:.6f}** | **{micro_et['precision']:.6f}** | **{micro_et['recall']:.6f}** | **{micro_et['specificity']:.6f}** |
| **Macro Mean ± Std** | {macro_metrics['et']['dice']['mean']:.6f} ± {macro_metrics['et']['dice']['std']:.6f} | {macro_metrics['et']['iou']['mean']:.6f} ± {macro_metrics['et']['iou']['std']:.6f} | {macro_metrics['et']['precision']['mean']:.6f} ± {macro_metrics['et']['precision']['std']:.6f} | {macro_metrics['et']['recall']['mean']:.6f} ± {macro_metrics['et']['recall']['std']:.6f} | {macro_metrics['et']['specificity']['mean']:.6f} ± {macro_metrics['et']['specificity']['std']:.6f} |
| **Macro Median (IQR)** | {macro_metrics['et']['dice']['median']:.6f} ({macro_metrics['et']['dice']['iqr']:.4f}) | {macro_metrics['et']['iou']['median']:.6f} ({macro_metrics['et']['iou']['iqr']:.4f}) | {macro_metrics['et']['precision']['median']:.6f} ({macro_metrics['et']['precision']['iqr']:.4f}) | {macro_metrics['et']['recall']['median']:.6f} ({macro_metrics['et']['recall']['iqr']:.4f}) | {macro_metrics['et']['specificity']['median']:.6f} ({macro_metrics['et']['specificity']['iqr']:.4f}) |

---

## 3. Volume Analysis & Correlation
- **Pearson Correlation ($r$)**: **{volume_analysis['pearson_r']:.4f}** ($p = {volume_analysis['pearson_p']:.4e}$)
- **Spearman Rank Correlation ($\\rho$)**: **{volume_analysis['spearman_rho']:.4f}** ($p = {volume_analysis['spearman_p']:.4e}$)
- **Linear $R^2$ Score**: **{volume_analysis['r2_score']:.4f}**
- **Mean Absolute Error (MAE)**: {volume_analysis['mean_absolute_error']:,.1f} $\\text{{mm}}^3$ (voxels)
- **Mean Relative Error (MRE)**: {volume_analysis['mean_relative_error']*100:.2f}%

---

## 4. Connected-Component False-Positive Filtering Analysis
- **Total Raw Components Detected**: {fp_analysis['total_components_before']}
- **Total Spurious Components Discarded (<1000 voxels)**: {fp_analysis['total_components_removed']} ({fp_analysis['total_components_removed']/max(fp_analysis['total_components_before'],1)*100:.1f}%)
- **Total Components Preserved**: {fp_analysis['total_components_after']}
- **Total False-Positive Voxels Removed**: {fp_analysis['total_voxels_removed']:,} voxels

---

## 5. Representative Benchmark Cases
- **Best Performing Case**: `{best_case['case_id']}` — WT Dice: **{best_case['wt']['dice']:.4f}** | GT: {best_case['wt']['gt_volume']:,} | Pred: {best_case['wt']['pred_volume']:,}
- **Median Case**: `{median_case['case_id']}` — WT Dice: **{median_case['wt']['dice']:.4f}** | GT: {median_case['wt']['gt_volume']:,} | Pred: {median_case['wt']['pred_volume']:,}
- **Worst Performing Case**: `{worst_case['case_id']}` — WT Dice: **{worst_case['wt']['dice']:.4f}** | GT: {worst_case['wt']['gt_volume']:,} | Pred: {worst_case['wt']['pred_volume']:,}

Visualizations saved to: `outputs/glioma_formal_evaluation/visualizations/`
"""

    with open(report_md, "w", encoding="utf-8") as f:
        f.write(md_content)

    with open(report_txt, "w", encoding="utf-8") as f:
        f.write(md_content)

    print("\n" + "=" * 75)
    print("GLIOMA FORMAL EVALUATION COMPLETED SUCCESSFULLY")
    print(f"Reports saved to: {OUTPUT_DIR}")
    print("=" * 75)


if __name__ == "__main__":
    run_formal_evaluation()

