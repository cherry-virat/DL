# ============================================================
# MENINGIOMA SWIN UNETR THRESHOLD & POST-PROCESSING ANALYSIS
# ============================================================
#
# Purpose:
#   Isolated evaluation of probability thresholds and 3D
#   connected-component filtering on the development set
#   for the frozen Meningioma Swin UNETR best checkpoint.
#
# Checkpoint:
#   outputs/meningioma_swin_unetr/checkpoints/meningioma_swin_best.pth
#
# Split:
#   data/meningioma_split.json (100 development cases)
#
# Isolated Outputs:
#   outputs/meningioma_swin_unetr/threshold_analysis/
#     - swin_threshold_analysis_report.json
#     - swin_threshold_analysis_summary.csv
#     - swin_threshold_analysis_summary.md
# ============================================================

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from scipy import ndimage
import torch
import torch.nn.functional as F

from models.meningioma_swin_unetr import create_meningioma_swin_unetr


# ============================================================
# PATHS
# ============================================================

PREPROCESSED_ROOT = PROJECT_ROOT / "data" / "meningioma_preprocessed"
SPLIT_FILE = PROJECT_ROOT / "data" / "meningioma_split.json"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "meningioma_swin_unetr"
    / "checkpoints"
    / "meningioma_swin_best.pth"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "meningioma_swin_unetr"
    / "threshold_analysis"
)


# ============================================================
# ANALYSIS GRID CONFIGURATION
# ============================================================

THRESHOLDS = [
    0.10,
    0.125,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
]

# 0 represents no filter (raw thresholded prediction)
MIN_COMPONENT_SIZES = [
    0,
    10,
    50,
    100,
    250,
    500,
    1000,
    2500,
]

# Standard 26-connectivity 3D structuring element matching baseline
STRUCTURE = np.ones((3, 3, 3), dtype=np.uint8)


# ============================================================
# SLIDING-WINDOW FULL-VOLUME INFERENCE
# ============================================================

def pad_volume(
    image: np.ndarray,
    patch_size: Tuple[int, int, int]
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Symmetric constant padding for volumes smaller than patch size."""
    pads = []
    for dim, patch_dim in zip(image.shape, patch_size):
        if dim < patch_dim:
            total_pad = patch_dim - dim
            before = total_pad // 2
            after = total_pad - before
        else:
            before, after = 0, 0
        pads.append((before, after))

    if any(b > 0 or a > 0 for b, a in pads):
        padded = np.pad(image, pads, mode="constant", constant_values=0)
    else:
        padded = image
    return padded, pads


def compute_starts(dim: int, patch_dim: int, overlap: float) -> List[int]:
    """Generates window starting indices along an axis with overlap."""
    if dim <= patch_dim:
        return [0]
    stride = int(patch_dim * (1.0 - overlap))
    if stride <= 0:
        stride = 1
    starts = list(range(0, dim - patch_dim + 1, stride))
    final_start = dim - patch_dim
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


@torch.no_grad()
def predict_full_volume_probabilities(
    model: torch.nn.Module,
    image: np.ndarray,
    patch_size: Tuple[int, int, int] = (64, 64, 64),
    overlap: float = 0.50,
    device: torch.device = torch.device("cuda"),
) -> Tuple[np.ndarray, int]:
    """
    Performs sliding-window inference and returns full-volume tumor probability map.
    Matches evaluate_meningioma_swin_unetr.py exactly.
    """
    original_shape = image.shape
    padded_image, pads = pad_volume(image, patch_size)
    padded_shape = padded_image.shape

    starts_d = compute_starts(padded_shape[0], patch_size[0], overlap)
    starts_h = compute_starts(padded_shape[1], patch_size[1], overlap)
    starts_w = compute_starts(padded_shape[2], patch_size[2], overlap)

    total_windows = len(starts_d) * len(starts_h) * len(starts_w)
    prob_sum = np.zeros(padded_shape, dtype=np.float32)
    pred_count = np.zeros(padded_shape, dtype=np.float32)

    for sd in starts_d:
        for sh in starts_h:
            for sw in starts_w:
                ed = sd + patch_size[0]
                eh = sh + patch_size[1]
                ew = sw + patch_size[2]

                patch = padded_image[sd:ed, sh:eh, sw:ew]
                patch_tensor = (
                    torch.from_numpy(patch.astype(np.float32, copy=False))
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(device)
                )

                if device.type == "cuda":
                    with torch.amp.autocast('cuda', enabled=True):
                        logits = model(patch_tensor)
                else:
                    logits = model(patch_tensor)

                probs = F.softmax(logits, dim=1)
                tumor_prob = probs[0, 1].detach().cpu().numpy()

                prob_sum[sd:ed, sh:eh, sw:ew] += tumor_prob
                pred_count[sd:ed, sh:eh, sw:ew] += 1.0

    avg_prob = prob_sum / np.maximum(pred_count, 1.0)

    # Invert padding
    crop_slices = []
    for axis, (before, after) in enumerate(pads):
        start = before
        stop = avg_prob.shape[axis] - after if after > 0 else None
        crop_slices.append(slice(start, stop))

    cropped_prob = avg_prob[tuple(crop_slices)]
    assert (
        cropped_prob.shape == original_shape
    ), f"Shape mismatch: {cropped_prob.shape} vs {original_shape}"
    return cropped_prob.astype(np.float32), total_windows


# ============================================================
# CONNECTED COMPONENT FILTERING
# ============================================================

def filter_connected_components(
    binary_mask: np.ndarray,
    min_size: int,
    structure: np.ndarray = STRUCTURE,
) -> np.ndarray:
    """
    Filters out connected components smaller than min_size voxels.
    Uses 26-connectivity matching the baseline implementation.
    """
    if min_size <= 0:
        return binary_mask.astype(bool)

    binary_bool = binary_mask.astype(bool)
    labeled, num_features = ndimage.label(binary_bool, structure=structure)
    if num_features == 0:
        return np.zeros_like(binary_bool, dtype=bool)

    component_sizes = np.bincount(labeled.ravel())
    keep = component_sizes >= min_size
    keep[0] = False  # background is never kept

    filtered = keep[labeled]
    return filtered.astype(bool)


# ============================================================
# METRICS COMPUTATION
# ============================================================

def compute_binary_metrics(
    pred_bool: np.ndarray,
    target_bool: np.ndarray,
) -> Dict[str, Any]:
    """Computes voxel confusion matrix and overlap metrics."""
    tp = int(np.logical_and(pred_bool, target_bool).sum())
    fp = int(np.logical_and(pred_bool, ~target_bool).sum())
    fn = int(np.logical_and(~pred_bool, target_bool).sum())
    tn = int(np.logical_and(~pred_bool, ~target_bool).sum())

    pred_voxels = int(pred_bool.sum())
    gt_voxels = int(target_bool.sum())

    denom_dice = pred_voxels + gt_voxels
    if denom_dice == 0:
        dice = 1.0
    else:
        dice = (2.0 * tp) / denom_dice

    denom_iou = tp + fp + fn
    if denom_iou == 0:
        iou = 1.0
    else:
        iou = tp / denom_iou

    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if gt_voxels == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if pred_voxels == 0 else 0.0)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "pred_voxels": pred_voxels,
        "gt_voxels": gt_voxels,
    }


# ============================================================
# THRESHOLD & FILTER SWEEP RUNNER
# ============================================================

def run_threshold_sweep(
    checkpoint_path: Path,
    output_dir: Path,
    thresholds: List[float] = THRESHOLDS,
    min_component_sizes: List[int] = MIN_COMPONENT_SIZES,
    overlap: float = 0.50,
    feature_size: int = 12,
    device_str: str = "cuda",
    smoke_test: bool = False,
    smoke_test_count: int = 2,
) -> Dict[str, Any]:
    print("=" * 80)
    print("MENINGIOMA SWIN UNETR THRESHOLD & CC-FILTER ANALYSIS")
    print("=" * 80)

    device = torch.device(
        device_str
        if torch.cuda.is_available() or device_str == "cpu"
        else "cpu"
    )
    print(f"Device:               {device}")
    if device.type == "cuda":
        print(f"GPU:                  {torch.cuda.get_device_name(0)}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load split
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(f"Split file not found: {SPLIT_FILE}")

    with open(SPLIT_FILE, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    dev_cases = list(split_data["dev_cases"])
    if smoke_test:
        dev_cases = dev_cases[:smoke_test_count]

    print(f"Checkpoint:           {checkpoint_path}")
    print(f"Feature size:         {feature_size}")
    print(f"Overlap:              {overlap}")
    print(f"Smoke test:           {smoke_test} (cases: {len(dev_cases)})")
    print(f"Thresholds ({len(thresholds)}):      {thresholds}")
    print(f"Min component sizes ({len(min_component_sizes)}): {min_component_sizes}")
    print(f"Total grid points:    {len(thresholds) * len(min_component_sizes)}")

    # Load model
    print("\nLoading Swin UNETR model...")
    model = create_meningioma_swin_unetr(
        in_channels=1,
        out_channels=2,
        feature_size=feature_size,
        use_checkpoint=True,
    )

    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_epoch = ckpt.get("epoch", "unknown")
    ckpt_best_dice = ckpt.get("best_dice", "unknown")

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.to(device)
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize results grid: grid_results[(tau, min_size)] = list of case metric dicts
    grid_results: Dict[Tuple[float, int], List[Dict[str, Any]]] = {
        (tau, ms): [] for tau in thresholds for ms in min_component_sizes
    }

    start_time = time.time()

    print("\n" + "=" * 80)
    print("EXECUTING INFERENCE & GRID EVALUATION ACROSS CASES")
    print("=" * 80)

    for case_idx, case_name in enumerate(dev_cases, start=1):
        case_start = time.time()
        case_dir = PREPROCESSED_ROOT / case_name
        t1c_path = case_dir / "t1c.npy"
        gtv_path = case_dir / "gtv.npy"

        if not t1c_path.exists() or not gtv_path.exists():
            print(f"  [WARN] Skipping case {case_name}: files missing on disk.")
            continue

        image = np.load(t1c_path).astype(np.float32)
        target = np.load(gtv_path).astype(np.uint8)
        target_bool = (target > 0)

        # 1. Full-volume inference (run ONCE per case)
        prob_map, n_windows = predict_full_volume_probabilities(
            model=model,
            image=image,
            patch_size=(64, 64, 64),
            overlap=overlap,
            device=device,
        )

        # 2. Evaluate all thresholds on this probability map
        for tau in thresholds:
            raw_binary = (prob_map >= tau)

            # 3. Evaluate all connected component sizes
            for min_size in min_component_sizes:
                if min_size == 0:
                    pred_mask = raw_binary
                else:
                    pred_mask = filter_connected_components(
                        raw_binary, min_size=min_size, structure=STRUCTURE
                    )

                metrics = compute_binary_metrics(pred_mask, target_bool)
                metrics["case_name"] = case_name
                metrics["threshold"] = tau
                metrics["min_component_size"] = min_size

                grid_results[(tau, min_size)].append(metrics)

        # Discard prob_map from memory
        del prob_map, image, target, target_bool
        if device.type == "cuda":
            torch.cuda.empty_cache()

        elapsed_case = time.time() - case_start
        print(
            f"  [{case_idx:03d}/{len(dev_cases):03d}] {case_name:30s} | "
            f"Windows: {n_windows:4d} | Grid points evaluated: {len(thresholds) * len(min_component_sizes)} | "
            f"Time: {elapsed_case:.1f}s"
        )

    total_duration = time.time() - start_time
    print("\nInference and grid evaluation complete in {:.1f}s.".format(total_duration))

    # ========================================================
    # AGGREGATE RESULTS ACROSS GRID POINTS
    # ========================================================
    summary_rows = []
    detailed_configurations = []

    for tau in thresholds:
        for min_size in min_component_sizes:
            case_metrics_list = grid_results[(tau, min_size)]
            num_cases = len(case_metrics_list)

            if num_cases == 0:
                continue

            dices = [m["dice"] for m in case_metrics_list]
            ious = [m["iou"] for m in case_metrics_list]
            precisions = [m["precision"] for m in case_metrics_list]
            recalls = [m["recall"] for m in case_metrics_list]

            total_tp = sum(m["tp"] for m in case_metrics_list)
            total_fp = sum(m["fp"] for m in case_metrics_list)
            total_fn = sum(m["fn"] for m in case_metrics_list)
            total_tn = sum(m["tn"] for m in case_metrics_list)

            denom_micro = 2 * total_tp + total_fp + total_fn
            micro_dice = (2.0 * total_tp) / denom_micro if denom_micro > 0 else 0.0

            cases_zero_dice = sum(1 for d in dices if d == 0.0)
            cases_pos_dice = sum(1 for d in dices if d > 0.0)
            cases_ge_50 = sum(1 for d in dices if d >= 0.50)
            cases_ge_70 = sum(1 for d in dices if d >= 0.70)
            cases_ge_80 = sum(1 for d in dices if d >= 0.80)

            mean_pred_voxels = float(np.mean([m["pred_voxels"] for m in case_metrics_list]))
            mean_gt_voxels = float(np.mean([m["gt_voxels"] for m in case_metrics_list]))

            summary_item = {
                "threshold": tau,
                "min_component_size": min_size,
                "num_cases": num_cases,
                "mean_dice": float(np.mean(dices)),
                "std_dice": float(np.std(dices)),
                "median_dice": float(np.median(dices)),
                "min_dice": float(np.min(dices)),
                "max_dice": float(np.max(dices)),
                "mean_iou": float(np.mean(ious)),
                "median_iou": float(np.median(ious)),
                "mean_precision": float(np.mean(precisions)),
                "mean_recall": float(np.mean(recalls)),
                "micro_dice": float(micro_dice),
                "total_tp": total_tp,
                "total_fp": total_fp,
                "total_fn": total_fn,
                "cases_with_dice_gt_0": cases_pos_dice,
                "cases_with_zero_dice": cases_zero_dice,
                "cases_with_dice_ge_0_50": cases_ge_50,
                "cases_with_dice_ge_0_70": cases_ge_70,
                "cases_with_dice_ge_0_80": cases_ge_80,
                "mean_pred_voxels": mean_pred_voxels,
                "mean_gt_voxels": mean_gt_voxels,
            }
            summary_rows.append(summary_item)

            detailed_configurations.append({
                "configuration": summary_item,
                "case_metrics": case_metrics_list,
            })

    # Find best configuration by mean Dice
    best_config = max(summary_rows, key=lambda x: x["mean_dice"])

    # Baseline comparison point (tau=0.50, min_component_size=0)
    baseline_point = next(
        (r for r in summary_rows if r["threshold"] == 0.50 and r["min_component_size"] == 0),
        None,
    )

    report_data = {
        "experiment": "meningioma_swin_unetr_threshold_analysis",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": ckpt_epoch,
        "checkpoint_best_patch_dice": ckpt_best_dice,
        "feature_size": feature_size,
        "overlap": overlap,
        "cases_evaluated": len(dev_cases),
        "total_duration_sec": total_duration,
        "thresholds_swept": thresholds,
        "min_component_sizes_swept": min_component_sizes,
        "best_configuration_by_mean_dice": best_config,
        "reference_point_tau_0_50_no_filter": baseline_point,
        "summary": summary_rows,
        "detailed_results": detailed_configurations,
    }

    # ========================================================
    # 1. SAVE JSON REPORT
    # ========================================================
    json_path = output_dir / "swin_threshold_analysis_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\n[OK] Detailed JSON report written to: {json_path}")

    # ========================================================
    # 2. SAVE CSV SUMMARY (FOR PLOTTING / SPREADSHEETS)
    # ========================================================
    csv_path = output_dir / "swin_threshold_analysis_summary.csv"
    fieldnames = [
        "threshold",
        "min_component_size",
        "num_cases",
        "mean_dice",
        "median_dice",
        "std_dice",
        "mean_iou",
        "median_iou",
        "mean_precision",
        "mean_recall",
        "micro_dice",
        "cases_with_dice_gt_0",
        "cases_with_zero_dice",
        "cases_with_dice_ge_0_50",
        "cases_with_dice_ge_0_70",
        "cases_with_dice_ge_0_80",
        "mean_pred_voxels",
        "mean_gt_voxels",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)
    print(f"[OK] Summary CSV written to: {csv_path}")

    # ========================================================
    # 3. SAVE CONCISE MARKDOWN REPORT
    # ========================================================
    md_path = output_dir / "swin_threshold_analysis_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Meningioma Swin UNETR Threshold & Connected-Component Analysis\n\n")
        f.write(f"- **Checkpoint**: `{checkpoint_path.name}` (Epoch: {ckpt_epoch})\n")
        f.write(f"- **Cases Evaluated**: {len(dev_cases)}\n")
        f.write(f"- **Sliding-Window Overlap**: {overlap}\n\n")

        f.write("## Best Operating Point\n\n")
        f.write(f"- **Optimal Threshold**: `{best_config['threshold']:.3f}`\n")
        f.write(f"- **Min Component Size**: `{best_config['min_component_size']}` voxels\n")
        f.write(f"- **Mean Case Dice**: **{best_config['mean_dice']:.4f}** ± {best_config['std_dice']:.4f}\n")
        f.write(f"- **Median Case Dice**: **{best_config['median_dice']:.4f}**\n")
        f.write(f"- **Mean Case IoU**: {best_config['mean_iou']:.4f}\n")
        f.write(f"- **Micro/Global Dice**: {best_config['micro_dice']:.4f}\n")
        f.write(f"- **Cases with Dice > 0**: {best_config['cases_with_dice_gt_0']} / {len(dev_cases)}\n")
        f.write(f"- **Cases with Dice ≥ 0.50**: {best_config['cases_with_dice_ge_0_50']} / {len(dev_cases)}\n\n")

        if baseline_point:
            f.write("## Comparison to Raw Reference (Threshold 0.50, No Filter)\n\n")
            f.write(f"- **Raw Reference Mean Dice**: {baseline_point['mean_dice']:.4f}\n")
            f.write(f"- **Raw Reference Median Dice**: {baseline_point['median_dice']:.4f}\n")
            f.write(f"- **Raw Reference Micro Dice**: {baseline_point['micro_dice']:.4f}\n")
            f.write(f"- **Delta in Mean Dice**: **{best_config['mean_dice'] - baseline_point['mean_dice']:+.4f}**\n\n")

        f.write("## Top 10 Configurations by Mean Case Dice\n\n")
        f.write("| Rank | Threshold | Min Component Size | Mean Dice | Median Dice | Mean IoU | Micro Dice | Cases Dice ≥ 0.50 |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        top_sorted = sorted(summary_rows, key=lambda x: x["mean_dice"], reverse=True)[:10]
        for rank, item in enumerate(top_sorted, start=1):
            f.write(
                f"| {rank} | {item['threshold']:.3f} | {item['min_component_size']} | "
                f"**{item['mean_dice']:.4f}** | {item['median_dice']:.4f} | "
                f"{item['mean_iou']:.4f} | {item['micro_dice']:.4f} | "
                f"{item['cases_with_dice_ge_0_50']}/{len(dev_cases)} |\n"
            )

    print(f"[OK] Markdown report written to: {md_path}")

    # Print summary table to console
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Optimal Configuration:  Threshold = {best_config['threshold']}, Min Component Size = {best_config['min_component_size']}")
    print(f"Best Mean Case Dice:    {best_config['mean_dice']:.4f} +- {best_config['std_dice']:.4f}")
    print(f"Best Median Case Dice:  {best_config['median_dice']:.4f}")
    print(f"Best Micro/Global Dice: {best_config['micro_dice']:.4f}")
    if baseline_point:
        print(f"Baseline (tau=0.50, no filter) Mean Dice: {baseline_point['mean_dice']:.4f}")
        print(f"Improvement over tau=0.50:               {best_config['mean_dice'] - baseline_point['mean_dice']:+.4f}")
    print("=" * 80)

    return report_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Meningioma Swin UNETR Threshold & Connected-Component Analysis"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="Path to Swin UNETR checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save analysis reports",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.50,
        help="Sliding-window overlap (default: 0.50)",
    )
    parser.add_argument(
        "--feature-size",
        type=int,
        default=12,
        help="Swin UNETR feature size (default: 12)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference (cuda or cpu)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run on small subset of development cases for verification",
    )
    parser.add_argument(
        "--smoke-test-count",
        type=int,
        default=2,
        help="Number of cases to process in smoke-test mode (default: 2)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_threshold_sweep(
        checkpoint_path=Path(args.checkpoint),
        output_dir=Path(args.output_dir),
        overlap=args.overlap,
        feature_size=args.feature_size,
        device_str=args.device,
        smoke_test=args.smoke_test,
        smoke_test_count=args.smoke_test_count,
    )

