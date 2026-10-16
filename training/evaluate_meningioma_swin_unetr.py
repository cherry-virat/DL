# ============================================================
# MENINGIOMA SWIN UNETR EVALUATION PIPELINE
# ============================================================
#
# Dataset:
#   BraTS2024-MEN-RT Development Split (100 Cases)
#
# Task:
#   Full-volume sliding window evaluation of Meningioma Swin UNETR.
#
# Output:
#   outputs/meningioma_swin_unetr/metrics/development_metrics.json
# ============================================================

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from models.meningioma_swin_unetr import create_meningioma_swin_unetr


# ============================================================
# PATHS
# ============================================================

PREPROCESSED_ROOT = PROJECT_ROOT / "data" / "meningioma_preprocessed"
SPLIT_FILE = PROJECT_ROOT / "data" / "meningioma_split.json"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs" / "meningioma_swin_unetr" / "checkpoints" / "meningioma_swin_best.pth"
DEFAULT_METRICS_DIR = PROJECT_ROOT / "outputs" / "meningioma_swin_unetr" / "metrics"


# ============================================================
# SLIDING WINDOW INFERENCE UTILITIES
# ============================================================

def pad_volume(image: np.ndarray, patch_size: Tuple[int, int, int]) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
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
def sliding_window_predict(
    model: torch.nn.Module,
    image: np.ndarray,
    patch_size: Tuple[int, int, int] = (64, 64, 64),
    overlap: float = 0.50,
    device: torch.device = torch.device("cpu"),
) -> Tuple[np.ndarray, int]:
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
                patch_tensor = torch.from_numpy(patch.astype(np.float32, copy=False)).unsqueeze(0).unsqueeze(0).to(device)

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
    assert cropped_prob.shape == original_shape, f"Shape mismatch: {cropped_prob.shape} vs {original_shape}"
    return cropped_prob.astype(np.float32), total_windows


# ============================================================
# EVALUATION METRIC CALCULATIONS
# ============================================================

def calculate_case_metrics(pred_binary: np.ndarray, target_binary: np.ndarray) -> Dict[str, float]:
    p = pred_binary.astype(bool)
    g = target_binary.astype(bool)

    tp = np.logical_and(p, g).sum()
    fp = np.logical_and(p, ~g).sum()
    fn = np.logical_and(~p, g).sum()
    tn = np.logical_and(~p, ~g).sum()

    p_sum = p.sum()
    g_sum = g.sum()

    # Dice
    denom_dice = p_sum + g_sum
    if denom_dice == 0:
        dice = 1.0
    else:
        dice = (2.0 * tp) / denom_dice

    # IoU / Jaccard
    denom_iou = tp + fp + fn
    if denom_iou == 0:
        iou = 1.0
    else:
        iou = tp / denom_iou

    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0 if g_sum == 0 else 0.0

    # Recall / Sensitivity
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0 if p_sum == 0 else 0.0

    # Specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 1.0

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "pred_voxels": int(p_sum),
        "gt_voxels": int(g_sum),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


# ============================================================
# EVALUATION RUNNER
# ============================================================

def evaluate_dev_set(
    checkpoint_path: Path,
    output_dir: Path,
    threshold: float = 0.50,
    overlap: float = 0.50,
    feature_size: int = 12,
    device_str: str = "cuda",
    smoke_test: bool = False,
    save_predictions: bool = False,
) -> Dict[str, Any]:
    print("=" * 80)
    print("MENINGIOMA SWIN UNETR FULL-VOLUME EVALUATION")
    print("=" * 80)

    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")
    print(f"Device:               {device}")
    if device.type == "cuda":
        print(f"GPU:                  {torch.cuda.get_device_name(0)}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Checkpoint:           {checkpoint_path}")
    print(f"Feature size:         {feature_size}")
    print(f"Threshold:            {threshold}")
    print(f"Overlap:              {overlap}")
    print(f"Smoke test mode:      {smoke_test}")

    # Load split
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(f"Split file not found: {SPLIT_FILE}")

    with open(SPLIT_FILE, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    dev_cases = list(split_data["dev_cases"])
    if smoke_test:
        dev_cases = dev_cases[:2]

    print(f"Total cases to eval:  {len(dev_cases)}")

    # Load Model
    print("\nLoading Swin UNETR model...")
    model = create_meningioma_swin_unetr(
        in_channels=1,
        out_channels=2,
        feature_size=feature_size,
        use_checkpoint=True,
    )

    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.to(device)
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)

    case_results = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    pred_volumes = []
    gt_volumes = []

    print("\n" + "=" * 80)
    print("RUNNING INFERENCE ON DEVELOPMENT SET")
    print("=" * 80)

    start_eval_time = time.time()

    for idx, case_name in enumerate(dev_cases, start=1):
        case_dir = PREPROCESSED_ROOT / case_name
        t1c_path = case_dir / "t1c.npy"
        gtv_path = case_dir / "gtv.npy"

        if not t1c_path.exists() or not gtv_path.exists():
            print(f"  [WARN] Skipping case {case_name}: files missing")
            continue

        case_start = time.time()
        image = np.load(t1c_path).astype(np.float32)
        target = np.load(gtv_path).astype(np.uint8)

        # Sliding window inference
        prob_map, n_windows = sliding_window_predict(
            model=model,
            image=image,
            patch_size=(64, 64, 64),
            overlap=overlap,
            device=device,
        )

        pred_binary = (prob_map >= threshold).astype(np.uint8)
        metrics = calculate_case_metrics(pred_binary, target)
        metrics["case_name"] = case_name
        metrics["windows"] = n_windows
        metrics["time_sec"] = time.time() - case_start

        if save_predictions:
            preds_dir = output_dir.parent / "predictions"
            preds_dir.mkdir(parents=True, exist_ok=True)
            np.save(preds_dir / f"{case_name}_pred.npy", pred_binary)
            np.save(preds_dir / f"{case_name}_prob.npy", prob_map)

        case_results.append(metrics)

        total_tp += metrics["tp"]
        total_fp += metrics["fp"]
        total_fn += metrics["fn"]
        total_tn += metrics["tn"]

        pred_volumes.append(metrics["pred_voxels"])
        gt_volumes.append(metrics["gt_voxels"])

        print(
            f"  [{idx:03d}/{len(dev_cases):03d}] {case_name:30s} | "
            f"Dice: {metrics['dice']:.4f} | IoU: {metrics['iou']:.4f} | "
            f"Pred: {metrics['pred_voxels']:6d} | GT: {metrics['gt_voxels']:6d} | ({metrics['time_sec']:.1f}s)"
        )

    eval_duration = time.time() - start_eval_time

    # Aggregations
    dice_list = [c["dice"] for c in case_results]
    iou_list = [c["iou"] for c in case_results]
    prec_list = [c["precision"] for c in case_results]
    rec_list = [c["recall"] for c in case_results]

    micro_dice = (2.0 * total_tp) / (2.0 * total_tp + total_fp + total_fn + 1e-8) if (2.0 * total_tp + total_fp + total_fn) > 0 else 1.0

    # Volume correlation
    pv = np.array(pred_volumes, dtype=np.float64)
    gv = np.array(gt_volumes, dtype=np.float64)
    if len(pv) > 1 and np.std(pv) > 1e-6 and np.std(gv) > 1e-6:
        r = float(np.corrcoef(pv, gv)[0, 1])
        r_squared = float(r ** 2)
    else:
        r = 0.0
        r_squared = 0.0

    summary_metrics = {
        "model_architecture": "SwinUNETR",
        "feature_size": feature_size,
        "checkpoint": str(checkpoint_path),
        "threshold": threshold,
        "overlap": overlap,
        "num_cases_evaluated": len(case_results),
        "evaluation_duration_sec": eval_duration,
        "case_dice": {
            "mean": float(np.mean(dice_list)),
            "std": float(np.std(dice_list)),
            "median": float(np.median(dice_list)),
            "min": float(np.min(dice_list)),
            "max": float(np.max(dice_list)),
        },
        "case_iou": {
            "mean": float(np.mean(iou_list)),
            "std": float(np.std(iou_list)),
            "median": float(np.median(iou_list)),
        },
        "precision": {
            "mean": float(np.mean(prec_list)),
            "std": float(np.std(prec_list)),
        },
        "recall": {
            "mean": float(np.mean(rec_list)),
            "std": float(np.std(rec_list)),
        },
        "micro_global_dice": float(micro_dice),
        "volume_correlation": {
            "pearson_r": r,
            "r_squared": r_squared,
        },
        "cases": case_results,
    }

    metrics_file = output_dir / "development_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Cases Evaluated:      {len(case_results)}")
    print(f"Mean Case Dice:       {summary_metrics['case_dice']['mean']:.4f} +- {summary_metrics['case_dice']['std']:.4f}")
    print(f"Median Case Dice:     {summary_metrics['case_dice']['median']:.4f}")
    print(f"Mean Case IoU:        {summary_metrics['case_iou']['mean']:.4f}")
    print(f"Micro/Global Dice:    {summary_metrics['micro_global_dice']:.4f}")
    print(f"Volume Correlation r: {r:.4f} (R^2 = {r_squared:.4f})")
    print(f"Results saved to:     {metrics_file}")
    print("=" * 80)

    return summary_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Meningioma Swin UNETR Development Evaluation")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_METRICS_DIR), help="Metrics output directory")
    parser.add_argument("--threshold", type=float, default=0.50, help="Probability threshold (default: 0.50)")
    parser.add_argument("--overlap", type=float, default=0.50, help="Sliding window overlap (default: 0.50)")
    parser.add_argument("--feature-size", type=int, default=12, help="Swin UNETR feature size")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device (cuda or cpu)")
    parser.add_argument("--smoke-test", action="store_true", help="Quick test on 2 cases")
    parser.add_argument("--save-predictions", action="store_true", help="Save prediction and probability maps to predictions/")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_dev_set(
        checkpoint_path=Path(args.checkpoint),
        output_dir=Path(args.output_dir),
        threshold=args.threshold,
        overlap=args.overlap,
        feature_size=args.feature_size,
        device_str=args.device,
        smoke_test=args.smoke_test,
        save_predictions=args.save_predictions,
    )

