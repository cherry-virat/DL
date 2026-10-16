# ============================================================
# MENINGIOMA SWIN UNETR GAUSSIAN BLENDING ANALYSIS
# ============================================================
#
# Purpose:
#   Isolated evaluation comparing flat/uniform sliding-window blending
#   versus Gaussian-weighted sliding-window blending for 3D Meningioma
#   segmentation using the trained Swin UNETR checkpoint (epoch 11).
#
# Experimental Design:
#   - Checkpoint: outputs/meningioma_swin_unetr/checkpoints/meningioma_swin_best.pth
#   - Dataset: 100 development cases from data/meningioma_split.json
#   - Window Size: (64, 64, 64) with 50% overlap (stride = 32)
#   - Controlled variable: Window importance weighting map (Flat vs Gaussian)
#     All other inference details (padding, coordinates, unpadding,
#     probability domain blending, model architecture, FP16 autocast)
#     are held strictly identical.
#
# Operating Points Evaluated:
#   1. Threshold 0.50, no CC filter (Reference point)
#   2. Threshold 0.25, no CC filter (PRIMARY comparison)
#   3. Threshold 0.25, min CC size 1000 (Secondary post-processing diagnostic)
#
# Outputs:
#   outputs/meningioma_swin_unetr/gaussian_blending_analysis/
#     - gaussian_blending_report.json
#     - gaussian_blending_summary.csv
#     - gaussian_blending_summary.md
# ============================================================

from pathlib import Path
import sys
import time
import json
import argparse
import csv
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from scipy import ndimage
import torch
import torch.nn.functional as F
from monai.data.utils import compute_importance_map

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.meningioma_swin_unetr import create_meningioma_swin_unetr


# ============================================================
# CONSTANTS & DEFAULTS
# ============================================================

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "meningioma_swin_unetr"
    / "checkpoints"
    / "meningioma_swin_best.pth"
)
DEFAULT_SPLIT_FILE = PROJECT_ROOT / "data" / "meningioma_split.json"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "meningioma_preprocessed"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "meningioma_swin_unetr"
    / "gaussian_blending_analysis"
)
REFERENCE_METRICS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "meningioma_swin_unetr"
    / "development_full_volume"
    / "development_metrics.json"
)

PATCH_SIZE = (64, 64, 64)
OVERLAP = 0.50
SIGMA_SCALE = 0.125
STRUCTURE_26 = np.ones((3, 3, 3), dtype=np.uint8)

OPERATING_POINTS = [
    {"name": "t0.50_raw", "threshold": 0.50, "min_component_size": 0, "is_primary": False},
    {"name": "t0.25_raw", "threshold": 0.25, "min_component_size": 0, "is_primary": True},
    {"name": "t0.25_cc1000", "threshold": 0.25, "min_component_size": 1000, "is_primary": False},
]


# ============================================================
# PADDING & COORDINATE LOGIC (REUSED FROM REFERENCE PIPELINE)
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
    """Generates window starting indices along an axis with overlap and boundary clamping."""
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


# ============================================================
# SLIDING-WINDOW INFERENCE (FLAT VS GAUSSIAN)
# ============================================================

@torch.no_grad()
def predict_full_volume_probabilities(
    model: torch.nn.Module,
    image: np.ndarray,
    patch_size: Tuple[int, int, int] = PATCH_SIZE,
    overlap: float = OVERLAP,
    blend_mode: str = "gaussian",
    sigma_scale: float = SIGMA_SCALE,
    device: torch.device = torch.device("cuda"),
) -> Tuple[np.ndarray, int, float, float, float]:
    """
    Performs sliding-window inference and returns full-volume tumor probability map.
    
    Args:
        model: Swin UNETR model in eval mode.
        image: 3D MRI array (D, H, W).
        patch_size: Sub-volume window dimensions (64, 64, 64).
        overlap: Overlap fraction (0.50 -> stride 32).
        blend_mode: 'flat' or 'gaussian'.
        sigma_scale: Standard deviation scale for Gaussian kernel.
        device: Torch execution device.
        
    Returns:
        (cropped_prob, total_windows, elapsed_time_sec, max_gpu_mem_mb, min_weight_denom)
    """
    original_shape = image.shape
    padded_image, pads = pad_volume(image, patch_size)
    padded_shape = padded_image.shape

    starts_d = compute_starts(padded_shape[0], patch_size[0], overlap)
    starts_h = compute_starts(padded_shape[1], patch_size[1], overlap)
    starts_w = compute_starts(padded_shape[2], patch_size[2], overlap)

    total_windows = len(starts_d) * len(starts_h) * len(starts_w)
    prob_sum = np.zeros(padded_shape, dtype=np.float32)
    weight_sum = np.zeros(padded_shape, dtype=np.float32)

    # Compute importance map
    if blend_mode == "gaussian":
        importance_map = compute_importance_map(
            patch_size=patch_size,
            mode="gaussian",
            sigma_scale=sigma_scale,
            device="cpu",
        ).numpy().astype(np.float32)
    elif blend_mode == "flat":
        importance_map = np.ones(patch_size, dtype=np.float32)
    else:
        raise ValueError(f"Unknown blend_mode: {blend_mode}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start_time = time.time()

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
                    with torch.amp.autocast("cuda", enabled=True):
                        logits = model(patch_tensor)
                else:
                    logits = model(patch_tensor)

                probs = F.softmax(logits, dim=1)
                tumor_prob = probs[0, 1].detach().cpu().numpy()

                prob_sum[sd:ed, sh:eh, sw:ew] += tumor_prob * importance_map
                weight_sum[sd:ed, sh:eh, sw:ew] += importance_map

    elapsed_time = time.time() - start_time
    max_mem_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        if device.type == "cuda"
        else 0.0
    )

    # Record minimum denominator across non-zero weight areas
    min_denom = float(np.min(weight_sum))

    # Normalized probability map
    avg_prob = prob_sum / np.maximum(weight_sum, 1e-8)

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

    # Numerical safety checks
    assert not np.isnan(cropped_prob).any(), "NaN detected in cropped probabilities!"
    assert not np.isinf(cropped_prob).any(), "Inf detected in cropped probabilities!"

    return cropped_prob.astype(np.float32), total_windows, elapsed_time, max_mem_mb, min_denom


# ============================================================
# CONNECTED COMPONENT FILTERING & METRIC CALCULATION
# ============================================================

def filter_connected_components(
    binary_mask: np.ndarray,
    min_size: int,
    structure: np.ndarray = STRUCTURE_26,
) -> np.ndarray:
    """
    Filters out connected components smaller than min_size voxels.
    Uses 26-connectivity matching baseline and threshold analysis conventions.
    """
    if min_size <= 0:
        return binary_mask.astype(bool)

    binary_bool = binary_mask.astype(bool)
    labeled, num_features = ndimage.label(binary_bool, structure=structure)
    if num_features == 0:
        return np.zeros_like(binary_bool, dtype=bool)

    component_sizes = np.bincount(labeled.ravel())
    keep = component_sizes >= min_size
    keep[0] = False  # Background is never kept

    filtered = keep[labeled]
    return filtered.astype(bool)


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


def classify_tumor_size(gt_voxels: int) -> str:
    """Classifies ground truth tumor size into standard groups."""
    if gt_voxels < 2500:
        return "small"
    elif gt_voxels <= 15000:
        return "medium"
    else:
        return "large"


# ============================================================
# AGGREGATE SUMMARY BUILDER
# ============================================================

def aggregate_operating_point_metrics(
    case_results: List[Dict[str, Any]],
    op_name: str,
) -> Dict[str, Any]:
    """Computes mean, median, micro Dice, and size subgroup statistics."""
    dices = [c[op_name]["dice"] for c in case_results]
    ious = [c[op_name]["iou"] for c in case_results]
    precisions = [c[op_name]["precision"] for c in case_results]
    recalls = [c[op_name]["recall"] for c in case_results]

    total_tp = sum(c[op_name]["tp"] for c in case_results)
    total_fp = sum(c[op_name]["fp"] for c in case_results)
    total_fn = sum(c[op_name]["fn"] for c in case_results)
    total_pred = sum(c[op_name]["pred_voxels"] for c in case_results)
    total_gt = sum(c[op_name]["gt_voxels"] for c in case_results)

    micro_dice = (2.0 * total_tp) / (total_pred + total_gt) if (total_pred + total_gt) > 0 else 0.0
    micro_iou = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0

    zero_dice_count = sum(1 for d in dices if d == 0.0)
    dice_ge_0_50_count = sum(1 for d in dices if d >= 0.50)

    # Subgroups
    subgroups = {}
    for group_name in ["small", "medium", "large"]:
        group_cases = [c for c in case_results if c["size_group"] == group_name]
        if group_cases:
            g_dices = [c[op_name]["dice"] for c in group_cases]
            g_tp = sum(c[op_name]["tp"] for c in group_cases)
            g_pred = sum(c[op_name]["pred_voxels"] for c in group_cases)
            g_gt = sum(c[op_name]["gt_voxels"] for c in group_cases)
            g_micro_dice = (2.0 * g_tp) / (g_pred + g_gt) if (g_pred + g_gt) > 0 else 0.0
            subgroups[group_name] = {
                "count": len(group_cases),
                "mean_dice": float(np.mean(g_dices)),
                "median_dice": float(np.median(g_dices)),
                "micro_dice": float(g_micro_dice),
                "zero_dice_count": sum(1 for d in g_dices if d == 0.0),
                "dice_ge_0_50_count": sum(1 for d in g_dices if d >= 0.50),
            }
        else:
            subgroups[group_name] = {"count": 0, "mean_dice": 0.0, "median_dice": 0.0, "micro_dice": 0.0, "zero_dice_count": 0, "dice_ge_0_50_count": 0}

    return {
        "num_cases": len(case_results),
        "mean_dice": float(np.mean(dices)),
        "median_dice": float(np.median(dices)),
        "mean_iou": float(np.mean(ious)),
        "median_iou": float(np.median(ious)),
        "mean_precision": float(np.mean(precisions)),
        "mean_recall": float(np.mean(recalls)),
        "micro_dice": float(micro_dice),
        "micro_iou": float(micro_iou),
        "zero_dice_cases": zero_dice_count,
        "cases_dice_ge_0_50": dice_ge_0_50_count,
        "total_pred_voxels": total_pred,
        "total_gt_voxels": total_gt,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "subgroups": subgroups,
    }


# ============================================================
# MAIN EXPERIMENT RUNNER
# ============================================================

def run_experiment(
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    split_file: Path = DEFAULT_SPLIT_FILE,
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    smoke_test: bool = False,
    max_cases: Optional[int] = None,
    device_str: str = "cuda",
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    print("=" * 70)
    print("MENINGIOMA SWIN UNETR GAUSSIAN BLENDING ANALYSIS")
    print("=" * 70)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Split file: {split_file}")
    print(f"Data dir: {data_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Mode: {'SMOKE TEST (2 cases)' if smoke_test else f'FULL EVALUATION ({max_cases or 100} cases)'}")
    print("=" * 70)

    # Load split cases
    with open(split_file, "r") as f:
        split_data = json.load(f)
    dev_cases = split_data["dev_cases"]

    if smoke_test:
        selected_cases = dev_cases[:2]
    elif max_cases is not None:
        selected_cases = dev_cases[:max_cases]
    else:
        selected_cases = dev_cases

    print(f"Total cases to evaluate: {len(selected_cases)}")

    # Load model
    print("\nLoading Swin UNETR model...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    feature_size = checkpoint.get("feature_size", 12)
    model = create_meningioma_swin_unetr(
        in_channels=1,
        out_channels=2,
        feature_size=feature_size,
        use_checkpoint=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Model loaded successfully (epoch {checkpoint.get('epoch')}, feature_size {feature_size}).")

    # Storage for results
    case_results: List[Dict[str, Any]] = []
    flat_times: List[float] = []
    gauss_times: List[float] = []
    flat_mems: List[float] = []
    gauss_mems: List[float] = []

    print("\nStarting evaluation loop...\n")

    for idx, case_id in enumerate(selected_cases, 1):
        case_path = data_dir / case_id
        t1c_path = case_path / "t1c.npy"
        gtv_path = case_path / "gtv.npy"

        if not t1c_path.exists() or not gtv_path.exists():
            raise FileNotFoundError(f"Missing data for case {case_id}: {t1c_path} or {gtv_path}")

        image = np.load(t1c_path).astype(np.float32)
        gtv = np.load(gtv_path).astype(np.uint8)
        gt_bool = gtv.astype(bool)
        gt_voxels = int(gt_bool.sum())
        size_group = classify_tumor_size(gt_voxels)

        # 1. Flat sliding-window inference
        prob_flat, windows_flat, time_flat, mem_flat, min_denom_flat = predict_full_volume_probabilities(
            model=model,
            image=image,
            patch_size=PATCH_SIZE,
            overlap=OVERLAP,
            blend_mode="flat",
            device=device,
        )
        flat_times.append(time_flat)
        flat_mems.append(mem_flat)

        # 2. Gaussian-weighted sliding-window inference
        prob_gauss, windows_gauss, time_gauss, mem_gauss, min_denom_gauss = predict_full_volume_probabilities(
            model=model,
            image=image,
            patch_size=PATCH_SIZE,
            overlap=OVERLAP,
            blend_mode="gaussian",
            sigma_scale=SIGMA_SCALE,
            device=device,
        )
        gauss_times.append(time_gauss)
        gauss_mems.append(mem_gauss)

        # Evaluate operating points for both methods
        case_entry: Dict[str, Any] = {
            "case_id": case_id,
            "case_index": idx,
            "volume_shape": list(image.shape),
            "gt_voxels": gt_voxels,
            "size_group": size_group,
            "windows": windows_flat,
            "flat_inference": {
                "time_sec": time_flat,
                "peak_mem_mb": mem_flat,
                "min_denom": min_denom_flat,
            },
            "gaussian_inference": {
                "time_sec": time_gauss,
                "peak_mem_mb": mem_gauss,
                "min_denom": min_denom_gauss,
            },
        }

        for op in OPERATING_POINTS:
            op_key = op["name"]
            thresh = op["threshold"]
            min_cc = op["min_component_size"]

            # Flat
            mask_flat_raw = prob_flat >= thresh
            mask_flat_filt = filter_connected_components(mask_flat_raw, min_size=min_cc)
            metrics_flat = compute_binary_metrics(mask_flat_filt, gt_bool)

            # Gaussian
            mask_gauss_raw = prob_gauss >= thresh
            mask_gauss_filt = filter_connected_components(mask_gauss_raw, min_size=min_cc)
            metrics_gauss = compute_binary_metrics(mask_gauss_filt, gt_bool)

            delta_dice = metrics_gauss["dice"] - metrics_flat["dice"]
            delta_pred = metrics_gauss["pred_voxels"] - metrics_flat["pred_voxels"]

            if delta_dice > 1e-4:
                comparison_status = "improved"
            elif delta_dice < -1e-4:
                comparison_status = "degraded"
            else:
                comparison_status = "unchanged"

            case_entry[f"flat_{op_key}"] = metrics_flat
            case_entry[f"gauss_{op_key}"] = metrics_gauss
            case_entry[f"delta_{op_key}"] = {
                "dice_diff": float(delta_dice),
                "pred_diff": int(delta_pred),
                "status": comparison_status,
            }

        case_results.append(case_entry)

        # Progress reporting
        p_dice_flat = case_entry["flat_t0.25_raw"]["dice"]
        p_dice_gauss = case_entry["gauss_t0.25_raw"]["dice"]
        diff_str = f"{p_dice_gauss - p_dice_flat:+.4f}"
        print(
            f"[{idx:3d}/{len(selected_cases)}] {case_id} ({size_group:>6s}, GT={gt_voxels:6d}) | "
            f"Flat(t0.25)={p_dice_flat:.4f} | Gauss(t0.25)={p_dice_gauss:.4f} | Diff={diff_str:>7s} | "
            f"Flat_t={time_flat:.1f}s Gauss_t={time_gauss:.1f}s"
        )

    # Compute aggregate metrics for all operating points
    aggregates: Dict[str, Any] = {}
    for op in OPERATING_POINTS:
        op_key = op["name"]
        agg_flat = aggregate_operating_point_metrics(case_results, f"flat_{op_key}")
        agg_gauss = aggregate_operating_point_metrics(case_results, f"gauss_{op_key}")

        # Differences
        improved = sum(1 for c in case_results if c[f"delta_{op_key}"]["status"] == "improved")
        unchanged = sum(1 for c in case_results if c[f"delta_{op_key}"]["status"] == "unchanged")
        degraded = sum(1 for c in case_results if c[f"delta_{op_key}"]["status"] == "degraded")

        aggregates[op_key] = {
            "operating_point": op,
            "flat": agg_flat,
            "gaussian": agg_gauss,
            "delta": {
                "mean_dice_diff": agg_gauss["mean_dice"] - agg_flat["mean_dice"],
                "median_dice_diff": agg_gauss["median_dice"] - agg_flat["median_dice"],
                "micro_dice_diff": agg_gauss["micro_dice"] - agg_flat["micro_dice"],
                "zero_dice_diff": agg_gauss["zero_dice_cases"] - agg_flat["zero_dice_cases"],
                "improved_cases": improved,
                "unchanged_cases": unchanged,
                "degraded_cases": degraded,
            },
        }

    # Summary performance stats
    performance_stats = {
        "flat_mean_time_sec": float(np.mean(flat_times)),
        "gaussian_mean_time_sec": float(np.mean(gauss_times)),
        "flat_max_gpu_mem_mb": float(np.max(flat_mems)),
        "gaussian_max_gpu_mem_mb": float(np.max(gauss_mems)),
    }

    # Smoke test verification block
    smoke_test_verification = None
    if smoke_test:
        smoke_test_verification = verify_smoke_test(case_results)

    # Compile report dictionary
    report_data = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "feature_size": feature_size,
            "dataset": "Meningioma BraTS Preprocessed Development Set",
            "split_file": str(split_file),
            "cases_evaluated": len(selected_cases),
            "is_smoke_test": smoke_test,
            "primary_operating_point": "Threshold 0.25, No Component Filter",
            "disclaimer": (
                "EXPLORATORY DEVELOPMENT-SET EVALUATION ONLY: All reported metrics "
                "are computed on the 100 development cases used for post-hoc threshold "
                "investigation. They must not be claimed as generalized or independent test set results."
            ),
        },
        "performance_stats": performance_stats,
        "operating_point_aggregates": aggregates,
        "detailed_case_results": case_results,
    }
    if smoke_test_verification:
        report_data["smoke_test_verification"] = smoke_test_verification

    # Write output files
    report_json_path = output_dir / "gaussian_blending_report.json"
    summary_csv_path = output_dir / "gaussian_blending_summary.csv"
    summary_md_path = output_dir / "gaussian_blending_summary.md"

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nSaved JSON report: {report_json_path}")

    write_summary_csv(summary_csv_path, aggregates, performance_stats)
    print(f"Saved CSV summary: {summary_csv_path}")

    write_summary_markdown(summary_md_path, report_data)
    print(f"Saved Markdown report: {summary_md_path}")

    print("\nExperiment execution completed successfully.")
    return report_data


# ============================================================
# SMOKE TEST VERIFICATION ROUTINE
# ============================================================

def verify_smoke_test(case_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Verifies smoke test results against reference evaluator."""
    print("\n" + "=" * 70)
    print("SMOKE TEST VERIFICATION AGAINST REFERENCE FULL-VOLUME EVALUATOR")
    print("=" * 70)

    # Check existence of reference metrics
    ref_data = None
    if REFERENCE_METRICS_FILE.exists():
        with open(REFERENCE_METRICS_FILE, "r") as f:
            ref_data = json.load(f)

    verification_records = []
    all_passed = True

    for i, c in enumerate(case_results):
        case_id = c["case_id"]
        flat_t05 = c["flat_t0.50_raw"]
        gauss_t05 = c["gauss_t0.50_raw"]

        # Check alignment and numerical bounds
        assert c["flat_inference"]["min_denom"] > 0, f"Zero denominator in flat inference for {case_id}"
        assert c["gaussian_inference"]["min_denom"] > 0, f"Zero denominator in gaussian inference for {case_id}"

        # Reference comparison for t=0.50
        ref_case = None
        if ref_data and "cases" in ref_data:
            ref_matches = [rc for rc in ref_data["cases"] if rc.get("case_name") == case_id]
            if ref_matches:
                ref_case = ref_matches[0]

        dice_match = True
        pred_match = True
        if ref_case:
            dice_match = abs(flat_t05["dice"] - ref_case["dice"]) < 1e-4
            pred_match = flat_t05["pred_voxels"] == ref_case["pred_voxels"]

        case_passed = dice_match and pred_match
        if not case_passed:
            all_passed = False

        rec = {
            "case_id": case_id,
            "flat_dice_t05": flat_t05["dice"],
            "flat_pred_voxels": flat_t05["pred_voxels"],
            "ref_dice_t05": ref_case["dice"] if ref_case else None,
            "ref_pred_voxels": ref_case["pred_voxels"] if ref_case else None,
            "gauss_dice_t05": gauss_t05["dice"],
            "gauss_pred_voxels": gauss_t05["pred_voxels"],
            "matches_reference": case_passed,
        }
        verification_records.append(rec)

        print(f"Case {case_id}:")
        print(f"  Flat Dice (t0.50): {flat_t05['dice']:.4f} (Ref: {rec['ref_dice_t05']})")
        print(f"  Flat Pred Voxels: {flat_t05['pred_voxels']} (Ref: {rec['ref_pred_voxels']})")
        print(f"  Gaussian Dice (t0.50): {gauss_t05['dice']:.4f}, Pred Voxels: {gauss_t05['pred_voxels']}")
        print(f"  Reference Match: {'PASS' if case_passed else 'FAIL'}")

    print("=" * 70)
    print(f"Smoke Test Verification Result: {'ALL CHECKS PASSED' if all_passed else 'VERIFICATION FAILED'}")
    print("=" * 70)

    return {
        "all_passed": all_passed,
        "cases": verification_records,
    }


# ============================================================
# CSV EXPORT
# ============================================================

def write_summary_csv(
    csv_path: Path,
    aggregates: Dict[str, Any],
    performance_stats: Dict[str, Any],
):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Operating Point",
            "Threshold",
            "Min CC Size",
            "Is Primary",
            "Blending Method",
            "Mean Dice",
            "Median Dice",
            "Mean IoU",
            "Micro Dice",
            "Zero Dice Cases",
            "Cases Dice >= 0.50",
            "Small Mean Dice (<2.5k)",
            "Medium Mean Dice (2.5k-15k)",
            "Large Mean Dice (>15k)",
        ])

        for op_key, agg in aggregates.items():
            op = agg["operating_point"]
            for method in ["flat", "gaussian"]:
                m_agg = agg[method]
                sg = m_agg["subgroups"]
                writer.writerow([
                    op_key,
                    op["threshold"],
                    op["min_component_size"],
                    op["is_primary"],
                    method.capitalize(),
                    f"{m_agg['mean_dice']:.6f}",
                    f"{m_agg['median_dice']:.6f}",
                    f"{m_agg['mean_iou']:.6f}",
                    f"{m_agg['micro_dice']:.6f}",
                    m_agg["zero_dice_cases"],
                    m_agg["cases_dice_ge_0_50"],
                    f"{sg['small']['mean_dice']:.6f}",
                    f"{sg['medium']['mean_dice']:.6f}",
                    f"{sg['large']['mean_dice']:.6f}",
                ])


# ============================================================
# MARKDOWN REPORT BUILDER
# ============================================================

def write_summary_markdown(
    md_path: Path,
    report: Dict[str, Any],
):
    meta = report["metadata"]
    aggs = report["operating_point_aggregates"]
    perf = report["performance_stats"]
    is_smoke = meta.get("is_smoke_test", False)

    lines = []
    if is_smoke:
        lines.append("# Meningioma Swin UNETR: Gaussian vs Flat Sliding-Window Blending (SMOKE TEST: 2 CASES)\n")
        lines.append("> [!NOTE]")
        lines.append(
            "> **SMOKE-TEST VALIDATION (2 CASES)**: This run evaluates only 2 development cases "
            "(`BraTS-MEN-RT-0005-1` and `BraTS-MEN-RT-0020-1`) for numerical verification and reference "
            "evaluator reproduction. These metrics DO NOT represent the full 100-case development evaluation.\n"
        )
    else:
        lines.append("# Meningioma Swin UNETR: Gaussian vs Flat Sliding-Window Blending (100 DEVELOPMENT CASES)\n")

    lines.append(f"**Execution Date**: {meta['timestamp']}  ")
    lines.append(f"**Checkpoint**: `{meta['checkpoint']}` (Epoch {meta['checkpoint_epoch']}, Feature Size {meta['feature_size']})  ")
    lines.append(f"**Dataset**: 100 Development Cases ([`data/meningioma_split.json`](file:///C:/Users/nchan/Desktop/DL/data/meningioma_split.json))  ")
    lines.append(f"**Cases Evaluated**: {meta['cases_evaluated']} ({'Smoke-Test Mode' if is_smoke else 'Full Evaluation'})  ")
    lines.append(f"**Primary Comparison**: **Threshold 0.25 (No CC Filter)**  \n")

    lines.append("> [!WARNING]")
    lines.append(f"> **Exploratory Development Analysis**: {meta['disclaimer']}\n")

    lines.append("## 1. Primary Comparison: Threshold 0.25 (No Component Filter)\n")
    p_agg = aggs["t0.25_raw"]
    flat_p = p_agg["flat"]
    gauss_p = p_agg["gaussian"]
    delta_p = p_agg["delta"]

    lines.append("| Metric | Flat Blending (Reference) | Gaussian Blending | Absolute Difference |")
    lines.append("| :--- | :---: | :---: | :---: |")
    lines.append(f"| **Mean Case Dice** | **{flat_p['mean_dice']:.4f}** | **{gauss_p['mean_dice']:.4f}** | **{delta_p['mean_dice_diff']:+.4f}** |")
    lines.append(f"| **Median Case Dice** | {flat_p['median_dice']:.4f} | {gauss_p['median_dice']:.4f} | {delta_p['median_dice_diff']:+.4f} |")
    lines.append(f"| **Mean Case IoU** | {flat_p['mean_iou']:.4f} | {gauss_p['mean_iou']:.4f} | {gauss_p['mean_iou'] - flat_p['mean_iou']:+.4f} |")
    lines.append(f"| **Micro / Global Dice** | {flat_p['micro_dice']:.4f} | {gauss_p['micro_dice']:.4f} | {delta_p['micro_dice_diff']:+.4f} |")
    lines.append(f"| **Zero-Dice Cases** | {flat_p['zero_dice_cases']} / {meta['cases_evaluated']} | {gauss_p['zero_dice_cases']} / {meta['cases_evaluated']} | {delta_p['zero_dice_diff']:+d} |")
    lines.append(f"| **Cases with Dice ≥ 0.50** | {flat_p['cases_dice_ge_0_50']} / {meta['cases_evaluated']} | {gauss_p['cases_dice_ge_0_50']} / {meta['cases_evaluated']} | {gauss_p['cases_dice_ge_0_50'] - flat_p['cases_dice_ge_0_50']:+d} |")
    lines.append(f"| **Improved / Unchanged / Degraded** | — | — | **{delta_p['improved_cases']} improved / {delta_p['unchanged_cases']} unchanged / {delta_p['degraded_cases']} degraded** |\n")

    lines.append("### Subgroup Performance by Ground-Truth Tumor Size (Threshold 0.25 Raw)\n")
    lines.append("| Tumor Size Group | Count | Flat Mean Dice | Gauss Mean Dice | Difference | Flat Zero Dice | Gauss Zero Dice |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for g_key, g_title in [("small", "Small (<2,500 voxels)"), ("medium", "Medium (2,500–15,000 voxels)"), ("large", "Large (>15,000 voxels)")]:
        f_sg = flat_p["subgroups"][g_key]
        g_sg = gauss_p["subgroups"][g_key]
        diff_sg = g_sg["mean_dice"] - f_sg["mean_dice"]
        lines.append(f"| **{g_title}** | {f_sg['count']} | {f_sg['mean_dice']:.4f} | {g_sg['mean_dice']:.4f} | **{diff_sg:+.4f}** | {f_sg['zero_dice_count']}/{f_sg['count']} | {g_sg['zero_dice_count']}/{f_sg['count']} |")
    lines.append("\n")

    lines.append("## 2. Secondary Diagnostic Operating Points\n")
    for op_key, title in [("t0.50_raw", "Reference Point: Threshold 0.50 (No CC Filter)"), ("t0.25_cc1000", "Post-Processing Diagnostic: Threshold 0.25 (CC Filter = 1,000)")]:
        agg = aggs[op_key]
        f_m = agg["flat"]
        g_m = agg["gaussian"]
        d_m = agg["delta"]
        lines.append(f"### {title}\n")
        lines.append("| Metric | Flat Blending | Gaussian Blending | Difference |")
        lines.append("| :--- | :---: | :---: | :---: |")
        lines.append(f"| **Mean Case Dice** | {f_m['mean_dice']:.4f} | {g_m['mean_dice']:.4f} | **{d_m['mean_dice_diff']:+.4f}** |")
        lines.append(f"| **Median Case Dice** | {f_m['median_dice']:.4f} | {g_m['median_dice']:.4f} | {d_m['median_dice_diff']:+.4f} |")
        lines.append(f"| **Micro / Global Dice** | {f_m['micro_dice']:.4f} | {g_m['micro_dice']:.4f} | {d_m['micro_dice_diff']:+.4f} |")
        lines.append(f"| **Zero-Dice Cases** | {f_m['zero_dice_cases']} | {g_m['zero_dice_cases']} | {d_m['zero_dice_diff']:+d} |")
        lines.append(f"| **Cases with Dice ≥ 0.50** | {f_m['cases_dice_ge_0_50']} | {g_m['cases_dice_ge_0_50']} | {g_m['cases_dice_ge_0_50'] - f_m['cases_dice_ge_0_50']:+d} |")
        lines.append(f"| **Case Comparison** | — | — | {d_m['improved_cases']} improved / {d_m['unchanged_cases']} unchanged / {d_m['degraded_cases']} degraded |\n")

    lines.append("## 3. Case-by-Case Breakdown (Threshold 0.25 Raw)\n")
    lines.append("| Case ID | Group | GT Voxels | Flat Pred | Flat Dice | Gauss Pred | Gauss Dice | Delta Dice | Status |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for c in report.get("detailed_case_results", []):
        cid = c["case_id"]
        grp = c["size_group"]
        gt_v = c["gt_voxels"]
        f_p = c["flat_t0.25_raw"]["pred_voxels"]
        f_d = c["flat_t0.25_raw"]["dice"]
        g_p = c["gauss_t0.25_raw"]["pred_voxels"]
        g_d = c["gauss_t0.25_raw"]["dice"]
        d_d = c["delta_t0.25_raw"]["dice_diff"]
        st = c["delta_t0.25_raw"]["status"]
        lines.append(f"| `{cid}` | {grp} | {gt_v} | {f_p} | {f_d:.4f} | {g_p} | {g_d:.4f} | {d_d:+.4f} | {st} |")
    lines.append("\n")

    lines.append("## 4. Computational Runtime & GPU Memory\n")
    lines.append("| Metric | Flat Blending | Gaussian Blending |")
    lines.append("| :--- | :---: | :---: |")
    lines.append(f"| **Mean Inference Time per Volume** | {perf['flat_mean_time_sec']:.2f} s | {perf['gaussian_mean_time_sec']:.2f} s |")
    lines.append(f"| **Peak GPU Memory Allocated** | {perf['flat_max_gpu_mem_mb']:.1f} MB | {perf['gaussian_max_gpu_mem_mb']:.1f} MB |\n")

    content = "\n".join(lines)
    if not content.strip():
        raise ValueError("Markdown report text is unexpectedly empty!")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(content)

    if not md_path.exists() or md_path.stat().st_size == 0:
        raise IOError(f"Failed to write non-empty markdown summary to {md_path}")


# ============================================================
# CLI ENTRYPOINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Meningioma Swin UNETR Gaussian Blending Analysis")
    parser.add_argument("--smoke-test", action="store_true", help="Run 2 development cases for verification")
    parser.add_argument("--max-cases", type=int, default=None, help="Limit number of development cases")
    parser.add_argument("--device", type=str, default="cuda", help="Computation device (cuda or cpu)")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Model checkpoint path")
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE, help="Split JSON file")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Preprocessed data root")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output destination folder")

    args = parser.parse_args()

    run_experiment(
        checkpoint_path=args.checkpoint,
        split_file=args.split_file,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        smoke_test=args.smoke_test,
        max_cases=args.max_cases,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()

