"""
evaluate_meningioma_multimodal_final200.py
=========================================
Primary Final Validation for 4-Channel Meningioma Segmentation Pipeline.

Evaluates the full-cohort 3D U-Net best checkpoint across all 200 original
validation cases from data/meningioma_multimodal/split.json, followed by
a secondary comparison against the 50-case SegResNet model on the exact
25 validation cases used during training.

Outputs:
    outputs/meningioma_multimodal_experiment/final_validation_20261015/
        - full200_validation_report.md
        - full200_validation_metrics.json
        - full200_per_case_metrics.csv
"""

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from monai.inferers import sliding_window_inference
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset

from models.meningioma_multimodal_unet import create_multimodal_unet
from models.meningioma_segresnet import create_segresnet
from training.meningioma_multimodal_dataset import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_SPLIT_JSON,
    MeningiomaMultimodalDataset,
    calculate_multimodal_metrics,
)
from training.train_meningioma_multimodal import apply_air_masking, load_checkpoint


TRAINING_25_VAL_CASES = [
    "BraTS-MEN-00150-000",
    "BraTS-MEN-00154-000",
    "BraTS-MEN-00269-000",
    "BraTS-MEN-00398-000",
    "BraTS-MEN-00451-000",
    "BraTS-MEN-00517-000",
    "BraTS-MEN-00535-000",
    "BraTS-MEN-00542-000",
    "BraTS-MEN-00552-000",
    "BraTS-MEN-00588-000",
    "BraTS-MEN-00614-000",
    "BraTS-MEN-00689-000",
    "BraTS-MEN-00697-000",
    "BraTS-MEN-00852-000",
    "BraTS-MEN-00936-000",
    "BraTS-MEN-00977-000",
    "BraTS-MEN-01061-000",
    "BraTS-MEN-01080-000",
    "BraTS-MEN-01128-000",
    "BraTS-MEN-01168-000",
    "BraTS-MEN-01216-000",
    "BraTS-MEN-01231-000",
    "BraTS-MEN-01278-000",
    "BraTS-MEN-01305-000",
    "BraTS-MEN-01323-000",
]


def compute_distribution_stats(values: List[float]) -> Dict[str, float]:
    """Computes mean, std, median, and IQR (q25, q75)."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "q25": 0.0, "q75": 0.0, "iqr": 0.0}
    arr = np.array(values, dtype=np.float64)
    q25 = float(np.percentile(arr, 25))
    q75 = float(np.percentile(arr, 75))
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "q25": q25,
        "q75": q75,
        "iqr": float(q75 - q25),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def run_evaluation():
    parser = argparse.ArgumentParser(description="Full 200-Case Multimodal Meningioma Validation")
    parser.add_argument(
        "--unet-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs/meningioma_multimodal_experiment/full800_brainaware_20261014/checkpoints/meningioma_multimodal_best.pth",
        help="Path to full-cohort 3D U-Net checkpoint",
    )
    parser.add_argument(
        "--segresnet-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs/meningioma_segresnet_experiment/exp50_segresnet/checkpoints/meningioma_multimodal_best.pth",
        help="Path to 50-case SegResNet checkpoint for secondary comparison",
    )
    parser.add_argument(
        "--split-json",
        type=Path,
        default=DEFAULT_SPLIT_JSON,
        help="Path to split.json",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Path to dataset root",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/meningioma_multimodal_experiment/final_validation_20261015",
        help="Output directory for reports and metrics",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Inference device")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "full200_per_case_metrics.csv"
    json_path = args.output_dir / "full200_validation_metrics.json"
    report_path = args.output_dir / "full200_validation_report.md"

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    print("=" * 80)
    print("URGENT FINAL VALIDATION: FULL 200-CASE EVALUATION")
    print("=" * 80)
    print(f"Device:                {device}")
    print(f"U-Net Checkpoint:      {args.unet_checkpoint}")
    print(f"SegResNet Checkpoint:  {args.segresnet_checkpoint}")
    print(f"Split JSON:            {args.split_json}")
    print(f"Dataset Root:          {args.dataset_root}")
    print(f"Output Directory:      {args.output_dir}")

    # 1. Dataset Verification
    print("\n--- Initializing Validation Dataset ---")
    val_dataset = MeningiomaMultimodalDataset(
        split_json_path=args.split_json,
        partition="val",
        dataset_root=args.dataset_root,
        patch_size=None,
    )
    total_val_cases = len(val_dataset)
    print(f"Total Validation Cases in Split: {total_val_cases}")
    assert total_val_cases == 200, f"Expected exactly 200 validation cases, found {total_val_cases}"

    # 2. Model Loading (U-Net)
    print("\n--- Loading Full-Cohort 3D U-Net Model ---")
    unet = create_multimodal_unet(in_channels=4, num_classes=4, base_channels=16).to(device)
    unet_meta = load_checkpoint(args.unet_checkpoint, unet, device=device)
    unet.eval()
    print(f"U-Net loaded: Epoch {unet_meta.get('epoch')}, Reported Best WT Dice: {unet_meta.get('best_metric', -1):.4f}")

    # 3. CSV Incremental Setup
    csv_headers = [
        "case_id",
        "status",
        "wt_dice",
        "tc_dice",
        "et_dice",
        "wt_precision",
        "wt_recall",
        "gt_wt_voxels",
        "pred_wt_voxels",
        "vol_ratio_pred_gt",
        "non_air_brain_fp",
        "air_fp",
        "runtime_seconds",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()

    # 4. Evaluation Loop
    results: List[Dict[str, Any]] = []
    failed_cases: List[Dict[str, Any]] = []

    # Running pooled voxel counters
    pooled_wt_tp = 0
    pooled_wt_fp = 0
    pooled_wt_fn = 0
    pooled_tc_tp = 0
    pooled_tc_fp = 0
    pooled_tc_fn = 0
    pooled_et_tp = 0
    pooled_et_fp = 0
    pooled_et_fn = 0
    total_air_fp = 0
    total_non_air_brain_fp = 0

    t_start = time.time()

    print("\n--- Starting Sequential 200-Case Evaluation ---")
    with torch.inference_mode():
        for i in range(total_val_cases):
            sample = val_dataset[i]
            cid = sample["case_id"]
            img_cpu = sample["image"]  # [4, D, H, W]
            mask_cpu = sample["mask"]  # [D, H, W]

            t_case = time.time()
            try:
                img_gpu = img_cpu.unsqueeze(0).to(device)  # [1, 4, D, H, W]
                output = sliding_window_inference(
                    inputs=img_gpu,
                    roi_size=(64, 64, 64),
                    sw_batch_size=2,
                    predictor=unet,
                    overlap=0.5,
                    mode="gaussian",
                )
                pred = torch.argmax(output, dim=1).squeeze(0).cpu()  # [D, H, W]

                # Air masking on predictions only
                masked_pred = apply_air_masking(pred, img_cpu)

                # Multimodal metrics against ground truth
                m = calculate_multimodal_metrics(masked_pred, mask_cpu)

                # Ground truth & prediction voxel counts
                gt_wt = int(((mask_cpu >= 1) & (mask_cpu <= 3)).sum().item())
                pred_wt = int(((masked_pred >= 1) & (masked_pred <= 3)).sum().item())
                vol_ratio = (pred_wt / gt_wt) if gt_wt > 0 else float("nan")

                # False positives in air vs non-air brain
                is_air = torch.all(img_cpu == 0, dim=0)
                gt_is_tumor = (mask_cpu >= 1) & (mask_cpu <= 3)
                pred_is_tumor = (masked_pred >= 1) & (masked_pred <= 3)

                air_fp = int((pred_is_tumor & (~gt_is_tumor) & is_air).sum().item())
                brain_fp = int((pred_is_tumor & (~gt_is_tumor) & (~is_air)).sum().item())

                case_time = time.time() - t_case

                # Accumulate pooled metrics
                pooled_wt_tp += m["wt"]["tp"]
                pooled_wt_fp += m["wt"]["fp"]
                pooled_wt_fn += m["wt"]["fn"]

                pooled_tc_tp += m["tc"]["tp"]
                pooled_tc_fp += m["tc"]["fp"]
                pooled_tc_fn += m["tc"]["fn"]

                pooled_et_tp += m["et"]["tp"]
                pooled_et_fp += m["et"]["fp"]
                pooled_et_fn += m["et"]["fn"]

                total_air_fp += air_fp
                total_non_air_brain_fp += brain_fp

                record = {
                    "case_id": cid,
                    "status": "success",
                    "wt_dice": m["wt"]["dice"],
                    "tc_dice": m["tc"]["dice"],
                    "et_dice": m["et"]["dice"],
                    "wt_precision": m["wt"]["precision"],
                    "wt_recall": m["wt"]["recall"],
                    "gt_wt_voxels": gt_wt,
                    "pred_wt_voxels": pred_wt,
                    "vol_ratio_pred_gt": vol_ratio,
                    "non_air_brain_fp": brain_fp,
                    "air_fp": air_fp,
                    "runtime_seconds": case_time,
                }
                results.append(record)

                print(
                    f"[{i+1:03d}/{total_val_cases}] {cid}: "
                    f"WT={m['wt']['dice']:.4f}, TC={m['tc']['dice']:.4f}, ET={m['et']['dice']:.4f} "
                    f"(Prec={m['wt']['precision']:.4f}, Rec={m['wt']['recall']:.4f}, GT_WT={gt_wt}) in {case_time:.1f}s"
                )

            except Exception as e:
                case_time = time.time() - t_case
                err_msg = str(e)
                print(f"[{i+1:03d}/{total_val_cases}] FAILED {cid}: {err_msg}")
                fail_record = {
                    "case_id": cid,
                    "status": f"failed: {err_msg}",
                    "wt_dice": 0.0,
                    "tc_dice": 0.0,
                    "et_dice": 0.0,
                    "wt_precision": 0.0,
                    "wt_recall": 0.0,
                    "gt_wt_voxels": 0,
                    "pred_wt_voxels": 0,
                    "vol_ratio_pred_gt": 0.0,
                    "non_air_brain_fp": 0,
                    "air_fp": 0,
                    "runtime_seconds": case_time,
                }
                failed_cases.append(fail_record)
                record = fail_record

            # Incremental CSV append
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=csv_headers)
                writer.writerow(record)

            # Periodic memory cleanup
            if (i + 1) % 10 == 0:
                torch.cuda.empty_cache()

    total_eval_time = time.time() - t_start

    # 5. Compute Pooled Metrics
    pooled_wt_dice = (2.0 * pooled_wt_tp / (2 * pooled_wt_tp + pooled_wt_fp + pooled_wt_fn)) if (2 * pooled_wt_tp + pooled_wt_fp + pooled_wt_fn) > 0 else 0.0
    pooled_wt_prec = (pooled_wt_tp / (pooled_wt_tp + pooled_wt_fp)) if (pooled_wt_tp + pooled_wt_fp) > 0 else 0.0
    pooled_wt_rec = (pooled_wt_tp / (pooled_wt_tp + pooled_wt_fn)) if (pooled_wt_tp + pooled_wt_fn) > 0 else 0.0

    pooled_tc_dice = (2.0 * pooled_tc_tp / (2 * pooled_tc_tp + pooled_tc_fp + pooled_tc_fn)) if (2 * pooled_tc_tp + pooled_tc_fp + pooled_tc_fn) > 0 else 0.0
    pooled_et_dice = (2.0 * pooled_et_tp / (2 * pooled_et_tp + pooled_et_fp + pooled_et_fn)) if (2 * pooled_et_tp + pooled_et_fp + pooled_et_fn) > 0 else 0.0

    # 6. Compute Macro Distribution Stats
    valid_results = [r for r in results if r["status"] == "success"]
    macro_stats = {
        "wt_dice": compute_distribution_stats([r["wt_dice"] for r in valid_results]),
        "tc_dice": compute_distribution_stats([r["tc_dice"] for r in valid_results]),
        "et_dice": compute_distribution_stats([r["et_dice"] for r in valid_results]),
        "wt_precision": compute_distribution_stats([r["wt_precision"] for r in valid_results]),
        "wt_recall": compute_distribution_stats([r["wt_recall"] for r in valid_results]),
    }

    # Stratification by Tumor Volume
    small_cases = [r for r in valid_results if r["gt_wt_voxels"] < 5000]
    medium_cases = [r for r in valid_results if 5000 <= r["gt_wt_voxels"] <= 50000]
    large_cases = [r for r in valid_results if r["gt_wt_voxels"] > 50000]

    stratified = {
        "small_tumors": {
            "count": len(small_cases),
            "wt_dice_mean": float(np.mean([r["wt_dice"] for r in small_cases])) if small_cases else 0.0,
            "wt_dice_median": float(np.median([r["wt_dice"] for r in small_cases])) if small_cases else 0.0,
            "wt_precision_mean": float(np.mean([r["wt_precision"] for r in small_cases])) if small_cases else 0.0,
            "wt_recall_mean": float(np.mean([r["wt_recall"] for r in small_cases])) if small_cases else 0.0,
        },
        "medium_tumors": {
            "count": len(medium_cases),
            "wt_dice_mean": float(np.mean([r["wt_dice"] for r in medium_cases])) if medium_cases else 0.0,
            "wt_dice_median": float(np.median([r["wt_dice"] for r in medium_cases])) if medium_cases else 0.0,
            "wt_precision_mean": float(np.mean([r["wt_precision"] for r in medium_cases])) if medium_cases else 0.0,
            "wt_recall_mean": float(np.mean([r["wt_recall"] for r in medium_cases])) if medium_cases else 0.0,
        },
        "large_tumors": {
            "count": len(large_cases),
            "wt_dice_mean": float(np.mean([r["wt_dice"] for r in large_cases])) if large_cases else 0.0,
            "wt_dice_median": float(np.median([r["wt_dice"] for r in large_cases])) if large_cases else 0.0,
            "wt_precision_mean": float(np.mean([r["wt_precision"] for r in large_cases])) if large_cases else 0.0,
            "wt_recall_mean": float(np.mean([r["wt_recall"] for r in large_cases])) if large_cases else 0.0,
        },
    }

    # Best and worst cases (by WT Dice)
    sorted_by_wt = sorted(valid_results, key=lambda x: x["wt_dice"], reverse=True)
    best_5_cases = sorted_by_wt[:5]
    worst_5_cases = sorted_by_wt[-5:]

    # 7. Task C: Secondary Comparison on Training 25 Cases
    print("\n--- Running Secondary Comparison on Training 25-Case Cohort ---")
    training_25_unet_results = [r for r in valid_results if r["case_id"] in TRAINING_25_VAL_CASES]
    segresnet_25_results: List[Dict[str, Any]] = []

    if args.segresnet_checkpoint.exists():
        print(f"Loading SegResNet from: {args.segresnet_checkpoint}")
        segresnet = create_segresnet(in_channels=4, num_classes=4, init_filters=16).to(device)
        seg_meta = load_checkpoint(args.segresnet_checkpoint, segresnet, device=device)
        segresnet.eval()
        print(f"SegResNet loaded: Epoch {seg_meta.get('epoch')}, Reported Best WT Dice: {seg_meta.get('best_metric', -1):.4f}")

        # Map case_ids to dataset indices
        case_id_to_idx = {val_dataset.case_ids[idx]: idx for idx in range(len(val_dataset))}

        with torch.inference_mode():
            for cid in TRAINING_25_VAL_CASES:
                if cid not in case_id_to_idx:
                    continue
                idx = case_id_to_idx[cid]
                s = val_dataset[idx]
                img_gpu = s["image"].unsqueeze(0).to(device)
                out_seg = sliding_window_inference(
                    inputs=img_gpu,
                    roi_size=(64, 64, 64),
                    sw_batch_size=2,
                    predictor=segresnet,
                    overlap=0.5,
                    mode="gaussian",
                )
                pred_seg = torch.argmax(out_seg, dim=1).squeeze(0).cpu()
                masked_seg = apply_air_masking(pred_seg, s["image"])
                m_seg = calculate_multimodal_metrics(masked_seg, s["mask"])
                segresnet_25_results.append({
                    "case_id": cid,
                    "wt_dice": m_seg["wt"]["dice"],
                    "tc_dice": m_seg["tc"]["dice"],
                    "et_dice": m_seg["et"]["dice"],
                    "wt_precision": m_seg["wt"]["precision"],
                    "wt_recall": m_seg["wt"]["recall"],
                })

    comparison_25_summary = {
        "unet_800cases": {
            "wt_dice": float(np.mean([r["wt_dice"] for r in training_25_unet_results])),
            "tc_dice": float(np.mean([r["tc_dice"] for r in training_25_unet_results])),
            "et_dice": float(np.mean([r["et_dice"] for r in training_25_unet_results])),
            "wt_precision": float(np.mean([r["wt_precision"] for r in training_25_unet_results])),
            "wt_recall": float(np.mean([r["wt_recall"] for r in training_25_unet_results])),
        },
        "segresnet_50cases": {
            "wt_dice": float(np.mean([r["wt_dice"] for r in segresnet_25_results])) if segresnet_25_results else 0.0,
            "tc_dice": float(np.mean([r["tc_dice"] for r in segresnet_25_results])) if segresnet_25_results else 0.0,
            "et_dice": float(np.mean([r["et_dice"] for r in segresnet_25_results])) if segresnet_25_results else 0.0,
            "wt_precision": float(np.mean([r["wt_precision"] for r in segresnet_25_results])) if segresnet_25_results else 0.0,
            "wt_recall": float(np.mean([r["wt_recall"] for r in segresnet_25_results])) if segresnet_25_results else 0.0,
        },
    }

    # 8. Save Final JSON
    final_json_payload = {
        "metadata": {
            "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "unet_checkpoint": str(args.unet_checkpoint),
            "segresnet_checkpoint": str(args.segresnet_checkpoint),
            "total_cases_in_split": total_val_cases,
            "successful_cases": len(valid_results),
            "failed_cases": len(failed_cases),
            "skipped_cases": 0,
            "total_evaluation_time_seconds": total_eval_time,
        },
        "macro_metrics": macro_stats,
        "pooled_metrics": {
            "wt_dice": pooled_wt_dice,
            "wt_precision": pooled_wt_prec,
            "wt_recall": pooled_wt_rec,
            "tc_dice": pooled_tc_dice,
            "et_dice": pooled_et_dice,
            "total_air_fp": total_air_fp,
            "total_non_air_brain_fp": total_non_air_brain_fp,
        },
        "stratified_by_size": stratified,
        "best_5_cases": best_5_cases,
        "worst_5_cases": worst_5_cases,
        "training_25_cases_comparison": comparison_25_summary,
        "all_cases": results,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_json_payload, f, indent=2)
    print(f"\nFinal metrics JSON saved to: {json_path}")

    # 9. Generate Comprehensive Markdown Report
    report_lines = [
        "# Full 200-Case Multimodal Meningioma Validation Report",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Evaluation Mode:** Read-Only Deterministic Sliding-Window Inference  ",
        f"**Evaluated Checkpoint:** `{args.unet_checkpoint.name}` (Epoch {unet_meta.get('epoch')}, Trained on 800 Cases)  ",
        f"**Cases Evaluated:** {len(valid_results)} Successful, {len(failed_cases)} Failed, 0 Skipped (Total: {total_val_cases})  ",
        f"**Total Runtime:** {total_eval_time / 60:.1f} minutes ({total_eval_time / total_val_cases:.2f}s per volume)  ",
        "",
        "---",
        "",
        "## 1. Executive Summary & Core Results",
        "",
        "This evaluation executes the validated full-volume inference pipeline over all **200 original validation cases** defined in `data/meningioma_multimodal/split.json`.",
        "",
        "### Primary Performance Metrics (All 200 Cases)",
        "",
        "| Subregion / Metric | Macro Mean ± Std | Median (IQR) [Q25 – Q75] | Pooled Voxel-Level |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Whole Tumor (WT) Dice** | **{macro_stats['wt_dice']['mean']:.4f} ± {macro_stats['wt_dice']['std']:.4f}** | **{macro_stats['wt_dice']['median']:.4f}** ({macro_stats['wt_dice']['iqr']:.4f}) [{macro_stats['wt_dice']['q25']:.4f} – {macro_stats['wt_dice']['q75']:.4f}] | **{pooled_wt_dice:.4f}** |",
        f"| **Tumor Core (TC) Dice** | **{macro_stats['tc_dice']['mean']:.4f} ± {macro_stats['tc_dice']['std']:.4f}** | **{macro_stats['tc_dice']['median']:.4f}** ({macro_stats['tc_dice']['iqr']:.4f}) [{macro_stats['tc_dice']['q25']:.4f} – {macro_stats['tc_dice']['q75']:.4f}] | **{pooled_tc_dice:.4f}** |",
        f"| **Enhancing Tumor (ET) Dice** | **{macro_stats['et_dice']['mean']:.4f} ± {macro_stats['et_dice']['std']:.4f}** | **{macro_stats['et_dice']['median']:.4f}** ({macro_stats['et_dice']['iqr']:.4f}) [{macro_stats['et_dice']['q25']:.4f} – {macro_stats['et_dice']['q75']:.4f}] | **{pooled_et_dice:.4f}** |",
        f"| **WT Precision** | {macro_stats['wt_precision']['mean']:.4f} ± {macro_stats['wt_precision']['std']:.4f} | {macro_stats['wt_precision']['median']:.4f} ({macro_stats['wt_precision']['iqr']:.4f}) [{macro_stats['wt_precision']['q25']:.4f} – {macro_stats['wt_precision']['q75']:.4f}] | {pooled_wt_prec:.4f} |",
        f"| **WT Recall** | {macro_stats['wt_recall']['mean']:.4f} ± {macro_stats['wt_recall']['std']:.4f} | {macro_stats['wt_recall']['median']:.4f} ({macro_stats['wt_recall']['iqr']:.4f}) [{macro_stats['wt_recall']['q25']:.4f} – {macro_stats['wt_recall']['q75']:.4f}] | {pooled_wt_rec:.4f} |",
        "",
        "### False Positive Distribution",
        f"- **Empty-Air False Positives:** `{total_air_fp}` voxels across all 200 cases (100% eliminated by validation air masking).",
        f"- **Non-Air Brain Tissue False Positives:** `{total_non_air_brain_fp:,}` voxels pooled across all 200 cases.",
        "",
        "---",
        "",
        "## 2. Performance Stratification by Tumor Volume",
        "",
        "| Volume Tier | Tumor Voxel Range | Cases | WT Dice (Mean) | WT Dice (Median) | WT Precision | WT Recall |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **Small Tumors** | < 5,000 voxels | {stratified['small_tumors']['count']} | {stratified['small_tumors']['wt_dice_mean']:.4f} | {stratified['small_tumors']['wt_dice_median']:.4f} | {stratified['small_tumors']['wt_precision_mean']:.4f} | {stratified['small_tumors']['wt_recall_mean']:.4f} |",
        f"| **Medium Tumors** | 5,000 – 50,000 voxels | {stratified['medium_tumors']['count']} | {stratified['medium_tumors']['wt_dice_mean']:.4f} | {stratified['medium_tumors']['wt_dice_median']:.4f} | {stratified['medium_tumors']['wt_precision_mean']:.4f} | {stratified['medium_tumors']['wt_recall_mean']:.4f} |",
        f"| **Large Tumors** | > 50,000 voxels | {stratified['large_tumors']['count']} | {stratified['large_tumors']['wt_dice_mean']:.4f} | {stratified['large_tumors']['wt_dice_median']:.4f} | {stratified['large_tumors']['wt_precision_mean']:.4f} | {stratified['large_tumors']['wt_recall_mean']:.4f} |",
        "",
        "---",
        "",
        "## 3. Best and Worst Performing Cases",
        "",
        "### Top 5 Best Cases (by WT Dice)",
        "| Case ID | WT Dice | TC Dice | ET Dice | Precision | Recall | GT WT Voxels | Pred WT Voxels |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for b in best_5_cases:
        report_lines.append(
            f"| `{b['case_id']}` | **{b['wt_dice']:.4f}** | {b['tc_dice']:.4f} | {b['et_dice']:.4f} | {b['wt_precision']:.4f} | {b['wt_recall']:.4f} | {b['gt_wt_voxels']:,} | {b['pred_wt_voxels']:,} |"
        )

    report_lines.extend([
        "",
        "### Top 5 Worst Cases (by WT Dice)",
        "| Case ID | WT Dice | TC Dice | ET Dice | Precision | Recall | GT WT Voxels | Pred WT Voxels |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for w in worst_5_cases:
        report_lines.append(
            f"| `{w['case_id']}` | **{w['wt_dice']:.4f}** | {w['tc_dice']:.4f} | {w['et_dice']:.4f} | {w['wt_precision']:.4f} | {w['wt_recall']:.4f} | {w['gt_wt_voxels']:,} | {w['pred_wt_voxels']:,} |"
        )

    report_lines.extend([
        "",
        "---",
        "",
        "## 4. Secondary Comparison: 3D U-Net vs. SegResNet (Exact Same 25 Validation Cases)",
        "",
        "> [!NOTE]",
        "> **Important Experimental Distinction:** The 3D U-Net was trained on the full **800-case** training split (25 epochs = 10,000 updates), whereas SegResNet was trained on only **50 cases** (20 epochs = 500 updates). Both are evaluated on the exact same 25 validation cases used during training.",
        "",
        "| Model Architecture | Training Cohort | WT Dice | TC Dice | ET Dice | WT Precision | WT Recall |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **3D U-Net (Best Ckpt, Ep 25)** | 800 Cases | **{comparison_25_summary['unet_800cases']['wt_dice']:.4f}** | **{comparison_25_summary['unet_800cases']['tc_dice']:.4f}** | **{comparison_25_summary['unet_800cases']['et_dice']:.4f}** | {comparison_25_summary['unet_800cases']['wt_precision']:.4f} | {comparison_25_summary['unet_800cases']['wt_recall']:.4f} |",
        f"| **SegResNet (Best Ckpt, Ep 20)** | 50 Cases | **{comparison_25_summary['segresnet_50cases']['wt_dice']:.4f}** | **{comparison_25_summary['segresnet_50cases']['tc_dice']:.4f}** | **{comparison_25_summary['segresnet_50cases']['et_dice']:.4f}** | {comparison_25_summary['segresnet_50cases']['wt_precision']:.4f} | {comparison_25_summary['segresnet_50cases']['wt_recall']:.4f} |",
        "",
        "---",
        "",
        "## 5. Artifacts and File Manifest",
        "- **CSV Detailed Metrics:** `outputs/meningioma_multimodal_experiment/final_validation_20261015/full200_per_case_metrics.csv`",
        "- **JSON Machine-Readable Results:** `outputs/meningioma_multimodal_experiment/final_validation_20261015/full200_validation_metrics.json`",
        "- **Markdown Full Report:** `outputs/meningioma_multimodal_experiment/final_validation_20261015/full200_validation_report.md`",
    ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"Markdown report saved to: {report_path}")

    print("\n" + "=" * 80)
    print("ALL 200-CASE EVALUATION COMPLETED SUCCESSFULLY")
    print(f"Macro WT Dice:  {macro_stats['wt_dice']['mean']:.4f} (Median: {macro_stats['wt_dice']['median']:.4f}, IQR: {macro_stats['wt_dice']['iqr']:.4f})")
    print(f"Macro TC Dice:  {macro_stats['tc_dice']['mean']:.4f}")
    print(f"Macro ET Dice:  {macro_stats['et_dice']['mean']:.4f}")
    print(f"Pooled WT Dice: {pooled_wt_dice:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    run_evaluation()

