"""
diagnose_meningioma_multimodal_pilot.py
======================================
Read-only diagnostic script for evaluating the 4-channel multimodal meningioma pilot checkpoint.
Performs detailed per-case and pooled voxel-level analysis on the 5 pilot validation cases.
Does NOT modify any checkpoint, training log, or dataset file.
"""

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset
from monai.inferers import sliding_window_inference

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.meningioma_multimodal_unet import create_multimodal_unet
from training.meningioma_multimodal_dataset import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_SPLIT_JSON,
    MeningiomaMultimodalDataset,
    calculate_multimodal_metrics,
)


def run_diagnostics():
    checkpoint_path = PROJECT_ROOT / "outputs/meningioma_multimodal_experiment/checkpoints/meningioma_multimodal_best.pth"
    split_json_path = PROJECT_ROOT / "data/meningioma_multimodal/split.json"
    dataset_root = Path(r"C:\3D_DATASET\BraTS2023_MEN")

    print("=" * 80)
    print("READ-ONLY DIAGNOSTIC AUDIT: PILOT CHECKPOINT EVALUATION")
    print("=" * 80)
    print(f"Target Checkpoint: {checkpoint_path}")
    print(f"Split JSON:        {split_json_path}")
    print(f"Dataset Root:      {dataset_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device:    {device}")

    # 1. Load Checkpoint (read-only, weights_only=False to support WindowsPath in args)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    chk = torch.load(checkpoint_path, map_location=device, weights_only=False)
    epoch = chk.get("epoch", "unknown")
    best_metric = chk.get("best_metric", "unknown")
    chk_args = chk.get("args", {})
    history = chk.get("history", {})
    print(f"Checkpoint Epoch:  {epoch}")
    print(f"Recorded Metric:   {best_metric}")
    print(f"Recorded History:  {history}")

    # 2. Instantiate Model
    model = create_multimodal_unet(
        in_channels=4,
        num_classes=4,
        base_channels=chk_args.get("base_channels", 16),
    ).to(device)

    model.load_state_dict(chk["model_state_dict"])
    model.eval()
    print("Model architecture instantiated and weights loaded successfully.")

    # 3. Load Validation Dataset
    val_dataset_full = MeningiomaMultimodalDataset(
        split_json_path=split_json_path,
        partition="val",
        dataset_root=dataset_root,
        patch_size=None,
    )

    pilot_val_cases = chk_args.get("pilot_val_cases", 5)
    indices = list(range(min(pilot_val_cases, len(val_dataset_full))))
    val_subset = Subset(val_dataset_full, indices)
    print(f"Evaluating {len(val_subset)} pilot validation cases...")

    patch_size_tuple = tuple(chk_args.get("patch_size", [64, 64, 64]))
    sw_batch_size = chk_args.get("sw_batch_size", 2)
    sw_overlap = chk_args.get("sw_overlap", 0.5)

    case_reports = []
    pooled_stats = {
        "gt_counts": {0: 0, 1: 0, 2: 0, 3: 0},
        "pred_counts": {0: 0, 1: 0, 2: 0, 3: 0},
        "air_voxels": 0,
        "brain_voxels": 0,
        "air_pred_tumor": 0,
        "brain_pred_tumor": 0,
        "wt_tp": 0, "wt_fp": 0, "wt_fn": 0,
        "tc_tp": 0, "tc_fp": 0, "tc_fn": 0,
        "et_tp": 0, "et_fp": 0, "et_fn": 0,
    }

    t_start = time.time()

    with torch.inference_mode():
        for i, sample in enumerate(val_subset):
            case_id = sample["case_id"]
            image = sample["image"].unsqueeze(0).to(device)  # [1, 4, D, H, W]
            mask = sample["mask"]  # [D, H, W]
            total_voxels = mask.numel()

            # Sliding-window inference
            t_case0 = time.time()
            output = sliding_window_inference(
                inputs=image,
                roi_size=patch_size_tuple,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=sw_overlap,
                mode="gaussian",
            )
            t_infer = time.time() - t_case0

            # Check finite outputs
            is_finite_output = bool(torch.isfinite(output).all().item())
            prob = torch.softmax(output, dim=1)
            is_finite_prob = bool(torch.isfinite(prob).all().item())

            # Logit summary
            mean_logits = output.mean(dim=(0, 2, 3, 4)).cpu().tolist()
            min_logits = output.amin(dim=(0, 2, 3, 4)).cpu().tolist()
            max_logits = output.amax(dim=(0, 2, 3, 4)).cpu().tolist()

            # Discrete prediction via argmax on channel dim=1
            pred = torch.argmax(output, dim=1).squeeze(0).cpu()  # [D, H, W]

            # Tissue mask: where non-zero in any channel
            img_cpu = sample["image"]  # [4, D, H, W]
            is_brain = (img_cpu.abs().sum(dim=0) > 1e-4)  # brain parenchyma
            is_air = ~is_brain

            brain_cnt = int(is_brain.sum().item())
            air_cnt = int(is_air.sum().item())

            # Voxel counts
            gt_counts = {c: int((mask == c).sum().item()) for c in range(4)}
            pred_counts = {c: int((pred == c).sum().item()) for c in range(4)}

            pred_tumor_mask = (pred >= 1)
            air_pred_tumor = int((is_air & pred_tumor_mask).sum().item())
            brain_pred_tumor = int((is_brain & pred_tumor_mask).sum().item())

            # Subregion ground truth and predicted counts
            gt_wt = gt_counts[1] + gt_counts[2] + gt_counts[3]
            gt_tc = gt_counts[1] + gt_counts[3]
            gt_et = gt_counts[3]

            pred_wt = pred_counts[1] + pred_counts[2] + pred_counts[3]
            pred_tc = pred_counts[1] + pred_counts[3]
            pred_et = pred_counts[3]

            # Volume ratios (pred / gt)
            ratio_wt = (pred_wt / gt_wt) if gt_wt > 0 else float("nan")
            ratio_tc = (pred_tc / gt_tc) if gt_tc > 0 else float("nan")
            ratio_et = (pred_et / gt_et) if gt_et > 0 else float("nan")

            # Standard metrics
            metrics = calculate_multimodal_metrics(pred, mask)

            # Fractions of total volume
            pred_fracs = {c: pred_counts[c] / total_voxels for c in range(4)}

            # Update pooled stats
            for c in range(4):
                pooled_stats["gt_counts"][c] += gt_counts[c]
                pooled_stats["pred_counts"][c] += pred_counts[c]
            pooled_stats["air_voxels"] += air_cnt
            pooled_stats["brain_voxels"] += brain_cnt
            pooled_stats["air_pred_tumor"] += air_pred_tumor
            pooled_stats["brain_pred_tumor"] += brain_pred_tumor
            pooled_stats["wt_tp"] += metrics["wt"]["tp"]
            pooled_stats["wt_fp"] += metrics["wt"]["fp"]
            pooled_stats["wt_fn"] += metrics["wt"]["fn"]
            pooled_stats["tc_tp"] += metrics["tc"]["tp"]
            pooled_stats["tc_fp"] += metrics["tc"]["fp"]
            pooled_stats["tc_fn"] += metrics["tc"]["fn"]
            pooled_stats["et_tp"] += metrics["et"]["tp"]
            pooled_stats["et_fp"] += metrics["et"]["fp"]
            pooled_stats["et_fn"] += metrics["et"]["fn"]

            report = {
                "case_id": case_id,
                "total_voxels": total_voxels,
                "brain_voxels": brain_cnt,
                "air_voxels": air_cnt,
                "air_pred_tumor": air_pred_tumor,
                "air_pred_tumor_pct": air_pred_tumor / max(air_cnt, 1) * 100,
                "brain_pred_tumor": brain_pred_tumor,
                "brain_pred_tumor_pct": brain_pred_tumor / max(brain_cnt, 1) * 100,
                "is_finite_output": is_finite_output,
                "is_finite_prob": is_finite_prob,
                "mean_logits": mean_logits,
                "min_logits": min_logits,
                "max_logits": max_logits,
                "gt_counts": gt_counts,
                "pred_counts": pred_counts,
                "pred_fracs": pred_fracs,
                "gt_subregions": {"wt": gt_wt, "tc": gt_tc, "et": gt_et},
                "pred_subregions": {"wt": pred_wt, "tc": pred_tc, "et": pred_et},
                "ratios": {"wt": ratio_wt, "tc": ratio_tc, "et": ratio_et},
                "metrics": metrics,
                "infer_time_seconds": t_infer,
            }
            case_reports.append(report)
            print(f"[{i+1}/{len(val_subset)}] {case_id} evaluated in {t_infer:.2f}s | WT Dice: {metrics['wt']['dice']:.4f} | Prec: {metrics['wt']['precision']:.4f} | Rec: {metrics['wt']['recall']:.4f}")

    total_time = time.time() - t_start

    # Macro averages
    macro_wt_dice = float(np.mean([r["metrics"]["wt"]["dice"] for r in case_reports]))
    macro_tc_dice = float(np.mean([r["metrics"]["tc"]["dice"] for r in case_reports]))
    macro_et_dice = float(np.mean([r["metrics"]["et"]["dice"] for r in case_reports]))
    macro_wt_prec = float(np.mean([r["metrics"]["wt"]["precision"] for r in case_reports]))
    macro_wt_rec = float(np.mean([r["metrics"]["wt"]["recall"] for r in case_reports]))

    macro_tc_prec = float(np.mean([r["metrics"]["tc"]["precision"] for r in case_reports]))
    macro_tc_rec = float(np.mean([r["metrics"]["tc"]["recall"] for r in case_reports]))
    macro_et_prec = float(np.mean([r["metrics"]["et"]["precision"] for r in case_reports]))
    macro_et_rec = float(np.mean([r["metrics"]["et"]["recall"] for r in case_reports]))

    # Pooled voxel-level metrics
    pooled_wt_dice = 2.0 * pooled_stats["wt_tp"] / (2 * pooled_stats["wt_tp"] + pooled_stats["wt_fp"] + pooled_stats["wt_fn"])
    pooled_wt_prec = pooled_stats["wt_tp"] / (pooled_stats["wt_tp"] + pooled_stats["wt_fp"]) if (pooled_stats["wt_tp"] + pooled_stats["wt_fp"]) > 0 else 0.0
    pooled_wt_rec = pooled_stats["wt_tp"] / (pooled_stats["wt_tp"] + pooled_stats["wt_fn"]) if (pooled_stats["wt_tp"] + pooled_stats["wt_fn"]) > 0 else 0.0

    pooled_tc_dice = 2.0 * pooled_stats["tc_tp"] / (2 * pooled_stats["tc_tp"] + pooled_stats["tc_fp"] + pooled_stats["tc_fn"])
    pooled_tc_prec = pooled_stats["tc_tp"] / (pooled_stats["tc_tp"] + pooled_stats["tc_fp"]) if (pooled_stats["tc_tp"] + pooled_stats["tc_fp"]) > 0 else 0.0
    pooled_tc_rec = pooled_stats["tc_tp"] / (pooled_stats["tc_tp"] + pooled_stats["tc_fn"]) if (pooled_stats["tc_tp"] + pooled_stats["tc_fn"]) > 0 else 0.0

    pooled_et_dice = 2.0 * pooled_stats["et_tp"] / (2 * pooled_stats["et_tp"] + pooled_stats["et_fp"] + pooled_stats["et_fn"])
    pooled_et_prec = pooled_stats["et_tp"] / (pooled_stats["et_tp"] + pooled_stats["et_fp"]) if (pooled_stats["et_tp"] + pooled_stats["et_fp"]) > 0 else 0.0
    pooled_et_rec = pooled_stats["et_tp"] / (pooled_stats["et_tp"] + pooled_stats["et_fn"]) if (pooled_stats["et_tp"] + pooled_stats["et_fn"]) > 0 else 0.0

    output_summary = {
        "case_reports": case_reports,
        "macro_metrics": {
            "wt_dice": macro_wt_dice,
            "tc_dice": macro_tc_dice,
            "et_dice": macro_et_dice,
            "wt_precision": macro_wt_prec,
            "wt_recall": macro_wt_rec,
            "tc_precision": macro_tc_prec,
            "tc_recall": macro_tc_rec,
            "et_precision": macro_et_prec,
            "et_recall": macro_et_rec,
        },
        "pooled_metrics": {
            "wt_dice": pooled_wt_dice,
            "tc_dice": pooled_tc_dice,
            "et_dice": pooled_et_dice,
            "wt_precision": pooled_wt_prec,
            "wt_recall": pooled_wt_rec,
            "tc_precision": pooled_tc_prec,
            "tc_recall": pooled_tc_rec,
            "et_precision": pooled_et_prec,
            "et_recall": pooled_et_rec,
        },
        "pooled_stats": pooled_stats,
        "total_eval_time_seconds": total_time,
    }

    # Print clean summary table
    print("\n" + "=" * 80)
    print("DETAILED PER-CASE DIAGNOSTIC RESULTS")
    print("=" * 80)
    for r in case_reports:
        cid = r["case_id"]
        gt = r["gt_subregions"]
        pr = r["pred_subregions"]
        rat = r["ratios"]
        m = r["metrics"]
        print(f"\nCase ID: {cid}")
        print(f"  GT Voxels:   0={r['gt_counts'][0]:,}, 1={r['gt_counts'][1]:,}, 2={r['gt_counts'][2]:,}, 3={r['gt_counts'][3]:,}")
        print(f"  Pred Voxels: 0={r['pred_counts'][0]:,}, 1={r['pred_counts'][1]:,}, 2={r['pred_counts'][2]:,}, 3={r['pred_counts'][3]:,}")
        print(f"  WT (1+2+3):  GT={gt['wt']:,} | Pred={pr['wt']:,} | Ratio={rat['wt']:.1f}x | Dice={m['wt']['dice']:.4f} | Prec={m['wt']['precision']:.4f} | Rec={m['wt']['recall']:.4f}")
        print(f"  TC (1+3):    GT={gt['tc']:,} | Pred={pr['tc']:,} | Ratio={rat['tc']:.1f}x | Dice={m['tc']['dice']:.4f} | Prec={m['tc']['precision']:.4f} | Rec={m['tc']['recall']:.4f}")
        print(f"  ET (3):      GT={gt['et']:,} | Pred={pr['et']:,} | Ratio={rat['et']:.1f}x | Dice={m['et']['dice']:.4f} | Prec={m['et']['precision']:.4f} | Rec={m['et']['recall']:.4f}")
        print(f"  Brain voxels pred tumor: {r['brain_pred_tumor']:,} / {r['brain_voxels']:,} ({r['brain_pred_tumor_pct']:.2f}%)")
        print(f"  Air voxels pred tumor:   {r['air_pred_tumor']:,} / {r['air_voxels']:,} ({r['air_pred_tumor_pct']:.2f}%)")
        print(f"  Mean Logits: Class0={r['mean_logits'][0]:.4f}, Class1={r['mean_logits'][1]:.4f}, Class2={r['mean_logits'][2]:.4f}, Class3={r['mean_logits'][3]:.4f}")

    print("\n" + "=" * 80)
    print("AGGREGATE METRICS COMPARISON")
    print("=" * 80)
    print(f"Macro Average Across 5 Cases:")
    print(f"  WT Dice: {macro_wt_dice:.4f} | Prec: {macro_wt_prec:.4f} | Rec: {macro_wt_rec:.4f}")
    print(f"  TC Dice: {macro_tc_dice:.4f} | Prec: {macro_tc_prec:.4f} | Rec: {macro_tc_rec:.4f}")
    print(f"  ET Dice: {macro_et_dice:.4f} | Prec: {macro_et_prec:.4f} | Rec: {macro_et_rec:.4f}")
    print(f"\nPooled Voxel-Level (All 5 Cases Combined):")
    print(f"  WT Dice: {pooled_wt_dice:.4f} | Prec: {pooled_wt_prec:.4f} | Rec: {pooled_wt_rec:.4f}")
    print(f"  TC Dice: {pooled_tc_dice:.4f} | Prec: {pooled_tc_prec:.4f} | Rec: {pooled_tc_rec:.4f}")
    print(f"  ET Dice: {pooled_et_dice:.4f} | Prec: {pooled_et_prec:.4f} | Rec: {pooled_et_rec:.4f}")

    # Save to a temporary json for the report generator to read
    out_json = PROJECT_ROOT / "outputs/meningioma_multimodal_experiment/diagnostic_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output_summary, f, indent=2)
    print(f"\nDiagnostic results saved to: {out_json}")


if __name__ == "__main__":
    run_diagnostics()
