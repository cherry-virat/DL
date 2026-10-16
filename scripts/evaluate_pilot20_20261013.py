"""
evaluate_pilot20_20261013.py
============================
Read-only evaluation script for the pilot20_20261013 checkpoint:
    outputs/meningioma_multimodal_experiment/pilot20_20261013/checkpoints/meningioma_multimodal_best.pth

Evaluates the exact same 5 validation cases with and without air masking.
Saves results inside:
    outputs/meningioma_multimodal_experiment/pilot20_20261013/pilot20_evaluation.json
"""

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

from monai.inferers import sliding_window_inference
import numpy as np
import torch
from torch.utils.data import Subset

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
from training.train_meningioma_multimodal import apply_air_masking, load_checkpoint


def run_pilot20_evaluation():
    run_dir = PROJECT_ROOT / "outputs/meningioma_multimodal_experiment/pilot20_20261013"
    checkpoint_path = run_dir / "checkpoints/meningioma_multimodal_best.pth"
    split_json_path = PROJECT_ROOT / "data/meningioma_multimodal/split.json"
    dataset_root = Path(r"C:\3D_DATASET\BraTS2023_MEN")

    print("=" * 80)
    print("READ-ONLY EVALUATION: PILOT20_20261013 CHECKPOINT")
    print("=" * 80)
    print(f"Target Checkpoint: {checkpoint_path}")
    print(f"Split JSON:        {split_json_path}")
    print(f"Dataset Root:      {dataset_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device:    {device}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    model = create_multimodal_unet(in_channels=4, num_classes=4, base_channels=16).to(device)
    chk_metadata = load_checkpoint(checkpoint_path, model, device=device)
    model.eval()
    print(f"Loaded checkpoint epoch: {chk_metadata.get('epoch')}, best metric: {chk_metadata.get('best_metric'):.4f}")

    val_dataset_full = MeningiomaMultimodalDataset(
        split_json_path=split_json_path,
        partition="val",
        dataset_root=dataset_root,
        patch_size=None,
    )

    target_case_ids = [
        "BraTS-MEN-00020-000",
        "BraTS-MEN-00025-000",
        "BraTS-MEN-00040-000",
        "BraTS-MEN-00043-000",
        "BraTS-MEN-00052-000",
    ]

    case_indices = [val_dataset_full.case_ids.index(cid) for cid in target_case_ids]
    val_subset = Subset(val_dataset_full, case_indices)
    print(f"Evaluating {len(val_subset)} target cases: {target_case_ids}")

    patch_size_tuple = (64, 64, 64)
    sw_batch_size = 2
    sw_overlap = 0.5

    case_evaluations: List[Dict[str, Any]] = []

    pooled = {
        "raw": {
            "wt_tp": 0, "wt_fp": 0, "wt_fn": 0,
            "tc_tp": 0, "tc_fp": 0, "tc_fn": 0,
            "et_tp": 0, "et_fp": 0, "et_fn": 0,
            "air_fp": 0, "non_air_fp": 0,
            "pred_counts": {0: 0, 1: 0, 2: 0, 3: 0},
        },
        "masked": {
            "wt_tp": 0, "wt_fp": 0, "wt_fn": 0,
            "tc_tp": 0, "tc_fp": 0, "tc_fn": 0,
            "et_tp": 0, "et_fp": 0, "et_fn": 0,
            "air_fp": 0, "non_air_fp": 0,
            "pred_counts": {0: 0, 1: 0, 2: 0, 3: 0},
        },
        "gt_counts": {0: 0, 1: 0, 2: 0, 3: 0},
        "total_voxels": 0,
        "total_air_voxels": 0,
        "total_brain_voxels": 0,
    }

    t0_all = time.time()

    with torch.inference_mode():
        for i, sample in enumerate(val_subset):
            cid = sample["case_id"]
            img_cpu = sample["image"]
            mask_cpu = sample["mask"]
            total_voxels = mask_cpu.numel()

            img_gpu = img_cpu.unsqueeze(0).to(device)

            t_case_start = time.time()
            output = sliding_window_inference(
                inputs=img_gpu,
                roi_size=patch_size_tuple,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=sw_overlap,
                mode="gaussian",
            )
            t_infer = time.time() - t_case_start

            has_nan_logits = bool(torch.isnan(output).any().item())
            has_inf_logits = bool(torch.isinf(output).any().item())
            probs = torch.softmax(output, dim=1)
            has_nan_probs = bool(torch.isnan(probs).any().item())
            has_inf_probs = bool(torch.isinf(probs).any().item())

            raw_pred = torch.argmax(output, dim=1).squeeze(0).cpu()
            masked_pred = apply_air_masking(raw_pred, img_cpu)

            is_air = torch.all(img_cpu == 0, dim=0)
            air_count = int(is_air.sum().item())
            brain_count = total_voxels - air_count
            air_proportion = air_count / total_voxels

            gt_counts = {c: int((mask_cpu == c).sum().item()) for c in range(4)}
            gt_wt = gt_counts[1] + gt_counts[2] + gt_counts[3]
            gt_tc = gt_counts[1] + gt_counts[3]
            gt_et = gt_counts[3]

            m_raw = calculate_multimodal_metrics(raw_pred, mask_cpu)
            m_masked = calculate_multimodal_metrics(masked_pred, mask_cpu)

            raw_counts = {c: int((raw_pred == c).sum().item()) for c in range(4)}
            raw_wt = raw_counts[1] + raw_counts[2] + raw_counts[3]
            raw_tc = raw_counts[1] + raw_counts[3]
            raw_et = raw_counts[3]

            masked_counts = {c: int((masked_pred == c).sum().item()) for c in range(4)}
            masked_wt = masked_counts[1] + masked_counts[2] + masked_counts[3]
            masked_tc = masked_counts[1] + masked_counts[3]
            masked_et = masked_counts[3]

            gt_is_tumor = (mask_cpu >= 1) & (mask_cpu <= 3)

            raw_is_tumor = (raw_pred >= 1) & (raw_pred <= 3)
            raw_fp_air = int((raw_is_tumor & (~gt_is_tumor) & is_air).sum().item())
            raw_fp_brain = int((raw_is_tumor & (~gt_is_tumor) & (~is_air)).sum().item())

            masked_is_tumor = (masked_pred >= 1) & (masked_pred <= 3)
            masked_fp_air = int((masked_is_tumor & (~gt_is_tumor) & is_air).sum().item())
            masked_fp_brain = int((masked_is_tumor & (~gt_is_tumor) & (~is_air)).sum().item())

            ratio_raw_wt = (raw_wt / gt_wt) if gt_wt > 0 else float("nan")
            ratio_raw_tc = (raw_tc / gt_tc) if gt_tc > 0 else float("nan")
            ratio_raw_et = (raw_et / gt_et) if gt_et > 0 else float("nan")

            ratio_masked_wt = (masked_wt / gt_wt) if gt_wt > 0 else float("nan")
            ratio_masked_tc = (masked_tc / gt_tc) if gt_tc > 0 else float("nan")
            ratio_masked_et = (masked_et / gt_et) if gt_et > 0 else float("nan")

            pooled["total_voxels"] += total_voxels
            pooled["total_air_voxels"] += air_count
            pooled["total_brain_voxels"] += brain_count
            for c in range(4):
                pooled["gt_counts"][c] += gt_counts[c]
                pooled["raw"]["pred_counts"][c] += raw_counts[c]
                pooled["masked"]["pred_counts"][c] += masked_counts[c]

            for reg in ["wt", "tc", "et"]:
                pooled["raw"][f"{reg}_tp"] += m_raw[reg]["tp"]
                pooled["raw"][f"{reg}_fp"] += m_raw[reg]["fp"]
                pooled["raw"][f"{reg}_fn"] += m_raw[reg]["fn"]

                pooled["masked"][f"{reg}_tp"] += m_masked[reg]["tp"]
                pooled["masked"][f"{reg}_fp"] += m_masked[reg]["fp"]
                pooled["masked"][f"{reg}_fn"] += m_masked[reg]["fn"]

            pooled["raw"]["air_fp"] += raw_fp_air
            pooled["raw"]["non_air_fp"] += raw_fp_brain
            pooled["masked"]["air_fp"] += masked_fp_air
            pooled["masked"]["non_air_fp"] += masked_fp_brain

            case_data = {
                "case_id": cid,
                "total_voxels": total_voxels,
                "air_voxels": air_count,
                "brain_voxels": brain_count,
                "air_proportion": air_proportion,
                "has_nan_logits": has_nan_logits,
                "has_inf_logits": has_inf_logits,
                "has_nan_probs": has_nan_probs,
                "has_inf_probs": has_inf_probs,
                "gt_counts": gt_counts,
                "gt_subregions": {"wt": gt_wt, "tc": gt_tc, "et": gt_et},
                "raw": {
                    "pred_counts": raw_counts,
                    "pred_subregions": {"wt": raw_wt, "tc": raw_tc, "et": raw_et},
                    "metrics": m_raw,
                    "ratios": {"wt": ratio_raw_wt, "tc": ratio_raw_tc, "et": ratio_raw_et},
                    "fp_air": raw_fp_air,
                    "fp_non_air": raw_fp_brain,
                },
                "masked": {
                    "pred_counts": masked_counts,
                    "pred_subregions": {"wt": masked_wt, "tc": masked_tc, "et": masked_et},
                    "metrics": m_masked,
                    "ratios": {"wt": ratio_masked_wt, "tc": ratio_masked_tc, "et": ratio_masked_et},
                    "fp_air": masked_fp_air,
                    "fp_non_air": masked_fp_brain,
                },
                "infer_time_seconds": t_infer,
            }
            case_evaluations.append(case_data)
            print(
                f"[{i+1}/5] {cid} ({t_infer:.2f}s) | "
                f"WT Dice: Raw={m_raw['wt']['dice']:.4f} -> Masked={m_masked['wt']['dice']:.4f} | "
                f"WT Prec: Raw={m_raw['wt']['precision']:.4f} -> Masked={m_masked['wt']['precision']:.4f} | "
                f"Air FP: {raw_fp_air:,} -> {masked_fp_air:,}"
            )

    total_eval_time = time.time() - t0_all

    def compute_macro_averages(mode_key: str):
        return {
            "wt_dice": float(np.mean([c[mode_key]["metrics"]["wt"]["dice"] for c in case_evaluations])),
            "wt_precision": float(np.mean([c[mode_key]["metrics"]["wt"]["precision"] for c in case_evaluations])),
            "wt_recall": float(np.mean([c[mode_key]["metrics"]["wt"]["recall"] for c in case_evaluations])),
            "tc_dice": float(np.mean([c[mode_key]["metrics"]["tc"]["dice"] for c in case_evaluations])),
            "tc_precision": float(np.mean([c[mode_key]["metrics"]["tc"]["precision"] for c in case_evaluations])),
            "tc_recall": float(np.mean([c[mode_key]["metrics"]["tc"]["recall"] for c in case_evaluations])),
            "et_dice": float(np.mean([c[mode_key]["metrics"]["et"]["dice"] for c in case_evaluations])),
            "et_precision": float(np.mean([c[mode_key]["metrics"]["et"]["precision"] for c in case_evaluations])),
            "et_recall": float(np.mean([c[mode_key]["metrics"]["et"]["recall"] for c in case_evaluations])),
            "ratio_wt": float(np.mean([c[mode_key]["ratios"]["wt"] for c in case_evaluations])),
            "ratio_tc": float(np.mean([c[mode_key]["ratios"]["tc"] for c in case_evaluations])),
            "ratio_et": float(np.mean([c[mode_key]["ratios"]["et"] for c in case_evaluations])),
        }

    macro_raw = compute_macro_averages("raw")
    macro_masked = compute_macro_averages("masked")

    def compute_pooled_metrics(mode_key: str):
        res = {}
        for reg in ["wt", "tc", "et"]:
            tp = pooled[mode_key][f"{reg}_tp"]
            fp = pooled[mode_key][f"{reg}_fp"]
            fn = pooled[mode_key][f"{reg}_fn"]
            denom_d = 2 * tp + fp + fn
            dice = (2.0 * tp / denom_d) if denom_d > 0 else (1.0 if (tp + fp + fn == 0) else 0.0)
            prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            res[f"{reg}_dice"] = float(dice)
            res[f"{reg}_precision"] = float(prec)
            res[f"{reg}_recall"] = float(rec)
            res[f"{reg}_tp"] = tp
            res[f"{reg}_fp"] = fp
            res[f"{reg}_fn"] = fn
        res["air_fp"] = pooled[mode_key]["air_fp"]
        res["non_air_fp"] = pooled[mode_key]["non_air_fp"]
        return res

    pooled_raw = compute_pooled_metrics("raw")
    pooled_masked = compute_pooled_metrics("masked")

    full_results = {
        "checkpoint": str(checkpoint_path),
        "case_evaluations": case_evaluations,
        "macro_metrics": {
            "before_air_masking": macro_raw,
            "after_air_masking": macro_masked,
        },
        "pooled_metrics": {
            "before_air_masking": pooled_raw,
            "after_air_masking": pooled_masked,
        },
        "pooled_counts": {
            "total_voxels": pooled["total_voxels"],
            "total_air_voxels": pooled["total_air_voxels"],
            "total_brain_voxels": pooled["total_brain_voxels"],
            "air_proportion": pooled["total_air_voxels"] / pooled["total_voxels"],
            "gt_counts": pooled["gt_counts"],
            "raw_pred_counts": pooled["raw"]["pred_counts"],
            "masked_pred_counts": pooled["masked"]["pred_counts"],
        },
        "evaluation_time_seconds": total_eval_time,
    }

    json_out = run_dir / "pilot20_evaluation.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nSaved evaluation to: {json_out}")

    print("\n" + "=" * 80)
    print("PILOT20_20261013 EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Macro Averages (5 cases):")
    print(f"  WT Dice:      {macro_raw['wt_dice']:.4f} -> {macro_masked['wt_dice']:.4f}")
    print(f"  WT Precision: {macro_raw['wt_precision']:.4f} -> {macro_masked['wt_precision']:.4f}")
    print(f"  WT Recall:    {macro_raw['wt_recall']:.4f} -> {macro_masked['wt_recall']:.4f}")
    print(f"  TC Dice:      {macro_raw['tc_dice']:.4f} -> {macro_masked['tc_dice']:.4f}")
    print(f"  ET Dice:      {macro_raw['et_dice']:.4f} -> {macro_masked['et_dice']:.4f}")
    print(f"Pooled Voxel-Level Metrics:")
    print(f"  WT Dice:      {pooled_raw['wt_dice']:.4f} -> {pooled_masked['wt_dice']:.4f}")
    print(f"  WT Precision: {pooled_raw['wt_precision']:.4f} -> {pooled_masked['wt_precision']:.4f}")
    print(f"  WT Recall:    {pooled_raw['wt_recall']:.4f} -> {pooled_masked['wt_recall']:.4f}")
    print(f"  Air FP:       {pooled_raw['air_fp']:,} -> {pooled_masked['air_fp']:,}")
    print(f"  Non-Air FP:   {pooled_raw['non_air_fp']:,} -> {pooled_masked['non_air_fp']:,}")


if __name__ == "__main__":
    run_pilot20_evaluation()

