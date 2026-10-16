"""
evaluate_multimodal_ensemble.py
===============================
Inference and Ensembling Evaluation Tool for Multimodal Meningioma Models.

Evaluates:
1. 3D U-Net Model alone
2. SegResNet Model alone (if provided)
3. Softmax Probability Ensemble (w1 * P_unet + w2 * P_segresnet)
4. Optional Test-Time Augmentation (TTA) with 3D spatial flips
5. Optional Connected-Component Noise Filtering

Outputs detailed comparative markdown reports and machine-readable JSON metrics.
"""

import argparse
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
import scipy.ndimage
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from models.meningioma_multimodal_unet import create_multimodal_unet
from models.meningioma_segresnet import create_segresnet
from training.meningioma_multimodal_dataset import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_SPLIT_JSON,
    MeningiomaMultimodalDataset,
    calculate_multimodal_metrics,
)
from training.train_meningioma_multimodal import (
    apply_air_masking,
    load_checkpoint,
    resolve_cohort_indices,
)


def predict_volume_probabilities(
    model: nn.Module,
    image: torch.Tensor,
    roi_size: Tuple[int, int, int] = (64, 64, 64),
    sw_batch_size: int = 2,
    overlap: float = 0.5,
    use_tta: bool = False,
) -> torch.Tensor:
    """
    Computes full-volume softmax probability map [1, num_classes, D, H, W]
    using sliding-window Gaussian inference, with optional 3D flip TTA.
    """
    model.eval()
    if not use_tta:
        logits = sliding_window_inference(
            inputs=image,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=overlap,
            mode="gaussian",
        )
        return torch.softmax(logits, dim=1)

    # 4-fold or 8-fold flip Test-Time Augmentation (TTA)
    flips = [
        [],        # original
        [2],       # flip D
        [3],       # flip H
        [4],       # flip W
        [2, 3],    # flip D, H
        [2, 4],    # flip D, W
        [3, 4],    # flip H, W
        [2, 3, 4], # flip D, H, W
    ]
    prob_accumulator = None
    for f_dims in flips:
        flipped_input = torch.flip(image, dims=f_dims) if f_dims else image
        logits = sliding_window_inference(
            inputs=flipped_input,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=overlap,
            mode="gaussian",
        )
        probs = torch.softmax(logits, dim=1)
        unflipped = torch.flip(probs, dims=f_dims) if f_dims else probs
        if prob_accumulator is None:
            prob_accumulator = unflipped
        else:
            prob_accumulator += unflipped

    return prob_accumulator / len(flips)


def filter_small_connected_components(
    pred_mask: torch.Tensor,
    min_size: int = 50,
) -> torch.Tensor:
    """Removes small isolated tumor clusters below min_size voxels."""
    if min_size <= 0:
        return pred_mask

    mask_np = pred_mask.numpy().astype(np.int32)
    binary_tumor = mask_np > 0
    labeled, num_features = scipy.ndimage.label(binary_tumor)
    if num_features == 0:
        return pred_mask

    counts = np.bincount(labeled.ravel())
    small_labels = np.where(counts < min_size)[0]
    cleaned = mask_np.copy()
    cleaned[np.isin(labeled, small_labels)] = 0
    return torch.from_numpy(cleaned).long()


def evaluate_cohort(
    models_dict: Dict[str, nn.Module],
    weights: Dict[str, float],
    dataset: Any,
    device: torch.device,
    roi_size: Tuple[int, int, int] = (64, 64, 64),
    sw_batch_size: int = 2,
    overlap: float = 0.5,
    use_tta: bool = False,
    min_component_size: int = 0,
) -> Dict[str, Any]:
    """
    Evaluates individual models and their weighted ensemble over the dataset.
    """
    eval_modes = list(models_dict.keys())
    if len(models_dict) > 1:
        eval_modes.append("ensemble")

    results_per_mode: Dict[str, List[Dict[str, Any]]] = {m: [] for m in eval_modes}
    t0 = time.time()

    with torch.inference_mode():
        for i in range(len(dataset)):
            sample = dataset[i]
            case_id = sample.get("case_id", f"case_{i:04d}")
            image = sample["image"].unsqueeze(0).to(device)  # [1, 4, D, H, W]
            gt_mask = sample["mask"]                        # [D, H, W]
            raw_input_cpu = sample["image"]

            probs_per_model: Dict[str, torch.Tensor] = {}
            for name, model in models_dict.items():
                p = predict_volume_probabilities(
                    model=model,
                    image=image,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    overlap=overlap,
                    use_tta=use_tta,
                )
                probs_per_model[name] = p

                # Evaluate individual model
                pred_ind = torch.argmax(p, dim=1).squeeze(0).cpu()
                pred_ind = apply_air_masking(pred_ind, raw_input_cpu)
                pred_ind = filter_small_connected_components(pred_ind, min_size=min_component_size)
                m_ind = calculate_multimodal_metrics(pred_ind, gt_mask)
                m_ind["case_id"] = case_id
                results_per_mode[name].append(m_ind)

            # Evaluate ensemble if multiple models present
            if len(models_dict) > 1:
                total_weight = sum(weights[name] for name in models_dict)
                ens_probs = sum(
                    (weights[name] / total_weight) * probs_per_model[name]
                    for name in models_dict
                )
                pred_ens = torch.argmax(ens_probs, dim=1).squeeze(0).cpu()
                pred_ens = apply_air_masking(pred_ens, raw_input_cpu)
                pred_ens = filter_small_connected_components(pred_ens, min_size=min_component_size)
                m_ens = calculate_multimodal_metrics(pred_ens, gt_mask)
                m_ens["case_id"] = case_id
                results_per_mode["ensemble"].append(m_ens)

            print(f"[{i+1}/{len(dataset)}] Evaluated {case_id}")

    elapsed = time.time() - t0

    # Aggregate summary metrics
    summary: Dict[str, Dict[str, float]] = {}
    for mode_name, case_metrics in results_per_mode.items():
        summary[mode_name] = {
            "wt_dice": float(np.mean([m["wt"]["dice"] for m in case_metrics])),
            "tc_dice": float(np.mean([m["tc"]["dice"] for m in case_metrics])),
            "et_dice": float(np.mean([m["et"]["dice"] for m in case_metrics])),
            "wt_precision": float(np.mean([m["wt"]["precision"] for m in case_metrics])),
            "wt_recall": float(np.mean([m["wt"]["recall"] for m in case_metrics])),
        }

    return {
        "summary": summary,
        "detailed_results": results_per_mode,
        "elapsed_seconds": elapsed,
        "cases_evaluated": len(dataset),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal Meningioma Ensemble Evaluator")
    parser.add_argument(
        "--unet-checkpoint",
        type=Path,
        required=True,
        help="Path to 3D U-Net checkpoint (.pth)",
    )
    parser.add_argument(
        "--segresnet-checkpoint",
        type=Path,
        default=None,
        help="Path to SegResNet checkpoint (.pth, optional for standalone or ensemble)",
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
        "--val-cases",
        type=int,
        default=25,
        help="Number of validation cases to evaluate (default: 25)",
    )
    parser.add_argument(
        "--cohort-selection",
        type=str,
        default="seed",
        choices=["seed", "prefix"],
        help="Cohort selection method ('seed' or 'prefix')",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "meningioma_ensemble_evaluation",
        help="Output directory for reports and metrics JSON",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Enable 3D Test-Time Augmentation (spatial flip averaging)",
    )
    parser.add_argument(
        "--min-component-size",
        type=int,
        default=50,
        help="Minimum connected component voxel size (removes isolated noise < N voxels, default: 50)",
    )
    parser.add_argument("--unet-weight", type=float, default=0.5, help="Ensemble weight for 3D U-Net")
    parser.add_argument("--segresnet-weight", type=float, default=0.5, help="Ensemble weight for SegResNet")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("MULTIMODAL MENINGIOMA ENSEMBLE EVALUATOR")
    print("=" * 80)
    print(f"Device:               {device}")
    print(f"3D U-Net Checkpoint:  {args.unet-checkpoint if hasattr(args, 'unet-checkpoint') else args.unet_checkpoint}")
    print(f"SegResNet Checkpoint: {args.segresnet_checkpoint}")
    print(f"Test-Time Aug (TTA):  {args.tta}")
    print(f"Min Component Size:   {args.min_component_size} voxels")
    print(f"Validation Cases:     {args.val_cases}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize validation dataset
    val_dataset = MeningiomaMultimodalDataset(
        split_json_path=args.split_json,
        partition="val",
        dataset_root=args.dataset_root,
        patch_size=None,
    )
    if args.val_cases is not None and args.val_cases < len(val_dataset):
        indices = resolve_cohort_indices(
            total_cases=len(val_dataset),
            requested_count=args.val_cases,
            seed=args.seed,
            mode=args.cohort_selection,
        )
        val_dataset = Subset(val_dataset, indices)

    models_dict = {}
    weights = {}

    # Load 3D U-Net
    print("\n--- Loading 3D U-Net ---")
    unet = create_multimodal_unet(in_channels=4, num_classes=4, base_channels=16).to(device)
    load_checkpoint(args.unet_checkpoint, unet, device=device)
    models_dict["unet"] = unet
    weights["unet"] = args.unet_weight

    # Load SegResNet (if provided)
    if args.segresnet_checkpoint is not None and args.segresnet_checkpoint.exists():
        print("\n--- Loading SegResNet ---")
        segresnet = create_segresnet(in_channels=4, num_classes=4, init_filters=16).to(device)
        load_checkpoint(args.segresnet_checkpoint, segresnet, device=device)
        models_dict["segresnet"] = segresnet
        weights["segresnet"] = args.segresnet_weight

    print(f"\n--- Running Evaluation over {len(val_dataset)} volumes ---")
    eval_results = evaluate_cohort(
        models_dict=models_dict,
        weights=weights,
        dataset=val_dataset,
        device=device,
        use_tta=args.tta,
        min_component_size=args.min_component_size,
    )

    # Save JSON results
    json_path = args.output_dir / "ensemble_evaluation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nJSON results saved to: {json_path}")

    # Generate Markdown Report
    report_path = args.output_dir / "ensemble_evaluation_report.md"
    summary = eval_results["summary"]

    lines = [
        "# Multimodal Meningioma Ensemble Evaluation Report",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Cases Evaluated:** {eval_results['cases_evaluated']}  ",
        f"**TTA Enabled:** {args.tta}  ",
        f"**Min Component Size:** {args.min_component_size} voxels  ",
        f"**Evaluation Duration:** {eval_results['elapsed_seconds']:.1f}s  \n",
        "## Summary Metrics Comparison\n",
        "| Model Configuration | WT Dice | TC Dice | ET Dice | WT Precision | WT Recall |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for mode_name, metrics in summary.items():
        title = mode_name.upper() if mode_name in ("unet", "segresnet") else "ENSEMBLE (U-Net + SegResNet)"
        lines.append(
            f"| **{title}** | **{metrics['wt_dice']:.4f}** | **{metrics['tc_dice']:.4f}** | "
            f"**{metrics['et_dice']:.4f}** | {metrics['wt_precision']:.4f} | {metrics['wt_recall']:.4f} |"
        )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Markdown report saved to: {report_path}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    for mode_name, metrics in summary.items():
        print(f"[{mode_name.upper()}] WT Dice: {metrics['wt_dice']:.4f} | TC Dice: {metrics['tc_dice']:.4f} | ET Dice: {metrics['et_dice']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()

