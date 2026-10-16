"""
Glioma Test-Time Augmentation (TTA) Soft-Voting Ensemble Evaluation.
===================================================================

Evaluates the frozen Glioma 3D U-Net checkpoint on the formal 20 validation cases
using 4-fold multi-axial spatial flip soft-voting ensemble:
  1. Original orientation
  2. Sagittal flip (X-axis)
  3. Coronal flip (Y-axis)
  4. Axial flip (Z-axis)

Compares Baseline vs TTA Soft-Voting Ensemble on:
  - WT Micro & Macro Dice
  - TC Micro & Macro Dice
  - ET Micro & Macro Dice
  - IoU, Precision, Recall
  - Volumetric Pearson r, R2, and MAE
"""

from pathlib import Path
import json
import sys
import random
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from scipy import ndimage, stats
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference

from preprocessing.glioma_dataset import DATASET_ROOT, get_cases, pad_to_patch_size
from preprocessing.glioma_preprocess import process_case
from preprocessing.glioma_split import create_train_val_split

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "glioma_training"
    / "glioma_best_medium_experiment.pth"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "glioma_tta_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROI_SIZE = (64, 64, 64)
PATCH_SIZE = ROI_SIZE
OVERLAP = 0.50
SW_BATCH_SIZE = 1
PROBABILITY_THRESHOLD = 0.60
MIN_COMPONENT_SIZE = 1000
CONNECTIVITY = 1


def filter_connected_components(binary_mask, min_size=MIN_COMPONENT_SIZE, connectivity=CONNECTIVITY):
    binary_mask = np.asarray(binary_mask, dtype=bool)
    if not np.any(binary_mask):
        return binary_mask.copy(), {"removed": 0}
    structure = ndimage.generate_binary_structure(3, connectivity)
    labeled_mask, num_features = ndimage.label(binary_mask, structure=structure)
    if num_features == 0:
        return binary_mask.copy(), {"removed": 0}
    component_sizes = np.bincount(labeled_mask.ravel())
    keep = component_sizes >= min_size
    keep[0] = False
    filtered_mask = keep[labeled_mask]
    return filtered_mask, {"removed": int(num_features - int(keep[1:].sum()))}


def compute_metrics(pred_b, gt_b):
    pred_b = np.asarray(pred_b, dtype=bool)
    gt_b = np.asarray(gt_b, dtype=bool)
    tp = int(np.logical_and(pred_b, gt_b).sum())
    fp = int(np.logical_and(pred_b, ~gt_b).sum())
    fn = int(np.logical_and(~pred_b, gt_b).sum())
    tn = int(np.logical_and(~pred_b, ~gt_b).sum())
    dice_denom = 2 * tp + fp + fn
    dice = float((2.0 * tp) / dice_denom) if dice_denom > 0 else (1.0 if (tp == 0 and fp == 0 and fn == 0) else 0.0)
    iou_denom = tp + fp + fn
    iou = float(tp / iou_denom) if iou_denom > 0 else (1.0 if (tp == 0 and fp == 0 and fn == 0) else 0.0)
    prec_denom = tp + fp
    precision = float(tp / prec_denom) if prec_denom > 0 else (1.0 if fn == 0 else 0.0)
    rec_denom = tp + fn
    recall = float(tp / rec_denom) if rec_denom > 0 else (1.0 if fp == 0 else 0.0)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "dice": dice, "iou": iou, "precision": precision, "recall": recall, "gt_vol": int(gt_b.sum()), "pred_vol": int(pred_b.sum())}


def load_glioma_model():
    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=0.0,
    ).to(DEVICE)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


@torch.inference_mode()
def predict_case(model, image_tensor, use_tta=True):
    """
    Inference with or without 4-fold multi-axial TTA soft voting.
    """
    # 1. Original orientation
    logits = sliding_window_inference(image_tensor, roi_size=ROI_SIZE, sw_batch_size=SW_BATCH_SIZE, predictor=model, overlap=OVERLAP, mode="gaussian")
    probs_orig = torch.softmax(logits, dim=1)

    if not use_tta:
        return probs_orig

    # 2. Sagittal flip (dim 4 -> X)
    img_flip_x = torch.flip(image_tensor, dims=[4])
    logits_x = sliding_window_inference(img_flip_x, roi_size=ROI_SIZE, sw_batch_size=SW_BATCH_SIZE, predictor=model, overlap=OVERLAP, mode="gaussian")
    probs_x = torch.flip(torch.softmax(logits_x, dim=1), dims=[4])

    # 3. Coronal flip (dim 3 -> Y)
    img_flip_y = torch.flip(image_tensor, dims=[3])
    logits_y = sliding_window_inference(img_flip_y, roi_size=ROI_SIZE, sw_batch_size=SW_BATCH_SIZE, predictor=model, overlap=OVERLAP, mode="gaussian")
    probs_y = torch.flip(torch.softmax(logits_y, dim=1), dims=[3])

    # 4. Axial flip (dim 2 -> Z)
    img_flip_z = torch.flip(image_tensor, dims=[2])
    logits_z = sliding_window_inference(img_flip_z, roi_size=ROI_SIZE, sw_batch_size=SW_BATCH_SIZE, predictor=model, overlap=OVERLAP, mode="gaussian")
    probs_z = torch.flip(torch.softmax(logits_z, dim=1), dims=[2])

    # Soft Voting Average (4 models in 1)
    probs_tta = (probs_orig + probs_x + probs_y + probs_z) / 4.0
    return probs_tta


def run_evaluation():
    print("=" * 70)
    print("GLIOMA TEST-TIME AUGMENTATION (TTA) SOFT-VOTING EVALUATION")
    print("=" * 70)
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    all_cases = get_cases(DATASET_ROOT)
    train_cases, val_cases = create_train_val_split(all_cases, train_ratio=0.90, seed=42)
    eval_cases = val_cases[:20]
    print(f"Total validation cases: {len(eval_cases)}")

    model = load_glioma_model()

    results_baseline = []
    results_tta = []

    print("\nProcessing cases side-by-side (Baseline vs TTA Soft-Voting)...")

    for idx, case_path in enumerate(eval_cases, start=1):
        case_name = case_path.name
        t0 = time.time()

        # Preprocess
        image, mask = process_case(case_path)
        image = np.asarray(image, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.int64)
        d, h, w = image.shape[1:]

        # Pad for patch sliding window
        mask_4d = np.expand_dims(mask, axis=0)
        padded_img, _ = pad_to_patch_size(image, mask_4d, PATCH_SIZE)
        tensor = torch.from_numpy(padded_img).float().unsqueeze(0).to(DEVICE)

        # Ground truth
        gt_mask = mask[:d, :h, :w]
        gt_wt = (gt_mask > 0)
        gt_tc = np.isin(gt_mask, [1, 3])
        gt_et = (gt_mask == 3)

        # A. Baseline prediction
        probs_base = predict_case(model, tensor, use_tta=False)
        wt_prob_base = probs_base[:, 1:, ...].sum(dim=1)[0].detach().cpu().numpy()[:d, :h, :w]
        classes_base = torch.argmax(probs_base, dim=1)[0].detach().cpu().numpy()[:d, :h, :w]
        wt_mask_base, _ = filter_connected_components(wt_prob_base >= PROBABILITY_THRESHOLD)
        multiclass_base = np.where(wt_mask_base, classes_base, 0)
        m_wt_b = compute_metrics(wt_mask_base, gt_wt)
        m_tc_b = compute_metrics(np.isin(multiclass_base, [1, 3]), gt_tc)
        m_et_b = compute_metrics(multiclass_base == 3, gt_et)

        # B. TTA Soft-Voting prediction
        probs_tta = predict_case(model, tensor, use_tta=True)
        wt_prob_tta = probs_tta[:, 1:, ...].sum(dim=1)[0].detach().cpu().numpy()[:d, :h, :w]
        classes_tta = torch.argmax(probs_tta, dim=1)[0].detach().cpu().numpy()[:d, :h, :w]
        wt_mask_tta, _ = filter_connected_components(wt_prob_tta >= PROBABILITY_THRESHOLD)
        multiclass_tta = np.where(wt_mask_tta, classes_tta, 0)
        m_wt_t = compute_metrics(wt_mask_tta, gt_wt)
        m_tc_t = compute_metrics(np.isin(multiclass_tta, [1, 3]), gt_tc)
        m_et_t = compute_metrics(multiclass_tta == 3, gt_et)

        dt = time.time() - t0
        print(f"[{idx:02d}/20] {case_name} ({dt:.1f}s) | Base WT Dice: {m_wt_b['dice']*100:.2f}% -> TTA WT Dice: {m_wt_t['dice']*100:.2f}% | Base TC: {m_tc_b['dice']*100:.2f}% -> TTA TC: {m_tc_t['dice']*100:.2f}% | Base ET: {m_et_b['dice']*100:.2f}% -> TTA ET: {m_et_t['dice']*100:.2f}%")

        results_baseline.append({"case": case_name, "wt": m_wt_b, "tc": m_tc_b, "et": m_et_b})
        results_tta.append({"case": case_name, "wt": m_wt_t, "tc": m_tc_t, "et": m_et_t})

    # Summary calculations
    def compute_aggregate(results):
        tp_wt = sum(r["wt"]["tp"] for r in results)
        fp_wt = sum(r["wt"]["fp"] for r in results)
        fn_wt = sum(r["wt"]["fn"] for r in results)
        micro_dice_wt = (2.0 * tp_wt) / (2 * tp_wt + fp_wt + fn_wt)
        micro_iou_wt = tp_wt / (tp_wt + fp_wt + fn_wt)
        micro_prec_wt = tp_wt / (tp_wt + fp_wt) if (tp_wt + fp_wt) > 0 else 1.0
        micro_rec_wt = tp_wt / (tp_wt + fn_wt) if (tp_wt + fn_wt) > 0 else 1.0

        tp_tc = sum(r["tc"]["tp"] for r in results)
        fp_tc = sum(r["tc"]["fp"] for r in results)
        fn_tc = sum(r["tc"]["fn"] for r in results)
        micro_dice_tc = (2.0 * tp_tc) / (2 * tp_tc + fp_tc + fn_tc) if (2 * tp_tc + fp_tc + fn_tc) > 0 else 0.0

        tp_et = sum(r["et"]["tp"] for r in results)
        fp_et = sum(r["et"]["fp"] for r in results)
        fn_et = sum(r["et"]["fn"] for r in results)
        micro_dice_et = (2.0 * tp_et) / (2 * tp_et + fp_et + fn_et) if (2 * tp_et + fp_et + fn_et) > 0 else 0.0

        macro_dice_wt = float(np.mean([r["wt"]["dice"] for r in results]))
        macro_dice_tc = float(np.mean([r["tc"]["dice"] for r in results]))
        macro_dice_et = float(np.mean([r["et"]["dice"] for r in results]))

        gt_vols = [r["wt"]["gt_vol"] for r in results]
        pred_vols = [r["wt"]["pred_vol"] for r in results]
        r_val, _ = stats.pearsonr(gt_vols, pred_vols)
        mae = float(np.mean(np.abs(np.array(pred_vols) - np.array(gt_vols))))

        return {
            "micro_dice_wt": float(micro_dice_wt),
            "micro_iou_wt": float(micro_iou_wt),
            "micro_prec_wt": float(micro_prec_wt),
            "micro_rec_wt": float(micro_rec_wt),
            "micro_dice_tc": float(micro_dice_tc),
            "micro_dice_et": float(micro_dice_et),
            "macro_dice_wt": macro_dice_wt,
            "macro_dice_tc": macro_dice_tc,
            "macro_dice_et": macro_dice_et,
            "pearson_r": float(r_val),
            "r2": float(r_val ** 2),
            "mae_mm3": mae,
        }

    agg_base = compute_aggregate(results_baseline)
    agg_tta = compute_aggregate(results_tta)

    print("\n" + "=" * 70)
    print("FINAL COMPARISON: BASELINE vs TTA SOFT-VOTING ENSEMBLE (20 CASES)")
    print("=" * 70)
    print(f"{'Metric':<30} | {'Baseline (No TTA)':<20} | {'TTA Soft-Voting':<20} | {'Difference':<15}")
    print("-" * 90)
    print(f"{'WT Micro Dice':<30} | {agg_base['micro_dice_wt']*100:.2f}%{'':<14} | {agg_tta['micro_dice_wt']*100:.2f}%{'':<14} | {(agg_tta['micro_dice_wt']-agg_base['micro_dice_wt'])*100:+.2f}%")
    print(f"{'WT Micro IoU':<30} | {agg_base['micro_iou_wt']*100:.2f}%{'':<14} | {agg_tta['micro_iou_wt']*100:.2f}%{'':<14} | {(agg_tta['micro_iou_wt']-agg_base['micro_iou_wt'])*100:+.2f}%")
    print(f"{'WT Micro Precision':<30} | {agg_base['micro_prec_wt']*100:.2f}%{'':<14} | {agg_tta['micro_prec_wt']*100:.2f}%{'':<14} | {(agg_tta['micro_prec_wt']-agg_base['micro_prec_wt'])*100:+.2f}%")
    print(f"{'WT Micro Recall':<30} | {agg_base['micro_rec_wt']*100:.2f}%{'':<14} | {agg_tta['micro_rec_wt']*100:.2f}%{'':<14} | {(agg_tta['micro_rec_wt']-agg_base['micro_rec_wt'])*100:+.2f}%")
    print(f"{'TC Micro Dice (Core)':<30} | {agg_base['micro_dice_tc']*100:.2f}%{'':<14} | {agg_tta['micro_dice_tc']*100:.2f}%{'':<14} | {(agg_tta['micro_dice_tc']-agg_base['micro_dice_tc'])*100:+.2f}%")
    print(f"{'ET Micro Dice (Enhancing)':<30} | {agg_base['micro_dice_et']*100:.2f}%{'':<14} | {agg_tta['micro_dice_et']*100:.2f}%{'':<14} | {(agg_tta['micro_dice_et']-agg_base['micro_dice_et'])*100:+.2f}%")
    print(f"{'WT Macro Dice':<30} | {agg_base['macro_dice_wt']*100:.2f}%{'':<14} | {agg_tta['macro_dice_wt']*100:.2f}%{'':<14} | {(agg_tta['macro_dice_wt']-agg_base['macro_dice_wt'])*100:+.2f}%")
    print(f"{'TC Macro Dice':<30} | {agg_base['macro_dice_tc']*100:.2f}%{'':<14} | {agg_tta['macro_dice_tc']*100:.2f}%{'':<14} | {(agg_tta['macro_dice_tc']-agg_base['macro_dice_tc'])*100:+.2f}%")
    print(f"{'ET Macro Dice':<30} | {agg_base['macro_dice_et']*100:.2f}%{'':<14} | {agg_tta['macro_dice_et']*100:.2f}%{'':<14} | {(agg_tta['macro_dice_et']-agg_base['macro_dice_et'])*100:+.2f}%")
    print(f"{'Pearson Volume r':<30} | {agg_base['pearson_r']:.4f}{'':<14} | {agg_tta['pearson_r']:.4f}{'':<14} | {agg_tta['pearson_r']-agg_base['pearson_r']:+.4f}")
    print(f"{'Volume R2':<30} | {agg_base['r2']:.4f}{'':<14} | {agg_tta['r2']:.4f}{'':<14} | {agg_tta['r2']-agg_base['r2']:+.4f}")
    print(f"{'MAE (mm3)':<30} | {agg_base['mae_mm3']:.1f} mm3{'':<10} | {agg_tta['mae_mm3']:.1f} mm3{'':<10} | {agg_tta['mae_mm3']-agg_base['mae_mm3']:+.1f} mm3")
    print("=" * 90)

    # Save comparison report
    output_report = {
        "baseline": agg_base,
        "tta_ensemble": agg_tta,
        "per_case_baseline": results_baseline,
        "per_case_tta": results_tta,
    }
    with open(OUTPUT_DIR / "tta_comparison_results.json", "w") as f:
        json.dump(output_report, f, indent=2)
    print(f"\n[OK] Results saved to {OUTPUT_DIR / 'tta_comparison_results.json'}")


if __name__ == "__main__":
    run_evaluation()
