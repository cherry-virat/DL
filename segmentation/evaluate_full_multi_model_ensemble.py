"""
Full 3-Model Soft-Voting Ensemble Evaluation.
============================================

Evaluates 3-Model Soft-Voting Ensembles against Single Baseline Models:
  1. Glioma: 20 Formal Validation Cases
     - Model 1: Baseline 3D U-Net
     - Model 2: Focal Weighted 3D U-Net (Seed 101)
     - Model 3: DiceFocal 3D U-Net (Seed 2024)
     - Ensemble Soft-Voting: (P1 + P2 + P3) / 3.0

  2. Meningioma: Development Cohort
     - Model 1: Baseline 3D U-Net
     - Model 2: High Sensitivity 3D U-Net (Seed 101)
     - Model 3: Cosine Annealed 3D U-Net (Seed 2024)
     - Ensemble Soft-Voting: (P1 + P2 + P3) / 3.0
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import time
import numpy as np
import torch
from scipy import ndimage, stats

from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference

from models.meningioma_model import create_meningioma_model
from preprocessing.glioma_dataset import DATASET_ROOT, get_cases, pad_to_patch_size
from preprocessing.glioma_preprocess import process_case
from preprocessing.glioma_split import create_train_val_split
from segmentation.meningioma_validation_inference import (
    pad_volume,
    calculate_positions,
    remove_small_components,
)

GLIOMA_CKPT_DIR = PROJECT_ROOT / "outputs" / "glioma_ensemble"
MENINGIOMA_CKPT_DIR = PROJECT_ROOT / "outputs" / "meningioma_ensemble"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "full_ensemble_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROI_SIZE = (64, 64, 64)
PATCH_SIZE = ROI_SIZE
OVERLAP = 0.50


def filter_cc(binary_mask, min_size=1000):
    binary_mask = np.asarray(binary_mask, dtype=bool)
    if not np.any(binary_mask):
        return binary_mask.copy()
    structure = ndimage.generate_binary_structure(3, 1)
    labeled_mask, num_features = ndimage.label(binary_mask, structure=structure)
    if num_features == 0:
        return binary_mask.copy()
    sizes = np.bincount(labeled_mask.ravel())
    keep = sizes >= min_size
    keep[0] = False
    return keep[labeled_mask]


def compute_metrics(pred_b, gt_b):
    pred_b = np.asarray(pred_b, dtype=bool)
    gt_b = np.asarray(gt_b, dtype=bool)
    tp = int(np.logical_and(pred_b, gt_b).sum())
    fp = int(np.logical_and(pred_b, ~gt_b).sum())
    fn = int(np.logical_and(~pred_b, gt_b).sum())
    dice = (2.0 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else (1.0 if (tp == 0 and fp == 0 and fn == 0) else 0.0)
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else (1.0 if (tp == 0 and fp == 0 and fn == 0) else 0.0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if fp == 0 else 0.0)
    return {"tp": tp, "fp": fp, "fn": fn, "dice": dice, "iou": iou, "precision": precision, "recall": recall, "gt_vol": int(gt_b.sum()), "pred_vol": int(pred_b.sum())}


def load_glioma_models():
    models = []
    for idx in [1, 2, 3]:
        p = GLIOMA_CKPT_DIR / f"glioma_model_{idx}.pth"
        if not p.exists():
            raise FileNotFoundError(f"Missing Glioma checkpoint: {p}")
        m = UNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=4,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            dropout=0.0,
        ).to(DEVICE)
        ckpt = torch.load(p, map_location=DEVICE)
        m.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
        m.eval()
        models.append(m)
    return models


def load_meningioma_models():
    models = []
    for idx in [1, 2, 3]:
        p = MENINGIOMA_CKPT_DIR / f"meningioma_model_{idx}.pth"
        if not p.exists():
            raise FileNotFoundError(f"Missing Meningioma checkpoint: {p}")
        m = create_meningioma_model(base_channels=16).to(DEVICE)
        ckpt = torch.load(p, map_location=DEVICE)
        m.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
        m.eval()
        models.append(m)
    return models


def evaluate_glioma_ensemble():
    print("=" * 80)
    print("EVALUATING 3-MODEL GLIOMA SOFT-VOTING ENSEMBLE (20 VALIDATION CASES)")
    print("=" * 80)

    models = load_glioma_models()
    all_cases = get_cases(DATASET_ROOT)
    _, val_cases = create_train_val_split(all_cases, train_ratio=0.90, seed=42)
    eval_cases = val_cases[:20]

    res_base = []
    res_ens = []

    for idx, case_path in enumerate(eval_cases, 1):
        image, mask = process_case(case_path)
        image = np.asarray(image, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.int64)
        d, h, w = image.shape[1:]

        padded_img, _ = pad_to_patch_size(image, np.expand_dims(mask, 0), PATCH_SIZE)
        tensor = torch.from_numpy(padded_img).float().unsqueeze(0).to(DEVICE)

        gt_mask = mask[:d, :h, :w]
        gt_wt = gt_mask > 0
        gt_tc = np.isin(gt_mask, [1, 3])
        gt_et = gt_mask == 3

        with torch.inference_mode():
            probs_list = []
            for m in models:
                logits = sliding_window_inference(tensor, roi_size=ROI_SIZE, sw_batch_size=1, predictor=m, overlap=OVERLAP, mode="gaussian")
                probs = torch.softmax(logits, dim=1)
                probs_list.append(probs)

            # Baseline = Model 1
            prob_base = probs_list[0][:, 1:, ...].sum(dim=1)[0].detach().cpu().numpy()[:d, :h, :w]
            classes_base = torch.argmax(probs_list[0], dim=1)[0].detach().cpu().numpy()[:d, :h, :w]

            # 3-Model Soft-Voting Ensemble Average
            prob_ens_tensor = (probs_list[0] + probs_list[1] + probs_list[2]) / 3.0
            prob_ens = prob_ens_tensor[:, 1:, ...].sum(dim=1)[0].detach().cpu().numpy()[:d, :h, :w]
            classes_ens = torch.argmax(prob_ens_tensor, dim=1)[0].detach().cpu().numpy()[:d, :h, :w]

        # Post-proc
        wt_base = filter_cc(prob_base >= 0.60, min_size=1000)
        multi_base = np.where(wt_base, classes_base, 0)
        m_wt_b = compute_metrics(wt_base, gt_wt)
        m_tc_b = compute_metrics(np.isin(multi_base, [1, 3]), gt_tc)
        m_et_b = compute_metrics(multi_base == 3, gt_et)

        wt_ens = filter_cc(prob_ens >= 0.55, min_size=1000)
        multi_ens = np.where(wt_ens, classes_ens, 0)
        m_wt_e = compute_metrics(wt_ens, gt_wt)
        m_tc_e = compute_metrics(np.isin(multi_ens, [1, 3]), gt_tc)
        m_et_e = compute_metrics(multi_ens == 3, gt_et)

        print(f"[{idx:02d}/20] {case_path.name} | Base WT Dice: {m_wt_b['dice']*100:.2f}% -> Ens WT Dice: {m_wt_e['dice']*100:.2f}% (Diff: {(m_wt_e['dice']-m_wt_b['dice'])*100:+.2f}%)")
        res_base.append({"wt": m_wt_b, "tc": m_tc_b, "et": m_et_b})
        res_ens.append({"wt": m_wt_e, "tc": m_tc_e, "et": m_et_e})

    def get_agg(results):
        tp_wt = sum(r["wt"]["tp"] for r in results)
        fp_wt = sum(r["wt"]["fp"] for r in results)
        fn_wt = sum(r["wt"]["fn"] for r in results)
        dice_wt = (2.0 * tp_wt) / (2 * tp_wt + fp_wt + fn_wt)
        iou_wt = tp_wt / (tp_wt + fp_wt + fn_wt)
        prec_wt = tp_wt / (tp_wt + fp_wt) if (tp_wt + fp_wt) > 0 else 1.0
        rec_wt = tp_wt / (tp_wt + fn_wt) if (tp_wt + fn_wt) > 0 else 1.0

        tp_tc = sum(r["tc"]["tp"] for r in results)
        fp_tc = sum(r["tc"]["fp"] for r in results)
        fn_tc = sum(r["tc"]["fn"] for r in results)
        dice_tc = (2.0 * tp_tc) / (2 * tp_tc + fp_tc + fn_tc) if (2 * tp_tc + fp_tc + fn_tc) > 0 else 0.0

        tp_et = sum(r["et"]["tp"] for r in results)
        fp_et = sum(r["et"]["fp"] for r in results)
        fn_et = sum(r["et"]["fn"] for r in results)
        dice_et = (2.0 * tp_et) / (2 * tp_et + fp_et + fn_et) if (2 * tp_et + fp_et + fn_et) > 0 else 0.0

        return {
            "wt_micro_dice": float(dice_wt),
            "wt_micro_iou": float(iou_wt),
            "wt_micro_prec": float(prec_wt),
            "wt_micro_rec": float(rec_wt),
            "tc_micro_dice": float(dice_tc),
            "et_micro_dice": float(dice_et),
            "wt_macro_dice": float(np.mean([r["wt"]["dice"] for r in results])),
            "tc_macro_dice": float(np.mean([r["tc"]["dice"] for r in results])),
            "et_macro_dice": float(np.mean([r["et"]["dice"] for r in results])),
        }

    agg_b = get_agg(res_base)
    agg_e = get_agg(res_ens)

    print("\n" + "=" * 80)
    print("GLIOMA 3-MODEL ENSEMBLE RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<25} | {'Single Model (Base)':<20} | {'3-Model Soft Ensemble':<22} | {'Gain':<12}")
    print("-" * 85)
    print(f"{'WT Micro Dice':<25} | {agg_b['wt_micro_dice']*100:.2f}%{'':<14} | {agg_e['wt_micro_dice']*100:.2f}%{'':<16} | {(agg_e['wt_micro_dice']-agg_b['wt_micro_dice'])*100:+.2f}%")
    print(f"{'WT Micro IoU':<25} | {agg_b['wt_micro_iou']*100:.2f}%{'':<14} | {agg_e['wt_micro_iou']*100:.2f}%{'':<16} | {(agg_e['wt_micro_iou']-agg_b['wt_micro_iou'])*100:+.2f}%")
    print(f"{'WT Micro Precision':<25} | {agg_b['wt_micro_prec']*100:.2f}%{'':<14} | {agg_e['wt_micro_prec']*100:.2f}%{'':<16} | {(agg_e['wt_micro_prec']-agg_b['wt_micro_prec'])*100:+.2f}%")
    print(f"{'WT Micro Recall':<25} | {agg_b['wt_micro_rec']*100:.2f}%{'':<14} | {agg_e['wt_micro_rec']*100:.2f}%{'':<16} | {(agg_e['wt_micro_rec']-agg_b['wt_micro_rec'])*100:+.2f}%")
    print(f"{'TC Micro Dice (Core)':<25} | {agg_b['tc_micro_dice']*100:.2f}%{'':<14} | {agg_e['tc_micro_dice']*100:.2f}%{'':<16} | {(agg_e['tc_micro_dice']-agg_b['tc_micro_dice'])*100:+.2f}%")
    print(f"{'ET Micro Dice (Enhancing)':<25} | {agg_b['et_micro_dice']*100:.2f}%{'':<14} | {agg_e['et_micro_dice']*100:.2f}%{'':<16} | {(agg_e['et_micro_dice']-agg_b['et_micro_dice'])*100:+.2f}%")
    print("=" * 85)


def evaluate_meningioma_ensemble(num_cases=25):
    print("\n" + "=" * 80)
    print(f"EVALUATING 3-MODEL MENINGIOMA SOFT-VOTING ENSEMBLE ({num_cases} DEV CASES)")
    print("=" * 80)

    models = load_meningioma_models()
    split_file = PROJECT_ROOT / "data" / "meningioma_split.json"
    with open(split_file, "r") as f:
        split_data = json.load(f)
    dev_cases = split_data.get("dev_cases", [])[:num_cases]
    prep_dir = PROJECT_ROOT / "data" / "meningioma_preprocessed"

    tp_b, fp_b, fn_b = 0, 0, 0
    tp_e, fp_e, fn_e = 0, 0, 0
    dices_b = []
    dices_e = []

    for idx, case_name in enumerate(dev_cases, 1):
        t1c_p = prep_dir / case_name / "t1c.npy"
        gtv_p = prep_dir / case_name / "gtv.npy"
        if not t1c_p.exists() or not gtv_p.exists():
            continue
        t1c = np.load(t1c_p)
        gtv = np.load(gtv_p) > 0

        padded, pad = pad_volume(t1c, (64, 64, 64))
        p_shape = padded.shape
        z_pos = calculate_positions(p_shape[0], 64, 0.5)
        y_pos = calculate_positions(p_shape[1], 64, 0.5)
        x_pos = calculate_positions(p_shape[2], 64, 0.5)

        prob_sums = [np.zeros(p_shape, dtype=np.float32) for _ in range(3)]
        cnt = np.zeros(p_shape, dtype=np.float32)

        with torch.inference_mode():
            for z in z_pos:
                for y in y_pos:
                    for x in x_pos:
                        patch = padded[z:z+64, y:y+64, x:x+64]
                        t_patch = torch.from_numpy(patch.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)
                        for m_idx, m in enumerate(models):
                            logits = m(t_patch)
                            pr = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()[0]
                            prob_sums[m_idx][z:z+64, y:y+64, x:x+64] += pr
                        cnt[z:z+64, y:y+64, x:x+64] += 1.0

        cnt = np.maximum(cnt, 1e-8)
        maps = [ps / cnt for ps in prob_sums]
        z_b, z_a = pad[0]
        y_b, y_a = pad[1]
        x_b, x_a = pad[2]
        z_e = None if z_a == 0 else -z_a
        y_e = None if y_a == 0 else -y_a
        x_e = None if x_a == 0 else -x_a

        map_b = maps[0][z_b:z_e, y_b:y_e, x_b:x_e]
        map_e = (maps[0] + maps[1] + maps[2])[z_b:z_e, y_b:y_e, x_b:x_e] / 3.0

        mask_b, _, _ = remove_small_components((map_b >= 0.125).astype(np.uint8), 2500)
        mask_e, _, _ = remove_small_components((map_e >= 0.125).astype(np.uint8), 2500)
        mask_b = mask_b > 0
        mask_e = mask_e > 0

        tpb = int(np.logical_and(mask_b, gtv).sum())
        fpb = int(np.logical_and(mask_b, ~gtv).sum())
        fnb = int(np.logical_and(~mask_b, gtv).sum())
        diceb = (2.0 * tpb) / (2 * tpb + fpb + fnb) if (2 * tpb + fpb + fnb) > 0 else (1.0 if (tpb==0 and fpb==0 and fnb==0) else 0.0)
        tp_b += tpb
        fp_b += fpb
        fn_b += fnb
        dices_b.append(diceb)

        tpe = int(np.logical_and(mask_e, gtv).sum())
        fpe = int(np.logical_and(mask_e, ~gtv).sum())
        fne = int(np.logical_and(~mask_e, gtv).sum())
        dicee = (2.0 * tpe) / (2 * tpe + fpe + fne) if (2 * tpe + fpe + fne) > 0 else (1.0 if (tpe==0 and fpe==0 and fne==0) else 0.0)
        tp_e += tpe
        fp_e += fpe
        fn_e += fne
        dices_e.append(dicee)

        print(f"[{idx:02d}/{len(dev_cases)}] {case_name} | Base Dice: {diceb*100:.2f}% -> Ens Dice: {dicee*100:.2f}% (Diff: {(dicee-diceb)*100:+.2f}%)")

    agg_dice_b = (2.0 * tp_b) / (2 * tp_b + fp_b + fn_b) if (2 * tp_b + fp_b + fn_b) > 0 else 0.0
    agg_dice_e = (2.0 * tp_e) / (2 * tp_e + fp_e + fn_e) if (2 * tp_e + fp_e + fn_e) > 0 else 0.0
    agg_iou_b = tp_b / (tp_b + fp_b + fn_b) if (tp_b + fp_b + fn_b) > 0 else 0.0
    agg_iou_e = tp_e / (tp_e + fp_e + fn_e) if (tp_e + fp_e + fn_e) > 0 else 0.0

    print("\n" + "=" * 80)
    print("MENINGIOMA 3-MODEL ENSEMBLE RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<25} | {'Single Model (Base)':<20} | {'3-Model Soft Ensemble':<22} | {'Gain':<12}")
    print("-" * 85)
    print(f"{'Micro Dice':<25} | {agg_dice_b*100:.2f}%{'':<14} | {agg_dice_e*100:.2f}%{'':<16} | {(agg_dice_e-agg_dice_b)*100:+.2f}%")
    print(f"{'Micro IoU':<25} | {agg_iou_b*100:.2f}%{'':<14} | {agg_iou_e*100:.2f}%{'':<16} | {(agg_iou_e-agg_iou_b)*100:+.2f}%")
    print(f"{'Macro Dice (Mean)':<25} | {np.mean(dices_b)*100:.2f}%{'':<14} | {np.mean(dices_e)*100:.2f}%{'':<16} | {(np.mean(dices_e)-np.mean(dices_b))*100:+.2f}%")
    print("=" * 85)


if __name__ == "__main__":
    evaluate_glioma_ensemble()
    evaluate_meningioma_ensemble()

