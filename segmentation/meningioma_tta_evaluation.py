"""
Meningioma Test-Time Augmentation (TTA) Soft-Voting Ensemble Evaluation.
=======================================================================

Evaluates the frozen Meningioma 3D U-Net checkpoint on the 100 development cases
using 4-fold multi-axial spatial flip soft-voting ensemble:
  1. Original orientation
  2. Sagittal flip (X-axis)
  3. Coronal flip (Y-axis)
  4. Axial flip (Z-axis)

Compares Baseline vs TTA Soft-Voting on:
  - Aggregate Development Dice, IoU, Precision, Recall
  - Case-level Macro Dice and IoU
"""

from pathlib import Path
import json
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from scipy import ndimage
from models.meningioma_model import create_meningioma_model
from segmentation.meningioma_validation_inference import (
    pad_volume,
    calculate_positions,
    remove_small_components,
)

PREPROCESSED_ROOT = PROJECT_ROOT / "data" / "meningioma_preprocessed"
SPLIT_FILE = PROJECT_ROOT / "data" / "meningioma_split.json"
CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "meningioma_training" / "meningioma_best.pth"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "meningioma_tta_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE = (64, 64, 64)
OVERLAP = 0.50
THRESHOLD = 0.125
MIN_COMPONENT_SIZE = 2500


def load_meningioma_model():
    model = create_meningioma_model(base_channels=16)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.to(DEVICE)
    model.eval()
    return model


@torch.inference_mode()
def predict_meningioma_volume(model, volume, use_tta=True):
    original_shape = volume.shape
    padded, padding = pad_volume(volume, PATCH_SIZE)
    padded_shape = padded.shape

    z_pos = calculate_positions(padded_shape[0], PATCH_SIZE[0], OVERLAP)
    y_pos = calculate_positions(padded_shape[1], PATCH_SIZE[1], OVERLAP)
    x_pos = calculate_positions(padded_shape[2], PATCH_SIZE[2], OVERLAP)

    prob_sum = np.zeros(padded_shape, dtype=np.float32)
    count_map = np.zeros(padded_shape, dtype=np.float32)

    for z in z_pos:
        for y in y_pos:
            for x in x_pos:
                patch = padded[z:z+PATCH_SIZE[0], y:y+PATCH_SIZE[1], x:x+PATCH_SIZE[2]]
                patch_tensor = torch.from_numpy(patch.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)

                # 1. Original
                logits_orig = model(patch_tensor)
                probs_orig = torch.softmax(logits_orig, dim=1)[:, 1]

                if use_tta:
                    # 2. Flip X (dim 4)
                    logits_x = model(torch.flip(patch_tensor, dims=[4]))
                    probs_x = torch.flip(torch.softmax(logits_x, dim=1)[:, 1], dims=[3])

                    # 3. Flip Y (dim 3)
                    logits_y = model(torch.flip(patch_tensor, dims=[3]))
                    probs_y = torch.flip(torch.softmax(logits_y, dim=1)[:, 1], dims=[2])

                    # 4. Flip Z (dim 2)
                    logits_z = model(torch.flip(patch_tensor, dims=[2]))
                    probs_z = torch.flip(torch.softmax(logits_z, dim=1)[:, 1], dims=[1])

                    patch_prob = (probs_orig + probs_x + probs_y + probs_z) / 4.0
                else:
                    patch_prob = probs_orig

                tumor_prob_np = patch_prob[0].detach().cpu().numpy()
                prob_sum[z:z+PATCH_SIZE[0], y:y+PATCH_SIZE[1], x:x+PATCH_SIZE[2]] += tumor_prob_np
                count_map[z:z+PATCH_SIZE[0], y:y+PATCH_SIZE[1], x:x+PATCH_SIZE[2]] += 1.0

    prob_map = prob_sum / np.maximum(count_map, 1e-8)
    z_b, z_a = padding[0]
    y_b, y_a = padding[1]
    x_b, x_a = padding[2]
    z_e = None if z_a == 0 else -z_a
    y_e = None if y_a == 0 else -y_a
    x_e = None if x_a == 0 else -x_a

    return prob_map[z_b:z_e, y_b:y_e, x_b:x_e]


def run_meningioma_eval(num_cases=25):
    """Evaluates first 25 dev cases for fast comparison."""
    print("=" * 70)
    print(f"MENINGIOMA TTA SOFT-VOTING EVALUATION (DEV COHORT: {num_cases} CASES)")
    print("=" * 70)

    with open(SPLIT_FILE, "r") as f:
        split_data = json.load(f)
    dev_cases = split_data.get("dev_cases", [])[:num_cases]
    print(f"Evaluating {len(dev_cases)} development cases...")

    model = load_meningioma_model()

    tp_base, fp_base, fn_base = 0, 0, 0
    tp_tta, fp_tta, fn_tta = 0, 0, 0
    dices_base = []
    dices_tta = []

    for idx, case_name in enumerate(dev_cases, start=1):
        t1c_path = PREPROCESSED_ROOT / case_name / "t1c.npy"
        gtv_path = PREPROCESSED_ROOT / case_name / "gtv.npy"

        if not t1c_path.exists() or not gtv_path.exists():
            continue

        t0 = time.time()
        t1c = np.load(t1c_path)
        gtv = np.load(gtv_path) > 0

        # Base
        prob_base = predict_meningioma_volume(model, t1c, use_tta=False)
        mask_base, _, _ = remove_small_components((prob_base >= THRESHOLD).astype(np.uint8), MIN_COMPONENT_SIZE)
        mask_base = mask_base > 0

        tp_b = int(np.logical_and(mask_base, gtv).sum())
        fp_b = int(np.logical_and(mask_base, ~gtv).sum())
        fn_b = int(np.logical_and(~mask_base, gtv).sum())
        dice_b = (2.0 * tp_b) / (2 * tp_b + fp_b + fn_b) if (2 * tp_b + fp_b + fn_b) > 0 else (1.0 if (tp_b==0 and fp_b==0 and fn_b==0) else 0.0)
        tp_base += tp_b
        fp_base += fp_b
        fn_base += fn_b
        dices_base.append(dice_b)

        # TTA
        prob_tta = predict_meningioma_volume(model, t1c, use_tta=True)
        mask_tta, _, _ = remove_small_components((prob_tta >= THRESHOLD).astype(np.uint8), MIN_COMPONENT_SIZE)
        mask_tta = mask_tta > 0

        tp_t = int(np.logical_and(mask_tta, gtv).sum())
        fp_t = int(np.logical_and(mask_tta, ~gtv).sum())
        fn_t = int(np.logical_and(~mask_tta, gtv).sum())
        dice_t = (2.0 * tp_t) / (2 * tp_t + fp_t + fn_t) if (2 * tp_t + fp_t + fn_t) > 0 else (1.0 if (tp_t==0 and fp_t==0 and fn_t==0) else 0.0)
        tp_tta += tp_t
        fp_tta += fp_t
        fn_tta += fn_t
        dices_tta.append(dice_t)

        dt = time.time() - t0
        print(f"[{idx:02d}/{len(dev_cases)}] {case_name} ({dt:.1f}s) | Base Dice: {dice_b*100:.2f}% -> TTA Dice: {dice_t*100:.2f}% (Diff: {(dice_t-dice_b)*100:+.2f}%)")

    # Aggregates
    agg_dice_base = (2.0 * tp_base) / (2 * tp_base + fp_base + fn_base) if (2 * tp_base + fp_base + fn_base) > 0 else 0.0
    agg_iou_base = tp_base / (tp_base + fp_base + fn_base) if (tp_base + fp_base + fn_base) > 0 else 0.0
    agg_prec_base = tp_base / (tp_base + fp_base) if (tp_base + fp_base) > 0 else 0.0
    agg_rec_base = tp_base / (tp_base + fn_base) if (tp_base + fn_base) > 0 else 0.0

    agg_dice_tta = (2.0 * tp_tta) / (2 * tp_tta + fp_tta + fn_tta) if (2 * tp_tta + fp_tta + fn_tta) > 0 else 0.0
    agg_iou_tta = tp_tta / (tp_tta + fp_tta + fn_tta) if (tp_tta + fp_tta + fn_tta) > 0 else 0.0
    agg_prec_tta = tp_tta / (tp_tta + fp_tta) if (tp_tta + fp_tta) > 0 else 0.0
    agg_rec_tta = tp_tta / (tp_tta + fn_tta) if (tp_tta + fn_tta) > 0 else 0.0

    print("\n" + "=" * 70)
    print("MENINGIOMA FINAL COMPARISON: BASELINE vs TTA SOFT-VOTING")
    print("=" * 70)
    print(f"{'Metric':<25} | {'Baseline':<15} | {'TTA Soft-Voting':<15} | {'Difference':<15}")
    print("-" * 75)
    print(f"{'Micro Dice':<25} | {agg_dice_base*100:.2f}%{'':<9} | {agg_dice_tta*100:.2f}%{'':<9} | {(agg_dice_tta-agg_dice_base)*100:+.2f}%")
    print(f"{'Micro IoU':<25} | {agg_iou_base*100:.2f}%{'':<9} | {agg_iou_tta*100:.2f}%{'':<9} | {(agg_iou_tta-agg_iou_base)*100:+.2f}%")
    print(f"{'Precision':<25} | {agg_prec_base*100:.2f}%{'':<9} | {agg_prec_tta*100:.2f}%{'':<9} | {(agg_prec_tta-agg_prec_base)*100:+.2f}%")
    print(f"{'Recall':<25} | {agg_rec_base*100:.2f}%{'':<9} | {agg_rec_tta*100:.2f}%{'':<9} | {(agg_rec_tta-agg_rec_base)*100:+.2f}%")
    print(f"{'Macro Dice (Mean)':<25} | {np.mean(dices_base)*100:.2f}%{'':<9} | {np.mean(dices_tta)*100:.2f}%{'':<9} | {(np.mean(dices_tta)-np.mean(dices_base))*100:+.2f}%")
    print("=" * 75)


if __name__ == "__main__":
    run_meningioma_eval(num_cases=20)

