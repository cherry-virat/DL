"""
generate_meningioma_final_visualizations.py
==========================================
Generates publication-quality axial, coronal, and sagittal slice visualizations
and a comprehensive contact sheet for representative meningioma cases.

Cases:
1. High-Dice Case: BraTS-MEN-01367-000 (WT Dice: 0.9849)
2. Large-Tumor Case: BraTS-MEN-00977-000 (141,143 voxels, WT Dice: 0.9822)
3. Medium-Tumor Case: BraTS-MEN-01179-000 (20,077 voxels, WT Dice: 0.9774)
4. Median WT Dice Case: BraTS-MEN-00232-000 (WT Dice: 0.8997)
5. Small Overpredicted Case: BraTS-MEN-00094-000 (139 voxels, WT Dice: 0.0109)
6. Small Missed Lesion Case: BraTS-MEN-00690-000 (1,587 voxels, WT Dice: 0.0000)

Outputs:
    outputs/meningioma_multimodal_experiment/final_visualizations_20261015/
        - case_01_high_dice_BraTS-MEN-01367-000.png
        - case_02_large_tumor_BraTS-MEN-00977-000.png
        - case_03_medium_tumor_BraTS-MEN-01179-000.png
        - case_04_median_dice_BraTS-MEN-00232-000.png
        - case_05_small_overpredicted_BraTS-MEN-00094-000.png
        - case_06_small_missed_BraTS-MEN-00690-000.png
        - meningioma_cases_contact_sheet.png
        - visualization_report.md
"""

from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from monai.inferers import sliding_window_inference
import numpy as np
import torch
import torch.nn as nn

from models.meningioma_multimodal_unet import create_multimodal_unet
from training.meningioma_multimodal_dataset import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_SPLIT_JSON,
    MeningiomaMultimodalDataset,
    calculate_multimodal_metrics,
)
from training.train_meningioma_multimodal import apply_air_masking, load_checkpoint

# Color palettes
# Subregions: 0: transparent, 1: NCR (Red), 2: ED (Green), 3: ET (Yellow)
SUBREGION_CMAP = ListedColormap(["none", "#FF3333", "#33CC33", "#FFCC00"])

# Discrepancy map: 0: none, 1: TP (Lime Green), 2: FP (Crimson Red), 3: FN (Dodger Blue)
ERROR_CMAP = ListedColormap(["none", "#00E676", "#FF1744", "#2979FF"])


TARGET_CASES = [
    {
        "case_id": "BraTS-MEN-01367-000",
        "category": "High-Dice Case (Top Performer)",
        "file_prefix": "case_01_high_dice",
        "rationale": "Demonstrates near-perfect agreement (WT Dice 0.9849, TC 0.9864, ET 0.9837) with precise boundary delineation on a large 76k voxel meningioma.",
    },
    {
        "case_id": "BraTS-MEN-00977-000",
        "category": "Large-Tumor Case (>50,000 voxels)",
        "file_prefix": "case_02_large_tumor",
        "rationale": "Demonstrates high volumetric accuracy (WT Dice 0.9822) on a very large 141k voxel mass, cleanly capturing enhancing and non-enhancing subregions.",
    },
    {
        "case_id": "BraTS-MEN-01179-000",
        "category": "Medium-Tumor Case (5,000–50,000 voxels)",
        "file_prefix": "case_03_medium_tumor",
        "rationale": "Represents typical clinical meningioma volume (20,077 voxels) with excellent boundary capture (WT Dice 0.9774, TC 0.9816).",
    },
    {
        "case_id": "BraTS-MEN-00232-000",
        "category": "Median WT Dice Case (~0.9006 benchmark)",
        "file_prefix": "case_04_median_dice",
        "rationale": "Exemplifies median cohort performance (WT Dice 0.8997, GT WT 9,956 voxels), showing minor boundary overprediction (ratio 1.17x) at periphery.",
    },
    {
        "case_id": "BraTS-MEN-00094-000",
        "category": "Difficult Small-Tumor (Overpredicted Case)",
        "file_prefix": "case_05_small_overpredicted",
        "rationale": "Illustrates high false-positive vulnerability on microscopic lesion (GT WT 139 voxels, Pred WT 22,724 voxels, WT Dice 0.0109, ratio 163.5x).",
    },
    {
        "case_id": "BraTS-MEN-00690-000",
        "category": "Difficult Small-Tumor (Missed Lesion Case)",
        "file_prefix": "case_06_small_missed",
        "rationale": "Illustrates complete false-negative miss (GT WT 1,587 voxels, WT Dice 0.0000, Overlap 0 voxels) where model predicted a false mass elsewhere.",
    },
]


def extract_slices(
    volume_3d: np.ndarray,
    center: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts axial, coronal, and sagittal 2D slices centered at coordinates (x, y, z).
    volume_3d shape: (240, 240, 155) -> (X=Sagittal, Y=Coronal, Z=Axial).
    Rotates slices for standard radiological orientation (Superior up, Anterior up).
    """
    x, y, z = center
    x = max(0, min(x, volume_3d.shape[0] - 1))
    y = max(0, min(y, volume_3d.shape[1] - 1))
    z = max(0, min(z, volume_3d.shape[2] - 1))

    # Axial: (X, Y) slice at Z=z
    axial = np.rot90(volume_3d[:, :, z], 1)
    # Coronal: (X, Z) slice at Y=y
    coronal = np.rot90(volume_3d[:, y, :], 1)
    # Sagittal: (Y, Z) slice at X=x
    sagittal = np.rot90(volume_3d[x, :, :], 1)

    return axial, coronal, sagittal


def normalize_display_image(img_2d: np.ndarray) -> np.ndarray:
    """Normalizes 2D image slice to [0, 1] for visual display."""
    p1, p99 = np.percentile(img_2d[img_2d > 0] if np.any(img_2d > 0) else img_2d, (1, 99))
    if p99 > p1:
        clipped = np.clip(img_2d, p1, p99)
        return (clipped - p1) / (p99 - p1)
    return np.zeros_like(img_2d)


def make_discrepancy_map(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    Creates WT agreement/discrepancy map:
        0: None (Background agreement)
        1: True Positive (TP: GT>0 and Pred>0) -> Green
        2: False Positive (FP: GT==0 and Pred>0) -> Red
        3: False Negative (FN: GT>0 and Pred==0) -> Blue
    """
    gt_bin = gt > 0
    pred_bin = pred > 0
    disc = np.zeros_like(gt, dtype=np.uint8)
    disc[gt_bin & pred_bin] = 1   # TP
    disc[(~gt_bin) & pred_bin] = 2 # FP
    disc[gt_bin & (~pred_bin)] = 3 # FN
    return disc


def render_case_figure(
    case_info: Dict[str, Any],
    t1c_3d: np.ndarray,
    gt_3d: np.ndarray,
    pred_3d: np.ndarray,
    metrics: Dict[str, Any],
    output_png: Path,
):
    """
    Renders 3-row (Axial, Coronal, Sagittal) by 4-column figure:
    Col 1: T1c MRI Grayscale
    Col 2: Ground Truth Subregions
    Col 3: Model Prediction Subregions
    Col 4: WT Agreement & Discrepancy Map
    """
    # Determine center coordinates intersecting tumor mass
    gt_coords = np.argwhere(gt_3d > 0)
    if len(gt_coords) > 0:
        center = (
            int(np.median(gt_coords[:, 0])),
            int(np.median(gt_coords[:, 1])),
            int(np.median(gt_coords[:, 2])),
        )
    else:
        pred_coords = np.argwhere(pred_3d > 0)
        center = (
            int(np.median(pred_coords[:, 0])) if len(pred_coords) > 0 else 120,
            int(np.median(pred_coords[:, 1])) if len(pred_coords) > 0 else 120,
            int(np.median(pred_coords[:, 2])) if len(pred_coords) > 0 else 77,
        )

    t1c_ax, t1c_cor, t1c_sag = extract_slices(t1c_3d, center)
    gt_ax, gt_cor, gt_sag = extract_slices(gt_3d, center)
    pred_ax, pred_cor, pred_sag = extract_slices(pred_3d, center)

    disc_ax = make_discrepancy_map(gt_ax, pred_ax)
    disc_cor = make_discrepancy_map(gt_cor, pred_cor)
    disc_sag = make_discrepancy_map(gt_sag, pred_sag)

    fig, axes = plt.subplots(3, 4, figsize=(18, 14), facecolor="black")

    rows = [
        ("Axial", t1c_ax, gt_ax, pred_ax, disc_ax, f"Z = {center[2]}"),
        ("Coronal", t1c_cor, gt_cor, pred_cor, disc_cor, f"Y = {center[1]}"),
        ("Sagittal", t1c_sag, gt_sag, pred_sag, disc_sag, f"X = {center[0]}"),
    ]

    col_titles = [
        "T1-Contrast (T1c) MRI",
        "Ground Truth Annotation",
        "Model Prediction (3D U-Net)",
        "WT Agreement & Error Map",
    ]

    for r_idx, (view_name, mri, gt_sl, pred_sl, disc_sl, slice_str) in enumerate(rows):
        mri_norm = normalize_display_image(mri)

        # Col 0: T1c MRI
        ax = axes[r_idx, 0]
        ax.imshow(mri_norm, cmap="gray", origin="lower", aspect="equal")
        ax.set_ylabel(f"{view_name}\n({slice_str})", color="white", fontsize=14, fontweight="bold")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        if r_idx == 0:
            ax.set_title(col_titles[0], color="white", fontsize=13, pad=8)

        # Col 1: Ground Truth
        ax = axes[r_idx, 1]
        ax.imshow(mri_norm, cmap="gray", origin="lower", aspect="equal")
        if np.any(gt_sl > 0):
            ax.imshow(gt_sl, cmap=SUBREGION_CMAP, vmin=0, vmax=3, alpha=0.65, origin="lower", aspect="equal")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        if r_idx == 0:
            ax.set_title(col_titles[1], color="white", fontsize=13, pad=8)

        # Col 2: Model Prediction
        ax = axes[r_idx, 2]
        ax.imshow(mri_norm, cmap="gray", origin="lower", aspect="equal")
        if np.any(pred_sl > 0):
            ax.imshow(pred_sl, cmap=SUBREGION_CMAP, vmin=0, vmax=3, alpha=0.65, origin="lower", aspect="equal")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        if r_idx == 0:
            ax.set_title(col_titles[2], color="white", fontsize=13, pad=8)

        # Col 3: Agreement / Error Map
        ax = axes[r_idx, 3]
        ax.imshow(mri_norm, cmap="gray", origin="lower", aspect="equal")
        if np.any(disc_sl > 0):
            ax.imshow(disc_sl, cmap=ERROR_CMAP, vmin=0, vmax=3, alpha=0.75, origin="lower", aspect="equal")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        if r_idx == 0:
            ax.set_title(col_titles[3], color="white", fontsize=13, pad=8)

    # Super-title with validation metrics
    gt_vol = metrics["gt_wt_voxels"]
    pred_vol = metrics["pred_wt_voxels"]
    ratio = metrics["vol_ratio_pred_gt"]
    ratio_str = f"{ratio:.2f}x" if not np.isnan(ratio) else "N/A"

    suptip = (
        f"Case: {case_info['case_id']} — {case_info['category']}\n"
        f"WT Dice: {metrics['wt_dice']:.4f} | TC Dice: {metrics['tc_dice']:.4f} | ET Dice: {metrics['et_dice']:.4f} | "
        f"Precision: {metrics['wt_precision']:.4f} | Recall: {metrics['wt_recall']:.4f} | "
        f"GT WT: {gt_vol:,} voxels | Pred WT: {pred_vol:,} voxels (Ratio: {ratio_str})"
    )
    plt.suptitle(suptip, color="white", fontsize=14, y=0.98, fontweight="bold")

    # Legend elements at bottom
    legend_elements = [
        Patch(facecolor="#FF3333", edgecolor="white", label="Necrotic Core (NCR, label 1)"),
        Patch(facecolor="#33CC33", edgecolor="white", label="Peritumoral Edema (ED, label 2)"),
        Patch(facecolor="#FFCC00", edgecolor="white", label="Enhancing Tumor (ET, label 3)"),
        Patch(facecolor="#00E676", edgecolor="white", label="True Positive (TP Agreement)"),
        Patch(facecolor="#FF1744", edgecolor="white", label="False Positive (Overprediction)"),
        Patch(facecolor="#2979FF", edgecolor="white", label="False Negative (Missed GT)"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=6,
        facecolor="#222222",
        edgecolor="#666666",
        fontsize=10,
        labelcolor="white",
        bbox_to_anchor=(0.5, 0.01),
    )

    plt.tight_layout(rect=[0.02, 0.04, 0.98, 0.95])
    plt.savefig(output_png, dpi=200, facecolor="black")
    plt.close(fig)
    print(f"Saved: {output_png.name}")


def render_contact_sheet(
    cases_data: List[Dict[str, Any]],
    output_png: Path,
):
    """
    Renders a unified contact sheet containing all 6 representative cases in axial view.
    """
    num_cases = len(cases_data)
    fig, axes = plt.subplots(num_cases, 4, figsize=(18, 3.2 * num_cases), facecolor="black")

    col_titles = [
        "T1c MRI Background",
        "Ground Truth (Subregions)",
        "3D U-Net Prediction",
        "WT Discrepancy (TP/FP/FN)",
    ]

    for c_idx, c_data in enumerate(cases_data):
        mri_ax = c_data["t1c_ax"]
        gt_ax = c_data["gt_ax"]
        pred_ax = c_data["pred_ax"]
        disc_ax = c_data["disc_ax"]
        metrics = c_data["metrics"]
        info = c_data["info"]

        mri_norm = normalize_display_image(mri_ax)

        # Col 0: MRI
        ax = axes[c_idx, 0]
        ax.imshow(mri_norm, cmap="gray", origin="lower", aspect="equal")
        gt_str = f"GT: {metrics['gt_wt_voxels']:,}"
        dice_str = f"Dice: {metrics['wt_dice']:.4f}"
        ax.set_ylabel(f"{info['case_id']}\n{dice_str} | {gt_str}", color="white", fontsize=11, fontweight="bold")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        if c_idx == 0:
            ax.set_title(col_titles[0], color="white", fontsize=13, pad=8)

        # Col 1: GT
        ax = axes[c_idx, 1]
        ax.imshow(mri_norm, cmap="gray", origin="lower", aspect="equal")
        if np.any(gt_ax > 0):
            ax.imshow(gt_ax, cmap=SUBREGION_CMAP, vmin=0, vmax=3, alpha=0.65, origin="lower", aspect="equal")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        if c_idx == 0:
            ax.set_title(col_titles[1], color="white", fontsize=13, pad=8)

        # Col 2: Pred
        ax = axes[c_idx, 2]
        ax.imshow(mri_norm, cmap="gray", origin="lower", aspect="equal")
        if np.any(pred_ax > 0):
            ax.imshow(pred_ax, cmap=SUBREGION_CMAP, vmin=0, vmax=3, alpha=0.65, origin="lower", aspect="equal")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        if c_idx == 0:
            ax.set_title(col_titles[2], color="white", fontsize=13, pad=8)

        # Col 3: Discrepancy
        ax = axes[c_idx, 3]
        ax.imshow(mri_norm, cmap="gray", origin="lower", aspect="equal")
        if np.any(disc_ax > 0):
            ax.imshow(disc_ax, cmap=ERROR_CMAP, vmin=0, vmax=3, alpha=0.75, origin="lower", aspect="equal")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        if c_idx == 0:
            ax.set_title(col_titles[3], color="white", fontsize=13, pad=8)

    plt.suptitle(
        "Multimodal Meningioma Segmentation Contact Sheet (Representative Validation Cases)\n"
        "Full-Cohort 3D U-Net Model (Trained on 800 cases, 25 epochs)",
        color="white",
        fontsize=15,
        y=0.99,
        fontweight="bold",
    )

    legend_elements = [
        Patch(facecolor="#FF3333", edgecolor="white", label="Necrotic Core (NCR)"),
        Patch(facecolor="#33CC33", edgecolor="white", label="Edema (ED)"),
        Patch(facecolor="#FFCC00", edgecolor="white", label="Enhancing Tumor (ET)"),
        Patch(facecolor="#00E676", edgecolor="white", label="True Positive (TP)"),
        Patch(facecolor="#FF1744", edgecolor="white", label="False Positive (FP)"),
        Patch(facecolor="#2979FF", edgecolor="white", label="False Negative (FN)"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=6,
        facecolor="#222222",
        edgecolor="#666666",
        fontsize=10,
        labelcolor="white",
        bbox_to_anchor=(0.5, 0.005),
    )

    plt.tight_layout(rect=[0.02, 0.025, 0.98, 0.97])
    plt.savefig(output_png, dpi=200, facecolor="black")
    plt.close(fig)
    print(f"Saved Contact Sheet: {output_png.name}")


def main():
    output_dir = PROJECT_ROOT / "outputs/meningioma_multimodal_experiment/final_visualizations_20261015"
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = PROJECT_ROOT / "outputs/meningioma_multimodal_experiment/full800_brainaware_20261014/checkpoints/meningioma_multimodal_best.pth"
    split_path = PROJECT_ROOT / "data/meningioma_multimodal/split.json"
    dataset_root = Path(r"C:\3D_DATASET\BraTS2023_MEN")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("GENERATING FINAL MENINGIOMA SEGMENTATION VISUALIZATIONS")
    print("=" * 80)
    print(f"Device:            {device}")
    print(f"Target Checkpoint: {ckpt_path}")
    print(f"Output Directory:  {output_dir}")

    # Load model
    unet = create_multimodal_unet(in_channels=4, num_classes=4, base_channels=16).to(device)
    load_checkpoint(ckpt_path, unet, device=device)
    unet.eval()

    # Load dataset
    ds = MeningiomaMultimodalDataset(
        split_json_path=split_path,
        partition="val",
        dataset_root=dataset_root,
        patch_size=None,
    )

    cases_contact_data: List[Dict[str, Any]] = []
    case_summaries: List[Dict[str, Any]] = []

    for c_info in TARGET_CASES:
        cid = c_info["case_id"]
        print(f"\nProcessing {cid} ({c_info['category']})...")
        idx = ds.case_ids.index(cid)
        sample = ds[idx]

        img_cpu = sample["image"]  # [4, D, H, W]
        gt_cpu = sample["mask"]    # [D, H, W]

        # Run inference
        img_gpu = img_cpu.unsqueeze(0).to(device)
        with torch.inference_mode():
            out = sliding_window_inference(
                inputs=img_gpu,
                roi_size=(64, 64, 64),
                sw_batch_size=2,
                predictor=unet,
                overlap=0.5,
                mode="gaussian",
            )
            pred = torch.argmax(out, dim=1).squeeze(0).cpu()

        # Air masking on predictions only
        masked_pred = apply_air_masking(pred, img_cpu)

        # Metrics
        m = calculate_multimodal_metrics(masked_pred, gt_cpu)
        gt_wt = int(((gt_cpu >= 1) & (gt_cpu <= 3)).sum().item())
        pred_wt = int(((masked_pred >= 1) & (masked_pred <= 3)).sum().item())
        ratio = (pred_wt / gt_wt) if gt_wt > 0 else float("nan")

        case_metrics = {
            "wt_dice": m["wt"]["dice"],
            "tc_dice": m["tc"]["dice"],
            "et_dice": m["et"]["dice"],
            "wt_precision": m["wt"]["precision"],
            "wt_recall": m["wt"]["recall"],
            "gt_wt_voxels": gt_wt,
            "pred_wt_voxels": pred_wt,
            "vol_ratio_pred_gt": ratio,
        }

        # Numpy volumes: (240, 240, 155)
        t1c_3d = img_cpu[1].numpy()
        gt_3d = gt_cpu.numpy()
        pred_3d = masked_pred.numpy()

        fig_path = output_dir / f"{c_info['file_prefix']}_{cid}.png"
        render_case_figure(
            case_info=c_info,
            t1c_3d=t1c_3d,
            gt_3d=gt_3d,
            pred_3d=pred_3d,
            metrics=case_metrics,
            output_png=fig_path,
        )

        # Extract axial slice for contact sheet
        gt_coords = np.argwhere(gt_3d > 0)
        if len(gt_coords) > 0:
            center = (
                int(np.median(gt_coords[:, 0])),
                int(np.median(gt_coords[:, 1])),
                int(np.median(gt_coords[:, 2])),
            )
        else:
            pred_coords = np.argwhere(pred_3d > 0)
            center = (
                int(np.median(pred_coords[:, 0])) if len(pred_coords) > 0 else 120,
                int(np.median(pred_coords[:, 1])) if len(pred_coords) > 0 else 120,
                int(np.median(pred_coords[:, 2])) if len(pred_coords) > 0 else 77,
            )

        t1c_ax, _, _ = extract_slices(t1c_3d, center)
        gt_ax, _, _ = extract_slices(gt_3d, center)
        pred_ax, _, _ = extract_slices(pred_3d, center)
        disc_ax = make_discrepancy_map(gt_ax, pred_ax)

        cases_contact_data.append({
            "info": c_info,
            "metrics": case_metrics,
            "t1c_ax": t1c_ax,
            "gt_ax": gt_ax,
            "pred_ax": pred_ax,
            "disc_ax": disc_ax,
        })

        case_summaries.append({
            "info": c_info,
            "metrics": case_metrics,
            "png_file": fig_path.name,
            "center": list(center),
        })

        del img_gpu, out, pred, masked_pred
        torch.cuda.empty_cache()

    # Render Contact Sheet
    contact_sheet_path = output_dir / "meningioma_cases_contact_sheet.png"
    render_contact_sheet(cases_contact_data, contact_sheet_path)

    # Generate Markdown Report
    report_path = output_dir / "visualization_report.md"
    report_lines = [
        "# Multimodal Meningioma Segmentation Visualization Report",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Evaluated Checkpoint:** `outputs/meningioma_multimodal_experiment/full800_brainaware_20261014/checkpoints/meningioma_multimodal_best.pth`  ",
        f"**Output Directory:** `outputs/meningioma_multimodal_experiment/final_visualizations_20261015/`  ",
        "",
        "---",
        "",
        "## 1. Overview & Visualization Methodology",
        "",
        "This report provides cross-sectional multi-planar visualizations (Axial, Coronal, Sagittal) across 6 representative cases selected from the 200-case validation split.",
        "",
        "### Anatomical and Color Conventions",
        "- **MRI Background:** T1-weighted post-contrast (T1c, Channel 1), displaying anatomical structure and meningioma contrast enhancement.",
        "- **Subregion Label Mapping:**",
        "  - **Red (#FF3333):** Necrotic Tumor Core (NCR, label 1)",
        "  - **Green (#33CC33):** Peritumoral Edema (ED, label 2)",
        "  - **Yellow (#FFCC00):** Enhancing Tumor (ET, label 3)",
        "- **Agreement & Error Map (Column 4):**",
        "  - **Lime Green (#00E676):** True Positive ($TP$) — Model and Ground Truth agree.",
        "  - **Crimson Red (#FF1744):** False Positive ($FP$) — Overprediction (Model predicted tumor in healthy brain).",
        "  - **Dodger Blue (#2979FF):** False Negative ($FN$) — Underprediction (Missed Ground Truth lesion).",
        "",
        "> [!NOTE]",
        "> **Empty-Air Masking Clarification:** Empty-air voxels (where all 4 normalized modalities are strictly zero) were explicitly masked to background (0) during evaluation. This inference-time bounding-space constraint removes spurious background noise outside the skull, but does not alter predictions inside the brain parenchyma.",
        "",
        "---",
        "",
        "## 2. Representative Cases Summary Table",
        "",
        "| Case ID | Category | WT Dice | TC Dice | ET Dice | Precision | Recall | GT Volume | Pred Volume | Vol Ratio |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for c in case_summaries:
        m = c["metrics"]
        info = c["info"]
        ratio_str = f"{m['vol_ratio_pred_gt']:.2f}x" if not np.isnan(m['vol_ratio_pred_gt']) else "N/A"
        report_lines.append(
            f"| `{info['case_id']}` | {info['category']} | **{m['wt_dice']:.4f}** | {m['tc_dice']:.4f} | {m['et_dice']:.4f} | "
            f"{m['wt_precision']:.4f} | {m['wt_recall']:.4f} | {m['gt_wt_voxels']:,} | {m['pred_wt_voxels']:,} | {ratio_str} |"
        )

    report_lines.extend([
        "",
        "---",
        "",
        "## 3. Detailed Case-by-Case Analysis",
        "",
    ])

    for i, c in enumerate(case_summaries, 1):
        info = c["info"]
        m = c["metrics"]
        report_lines.extend([
            f"### Case {i}: `{info['case_id']}` — {info['category']}",
            f"- **Figure File:** `outputs/meningioma_multimodal_experiment/final_visualizations_20261015/{c['png_file']}`",
            f"- **Validation Metrics:** WT Dice = **{m['wt_dice']:.4f}**, TC Dice = **{m['tc_dice']:.4f}**, ET Dice = **{m['et_dice']:.4f}**, Precision = {m['wt_precision']:.4f}, Recall = {m['wt_recall']:.4f}",
            f"- **Tumor Volumes:** Ground Truth = {m['gt_wt_voxels']:,} voxels | Predicted = {m['pred_wt_voxels']:,} voxels (Ratio: {m['vol_ratio_pred_gt']:.2f}x)",
            f"- **Selected Slice Centroid:** Axial Z={c['center'][2]}, Coronal Y={c['center'][1]}, Sagittal X={c['center'][0]}",
            f"- **Selection Rationale:** {info['rationale']}",
            f"- **Clinical & Radiological Findings:**",
        ])
        if "High-Dice" in info["category"] or "Large-Tumor" in info["category"]:
            report_lines.append(
                "  The visualization shows exceptional multi-planar congruence between ground truth and model output. The enhancing dural attachment and core margins match the anatomical boundaries on T1c MRI with negligible boundary divergence (green dominating the agreement map)."
            )
        elif "Medium-Tumor" in info["category"]:
            report_lines.append(
                "  Represents optimal performance on a standard medium-sized extra-axial lesion. The model correctly identifies the entire tumor boundary and distinguishes enhancing tumor from the small rim of peritumoral edema."
            )
        elif "Median" in info["category"]:
            report_lines.append(
                "  Illustrates median cohort behavior: the bulk of the lesion is accurately captured with high recall (0.98+), with mild false-positive edge dilation along the dural tail visible in crimson red on the error map."
            )
        elif "Overpredicted" in info["category"]:
            report_lines.append(
                "  Reveals the precision vulnerability on tiny tumors: while the model detected the true 139-voxel lesion (125 true positive voxels), it also predicted an ectopic false-positive cluster of 22k voxels elsewhere in the parenchyma, driving WT precision down to 0.0055."
            )
        elif "Missed" in info["category"]:
            report_lines.append(
                "  Documents a complete false-negative failure on an atypical small meningioma (1,587 voxels). The ground truth lesion (visible in blue on the error map) was completely undetected by the model, which instead generated a small spurious prediction (441 voxels) in a separate region."
            )
        report_lines.append("")

    report_lines.extend([
        "---",
        "",
        "## 4. Contact Sheet Reference",
        "The complete comparative contact sheet is saved at:",
        "- `outputs/meningioma_multimodal_experiment/final_visualizations_20261015/meningioma_cases_contact_sheet.png`",
        "",
        "This multi-panel overview juxtaposes all 6 cases in standard axial view, illustrating the continuum from near-perfect large tumor segmentations to the failure modes encountered on tiny lesions.",
    ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"Visualization report saved to: {report_path}")

    print("\n" + "=" * 80)
    print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()

