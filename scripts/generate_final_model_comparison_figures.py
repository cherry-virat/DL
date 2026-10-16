"""
generate_final_model_comparison_figures.py
==========================================
Generates comprehensive comparative plots, statistical figures, and selected-case
multi-planar visualizations for both the Glioma and Multimodal Meningioma
4-channel 3D U-Net models.

Outputs saved to:
    outputs/final_model_comparison_20261016/
"""

import json
import shutil
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet

from preprocessing.glioma_dataset import DATASET_ROOT as GLIOMA_DATASET_ROOT, pad_to_patch_size
from preprocessing.glioma_preprocess import process_case as process_glioma_case
from segmentation.glioma_formal_evaluation import (
    filter_connected_components,
    generate_orthogonal_visualization as generate_glioma_orthogonal_vis,
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "final_model_comparison_20261016"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GLIOMA_CSV = PROJECT_ROOT / "outputs" / "glioma_formal_evaluation" / "per_case_metrics.csv"
GLIOMA_JSON = PROJECT_ROOT / "outputs" / "glioma_formal_evaluation" / "aggregate_metrics.json"
GLIOMA_VIS_DIR = PROJECT_ROOT / "outputs" / "glioma_formal_evaluation" / "visualizations"
GLIOMA_CKPT = PROJECT_ROOT / "outputs" / "glioma_training" / "glioma_best_medium_experiment.pth"

MENINGIOMA_CSV = PROJECT_ROOT / "outputs" / "meningioma_multimodal_experiment" / "final_validation_20261015" / "full200_per_case_metrics.csv"
MENINGIOMA_JSON = PROJECT_ROOT / "outputs" / "meningioma_multimodal_experiment" / "final_validation_20261015" / "full200_validation_metrics.json"
MENINGIOMA_VIS_DIR = PROJECT_ROOT / "outputs" / "meningioma_multimodal_experiment" / "final_visualizations_20261015"


def generate_statistical_comparison_plots():
    print("\n--- Generating Statistical Comparison Plots ---")
    df_gli = pd.read_csv(GLIOMA_CSV)
    df_men = pd.read_csv(MENINGIOMA_CSV)

    # 1. Dice Distribution Comparison (Boxplots + Jitter)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    plt.suptitle("4-Channel 3D U-Net: Subregion Dice Score Distributions", fontsize=15, fontweight="bold")

    # Glioma (20 cases)
    gli_data = [df_gli["WT_Dice"], df_gli["TC_Dice"], df_gli["ET_Dice"]]
    bp1 = axes[0].boxplot(gli_data, tick_labels=["Whole Tumor (WT)", "Tumor Core (TC)", "Enhancing (ET)"], patch_artist=True)
    colors1 = ["#4CAF50", "#2196F3", "#FF9800"]
    for patch, color in zip(bp1["boxes"], colors1):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for i, col in enumerate(["WT_Dice", "TC_Dice", "ET_Dice"]):
        y = df_gli[col]
        x = np.random.normal(i + 1, 0.04, size=len(y))
        axes[0].plot(x, y, "o", color=colors1[i], alpha=0.8, markersize=6)
    axes[0].set_title(f"Glioma (N={len(df_gli)} Validation Cases)\nMacro WT Mean: 84.78%, Median: 90.13%", fontsize=12)
    axes[0].set_ylabel("Dice Similarity Coefficient", fontsize=12)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Meningioma (200 cases)
    men_data = [df_men["wt_dice"], df_men["tc_dice"], df_men["et_dice"]]
    bp2 = axes[1].boxplot(men_data, tick_labels=["Whole Tumor (WT)", "Tumor Core (TC)", "Enhancing (ET)"], patch_artist=True)
    colors2 = ["#4CAF50", "#2196F3", "#FF9800"]
    for patch, color in zip(bp2["boxes"], colors2):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for i, col in enumerate(["wt_dice", "tc_dice", "et_dice"]):
        y = df_men[col]
        x = np.random.normal(i + 1, 0.06, size=len(y))
        axes[1].plot(x, y, "o", color=colors2[i], alpha=0.3, markersize=4)
    axes[1].set_title(f"Meningioma (N={len(df_men)} Holdout Cases)\nMacro WT Mean: 79.91%, Median: 90.06%, Pooled: 92.73%", fontsize=12)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot1_path = OUTPUT_DIR / "dice_distribution_comparison.png"
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"Saved: {plot1_path}")

    # 2. Volume vs. Dice Correlation Plot (Log Scale)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    plt.suptitle("Tumor Volume vs. WT Dice: Concentration of Errors in Small Tumors", fontsize=15, fontweight="bold")

    # Glioma
    axes[0].scatter(df_gli["GT_WT_Volume_mm3"], df_gli["WT_Dice"], color="#1565C0", s=60, alpha=0.85, edgecolors="black", linewidths=0.5)
    axes[0].set_xscale("log")
    axes[0].axvline(5000, color="red", linestyle="--", linewidth=1.5, label="Small Tumor Threshold (5,000 voxels)")
    axes[0].set_title(f"Glioma (N={len(df_gli)} Cases)\nFew small cases, stable Dice above 10k voxels", fontsize=12)
    axes[0].set_xlabel("Ground Truth Volume (mm³ or voxels, log scale)", fontsize=11)
    axes[0].set_ylabel("Whole Tumor (WT) Dice", fontsize=11)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="lower right")

    # Meningioma
    axes[1].scatter(df_men["gt_wt_voxels"], df_men["wt_dice"], color="#2E7D32", s=35, alpha=0.6, edgecolors="black", linewidths=0.3)
    axes[1].set_xscale("log")
    axes[1].axvline(5000, color="red", linestyle="--", linewidth=1.5, label="Small Tumor Threshold (<5,000 voxels: 75 cases)")
    axes[1].axhline(0.90, color="gray", linestyle=":", linewidth=1.0, label="90% Dice Benchmark")
    axes[1].set_title(f"Meningioma (N={len(df_men)} Cases)\nClear transition: >5k voxels (mean 88-95%), <5k voxels drops to 60.5%", fontsize=12)
    axes[1].set_xlabel("Ground Truth Volume (voxels, log scale)", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plot2_path = OUTPUT_DIR / "volume_vs_dice_correlation.png"
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"Saved: {plot2_path}")

    # 3. Macro vs Median vs Pooled Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = ["Glioma\nWhole Tumor", "Glioma\nTumor Core", "Glioma\nEnhancing Tumor", 
                   "Meningioma\nWhole Tumor", "Meningioma\nTumor Core", "Meningioma\nEnhancing Tumor"]
    macro_means = [84.78, 72.40, 70.72, 79.91, 81.41, 82.08]
    medians = [90.13, 78.71, 78.74, 90.06, 89.79, 91.08]
    pooled = [90.42, 76.66, 84.01, 92.73, 91.38, 92.77]

    x = np.arange(len(categories))
    width = 0.25

    rects1 = ax.bar(x - width, macro_means, width, label="Macro Mean Dice (%)", color="#90CAF9")
    rects2 = ax.bar(x, medians, width, label="Median Dice (%)", color="#1E88E5")
    rects3 = ax.bar(x + width, pooled, width, label="Pooled / Micro Dice (%)", color="#0D47A1")

    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Comparison of Macro Mean, Median, and Pooled/Micro Dice Scores", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")

    # Add data labels
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plot3_path = OUTPUT_DIR / "subregion_metrics_comparison.png"
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    print(f"Saved: {plot3_path}")


def generate_glioma_small_case_visualization():
    """Generates orthogonal visualization for BraTS-GLI-01341-000 (smallest glioma case: 3,340 mm3)."""
    case_id = "BraTS-GLI-01341-000"
    out_img = OUTPUT_DIR / f"glioma_case_small_tumor_{case_id}.png"
    if out_img.exists():
        print(f"Already exists: {out_img}")
        return

    print(f"\n--- Generating Glioma Small Case Visualization ({case_id}) ---")
    case_dir = GLIOMA_DATASET_ROOT / case_id
    if not case_dir.exists():
        cand = GLIOMA_DATASET_ROOT / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData" / case_id
        if cand.exists():
            case_dir = cand

    if not case_dir.exists():
        print(f"Warning: Could not locate directory for {case_id}")
        return

    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=0.0,
    ).to(device)

    ckpt = torch.load(GLIOMA_CKPT, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    image, mask = process_glioma_case(case_dir)
    image = np.asarray(image, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.int64)

    original_shape = tuple(image.shape[1:])
    d, h, w = original_shape

    mask_4d = np.expand_dims(mask, axis=0)
    padded_image, _ = pad_to_patch_size(image, mask_4d, (64, 64, 64))

    image_tensor = torch.from_numpy(padded_image).float().unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = sliding_window_inference(
            inputs=image_tensor,
            roi_size=(64, 64, 64),
            sw_batch_size=1,
            predictor=model,
            overlap=0.50,
            mode="gaussian",
        )
        probabilities = torch.softmax(logits, dim=1)
        wt_prob = probabilities[:, 1:, ...].sum(dim=1)[0].detach().cpu().numpy()[:d, :h, :w]

    raw_wt_mask = (wt_prob >= 0.60)
    final_wt_mask, _ = filter_connected_components(raw_wt_mask, min_size=1000, connectivity=1)

    gt_wt = (mask[:d, :h, :w] > 0)
    dice_val = (2.0 * np.logical_and(final_wt_mask, gt_wt).sum()) / (final_wt_mask.sum() + gt_wt.sum())

    generate_glioma_orthogonal_vis(case_id, image, mask[:d, :h, :w], final_wt_mask, out_img, dice_val)
    print(f"Saved: {out_img}")


def copy_selected_visualizations():
    print("\n--- Organizing Selected Case Visualizations ---")
    # Glioma existing
    gli_copies = {
        GLIOMA_VIS_DIR / "best_BraTS-GLI-01381-000_orthogonal_visualization.png": OUTPUT_DIR / "glioma_case_successful_BraTS-GLI-01381-000.png",
        GLIOMA_VIS_DIR / "median_BraTS-GLI-01397-000_orthogonal_visualization.png": OUTPUT_DIR / "glioma_case_median_BraTS-GLI-01397-000.png",
        GLIOMA_VIS_DIR / "worst_BraTS-GLI-01308-000_orthogonal_visualization.png": OUTPUT_DIR / "glioma_case_poor_BraTS-GLI-01308-000.png",
    }
    for src, dst in gli_copies.items():
        if src.exists():
            shutil.copy2(src, dst)
            print(f"Copied: {dst.name}")

    # Meningioma existing
    men_copies = {
        MENINGIOMA_VIS_DIR / "case_01_high_dice_BraTS-MEN-01367-000.png": OUTPUT_DIR / "meningioma_case_successful_BraTS-MEN-01367-000.png",
        MENINGIOMA_VIS_DIR / "case_04_median_dice_BraTS-MEN-00232-000.png": OUTPUT_DIR / "meningioma_case_median_BraTS-MEN-00232-000.png",
        MENINGIOMA_VIS_DIR / "case_05_small_overpredicted_BraTS-MEN-00094-000.png": OUTPUT_DIR / "meningioma_case_small_overpredicted_BraTS-MEN-00094-000.png",
        MENINGIOMA_VIS_DIR / "case_06_small_missed_BraTS-MEN-00690-000.png": OUTPUT_DIR / "meningioma_case_small_missed_BraTS-MEN-00690-000.png",
        MENINGIOMA_VIS_DIR / "case_02_large_tumor_BraTS-MEN-00977-000.png": OUTPUT_DIR / "meningioma_case_large_BraTS-MEN-00977-000.png",
        MENINGIOMA_VIS_DIR / "case_03_medium_tumor_BraTS-MEN-01179-000.png": OUTPUT_DIR / "meningioma_case_medium_BraTS-MEN-01179-000.png",
        MENINGIOMA_VIS_DIR / "meningioma_cases_contact_sheet.png": OUTPUT_DIR / "meningioma_cases_contact_sheet.png",
    }
    for src, dst in men_copies.items():
        if src.exists():
            shutil.copy2(src, dst)
            print(f"Copied: {dst.name}")


def main():
    print("=" * 70)
    print("GENERATING FINAL COMPARATIVE ARTIFACTS AND VISUALIZATIONS")
    print("=" * 70)
    generate_statistical_comparison_plots()
    generate_glioma_small_case_visualization()
    copy_selected_visualizations()
    print("\nAll comparative figures generated successfully.")


if __name__ == "__main__":
    main()
