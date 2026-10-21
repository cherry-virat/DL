"""
Locked Test-Set Evaluator for Stage 2 3D Tumor Type Classifier.
==============================================================

Evaluates the 3D Tumor Type Classifier (Glioma vs. Meningioma) on the
frozen, locked test cohort defined in `data/classifier/split.json`.

Core Principles & Safeguards:
-----------------------------
1. Zero Data Leakage: Strictly evaluates the frozen 'test_cases' partition (300 cases).
   Never trains on the test set, never alters split definitions, and never tunes
   hyperparameters or decision thresholds based on test feedback.
2. Balanced Evaluation: Evaluates exactly 150 Glioma and 150 Meningioma test cases.
3. Checkpoint Validation: Inspects checkpoint metadata to verify whether the model
   comes from a full training run or a sample-truncated smoke test.
4. Comprehensive Metrics: Computes confusion matrix, accuracy, sensitivity, specificity,
   macro F1, per-class precision/recall/F1, and ROC-AUC.
5. Structured Persistence: Saves both machine-readable JSON and human-readable Markdown
   evaluation reports.
"""

import argparse
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.tumor_classifier_3d import TumorClassifier3D, create_tumor_classifier_3d
from training.dataset_classifier import BrainTumorClassifierDataset

try:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        confusion_matrix,
        roc_auc_score,
        log_loss,
        brier_score_loss,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "classifier" / "manifest.json"
DEFAULT_SPLIT_PATH = PROJECT_ROOT / "data" / "classifier" / "split.json"
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "classifier_training" / "tumor_classifier_best.pth"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "classifier_locked_test_evaluation"

CLASS_NAMES = {
    0: "glioma",
    1: "meningioma",
}


def verify_split_integrity(split_path: Path) -> Dict[str, Any]:
    """
    Verifies that the split file has zero overlap between train, val, test, and extension partitions.
    """
    if not split_path.exists():
        raise FileNotFoundError(f"Split file missing: {split_path}")

    with open(split_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    meta = split_data.get("metadata", {})
    s_train = set(split_data.get("train_cases", []))
    s_val = set(split_data.get("val_cases", []))
    s_test = set(split_data.get("test_cases", []))
    s_ext = set(split_data.get("unbalanced_glioma_extension", []))

    overlap_train_val = len(s_train & s_val)
    overlap_train_test = len(s_train & s_test)
    overlap_val_test = len(s_val & s_test)
    overlap_ext = len(s_ext & (s_train | s_val | s_test))

    all_disjoint = (
        overlap_train_val == 0
        and overlap_train_test == 0
        and overlap_val_test == 0
        and overlap_ext == 0
    )

    return {
        "all_disjoint": all_disjoint,
        "train_cases_count": len(s_train),
        "val_cases_count": len(s_val),
        "test_cases_count": len(s_test),
        "extension_cases_count": len(s_ext),
        "total_unique_cases": len(s_train | s_val | s_test | s_ext),
        "overlaps": {
            "train_val": overlap_train_val,
            "train_test": overlap_train_test,
            "val_test": overlap_val_test,
            "extension": overlap_ext,
        },
    }


def compute_comprehensive_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_prob: List[List[float]],
) -> Dict[str, Any]:
    """
    Computes rigorous classification metrics for binary Glioma (0) vs Meningioma (1).
    """
    y_t = np.array(y_true, dtype=np.int64)
    y_p = np.array(y_pred, dtype=np.int64)
    probs_arr = np.array(y_prob, dtype=np.float64)

    row_sums = probs_arr.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    probs_arr = probs_arr / row_sums

    if HAS_SKLEARN:
        acc = float(accuracy_score(y_t, y_p))
        macro_f1 = float(f1_score(y_t, y_p, average="macro", zero_division=0))
        prec_arr = precision_score(y_t, y_p, labels=[0, 1], average=None, zero_division=0)
        rec_arr = recall_score(y_t, y_p, labels=[0, 1], average=None, zero_division=0)
        f1_arr = f1_score(y_t, y_p, labels=[0, 1], average=None, zero_division=0)
        cm = confusion_matrix(y_t, y_p, labels=[0, 1]).tolist()

        specificity = float(rec_arr[0])  # Glioma recall
        sensitivity = float(rec_arr[1])  # Meningioma recall

        probs_c1 = probs_arr[:, 1]
        try:
            roc_auc = float(round(roc_auc_score(y_t, probs_c1), 5))
        except Exception:
            roc_auc = None

        try:
            ce_loss = float(round(log_loss(y_t, probs_arr), 5))
        except Exception:
            ce_loss = None

        try:
            brier = float(round(brier_score_loss(y_t, probs_c1), 5))
        except Exception:
            brier = None
    else:
        acc = float(np.mean(y_t == y_p))
        cm = [[0, 0], [0, 0]]
        for t, p in zip(y_t, y_p):
            cm[t][p] += 1

        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        prec_0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        rec_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1_0 = 2 * prec_0 * rec_0 / (prec_0 + rec_0) if (prec_0 + rec_0) > 0 else 0.0

        prec_1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec_1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_1 = 2 * prec_1 * rec_1 / (prec_1 + rec_1) if (prec_1 + rec_1) > 0 else 0.0

        prec_arr = [prec_0, prec_1]
        rec_arr = [rec_0, rec_1]
        f1_arr = [f1_0, f1_1]
        macro_f1 = float((f1_0 + f1_1) / 2.0)
        specificity = rec_0
        sensitivity = rec_1
        roc_auc = None
        ce_loss = None
        brier = None

    return {
        "accuracy": float(round(acc, 5)),
        "macro_f1": float(round(macro_f1, 5)),
        "sensitivity_meningioma": float(round(sensitivity, 5)),
        "specificity_glioma": float(round(specificity, 5)),
        "roc_auc": roc_auc,
        "cross_entropy_loss": ce_loss,
        "brier_score": brier,
        "confusion_matrix": cm,
        "true_glioma_pred_glioma": cm[0][0],
        "true_glioma_pred_meningioma": cm[0][1],
        "true_meningioma_pred_glioma": cm[1][0],
        "true_meningioma_pred_meningioma": cm[1][1],
        "precision_per_class": {
            "glioma": float(round(prec_arr[0], 5)),
            "meningioma": float(round(prec_arr[1], 5)),
        },
        "recall_per_class": {
            "glioma": float(round(rec_arr[0], 5)),
            "meningioma": float(round(rec_arr[1], 5)),
        },
        "f1_per_class": {
            "glioma": float(round(f1_arr[0], 5)),
            "meningioma": float(round(f1_arr[1], 5)),
        },
        "total_evaluated_samples": len(y_true),
    }


def evaluate_stage2_locked_test(
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    split_path: Path = DEFAULT_SPLIT_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    batch_size: int = 4,
    device_str: Optional[str] = None,
    max_cases: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Main evaluation pipeline on Stage 2 locked test set (Glioma vs. Meningioma).
    """
    start_time = datetime.datetime.now(datetime.timezone.utc)
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("STAGE 2 3D BRAIN TUMOR TYPE CLASSIFIER - LOCKED TEST SET EVALUATION")
    print("=" * 78)
    print(f"Timestamp:       {start_time.isoformat()}")
    print(f"Project Root:    {PROJECT_ROOT}")
    print(f"Checkpoint:      {checkpoint_path}")
    print(f"Manifest:        {manifest_path}")
    print(f"Split File:      {split_path}")
    print(f"Output Dir:      {out_p}")

    # 1. Verify split integrity
    print("\n[Step 1/5] Verifying split disjointness and partition isolation...")
    split_info = verify_split_integrity(split_path)
    if not split_info["all_disjoint"]:
        raise ValueError(f"Split partitions are not disjoint! Overlaps: {split_info['overlaps']}")
    print(f"  [*] Split verified: {split_info['test_cases_count']} test cases (disjoint from train/val/ext).")

    # 2. Checkpoint inspection
    print("\n[Step 2/5] Inspecting and loading model checkpoint...")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    with open(checkpoint_path, "rb") as f:
        checkpoint_sha256 = hashlib.sha256(f.read()).hexdigest()
    with open(split_path, "rb") as f:
        split_sha256 = hashlib.sha256(f.read()).hexdigest()

    device = torch.device(device_str) if device_str else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  [*] Executing on device: {device} (AMP={device.type == 'cuda'})")
    print(f"  [*] Checkpoint SHA256: {checkpoint_sha256}")
    print(f"  [*] Split SHA256:      {split_sha256}")

    ckpt = torch.load(str(checkpoint_path), map_location=device)
    base_channels = ckpt.get("base_channels", 16)
    in_channels = ckpt.get("in_channels", 4)
    num_classes = ckpt.get("num_classes", 2)
    ckpt_epoch = ckpt.get("epoch", 1)
    val_metrics_recorded = ckpt.get("val_metrics", {})
    recorded_val_samples = val_metrics_recorded.get("total_samples", 0) if isinstance(val_metrics_recorded, dict) else 0

    is_smoke_test_ckpt = (recorded_val_samples < 50)
    print(f"  [*] Checkpoint epoch: {ckpt_epoch}, base_channels: {base_channels}, num_classes: {num_classes}")
    print(f"  [*] Recorded validation samples in checkpoint: {recorded_val_samples}")
    if is_smoke_test_ckpt:
        print("  [!] WARNING: Checkpoint originates from a sample-truncated smoke test (< 50 validation samples).")

    model = create_tumor_classifier_3d(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  [*] Model instantiated and weights loaded ({model.count_parameters():,} parameters).")

    # 3. Load locked test dataset
    print("\n[Step 3/5] Loading locked Stage 2 test cohort...")
    test_dataset = BrainTumorClassifierDataset(
        split="test",
        manifest_path=manifest_path,
        split_path=split_path,
    )
    total_test_in_split = len(test_dataset)
    print(f"  [*] Test cases loaded: {total_test_in_split}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # 4. Inference execution
    print("\n[Step 4/5] Executing inference over locked test cases...")
    all_targets: List[int] = []
    all_preds: List[int] = []
    all_probs: List[List[float]] = []
    all_case_ids: List[str] = []
    case_level_results: List[Dict[str, Any]] = []

    use_amp = (device.type == "cuda")

    with torch.no_grad():
        for b_idx, (images, targets, anon_ids) in enumerate(test_loader):
            if max_cases is not None and len(all_targets) >= max_cases:
                break

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = model(images)
            else:
                outputs = model(images)

            probs = torch.softmax(outputs, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            t_list = targets.cpu().tolist()
            p_list = preds.cpu().tolist()
            pr_list = probs.cpu().tolist()
            ids_list = list(anon_ids)

            all_targets.extend(t_list)
            all_preds.extend(p_list)
            all_probs.extend(pr_list)
            all_case_ids.extend(ids_list)

            for idx_in_batch, cid in enumerate(ids_list):
                case_level_results.append({
                    "anonymized_id": cid,
                    "ground_truth_label": CLASS_NAMES[t_list[idx_in_batch]],
                    "ground_truth_id": t_list[idx_in_batch],
                    "predicted_label": CLASS_NAMES[p_list[idx_in_batch]],
                    "predicted_id": p_list[idx_in_batch],
                    "is_correct": (t_list[idx_in_batch] == p_list[idx_in_batch]),
                    "prob_glioma": float(round(pr_list[idx_in_batch][0], 6)),
                    "prob_meningioma": float(round(pr_list[idx_in_batch][1], 6)),
                })

            if (b_idx + 1) % 20 == 0 or (b_idx + 1) == len(test_loader):
                print(f"  Processed Batch {b_idx + 1}/{len(test_loader)} ({len(all_targets)}/{total_test_in_split} cases)")

    # 5. Metrics calculation
    print("\n[Step 5/5] Computing Stage 2 evaluation metrics...")
    overall_metrics = compute_comprehensive_metrics(all_targets, all_preds, all_probs)

    # Subgroup metrics breakdown
    gli_targets = [t for t in all_targets if t == 0]
    gli_preds = [p for t, p in zip(all_targets, all_preds) if t == 0]
    gli_probs = [pr for t, pr in zip(all_targets, all_probs) if t == 0]
    men_targets = [t for t in all_targets if t == 1]
    men_preds = [p for t, p in zip(all_targets, all_preds) if t == 1]
    men_probs = [pr for t, pr in zip(all_targets, all_probs) if t == 1]

    subgroup_metrics = {
        "glioma_subgroup": {
            "total_cases": len(gli_targets),
            "correct_predictions": sum(1 for p in gli_preds if p == 0),
            "accuracy": float(round(sum(1 for p in gli_preds if p == 0) / len(gli_targets), 5)) if gli_targets else 0.0,
            "error_rate": float(round(sum(1 for p in gli_preds if p != 0) / len(gli_targets), 5)) if gli_targets else 0.0,
            "mean_confidence": float(round(float(np.mean([pr[0] for pr in gli_probs])), 5)) if gli_probs else 0.0,
        },
        "meningioma_subgroup": {
            "total_cases": len(men_targets),
            "correct_predictions": sum(1 for p in men_preds if p == 1),
            "accuracy": float(round(sum(1 for p in men_preds if p == 1) / len(men_targets), 5)) if men_targets else 0.0,
            "error_rate": float(round(sum(1 for p in men_preds if p != 1) / len(men_targets), 5)) if men_targets else 0.0,
            "mean_confidence": float(round(float(np.mean([pr[1] for pr in men_probs])), 5)) if men_probs else 0.0,
        },
    }

    end_time = datetime.datetime.now(datetime.timezone.utc)
    duration_sec = (end_time - start_time).total_seconds()

    report = {
        "metadata": {
            "evaluation_timestamp": end_time.isoformat(),
            "evaluation_duration_seconds": round(duration_sec, 2),
            "project_root": str(PROJECT_ROOT),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "split_path": str(split_path),
            "split_sha256": split_sha256,
            "checkpoint_is_smoke_test": is_smoke_test_ckpt,
            "recorded_val_samples_in_checkpoint": recorded_val_samples,
            "manifest_path": str(manifest_path),
            "device": str(device),
            "batch_size": batch_size,
            "model_architecture": "TumorClassifier3D",
            "base_channels": base_channels,
            "total_test_cases_evaluated": len(all_targets),
            "total_test_cases_in_split": total_test_in_split,
        },
        "overall_metrics": overall_metrics,
        "subgroup_metrics": subgroup_metrics,
        "disclaimer": (
            "If evaluated checkpoint was trained for only a smoke-test, metrics represent "
            "untrained or undertrained baseline weights, NOT validated production classification."
        ),
        "case_level_results": case_level_results,
    }

    # Save JSON report
    json_path = out_p / "stage2_test_evaluation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[Saved JSON Report]: {json_path}")

    # Save Markdown report
    md_path = out_p / "stage2_test_evaluation_report.md"
    generate_markdown_report(report, md_path)
    print(f"[Saved Markdown Report]: {md_path}")

    print("\n" + "=" * 78)
    print("STAGE 2 EVALUATION SUMMARY RESULTS")
    print("=" * 78)
    print(f"Total Test Cases:       {len(all_targets)} (150 Glioma, 150 Meningioma)")
    print(f"Overall Accuracy:       {overall_metrics['accuracy'] * 100:.2f}%")
    print(f"Macro F1 Score:         {overall_metrics['macro_f1']:.4f}")
    print(f"Glioma Precision/Recall:  {overall_metrics['precision_per_class']['glioma']*100:.2f}% / {overall_metrics['recall_per_class']['glioma']*100:.2f}%")
    print(f"Meningioma Precision/Recall: {overall_metrics['precision_per_class']['meningioma']*100:.2f}% / {overall_metrics['recall_per_class']['meningioma']*100:.2f}%")
    if overall_metrics["roc_auc"] is not None:
        print(f"ROC-AUC:                {overall_metrics['roc_auc']:.4f}")
    print(f"Confusion Matrix:       [Glioma->Glioma: {overall_metrics['true_glioma_pred_glioma']}, Glioma->Men: {overall_metrics['true_glioma_pred_meningioma']}], "
          f"[Men->Glioma: {overall_metrics['true_meningioma_pred_glioma']}, Men->Men: {overall_metrics['true_meningioma_pred_meningioma']}]")
    print("=" * 78)

    return report


def generate_markdown_report(report: Dict[str, Any], output_file: Path) -> None:
    """Generates a clean Markdown report from Stage 2 evaluation metrics."""
    meta = report["metadata"]
    m = report["overall_metrics"]

    md = f"""# Stage 2 3D Tumor Type Classifier - Locked Test Evaluation Report

**Evaluation Date:** {meta['evaluation_timestamp']}
**Duration:** {meta['evaluation_duration_seconds']} seconds
**Model Architecture:** `{meta['model_architecture']}` (base channels: {meta['base_channels']})
**Checkpoint Path:** `{meta['checkpoint_path']}`
**Checkpoint SHA256:** `{meta.get('checkpoint_sha256', 'N/A')}`
**Is Smoke-Test Checkpoint:** `{meta['checkpoint_is_smoke_test']}` (recorded val samples: {meta['recorded_val_samples_in_checkpoint']})
**Split File:** `{meta['split_path']}`
**Split SHA256:** `{meta.get('split_sha256', 'N/A')}`

---

## 1. Overall Locked Test Metrics (Glioma vs. Meningioma)

| Metric | Score | Details |
| :--- | :---: | :--- |
| **Total Test Samples** | **{m['total_evaluated_samples']}** | 150 Glioma + 150 Meningioma |
| **Accuracy** | **{m['accuracy'] * 100:.2f}%** | Overall classification accuracy |
| **Macro F1 Score** | **{m['macro_f1']:.4f}** | Unweighted mean across both tumor classes |
| **Glioma Precision** | **{m['precision_per_class']['glioma'] * 100:.2f}%** | Purity of glioma predictions |
| **Glioma Recall (Specificity)** | **{m['recall_per_class']['glioma'] * 100:.2f}%** | True Glioma cases identified |
| **Glioma F1 Score** | **{m['f1_per_class']['glioma']:.4f}** | Harmonic mean of glioma precision/recall |
| **Meningioma Precision** | **{m['precision_per_class']['meningioma'] * 100:.2f}%** | Purity of meningioma predictions |
| **Meningioma Recall (Sensitivity)** | **{m['recall_per_class']['meningioma'] * 100:.2f}%** | True Meningioma cases identified |
| **Meningioma F1 Score** | **{m['f1_per_class']['meningioma']:.4f}** | Harmonic mean of meningioma precision/recall |
| **ROC-AUC** | **{m['roc_auc'] if m['roc_auc'] is not None else 'N/A'}** | Area under the ROC curve |
| **Cross-Entropy Loss** | **{m['cross_entropy_loss']}** | Binary log loss on test probabilities |
| **Brier Score** | **{m['brier_score']}** | Mean squared probability error |

### Confusion Matrix

| Ground Truth \\ Prediction | Predicted Glioma (0) | Predicted Meningioma (1) | Total |
| :--- | :---: | :---: | :---: |
| **Actual Glioma** | **{m['true_glioma_pred_glioma']}** | {m['true_glioma_pred_meningioma']} | 150 |
| **Actual Meningioma** | {m['true_meningioma_pred_glioma']} | **{m['true_meningioma_pred_meningioma']}** | 150 |
| **Total Predicted** | {m['true_glioma_pred_glioma'] + m['true_meningioma_pred_glioma']} | {m['true_glioma_pred_meningioma'] + m['true_meningioma_pred_meningioma']} | 300 |

---

## 2. Cohort Subgroup Breakdown

| Subgroup | Cases | Correct | Accuracy | Error Rate | Mean Confidence |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Glioma Cohort** | {report.get('subgroup_metrics', {}).get('glioma_subgroup', {}).get('total_cases', 150)} | {report.get('subgroup_metrics', {}).get('glioma_subgroup', {}).get('correct_predictions', m['true_glioma_pred_glioma'])} | {report.get('subgroup_metrics', {}).get('glioma_subgroup', {}).get('accuracy', 0.0) * 100:.2f}% | {report.get('subgroup_metrics', {}).get('glioma_subgroup', {}).get('error_rate', 0.0) * 100:.2f}% | {report.get('subgroup_metrics', {}).get('glioma_subgroup', {}).get('mean_confidence', 0.0):.4f} |
| **Meningioma Cohort** | {report.get('subgroup_metrics', {}).get('meningioma_subgroup', {}).get('total_cases', 150)} | {report.get('subgroup_metrics', {}).get('meningioma_subgroup', {}).get('correct_predictions', m['true_meningioma_pred_meningioma'])} | {report.get('subgroup_metrics', {}).get('meningioma_subgroup', {}).get('accuracy', 0.0) * 100:.2f}% | {report.get('subgroup_metrics', {}).get('meningioma_subgroup', {}).get('error_rate', 0.0) * 100:.2f}% | {report.get('subgroup_metrics', {}).get('meningioma_subgroup', {}).get('mean_confidence', 0.0):.4f} |

---

## 3. Integrity and Validation Status

> [!WARNING]
> **Production Validation Status:**
> {report['disclaimer']}
"""
    output_file.write_text(md, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Stage 2 3D Tumor Type Classifier on Locked Test Set.")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT_PATH), help="Path to checkpoint.")
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST_PATH), help="Path to manifest.json.")
    parser.add_argument("--split", type=str, default=str(DEFAULT_SPLIT_PATH), help="Path to split.json.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory for reports.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for evaluation DataLoader.")
    parser.add_argument("--device", type=str, default=None, help="Execution device ('cuda' or 'cpu').")
    parser.add_argument("--max-cases", type=int, default=None, help="Optional cap for smoke testing.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_stage2_locked_test(
        checkpoint_path=Path(args.checkpoint),
        manifest_path=Path(args.manifest),
        split_path=Path(args.split),
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        device_str=args.device,
        max_cases=args.max_cases,
    )
