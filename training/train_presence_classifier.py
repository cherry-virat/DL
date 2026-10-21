"""
Training and Validation Engine for 3D Brain Tumor Presence Classifier (Stage 1).
================================================================================

Trains the lightweight 3D ResNet classifier to discriminate between:
  - Class 0: Healthy Control (Normative structural MRI: IXI)
  - Class 1: Tumor Present (Pathological structural MRI: BraTS GLI & MEN)

Design Principles:
------------------
1. Disjoint Partitioning & Strict Test Isolation: Strictly uses pre-defined 'train'
   and 'val' splits from `data/tumor_presence/split.json`. The locked 'test' split
   is strictly isolated and NEVER accessed during training, checkpoint selection, or early stopping.
2. Robust QC Enforcement: Integrates `filter_non_qc=True` on dataset instantiation,
   ensuring no unverified, pending, flagged, or excluded scans enter training or validation.
3. Class Imbalance Mitigation: Calculates dynamic inverse-frequency class weights
   for CrossEntropyLoss to counterbalance the ~3.8:1 tumor-to-healthy ratio.
4. Comprehensive Metrics Tracking: Computes Loss, Accuracy, Macro F1, Sensitivity (Tumor Recall),
   Specificity (Healthy Recall), Per-class Precision/Recall/F1, Confusion Matrix, and ROC-AUC.
5. Dedicated Output Directory: Saves checkpoints and logs strictly to
   `outputs/classifier_training/`, with zero risk of overwriting segmentation checkpoints.
6. Safe Hardware Scaling: Designed for 6 GB RTX 3050 Laptop GPU using automatic mixed
   precision (AMP) with dynamic GradScaler.
7. Non-executable on Import: Protected by a `if __name__ == "__main__":` entrypoint.
"""

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.tumor_presence_classifier_3d import TumorPresenceClassifier3D, create_tumor_presence_classifier_3d
from training.dataset_tumor_presence import TumorPresenceDataset

# Try importing sklearn metrics with fallback to pure numpy
try:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        confusion_matrix,
        roc_auc_score,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "tumor_presence" / "manifest.json"
DEFAULT_SPLIT_PATH = PROJECT_ROOT / "data" / "tumor_presence" / "split.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "classifier_training"

CLASS_NAMES = {
    0: "healthy_control",
    1: "tumor_present",
}


def set_seed(seed: int = 42) -> None:
    """Sets deterministic random seeds across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_presence_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_prob: Optional[List[List[float]]] = None,
) -> Dict[str, Any]:
    """
    Computes standard classification evaluation metrics for tumor presence detection.

    Parameters:
        y_true: Ground truth integer class labels (0: Healthy, 1: Tumor).
        y_pred: Predicted integer class labels (0 or 1).
        y_prob: Optional predicted class probability distributions.

    Returns:
        Dictionary containing accuracy, macro_f1, sensitivity, specificity,
        per-class metrics, confusion matrix, and ROC-AUC (when computable).
    """
    if len(y_true) == 0:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
            "roc_auc": None,
            "precision_per_class": {"healthy_control": 0.0, "tumor_present": 0.0},
            "recall_per_class": {"healthy_control": 0.0, "tumor_present": 0.0},
            "f1_per_class": {"healthy_control": 0.0, "tumor_present": 0.0},
            "confusion_matrix": [[0, 0], [0, 0]],
            "total_samples": 0,
        }

    y_t = np.array(y_true, dtype=np.int64)
    y_p = np.array(y_pred, dtype=np.int64)

    if HAS_SKLEARN:
        acc = float(accuracy_score(y_t, y_p))
        macro_f1 = float(f1_score(y_t, y_p, average="macro", zero_division=0))
        prec_arr = precision_score(y_t, y_p, labels=[0, 1], average=None, zero_division=0)
        rec_arr = recall_score(y_t, y_p, labels=[0, 1], average=None, zero_division=0)
        f1_arr = f1_score(y_t, y_p, labels=[0, 1], average=None, zero_division=0)
        cm = confusion_matrix(y_t, y_p, labels=[0, 1]).tolist()

        # Specificity (Class 0 recall) and Sensitivity (Class 1 recall)
        specificity = float(rec_arr[0])
        sensitivity = float(rec_arr[1])

        # ROC-AUC if probabilities are available and both classes exist in y_true
        roc_auc = None
        if y_prob is not None and len(np.unique(y_t)) > 1:
            try:
                probs_class1 = np.array(y_prob)[:, 1]
                roc_auc = float(round(roc_auc_score(y_t, probs_class1), 4))
            except Exception:
                roc_auc = None
    else:
        # Pure numpy fallback
        acc = float((y_t == y_p).mean())
        cm = [[0, 0], [0, 0]]
        for t, p in zip(y_t, y_p):
            if t in (0, 1) and p in (0, 1):
                cm[t][p] += 1

        prec_arr, rec_arr, f1_arr = [], [], []
        for c in (0, 1):
            tp = cm[c][c]
            fp = sum(cm[other][c] for other in (0, 1) if other != c)
            fn = sum(cm[c][other] for other in (0, 1) if other != c)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            prec_arr.append(prec)
            rec_arr.append(rec)
            f1_arr.append(f1)
        macro_f1 = float(np.mean(f1_arr))
        specificity = float(rec_arr[0])
        sensitivity = float(rec_arr[1])
        roc_auc = None

    return {
        "accuracy": float(round(acc, 4)),
        "macro_f1": float(round(macro_f1, 4)),
        "sensitivity": float(round(sensitivity, 4)),
        "specificity": float(round(specificity, 4)),
        "roc_auc": roc_auc,
        "precision_per_class": {
            "healthy_control": float(round(prec_arr[0], 4)),
            "tumor_present": float(round(prec_arr[1], 4)),
        },
        "recall_per_class": {
            "healthy_control": float(round(rec_arr[0], 4)),
            "tumor_present": float(round(rec_arr[1], 4)),
        },
        "f1_per_class": {
            "healthy_control": float(round(f1_arr[0], 4)),
            "tumor_present": float(round(f1_arr[1], 4)),
        },
        "confusion_matrix": cm,
        "total_samples": len(y_true),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    use_amp: bool = True,
    max_batches: Optional[int] = None,
) -> Tuple[float, float]:
    """Executes one epoch of presence model training."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (images, targets, _) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        if use_amp and device.type == "cuda" and scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        running_loss += float(loss.item()) * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += int((preds == targets).sum().item())
        total += int(images.size(0))

        if (i + 1) % 100 == 0 or (i + 1) == len(loader):
            print(f"  Batch {i+1}/{len(loader)} | Running Loss: {running_loss/total:.4f} | Running Acc: {correct/total:.3f}", flush=True)

    avg_loss = (running_loss / total) if total > 0 else 0.0
    accuracy = (correct / total) if total > 0 else 0.0
    return avg_loss, accuracy


def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
    max_batches: Optional[int] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Evaluates the presence model on validation data without tracking gradients."""
    model.eval()
    running_loss = 0.0
    total = 0

    all_targets: List[int] = []
    all_preds: List[int] = []
    all_probs: List[List[float]] = []

    with torch.no_grad():
        for i, (images, targets, _) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if use_amp and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)

            running_loss += float(loss.item()) * images.size(0)
            total += int(images.size(0))

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_targets.extend(targets.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

            if (i + 1) % 50 == 0 or (i + 1) == len(loader):
                print(f"  Val Batch {i+1}/{len(loader)}", flush=True)

    avg_loss = (running_loss / total) if total > 0 else 0.0
    metrics = compute_presence_metrics(all_targets, all_preds, all_probs)
    return avg_loss, metrics


def compute_class_weights(dataset: TumorPresenceDataset) -> torch.Tensor:
    """
    Computes inverse frequency loss weights for healthy (0) and tumor (1) classes.
    """
    labels = [int(rec["class_id"]) for rec in dataset.case_records]
    n_healthy = max(1, labels.count(0))
    n_tumor = max(1, labels.count(1))

    # Weight healthy class proportionally to balance gradient contributions
    healthy_weight = float(n_tumor / n_healthy)
    tumor_weight = 1.0
    return torch.tensor([healthy_weight, tumor_weight], dtype=torch.float32)


def run_presence_training(
    manifest_path: Union[str, Path] = DEFAULT_MANIFEST_PATH,
    split_path: Union[str, Path] = DEFAULT_SPLIT_PATH,
    output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
    epochs: int = 20,
    batch_size: int = 2,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    base_channels: int = 16,
    dropout_rate: float = 0.2,
    patience: int = 7,
    seed: int = 42,
    device_str: Optional[str] = None,
    num_workers: int = 0,
    smoke_test: bool = False,
    resume_checkpoint: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Main training workflow for the 3D Tumor Presence Classifier (Stage 1).
    """
    set_seed(seed)

    manifest_p = Path(manifest_path)
    split_p = Path(split_path)
    out_dir = Path(output_dir)

    if not manifest_p.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_p}")
    if not split_p.exists():
        raise FileNotFoundError(f"Split file not found: {split_p}")

    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # Load datasets with strict QC filtering
    train_dataset = TumorPresenceDataset(
        split="train",
        manifest_path=manifest_p,
        split_path=split_p,
        filter_non_qc=True,
    )
    val_dataset = TumorPresenceDataset(
        split="val",
        manifest_path=manifest_p,
        split_path=split_p,
        filter_non_qc=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Instantiate Stage 1 model (2 input channels: T1, T2)
    model = create_tumor_presence_classifier_3d(
        in_channels=2,
        num_classes=2,
        base_channels=base_channels,
        dropout_rate=dropout_rate,
    ).to(device)

    # Class-weighted loss to counter class imbalance
    class_weights = compute_class_weights(train_dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    start_epoch = 1
    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0

    history: Dict[str, List] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "val_sensitivity": [],
        "val_specificity": [],
        "val_roc_auc": [],
        "lr": [],
    }

    # Handle resumption if requested
    if resume_checkpoint is not None and Path(resume_checkpoint).exists():
        ckpt = torch.load(str(resume_checkpoint), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_f1 = ckpt.get("best_val_f1", -1.0)
        best_epoch = ckpt.get("best_epoch", 0)
        print(f"[*] Resumed Stage 1 checkpoint from {resume_checkpoint} (epoch {start_epoch}, best F1: {best_val_f1:.4f})")

    max_batches = 2 if smoke_test else None
    actual_epochs = start_epoch if smoke_test else epochs

    print(f"Starting 3D Tumor Presence Classifier Training on {device} (AMP={use_amp})")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Class weights: healthy={class_weights[0]:.2f}, tumor={class_weights[1]:.2f}")
    print(f"Model parameters: {model.count_parameters():,}")

    for epoch in range(start_epoch, actual_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            max_batches=max_batches,
        )

        val_loss, val_metrics = validate_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            max_batches=max_batches,
        )

        current_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()

        val_f1 = val_metrics["macro_f1"]
        val_acc = val_metrics["accuracy"]
        val_sens = val_metrics["sensitivity"]
        val_spec = val_metrics["specificity"]

        history["train_loss"].append(round(train_loss, 5))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 5))
        history["val_acc"].append(round(val_acc, 4))
        history["val_macro_f1"].append(round(val_f1, 4))
        history["val_sensitivity"].append(round(val_sens, 4))
        history["val_specificity"].append(round(val_spec, 4))
        history["val_roc_auc"].append(val_metrics.get("roc_auc"))
        history["lr"].append(current_lr)

        print(
            f"Epoch {epoch:02d}/{actual_epochs} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.3f}, F1: {val_f1:.3f}, "
            f"Sens: {val_sens:.3f}, Spec: {val_spec:.3f}"
        )

        # Checkpoint selection based on validation macro F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0

            best_ckpt_path = out_dir / "tumor_presence_best.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_f1": best_val_f1,
                    "val_metrics": val_metrics,
                    "architecture": "TumorPresenceClassifier3D",
                    "in_channels": 2,
                    "num_classes": 2,
                    "base_channels": base_channels,
                    "dropout_rate": dropout_rate,
                    "training_metadata": {
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "seed": seed,
                        "batch_size": batch_size,
                        "learning_rate": lr,
                        "class_weights": class_weights.cpu().tolist(),
                    },
                },
                best_ckpt_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience and not smoke_test:
                print(f"Early stopping triggered at epoch {epoch} (patience={patience}).")
                break

    # Save latest checkpoint
    latest_ckpt_path = out_dir / "tumor_presence_latest.pth"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "history": history,
            "architecture": "TumorPresenceClassifier3D",
            "base_channels": base_channels,
        },
        latest_ckpt_path,
    )

    # Save training history JSON
    history_path = log_dir / "presence_training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_val_f1": best_val_f1,
                "history": history,
                "final_val_metrics": val_metrics,
            },
            f,
            indent=2,
        )

    return {
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "history": history,
        "best_checkpoint": str(out_dir / "tumor_presence_best.pth"),
        "latest_checkpoint": str(latest_ckpt_path),
        "history_file": str(history_path),
    }


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for Stage 1 training invocation."""
    parser = argparse.ArgumentParser(
        description="Train the 3D Brain Tumor Presence Classifier (Stage 1: Healthy Control vs. Tumor Present)."
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per forward pass.")
    parser.add_argument("--lr", type=float, default=1e-4, help="AdamW initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--base-channels", type=int, default=16, help="Initial channel width.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Head dropout probability.")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--smoke-test", action="store_true", help="Execute 2 mini-batches for verification.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_presence_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        base_channels=args.base_channels,
        dropout_rate=args.dropout,
        patience=args.patience,
        seed=args.seed,
        num_workers=args.num_workers,
        smoke_test=args.smoke_test,
        resume_checkpoint=args.resume,
        output_dir=args.output_dir,
    )
