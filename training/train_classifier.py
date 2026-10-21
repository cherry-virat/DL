"""
Training and Validation Engine for 3D Brain Tumor Classifier.
=============================================================

Trains the lightweight 3D ResNet classifier to discriminate between:
  - Class 0: Glioma
  - Class 1: Meningioma

Design Principles:
------------------
1. Disjoint Partitioning: Strictly uses pre-defined 'train' and 'val' splits from
   `data/classifier/split.json`. The 'test' split is strictly locked and never accessed
   during model training, checkpoint selection, or early stopping.
2. Safe Hardware Scaling: Designed for the 6 GB RTX 3050 Laptop GPU using automatic
   mixed precision (AMP) with dynamic GradScaler.
3. Metric Tracking: Tracks Loss, Accuracy, Macro F1, Per-class Precision/Recall, and
   Confusion Matrix per epoch.
4. Model Persistence: Saves the best checkpoint and full training history JSON to
   `outputs/classifier_training/`.
5. Non-executable on Import: Protected by a `if __name__ == "__main__":` entrypoint.
   Importing this module executes zero training and accesses no datasets.
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

from models.tumor_classifier_3d import TumorClassifier3D, create_tumor_classifier_3d
from training.dataset_classifier import BrainTumorClassifierDataset

# Try importing sklearn metrics with fallback to pure numpy
try:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        confusion_matrix,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "classifier" / "manifest.json"
DEFAULT_SPLIT_PATH = PROJECT_ROOT / "data" / "classifier" / "split.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "classifier_training"

CLASS_NAMES = {
    0: "glioma",
    1: "meningioma",
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


def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_prob: Optional[List[List[float]]] = None,
) -> Dict[str, Any]:
    """
    Computes standard classification evaluation metrics.

    Parameters:
        y_true: Ground truth integer class labels (0 or 1).
        y_pred: Predicted integer class labels (0 or 1).
        y_prob: Optional predicted class probability distributions.

    Returns:
        Dictionary containing accuracy, macro_f1, per-class metrics, and confusion matrix.
    """
    if len(y_true) == 0:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "precision_per_class": {"glioma": 0.0, "meningioma": 0.0},
            "recall_per_class": {"glioma": 0.0, "meningioma": 0.0},
            "f1_per_class": {"glioma": 0.0, "meningioma": 0.0},
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

    return {
        "accuracy": float(round(acc, 4)),
        "macro_f1": float(round(macro_f1, 4)),
        "precision_per_class": {
            "glioma": float(round(prec_arr[0], 4)),
            "meningioma": float(round(prec_arr[1], 4)),
        },
        "recall_per_class": {
            "glioma": float(round(rec_arr[0], 4)),
            "meningioma": float(round(rec_arr[1], 4)),
        },
        "f1_per_class": {
            "glioma": float(round(f1_arr[0], 4)),
            "meningioma": float(round(f1_arr[1], 4)),
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
    """
    Executes one epoch of model training.

    Returns:
        (avg_loss, accuracy)
    """
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
    """
    Evaluates the model on validation data without tracking gradients.

    Returns:
        (avg_loss, metrics_dict)
    """
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
    metrics = compute_classification_metrics(all_targets, all_preds, all_probs)
    return avg_loss, metrics


def run_training(
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
    Main training workflow for the 3D Tumor Classifier.

    Parameters:
        manifest_path: Path to dataset manifest.json.
        split_path: Path to dataset split.json.
        output_dir: Destination directory for checkpoints and logs.
        epochs: Number of training epochs.
        batch_size: Batch size per forward step (default 2 for 6 GB VRAM).
        lr: Initial AdamW learning rate.
        weight_decay: L2 regularization penalty.
        base_channels: Initial 3D ResNet channel width.
        dropout_rate: Linear head dropout rate.
        patience: Early stopping patience epochs on validation macro F1.
        seed: Random seed for reproducibility.
        device_str: Device to execute on ('cuda' or 'cpu').
        num_workers: DataLoader background workers.
        smoke_test: If True, evaluates only 2 mini-batches per split for verification.

    Returns:
        Summary dict containing training history and best checkpoint information.
    """
    set_seed(seed)

    manifest_p = Path(manifest_path)
    split_p = Path(split_path)
    out_dir = Path(output_dir)

    if not manifest_p.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_p}")
    if not split_p.exists():
        raise FileNotFoundError(f"Split file not found: {split_p}")

    ckpt_dir = out_dir / "checkpoints"
    log_dir = out_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # Load datasets
    train_dataset = BrainTumorClassifierDataset(
        split="train",
        manifest_path=manifest_p,
        split_path=split_p,
    )
    val_dataset = BrainTumorClassifierDataset(
        split="val",
        manifest_path=manifest_p,
        split_path=split_p,
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

    # Instantiate model
    model = create_tumor_classifier_3d(
        in_channels=4,
        num_classes=2,
        base_channels=base_channels,
        dropout_rate=dropout_rate,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    start_epoch = 1
    best_val_f1 = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    if resume_checkpoint is not None and Path(resume_checkpoint).exists():
        ckpt = torch.load(str(resume_checkpoint), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_f1 = ckpt.get("best_val_f1", -1.0)
        best_epoch = ckpt.get("epoch", 0)
        print(f"[*] Resumed Stage 2 checkpoint from {resume_checkpoint} (epoch {start_epoch}, best F1: {best_val_f1:.4f})")

    history: Dict[str, List] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "lr": [],
    }

    max_batches = 2 if smoke_test else None
    actual_epochs = start_epoch if smoke_test else epochs

    print(f"Starting 3D Tumor Classifier Training on {device} (AMP={use_amp})")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
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

        history["train_loss"].append(round(train_loss, 5))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 5))
        history["val_acc"].append(round(val_acc, 4))
        history["val_macro_f1"].append(round(val_f1, 4))
        history["lr"].append(current_lr)

        print(
            f"Epoch {epoch:02d}/{actual_epochs} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.3f}, Macro F1: {val_f1:.3f}"
        )

        # Checkpoint selection based on validation macro F1 with val_loss tie-breaker
        is_better = False
        if val_f1 > best_val_f1:
            is_better = True
        elif val_f1 == best_val_f1 and val_loss < best_val_loss:
            is_better = True

        if is_better:
            best_val_f1 = val_f1
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            ckpt_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_f1": best_val_f1,
                "val_metrics": val_metrics,
                "architecture": "TumorClassifier3D",
                "in_channels": 4,
                "num_classes": 2,
                "base_channels": base_channels,
                "dropout_rate": dropout_rate,
                "training_metadata": {
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "seed": seed,
                    "batch_size": batch_size,
                    "learning_rate": lr,
                },
            }
            if smoke_test:
                smoke_ckpt_path = ckpt_dir / "classifier_smoke_test.pth"
                torch.save(ckpt_dict, smoke_ckpt_path)
            else:
                best_ckpt_path = ckpt_dir / "classifier_best.pth"
                canonical_ckpt_path = out_dir / "tumor_classifier_best.pth"
                torch.save(ckpt_dict, best_ckpt_path)
                torch.save(ckpt_dict, canonical_ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= patience and not smoke_test:
                print(f"Early stopping triggered at epoch {epoch} (patience={patience}).")
                break

    # Save last checkpoint
    last_ckpt_path = ckpt_dir / "classifier_last.pth"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "history": history,
            "architecture": "TumorClassifier3D",
            "base_channels": base_channels,
        },
        last_ckpt_path,
    )

    # Save training history JSON
    history_path = log_dir / "training_history.json"
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
        "best_checkpoint": str(ckpt_dir / "classifier_best.pth"),
        "history_file": str(history_path),
    }


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for training invocation."""
    parser = argparse.ArgumentParser(
        description="Train the 3D Multimodal Brain Tumor Classifier (Glioma vs. Meningioma)."
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per forward pass.")
    parser.add_argument("--lr", type=float, default=1e-4, help="AdamW initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--base-channels", type=int, default=16, help="Initial channel width.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Head dropout probability.")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu').")
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=str(DEFAULT_MANIFEST_PATH),
        help="Path to manifest.json.",
    )
    parser.add_argument(
        "--split-path",
        type=str,
        default=str(DEFAULT_SPLIT_PATH),
        help="Path to split.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save checkpoints and logs.",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run 2 mini-batches only for pipeline verification.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(
        manifest_path=args.manifest_path,
        split_path=args.split_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        base_channels=args.base_channels,
        dropout_rate=args.dropout,
        patience=args.patience,
        seed=args.seed,
        device_str=args.device,
        num_workers=args.num_workers,
        smoke_test=args.smoke_test,
        resume_checkpoint=args.resume,
    )
