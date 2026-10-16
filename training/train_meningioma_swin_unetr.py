# ============================================================
# MENINGIOMA SWIN UNETR TRAINING PIPELINE
# ============================================================
#
# Dataset:
#   BraTS2024-MEN-RT (Preprocessed T1c MRI)
#
# Task:
#   Binary meningioma GTV segmentation
#
# Model:
#   Swin UNETR (1 input channel, 2 output classes, feature_size=12)
#
# Isolated Output Directory:
#   outputs/meningioma_swin_unetr/
# ============================================================

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.meningioma_swin_unetr import create_meningioma_swin_unetr
from preprocessing.meningioma_dataset import MeningiomaPatchDataset


# ============================================================
# LOSS FUNCTIONS & METRICS
# ============================================================

class DiceLoss(nn.Module):
    """Soft Dice Loss for multi-class/binary one-hot targets."""

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        num_classes = logits.shape[1]

        target_one_hot = F.one_hot(target.long(), num_classes=num_classes)
        target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()

        dims = (0, 2, 3, 4)
        intersection = torch.sum(probs * target_one_hot, dim=dims)
        cardinality = torch.sum(probs, dim=dims) + torch.sum(target_one_hot, dim=dims)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class DiceCELoss(nn.Module):
    """Combined Dice and Cross-Entropy loss."""

    def __init__(self, dice_weight: float = 0.5, ce_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice_loss(logits, target) + self.ce_weight * self.ce_loss(logits, target)


def calculate_tumor_dice(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> float:
    """Calculates Dice score for Class 1 (Tumor GTV)."""
    preds = torch.argmax(logits, dim=1)
    pred_tumor = (preds == 1)
    target_tumor = (target == 1)

    intersection = (pred_tumor & target_tumor).sum().float().item()
    pred_vol = pred_tumor.sum().float().item()
    target_vol = target_tumor.sum().float().item()

    denominator = pred_vol + target_vol
    if denominator == 0:
        return 1.0
    return (2.0 * intersection + smooth) / (denominator + smooth)


# ============================================================
# TRAINING & VALIDATION EPOCH
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    grad_clip: float = 5.0,
    max_batches: Optional[int] = None,
) -> dict:
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    batch_count = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(loader, start=1):
        if max_batches is not None and batch_idx > max_batches:
            break

        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        target = masks.squeeze(1)

        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            with torch.amp.autocast('cuda', enabled=True):
                logits = model(images)
                loss = criterion(logits, target)
        else:
            logits = model(images)
            loss = criterion(logits, target)

        if device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        with torch.no_grad():
            dice = calculate_tumor_dice(logits, target)

        running_loss += loss.detach().item()
        running_dice += dice
        batch_count += 1

        if batch_idx == 1 or batch_idx % 25 == 0 or (max_batches and batch_idx == max_batches):
            print(f"    Batch {batch_idx:03d} | Loss: {loss.detach().item():.4f} | Tumor Dice: {dice:.4f}")

    elapsed = time.time() - start_time
    return {
        "loss": running_loss / max(batch_count, 1),
        "dice": running_dice / max(batch_count, 1),
        "batches": batch_count,
        "elapsed_seconds": elapsed,
    }


def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> dict:
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    batch_count = 0
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            if max_batches is not None and batch_idx > max_batches:
                break

            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            target = masks.squeeze(1)

            if device.type == "cuda":
                with torch.amp.autocast('cuda', enabled=True):
                    logits = model(images)
                    loss = criterion(logits, target)
            else:
                logits = model(images)
                loss = criterion(logits, target)

            dice = calculate_tumor_dice(logits, target)

            running_loss += loss.item()
            running_dice += dice
            batch_count += 1

    elapsed = time.time() - start_time
    return {
        "loss": running_loss / max(batch_count, 1),
        "dice": running_dice / max(batch_count, 1),
        "batches": batch_count,
        "elapsed_seconds": elapsed,
    }


# ============================================================
# MAIN TRAINING RUNNER
# ============================================================

def run_training(args):
    print("=" * 80)
    print("MENINGIOMA SWIN UNETR TRAINING")
    print("=" * 80)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Device:               {device}")
    if device.type == "cuda":
        print(f"GPU:                  {torch.cuda.get_device_name(0)}")

    output_dir = Path(args.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    logs_dir = output_dir / "logs"
    predictions_dir = output_dir / "predictions"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic_test:
        print("\n" + "=" * 80)
        print("RUNNING SYNTHETIC MODEL SMOKE TEST")
        print("=" * 80)
        model = create_meningioma_swin_unetr(
            in_channels=1,
            out_channels=2,
            feature_size=args.feature_size,
            use_checkpoint=True,
        ).to(device)
        model.train()
        synthetic_input = torch.randn(1, 1, 64, 64, 64, device=device)
        synthetic_target = torch.randint(0, 2, (1, 64, 64, 64), device=device)
        criterion = DiceCELoss(dice_weight=0.5, ce_weight=0.5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        optimizer.zero_grad()
        if device.type == "cuda":
            with torch.amp.autocast('cuda', enabled=True):
                logits = model(synthetic_input)
                loss = criterion(logits, synthetic_target)
        else:
            logits = model(synthetic_input)
            loss = criterion(logits, synthetic_target)

        loss.backward()
        optimizer.step()
        print(f"[PASS] Synthetic forward, loss ({loss.item():.4f}), and backward pass completed.")
        return

    print(f"Output directory:     {output_dir}")
    print(f"Smoke test mode:      {args.smoke_test}")
    print(f"Feature size:         {args.feature_size}")
    print(f"Batch size:           {args.batch_size}")
    print(f"Learning rate:        {args.lr}")
    print(f"Epochs:               {1 if args.smoke_test else args.epochs}")

    # Datasets
    print("\nLoading datasets...")
    train_dataset = MeningiomaPatchDataset(split="train")
    dev_dataset = MeningiomaPatchDataset(split="dev")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Train samples:        {len(train_dataset)} ({len(train_loader)} batches)")
    print(f"Dev samples:          {len(dev_dataset)} ({len(dev_loader)} batches)")

    # Model instantiation
    print("\nInstantiating Swin UNETR model...")
    model = create_meningioma_swin_unetr(
        in_channels=1,
        out_channels=2,
        feature_size=args.feature_size,
        use_checkpoint=True,
    )
    model.to(device)

    criterion = DiceCELoss(dice_weight=0.5, ce_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    best_dice = -1.0
    history = {
        "epochs": [],
        "train_loss": [],
        "train_dice": [],
        "dev_loss": [],
        "dev_dice": [],
        "learning_rates": [],
    }

    num_epochs = 1 if args.smoke_test else args.epochs
    max_train_batches = 2 if args.smoke_test else None
    max_dev_batches = 2 if args.smoke_test else None

    print("\n" + "=" * 80)
    print("STARTING TRAINING LOOP")
    print("=" * 80)

    try:
        for epoch in range(1, num_epochs + 1):
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"\nEpoch [{epoch:02d}/{num_epochs:02d}] (LR: {current_lr:.6f})")

            # Training
            train_metrics = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                grad_clip=args.grad_clip,
                max_batches=max_train_batches,
            )

            # Validation
            dev_metrics = validate_one_epoch(
                model=model,
                loader=dev_loader,
                criterion=criterion,
                device=device,
                max_batches=max_dev_batches,
            )

            scheduler.step()

            print(f"  Summary Epoch {epoch:02d}:")
            print(f"    Train Loss: {train_metrics['loss']:.4f} | Train Dice: {train_metrics['dice']:.4f} ({train_metrics['elapsed_seconds']:.1f}s)")
            print(f"    Dev Loss:   {dev_metrics['loss']:.4f} | Dev Dice:   {dev_metrics['dice']:.4f} ({dev_metrics['elapsed_seconds']:.1f}s)")

            history["epochs"].append(epoch)
            history["train_loss"].append(train_metrics["loss"])
            history["train_dice"].append(train_metrics["dice"])
            history["dev_loss"].append(dev_metrics["loss"])
            history["dev_dice"].append(dev_metrics["dice"])
            history["learning_rates"].append(current_lr)

            # Save last checkpoint
            last_ckpt_path = checkpoints_dir / "meningioma_swin_last.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_dice": best_dice,
                    "history": history,
                    "architecture": "SwinUNETR",
                    "feature_size": args.feature_size,
                },
                last_ckpt_path,
            )

            # Save best checkpoint
            if dev_metrics["dice"] > best_dice:
                best_dice = dev_metrics["dice"]
                best_ckpt_path = checkpoints_dir / "meningioma_swin_best.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_dice": best_dice,
                        "history": history,
                        "architecture": "SwinUNETR",
                        "feature_size": args.feature_size,
                    },
                    best_ckpt_path,
                )
                print(f"    [BEST] Saved new best checkpoint -> {best_ckpt_path.name} (Dev Dice: {best_dice:.4f})")

            # Save history log JSON
            history_path = logs_dir / "training_history.json"
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

    except torch.cuda.OutOfMemoryError as e:
        print("\n" + "!" * 80)
        print("CUDA OUT OF MEMORY ERROR OCCURRED DURING TRAINING")
        print("Recommendations:")
        print("1. Ensure feature_size=12 and use_checkpoint=True are configured.")
        print("2. Set batch_size=1.")
        print("3. Ensure torch.cuda.empty_cache() is invoked between runs.")
        print("!" * 80)
        torch.cuda.empty_cache()
        raise e

    print("\n" + "=" * 80)
    print("TRAINING PROCESS COMPLETED SUCCESSFULLY")
    print(f"Best Dev Dice:        {best_dice:.4f}")
    print(f"Checkpoints dir:      {checkpoints_dir}")
    print(f"Logs file:            {logs_dir / 'training_history.json'}")
    print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(description="Isolated Meningioma Swin UNETR Training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1 for 6GB VRAM safety)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay for AdamW")
    parser.add_argument("--grad-clip", type=float, default=5.0, help="Gradient norm clipping")
    parser.add_argument("--feature-size", type=int, default=12, help="Swin UNETR feature size")
    parser.add_argument("--device", type=str, default="cuda", help="Target device (cuda or cpu)")
    parser.add_argument("--output-dir", type=str, default="outputs/meningioma_swin_unetr", help="Experiment output directory")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1 epoch with limited real batches for quick verification")
    parser.add_argument("--synthetic-test", action="store_true", help="Run synthetic tensor forward/backward test without loading dataset")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(args)
