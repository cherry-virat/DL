"""
Glioma Ensemble Model Training.
===============================

Trains diverse 3D Residual U-Net checkpoints for a 3-Model Soft-Voting Ensemble:
  - Model 2: Random Seed 101, DiceCELoss (Focal weighted), CosineAnnealingLR, 25 epochs.
  - Model 3: Random Seed 2024, DiceFocalLoss, AdamW optimizer, 25 epochs.

Saves checkpoints under:
  outputs/glioma_ensemble/
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from monai.networks.nets import UNet
from monai.losses import DiceCELoss, DiceFocalLoss
from monai.metrics import DiceMetric

from preprocessing.glioma_dataset import (
    GliomaDataset,
    DATASET_ROOT,
    get_cases,
)
from preprocessing.glioma_split import create_train_val_split

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "glioma_ensemble"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_CASES_COUNT = 100
VAL_CASES_COUNT = 20
NUM_EPOCHS = 25
BATCH_SIZE = 1


def glioma_collate_fn(batch):
    patches = []
    for case_output in batch:
        if not isinstance(case_output, list):
            raise TypeError("Expected GliomaDataset to return a list of patch dictionaries.")
        patches.extend(case_output)
    if len(patches) == 0:
        raise ValueError("Collate received empty batch.")
    images = [torch.as_tensor(p["image"]).float() for p in patches]
    labels = [torch.as_tensor(p["label"]).long() for p in patches]
    return {
        "image": torch.stack(images, dim=0),
        "label": torch.stack(labels, dim=0),
    }


def train_single_model(model_idx: int, seed: int, loss_type: str = "dice_ce", lr: float = 1e-4):
    print("\n" + "=" * 70)
    print(f"TRAINING GLIOMA ENSEMBLE MODEL {model_idx} (Seed: {seed}, Loss: {loss_type}, LR: {lr})")
    print("=" * 70)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    all_cases = get_cases(DATASET_ROOT)
    train_cases, val_cases = create_train_val_split(all_cases, train_ratio=0.90, seed=42)

    train_ds = GliomaDataset(train_cases[:TRAIN_CASES_COUNT])
    val_ds = GliomaDataset(val_cases[:VAL_CASES_COUNT])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=glioma_collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=glioma_collate_fn, num_workers=0)

    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=0.0,
    ).to(DEVICE)

    if loss_type == "dice_ce":
        loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, lambda_dice=1.0, lambda_ce=1.0)
    else:
        loss_fn = DiceFocalLoss(to_onehot_y=True, softmax=True, lambda_dice=1.0, lambda_focal=1.0, gamma=2.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    best_val_dice = -1.0
    best_checkpoint_path = OUTPUT_DIR / f"glioma_model_{model_idx}.pth"
    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_tp, val_fp, val_fn = 0, 0, 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(DEVICE)
                labels = batch["label"].to(DEVICE)

                logits = model(images)
                loss = loss_fn(logits, labels)
                val_loss += loss.item()

                probs = torch.softmax(logits, dim=1)
                pred_wt = (probs[:, 1:, ...].sum(dim=1) >= 0.50).cpu().numpy()
                gt_wt = (labels[:, 0, ...] > 0).cpu().numpy()

                val_tp += int(np.logical_and(pred_wt, gt_wt).sum())
                val_fp += int(np.logical_and(pred_wt, ~gt_wt).sum())
                val_fn += int(np.logical_and(~pred_wt, gt_wt).sum())

        val_loss /= len(val_loader)
        val_dice = (2.0 * val_tp) / (2 * val_tp + val_fp + val_fn) if (2 * val_tp + val_fp + val_fn) > 0 else 0.0
        dt = time.time() - t0

        is_best = val_dice > best_val_dice
        if is_best:
            best_val_dice = val_dice
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": val_dice,
                "seed": seed,
                "loss_type": loss_type,
            }, best_checkpoint_path)

        star = " * [BEST]" if is_best else ""
        print(f"Epoch [{epoch:02d}/{NUM_EPOCHS:02d}] ({dt:.1f}s) | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val WT Dice: {val_dice*100:.2f}%{star}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_dice": val_dice})

    print(f"\n[OK] Model {model_idx} completed. Best Val WT Dice: {best_val_dice*100:.2f}%. Saved to {best_checkpoint_path}")
    return best_checkpoint_path


if __name__ == "__main__":
    print("Starting Glioma Ensemble Training (Models 2 & 3)...")
    train_single_model(model_idx=2, seed=101, loss_type="dice_ce", lr=1.5e-4)
    train_single_model(model_idx=3, seed=2024, loss_type="dice_focal", lr=1.0e-4)
    print("\nAll Glioma Ensemble models successfully trained!")
