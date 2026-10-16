"""
Meningioma Ensemble Model Training.
===================================

Trains diverse 3D Residual U-Net checkpoints for a 3-Model Soft-Voting Ensemble:
  - Model 2: Seed 101, Weighted DiceCELoss (tumor weight 2.0), AdamW, lr=1.5e-4, 25 epochs.
  - Model 3: Seed 2024, DiceLoss + FocalLoss, CosineAnnealingLR, 25 epochs.

Saves checkpoints under:
  outputs/meningioma_ensemble/
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.meningioma_model import create_meningioma_model
from preprocessing.meningioma_dataset import MeningiomaPatchDataset

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "meningioma_ensemble"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 25
BATCH_SIZE = 2


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)
        target_one_hot = F.one_hot(target.long(), num_classes=2).permute(0, 4, 1, 2, 3).float()
        dims = (0, 2, 3, 4)
        intersection = torch.sum(probs * target_one_hot, dim=dims)
        denominator = torch.sum(probs, dim=dims) + torch.sum(target_one_hot, dim=dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice[1]  # Tumor class loss


class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=0.5, dice_weight=0.5, tumor_weight=2.0):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        weights = torch.tensor([1.0, tumor_weight], device=DEVICE)
        self.ce = nn.CrossEntropyLoss(weight=weights)
        self.dice = DiceLoss()

    def forward(self, logits, target):
        return self.ce_weight * self.ce(logits, target) + self.dice_weight * self.dice(logits, target)


def calculate_tumor_dice(logits, target):
    pred = torch.argmax(logits, dim=1) == 1
    gt = target == 1
    intersection = torch.sum(pred & gt).float()
    pred_vol = torch.sum(pred).float()
    gt_vol = torch.sum(gt).float()
    denom = pred_vol + gt_vol
    if denom == 0:
        return torch.tensor(1.0, device=logits.device)
    return (2.0 * intersection + 1e-5) / (denom + 1e-5)


def train_single_meningioma_model(model_idx: int, seed: int, tumor_weight: float = 2.0, lr: float = 1e-4):
    print("\n" + "=" * 70)
    print(f"TRAINING MENINGIOMA ENSEMBLE MODEL {model_idx} (Seed: {seed}, Tumor Weight: {tumor_weight}, LR: {lr})")
    print("=" * 70)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_ds = MeningiomaPatchDataset(split="train")
    dev_ds = MeningiomaPatchDataset(split="dev")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=0)
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=0)

    model = create_meningioma_model(base_channels=16).to(DEVICE)
    criterion = CombinedLoss(ce_weight=0.4, dice_weight=0.6, tumor_weight=tumor_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    best_dev_dice = -1.0
    best_checkpoint_path = OUTPUT_DIR / f"meningioma_model_{model_idx}.pth"
    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            images = batch["image"].to(DEVICE, non_blocking=True)
            masks = batch["mask"].to(DEVICE, non_blocking=True).squeeze(1)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # Validation on Dev split
        model.eval()
        dev_loss = 0.0
        dev_dice_sum = 0.0

        with torch.no_grad():
            for batch in dev_loader:
                images = batch["image"].to(DEVICE, non_blocking=True)
                masks = batch["mask"].to(DEVICE, non_blocking=True).squeeze(1)

                logits = model(images)
                loss = criterion(logits, masks)
                dev_loss += loss.item()
                dice = calculate_tumor_dice(logits, masks)
                dev_dice_sum += dice.item()

        dev_loss /= len(dev_loader)
        dev_dice = dev_dice_sum / len(dev_loader)
        dt = time.time() - t0

        is_best = dev_dice > best_dev_dice
        if is_best:
            best_dev_dice = dev_dice
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "dev_dice": dev_dice,
                "seed": seed,
            }, best_checkpoint_path)

        star = " * [BEST]" if is_best else ""
        print(f"Epoch [{epoch:02d}/{NUM_EPOCHS:02d}] ({dt:.1f}s) | Train Loss: {train_loss:.4f} | Dev Loss: {dev_loss:.4f} | Dev Dice: {dev_dice*100:.2f}%{star}")
        history.append({"epoch": epoch, "train_loss": train_loss, "dev_loss": dev_loss, "dev_dice": dev_dice})

    print(f"\n[OK] Meningioma Model {model_idx} completed. Best Dev Dice: {best_dev_dice*100:.2f}%. Saved to {best_checkpoint_path}")
    return best_checkpoint_path


if __name__ == "__main__":
    print("Starting Meningioma Ensemble Training (Models 2 & 3)...")
    train_single_meningioma_model(model_idx=2, seed=101, tumor_weight=2.5, lr=1.5e-4)
    train_single_meningioma_model(model_idx=3, seed=2024, tumor_weight=1.5, lr=1.0e-4)
    print("\nAll Meningioma Ensemble models successfully trained!")
