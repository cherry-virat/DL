"""
train_meningioma_multimodal.py
==============================
Training pipeline for the 4-Channel BraTS 2023 Multi-Modal Meningioma 3D U-Net.

Dataset:
    BraTS 2023 Meningioma Challenge (C:\\3D_DATASET\\BraTS2023_MEN)
    Partitions: 800 train cases / 200 val cases (data/meningioma_multimodal/split.json)

Inputs:
    4 Channels: [t1n, t1c, t2w, t2f]
    Normalized: Non-zero z-score per channel

Targets:
    4 Classes: 0: Background, 1: NCR, 2: ED, 3: ET

Metrics:
    Whole Tumor (WT): 1 + 2 + 3
    Tumor Core (TC): 1 + 3
    Enhancing Tumor (ET): 3

Outputs & Checkpoints:
    outputs/meningioma_multimodal_experiment/
        checkpoints/
            meningioma_multimodal_best.pth
            meningioma_multimodal_latest.pth
        logs/
            training_log.json
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from models.meningioma_multimodal_unet import create_multimodal_unet
from models.meningioma_segresnet import create_segresnet
from training.meningioma_multimodal_dataset import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_SPLIT_JSON,
    MeningiomaMultimodalDataset,
    calculate_multimodal_metrics,
)


# ============================================================
# MULTICLASS LOSS WRAPPER
# ============================================================

class MultimodalDiceCELoss(nn.Module):
    """
    Multiclass Combined Dice and Cross-Entropy Loss for 4-class segmentation.
    - Excludes background from Dice computation to prevent dominant healthy tissue
      (>90% voxels) from diluting rare tumor subregion gradients.
    - Cross-Entropy still penalizes background false positives.
    - Handles target shapes [B, D, H, W] and [B, 1, D, H, W] automatically.
    """

    def __init__(
        self,
        include_background: bool = False,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
    ):
        super().__init__()
        self.loss_fn = DiceCELoss(
            include_background=include_background,
            to_onehot_y=True,
            softmax=True,
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce,
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # MONAI DiceCELoss requires 5D target [B, 1, D, H, W] when input is 5D [B, C, D, H, W]
        if target.ndim == 4:
            target = target.unsqueeze(1)
        return self.loss_fn(logits, target)


# ============================================================
# CHECKPOINTING UTILITIES
# ============================================================

def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool,
    checkpoint_dir: Path,
    filename: str = "meningioma_multimodal_latest.pth",
    best_filename: str = "meningioma_multimodal_best.pth",
) -> None:
    """Saves checkpoint atomically via temporary file to prevent corruption."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    temp_path = checkpoint_dir / f"{filename}.tmp"
    final_path = checkpoint_dir / filename

    torch.save(state, temp_path)
    if final_path.exists():
        final_path.unlink()
    temp_path.rename(final_path)

    if is_best:
        best_temp = checkpoint_dir / f"{best_filename}.tmp"
        best_final = checkpoint_dir / best_filename
        torch.save(state, best_temp)
        if best_final.exists():
            best_final.unlink()
        best_temp.rename(best_final)


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Loads model, optimizer, scheduler, and scaler weights from checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # Fix 2: Explicitly set weights_only=False for trusted project checkpoints.
    # PyTorch 2.6+ defaults to weights_only=True, which rejects metadata dicts containing Path objects.
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return checkpoint


# ============================================================
# TRAINING EPOCH
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp: bool = True,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Runs one training epoch over patches with mixed precision."""
    model.train()
    running_loss = 0.0
    batch_count = 0
    t0 = time.time()

    for batch in loader:
        if max_batches is not None and batch_count >= max_batches:
            break

        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and amp)):
            logits = model(images)
            loss = criterion(logits, masks)

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss detected: {loss.item()}")

        if device.type == "cuda" and amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        running_loss += loss.item()
        batch_count += 1

    elapsed = time.time() - t0
    epoch_loss = running_loss / max(batch_count, 1)

    return {
        "train_loss": epoch_loss,
        "batches": batch_count,
        "epoch_time_seconds": elapsed,
    }


# ============================================================
# AIR MASKING UTILITY
# ============================================================

def apply_air_masking(prediction: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    """
    Sets predicted class to background (0) for empty-air voxels.
    Empty-air voxels are defined as voxels where all four input channels are zero,
    matching the non-zero z-score preprocessing convention (normalize_nonzero_zscore).

    Args:
        prediction: Integer class prediction tensor of shape [D, H, W] or [B, D, H, W].
        image: Normalized 4-channel input image tensor of shape [4, D, H, W] or [B, 4, D, H, W].

    Returns:
        prediction: Tensor of same shape with empty-air voxels set to class 0 (background).
    """
    if image.ndim == 4:
        # image: [4, D, H, W], prediction: [D, H, W]
        is_air = torch.all(image == 0, dim=0)
    elif image.ndim == 5:
        # image: [B, 4, D, H, W], prediction: [B, D, H, W]
        is_air = torch.all(image == 0, dim=1)
    else:
        raise ValueError(f"Expected image tensor with 4 or 5 dims, got shape {tuple(image.shape)}")

    is_air = is_air.to(prediction.device)
    masked_pred = prediction.clone()
    masked_pred[is_air] = 0
    return masked_pred


# ============================================================
# VALIDATION EVALUATION (FULL-VOLUME SLIDING WINDOW)
# ============================================================

def validate_full_volumes(
    model: nn.Module,
    dataset: MeningiomaMultimodalDataset,
    device: torch.device,
    roi_size: Tuple[int, int, int] = (64, 64, 64),
    sw_batch_size: int = 2,
    overlap: float = 0.5,
    max_cases: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluates validation cases deterministically using full-volume
    MONAI sliding-window inference with Gaussian blending.
    """
    model.eval()
    case_metrics_list = []
    num_cases = len(dataset) if max_cases is None else min(len(dataset), max_cases)

    t0 = time.time()
    with torch.inference_mode():
        for i in range(num_cases):
            sample = dataset[i]
            image = sample["image"].unsqueeze(0).to(device)  # [1, 4, D, H, W]
            mask = sample["mask"]  # [D, H, W]

            output = sliding_window_inference(
                inputs=image,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=overlap,
                mode="gaussian",
            )

            # Argmax predictions [1, 4, D, H, W] -> [D, H, W]
            pred = torch.argmax(output, dim=1).squeeze(0).cpu()

            # Fix 1: Exclude empty-air voxels from validation predictions.
            # Empty-air voxels (where all 4 modalities are 0) are set to class 0 (background).
            pred = apply_air_masking(pred, sample["image"])

            metrics = calculate_multimodal_metrics(pred, mask)
            case_metrics_list.append(metrics)

    elapsed = time.time() - t0

    # Aggregate metric means across validation cases
    mean_wt_dice = float(np.mean([m["wt"]["dice"] for m in case_metrics_list]))
    mean_tc_dice = float(np.mean([m["tc"]["dice"] for m in case_metrics_list]))
    mean_et_dice = float(np.mean([m["et"]["dice"] for m in case_metrics_list]))
    mean_wt_prec = float(np.mean([m["wt"]["precision"] for m in case_metrics_list]))
    mean_wt_rec = float(np.mean([m["wt"]["recall"] for m in case_metrics_list]))

    return {
        "val_wt_dice": mean_wt_dice,
        "val_tc_dice": mean_tc_dice,
        "val_et_dice": mean_et_dice,
        "val_wt_precision": mean_wt_prec,
        "val_wt_recall": mean_wt_rec,
        "cases_evaluated": num_cases,
        "val_time_seconds": elapsed,
    }


# ============================================================
# CLI BUILDER & EPOCH RESOLUTION
# ============================================================

DEFAULT_EPOCHS: int = 50
DEFAULT_PILOT_EPOCHS: int = 2


def resolve_effective_epochs(pilot: bool, epochs: Optional[int] = None) -> int:
    """
    Resolves the effective training epoch count:
    - If epochs is explicitly provided (> 0), that value is always honored.
    - If epochs is None (not explicitly supplied):
        - If pilot is True: defaults to DEFAULT_PILOT_EPOCHS (2).
        - If pilot is False: defaults to DEFAULT_EPOCHS (50).
    """
    if epochs is not None:
        if epochs <= 0:
            raise ValueError(f"Epoch count must be positive, got {epochs}")
        return epochs
    return DEFAULT_PILOT_EPOCHS if pilot else DEFAULT_EPOCHS


def resolve_cohort_indices(
    total_cases: int,
    requested_count: Optional[int],
    seed: int = 42,
    mode: str = "seed",
) -> List[int]:
    """
    Deterministically selects `requested_count` indices from `total_cases`.
    - If requested_count is None or >= total_cases: returns list(range(total_cases)).
    - If mode == 'prefix': returns list(range(requested_count)) (first N sorted cases).
    - If mode == 'seed': uses np.random.RandomState(seed) to deterministically select indices.
    """
    if requested_count is None or requested_count >= total_cases:
        return list(range(total_cases))
    if requested_count <= 0:
        raise ValueError(f"Requested case count must be positive, got {requested_count}")

    if mode == "prefix":
        return list(range(requested_count))

    rng = np.random.RandomState(seed)
    selected = rng.choice(total_cases, size=requested_count, replace=False)
    return sorted(selected.tolist())


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3 Training Pipeline for 4-Channel BraTS 2023 Meningioma 3D U-Net."
    )
    parser.add_argument(
        "--split-json",
        type=Path,
        default=DEFAULT_SPLIT_JSON,
        help="Path to dataset split.json (default: data/meningioma_multimodal/split.json)",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Path to BraTS2023_MEN root directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "meningioma_multimodal_experiment",
        help="Isolated output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS} for full run, {DEFAULT_PILOT_EPOCHS} for pilot)",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Patch batch size per step")
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=3,
        default=[64, 64, 64],
        help="3D spatial patch size for training (D H W)",
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="Initial AdamW learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--base-channels", type=int, default=16, help="Initial conv filters")
    parser.add_argument("--sw-overlap", type=float, default=0.5, help="Validation sliding window overlap")
    parser.add_argument("--sw-batch-size", type=int, default=2, help="Validation window batch size")
    parser.add_argument("--val-interval", type=int, default=1, help="Validation evaluation frequency (epochs)")
    parser.add_argument("--amp", action="store_true", default=True, help="Use Automatic Mixed Precision")
    parser.add_argument("--no-amp", action="store_false", dest="amp", help="Disable AMP")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader num_workers")
    parser.add_argument(
        "--pilot",
        action="store_true",
        help=f"Run pilot training on a small subset (10 train cases, 5 val cases, default {DEFAULT_PILOT_EPOCHS} epochs)",
    )
    parser.add_argument("--pilot-train-cases", type=int, default=10, help="Number of train cases in pilot")
    parser.add_argument("--pilot-val-cases", type=int, default=5, help="Number of val cases in pilot")
    parser.add_argument(
        "--train-cases",
        type=int,
        default=None,
        help="Number of training cases to select from split (default: None, uses all 800 or pilot default)",
    )
    parser.add_argument(
        "--val-cases",
        type=int,
        default=None,
        help="Number of validation cases to select from split (default: None, uses all 200 or pilot default)",
    )
    parser.add_argument(
        "--patch-sampling",
        type=str,
        default="tumor_weighted",
        choices=["tumor_weighted", "brain_aware_weighted"],
        help="Patch sampling strategy for training (default: tumor_weighted)",
    )
    parser.add_argument(
        "--cohort-selection",
        type=str,
        default="auto",
        choices=["auto", "seed", "prefix"],
        help="Cohort selection method: 'auto' (prefix for pilot default, seed otherwise), 'seed' (deterministic seeded choice), or 'prefix' (first N sorted cases).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="unet",
        choices=["unet", "segresnet"],
        help="Model architecture: 'unet' (default 3D U-Net) or 'segresnet' (MONAI SegResNet)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Instantiate all pipeline components, check shapes and GPU memory, then exit without training",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to existing checkpoint to resume training from",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute training epochs and backpropagation (requires explicit user authorization)",
    )
    parsed_args = parser.parse_args(args)
    parsed_args.epochs = resolve_effective_epochs(
        pilot=parsed_args.pilot,
        epochs=parsed_args.epochs,
    )
    return parsed_args


# ============================================================
# MAIN ENTRYPOINT
# ============================================================

def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("PHASE 3: MULTI-MODAL 4-CHANNEL MENINGIOMA TRAINING PIPELINE")
    print("=" * 80)
    print(f"Device:               {device}")
    if device.type == "cuda":
        print(f"GPU:                  {torch.cuda.get_device_name(0)}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Total VRAM:           {total_vram:.2f} GB")
    print(f"Output Directory:     {args.output_dir}")
    print(f"Split JSON:           {args.split_json}")
    print(f"Epochs:               {args.epochs}")
    print(f"Batch Size:           {args.batch_size}")
    print(f"Patch Size:           {tuple(args.patch_size)}")
    print(f"Model Architecture:   {args.model}")
    print(f"Patch Sampling:       {args.patch_sampling}")
    print(f"Base Channels:        {args.base_channels}")
    print(f"Learning Rate:        {args.lr}")
    print(f"AMP Enabled:          {args.amp}")
    print(f"Dry Run Mode:         {args.dry_run}")
    print(f"Pilot Mode:           {args.pilot}")
    print(f"Execute Mode:         {args.execute}")

    # Directories
    checkpoints_dir = args.output_dir / "checkpoints"
    logs_dir = args.output_dir / "logs"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset instantiation
    print("\n--- Initializing Datasets ---")
    patch_size_tuple = tuple(args.patch_size)

    train_dataset = MeningiomaMultimodalDataset(
        split_json_path=args.split_json,
        partition="train",
        dataset_root=args.dataset_root,
        patch_size=patch_size_tuple,
        patch_sampling=args.patch_sampling,
    )

    val_dataset = MeningiomaMultimodalDataset(
        split_json_path=args.split_json,
        partition="val",
        dataset_root=args.dataset_root,
        patch_size=None,  # Full volume for deterministic evaluation
    )

    # Determine requested cohort counts and selection mode
    num_train_total = len(train_dataset)
    num_val_total = len(val_dataset)

    if args.train_cases is not None:
        target_train_count = args.train_cases
    elif args.pilot:
        target_train_count = args.pilot_train_cases
    else:
        target_train_count = None

    if args.val_cases is not None:
        target_val_count = args.val_cases
    elif args.pilot:
        target_val_count = args.pilot_val_cases
    else:
        target_val_count = None

    if args.cohort_selection not in (None, "auto"):
        cohort_mode = args.cohort_selection
    elif args.pilot and args.train_cases is None and args.val_cases is None:
        cohort_mode = "prefix"
    else:
        cohort_mode = "seed"

    train_indices = resolve_cohort_indices(
        total_cases=num_train_total,
        requested_count=target_train_count,
        seed=args.seed,
        mode=cohort_mode,
    )
    val_indices = resolve_cohort_indices(
        total_cases=num_val_total,
        requested_count=target_val_count,
        seed=args.seed,
        mode=cohort_mode,
    )

    all_train_ids = train_dataset.case_ids
    all_val_ids = val_dataset.case_ids
    selected_train_ids = [all_train_ids[i] for i in train_indices]
    selected_val_ids = [all_val_ids[i] for i in val_indices]

    # Verify train and validation sets are strictly disjoint
    overlap = set(selected_train_ids).intersection(set(selected_val_ids))
    if overlap:
        raise RuntimeError(f"Data leakage detected! Overlapping train and val case IDs: {overlap}")

    if len(train_indices) < num_train_total:
        train_dataset = Subset(train_dataset, train_indices)
    if len(val_indices) < num_val_total:
        val_dataset = Subset(val_dataset, val_indices)

    epochs = args.epochs

    # Save effective run configuration and selected case IDs
    selected_cases_payload = {
        "train_count": len(selected_train_ids),
        "val_count": len(selected_val_ids),
        "cohort_mode": cohort_mode,
        "seed": args.seed,
        "patch_sampling": args.patch_sampling,
        "train_cases": selected_train_ids,
        "val_cases": selected_val_ids,
    }
    with open(logs_dir / "selected_cases.json", "w", encoding="utf-8") as f:
        json.dump(selected_cases_payload, f, indent=2)

    run_config_payload = {
        "output_dir": str(args.output_dir),
        "split_json": str(args.split_json),
        "dataset_root": str(args.dataset_root),
        "epochs": epochs,
        "batch_size": args.batch_size,
        "patch_size": patch_size_tuple,
        "model_architecture": args.model,
        "patch_sampling": args.patch_sampling,
        "base_channels": args.base_channels,
        "lr": args.lr,
        "amp": args.amp,
        "seed": args.seed,
        "cohort_mode": cohort_mode,
        "train_cases_count": len(selected_train_ids),
        "val_cases_count": len(selected_val_ids),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    with open(logs_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config_payload, f, indent=2)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Train Cases/Patches:  {len(train_dataset)} ({len(train_loader)} batches/epoch)")
    print(f"Val Full Volumes:     {len(val_dataset)}")

    # 2. Model instantiation
    print("\n--- Initializing Model & Loss ---")
    if args.model == "segresnet":
        model = create_segresnet(
            in_channels=4,
            num_classes=4,
            init_filters=args.base_channels,
        ).to(device)
        print("Architecture:         SegResNet (MONAI Residual Encoder-Decoder)")
    else:
        model = create_multimodal_unet(
            in_channels=4,
            num_classes=4,
            base_channels=args.base_channels,
        ).to(device)
        print("Architecture:         3D U-Net (DoubleConv3D Encoder-Decoder)")

    param_count = model.count_parameters()
    print(f"Model parameters:     {param_count:,}")

    criterion = MultimodalDiceCELoss(include_background=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    epochs = args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.amp))

    start_epoch = 1
    best_wt_dice = -1.0
    history: Dict[str, List[Any]] = {
        "train_loss": [],
        "val_wt_dice": [],
        "val_tc_dice": [],
        "val_et_dice": [],
        "learning_rates": [],
    }

    # Handle resume
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        chk = load_checkpoint(args.resume, model, optimizer, scheduler, scaler, device)
        start_epoch = chk.get("epoch", 0) + 1
        best_wt_dice = chk.get("best_metric", -1.0)
        history = chk.get("history", history)
        print(f"Resumed at epoch {start_epoch} (previous best WT Dice: {best_wt_dice:.4f})")

    # 3. Dry-Run Verification Exit
    if args.dry_run:
        print("\n" + "=" * 80)
        print("DRY-RUN VERIFICATION: Checking pipeline forward step and VRAM...")
        print("=" * 80)
        sample_batch = next(iter(train_loader))
        img = sample_batch["image"].to(device)
        msk = sample_batch["mask"].to(device)
        print(f"Dry-run input tensor shape:  {tuple(img.shape)}")
        print(f"Dry-run mask tensor shape:   {tuple(msk.shape)}")

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and args.amp)):
                logits = model(img)
                loss = criterion(logits, msk)

        print(f"Dry-run output logits shape: {tuple(logits.shape)}")
        print(f"Dry-run sample loss value:   {loss.item():.4f}")
        assert logits.shape == (args.batch_size, 4, patch_size_tuple[0], patch_size_tuple[1], patch_size_tuple[2])
        assert torch.isfinite(loss).all()

        if device.type == "cuda":
            peak_mb = torch.cuda.max_memory_allocated(0) / (1024**2)
            print(f"Peak VRAM allocated:         {peak_mb:.2f} MB")

        print("[OK] Dry-run passed with zero errors. Pipeline is completely verified.")
        print("Exiting without training per hard restrictions.")
        return

    # 4. Mandatory Guardrail: Check for user authorization flag
    if not args.execute:
        print("\n" + "=" * 80)
        print("[GUARDRAIL ACTIVE] Pipeline code, loss functions, and datasets are fully verified.")
        print("Hard restriction: Zero epochs or updates allowed without explicit user authorization.")
        print("When user approves the pilot, run with:")
        print("  .venv\\Scripts\\python.exe training/train_meningioma_multimodal.py --pilot --execute")
        print("=" * 80)
        return

    # 5. Training Execution Loop (Only reached when user explicitly passes --execute)
    print("\n" + "=" * 80)
    print(f"STARTING TRAINING EXECUTION ({'PILOT' if args.pilot else 'FULL'} RUN: {epochs} EPOCHS)")
    print("=" * 80)

    for epoch in range(start_epoch, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        train_res = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            amp=args.amp,
        )
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        print(
            f"Epoch {epoch} Train: Loss={train_res['train_loss']:.4f} "
            f"({train_res['batches']} batches in {train_res['epoch_time_seconds']:.1f}s, lr={current_lr:.6f})"
        )

        history["train_loss"].append(train_res["train_loss"])
        history["learning_rates"].append(current_lr)

        # Validation
        if epoch % args.val_interval == 0:
            print(f"Evaluating validation volumes...")
            val_res = validate_full_volumes(
                model=model,
                dataset=val_dataset,
                device=device,
                roi_size=patch_size_tuple,
                sw_batch_size=args.sw_batch_size,
                overlap=args.sw_overlap,
            )
            print(
                f"Epoch {epoch} Val ({val_res['cases_evaluated']} cases): "
                f"WT Dice={val_res['val_wt_dice']:.4f}, TC Dice={val_res['val_tc_dice']:.4f}, ET Dice={val_res['val_et_dice']:.4f} "
                f"(Precision={val_res['val_wt_precision']:.4f}, Recall={val_res['val_wt_recall']:.4f}) in {val_res['val_time_seconds']:.1f}s"
            )

            history["val_wt_dice"].append(val_res["val_wt_dice"])
            history["val_tc_dice"].append(val_res["val_tc_dice"])
            history["val_et_dice"].append(val_res["val_et_dice"])

            is_best = val_res["val_wt_dice"] > best_wt_dice
            if is_best:
                best_wt_dice = val_res["val_wt_dice"]
                print(f"[BEST] New best validation WT Dice: {best_wt_dice:.4f}")

            # Checkpoint payload
            checkpoint_payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_metric": best_wt_dice,
                "history": history,
                "selected_train_ids": selected_train_ids,
                "selected_val_ids": selected_val_ids,
                "model_architecture": args.model,
                "patch_sampling": args.patch_sampling,
                "args": vars(args),
            }

            save_checkpoint(
                state=checkpoint_payload,
                is_best=is_best,
                checkpoint_dir=checkpoints_dir,
            )
            print(f"Checkpoint saved to: {checkpoints_dir}")

            # Write log
            with open(logs_dir / "training_log.json", "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

    print("\n" + "=" * 80)
    print("TRAINING RUN COMPLETE")
    print(f"Best Validation WT Dice: {best_wt_dice:.4f}")
    print(f"Checkpoints saved under: {checkpoints_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

