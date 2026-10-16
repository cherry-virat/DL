"""
test_meningioma_multimodal_loader.py
====================================
Comprehensive verification and feasibility script for Phase 2:
1. Validates the 800/200 patient-level split (0 overlap).
2. Tests streaming data loading for cases from both source folders:
   - BraTS-MEN-Train
   - BraTS-MEN-Train-Fix
3. Verifies 4 channels [t1n, t1c, t2w, t2f] and multiclass labels [0..3].
4. Checks normalization (finite, zero-background, non-zero unit variance).
5. Runs small batch loading via PyTorch DataLoader.
6. Conducts isolated model forward-pass feasibility test on CUDA (RTX 3050).
   (NO backward pass, NO optimizer, NO training epoch).
"""

import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.meningioma_multimodal_unet import create_multimodal_unet
from training.meningioma_multimodal_dataset import (
    MeningiomaMultimodalDataset,
    calculate_multimodal_metrics,
    get_multimodal_dataloader,
)


def run_loader_verification():
    print("=" * 80)
    print("PHASE 2: BRA TS 2023 MULTIMODAL MENINGIOMA DATA PIPELINE VERIFICATION")
    print("=" * 80)

    split_file = PROJECT_ROOT / "data" / "meningioma_multimodal" / "split.json"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file missing: {split_file}")

    with open(split_file, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    train_cases = set(split_data["train_cases"])
    val_cases = set(split_data["val_cases"])
    overlap = train_cases.intersection(val_cases)

    print("\n--- 1. Split Verification ---")
    print(f"Total train cases: {len(train_cases)}")
    print(f"Total val cases:   {len(val_cases)}")
    print(f"Train/Val overlap: {len(overlap)} cases")
    assert len(train_cases) == 800, f"Expected 800 train cases, got {len(train_cases)}"
    assert len(val_cases) == 200, f"Expected 200 val cases, got {len(val_cases)}"
    assert len(overlap) == 0, f"Overlap detected: {overlap}"
    print("[OK] Split count and disjointness verified.")

    # Find sample case from BraTS-MEN-Train and BraTS-MEN-Train-Fix
    case_details = split_data.get("case_details", {})
    train_sample_id = None
    fix_sample_id = None
    for cid, info in case_details.items():
        if info["source_folder"] == "BraTS-MEN-Train" and train_sample_id is None:
            train_sample_id = cid
        if info["source_folder"] == "BraTS-MEN-Train-Fix" and fix_sample_id is None:
            fix_sample_id = cid
        if train_sample_id and fix_sample_id:
            break

    print("\n--- 2. Dual-Provenance Case Loading Test ---")
    print(f"Selected BraTS-MEN-Train sample:     {train_sample_id}")
    print(f"Selected BraTS-MEN-Train-Fix sample: {fix_sample_id}")

    dataset_full = MeningiomaMultimodalDataset(
        split_json_path=split_file,
        partition="all",
        patch_size=None,  # Full volume mode
    )

    for target_id in [train_sample_id, fix_sample_id]:
        idx = dataset_full.case_ids.index(target_id)
        t0 = time.time()
        sample = dataset_full[idx]
        elapsed = time.time() - t0

        img = sample["image"]
        msk = sample["mask"]
        cid = sample["case_id"]
        src = sample["source_folder"]

        print(f"\nCase: {cid} (Source: {src}) loaded in {elapsed:.3f}s")
        print(f"  Image shape: {tuple(img.shape)}, dtype: {img.dtype}")
        print(f"  Mask shape:  {tuple(msk.shape)}, dtype: {msk.dtype}")

        # Assertions
        assert img.shape == (4, 240, 240, 155), f"Unexpected image shape: {img.shape}"
        assert msk.shape == (240, 240, 155), f"Unexpected mask shape: {msk.shape}"
        assert torch.isfinite(img).all(), f"Non-finite image values detected in {cid}!"

        unique_labels = torch.unique(msk).tolist()
        print(f"  Unique labels present: {unique_labels}")
        assert all(lbl in [0, 1, 2, 3] for lbl in unique_labels), f"Invalid labels in {cid}: {unique_labels}"

        # Channel normalization checks
        for c in range(4):
            ch_data = img[c].numpy()
            nz = ch_data[ch_data != 0]
            if len(nz) > 0:
                print(f"  Channel {c} non-zero voxels: count={len(nz):,}, mean={nz.mean():.4f}, std={nz.std():.4f}")
                assert abs(nz.mean()) < 0.1, f"Mean not centered around zero: {nz.mean()}"
                assert abs(nz.std() - 1.0) < 0.1, f"Std not close to 1: {nz.std()}"

    print("[OK] Dual-provenance full-volume loading verified.")

    print("\n--- 3. Batch Loading via DataLoader (Patch Mode) ---")
    patch_size = (64, 64, 64)
    batch_size = 2
    loader = get_multimodal_dataloader(
        split_json_path=split_file,
        partition="train",
        batch_size=batch_size,
        shuffle=True,
        patch_size=patch_size,
        num_workers=0,
    )

    t0 = time.time()
    batch = next(iter(loader))
    b_time = time.time() - t0

    b_img = batch["image"]
    b_msk = batch["mask"]
    print(f"Loaded batch of size {batch_size} in {b_time:.3f}s:")
    print(f"  Batch image shape: {tuple(b_img.shape)}")
    print(f"  Batch mask shape:  {tuple(b_msk.shape)}")
    print(f"  Batch cases:       {batch['case_id']}")
    assert b_img.shape == (batch_size, 4, 64, 64, 64)
    assert b_msk.shape == (batch_size, 64, 64, 64)
    print("[OK] Patch extraction and batching verified.")

    print("\n--- 4. Multimodal Metric Calculation Test ---")
    # Synthetic verification of calculate_multimodal_metrics
    synth_pred = torch.zeros((64, 64, 64), dtype=torch.long)
    synth_gt = torch.zeros((64, 64, 64), dtype=torch.long)
    # Set matching labels in center
    synth_gt[20:30, 20:30, 20:30] = 3  # Enhancing tumor (1000 voxels)
    synth_pred[20:30, 20:30, 20:30] = 3
    synth_gt[30:35, 30:35, 30:35] = 2  # Edema (125 voxels)
    synth_pred[30:35, 30:35, 30:35] = 2

    metrics = calculate_multimodal_metrics(synth_pred, synth_gt)
    print(f"  WT Dice: {metrics['wt']['dice']:.4f}")
    print(f"  TC Dice: {metrics['tc']['dice']:.4f}")
    print(f"  ET Dice: {metrics['et']['dice']:.4f}")
    assert metrics["wt"]["dice"] == 1.0, "Expected perfect WT dice"
    assert metrics["et"]["dice"] == 1.0, "Expected perfect ET dice"
    print("[OK] Multimodal metric calculation verified.")

    print("\n--- 5. Model Forward-Pass Feasibility Test ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target device: {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name} ({total_vram_gb:.2f} GB total VRAM)")
        torch.cuda.reset_peak_memory_stats(0)

    model = create_multimodal_unet(in_channels=4, num_classes=4, base_channels=16)
    model = model.to(device)
    model.eval()

    # Pass the loaded batch or a patch through model
    input_patch = b_img[:1].to(device)  # [1, 4, 64, 64, 64]
    print(f"Forward input shape:  {tuple(input_patch.shape)}")

    with torch.no_grad():
        t0 = time.time()
        output_logits = model(input_patch)
        fwd_time = time.time() - t0

    print(f"Forward output shape: {tuple(output_logits.shape)}")
    print(f"Forward pass time:   {fwd_time:.4f}s")
    assert output_logits.shape == (1, 4, 64, 64, 64), f"Output shape mismatch: {output_logits.shape}"
    assert torch.isfinite(output_logits).all(), "Non-finite logits returned by model!"

    if torch.cuda.is_available():
        peak_alloc = torch.cuda.max_memory_allocated(0) / (1024**2)
        peak_resv = torch.cuda.max_memory_reserved(0) / (1024**2)
        print(f"Peak GPU VRAM Allocated: {peak_alloc:.2f} MB")
        print(f"Peak GPU VRAM Reserved:  {peak_resv:.2f} MB")
        assert peak_resv < 4000, "VRAM exceeded safe threshold for 6GB GPU!"

    print("[OK] Model forward-pass feasibility test completed successfully!")
    print("\n" + "=" * 80)
    print("ALL PHASE 2 VERIFICATION CHECKS PASSED PERFECTLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_loader_verification()

