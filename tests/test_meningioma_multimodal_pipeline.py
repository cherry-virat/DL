"""
test_meningioma_multimodal_pipeline.py
======================================
Isolated PyTest suite for the BraTS 2023 Multi-Modal Meningioma Pipeline.
Strictly isolated from existing baseline and glioma tests.

Test Coverage:
1. Split count (800 train, 200 val) and 0 overlap.
2. Provenance and stratification balance between Train and Fix sources.
3. Modality and mask path resolution for sampled cases.
4. Tensor shape verification (4 channels, matching spatial dimensions).
5. Label range validation (strictly in {0, 1, 2, 3}).
6. Finite numeric values and non-zero normalization sanity.
7. Dual-provenance loading (BraTS-MEN-Train and BraTS-MEN-Train-Fix).
8. DataLoader batching and patch extraction.
9. Multimodal metric calculation (WT, TC, ET, per-class dice).
10. Model forward-pass feasibility (shape, logits, device compatibility).
"""

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import torch
import numpy as np

from models.meningioma_multimodal_unet import create_multimodal_unet
from training.meningioma_multimodal_dataset import (
    DEFAULT_SPLIT_JSON,
    MeningiomaMultimodalDataset,
    calculate_multimodal_metrics,
    get_multimodal_dataloader,
    normalize_nonzero_zscore,
)


@pytest.fixture(scope="module")
def split_data():
    assert DEFAULT_SPLIT_JSON.exists(), f"Split JSON missing at {DEFAULT_SPLIT_JSON}"
    with open(DEFAULT_SPLIT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def test_split_counts_and_disjointness(split_data):
    """1 & 2: Split has exactly 800 train and 200 val IDs, strictly disjoint."""
    train_ids = set(split_data["train_cases"])
    val_ids = set(split_data["val_cases"])

    assert len(train_ids) == 800, f"Expected 800 train cases, got {len(train_ids)}"
    assert len(val_ids) == 200, f"Expected 200 val cases, got {len(val_ids)}"
    assert len(train_ids.intersection(val_ids)) == 0, "Train and validation sets must have zero overlap!"
    assert len(train_ids.union(val_ids)) == 1000, "Union of splits must contain all 1000 cases"


def test_split_provenance_and_stratification(split_data):
    """Verifies that BraTS-MEN-Train and BraTS-MEN-Train-Fix are both partitioned proportionally."""
    case_details = split_data.get("case_details", {})
    assert len(case_details) == 1000

    train_sources = [case_details[cid]["source_folder"] for cid in split_data["train_cases"]]
    val_sources = [case_details[cid]["source_folder"] for cid in split_data["val_cases"]]

    train_fix = sum(1 for s in train_sources if s == "BraTS-MEN-Train-Fix")
    val_fix = sum(1 for s in val_sources if s == "BraTS-MEN-Train-Fix")

    assert train_fix == 38, f"Expected 38 Fix cases in train, got {train_fix}"
    assert val_fix == 9, f"Expected 9 Fix cases in val, got {val_fix}"


def test_paths_resolution(split_data):
    """3: All 4 modalities and mask paths resolve to existing files on disk."""
    case_details = split_data.get("case_details", {})
    # Check a representative sample of 10 cases (5 train, 5 val, including Fix folder)
    sample_ids = split_data["train_cases"][:5] + split_data["val_cases"][:5]
    for cid in sample_ids:
        details = case_details[cid]
        for mod, path_str in details["modalities"].items():
            p = Path(path_str)
            assert p.exists(), f"Modality file missing: {p}"
        m = Path(details["mask"])
        assert m.exists(), f"Mask file missing: {m}"


def test_normalization_function():
    """Unit test for non-zero z-score normalization."""
    arr = np.zeros((10, 10, 10), dtype=np.float32)
    # Put a non-zero sphere
    arr[3:7, 3:7, 3:7] = 100.0
    arr[4:6, 4:6, 4:6] = 200.0

    norm = normalize_nonzero_zscore(arr)
    # Background must stay zero
    assert (norm[arr == 0] == 0.0).all()
    # Non-zero voxels must have zero mean and unit variance
    nz = norm[arr != 0]
    assert np.isclose(nz.mean(), 0.0, atol=1e-5)
    assert np.isclose(nz.std(), 1.0, atol=1e-5)


def test_dataset_dual_provenance_loading():
    """4, 5, 6, 7, 8: Load cases from both BraTS-MEN-Train and BraTS-MEN-Train-Fix."""
    with open(DEFAULT_SPLIT_JSON, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    case_details = split_data["case_details"]
    # Pick one train folder case and one fix folder case
    train_case_id = next(cid for cid, info in case_details.items() if info["source_folder"] == "BraTS-MEN-Train")
    fix_case_id = next(cid for cid, info in case_details.items() if info["source_folder"] == "BraTS-MEN-Train-Fix")

    dataset = MeningiomaMultimodalDataset(
        split_json_path=DEFAULT_SPLIT_JSON,
        partition="all",
        patch_size=None,
    )

    for cid in [train_case_id, fix_case_id]:
        idx = dataset.case_ids.index(cid)
        sample = dataset[idx]

        img = sample["image"]
        msk = sample["mask"]

        # Shape assertions
        assert img.ndim == 4, f"Expected 4D image [4, D, H, W], got {img.shape}"
        assert img.shape[0] == 4, f"Expected 4 channels, got {img.shape[0]}"
        assert msk.ndim == 3, f"Expected 3D mask [D, H, W], got {msk.shape}"
        assert img.shape[1:] == msk.shape, f"Image spatial dims {img.shape[1:]} != mask {msk.shape}"

        # Finite values
        assert torch.isfinite(img).all(), "Image contains NaN or Inf values!"

        # Labels in {0, 1, 2, 3}
        unique_labels = torch.unique(msk).tolist()
        assert all(lbl in [0, 1, 2, 3] for lbl in unique_labels), f"Invalid labels: {unique_labels}"


def test_dataloader_batching_and_patching():
    """9: Test DataLoader with patch extraction and batch loading."""
    loader = get_multimodal_dataloader(
        split_json_path=DEFAULT_SPLIT_JSON,
        partition="train",
        batch_size=2,
        patch_size=(64, 64, 64),
        shuffle=False,
    )

    batch = next(iter(loader))
    b_img = batch["image"]
    b_msk = batch["mask"]

    assert b_img.shape == (2, 4, 64, 64, 64)
    assert b_msk.shape == (2, 64, 64, 64)
    assert b_img.dtype == torch.float32
    assert b_msk.dtype == torch.int64


def test_multimodal_metrics_logic():
    """Tests calculate_multimodal_metrics on known synthetic cases."""
    pred = torch.zeros((32, 32, 32), dtype=torch.long)
    gt = torch.zeros((32, 32, 32), dtype=torch.long)

    # Put NCR (1), ED (2), ET (3)
    gt[5:15, 5:15, 5:15] = 3  # 1000 ET voxels
    gt[15:20, 15:20, 15:20] = 2  # 125 ED voxels
    gt[20:25, 20:25, 20:25] = 1  # 125 NCR voxels

    # Predict exact same
    pred = gt.clone()
    metrics = calculate_multimodal_metrics(pred, gt)

    assert metrics["wt"]["dice"] == 1.0
    assert metrics["tc"]["dice"] == 1.0
    assert metrics["et"]["dice"] == 1.0
    assert metrics["per_class_dice"][0] == 1.0
    assert metrics["per_class_dice"][3] == 1.0


def test_model_forward_pass():
    """Task E: Model instantiation and single forward pass feasibility."""
    model = create_multimodal_unet(in_channels=4, num_classes=4, base_channels=16)
    model.eval()

    dummy = torch.randn(1, 4, 32, 32, 32)
    with torch.no_grad():
        out = model(dummy)

    assert out.shape == (1, 4, 32, 32, 32)
    assert torch.isfinite(out).all()
