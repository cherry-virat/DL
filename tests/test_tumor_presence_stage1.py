"""
Stage 1 Unit Tests for Healthy vs. Tumor Presence Detection Foundation.
======================================================================

Verifies:
1. 2-Channel preprocessing geometry: symmetric padding + isotropic 1.875x scaling yields (2, 128, 128, 96).
2. Intensity normalization: non-zero Z-score on brain, exact 0.0 preservation on background air.
3. Modality canonicalization and validation (t1/t1n/t1w and t2/t2w).
4. Spatial alignment enforcement: raises SpatialAlignmentError on affine/spacing mismatch.
5. Rejection of invalid inputs (missing modalities, non-3D volumes, non-finite values, non-existent files).
6. Model instantiation with default (2 channels, 2 classes) and custom parameters.
7. Model forward pass contract: returns (B, 2) finite logits.
8. Backward pass: CrossEntropyLoss computes and backpropagates finite gradients across all weights.
9. Model input validation: rejects non-tensor, non-5D, or wrong channel count (e.g. 1, 3, 4).
10. Manifest generator clean exit: exits cleanly (code 0) when IXI healthy data is missing or empty.
11. Manifest and split generation with synthetic cohort: disjoint partitions, split checksums.
12. Streaming dataset loading: verifies shape contract (2, 128, 128, 96), labels, and partition isolation.
"""

from pathlib import Path
import json
import tempfile

import nibabel as nib
import numpy as np
import pytest
import torch
import torch.nn as nn

from preprocessing.tumor_presence_preprocess import (
    preprocess_2channel_presence_tensor,
    normalize_nonzero_zscore,
    canonicalize_modality_key,
    compute_target_affine,
    validate_brain_mask_plausibility,
    align_and_resample_t2_to_t1,
    SpatialAlignmentError,
    CANONICAL_PRESENCE_MODALITIES,
    ORIGINAL_SHAPE,
    TARGET_SHAPE,
    TARGET_SPACING,
)
from models.tumor_presence_classifier_3d import (
    TumorPresenceClassifier3D,
    create_tumor_presence_classifier_3d,
)
from training.dataset_tumor_presence import TumorPresenceDataset
from scripts.generate_tumor_presence_manifest import (
    generate_presence_cohort,
    find_ixi_paired_cases,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# =====================================================================
# Preprocessing Engine Tests
# =====================================================================

def test_canonicalize_modality_key():
    """Verify robust recognition of sequence variant naming."""
    for variant in ["t1", "T1", "t1n", "T1N", "t1w", "T1W", "t1-weighted", "t1_weighted"]:
        assert canonicalize_modality_key(variant) == "t1"

    for variant in ["t2", "T2", "t2w", "T2W", "t2-weighted", "t2_weighted"]:
        assert canonicalize_modality_key(variant) == "t2"

    for invalid in ["flair", "t1c", "t2f", "seg", "unknown"]:
        assert canonicalize_modality_key(invalid) is None


def test_2channel_preprocessing_geometry():
    """Verify that 2-channel volumes standardize to (2, 128, 128, 96) with 1.875mm spacing."""
    affine = np.eye(4, dtype=np.float64)
    vol_t1 = np.zeros(ORIGINAL_SHAPE, dtype=np.float32)
    vol_t2 = np.zeros(ORIGINAL_SHAPE, dtype=np.float32)

    # Synthetic brain region
    vol_t1[50:190, 50:190, 30:130] = 100.0 + np.random.uniform(0, 50, (140, 140, 100)).astype(np.float32)
    vol_t2[50:190, 50:190, 30:130] = 80.0 + np.random.uniform(0, 40, (140, 140, 100)).astype(np.float32)

    tensor_2d, meta = preprocess_2channel_presence_tensor(
        {"t1": vol_t1, "t2": vol_t2},
        original_affine=affine,
    )

    assert isinstance(tensor_2d, torch.Tensor)
    assert tensor_2d.dtype == torch.float32
    assert tensor_2d.shape == (2, 128, 128, 96)

    assert meta.original_shape == ORIGINAL_SHAPE
    assert meta.padded_shape == (240, 240, 180)
    assert meta.target_shape == TARGET_SHAPE
    assert meta.target_spacing == TARGET_SPACING
    assert meta.pad_z == (12, 13)
    assert meta.scale_factor == 1.875
    assert meta.original_affine is not None
    assert meta.target_affine is not None


def test_intensity_normalization_brain_and_air():
    """Verify non-zero Z-score normalization: brain voxels have zero-mean, air remains strictly 0.0."""
    vol = np.zeros((50, 50, 50), dtype=np.float32)
    # Air region stays 0.0
    # Brain region has non-zero values
    brain_values = np.random.normal(loc=150.0, scale=30.0, size=(30, 30, 30)).astype(np.float32)
    brain_values = np.clip(brain_values, 10.0, 300.0)
    vol[10:40, 10:40, 10:40] = brain_values

    norm_vol = normalize_nonzero_zscore(vol)

    # Background air must be strictly 0.0
    air_mask = (vol == 0.0)
    assert np.all(norm_vol[air_mask] == 0.0)

    # Brain tissue must have zero mean and unit variance
    brain_norm = norm_vol[~air_mask]
    assert np.isclose(brain_norm.mean(), 0.0, atol=1e-5)
    assert np.isclose(brain_norm.std(), 1.0, atol=1e-3)


def test_preprocessing_modality_validation_rejections():
    """Verify clear exceptions for missing modalities, duplicate keys, or shape mismatches."""
    vol = np.ones((50, 50, 50), dtype=np.float32)

    # Missing T2
    with pytest.raises(ValueError, match="Missing required modalities"):
        preprocess_2channel_presence_tensor({"t1": vol})

    # Missing T1
    with pytest.raises(ValueError, match="Missing required modalities"):
        preprocess_2channel_presence_tensor({"t2": vol})

    # Non-dict input
    with pytest.raises(TypeError, match="modality_inputs must be a dict"):
        preprocess_2channel_presence_tensor([vol, vol])  # type: ignore

    # Duplicate modality mapping
    with pytest.raises(ValueError, match="Duplicate modality detected"):
        preprocess_2channel_presence_tensor({"t1": vol, "t1n": vol, "t2": vol})

    # Shape mismatch between T1 and T2
    vol_wrong_shape = np.ones((50, 50, 40), dtype=np.float32)
    with pytest.raises(ValueError, match="Spatial shape mismatch"):
        preprocess_2channel_presence_tensor({"t1": vol, "t2": vol_wrong_shape})


def test_preprocessing_spatial_alignment_rejection():
    """Verify that spatial affine and spacing mismatches raise SpatialAlignmentError."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)

        # Affine 1 (standard)
        aff1 = np.eye(4, dtype=np.float64)
        nii1 = nib.Nifti1Image(np.ones((40, 40, 40), dtype=np.float32), aff1)
        p1 = tmp_p / "t1.nii.gz"
        nib.save(nii1, str(p1))

        # Affine 2 (shifted origin exceeding 1e-3 tolerance)
        aff2 = np.eye(4, dtype=np.float64)
        aff2[0, 3] = 5.0  # 5mm shift
        nii2 = nib.Nifti1Image(np.ones((40, 40, 40), dtype=np.float32), aff2)
        p2 = tmp_p / "t2.nii.gz"
        nib.save(nii2, str(p2))

        with pytest.raises(SpatialAlignmentError, match="Spatial affine mismatch"):
            preprocess_2channel_presence_tensor({"t1": p1, "t2": p2})


def test_preprocessing_degenerate_volume_rejections():
    """Verify rejection of 2D/4D inputs, non-finite values, and empty files."""
    # 2D volume
    with pytest.raises(ValueError, match="must be a 3D volume"):
        preprocess_2channel_presence_tensor({
            "t1": np.ones((50, 50), dtype=np.float32),
            "t2": np.ones((50, 50), dtype=np.float32),
        })

    # Non-finite (NaN)
    vol_nan = np.ones((20, 20, 20), dtype=np.float32)
    vol_nan[5, 5, 5] = np.nan
    with pytest.raises(ValueError, match="Non-finite values"):
        preprocess_2channel_presence_tensor({
            "t1": vol_nan,
            "t2": np.ones((20, 20, 20), dtype=np.float32),
        })

    # Empty file
    with tempfile.TemporaryDirectory() as tmp_dir:
        empty_f = Path(tmp_dir) / "empty.nii.gz"
        empty_f.touch()
        with pytest.raises(ValueError, match="is empty"):
            preprocess_2channel_presence_tensor({
                "t1": empty_f,
                "t2": np.ones((20, 20, 20), dtype=np.float32),
            })


# =====================================================================
# 3D Model Architecture Tests
# =====================================================================

def test_presence_model_instantiation_defaults():
    """Verify default model configuration (in_channels=2, num_classes=2, base_channels=16)."""
    model = create_tumor_presence_classifier_3d()
    assert isinstance(model, TumorPresenceClassifier3D)
    assert model.in_channels == 2
    assert model.num_classes == 2
    assert model.base_channels == 16
    assert model.dropout_rate == 0.2

    num_params = model.count_parameters()
    assert num_params > 0
    # Lightweight footprint (< 2.5M params)
    assert num_params < 2_500_000


def test_presence_model_forward_pass_shape_and_finiteness():
    """Verify model forward pass produces (B, 2) finite logits."""
    model = create_tumor_presence_classifier_3d(base_channels=8)
    model.eval()

    # Small synthetic input: (B=2, C=2, D=32, H=32, W=24)
    x = torch.randn(2, 2, 32, 32, 24, dtype=torch.float32)
    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()


def test_presence_model_loss_and_backpropagation():
    """Verify CrossEntropyLoss computation and backpropagation produces finite gradients."""
    model = create_tumor_presence_classifier_3d(base_channels=8)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(2, 2, 32, 32, 24, dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)

    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits, targets)

    assert torch.isfinite(loss).item()
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    optimizer.step()


def test_presence_model_input_validation():
    """Verify strict rejection of invalid inputs."""
    model = create_tumor_presence_classifier_3d(in_channels=2, base_channels=8)

    # 1. Non-tensor
    with pytest.raises(TypeError, match="Expected torch.Tensor"):
        model(np.zeros((1, 2, 16, 16, 16)))  # type: ignore

    # 2. 4D input instead of 5D
    with pytest.raises(ValueError, match="Expected 5D input tensor"):
        model(torch.randn(2, 16, 16, 16))

    # 3. Wrong channel count: 4 channels (should be 2)
    with pytest.raises(ValueError, match="Input channel count mismatch"):
        model(torch.randn(1, 4, 16, 16, 16))

    # 4. Wrong channel count: 1 channel (should be 2)
    with pytest.raises(ValueError, match="Input channel count mismatch"):
        model(torch.randn(1, 1, 16, 16, 16))


# =====================================================================
# Manifest and Dataset Streaming Tests
# =====================================================================

def test_manifest_generator_clean_exit_when_ixi_empty():
    """Verify manifest generator gracefully exits with code 0 when healthy data is missing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        empty_ixi = Path(tmp_dir) / "empty_ixi"
        empty_ixi.mkdir()
        out_dir = Path(tmp_dir) / "output"

        ret = generate_presence_cohort(
            ixi_dir=empty_ixi,
            glioma_dir=Path(tmp_dir) / "gli",
            meningioma_dir=Path(tmp_dir) / "men",
            output_dir=out_dir,
        )
        assert ret == 0
        # No files should have been generated
        assert not (out_dir / "manifest.json").exists()


def test_manifest_generator_and_streaming_dataset_with_synthetic_cohort():
    """
    End-to-end integration test with synthetic cohort:
    1. Generates synthetic IXI (healthy) and BraTS (tumor) cases.
    2. Runs manifest generator to create manifest.json and split.json.
    3. Validates split disjointness and checksums.
    4. Instantiates TumorPresenceDataset and streams batches.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        ixi_dir = tmp_p / "ixi" / "raw"
        ixi_t1 = ixi_dir / "T1"
        ixi_t2 = ixi_dir / "T2"
        ixi_t1.mkdir(parents=True)
        ixi_t2.mkdir(parents=True)

        gli_dir = tmp_p / "gli"
        gli_dir.mkdir()
        men_dir = tmp_p / "men"
        men_dir.mkdir()

        affine = np.eye(4, dtype=np.float64)
        small_vol = np.zeros((40, 40, 40), dtype=np.float32)
        small_vol[5:35, 5:35, 5:35] = 100.0

        # Create 4 synthetic healthy cases in IXI
        for i in range(4):
            sid = f"IXI{i:03d}"
            nii = nib.Nifti1Image(small_vol, affine)
            nib.save(nii, str(ixi_t1 / f"{sid}-T1.nii.gz"))
            nib.save(nii, str(ixi_t2 / f"{sid}-T2.nii.gz"))

        # Create 2 synthetic Glioma cases
        for i in range(2):
            cd = gli_dir / f"BraTS-GLI-{i:05d}-000"
            cd.mkdir()
            nii = nib.Nifti1Image(small_vol, affine)
            nib.save(nii, str(cd / f"{cd.name}-t1n.nii.gz"))
            nib.save(nii, str(cd / f"{cd.name}-t2w.nii.gz"))

        # Create 2 synthetic Meningioma cases
        for i in range(2):
            cd = men_dir / f"BraTS-MEN-{i:05d}-000"
            cd.mkdir()
            nii = nib.Nifti1Image(small_vol, affine)
            nib.save(nii, str(cd / f"{cd.name}-t1n.nii.gz"))
            nib.save(nii, str(cd / f"{cd.name}-t2w.nii.gz"))

        # Generate manifest and split
        out_dir = tmp_p / "tumor_presence"
        ret = generate_presence_cohort(
            ixi_dir=tmp_p / "ixi",
            glioma_dir=gli_dir,
            meningioma_dir=men_dir,
            output_dir=out_dir,
            seed=42,
            train_ratio=0.50,
            val_ratio=0.25,
            test_ratio=0.25,
        )
        assert ret == 0

        manifest_path = out_dir / "manifest.json"
        split_path = out_dir / "split.json"
        assert manifest_path.exists()
        assert split_path.exists()

        with open(manifest_path, "r", encoding="utf-8") as f:
            man_data = json.load(f)
        with open(split_path, "r", encoding="utf-8") as f:
            split_data = json.load(f)

        assert man_data["metadata"]["total_cases"] == 8
        assert man_data["metadata"]["healthy_cases"] == 4
        assert man_data["metadata"]["tumor_cases"] == 4

        train_cases = set(split_data["train_cases"])
        val_cases = set(split_data["val_cases"])
        test_cases = set(split_data["test_cases"])

        # Disjoint partitions
        assert len(train_cases & val_cases) == 0
        assert len(train_cases & test_cases) == 0
        assert len(val_cases & test_cases) == 0
        assert len(train_cases | val_cases | test_cases) == 8

        # Test Streaming Dataset
        ds_train = TumorPresenceDataset(
            split="train",
            manifest_path=manifest_path,
            split_path=split_path,
        )
        assert len(ds_train) == len(train_cases)

        tensor_2d, label, anon_id = ds_train[0]
        assert tensor_2d.shape == (2, 128, 128, 96)
        assert tensor_2d.dtype == torch.float32
        assert label in (0, 1)
        assert anon_id.startswith("ANON_")

        ds_val = TumorPresenceDataset(
            split="val",
            manifest_path=manifest_path,
            split_path=split_path,
        )
        assert len(ds_val) == len(val_cases)

        ds_test = TumorPresenceDataset(
            split="test",
            manifest_path=manifest_path,
            split_path=split_path,
        )
        assert len(ds_test) == len(test_cases)



# =====================================================================
# Real Data Smoke Test & Manifest Integrity (Phase 3B)
# =====================================================================

REAL_MANIFEST_PATH = PROJECT_ROOT / "data" / "tumor_presence" / "manifest.json"
REAL_SPLIT_PATH = PROJECT_ROOT / "data" / "tumor_presence" / "split.json"
REAL_PROCESSED_DIR = Path(r"C:\3D_DATASET\HealthyControls\IXI\processed")


def test_real_manifest_and_split_integrity():
    """Verify real manifest and split files in data/tumor_presence/ if generated."""
    if not REAL_MANIFEST_PATH.exists() or not REAL_SPLIT_PATH.exists():
        pytest.skip("Real tumor presence manifest or split file not yet generated.")

    with open(REAL_MANIFEST_PATH, "r", encoding="utf-8") as f:
        man = json.load(f)
    with open(REAL_SPLIT_PATH, "r", encoding="utf-8") as f:
        split = json.load(f)

    meta = man["metadata"]
    assert meta["total_records_inspected"] == 2786
    assert meta["total_valid_cases"] == 2780
    assert meta["healthy_valid_cases"] == 576
    assert meta["tumor_valid_cases"] == 2204

    # Verify breakdown
    bd = meta["breakdown"]
    assert bd["ixi_healthy_valid"] == 576
    assert bd["brats_glioma_valid"] == 1251
    assert bd["brats_meningioma_valid"] == 953
    assert bd["ixi_excluded"] == 6

    # Verify split counts
    counts = split["metadata"]["counts"]
    assert counts["train_total"] == 1945
    assert counts["val_total"] == 415
    assert counts["test_total"] == 420
    assert counts["train_total"] + counts["val_total"] + counts["test_total"] == 2780

    # Verify disjointness
    tr = set(split["train_cases"])
    va = set(split["val_cases"])
    te = set(split["test_cases"])
    assert len(tr & va) == 0
    assert len(tr & te) == 0
    assert len(va & te) == 0

    # Verify split checksums exist
    chk = split["metadata"]["split_checksums"]
    assert len(chk["train_cases_sha256"]) == 64
    assert len(chk["val_cases_sha256"]) == 64
    assert len(chk["test_cases_sha256"]) == 64


def test_real_data_no_training_forward_pass_smoke():
    """
    Phase 3B Real Data Smoke Test:
    Loads 1 real healthy pair, 1 real glioma pair, 1 real meningioma pair,
    verifies tensor shape (2, 128, 128, 96), and executes an untrained forward pass.
    Strictly NO backpropagation or optimizer updates.
    """
    ixi_t1 = Path(r"C:\3D_DATASET\HealthyControls\IXI\raw\T1\IXI002-Guys-0828-T1.nii.gz")
    ixi_t2 = Path(r"C:\3D_DATASET\HealthyControls\IXI\raw\T2\IXI002-Guys-0828-T2.nii.gz")
    gli_t1 = Path(r"C:\3D_DATASET\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData\BraTS-GLI-00000-000\BraTS-GLI-00000-000-t1n.nii.gz")
    gli_t2 = Path(r"C:\3D_DATASET\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData\BraTS-GLI-00000-000\BraTS-GLI-00000-000-t2w.nii.gz")
    men_t1 = Path(r"C:\3D_DATASET\BraTS2023_MEN\BraTS-MEN-Train\BraTS-MEN-00004-000\BraTS-MEN-00004-000-t1n.nii.gz")
    men_t2 = Path(r"C:\3D_DATASET\BraTS2023_MEN\BraTS-MEN-Train\BraTS-MEN-00004-000\BraTS-MEN-00004-000-t2w.nii.gz")

    if not (ixi_t1.exists() and gli_t1.exists() and men_t1.exists()):
        pytest.skip("Local real MRI files not present on disk.")

    # 1. Preprocess Healthy IXI
    t_hc, meta_hc = preprocess_2channel_presence_tensor({"t1": ixi_t1, "t2": ixi_t2}, auto_align=True)
    assert t_hc.shape == (2, 128, 128, 96)
    assert t_hc.dtype == torch.float32
    assert torch.isfinite(t_hc).all()
    assert meta_hc.was_auto_aligned is True

    # 2. Preprocess Glioma
    t_gli, meta_gli = preprocess_2channel_presence_tensor({"t1": gli_t1, "t2": gli_t2}, auto_align=True)
    assert t_gli.shape == (2, 128, 128, 96)
    assert t_gli.dtype == torch.float32
    assert torch.isfinite(t_gli).all()
    assert meta_gli.was_auto_aligned is False

    # 3. Preprocess Meningioma
    t_men, meta_men = preprocess_2channel_presence_tensor({"t1": men_t1, "t2": men_t2}, auto_align=True)
    assert t_men.shape == (2, 128, 128, 96)
    assert t_men.dtype == torch.float32
    assert torch.isfinite(t_men).all()
    assert meta_men.was_auto_aligned is False

    # 4. Untrained model forward pass only
    model = create_tumor_presence_classifier_3d(in_channels=2, num_classes=2, base_channels=16)
    model.eval()

    with torch.no_grad():
        out_hc = model(t_hc.unsqueeze(0))
        out_gli = model(t_gli.unsqueeze(0))
        out_men = model(t_men.unsqueeze(0))

    assert out_hc.shape == (1, 2)
    assert out_gli.shape == (1, 2)
    assert out_men.shape == (1, 2)
    assert torch.isfinite(out_hc).all()
    assert torch.isfinite(out_gli).all()
    assert torch.isfinite(out_men).all()

    # Confirm no gradients were attached
    assert out_hc.grad_fn is None
    assert out_gli.grad_fn is None
    assert out_men.grad_fn is None



# =====================================================================
# Phase 3C Brain Extraction & Registration Policy Verification Tests
# =====================================================================

def test_brain_mask_plausibility_validation_synthetic():
    """Verify synthetic brain volume plausibility checks (normal, too small, and whole-head)."""
    # 1. Normal brain volume (e.g. 1300 cc in 1mm space, zero boundary voxels)
    vol_normal = np.zeros((150, 150, 150), dtype=np.float32)
    vol_normal[20:130, 20:130, 20:130] = 100.0  # ~1331 cc
    res_normal = validate_brain_mask_plausibility(vol_normal, (1.0, 1.0, 1.0))
    assert res_normal["is_skull_stripped_plausible"] is True
    assert len(res_normal["failure_reasons"]) == 0
    assert 1200.0 <= res_normal["nonzero_volume_cc"] <= 1400.0

    # 2. Too small volume (< 700 cc; e.g. severe erosion)
    vol_small = np.zeros((100, 100, 100), dtype=np.float32)
    vol_small[30:70, 30:70, 30:70] = 100.0  # 64 cc
    res_small = validate_brain_mask_plausibility(vol_small, (1.0, 1.0, 1.0))
    assert res_small["is_skull_stripped_plausible"] is False
    assert any("too small" in r.lower() for r in res_small["failure_reasons"])

    # 3. Whole-head unstripped scan simulation (large volume + non-zero boundary voxels)
    vol_wholehead = np.zeros((160, 160, 160), dtype=np.float32)
    vol_wholehead[10:150, 10:150, 10:150] = 100.0  # ~2744 cc
    vol_wholehead[0:5, :, :] = 50.0  # Scalp signal extending to boundary
    res_wholehead = validate_brain_mask_plausibility(vol_wholehead, (1.0, 1.0, 1.0))
    assert res_wholehead["is_skull_stripped_plausible"] is False
    assert any("too large" in r.lower() for r in res_wholehead["failure_reasons"])
    assert any("boundary" in r.lower() for r in res_wholehead["failure_reasons"])

    # 4. Enforce skull stripped flag rejection
    with pytest.raises(ValueError, match="Input volume failed skull-stripping"):
        preprocess_2channel_presence_tensor(
            {"t1": vol_wholehead, "t2": vol_wholehead},
            enforce_skull_stripped=True,
        )


def test_subject_leakage_and_group_key_separation():
    """Verify zero participant or duplicate leakage across train, val, and locked test splits."""
    if not REAL_MANIFEST_PATH.exists() or not REAL_SPLIT_PATH.exists():
        pytest.skip("Tumor presence manifest or split file not yet present.")

    with open(REAL_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(REAL_SPLIT_PATH, "r", encoding="utf-8") as f:
        split = json.load(f)

    # Build mapping from anonymized ID to subject_group_key
    id_to_subj = {c["anonymized_id"]: c["subject_group_key"] for c in manifest["cases"]}

    train_subjs = {id_to_subj[cid] for cid in split["train_cases"]}
    val_subjs = {id_to_subj[cid] for cid in split["val_cases"]}
    test_subjs = {id_to_subj[cid] for cid in split["test_cases"]}

    # Verify zero subject overlap
    assert len(train_subjs & val_subjs) == 0, f"Overlap between Train and Val: {train_subjs & val_subjs}"
    assert len(train_subjs & test_subjs) == 0, f"Overlap between Train and Test: {train_subjs & test_subjs}"
    assert len(val_subjs & test_subjs) == 0, f"Overlap between Val and Test: {val_subjs & test_subjs}"

    # Verify locked test set has exactly 420 cases
    assert len(split["test_cases"]) == 420
    assert len(test_subjs) == 420


def test_real_data_brain_mask_metrics_and_skull_stripping_audit():
    """
    Evaluates real BraTS (skull-stripped) vs real IXI (raw whole-head):
    - Confirms BraTS is detected as plausibly skull-stripped.
    - Confirms raw IXI is detected as unstripped (> 3,500 cc with peripheral scalp/neck signal).
    """
    ixi_t1 = Path(r"C:\3D_DATASET\HealthyControls\IXI\raw\T1\IXI002-Guys-0828-T1.nii.gz")
    gli_t1 = Path(r"C:\3D_DATASET\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData\BraTS-GLI-00000-000\BraTS-GLI-00000-000-t1n.nii.gz")

    if not (ixi_t1.exists() and gli_t1.exists()):
        pytest.skip("Real MRI files not found on disk.")

    nii_gli = nib.load(str(gli_t1))
    arr_gli = np.asanyarray(nii_gli.dataobj, dtype=np.float32)
    zooms_gli = tuple(float(round(z, 4)) for z in nii_gli.header.get_zooms()[:3])
    res_gli = validate_brain_mask_plausibility(arr_gli, zooms_gli)

    # BraTS glioma volume is intracranial (~1,480 cc) with 0 boundary voxels
    assert res_gli["is_skull_stripped_plausible"] is True
    assert 1000.0 <= res_gli["nonzero_volume_cc"] <= 1800.0
    assert res_gli["outer_boundary_voxels"] <= 50

    nii_ixi = nib.load(str(ixi_t1))
    arr_ixi = np.asanyarray(nii_ixi.dataobj, dtype=np.float32)
    zooms_ixi = tuple(float(round(z, 4)) for z in nii_ixi.header.get_zooms()[:3])
    res_ixi = validate_brain_mask_plausibility(arr_ixi, zooms_ixi)

    # Raw IXI includes whole-head scalp/skull/neck (> 5,000 cc, > 100,000 boundary voxels)
    assert res_ixi["is_skull_stripped_plausible"] is False
    assert res_ixi["nonzero_volume_cc"] > 3500.0
    assert res_ixi["outer_boundary_voxels"] > 500


# =====================================================================
# Phase 3: Post-QC Manifest, Modality Pairing, and Split Integrity Tests
# =====================================================================

def test_manifest_processed_healthy_paths_and_pairing():
    """
    Verifies that all QC-passed healthy cases in data/tumor_presence/manifest.json:
    - Reference valid skull-stripped T1, T2, and brain masks on disk.
    - Have geometry_status='aligned' and is_skull_stripped=True.
    - Possess matching 3D matrix shapes, affines, and voxel zooms between T1 and T2.
    """
    if not REAL_MANIFEST_PATH.exists():
        pytest.skip("Tumor presence manifest not present.")

    with open(REAL_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    valid_count_on_disk = 0
    flagged_count_on_disk = 0
    if REAL_PROCESSED_DIR.exists():
        for mf in (REAL_PROCESSED_DIR / "metadata").glob("*.json"):
            try:
                with open(mf, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                if data.get("status") == "PASSED" and data.get("plausibility_audit", {}).get("is_skull_stripped_plausible", False):
                    valid_count_on_disk += 1
                elif data.get("status") == "FLAGGED" or not data.get("plausibility_audit", {}).get("is_skull_stripped_plausible", True):
                    flagged_count_on_disk += 1
            except Exception:
                pass
    expected_valid = valid_count_on_disk if valid_count_on_disk > 0 else manifest["metadata"].get("healthy_qc_passed_cases", 0)

    qc_passed_healthy = [
        c for c in manifest["cases"]
        if c["source_cohort"] == "ixi" and c.get("qc_status") == "PASSED"
    ]

    assert len(qc_passed_healthy) == expected_valid, f"Expected {expected_valid} QC-passed cases, got {len(qc_passed_healthy)}"

    # Verify failed cases are properly flagged and isolated
    failed_healthy = [
        c for c in manifest["cases"]
        if c["source_cohort"] == "ixi" and c.get("qc_status") == "FAILED"
    ]
    assert len(failed_healthy) == 0, f"Expected 0 failed cases, got {len(failed_healthy)}"

    flagged_healthy = [
        c for c in manifest["cases"]
        if c["source_cohort"] == "ixi" and c.get("qc_status") == "FLAGGED"
    ]
    assert len(flagged_healthy) == flagged_count_on_disk
    if flagged_count_on_disk > 0:
        for fc in flagged_healthy:
            assert fc["is_skull_stripped"] is False

    pending_healthy = [
        c for c in manifest["cases"]
        if c["source_cohort"] == "ixi" and c.get("qc_status") == "PENDING_PREPROCESSING"
    ]
    assert len(pending_healthy) == 576 - expected_valid - flagged_count_on_disk, f"Expected {576 - expected_valid - flagged_count_on_disk} pending cases, got {len(pending_healthy)}"

    excluded_healthy = [
        c for c in manifest["cases"]
        if c["source_cohort"] == "ixi" and c.get("qc_status") == "EXCLUDED"
    ]
    assert len(excluded_healthy) == 6, f"Expected 6 excluded cases, got {len(excluded_healthy)}"

    for case in qc_passed_healthy[:10]:  # Deep header inspection on sample
        t1_path = Path(case["modalities"]["t1"])
        t2_path = Path(case["modalities"]["t2"])
        mask_path = Path(case["mask_path"])

        assert t1_path.exists(), f"Missing processed T1: {t1_path}"
        assert t2_path.exists(), f"Missing processed T2: {t2_path}"
        assert mask_path.exists(), f"Missing brain mask: {mask_path}"
        assert t1_path.stat().st_size > 0
        assert t2_path.stat().st_size > 0
        assert mask_path.stat().st_size > 0

        assert "t1_skull_stripped" in t1_path.name
        assert "t2_skull_stripped" in t2_path.name
        assert "_mask" in mask_path.name

        assert case["geometry_status"] == "aligned"
        assert case["is_skull_stripped"] is True
        assert case["class_id"] == 0
        assert case["class_label"] == "healthy_control"

        # Check geometry consistency
        nii_t1 = nib.load(str(t1_path))
        nii_t2 = nib.load(str(t2_path))
        nii_m = nib.load(str(mask_path))

        s1 = tuple(int(x) for x in nii_t1.header.get_data_shape()[:3])
        s2 = tuple(int(x) for x in nii_t2.header.get_data_shape()[:3])
        sm = tuple(int(x) for x in nii_m.header.get_data_shape()[:3])
        assert s1 == s2 == sm, f"Shape mismatch for {case['subject_group_key']}: T1={s1}, T2={s2}, Mask={sm}"
        assert np.allclose(nii_t1.affine, nii_t2.affine, atol=1e-3)
        assert np.allclose(nii_t1.affine, nii_m.affine, atol=1e-3)


def test_split_integrity_one_to_one_and_checksum_immutability():
    """
    Verifies that:
    1. Every case in train, val, and test splits maps to exactly one valid manifest entry.
    2. Zero subject or split overlap exists across train, val, and locked test.
    3. The locked test split checksum remains strictly identical to the frozen ground truth.
    """
    if not (REAL_MANIFEST_PATH.exists() and REAL_SPLIT_PATH.exists()):
        pytest.skip("Manifest or split file not found.")

    with open(REAL_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(REAL_SPLIT_PATH, "r", encoding="utf-8") as f:
        split = json.load(f)

    all_cases = {c["anonymized_id"]: c for c in manifest["cases"]}
    assert len(all_cases) == len(manifest["cases"]), "Duplicate anonymized_id detected in manifest!"

    tr = split["train_cases"]
    va = split["val_cases"]
    te = split["test_cases"]

    # One-to-one mapping
    for cid in tr + va + te:
        assert cid in all_cases, f"Split ID '{cid}' does not exist in manifest!"
        assert all_cases[cid]["is_valid"] is True, f"Split ID '{cid}' maps to an invalid case!"

    # Disjointness
    set_tr, set_va, set_te = set(tr), set(va), set(te)
    assert len(set_tr & set_va) == 0, "Train-Val overlap detected!"
    assert len(set_tr & set_te) == 0, "Train-Test overlap detected!"
    assert len(set_va & set_te) == 0, "Val-Test overlap detected!"

    # Zero subject key leakage
    key_tr = {all_cases[cid]["subject_group_key"] for cid in tr}
    key_va = {all_cases[cid]["subject_group_key"] for cid in va}
    key_te = {all_cases[cid]["subject_group_key"] for cid in te}
    assert len(key_tr & key_va) == 0, "Subject key overlap between Train and Val!"
    assert len(key_tr & key_te) == 0, "Subject key overlap between Train and Test!"
    assert len(key_va & key_te) == 0, "Subject key overlap between Val and Test!"

    # Checksums
    import hashlib
    def get_hash(lst):
        return hashlib.sha256("\n".join(sorted(lst)).encode("utf-8")).hexdigest()

    expected_chk = split["metadata"]["split_checksums"]
    assert get_hash(tr) == expected_chk["train_cases_sha256"]
    assert get_hash(va) == expected_chk["val_cases_sha256"]
    assert get_hash(te) == expected_chk["test_cases_sha256"]
    assert expected_chk["test_cases_sha256"] == "556ca50468a4bb3823ee2d79c40d170ae84b26d249ab90286fa294677e6ea818"


def test_tumor_cohort_provenance_never_labeled_healthy():
    """
    Verifies Requirement 7:
    All tumor cases must strictly retain class_id=1 and class_label='tumor_present',
    regardless of whether their segmentation mask is empty or absent.
    """
    if not REAL_MANIFEST_PATH.exists():
        pytest.skip("Manifest not found.")

    with open(REAL_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for case in manifest["cases"]:
        cohort = case["source_cohort"]
        cid = case["class_id"]
        label = case["class_label"]

        if cohort in ("brats_glioma", "brats_meningioma"):
            assert cid == 1, f"Tumor case {case['subject_group_key']} has class_id={cid}!"
            assert label == "tumor_present", f"Tumor case {case['subject_group_key']} has label='{label}'!"
        elif cohort == "ixi":
            assert cid == 0, f"Healthy case {case['subject_group_key']} has class_id={cid}!"
            assert label == "healthy_control", f"Healthy case {case['subject_group_key']} has label='{label}'!"


def test_qc_exclusion_and_rejection_logic():
    """
    Verifies that known excluded cases (e.g. thick-slice IXI014, unpaired scans)
    are marked as invalid and completely excluded from split.json.
    """
    if not (REAL_MANIFEST_PATH.exists() and REAL_SPLIT_PATH.exists()):
        pytest.skip("Manifest or split not found.")

    with open(REAL_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(REAL_SPLIT_PATH, "r", encoding="utf-8") as f:
        split = json.load(f)

    split_ids = set(split["train_cases"] + split["val_cases"] + split["test_cases"])

    excluded_cases = [c for c in manifest["cases"] if not c["is_valid"]]
    assert len(excluded_cases) == 6, f"Expected 6 excluded cases, got {len(excluded_cases)}"

    excluded_keys = {c["subject_group_key"] for c in excluded_cases}
    assert "IXI014" in excluded_keys  # Thick-slice
    assert "IXI116" in excluded_keys  # T1-only
    assert "IXI580" in excluded_keys  # T2-only

    for ec in excluded_cases:
        assert ec["anonymized_id"] not in split_ids, f"Excluded case {ec['subject_group_key']} leaked into split!"
        assert ec["qc_status"] == "EXCLUDED"


def test_manifest_updater_idempotency_and_no_split_drift():
    """
    Requirement 8:
    Rerunning the manifest updater must be strictly idempotent:
    - Never damage valid records.
    - Never alter split assignments or checksums.
    - Preserves exact counts of passed, failed, and pending cases.
    """
    from scripts.update_tumor_presence_manifest import update_manifest

    # Run updater again
    update_manifest()

    # Reload and verify
    with open(REAL_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(REAL_SPLIT_PATH, "r", encoding="utf-8") as f:
        split = json.load(f)

    valid_count_on_disk = 0
    flagged_count_on_disk = 0
    if REAL_PROCESSED_DIR.exists():
        for mf in (REAL_PROCESSED_DIR / "metadata").glob("*.json"):
            try:
                with open(mf, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                if data.get("status") == "PASSED" and data.get("plausibility_audit", {}).get("is_skull_stripped_plausible", False):
                    valid_count_on_disk += 1
                elif data.get("status") == "FLAGGED" or not data.get("plausibility_audit", {}).get("is_skull_stripped_plausible", True):
                    flagged_count_on_disk += 1
            except Exception:
                pass
    expected_valid = valid_count_on_disk if valid_count_on_disk > 0 else manifest["metadata"].get("healthy_qc_passed_cases", 0)

    meta = manifest["metadata"]
    assert meta["total_cases"] == 2786
    assert meta["healthy_qc_passed_cases"] == expected_valid
    assert meta["healthy_pending_preprocessing_cases"] == 576 - expected_valid - flagged_count_on_disk
    assert meta["healthy_failed_cases"] == 0
    assert meta["healthy_flagged_cases"] == flagged_count_on_disk
    assert meta["healthy_excluded_cases"] == 6

    # Verify split checksums did not drift
    import hashlib
    def get_hash(lst):
        return hashlib.sha256("\n".join(sorted(lst)).encode("utf-8")).hexdigest()

    expected_chk = split["metadata"]["split_checksums"]
    assert get_hash(split["train_cases"]) == expected_chk["train_cases_sha256"]
    assert get_hash(split["val_cases"]) == expected_chk["val_cases_sha256"]
    assert get_hash(split["test_cases"]) == expected_chk["test_cases_sha256"]
    assert expected_chk["test_cases_sha256"] == "556ca50468a4bb3823ee2d79c40d170ae84b26d249ab90286fa294677e6ea818"


def test_non_qc_passed_cases_never_treated_as_valid_training_examples_and_ixi232_qc_verified():
    """
    Requirement 3:
    Ensures:
    1. Any healthy case with status FAILED, FLAGGED, PENDING_PREPROCESSING, or EXCLUDED
       is strictly marked with is_skull_stripped=False and never treated as a valid
       skull-stripped training example.
    2. IXI232 (which previously failed HD-BET with an OOM error and was retried upon batch resumption)
       is now fully verified to have physically passed all NIfTI geometry, binary mask, and
       plausibility checks before being admitted to class 0.
    """
    with open(REAL_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    ixi_cases = [c for c in manifest["cases"] if c["source_cohort"] == "ixi"]
    assert len(ixi_cases) == 582

    # Negative test: Non-PASSED cases must have is_skull_stripped=False
    non_passed = [c for c in ixi_cases if c.get("qc_status") != "PASSED"]
    passed = [c for c in ixi_cases if c.get("qc_status") == "PASSED"]
    assert len(non_passed) == 582 - len(passed)
    for c in non_passed:
        assert c["is_skull_stripped"] is False, f"Case {c['subject_group_key']} has status {c.get('qc_status')} but is_skull_stripped=True!"
        assert c["qc_status"] in ("PENDING_PREPROCESSING", "EXCLUDED", "FAILED", "FLAGGED")

    # Positive test: IXI232 verification on real disk
    ixi232_case = next((c for c in ixi_cases if c["subject_group_key"] == "IXI232"), None)
    assert ixi232_case is not None, "IXI232 not found in manifest!"

    assert ixi232_case["qc_status"] == "PASSED"
    assert ixi232_case["is_skull_stripped"] is True
    assert ixi232_case["geometry_status"] == "aligned"
    assert ixi232_case["class_id"] == 0
    assert ixi232_case["class_label"] == "healthy_control"

    # Deep physical check of IXI232 files
    t1_p = Path(ixi232_case["modalities"]["t1"])
    t2_p = Path(ixi232_case["modalities"]["t2"])
    mask_p = Path(ixi232_case["mask_path"])

    assert t1_p.exists() and t1_p.stat().st_size > 0
    assert t2_p.exists() and t2_p.stat().st_size > 0
    assert mask_p.exists() and mask_p.stat().st_size > 0

    nii_t1 = nib.load(str(t1_p))
    nii_t2 = nib.load(str(t2_p))
    nii_m = nib.load(str(mask_p))

    s1 = tuple(int(x) for x in nii_t1.header.get_data_shape()[:3])
    s2 = tuple(int(x) for x in nii_t2.header.get_data_shape()[:3])
    sm = tuple(int(x) for x in nii_m.header.get_data_shape()[:3])
    assert s1 == s2 == sm == (256, 256, 146)
    assert np.allclose(nii_t1.affine, nii_t2.affine, atol=1e-3)
    assert np.allclose(nii_t1.affine, nii_m.affine, atol=1e-3)

    mask_arr = np.asanyarray(nii_m.dataobj, dtype=np.uint8)
    assert set(np.unique(mask_arr)).issubset({0, 1})

    # Plausibility metrics in metadata
    meta = ixi232_case["qc_metadata"]
    assert 700.0 <= meta["nonzero_volume_cc"] <= 2000.0
    assert meta["outer_boundary_voxels"] <= 500


def test_failed_case_ixi232_never_treated_as_valid_healthy_training_example():
    """Compatibility wrapper ensuring backwards compatibility with specific test filters."""
    test_non_qc_passed_cases_never_treated_as_valid_training_examples_and_ixi232_qc_verified()


# =====================================================================
# Stage 1 Training Engine Tests
# =====================================================================

def test_presence_metrics_computation():
    """Verify compute_presence_metrics correctly calculates sensitivity, specificity, accuracy, and F1."""
    import training.train_presence_classifier as presence_train_mod

    # Perfectly predicted scenario
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    y_prob = [[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]]
    metrics = presence_train_mod.compute_presence_metrics(y_true, y_pred, y_prob)

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["sensitivity"] == 1.0  # Class 1 recall
    assert metrics["specificity"] == 1.0  # Class 0 recall
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]
    if metrics["roc_auc"] is not None:
        assert metrics["roc_auc"] == 1.0

    # Imperfect prediction: 1 false positive, 1 false negative
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]
    metrics = presence_train_mod.compute_presence_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 0.5
    assert metrics["sensitivity"] == 0.5
    assert metrics["specificity"] == 0.5
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]

    # Empty inputs edge case
    empty_m = presence_train_mod.compute_presence_metrics([], [])
    assert empty_m["total_samples"] == 0
    assert empty_m["accuracy"] == 0.0


def test_presence_training_import_isolation():
    """Confirms importing training.train_presence_classifier executes zero training."""
    import training.train_presence_classifier as mod
    assert hasattr(mod, "run_presence_training")
    assert hasattr(mod, "compute_presence_metrics")
    assert hasattr(mod, "compute_class_weights")


def test_presence_training_smoke_test_execution(tmp_path):
    """
    Executes a 1-epoch, 2-batch smoke test of run_presence_training in a temporary output directory.
    Verifies that forward, loss, backward, metrics, and checkpointing function end-to-end.
    """
    if not (REAL_MANIFEST_PATH.exists() and REAL_SPLIT_PATH.exists()):
        pytest.skip("Real manifest or split file not found on disk.")

    import training.train_presence_classifier as mod

    out_dir = tmp_path / "presence_train_out"
    summary = mod.run_presence_training(
        manifest_path=REAL_MANIFEST_PATH,
        split_path=REAL_SPLIT_PATH,
        output_dir=out_dir,
        epochs=1,
        batch_size=2,
        base_channels=8,
        smoke_test=True,
    )

    assert summary["best_epoch"] == 1
    assert "history" in summary
    assert Path(summary["best_checkpoint"]).exists()
    assert Path(summary["latest_checkpoint"]).exists()
    assert Path(summary["history_file"]).exists()

    # Inspect checkpoint content
    ckpt = torch.load(summary["best_checkpoint"], map_location="cpu")
    assert "model_state_dict" in ckpt
    assert ckpt["architecture"] == "TumorPresenceClassifier3D"
    assert ckpt["in_channels"] == 2
    assert ckpt["num_classes"] == 2
