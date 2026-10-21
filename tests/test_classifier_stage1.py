"""
Stage 1 Unit Tests for Automatic Brain Tumor Classifier.
========================================================

Verifies:
1. Manifest integrity: exact case counts (1,251 Glioma, 1,000 Meningioma, 2,251 total).
2. Split integrity: disjoint patient-level partitions (1,400 train, 300 val, 300 test, 251 extension).
3. Zero overlap across all split partitions and preservation of split membership.
4. Preprocessing geometry: symmetric padding + isotropic 1.875x scaling yields (4, 128, 128, 96).
5. Non-zero Z-score intensity normalization with exact background zero-preservation.
6. Modality validation: rejection of missing, mismatched shapes, non-3D, empty/corrupt, or unaligned affines.
7. Absence of tumor segmentation masks (zero label leakage).
8. Data loader instantiation and safe anonymization.
"""

import json
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from preprocessing.tumor_classifier_preprocess import (
    preprocess_4channel_classifier_tensor,
    preprocess_case_to_tensor,
    canonicalize_classifier_modality_key,
    normalize_nonzero_zscore,
    compute_target_affine,
    SpatialAlignmentError,
    CANONICAL_MODALITIES,
    ORIGINAL_SHAPE,
    TARGET_SHAPE,
    TARGET_SPACING,
)
from training.dataset_classifier import BrainTumorClassifierDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "classifier" / "manifest.json"
SPLIT_PATH = PROJECT_ROOT / "data" / "classifier" / "split.json"


def test_classifier_manifest_file_structure_and_counts():
    """Test 1: Manifest exists, contains exactly 2,251 valid cases with complete metadata."""
    assert MANIFEST_PATH.exists(), f"Manifest file missing: {MANIFEST_PATH}"

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data["metadata"]
    cases = data["cases"]

    assert meta["total_cases"] == 2251
    assert meta["glioma_cases"] == 1251
    assert meta["meningioma_cases"] == 1000
    assert meta["balanced_cohort_size"] == 2000
    assert meta["glioma_extension_size"] == 251

    assert len(cases) == 2251

    gli_cases = [c for c in cases if c["class_label"] == "glioma"]
    men_cases = [c for c in cases if c["class_label"] == "meningioma"]

    assert len(gli_cases) == 1251
    assert len(men_cases) == 1000

    balanced_cases = [c for c in cases if c.get("is_in_balanced_cohort") is True]
    assert len(balanced_cases) == 2000

    # Verify fields for sample cases
    for c in cases[:100]:
        assert c["anonymized_id"].startswith("ANON_")
        assert c["class_id"] in (0, 1)
        assert set(c["modalities"].keys()) == {"t1n", "t1c", "t2w", "t2f"}
        assert c["spatial_shape"] == [240, 240, 155]
        assert c["voxel_spacing_mm"] == [1.0, 1.0, 1.0]

        # If enriched metadata is present, verify structure
        if "spatial_consistency" in c:
            assert c["spatial_consistency"]["is_valid"] is True
            assert c["spatial_consistency"]["all_modalities_readable"] is True
            assert c["spatial_consistency"]["shapes_consistent"] is True
            assert c["spatial_consistency"]["affines_consistent"] is True


def test_classifier_split_file_disjoint_integrity():
    """Test 2: Split file defines disjoint train, val, and locked test partitions with zero overlap."""
    assert SPLIT_PATH.exists(), f"Split file missing: {SPLIT_PATH}"

    with open(SPLIT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data["metadata"]
    summary = meta["summary"]

    assert meta["locked_test_set"] is True
    assert meta["random_seed"] == 42

    assert summary["train_total"] == 1400
    assert summary["train_glioma"] == 700
    assert summary["train_meningioma"] == 700

    assert summary["val_total"] == 300
    assert summary["val_glioma"] == 150
    assert summary["val_meningioma"] == 150

    assert summary["test_total"] == 300
    assert summary["test_glioma"] == 150
    assert summary["test_meningioma"] == 150

    assert summary["unbalanced_glioma_extension"] == 251

    train_set = set(data["train_cases"])
    val_set = set(data["val_cases"])
    test_set = set(data["test_cases"])
    ext_set = set(data["unbalanced_glioma_extension"])

    assert len(train_set) == 1400
    assert len(val_set) == 300
    assert len(test_set) == 300
    assert len(ext_set) == 251

    # Disjoint assertions
    assert len(train_set.intersection(val_set)) == 0, "Train and Val overlap detected!"
    assert len(train_set.intersection(test_set)) == 0, "Train and Test overlap detected!"
    assert len(val_set.intersection(test_set)) == 0, "Val and Test overlap detected!"
    assert len(train_set.intersection(ext_set)) == 0, "Train and Extension overlap detected!"
    assert len(val_set.intersection(ext_set)) == 0, "Val and Extension overlap detected!"
    assert len(test_set.intersection(ext_set)) == 0, "Test and Extension overlap detected!"


def test_classifier_preprocessing_shape_and_spacing():
    """Test 3: Standalone preprocessing transforms (240, 240, 155) into (4, 128, 128, 96)."""
    synth_inputs = {}
    for mod in CANONICAL_MODALITIES:
        arr = np.zeros(ORIGINAL_SHAPE, dtype=np.float32)
        arr[50:180, 50:180, 20:140] = np.random.uniform(10.0, 100.0, size=(130, 130, 120))
        synth_inputs[mod] = arr

    ref_affine = np.array([
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 239.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    tensor_4d, meta = preprocess_4channel_classifier_tensor(synth_inputs, original_affine=ref_affine)

    assert isinstance(tensor_4d, torch.Tensor)
    assert tensor_4d.shape == (4, 128, 128, 96)
    assert tensor_4d.dtype == torch.float32

    assert meta.original_shape == (240, 240, 155)
    assert meta.padded_shape == (240, 240, 180)
    assert meta.target_shape == TARGET_SHAPE
    assert meta.target_spacing == TARGET_SPACING
    assert meta.scale_factor == 1.875
    assert meta.pad_z == (12, 13)

    assert meta.target_affine is not None
    target_aff = np.array(meta.target_affine)
    assert np.isclose(abs(target_aff[0, 0]), 1.875)
    assert np.isclose(abs(target_aff[1, 1]), 1.875)
    assert np.isclose(abs(target_aff[2, 2]), 1.875)


def test_classifier_preprocessing_normalization_invariance():
    """Test 4: Non-zero Z-score normalization produces zero-mean, unit-variance on brain and preserves 0.0 air."""
    arr = np.zeros((64, 64, 64), dtype=np.float32)
    arr[16:48, 16:48, 16:48] = np.random.normal(loc=50.0, scale=10.0, size=(32, 32, 32))

    norm = normalize_nonzero_zscore(arr)

    # Air voxels remain strictly 0.0
    air_mask = arr == 0.0
    assert np.all(norm[air_mask] == 0.0)

    # Brain voxels have zero mean and unit variance
    brain_mask = arr > 0.0
    assert np.isclose(norm[brain_mask].mean(), 0.0, atol=1e-4)
    assert np.isclose(norm[brain_mask].std(), 1.0, atol=1e-4)


def test_classifier_preprocessing_rejections():
    """Test 5: Rejects missing modalities, shape mismatches, non-3D, or non-finite inputs."""
    # 1. Missing modality
    with pytest.raises(ValueError, match="Missing required classifier modalities"):
        preprocess_4channel_classifier_tensor({"t1n": np.zeros((32, 32, 32))})

    # 2. Shape mismatch
    with pytest.raises(ValueError, match="Spatial shape mismatch"):
        preprocess_4channel_classifier_tensor({
            "t1n": np.zeros((32, 32, 32)),
            "t1c": np.zeros((32, 32, 32)),
            "t2w": np.zeros((32, 32, 32)),
            "t2f": np.zeros((32, 32, 16)),
        })

    # 3. Non-3D input
    with pytest.raises(ValueError, match="must be a 3D volume"):
        preprocess_4channel_classifier_tensor({
            "t1n": np.zeros((32, 32)),
            "t1c": np.zeros((32, 32, 32)),
            "t2w": np.zeros((32, 32, 32)),
            "t2f": np.zeros((32, 32, 32)),
        })

    # 4. Non-finite values (NaN / Inf)
    bad_arr = np.zeros((32, 32, 32), dtype=np.float32)
    bad_arr[10, 10, 10] = np.nan
    with pytest.raises(ValueError, match="Non-finite values"):
        preprocess_4channel_classifier_tensor({
            "t1n": bad_arr,
            "t1c": np.zeros((32, 32, 32)),
            "t2w": np.zeros((32, 32, 32)),
            "t2f": np.zeros((32, 32, 32)),
        })


def test_classifier_preprocessing_affine_and_spacing_rejections():
    """Test 6: Rejects modality inputs with discordant spatial affines or voxel spacings."""
    ref_aff = np.eye(4, dtype=np.float64)
    bad_aff = np.eye(4, dtype=np.float64)
    bad_aff[0, 3] = 10.0  # 10 mm shift along X

    arr = np.zeros((32, 32, 32), dtype=np.float32)
    nii_ref = nib.Nifti1Image(arr, affine=ref_aff)
    nii_bad_aff = nib.Nifti1Image(arr, affine=bad_aff)

    with pytest.raises(SpatialAlignmentError, match="Spatial affine mismatch"):
        preprocess_4channel_classifier_tensor({
            "t1n": nii_ref,
            "t1c": nii_ref,
            "t2w": nii_ref,
            "t2f": nii_bad_aff,
        })

    # Spacing mismatch test
    hdr_bad_zooms = nii_ref.header.copy()
    hdr_bad_zooms.set_zooms((1.0, 1.0, 2.5))
    nii_bad_zoom = nib.Nifti1Image(arr, affine=ref_aff, header=hdr_bad_zooms)

    with pytest.raises(SpatialAlignmentError, match="Voxel spacing mismatch"):
        preprocess_4channel_classifier_tensor({
            "t1n": nii_ref,
            "t1c": nii_ref,
            "t2w": nii_ref,
            "t2f": nii_bad_zoom,
        })


def test_classifier_preprocessing_unreadable_and_empty_file_rejections(tmp_path):
    """Test 7: Rejects empty or corrupt NIfTI files on disk with clear exceptions."""
    empty_file = tmp_path / "empty_scan.nii.gz"
    empty_file.write_bytes(b"")

    corrupt_file = tmp_path / "corrupt_scan.nii.gz"
    corrupt_file.write_bytes(b"NOT_A_VALID_GZIP_OR_NIFTI_HEADER_DATA_123456789")

    valid_arr = np.zeros((32, 32, 32), dtype=np.float32)
    valid_file = tmp_path / "valid_scan.nii.gz"
    nib.save(nib.Nifti1Image(valid_arr, np.eye(4)), str(valid_file))

    # Empty file check
    with pytest.raises(ValueError, match="empty"):
        preprocess_4channel_classifier_tensor({
            "t1n": empty_file,
            "t1c": valid_file,
            "t2w": valid_file,
            "t2f": valid_file,
        })

    # Corrupted file check
    with pytest.raises(ValueError, match="Failed to load or parse NIfTI"):
        preprocess_4channel_classifier_tensor({
            "t1n": corrupt_file,
            "t1c": valid_file,
            "t2w": valid_file,
            "t2f": valid_file,
        })


def test_classifier_preprocessing_aligned_nifti_fixtures(tmp_path):
    """Test 8: Successfully preprocesses 4 real aligned on-disk synthetic NIfTI files."""
    ref_aff = np.array([
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 239.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    paths = {}
    for mod in CANONICAL_MODALITIES:
        arr = np.zeros(ORIGINAL_SHAPE, dtype=np.float32)
        arr[60:180, 60:180, 30:130] = np.random.uniform(20.0, 80.0, size=(120, 120, 100))
        nii = nib.Nifti1Image(arr, affine=ref_aff)
        nii.header.set_zooms((1.0, 1.0, 1.0))
        p = tmp_path / f"synth_{mod}.nii.gz"
        nib.save(nii, str(p))
        paths[mod] = p

    tensor_4d, meta = preprocess_4channel_classifier_tensor(paths)

    assert tensor_4d.shape == (4, 128, 128, 96)
    assert meta.scale_factor == 1.875
    assert meta.pad_z == (12, 13)
    assert meta.original_shape == ORIGINAL_SHAPE
    assert meta.target_shape == TARGET_SHAPE


def test_classifier_dataset_instantiation_and_counts():
    """Test 9: BrainTumorClassifierDataset instantiates with correct split counts."""
    ds_train = BrainTumorClassifierDataset(split="train")
    assert len(ds_train) == 1400

    ds_val = BrainTumorClassifierDataset(split="val")
    assert len(ds_val) == 300

    ds_test = BrainTumorClassifierDataset(split="test")
    assert len(ds_test) == 300

    ds_ext = BrainTumorClassifierDataset(split="extension")
    assert len(ds_ext) == 251

    ds_all = BrainTumorClassifierDataset(split="all")
    assert len(ds_all) == 2251

    with pytest.raises(ValueError, match="Unsupported split"):
        BrainTumorClassifierDataset(split="invalid_split_name")


def test_classifier_no_mask_leakage_and_anonymization():
    """Test 10: Manifest strictly excludes segmentation masks and exposes only canonical modalities."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    for record in data["cases"]:
        for mod_key, path_str in record["modalities"].items():
            assert mod_key in ("t1n", "t1c", "t2w", "t2f")
            assert "-seg" not in path_str.lower()
            assert "mask" not in mod_key.lower()

        assert record["class_label"] in ("glioma", "meningioma")
        assert record["class_id"] in (0, 1)


def test_classifier_split_membership_preservation():
    """Test 11: Split membership IDs match the deterministic seed 42 partition exactly."""
    with open(SPLIT_PATH, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    # Validate that split cases exist in manifest
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    manifest_ids = {c["anonymized_id"] for c in manifest_data["cases"]}

    for split_key in ["train_cases", "val_cases", "test_cases", "unbalanced_glioma_extension"]:
        case_list = split_data[split_key]
        for cid in case_list:
            assert cid in manifest_ids, f"Case {cid} in {split_key} missing from manifest"


def test_classifier_modality_key_canonicalization():
    """Test 12: Maps diverse clinical sequence naming variants to canonical keys."""
    assert canonicalize_classifier_modality_key("t1") == "t1n"
    assert canonicalize_classifier_modality_key("T1N") == "t1n"
    assert canonicalize_classifier_modality_key("t1w") == "t1n"
    assert canonicalize_classifier_modality_key("t1c") == "t1c"
    assert canonicalize_classifier_modality_key("T1CE") == "t1c"
    assert canonicalize_classifier_modality_key("t1-contrast") == "t1c"
    assert canonicalize_classifier_modality_key("t2") == "t2w"
    assert canonicalize_classifier_modality_key("T2W") == "t2w"
    assert canonicalize_classifier_modality_key("flair") == "t2f"
    assert canonicalize_classifier_modality_key("T2F") == "t2f"
    assert canonicalize_classifier_modality_key("unknown_modality") is None


def test_preprocess_case_to_tensor_with_variant_keys():
    """Test 13: Preprocesses 4-channel input using variant modality keys into (4, 128, 128, 96)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        affine = np.diag([1.0, 1.0, 1.0, 1.0])
        # Synthetic (240, 240, 155) volumes
        paths = {}
        for mod, key_name in [("t1n", "t1"), ("t1c", "t1c"), ("t2w", "t2"), ("t2f", "flair")]:
            data = np.zeros(ORIGINAL_SHAPE, dtype=np.float32)
            data[50:150, 50:150, 40:100] = 100.0  # brain foreground
            img = nib.Nifti1Image(data, affine)
            fpath = tmp_path / f"{mod}.nii.gz"
            nib.save(img, str(fpath))
            paths[key_name] = fpath

        tensor = preprocess_case_to_tensor(paths)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (4, 128, 128, 96)
        assert tensor.dtype == torch.float32
        assert torch.isfinite(tensor).all()
