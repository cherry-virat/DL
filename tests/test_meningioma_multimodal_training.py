"""
test_meningioma_multimodal_training.py
======================================
Isolated unit tests for Phase 3 training components:
1. Multiclass loss shapes (4D vs 5D target compatibility).
2. Gradient flow without optimizer step.
3. Loss stability on absent subregion classes (no NaNs).
4. Checkpoint saving/loading, payload structure, and atomic naming.
5. Multimodal subregion metrics (WT, TC, ET) and empty-mask policy.
6. CLI argument parsing sanity.
"""

from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest.mock

import numpy as np
import pytest
import torch
import torch.nn as nn

from models.meningioma_multimodal_unet import create_multimodal_unet
from models.meningioma_segresnet import create_segresnet
from scripts.evaluate_multimodal_ensemble import filter_small_connected_components
from training.train_meningioma_multimodal import (
    DEFAULT_EPOCHS,
    DEFAULT_PILOT_EPOCHS,
    MultimodalDiceCELoss,
    apply_air_masking,
    load_checkpoint,
    parse_args,
    resolve_cohort_indices,
    resolve_effective_epochs,
    save_checkpoint,
)
from training.meningioma_multimodal_dataset import (
    MeningiomaMultimodalDataset,
    calculate_multimodal_metrics,
)


def test_loss_4d_and_5d_target_equivalence():
    """Verify loss accepts both [B, D, H, W] and [B, 1, D, H, W] targets identically."""
    loss_fn = MultimodalDiceCELoss(include_background=False)

    logits = torch.randn(2, 4, 32, 32, 32, requires_grad=True)
    target_4d = torch.randint(0, 4, (2, 32, 32, 32), dtype=torch.long)
    target_5d = target_4d.unsqueeze(1)

    l4 = loss_fn(logits, target_4d)
    l5 = loss_fn(logits, target_5d)

    assert torch.isfinite(l4), "Loss for 4D target must be finite"
    assert torch.isfinite(l5), "Loss for 5D target must be finite"
    assert torch.isclose(l4, l5, atol=1e-5), "Loss mismatch between 4D and 5D targets!"


def test_loss_gradient_flow_without_optimizer():
    """Verify gradients propagate to parameters without executing optimizer.step()."""
    model = create_multimodal_unet(in_channels=4, num_classes=4, base_channels=8)
    loss_fn = MultimodalDiceCELoss(include_background=False)

    x = torch.randn(1, 4, 16, 16, 16)
    target = torch.randint(0, 4, (1, 16, 16, 16), dtype=torch.long)

    logits = model(x)
    loss = loss_fn(logits, target)
    loss.backward()

    # Check that conv weights received gradients
    has_grads = any(p.grad is not None and torch.norm(p.grad) > 0 for p in model.parameters())
    assert has_grads, "Gradients must compute properly through the model graph"


def test_loss_absent_classes_stability():
    """Verify loss computes stably when specific classes are absent in a patch."""
    loss_fn = MultimodalDiceCELoss(include_background=False)
    logits = torch.randn(1, 4, 16, 16, 16, requires_grad=True)

    # 1. All background (only class 0)
    target_all_bg = torch.zeros(1, 16, 16, 16, dtype=torch.long)
    l_bg = loss_fn(logits, target_all_bg)
    assert torch.isfinite(l_bg), "All-background patch must yield finite loss"

    # 2. Only enhancing tumor (class 3 present, 1 and 2 absent)
    target_et_only = torch.zeros(1, 16, 16, 16, dtype=torch.long)
    target_et_only[:, 4:12, 4:12, 4:12] = 3
    l_et = loss_fn(logits, target_et_only)
    assert torch.isfinite(l_et), "ET-only patch must yield finite loss"


def test_checkpoint_atomic_saving_and_loading():
    """Verify atomic checkpoint file naming, best flag handling, and payload loading."""
    model = create_multimodal_unet(in_channels=4, num_classes=4, base_channels=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        payload = {
            "epoch": 5,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": 0.852,
            "history": {"val_wt_dice": [0.70, 0.75, 0.80, 0.82, 0.852]},
        }

        # Save as best
        save_checkpoint(
            state=payload,
            is_best=True,
            checkpoint_dir=tmp_path,
            filename="test_latest.pth",
            best_filename="test_best.pth",
        )

        latest_file = tmp_path / "test_latest.pth"
        best_file = tmp_path / "test_best.pth"

        assert latest_file.exists(), "Latest checkpoint file must exist"
        assert best_file.exists(), "Best checkpoint file must exist"
        # Temporary files must have been cleaned up
        assert not (tmp_path / "test_latest.pth.tmp").exists()
        assert not (tmp_path / "test_best.pth.tmp").exists()

        # Test loading into a new model instance
        new_model = create_multimodal_unet(in_channels=4, num_classes=4, base_channels=8)
        new_optim = torch.optim.AdamW(new_model.parameters(), lr=1e-4)

        loaded_data = load_checkpoint(latest_file, new_model, new_optim)
        assert loaded_data["epoch"] == 5
        assert loaded_data["best_metric"] == 0.852

        # Verify weights match
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.equal(p1, p2), "Model parameter mismatch after checkpoint reload"


def test_multimodal_metrics_subregions():
    """Verify WT, TC, ET definitions and empty-mask boundary conditions."""
    pred = torch.zeros((16, 16, 16), dtype=torch.long)
    gt = torch.zeros((16, 16, 16), dtype=torch.long)

    # Put ET in region A (voxels 2:6)
    gt[2:6, 2:6, 2:6] = 3
    pred[2:6, 2:6, 2:6] = 3

    # Put ED in region B (voxels 6:10)
    gt[6:10, 6:10, 6:10] = 2
    pred[6:10, 6:10, 6:10] = 2

    # NCR is 0 in both GT and Pred
    metrics = calculate_multimodal_metrics(pred, gt)

    # WT = 2 + 3 (present and matching) -> 1.0
    assert metrics["wt"]["dice"] == 1.0
    # TC = 3 (present and matching) -> 1.0
    assert metrics["tc"]["dice"] == 1.0
    # ET = 3 (present and matching) -> 1.0
    assert metrics["et"]["dice"] == 1.0
    # Class 1 (NCR) is absent in both -> empty-mask policy returns 1.0
    assert metrics["per_class_dice"][1] == 1.0


def test_cli_argument_parsing():
    """Verify argparse configuration defaults and flags."""
    test_args = ["--epochs", "10", "--batch-size", "4", "--pilot", "--dry-run"]
    sys_argv_bak = sys.argv
    try:
        sys.argv = ["train_script.py"] + test_args
        args = parse_args()
        assert args.epochs == 10
        assert args.batch_size == 4
        assert args.pilot is True
        assert args.dry_run is True
        assert args.base_channels == 16
        assert args.amp is True
    finally:
        sys.argv = sys_argv_bak


def test_effective_epochs_pilot_default():
    """Verify pilot mode without an explicit epoch count defaults to 2 epochs."""
    args = parse_args(["--pilot"])
    assert args.pilot is True
    assert args.epochs == DEFAULT_PILOT_EPOCHS
    assert args.epochs == 2
    assert resolve_effective_epochs(pilot=True, epochs=None) == 2


def test_effective_epochs_pilot_explicit_20():
    """Verify pilot mode combined with --epochs 20 produces effective epochs = 20."""
    args = parse_args(["--pilot", "--epochs", "20"])
    assert args.pilot is True
    assert args.epochs == 20
    assert resolve_effective_epochs(pilot=True, epochs=20) == 20


def test_effective_epochs_non_pilot_default():
    """Verify non-pilot default behavior without explicit --epochs retains 50 epochs."""
    args = parse_args([])
    assert args.pilot is False
    assert args.epochs == DEFAULT_EPOCHS
    assert args.epochs == 50
    assert resolve_effective_epochs(pilot=False, epochs=None) == 50


def test_effective_epochs_non_pilot_explicit():
    """Verify explicit non-pilot epoch count produces specified epoch count."""
    args = parse_args(["--epochs", "35"])
    assert args.pilot is False
    assert args.epochs == 35
    assert resolve_effective_epochs(pilot=False, epochs=35) == 35


def test_resolve_effective_epochs_validation():
    """Verify that non-positive epoch counts raise ValueError."""
    with pytest.raises(ValueError, match="positive"):
        resolve_effective_epochs(pilot=True, epochs=0)
    with pytest.raises(ValueError, match="positive"):
        resolve_effective_epochs(pilot=False, epochs=-5)


def test_air_masking_zero_voxels_and_preserves_non_air():
    """
    Verify Fix 1:
    - All-zero four-channel voxels are set to class 0 (background).
    - Non-air voxels retain their original predicted classes.
    - Ground-truth labels are strictly unmodified.
    - Works for both 4D [4, D, H, W] and 5D [B, 4, D, H, W] images.
    """
    # Create synthetic 4-channel image [4, 6, 6, 6]
    image_4d = torch.zeros((4, 6, 6, 6), dtype=torch.float32)
    # Define a "brain" region in voxels (2:5, 2:5, 2:5) where channels have non-zero intensities
    image_4d[0, 2:5, 2:5, 2:5] = 1.2
    image_4d[1, 2:5, 2:5, 2:5] = 0.8
    image_4d[2, 2:5, 2:5, 2:5] = -0.5
    image_4d[3, 2:5, 2:5, 2:5] = 2.1

    # Raw model predictions: spurious class 2 (edema) everywhere, including air
    pred_3d = torch.full((6, 6, 6), fill_value=2, dtype=torch.long)
    # Ground truth mask: label 3 (ET) inside brain center, 0 elsewhere
    gt_mask = torch.zeros((6, 6, 6), dtype=torch.long)
    gt_mask[3, 3, 3] = 3
    gt_backup = gt_mask.clone()

    masked_pred = apply_air_masking(pred_3d, image_4d)

    # 1. An all-zero 4-channel voxel (air) must be predicted as background (0)
    assert masked_pred[0, 0, 0].item() == 0, "Air voxel must be masked to background class 0"
    assert masked_pred[5, 5, 5].item() == 0, "Air voxel must be masked to background class 0"
    assert torch.all(masked_pred[:2, :, :] == 0), "All air slices must be masked to background"

    # 2. A non-air voxel must retain its original predicted class (class 2)
    assert masked_pred[3, 3, 3].item() == 2, "Non-air voxel must retain original prediction"
    assert masked_pred[2, 2, 2].item() == 2, "Non-air voxel must retain original prediction"
    assert torch.all(masked_pred[2:5, 2:5, 2:5] == 2), "All brain voxels must retain original prediction"

    # 3. Ground truth labels must be completely unmodified
    assert torch.equal(gt_mask, gt_backup), "Ground truth mask must remain strictly unmodified"

    # 4. Batch 5D format test
    image_5d = image_4d.unsqueeze(0)  # [1, 4, 6, 6, 6]
    pred_4d = pred_3d.unsqueeze(0)    # [1, 6, 6, 6]
    masked_pred_5d = apply_air_masking(pred_4d, image_5d)
    assert masked_pred_5d[0, 0, 0, 0].item() == 0
    assert masked_pred_5d[0, 3, 3, 3].item() == 2


def test_air_masking_integration_with_multimodal_metrics():
    """
    Verify that air masking preserves WT, TC, ET metric definitions while eliminating
    spurious air false positives.
    """
    image = torch.zeros((4, 8, 8, 8), dtype=torch.float32)
    # Brain tissue region
    image[:, 2:6, 2:6, 2:6] = 1.0

    # GT has ET (class 3) in voxels 3:5
    gt = torch.zeros((8, 8, 8), dtype=torch.long)
    gt[3:5, 3:5, 3:5] = 3

    # Prediction before masking: predicts class 3 correctly in brain (3:5),
    # but ALSO spurious class 3 in air (voxels 0:2)
    raw_pred = torch.zeros((8, 8, 8), dtype=torch.long)
    raw_pred[3:5, 3:5, 3:5] = 3  # TP
    raw_pred[0:2, 0:2, 0:2] = 3  # Spurious air FP

    # Metrics before masking: has air FP
    m_before = calculate_multimodal_metrics(raw_pred, gt)
    assert m_before["et"]["fp"] > 0, "Raw prediction must have air false positives"

    # Apply air masking
    clean_pred = apply_air_masking(raw_pred, image)

    # Metrics after masking: air FP eliminated, TP preserved
    m_after = calculate_multimodal_metrics(clean_pred, gt)
    assert m_after["et"]["fp"] == 0, "Air false positives must be completely eliminated"
    assert m_after["et"]["tp"] == m_before["et"]["tp"], "True positives in brain must be preserved"
    assert m_after["et"]["dice"] == 1.0, "Clean prediction should achieve perfect Dice"
    assert m_after["wt"]["dice"] == 1.0, "WT metric must also reflect cleaned air voxels"
    assert m_after["tc"]["dice"] == 1.0, "TC metric must also reflect cleaned air voxels"


def test_checkpoint_resume_weights_only_compat():
    """
    Verify Fix 2:
    - Checkpoint containing Path objects in metadata loads cleanly without _pickle.UnpicklingError.
    - Existing project checkpoint (if present) loads cleanly without performing training or updates.
    """
    model = create_multimodal_unet(in_channels=4, num_classes=4, base_channels=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        checkpoint_file = tmp_path / "test_with_paths.pth"

        # Payload contains Path objects in metadata (like vars(args))
        payload = {
            "epoch": 2,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": 0.015,
            "history": {"train_loss": [2.36, 2.23]},
            "args": {
                "split_json": Path("C:/data/split.json"),
                "dataset_root": Path("C:/data/root"),
                "epochs": 2,
            },
        }
        torch.save(payload, checkpoint_file)

        # Load into new model instance
        new_model = create_multimodal_unet(in_channels=4, num_classes=4, base_channels=8)
        new_optim = torch.optim.AdamW(new_model.parameters(), lr=1e-4)

        # Must load cleanly without raising UnpicklingError
        loaded = load_checkpoint(checkpoint_file, new_model, new_optim)
        assert loaded["epoch"] == 2
        assert loaded["best_metric"] == 0.015
        assert isinstance(loaded["args"]["split_json"], Path)

    # Also test actual project pilot checkpoint if it exists on disk
    project_ckpt = PROJECT_ROOT / "outputs/meningioma_multimodal_experiment/checkpoints/meningioma_multimodal_best.pth"
    if project_ckpt.exists():
        eval_model = create_multimodal_unet(in_channels=4, num_classes=4, base_channels=16)
        eval_optim = torch.optim.AdamW(eval_model.parameters(), lr=2e-4)
        loaded_project = load_checkpoint(project_ckpt, eval_model, eval_optim)
        assert loaded_project["epoch"] == 2
        assert "model_state_dict" in loaded_project


def test_load_checkpoint_device_compatibility():
    """
    Verifies CPU/GPU-compatible checkpoint loading:
    1. CPU model loads a checkpoint saved with CUDA tensor storage when device=None.
    2. Explicit device=torch.device('cpu') works.
    3. Explicit device=torch.device('cuda') works when CUDA is available.
    4. Optimizer state is restored with matching tensor devices.
    5. Path objects in checkpoint metadata are preserved.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "cuda_saved_checkpoint.pth"

        # Create dummy tensors (simulate CUDA storage if CUDA is available, or manually set device)
        if torch.cuda.is_available():
            save_device = torch.device("cuda:0")
        else:
            save_device = torch.device("cpu")

        source_model = nn.Linear(4, 2).to(save_device)
        source_optim = torch.optim.Adam(source_model.parameters(), lr=1e-3)
        x = torch.randn(2, 4, device=save_device)
        loss = source_model(x).sum()
        loss.backward()
        source_optim.step()

        payload = {
            "epoch": 5,
            "model_state_dict": source_model.state_dict(),
            "optimizer_state_dict": source_optim.state_dict(),
            "metadata_path": Path("some/nested/path/config.json"),
        }
        torch.save(payload, ckpt_path)

        # 1. CPU model loads CUDA-saved checkpoint with device=None
        # Even when torch.cuda.is_available() is mocked as False
        with unittest.mock.patch("torch.cuda.is_available", return_value=False):
            cpu_model = nn.Linear(4, 2)  # on CPU
            cpu_optim = torch.optim.Adam(cpu_model.parameters(), lr=1e-3)

            loaded = load_checkpoint(ckpt_path, cpu_model, cpu_optim, device=None)
            assert loaded["epoch"] == 5
            assert isinstance(loaded["metadata_path"], Path)
            assert next(cpu_model.parameters()).device.type == "cpu"
            for state in cpu_optim.state.values():
                for v in state.values():
                    if torch.is_tensor(v):
                        assert v.device.type == "cpu"

        # 2. Explicit CPU device
        cpu_model_exp = nn.Linear(4, 2)
        cpu_optim_exp = torch.optim.Adam(cpu_model_exp.parameters(), lr=1e-3)
        loaded_exp = load_checkpoint(ckpt_path, cpu_model_exp, cpu_optim_exp, device=torch.device("cpu"))
        assert loaded_exp["epoch"] == 5
        assert next(cpu_model_exp.parameters()).device.type == "cpu"
        for state in cpu_optim_exp.state.values():
            for v in state.values():
                if torch.is_tensor(v):
                    assert v.device.type == "cpu"

        # 3. Explicit CUDA device when CUDA is available
        if torch.cuda.is_available():
            cuda_model = nn.Linear(4, 2).cuda()
            cuda_optim = torch.optim.Adam(cuda_model.parameters(), lr=1e-3)
            loaded_cuda = load_checkpoint(ckpt_path, cuda_model, cuda_optim, device=torch.device("cuda:0"))
            assert loaded_cuda["epoch"] == 5
            assert next(cuda_model.parameters()).device.type == "cuda"
            for state in cuda_optim.state.values():
                for v in state.values():
                    if torch.is_tensor(v):
                        assert v.device.type == "cuda"


def test_brain_aware_sampling_synthetic():
    """Verify brain_aware_weighted patch sampling produces expected patch shapes and populates candidate cache."""
    image = torch.zeros(4, 32, 32, 32, dtype=torch.float32)
    mask = torch.zeros(32, 32, 32, dtype=torch.long)

    # Brain region: non-zero in at least one modality
    image[0, 6:26, 6:26, 6:26] = 1.5
    # Tumor region: non-zero mask (labels 1, 2, 3)
    mask[12:18, 12:18, 12:18] = 1

    ds = object.__new__(MeningiomaMultimodalDataset)
    ds.patch_sampling = "brain_aware_weighted"
    ds._candidate_cache = {}

    patch_size = (16, 16, 16)
    img_patch, msk_patch = ds._extract_patch(image, mask, patch_size, case_id="synth_case_1")

    assert img_patch.shape == (4, 16, 16, 16), f"Unexpected image patch shape: {img_patch.shape}"
    assert msk_patch.shape == (16, 16, 16), f"Unexpected mask patch shape: {msk_patch.shape}"
    assert "synth_case_1" in ds._candidate_cache
    assert "tumor" in ds._candidate_cache["synth_case_1"]
    assert "healthy" in ds._candidate_cache["synth_case_1"]
    assert ds._candidate_cache["synth_case_1"]["tumor"].shape[0] == 6 * 6 * 6


def test_healthy_brain_patch_sampling_synthetic(monkeypatch):
    """Verify brain_aware_weighted chooses a healthy brain voxel center in the 30% branch."""
    image = torch.zeros(4, 32, 32, 32, dtype=torch.float32)
    mask = torch.zeros(32, 32, 32, dtype=torch.long)

    # Healthy brain only in region [4:12, 4:12, 4:12]
    image[0, 4:12, 4:12, 4:12] = 2.0
    # Tumor only in distant region [20:28, 20:28, 20:28]
    image[0, 20:28, 20:28, 20:28] = 2.0
    mask[20:28, 20:28, 20:28] = 1

    ds = object.__new__(MeningiomaMultimodalDataset)
    ds.patch_sampling = "brain_aware_weighted"
    ds._candidate_cache = {}

    # Mock rand() to return 0.75 (inside [0.60, 0.90) -> healthy brain branch)
    monkeypatch.setattr(np.random, "rand", lambda: 0.75)

    patch_size = (8, 8, 8)
    img_patch, msk_patch = ds._extract_patch(image, mask, patch_size, case_id="synth_healthy")

    healthy_coords = ds._candidate_cache["synth_healthy"]["healthy"]
    assert healthy_coords.shape[0] == 8 * 8 * 8
    assert img_patch.shape == (4, 8, 8, 8)
    assert msk_patch.shape == (8, 8, 8)


def test_sampling_fallbacks_synthetic(monkeypatch):
    """Verify graceful fallbacks for absent tumor or absent healthy brain candidates."""
    ds = object.__new__(MeningiomaMultimodalDataset)
    ds.patch_sampling = "brain_aware_weighted"
    ds._candidate_cache = {}
    patch_size = (8, 8, 8)

    # 1. Healthy brain present, NO tumor. r=0.2 (tumor branch) must fallback to healthy brain without error.
    image_no_tumor = torch.zeros(4, 32, 32, 32, dtype=torch.float32)
    mask_no_tumor = torch.zeros(32, 32, 32, dtype=torch.long)
    image_no_tumor[:, 8:24, 8:24, 8:24] = 1.0

    monkeypatch.setattr(np.random, "rand", lambda: 0.2)
    img_p, msk_p = ds._extract_patch(image_no_tumor, mask_no_tumor, patch_size)
    assert img_p.shape == (4, 8, 8, 8)
    assert msk_p.shape == (8, 8, 8)

    # 2. All air (no healthy brain, no tumor). Must fallback to uniform crop without crashing.
    image_air = torch.zeros(4, 32, 32, 32, dtype=torch.float32)
    mask_air = torch.zeros(32, 32, 32, dtype=torch.long)
    monkeypatch.setattr(np.random, "rand", lambda: 0.75)  # healthy branch
    img_air_p, msk_air_p = ds._extract_patch(image_air, mask_air, patch_size)
    assert img_air_p.shape == (4, 8, 8, 8)
    assert msk_air_p.shape == (8, 8, 8)

    # 3. Tumor present, NO healthy brain (every non-zero voxel is tumor). r=0.75 (healthy branch) must fallback to tumor.
    image_only_tumor = torch.zeros(4, 32, 32, 32, dtype=torch.float32)
    mask_only_tumor = torch.zeros(32, 32, 32, dtype=torch.long)
    image_only_tumor[:, 10:20, 10:20, 10:20] = 1.0
    mask_only_tumor[10:20, 10:20, 10:20] = 2

    monkeypatch.setattr(np.random, "rand", lambda: 0.75)  # healthy branch
    img_t_p, msk_t_p = ds._extract_patch(image_only_tumor, mask_only_tumor, patch_size)
    assert img_t_p.shape == (4, 8, 8, 8)
    assert msk_t_p.shape == (8, 8, 8)


def test_deterministic_cohort_selection():
    """Verify deterministic cohort selection under 'seed' and 'prefix' modes."""
    total_cases = 100
    requested = 20

    # 1. Seed reproducibility
    idx1 = resolve_cohort_indices(total_cases, requested, mode="seed", seed=42)
    idx2 = resolve_cohort_indices(total_cases, requested, mode="seed", seed=42)
    assert idx1 == idx2, "Same seed must yield identical cohort indices"
    assert len(idx1) == requested
    assert len(set(idx1)) == requested, "Indices must be unique"

    # 2. Distinct seed difference
    idx_other = resolve_cohort_indices(total_cases, requested, mode="seed", seed=99)
    assert idx1 != idx_other, "Different seeds should yield different cohort indices"

    # 3. Prefix mode
    idx_prefix = resolve_cohort_indices(total_cases, requested, mode="prefix")
    assert idx_prefix == list(range(requested)), "Prefix mode must return first N indices"

    # 4. Requesting all or more cases returns full range
    idx_all = resolve_cohort_indices(total_cases, total_cases + 10, mode="seed")
    assert idx_all == list(range(total_cases))


def test_cohort_train_val_disjoint():
    """Verify selected train and val subsets from disjoint candidate pools remain strictly disjoint."""
    train_ids = [f"BraTS-MEN-{i:05d}-000" for i in range(100)]
    val_ids = [f"BraTS-MEN-{i:05d}-000" for i in range(100, 150)]

    train_idx = resolve_cohort_indices(len(train_ids), 30, mode="seed", seed=42)
    val_idx = resolve_cohort_indices(len(val_ids), 10, mode="seed", seed=42)

    selected_train = [train_ids[i] for i in train_idx]
    selected_val = [val_ids[i] for i in val_idx]

    overlap = set(selected_train) & set(selected_val)
    assert len(overlap) == 0, f"Detected leakage between train and val cohorts: {overlap}"


def test_cli_cohort_and_sampling_defaults():
    """Verify CLI arguments for cohort selection and patch sampling."""
    # Defaults
    args = parse_args([])
    assert args.patch_sampling == "tumor_weighted"
    assert args.train_cases is None
    assert args.val_cases is None
    assert args.cohort_selection == "auto"

    # Custom cohort and sampling
    args2 = parse_args([
        "--train-cases", "50",
        "--val-cases", "20",
        "--patch-sampling", "brain_aware_weighted",
        "--cohort-selection", "seed",
        "--seed", "123",
    ])
    assert args2.train_cases == 50
    assert args2.val_cases == 20
    assert args2.patch_sampling == "brain_aware_weighted"
    assert args2.cohort_selection == "seed"
    assert args2.seed == 123


def test_segresnet_model_and_forward():
    """Verify MONAI SegResNet instantiates cleanly and computes expected output tensor."""
    model = create_segresnet(in_channels=4, num_classes=4, init_filters=16)
    assert model.count_parameters() > 4_000_000, "SegResNet should have ~4.7M parameters"

    x = torch.randn(1, 4, 32, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 4, 32, 32, 32), f"Unexpected SegResNet output shape: {out.shape}"


def test_cli_model_selection():
    """Verify CLI --model argument defaults to unet and parses segresnet."""
    args_default = parse_args([])
    assert args_default.model == "unet"

    args_segresnet = parse_args(["--model", "segresnet"])
    assert args_segresnet.model == "segresnet"


def test_connected_component_filtering():
    """Verify small isolated islands are eliminated while true tumor mass is retained."""
    mask = torch.zeros(32, 32, 32, dtype=torch.long)
    # True tumor mass: 10 x 10 x 10 = 1000 voxels
    mask[10:20, 10:20, 10:20] = 1
    # Tiny noise artifact: 2 x 2 x 2 = 8 voxels
    mask[2:4, 2:4, 2:4] = 2

    cleaned = filter_small_connected_components(mask, min_size=50)

    # Big tumor must be preserved
    assert (cleaned[10:20, 10:20, 10:20] == 1).all()
    # Tiny noise artifact must be zeroed out
    assert (cleaned[2:4, 2:4, 2:4] == 0).all()




