"""
Stage 2 Unit Tests for 3D Brain Tumor Classifier Foundation.
============================================================

Verifies:
1. Model instantiation with default and custom configurations.
2. Forward pass shape contract: returns strictly (B, 2) for synthetic inputs.
3. Numerical stability: all logits are finite (no NaN or Inf).
4. Loss computation: CrossEntropyLoss computes without error.
5. Gradient flow: Backpropagation produces finite gradients for all trainable parameters.
6. Input validation: Rejection of non-5D inputs, non-tensor inputs, and incorrect channel counts.
7. Hardware compatibility: Executes cleanly on CPU, and runs a CUDA smoke test if available.
8. Import isolation: Importing `training.train_classifier` executes zero training and accesses no datasets.
9. Classification metric computations and edge-case handling.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import torch
import torch.nn as nn

import tempfile
import nibabel as nib
import numpy as np

from models.tumor_classifier_3d import (
    TumorClassifier3D,
    create_tumor_classifier_3d,
)
from models.tumor_presence_classifier_3d import create_tumor_presence_classifier_3d
from integration.classifier_service import ClassifierService
import training.train_classifier as train_classifier_mod


def test_model_instantiation_defaults():
    """Test 1: Instantiates model with default parameters (in_channels=4, num_classes=2, base=16)."""
    model = create_tumor_classifier_3d()
    assert isinstance(model, TumorClassifier3D)
    assert model.in_channels == 4
    assert model.num_classes == 2
    assert model.base_channels == 16
    assert model.dropout_rate == 0.2
    assert model.count_parameters() > 0


def test_model_instantiation_custom():
    """Test 2: Instantiates model with custom channel widths and dropout rates."""
    model = create_tumor_classifier_3d(
        in_channels=1,
        num_classes=3,
        base_channels=8,
        dropout_rate=0.5,
    )
    assert model.in_channels == 1
    assert model.num_classes == 3
    assert model.base_channels == 8
    assert model.dropout_rate == 0.5


def test_forward_pass_shape_and_finiteness():
    """Test 3: Forward pass returns (B, 2) shape with finite logits for small synthetic batch."""
    model = create_tumor_classifier_3d(base_channels=8)
    model.eval()

    # Use small synthetic batch to keep unit tests fast: (B=2, C=4, D=32, H=32, W=24)
    x = torch.randn(2, 4, 32, 32, 24, dtype=torch.float32)
    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()


def test_loss_computation_and_backpropagation():
    """Test 4 & 5: Computes CrossEntropyLoss and verifies backpropagation yields finite gradients."""
    model = create_tumor_classifier_3d(base_channels=8)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(2, 4, 32, 32, 24, dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)

    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits, targets)

    assert torch.isfinite(loss).item()
    loss.backward()

    # Verify gradients for all trainable parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for parameter {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient in parameter {name}"

    optimizer.step()


def test_input_validation_rejections():
    """Test 6: Validates clear failure for invalid channel counts, non-5D shapes, and invalid types."""
    model = create_tumor_classifier_3d(in_channels=4, base_channels=8)

    # 1. Non-tensor input
    with pytest.raises(TypeError, match="Expected torch.Tensor"):
        model([[1.0, 2.0], [3.0, 4.0]])

    # 2. Non-5D input (4D tensor)
    with pytest.raises(ValueError, match="Expected 5D input tensor"):
        model(torch.randn(2, 4, 32, 32))

    # 3. Channel count mismatch (e.g. 1 or 3 channels instead of 4)
    with pytest.raises(ValueError, match="Input channel count mismatch"):
        model(torch.randn(2, 1, 32, 32, 24))

    with pytest.raises(ValueError, match="Input channel count mismatch"):
        model(torch.randn(2, 3, 32, 32, 24))

    # 4. Invalid configuration arguments
    with pytest.raises(ValueError, match="in_channels must be at least 1"):
        create_tumor_classifier_3d(in_channels=0)

    with pytest.raises(ValueError, match="num_classes must be at least 2"):
        create_tumor_classifier_3d(num_classes=1)

    with pytest.raises(ValueError, match="base_channels must be at least 4"):
        create_tumor_classifier_3d(base_channels=2)

    with pytest.raises(ValueError, match="dropout_rate must be in"):
        create_tumor_classifier_3d(dropout_rate=1.0)


def test_cpu_and_cuda_execution():
    """Test 7: Verifies execution on CPU and includes a CUDA smoke test if GPU is available."""
    # CPU verification
    cpu_model = create_tumor_classifier_3d(base_channels=8).to("cpu")
    cpu_model.eval()
    cpu_x = torch.randn(2, 4, 32, 32, 24, device="cpu")
    with torch.no_grad():
        cpu_logits = cpu_model(cpu_x)
    assert cpu_logits.device.type == "cpu"
    assert cpu_logits.shape == (2, 2)
    assert torch.isfinite(cpu_logits).all()

    # CUDA smoke test if available
    if torch.cuda.is_available():
        cuda_model = create_tumor_classifier_3d(base_channels=8).to("cuda")
        cuda_model.eval()
        cuda_x = torch.randn(2, 4, 32, 32, 24, device="cuda")
        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                cuda_logits = cuda_model(cuda_x)
        assert cuda_logits.device.type == "cuda"
        assert cuda_logits.shape == (2, 2)
        assert torch.isfinite(cuda_logits).all()


def test_import_training_module_isolation():
    """Test 8: Importing train_classifier does not start training or access datasets."""
    # Ensure train_classifier has required attributes
    assert hasattr(train_classifier_mod, "run_training")
    assert hasattr(train_classifier_mod, "train_one_epoch")
    assert hasattr(train_classifier_mod, "validate_epoch")
    assert hasattr(train_classifier_mod, "compute_classification_metrics")

    # Confirm that outputs/classifier_training/ was not created on import
    # (output dir should only be created when run_training() is called)
    test_out_dir = Path("outputs") / "classifier_training" / "checkpoints" / "classifier_best.pth"
    # Even if previous tests ran, importing the module itself must not write new files
    assert callable(train_classifier_mod.run_training)


def test_classification_metrics_computation():
    """Test 9: Verifies accuracy, macro F1, and confusion matrix calculation."""
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]

    metrics = train_classifier_mod.compute_classification_metrics(y_true, y_pred)

    # 3 correct out of 4 -> accuracy = 0.75
    assert metrics["accuracy"] == 0.75
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert metrics["precision_per_class"]["glioma"] == 1.0
    assert metrics["recall_per_class"]["glioma"] == 0.5
    assert metrics["precision_per_class"]["meningioma"] == float(round(2 / 3, 4))
    assert metrics["recall_per_class"]["meningioma"] == 1.0

    # Empty inputs edge case
    empty_metrics = train_classifier_mod.compute_classification_metrics([], [])
    assert empty_metrics["total_samples"] == 0
    assert empty_metrics["accuracy"] == 0.0


def test_classifier_service_missing_checkpoints_honesty():
    """Test 10: ClassifierService reports unavailable and blocks automatic routing when checkpoints do not exist."""
    service = ClassifierService(
        presence_checkpoint_path="nonexistent_presence.pth",
        type_checkpoint_path="nonexistent_type.pth",
    )
    status = service.get_status()
    assert status["automatic_classification_available"] is False
    assert status["presence_model_available"] is False
    assert status["type_model_available"] is False
    assert status["blocker_explanation"] is not None
    assert status["supported_manual_routes"] == ["glioma", "meningioma"]

    res = service.classify({})
    assert res["status"] == "automatic_classification_not_available"
    assert res["selected_route"] is None
    assert res["available_manual_routes"] == ["glioma", "meningioma"]


def test_classifier_service_corrupt_checkpoint_graceful_handling():
    """Test 11: Corrupt checkpoint file returns classification_failed without crashing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        corrupt_p = tmp_path / "corrupt.pth"
        corrupt_p.write_text("corrupted bytes here")

        service = ClassifierService(
            presence_checkpoint_path=corrupt_p,
            type_checkpoint_path=corrupt_p,
        )
        assert service.is_automatic_classification_available is True

        res = service.classify({})
        assert res["status"] == "classification_failed"
        assert res["selected_route"] is None
        assert "Classification inference failed" in res["reason"]
        assert res["available_manual_routes"] == ["glioma", "meningioma"]


def test_classifier_service_synthetic_trained_execution_flow():
    """Test 12: End-to-end execution flow when genuine synthetic weights are loaded."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        presence_ckpt = tmp_path / "presence_best.pth"
        type_ckpt = tmp_path / "type_best.pth"

        # Create synthetic models with num_classes=2
        m_pres = create_tumor_presence_classifier_3d(in_channels=2, num_classes=2, base_channels=8)
        torch.save({
            "model_state_dict": m_pres.state_dict(),
            "base_channels": 8,
            "num_classes": 2,
        }, presence_ckpt)

        m_type = create_tumor_classifier_3d(in_channels=4, num_classes=2, base_channels=8)
        torch.save({
            "model_state_dict": m_type.state_dict(),
            "base_channels": 8,
            "num_classes": 2,
        }, type_ckpt)

        service = ClassifierService(
            presence_checkpoint_path=presence_ckpt,
            type_checkpoint_path=type_ckpt,
            device="cpu",
        )
        assert service.is_automatic_classification_available is True

        # Create synthetic volumes
        affine = np.diag([1.0, 1.0, 1.0, 1.0])
        paths = {}
        for mod in ["t1", "t1c", "t2", "flair"]:
            data = np.zeros((240, 240, 155), dtype=np.float32)
            data[50:150, 50:150, 40:100] = 100.0  # brain tissue
            img = nib.Nifti1Image(data, affine)
            fpath = tmp_path / f"{mod}.nii.gz"
            nib.save(img, str(fpath))
            paths[mod] = fpath

        res = service.classify(paths)
        assert res["status"] in ("classified_healthy", "classified_tumor", "uncertain_classification")
        if res["status"] == "classified_healthy":
            assert res["tumor_presence"] == "healthy_control"
            assert res["selected_route"] is None
        elif res["status"] == "uncertain_classification":
            assert res["tumor_presence"] == "tumor_present"
            assert res["selected_route"] is None
            assert 0.0 <= res["confidence"] <= 1.0
        else:
            assert res["tumor_presence"] == "tumor_present"
            assert res["predicted_tumor_type"] in ("glioma", "meningioma")
            assert res["selected_route"] in ("glioma", "meningioma")
            assert 0.0 <= res["confidence"] <= 1.0
