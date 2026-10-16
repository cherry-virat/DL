"""
Unit and Integration Tests for Thin Inference Services.
======================================================

Tests:
1. Service imports and class instantiation.
2. GliomaInferenceService with synthetic 4-channel input returning valid SegmentationResult.
3. Glioma subregion breakdown (WT, TC, ET), volumetric quantification, and JSON serialization.
4. MeningiomaInferenceService with synthetic T1c input returning valid SegmentationResult.
5. Meningioma GTV volume, spatial restoration status, and JSON serialization.
6. Confirmation that Meningioma offline evaluation contains NO fake Dice/IoU.
7. Input validation error handling (missing modalities, bad dimensions).
"""

import json
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest
import torch

from integration.result_schema import (
    TumorType,
    PipelineStatus,
    SegmentationResult,
)
from integration.glioma_inference_service import (
    GliomaInferenceService,
    run_glioma_inference,
)
from integration.meningioma_inference_service import (
    MeningiomaInferenceService,
    run_meningioma_inference,
)


class MockGliomaModel(torch.nn.Module):
    """Fast, deterministic mock of 4-class Glioma 3D U-Net."""
    def __init__(self):
        super().__init__()
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x is (B, 4, D, H, W)
        b, c, d, h, w = x.shape
        # Create logits with background (class 0) dominant everywhere
        logits = torch.zeros((b, 4, d, h, w), dtype=torch.float32, device=x.device)
        logits[:, 0, :, :, :] = 2.0  # background higher by default

        # Place a simulated tumor blob in center
        cd, ch, cw = d // 2, h // 2, w // 2
        # Whole Tumor region (classes 1, 2, 3)
        logits[:, 2, cd-6:cd+6, ch-6:ch+6, cw-6:cw+6] = 5.0  # Edema (class 2)
        logits[:, 1, cd-3:cd+3, ch-3:ch+3, cw-3:cw+3] = 6.0  # Necrosis (class 1)
        logits[:, 3, cd-2:cd+2, ch-2:ch+2, cw-2:cw+2] = 7.0  # Enhancing tumor (class 3)
        return logits


class MockMeningiomaModel(torch.nn.Module):
    """Fast, deterministic mock of binary Meningioma 3D U-Net."""
    def __init__(self):
        super().__init__()
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x is (B, 1, D, H, W)
        b, c, d, h, w = x.shape
        logits = torch.zeros((b, 2, d, h, w), dtype=torch.float32, device=x.device)
        logits[:, 0, :, :, :] = 2.0  # background dominant

        # Simulated dural meningioma blob
        cd, ch, cw = d // 2, h // 2, w // 2
        logits[:, 1, cd-8:cd+8, ch-8:ch+8, cw-8:cw+8] = 6.0  # GTV (class 1)
        return logits


def test_service_imports_and_instantiation():
    """Test 1: Verify services can be imported and initialized."""
    glioma_svc = GliomaInferenceService()
    assert glioma_svc is not None
    assert glioma_svc.checkpoint_path.name == "glioma_best_medium_experiment.pth"

    meningioma_svc = MeningiomaInferenceService()
    assert meningioma_svc is not None
    assert meningioma_svc.checkpoint_path.name == "meningioma_best.pth"


def test_glioma_inference_service_synthetic():
    """Test 2 & 3: Glioma inference with mock model returning complete SegmentationResult."""
    mock_model = MockGliomaModel()
    service = GliomaInferenceService(model=mock_model)

    # Synthetic 4-channel input (4, 64, 64, 64)
    synthetic_mri = np.zeros((4, 64, 64, 64), dtype=np.float32)
    # Give some non-zero foreground
    synthetic_mri[:, 10:54, 10:54, 10:54] = 1.0

    result = service.predict(
        case_input=synthetic_mri,
        request_id="REQ-GLI-TEST-001",
        patient_id="PATIENT-GLI-001",
    )

    # Validate output type and contract
    assert isinstance(result, SegmentationResult)
    assert result.request.tumor_type == TumorType.GLIOMA
    assert result.request.request_id == "REQ-GLI-TEST-001"
    assert result.request.patient_id == "PATIENT-GLI-001"
    assert result.quality_status.pipeline_frozen is True
    assert result.quality_status.qc_verdict == "PASS WITH DIFFICULT CASES"

    # Validate Glioma specific subregions
    assert result.glioma is not None
    assert result.meningioma is None
    sub = result.glioma.subregions
    assert sub.wt_voxels > 0
    assert sub.tc_voxels > 0
    assert sub.et_voxels > 0
    assert sub.wt_volume_cm3 > 0.0

    # Validate Volumetric consistency
    assert result.volumetrics.total_tumor_voxels == sub.wt_voxels
    assert result.volumetrics.total_tumor_volume_mm3 == sub.wt_volume_mm3

    # Validate JSON serialization and deserialization
    json_str = result.to_json()
    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert parsed["request"]["tumor_type"] == "glioma"
    assert parsed["glioma"]["subregions"]["wt_voxels"] == sub.wt_voxels

    # Validate QC metadata provenance
    assert result.quality_status.qc_notes is not None
    assert "20 internal validation cases" in result.quality_status.qc_notes
    assert "0.8478" in result.quality_status.qc_notes
    assert "not an independent test cohort" in result.quality_status.qc_notes

    # Test round trip
    restored = SegmentationResult.from_json(json_str)
    assert restored.glioma.subregions.wt_voxels == sub.wt_voxels


def test_glioma_mask_saving_and_gtv_binary_equivalence(tmp_path):
    """Test: Glioma inference saves separate multiclass and binary GTV masks with exact equivalence."""
    import SimpleITK as sitk

    mock_model = MockGliomaModel()
    service = GliomaInferenceService(model=mock_model)

    synthetic_mri = np.zeros((4, 64, 64, 64), dtype=np.float32)
    synthetic_mri[:, 10:54, 10:54, 10:54] = 1.0

    save_multi_path = tmp_path / "case_glioma_seg.nii.gz"
    result = service.predict(
        case_input=synthetic_mri,
        request_id="REQ-GLI-SEP-001",
        save_mask_path=save_multi_path,
    )

    # Both mask references must be populated
    assert result.glioma.masks.multiclass_mask is not None
    assert result.glioma.masks.wt_mask is not None

    p_multi = Path(result.glioma.masks.multiclass_mask.mask_path)
    p_gtv = Path(result.glioma.masks.wt_mask.mask_path)

    # Distinct file paths on disk
    assert p_multi != p_gtv
    assert p_multi.exists()
    assert p_gtv.exists()
    assert p_gtv.name.endswith("_gtv.nii.gz")

    # Read both images
    img_multi = sitk.ReadImage(str(p_multi))
    img_gtv = sitk.ReadImage(str(p_gtv))

    arr_multi = sitk.GetArrayFromImage(img_multi)
    arr_gtv = sitk.GetArrayFromImage(img_gtv)

    # Check labels
    unique_multi = set(np.unique(arr_multi))
    assert unique_multi.issubset({0, 1, 2, 3})
    assert any(c in unique_multi for c in [1, 2, 3])

    unique_gtv = set(np.unique(arr_gtv))
    assert unique_gtv.issubset({0, 1})
    assert 1 in unique_gtv

    # Verify binary equivalence: (arr_multi > 0) == (arr_gtv == 1)
    assert np.array_equal((arr_multi > 0).astype(np.uint8), arr_gtv)
    assert int((arr_gtv == 1).sum()) == result.glioma.subregions.wt_voxels
    assert int((arr_gtv == 1).sum()) == result.volumetrics.total_tumor_voxels

    # Verify matching spatial geometry
    assert img_multi.GetSize() == img_gtv.GetSize()
    assert img_multi.GetSpacing() == img_gtv.GetSpacing()
    assert img_multi.GetOrigin() == img_gtv.GetOrigin()
    assert img_multi.GetDirection() == img_gtv.GetDirection()


def test_meningioma_inference_service_synthetic():
    """Test 4, 5 & 6: Meningioma inference with mock model, GTV volume, zero fake validation Dice."""
    mock_model = MockMeningiomaModel()
    service = MeningiomaInferenceService(model=mock_model)

    # Synthetic single T1c input (64, 64, 64)
    synthetic_t1c = np.zeros((64, 64, 64), dtype=np.float32)
    synthetic_t1c[10:54, 10:54, 10:54] = 1.5

    result = service.predict(
        t1c_input=synthetic_t1c,
        request_id="REQ-MEN-TEST-002",
        patient_id="PATIENT-MEN-002",
    )

    # Validate output type and contract
    assert isinstance(result, SegmentationResult)
    assert result.request.tumor_type == TumorType.MENINGIOMA
    assert result.request.request_id == "REQ-MEN-TEST-002"
    assert result.quality_status.pipeline_frozen is True
    assert result.quality_status.qc_verdict == "PASS WITH UNCERTAINTY"

    # Validate Meningioma specific measurements
    assert result.meningioma is not None
    assert result.glioma is None
    assert result.meningioma.gtv_voxels > 0
    assert result.meningioma.gtv_volume_cm3 > 0.0
    assert result.meningioma.label_definitions[1] == "Meningioma GTV (Gross Tumor Volume)"

    # CRITICAL CHECK: No fake Dice/IoU
    assert result.offline_evaluation.has_ground_truth is False
    assert result.offline_evaluation.dice is None
    assert result.offline_evaluation.iou is None

    # Validate JSON serialization
    json_str = result.to_json()
    parsed = json.loads(json_str)
    assert parsed["request"]["tumor_type"] == "meningioma"
    assert parsed["offline_evaluation"]["dice"] is None

    restored = SegmentationResult.from_json(json_str)
    assert restored.meningioma.gtv_voxels == result.meningioma.gtv_voxels


def test_invalid_input_rejections():
    """Test 7: Error handling on invalid modality inputs."""
    glioma_svc = GliomaInferenceService(model=MockGliomaModel())

    # Missing modality dictionary keys
    with pytest.raises(ValueError):
        glioma_svc.predict({"T1": "nonexistent.nii.gz"})

    # Invalid numpy dimensions (3D instead of 4D)
    with pytest.raises(ValueError):
        glioma_svc.predict(np.zeros((64, 64, 64), dtype=np.float32))

    # Unsupported input type
    with pytest.raises(TypeError):
        glioma_svc.predict(12345)

    men_svc = MeningiomaInferenceService(model=MockMeningiomaModel())
    with pytest.raises(TypeError):
        men_svc.predict(12345)

