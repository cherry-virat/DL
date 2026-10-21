"""
Unit and Integration Tests for Unified Inference Router.
========================================================

Tests:
1. Router imports and class instantiation.
2. GLIOMA routing dispatches to GliomaInferenceService with correct result.
3. MENINGIOMA routing dispatches to MeningiomaInferenceService with correct result.
4. String case-insensitivity ('GLIOMA', 'glioma', 'Meningioma') normalization.
5. Rejection of unsupported or invalid tumor types.
6. Rejection of invalid inputs for Glioma (missing modalities, bad dimensions).
7. Rejection of invalid inputs for Meningioma (missing T1c, bad dimensions).
8. Verification that returned object is a SegmentationResult matching requested tumor type.
9. Fast mock execution without expensive GPU load.
"""

import json
from unittest.mock import MagicMock
import numpy as np
import pytest
import torch

from integration.result_schema import (
    TumorType,
    PipelineStatus,
    SegmentationResult,
)
from integration.inference_router import (
    InferenceRouter,
    run_inference,
)
from integration.glioma_inference_service import (
    GliomaInferenceService,
)
from integration.meningioma_multimodal_service import (
    MeningiomaMultimodalInferenceService,
)


class MockGliomaModel(torch.nn.Module):
    """Fast deterministic mock of 4-class Glioma 3D U-Net."""
    def __init__(self):
        super().__init__()
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, d, h, w = x.shape
        logits = torch.zeros((b, 4, d, h, w), dtype=torch.float32, device=x.device)
        logits[:, 0, :, :, :] = 2.0  # background
        cd, ch, cw = d // 2, h // 2, w // 2
        logits[:, 2, cd-6:cd+6, ch-6:ch+6, cw-6:cw+6] = 5.0
        logits[:, 1, cd-3:cd+3, ch-3:ch+3, cw-3:cw+3] = 6.0
        logits[:, 3, cd-2:cd+2, ch-2:ch+2, cw-2:cw+2] = 7.0
        return logits


class MockMeningiomaModel(torch.nn.Module):
    """Fast deterministic mock of 4-class Multimodal Meningioma 3D U-Net."""
    def __init__(self):
        super().__init__()
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, d, h, w = x.shape
        logits = torch.zeros((b, 4, d, h, w), dtype=torch.float32, device=x.device)
        logits[:, 0, :, :, :] = 2.0  # background
        cd, ch, cw = d // 2, h // 2, w // 2
        logits[:, 2, cd-6:cd+6, ch-6:ch+6, cw-6:cw+6] = 5.0
        logits[:, 1, cd-3:cd+3, ch-3:ch+3, cw-3:cw+3] = 6.0
        logits[:, 3, cd-2:cd+2, ch-2:ch+2, cw-2:cw+2] = 7.0
        return logits


def test_router_imports_and_instantiation():
    """Test 1: Router imports correctly and can be instantiated."""
    router = InferenceRouter()
    assert router is not None
    assert router.glioma_service is not None
    assert router.meningioma_service is not None


def test_router_dispatches_glioma_enum_and_string():
    """Test 2 & 8 & 9: GLIOMA dispatches to Glioma service returning SegmentationResult."""
    mock_glioma = GliomaInferenceService(model=MockGliomaModel())
    router = InferenceRouter(glioma_service=mock_glioma)

    # 4-channel input array (4, 64, 64, 64)
    glioma_input = np.zeros((4, 64, 64, 64), dtype=np.float32)
    glioma_input[:, 10:54, 10:54, 10:54] = 1.0

    # 1. Dispatch using Enum
    res_enum = router.route(
        tumor_type=TumorType.GLIOMA,
        input_data=glioma_input,
        request_id="REQ-ROUTER-GLI-01",
        patient_id="PATIENT-01",
    )
    assert isinstance(res_enum, SegmentationResult)
    assert res_enum.request.tumor_type == TumorType.GLIOMA
    assert res_enum.request.request_id == "REQ-ROUTER-GLI-01"
    assert res_enum.glioma is not None
    assert res_enum.meningioma is None
    assert res_enum.glioma.subregions.wt_voxels > 0

    # 2. Dispatch using string case-insensitively
    res_str = router.route(
        tumor_type="gLiOmA",
        input_data=glioma_input,
        request_id="REQ-ROUTER-GLI-02",
    )
    assert isinstance(res_str, SegmentationResult)
    assert res_str.request.tumor_type == TumorType.GLIOMA

    # 3. Convenience function test
    res_func = run_inference(
        tumor_type="glioma",
        input_data=glioma_input,
        router=router,
        request_id="REQ-ROUTER-GLI-03",
    )
    assert isinstance(res_func, SegmentationResult)
    assert res_func.request.tumor_type == TumorType.GLIOMA


def test_router_dispatches_meningioma_enum_and_string():
    """Test 3 & 8 & 9: MENINGIOMA dispatches to Meningioma multimodal service returning SegmentationResult."""
    mock_meningioma = MeningiomaMultimodalInferenceService(model=MockMeningiomaModel())
    router = InferenceRouter(meningioma_service=mock_meningioma)

    # 4-channel input array (4, 64, 64, 64)
    men_input = np.zeros((4, 64, 64, 64), dtype=np.float32)
    men_input[:, 10:54, 10:54, 10:54] = 1.5

    # 1. Dispatch using Enum
    res_enum = router.route(
        tumor_type=TumorType.MENINGIOMA,
        input_data=men_input,
        request_id="REQ-ROUTER-MEN-01",
        patient_id="PATIENT-02",
    )
    assert isinstance(res_enum, SegmentationResult)
    assert res_enum.request.tumor_type == TumorType.MENINGIOMA
    assert res_enum.request.request_id == "REQ-ROUTER-MEN-01"
    assert res_enum.meningioma is not None
    assert res_enum.meningioma_multimodal is not None
    assert res_enum.glioma is None
    assert res_enum.volumetrics.total_tumor_voxels > 0
    assert res_enum.offline_evaluation is None

    # 2. Dispatch using string case-insensitively
    res_str = router.route(
        tumor_type="MeNiNgIoMa",
        input_data=men_input,
        request_id="REQ-ROUTER-MEN-02",
    )
    assert isinstance(res_str, SegmentationResult)
    assert res_str.request.tumor_type == TumorType.MENINGIOMA

    # 3. Convenience function test
    res_func = run_inference(
        tumor_type="meningioma",
        input_data=men_input,
        router=router,
        request_id="REQ-ROUTER-MEN-03",
    )
    assert isinstance(res_func, SegmentationResult)
    assert res_func.request.tumor_type == TumorType.MENINGIOMA


def test_unsupported_tumor_type_rejection():
    """Test 5: Router rejects unsupported or invalid tumor types with clear error message."""
    router = InferenceRouter()
    dummy_input = np.zeros((64, 64, 64), dtype=np.float32)

    with pytest.raises(ValueError, match="Unsupported tumor type"):
        router.route("sarcoma", dummy_input)

    with pytest.raises(ValueError, match="Unsupported tumor type"):
        router.route("metastasis", dummy_input)

    with pytest.raises(TypeError, match="Expected tumor_type"):
        router.route(12345, dummy_input)

    with pytest.raises(TypeError, match="Expected tumor_type"):
        router.route(None, dummy_input)


def test_invalid_glioma_inputs_rejection():
    """Test 6: Router validates and rejects invalid Glioma input structures."""
    router = InferenceRouter(glioma_service=GliomaInferenceService(model=MockGliomaModel()))

    # Missing input
    with pytest.raises(ValueError, match="input_data cannot be None"):
        router.route("glioma", None)

    # Dictionary missing required modalities (e.g. only T1 provided)
    with pytest.raises(ValueError, match="Missing required modalities"):
        router.route("glioma", {"T1": "file_t1.nii.gz"})

    # Invalid array dimension (3D instead of 4D)
    with pytest.raises(ValueError, match="Invalid array shape for Glioma"):
        router.route("glioma", np.zeros((64, 64, 64), dtype=np.float32))

    # Invalid array channel count (1 channel instead of 4)
    with pytest.raises(ValueError, match="Invalid array shape for Glioma"):
        router.route("glioma", np.zeros((1, 64, 64, 64), dtype=np.float32))

    # Unsupported data type
    with pytest.raises(TypeError, match="Unsupported input_data type"):
        router.route("glioma", [1, 2, 3])


def test_invalid_meningioma_inputs_rejection():
    """Test 7: Router validates and rejects invalid Meningioma input structures."""
    router = InferenceRouter(meningioma_service=MeningiomaMultimodalInferenceService(model=MockMeningiomaModel()))

    # Missing input
    with pytest.raises(ValueError, match="input_data cannot be None"):
        router.route("meningioma", None)

    # Dictionary without all 4 modalities
    with pytest.raises(ValueError, match="Missing required modalities"):
        router.route("meningioma", {"t1c": "t1c.nii.gz"})

    # Invalid array shape (3D or 1-channel instead of 4D)
    with pytest.raises(ValueError, match="Invalid array shape for Meningioma"):
        router.route("meningioma", np.zeros((64, 64, 64), dtype=np.float32))

    with pytest.raises(ValueError, match="Invalid array shape for Meningioma"):
        router.route("meningioma", np.zeros((1, 64, 64, 64), dtype=np.float32))

    # Unsupported data type
    with pytest.raises(TypeError, match="Unsupported input_data type"):
        router.route("meningioma", 999.0)


def test_router_correct_model_selection_isolation():
    """Test 8: Verifies strict model isolation - glioma routes ONLY to glioma service and meningioma ONLY to meningioma service."""
    mock_glioma_svc = MagicMock()
    mock_meningioma_svc = MagicMock()

    mock_glioma_svc.predict.return_value = MagicMock(spec=SegmentationResult)
    mock_meningioma_svc.predict.return_value = MagicMock(spec=SegmentationResult)

    router = InferenceRouter(
        glioma_service=mock_glioma_svc,
        meningioma_service=mock_meningioma_svc,
    )

    dummy_glioma_input = np.zeros((4, 64, 64, 64), dtype=np.float32)
    dummy_men_input = np.zeros((4, 64, 64, 64), dtype=np.float32)

    # 1. Route Glioma
    router.route(tumor_type="glioma", input_data=dummy_glioma_input, request_id="REQ-ISO-GLI")
    mock_glioma_svc.predict.assert_called_once()
    mock_meningioma_svc.predict.assert_not_called()

    # Reset mocks
    mock_glioma_svc.reset_mock()
    mock_meningioma_svc.reset_mock()

    # 2. Route Meningioma
    router.route(tumor_type="meningioma", input_data=dummy_men_input, request_id="REQ-ISO-MEN")
    mock_meningioma_svc.predict.assert_called_once()
    mock_glioma_svc.predict.assert_not_called()

    # Reset mocks
    mock_glioma_svc.reset_mock()
    mock_meningioma_svc.reset_mock()

    # 3. Route Meningioma Multimodal alias
    router.route(tumor_type="meningioma_multimodal", input_data=dummy_men_input, request_id="REQ-ISO-MULTI")
    mock_meningioma_svc.predict.assert_called_once()
    mock_glioma_svc.predict.assert_not_called()


def test_router_unexpected_and_duplicate_modalities():
    """Test 9: Verifies router rejects unexpected and duplicate modalities in dictionary inputs."""
    router = InferenceRouter(
        glioma_service=GliomaInferenceService(model=MockGliomaModel()),
        meningioma_service=MeningiomaMultimodalInferenceService(model=MockMeningiomaModel()),
    )

    # Glioma: unexpected modality
    with pytest.raises(ValueError, match="Unexpected modality"):
        router.route("glioma", {"T1": "a", "T1c": "b", "T2": "c", "FLAIR": "d", "DWI": "e"})

    # Glioma: duplicate modality (T1 and t1n both resolve to T1)
    with pytest.raises(ValueError, match="Duplicate modality"):
        router.route("glioma", {"T1": "a", "t1n": "b", "T1c": "c", "T2": "d", "FLAIR": "e"})

    # Meningioma: unexpected modality
    with pytest.raises(ValueError, match="Unexpected modality"):
        router.route("meningioma", {"t1n": "a", "t1c": "b", "t2w": "c", "t2f": "d", "PET": "e"})

    # Meningioma: duplicate modality (t1n and T1 both resolve to t1n)
    with pytest.raises(ValueError, match="Duplicate modality"):
        router.route("meningioma", {"t1n": "a", "T1": "b", "t1c": "c", "t2w": "d", "t2f": "e"})


def test_router_inference_failure_propagation():
    """Test 10: Verifies router propagates inference runtime errors cleanly."""
    mock_failing_glioma = MagicMock()
    mock_failing_glioma.predict.side_effect = RuntimeError("CUDA OOM or model execution failure")

    router = InferenceRouter(glioma_service=mock_failing_glioma)
    dummy_input = np.zeros((4, 64, 64, 64), dtype=np.float32)

    with pytest.raises(RuntimeError, match="CUDA OOM or model execution failure"):
        router.route("glioma", dummy_input)
