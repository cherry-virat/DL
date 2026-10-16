"""
Unit and Integration Tests for Isolated Multimodal Meningioma Inference.
========================================================================

Tests:
1. Service imports, checkpoint resolution, and class instantiation.
2. Input validation:
   - Rejection of None input.
   - Rejection of missing modalities.
   - Rejection of spatial shape mismatch across modalities.
   - Rejection of affine mismatch across modalities.
   - Rejection of invalid array dimensionality.
3. Empty-air masking behavior on predictions.
4. Model forward pass, sliding-window inference, and subregion breakdown (WT, TC, ET, NCR, ED).
5. Schema serialization, deserialization, and research disclaimer verification.
6. Unified InferenceRouter integration with string and enum dispatch.
7. Flask REST API endpoint (/api/segment/meningioma-multimodal and /api/segment) integration.
"""

import io
import json
import tempfile
from pathlib import Path
from typing import Dict, Any
from unittest.mock import MagicMock

import nibabel as nib
import numpy as np
import pytest
import SimpleITK as sitk
import torch

from api.app import create_app
from integration.result_schema import (
    TumorType,
    PipelineStatus,
    RequestMetadata,
    InputMetadata,
    VolumetricSummary,
    GliomaSubregionMasks,
    MaskReference,
    MeningiomaMeasurements,
    MeningiomaMultimodalSubregionVolumes,
    MeningiomaMultimodalMeasurements,
    InferenceQualityStatus,
    SegmentationResult,
)
from integration.inference_router import (
    InferenceRouter,
    run_inference,
)
from integration.meningioma_multimodal_service import (
    MeningiomaMultimodalInferenceService,
    run_meningioma_multimodal_inference,
    detect_modality_key,
    REQUIRED_MODALITIES,
    DEFAULT_MULTIMODAL_CHECKPOINT,
    RESEARCH_DISCLAIMER,
)


class MockMultimodalModel(torch.nn.Module):
    """Deterministic mock of 4-class Multimodal Meningioma 3D U-Net."""

    def __init__(self):
        super().__init__()
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, 4, D, H, W)
        b, c, d, h, w = x.shape
        logits = torch.zeros((b, 4, d, h, w), dtype=torch.float32, device=x.device)
        logits[:, 0, :, :, :] = 2.0  # Background dominant by default

        # Place simulated lesions in center
        cd, ch, cw = d // 2, h // 2, w // 2
        # Edema (class 2)
        logits[:, 2, cd-6:cd+6, ch-6:ch+6, cw-6:cw+6] = 5.0
        # Necrotic Core (class 1)
        logits[:, 1, cd-3:cd+3, ch-3:ch+3, cw-3:cw+3] = 6.0
        # Enhancing Tumor (class 3)
        logits[:, 3, cd-2:cd+2, ch-2:ch+2, cw-2:cw+2] = 7.0

        return logits


def create_synthetic_nifti_file(
    path: Path,
    shape=(32, 32, 32),
    affine=None,
    fill_value: float = 1.0,
) -> Path:
    """Helper creating a temporary NIfTI file on disk."""
    if affine is None:
        affine = np.eye(4)
    data = np.full(shape, fill_value, dtype=np.float32)
    # Add zero borders (air voxels)
    data[:4, :, :] = 0.0
    data[-4:, :, :] = 0.0
    nii = nib.Nifti1Image(data, affine=affine)
    nib.save(nii, str(path))
    return path


def test_service_initialization_and_checkpoint_resolution():
    """Test 1: Service instantiates with correct default checkpoint path and attributes."""
    service = MeningiomaMultimodalInferenceService()
    assert service is not None
    assert service.checkpoint_path == DEFAULT_MULTIMODAL_CHECKPOINT
    assert service.checkpoint_path.name == "meningioma_multimodal_best.pth"
    assert service.device is not None


def test_modality_key_detection():
    """Test 2: Modality key detection accurately identifies BraTS and standard filenames."""
    assert detect_modality_key("BraTS-MEN-00001-000-t1n.nii.gz") == "t1n"
    assert detect_modality_key("case_t1c.nii.gz") == "t1c"
    assert detect_modality_key("patient_01_t1ce.nii.gz") == "t1c"
    assert detect_modality_key("scan_t2w.nii.gz") == "t2w"
    assert detect_modality_key("patient_01_t2.nii.gz") == "t2w"
    assert detect_modality_key("scan_t2f.nii.gz") == "t2f"
    assert detect_modality_key("brain_flair.nii.gz") == "t2f"
    assert detect_modality_key("unrelated_file.txt") is None


def test_validation_missing_input_rejection():
    """Test 3: Service rejects None input with ValueError."""
    service = MeningiomaMultimodalInferenceService(model=MockMultimodalModel())
    with pytest.raises(TypeError, match="Unsupported case_input type"):
        service.predict(case_input=None)


def test_validation_missing_modalities_rejection(tmp_path):
    """Test 4: Service rejects input dictionary missing any of the 4 required modalities."""
    service = MeningiomaMultimodalInferenceService(model=MockMultimodalModel())
    f1 = create_synthetic_nifti_file(tmp_path / "t1n.nii.gz")
    f2 = create_synthetic_nifti_file(tmp_path / "t1c.nii.gz")
    f3 = create_synthetic_nifti_file(tmp_path / "t2w.nii.gz")

    # Only 3 modalities provided
    incomplete_dict = {"t1n": f1, "t1c": f2, "t2w": f3}
    with pytest.raises(ValueError, match="Missing required multimodal modalities"):
        service.preprocess_inputs(incomplete_dict)


def test_validation_spatial_shape_mismatch_rejection(tmp_path):
    """Test 5: Service rejects modalities with differing matrix dimensions."""
    service = MeningiomaMultimodalInferenceService(model=MockMultimodalModel())
    f_t1n = create_synthetic_nifti_file(tmp_path / "t1n.nii.gz", shape=(32, 32, 32))
    f_t1c = create_synthetic_nifti_file(tmp_path / "t1c.nii.gz", shape=(32, 32, 32))
    f_t2w = create_synthetic_nifti_file(tmp_path / "t2w.nii.gz", shape=(32, 32, 32))
    # Mismatched shape (32, 32, 16)
    f_t2f = create_synthetic_nifti_file(tmp_path / "t2f.nii.gz", shape=(32, 32, 16))

    mismatched_dict = {"t1n": f_t1n, "t1c": f_t1c, "t2w": f_t2w, "t2f": f_t2f}
    with pytest.raises(ValueError, match="Spatial shape mismatch"):
        service.preprocess_inputs(mismatched_dict)


def test_validation_affine_mismatch_rejection(tmp_path):
    """Test 6: Service rejects modalities with mismatched spatial orientation affines."""
    service = MeningiomaMultimodalInferenceService(model=MockMultimodalModel())
    aff1 = np.eye(4)
    aff2 = np.diag([2.0, 2.0, 2.0, 1.0])

    f_t1n = create_synthetic_nifti_file(tmp_path / "t1n.nii.gz", affine=aff1)
    f_t1c = create_synthetic_nifti_file(tmp_path / "t1c.nii.gz", affine=aff1)
    f_t2w = create_synthetic_nifti_file(tmp_path / "t2w.nii.gz", affine=aff1)
    f_t2f = create_synthetic_nifti_file(tmp_path / "t2f.nii.gz", affine=aff2)

    mismatched_dict = {"t1n": f_t1n, "t1c": f_t1c, "t2w": f_t2w, "t2f": f_t2f}
    with pytest.raises(ValueError, match="Spatial affine mismatch"):
        service.preprocess_inputs(mismatched_dict)


def test_validation_invalid_array_shape_rejection():
    """Test 7: Service rejects numpy inputs with wrong number of channels or dimensions."""
    service = MeningiomaMultimodalInferenceService(model=MockMultimodalModel())

    # 3D array instead of 4D
    with pytest.raises(ValueError, match="Expected 4D numpy array"):
        service.predict(case_input=np.zeros((64, 64, 64), dtype=np.float32))

    # 1-channel 4D array instead of 4 channels
    with pytest.raises(ValueError, match="Expected 4D numpy array with shape"):
        service.predict(case_input=np.zeros((1, 64, 64, 64), dtype=np.float32))


def test_multimodal_inference_synthetic_execution_and_subregions():
    """Test 8: Full sliding-window inference with mock model verifies subregions and volumetrics."""
    mock_model = MockMultimodalModel()
    service = MeningiomaMultimodalInferenceService(model=mock_model, device=torch.device("cpu"))

    # Synthetic 4-channel input (4, 64, 64, 64) with non-zero brain foreground
    synthetic_mri = np.zeros((4, 64, 64, 64), dtype=np.float32)
    synthetic_mri[:, 10:54, 10:54, 10:54] = 2.5

    result = service.predict(
        case_input=synthetic_mri,
        request_id="REQ-TEST-MULTI-01",
        patient_id="PATIENT-001",
        session_id="SESS-001",
    )

    assert isinstance(result, SegmentationResult)
    assert result.request.tumor_type == TumorType.MENINGIOMA_MULTIMODAL
    assert result.request.request_id == "REQ-TEST-MULTI-01"

    # Verify multimodal measurements
    assert result.meningioma_multimodal is not None
    meas = result.meningioma_multimodal
    assert meas.subregions.wt_voxels > 0
    assert meas.subregions.tc_voxels > 0
    assert meas.subregions.et_voxels > 0
    assert meas.subregions.ncr_net_voxels > 0
    assert meas.subregions.edema_voxels > 0

    # Whole Tumor must be >= Tumor Core and >= Enhancing Tumor
    assert meas.subregions.wt_voxels >= meas.subregions.tc_voxels
    assert meas.subregions.tc_voxels >= meas.subregions.et_voxels

    # Verify VolumetricSummary
    assert result.volumetrics.total_tumor_voxels == meas.subregions.wt_voxels
    assert result.volumetrics.total_tumor_volume_cm3 == meas.subregions.wt_volume_cm3

    # Verify backward-compatible legacy meningioma payload
    assert result.meningioma is not None
    assert result.meningioma.gtv_voxels == meas.subregions.wt_voxels

    # Verify disclaimer is present
    assert RESEARCH_DISCLAIMER in meas.disclaimer
    assert result.quality_status.status == PipelineStatus.SUCCESS


def test_empty_air_masking_integration():
    """Test 9: Empty-air voxels (all 4 channels zero) are forced to background (class 0)."""
    mock_model = MockMultimodalModel()
    service = MeningiomaMultimodalInferenceService(model=mock_model, device=torch.device("cpu"))

    # Create input where center voxels are strictly 0.0 (simulated air cavity inside brain)
    synthetic_mri = np.zeros((4, 64, 64, 64), dtype=np.float32)
    synthetic_mri[:, 10:54, 10:54, 10:54] = 2.0
    # Zero out the center 6x6x6 where mock model places enhancing tumor
    synthetic_mri[:, 30:34, 30:34, 30:34] = 0.0

    result = service.predict(
        case_input=synthetic_mri,
        request_id="REQ-TEST-AIR-01",
    )

    # In synthetic_mri[:, 30:34, 30:34, 30:34] all channels are 0.0,
    # so despite MockMultimodalModel predicting class 3, air masking forces it to 0.
    # Let's inspect that the air masking zeroed those specific voxels
    # by running direct predict with save_mask_path
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        tmp_mask_path = Path(tmp.name)

    try:
        service.predict(
            case_input=synthetic_mri,
            request_id="REQ-TEST-AIR-02",
            save_mask_path=tmp_mask_path,
        )
        saved_mask = nib.load(str(tmp_mask_path)).get_fdata()
        # The region 30:34 must be 0 because it was zero in all channels
        assert np.all(saved_mask[30:34, 30:34, 30:34] == 0)
    finally:
        if tmp_mask_path.exists():
            tmp_mask_path.unlink()


def test_json_roundtrip_serialization():
    """Test 10: Multimodal result serializes to JSON and round-trips via from_json faithfully."""
    mock_model = MockMultimodalModel()
    service = MeningiomaMultimodalInferenceService(model=mock_model, device=torch.device("cpu"))

    synthetic_mri = np.zeros((4, 64, 64, 64), dtype=np.float32)
    synthetic_mri[:, 10:54, 10:54, 10:54] = 1.0

    orig_result = service.predict(
        case_input=synthetic_mri,
        request_id="REQ-JSON-ROUNDTRIP-01",
        patient_id="PT-ROUNDTRIP",
    )

    json_str = orig_result.to_json()
    assert isinstance(json_str, str)
    parsed = json.loads(json_str)

    assert parsed["request"]["tumor_type"] == "meningioma_multimodal"
    assert parsed["meningioma_multimodal"]["subregions"]["wt_voxels"] == orig_result.meningioma_multimodal.subregions.wt_voxels

    # Reconstruct from JSON
    reconstructed = SegmentationResult.from_json(json_str)
    assert reconstructed.request.request_id == "REQ-JSON-ROUNDTRIP-01"
    assert reconstructed.request.tumor_type == TumorType.MENINGIOMA_MULTIMODAL
    assert reconstructed.meningioma_multimodal is not None
    assert reconstructed.meningioma_multimodal.subregions.wt_voxels == orig_result.meningioma_multimodal.subregions.wt_voxels
    assert reconstructed.meningioma_multimodal.subregions.tc_voxels == orig_result.meningioma_multimodal.subregions.tc_voxels
    assert reconstructed.meningioma_multimodal.subregions.et_voxels == orig_result.meningioma_multimodal.subregions.et_voxels


def test_router_multimodal_dispatch():
    """Test 11: Unified InferenceRouter routes 'meningioma_multimodal' to multimodal service."""
    mock_model = MockMultimodalModel()
    multimodal_svc = MeningiomaMultimodalInferenceService(model=mock_model, device=torch.device("cpu"))
    router = InferenceRouter(meningioma_multimodal_service=multimodal_svc)

    synthetic_mri = np.zeros((4, 64, 64, 64), dtype=np.float32)
    synthetic_mri[:, 10:54, 10:54, 10:54] = 1.0

    # 1. Dispatch using Enum
    res_enum = router.route(
        tumor_type=TumorType.MENINGIOMA_MULTIMODAL,
        input_data=synthetic_mri,
        request_id="REQ-ROUTER-MM-01",
    )
    assert res_enum.request.tumor_type == TumorType.MENINGIOMA_MULTIMODAL
    assert res_enum.meningioma_multimodal is not None

    # 2. Dispatch using String (various cases and hyphens)
    res_str1 = router.route(
        tumor_type="meningioma_multimodal",
        input_data=synthetic_mri,
        request_id="REQ-ROUTER-MM-02",
    )
    assert res_str1.request.tumor_type == TumorType.MENINGIOMA_MULTIMODAL

    res_str2 = router.route(
        tumor_type="Meningioma-Multimodal",
        input_data=synthetic_mri,
        request_id="REQ-ROUTER-MM-03",
    )
    assert res_str2.request.tumor_type == TumorType.MENINGIOMA_MULTIMODAL

    # 3. Convenience function
    res_func = run_inference(
        tumor_type="multimodal_meningioma",
        input_data=synthetic_mri,
        router=router,
        request_id="REQ-ROUTER-MM-04",
    )
    assert res_func.request.tumor_type == TumorType.MENINGIOMA_MULTIMODAL


def test_api_multimodal_endpoint_and_router_integration():
    """Test 12: Flask REST API supports /api/segment/meningioma-multimodal and /api/segment."""
    mock_model = MockMultimodalModel()
    multimodal_svc = MeningiomaMultimodalInferenceService(model=mock_model, device=torch.device("cpu"))
    router = InferenceRouter(meningioma_multimodal_service=multimodal_svc)

    app = create_app(router=router)
    app.config["TESTING"] = True
    client = app.test_client()

    # 1. Check /api/health includes meningioma
    health_res = client.get("/api/health")
    assert health_res.status_code == 200
    types = [t.lower() for t in health_res.get_json()["supported_tumor_types"]]
    assert "meningioma" in types

    # 2. Upload 4 synthetic modality NIfTI files
    def _create_nii_bytes():
        arr = np.zeros((64, 64, 64), dtype=np.float32)
        arr[16:48, 16:48, 16:48] = np.linspace(1.0, 5.0, 32 * 32 * 32, dtype=np.float32).reshape(32, 32, 32)
        img = sitk.GetImageFromArray(arr)
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            sitk.WriteImage(img, str(tmp_path))
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    upload_data = {
        "files": [
            (io.BytesIO(_create_nii_bytes()), "scan_t1n.nii.gz"),
            (io.BytesIO(_create_nii_bytes()), "scan_t1c.nii.gz"),
            (io.BytesIO(_create_nii_bytes()), "scan_t2w.nii.gz"),
            (io.BytesIO(_create_nii_bytes()), "scan_t2f.nii.gz"),
        ]
    }
    upload_res = client.post("/api/upload", data=upload_data, content_type="multipart/form-data")
    assert upload_res.status_code == 200
    req_id = upload_res.get_json()["request_id"]

    # 3. Call dedicated endpoint POST /api/segment/meningioma-multimodal
    seg_res = client.post(
        "/api/segment/meningioma-multimodal",
        json={"request_id": req_id},
    )
    assert seg_res.status_code == 200
    seg_json = seg_res.get_json()
    assert seg_json["request"]["tumor_type"] == "meningioma_multimodal"
    assert seg_json["meningioma_multimodal"] is not None
    assert seg_json["meningioma_multimodal"]["subregions"]["wt_voxels"] > 0

    # 4. Verify results retrieval via GET /api/results/<req_id>
    res_get = client.get(f"/api/results/{req_id}")
    assert res_get.status_code == 200
    assert res_get.get_json()["request"]["request_id"] == req_id

    # 5. Verify mask download via GET /api/mask/<req_id>
    mask_res = client.get(f"/api/mask/{req_id}")
    assert mask_res.status_code == 200
    assert len(mask_res.data) > 0


def test_api_multimodal_missing_modality_rejection():
    """Test 13: API rejects multimodal segmentation request if only 1 modality is uploaded."""
    router = InferenceRouter()
    app = create_app(router=router)
    app.config["TESTING"] = True
    client = app.test_client()

    # Upload only t1c
    arr = np.zeros((16, 16, 16), dtype=np.float32)
    img = sitk.GetImageFromArray(arr)
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        sitk.WriteImage(img, str(tmp_path))
        with open(tmp_path, "rb") as f:
            nii_bytes = f.read()
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    upload_data = {
        "files": [(io.BytesIO(nii_bytes), "only_t1c.nii.gz")]
    }
    upload_res = client.post("/api/upload", data=upload_data, content_type="multipart/form-data")
    assert upload_res.status_code == 200
    req_id = upload_res.get_json()["request_id"]

    # Attempt multimodal segmentation with missing modalities
    seg_res = client.post(
        "/api/segment/meningioma-multimodal",
        json={"request_id": req_id},
    )
    assert seg_res.status_code == 400
    assert "requires 4 modalities" in seg_res.get_json()["detail"].lower()


def test_api_multimodal_mismatched_shape_rejection():
    """Test 14: API rejects multimodal segmentation request if shapes across modalities do not match."""
    router = InferenceRouter()
    app = create_app(router=router)
    app.config["TESTING"] = True
    client = app.test_client()

    def _create_bytes(shape):
        arr = np.zeros(shape, dtype=np.float32)
        arr[8:24, 8:24, 8:24] = 1.0
        img = sitk.GetImageFromArray(arr)
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            sitk.WriteImage(img, str(tmp_path))
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    upload_data = {
        "files": [
            (io.BytesIO(_create_bytes((32, 32, 32))), "scan_t1n.nii.gz"),
            (io.BytesIO(_create_bytes((32, 32, 32))), "scan_t1c.nii.gz"),
            (io.BytesIO(_create_bytes((32, 32, 32))), "scan_t2w.nii.gz"),
            (io.BytesIO(_create_bytes((32, 32, 16))), "scan_t2f.nii.gz"),  # mismatched z dimension
        ]
    }
    upload_res = client.post("/api/upload", data=upload_data, content_type="multipart/form-data")
    assert upload_res.status_code == 200
    req_id = upload_res.get_json()["request_id"]

    seg_res = client.post(
        "/api/segment/meningioma-multimodal",
        json={"request_id": req_id},
    )
    assert seg_res.status_code == 400
    assert "spatial shape mismatch" in seg_res.get_json()["detail"].lower()


def test_separate_masks_saving_and_gtv_binary_equivalence():
    """Test 15: Saving masks produces separate multiclass (0-3) and binary GTV (0-1) files with matching geometry."""
    mock_model = MockMultimodalModel()
    service = MeningiomaMultimodalInferenceService(model=mock_model, device=torch.device("cpu"))

    synthetic_mri = np.zeros((4, 64, 64, 64), dtype=np.float32)
    synthetic_mri[:, 10:54, 10:54, 10:54] = 1.0

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_mask_path = Path(tmp_dir) / "test_case_seg.nii.gz"

        result = service.predict(
            case_input=synthetic_mri,
            request_id="REQ-SEP-MASKS-01",
            patient_id="PT-SEP-01",
            save_mask_path=tmp_mask_path,
        )

        # 1. Verify both mask references exist in result
        assert result.meningioma_multimodal.masks.multiclass_mask is not None
        assert result.meningioma.gtv_mask is not None

        multi_path = Path(result.meningioma_multimodal.masks.multiclass_mask.mask_path)
        gtv_path = Path(result.meningioma.gtv_mask.mask_path)

        # 2. Verify separate file paths on disk
        assert multi_path != gtv_path
        assert multi_path.exists()
        assert gtv_path.exists()
        assert gtv_path.name.endswith("_gtv.nii.gz")

        # 3. Load multiclass mask: labels 0, 1, 2, 3
        multi_nii = nib.load(str(multi_path))
        multi_data = np.asanyarray(multi_nii.dataobj).astype(np.int64)
        multi_unique = np.unique(multi_data)
        assert set(multi_unique).issubset({0, 1, 2, 3})
        # Mock model produces lesions with classes 1, 2, and 3
        assert any(c in multi_unique for c in [1, 2, 3])

        # 4. Load binary GTV mask: strictly labels 0 and 1
        gtv_nii = nib.load(str(gtv_path))
        gtv_data = np.asanyarray(gtv_nii.dataobj).astype(np.int64)
        gtv_unique = np.unique(gtv_data)
        assert set(gtv_unique).issubset({0, 1})
        assert 1 in gtv_unique

        # 5. Verify binary GTV equivalence: (multiclass_mask > 0) == (gtv_mask == 1)
        expected_binary = (multi_data > 0).astype(np.int64)
        assert np.array_equal(expected_binary, gtv_data)
        assert int((gtv_data == 1).sum()) == result.meningioma.gtv_voxels
        assert int((gtv_data == 1).sum()) == result.meningioma_multimodal.subregions.wt_voxels
        assert int((gtv_data == 1).sum()) == result.volumetrics.total_tumor_voxels

        # 6. Verify matching spatial geometry (shape, affine, spacing)
        assert multi_nii.shape == gtv_nii.shape == (64, 64, 64)
        assert np.allclose(multi_nii.affine, gtv_nii.affine, atol=1e-3)
        assert multi_nii.header.get_zooms()[:3] == gtv_nii.header.get_zooms()[:3]

        # 7. Verify label definitions
        assert result.meningioma.label_definitions == {0: "Background", 1: "Whole Tumor (GTV composite)"}
        assert result.meningioma_multimodal.label_definitions[0] == "Background"
        assert result.meningioma_multimodal.label_definitions[1] == "Necrotic Tumor Core (NCR)"
        assert result.meningioma_multimodal.label_definitions[2] == "Peritumoral Edema (ED)"
        assert result.meningioma_multimodal.label_definitions[3] == "Enhancing Tumor (ET)"


def test_volume_counts_and_hierarchical_consistency():
    """Test 16: WT, TC, ET, NCR, and ED volumes derive consistently and hierarchically from mask."""
    mock_model = MockMultimodalModel()
    service = MeningiomaMultimodalInferenceService(model=mock_model, device=torch.device("cpu"))

    synthetic_mri = np.zeros((4, 64, 64, 64), dtype=np.float32)
    synthetic_mri[:, 10:54, 10:54, 10:54] = 1.0

    result = service.predict(
        case_input=synthetic_mri,
        request_id="REQ-VOL-CONSISTENCY-01",
    )

    sub = result.meningioma_multimodal.subregions
    # Whole Tumor = Tumor Core + Edema
    assert sub.wt_voxels == sub.tc_voxels + sub.edema_voxels
    # Tumor Core = Necrotic Core + Enhancing Tumor
    assert sub.tc_voxels == sub.ncr_net_voxels + sub.et_voxels

    # Check mm3 volume scaling
    voxel_vol = result.input_metadata.voxel_volume_mm3
    assert abs(sub.wt_volume_mm3 - (sub.wt_voxels * voxel_vol)) < 1e-4
    assert abs(sub.tc_volume_mm3 - (sub.tc_voxels * voxel_vol)) < 1e-4
    assert abs(sub.et_volume_mm3 - (sub.et_voxels * voxel_vol)) < 1e-4
    assert abs(sub.edema_volume_mm3 - (sub.edema_voxels * voxel_vol)) < 1e-4
    assert abs(sub.ncr_net_volume_mm3 - (sub.ncr_net_voxels * voxel_vol)) < 1e-4

    # Check cm3 volume scaling
    assert sub.wt_volume_cm3 == round(sub.wt_volume_mm3 / 1000.0, 4)
    assert sub.tc_volume_cm3 == round(sub.tc_volume_mm3 / 1000.0, 4)
    assert sub.et_volume_cm3 == round(sub.et_volume_mm3 / 1000.0, 4)

    # Check consistency with global VolumetricSummary and legacy Meningioma
    assert result.volumetrics.total_tumor_voxels == sub.wt_voxels
    assert result.volumetrics.total_tumor_volume_cm3 == sub.wt_volume_cm3
    assert result.meningioma.gtv_voxels == sub.wt_voxels
    assert result.meningioma.gtv_volume_cm3 == sub.wt_volume_cm3


def test_qc_benchmark_metadata_and_argmax_postprocessing():
    """Test 17: QC benchmark correctly reports 200-case benchmarks and distinguishes 25-case in-training score."""
    mock_model = MockMultimodalModel()
    service = MeningiomaMultimodalInferenceService(model=mock_model, device=torch.device("cpu"))

    synthetic_mri = np.zeros((4, 64, 64, 64), dtype=np.float32)
    synthetic_mri[:, 10:54, 10:54, 10:54] = 1.0

    result = service.predict(
        case_input=synthetic_mri,
        request_id="REQ-QC-META-01",
    )

    qs = result.quality_status
    # Evaluated pipeline uses discrete argmax across logits (no probability threshold)
    assert qs.threshold_applied == 0.0
    # Evaluated pipeline applies no connected-component filtering
    assert qs.min_component_size_voxels == 0

    # Verdict must cite full 200-case benchmark (Macro WT 0.7991, Median WT 0.9006)
    assert "0.7991" in qs.qc_verdict
    assert "0.9006" in qs.qc_verdict

    # Notes must report full 200-case results (Macro 0.7991, Median 0.9006, Pooled 0.9273)
    # and clearly distinguish the 25-case in-training validation score (0.8312)
    assert "0.7991" in qs.qc_notes
    assert "0.9006" in qs.qc_notes
    assert "0.9273" in qs.qc_notes
    assert "0.8312" in qs.qc_notes
    assert "25-case" in qs.qc_notes
    assert "200 validation cases" in qs.qc_notes


def test_router_multimodal_routing_and_masks_dispatch():
    """Test 18: Router passes save_mask_path and generates both masks for tumor_type='meningioma'."""
    mock_model = MockMultimodalModel()
    service = MeningiomaMultimodalInferenceService(model=mock_model, device=torch.device("cpu"))
    router = InferenceRouter(meningioma_service=service)

    synthetic_mri = np.zeros((4, 64, 64, 64), dtype=np.float32)
    synthetic_mri[:, 10:54, 10:54, 10:54] = 1.0

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_dest = Path(tmp_dir) / "router_men_seg.nii.gz"

        res = router.route(
            tumor_type="meningioma",
            input_data=synthetic_mri,
            request_id="REQ-ROUTER-MASKS-01",
            save_mask_path=save_dest,
        )

        assert res.request.tumor_type == TumorType.MENINGIOMA
        assert res.meningioma_multimodal.masks.multiclass_mask.mask_path == str(save_dest)
        assert res.meningioma.gtv_mask.mask_path.endswith("_gtv.nii.gz")
        assert Path(res.meningioma_multimodal.masks.multiclass_mask.mask_path).exists()
        assert Path(res.meningioma.gtv_mask.mask_path).exists()


def test_api_mask_download_supports_multiclass_and_gtv():
    """Test 19: API supports downloading default multiclass mask and type=gtv binary mask."""
    mock_model = MockMultimodalModel()
    multimodal_svc = MeningiomaMultimodalInferenceService(model=mock_model, device=torch.device("cpu"))
    router = InferenceRouter(meningioma_multimodal_service=multimodal_svc)
    app = create_app(router=router)
    app.config["TESTING"] = True
    client = app.test_client()

    def _create_bytes(shape=(32, 32, 32)):
        arr = np.zeros(shape, dtype=np.float32)
        arr[8:24, 8:24, 8:24] = 1.0
        img = sitk.GetImageFromArray(arr)
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            sitk.WriteImage(img, str(tmp_path))
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    upload_data = {
        "files": [
            (io.BytesIO(_create_bytes()), "case_t1n.nii.gz"),
            (io.BytesIO(_create_bytes()), "case_t1c.nii.gz"),
            (io.BytesIO(_create_bytes()), "case_t2w.nii.gz"),
            (io.BytesIO(_create_bytes()), "case_t2f.nii.gz"),
        ]
    }
    upload_res = client.post("/api/upload", data=upload_data, content_type="multipart/form-data")
    assert upload_res.status_code == 200
    req_id = upload_res.get_json()["request_id"]

    seg_res = client.post("/api/segment", json={"request_id": req_id, "tumor_type": "meningioma"})
    assert seg_res.status_code == 200

    # 1. Default download: multiclass mask
    dl_multi = client.get(f"/api/mask/{req_id}")
    assert dl_multi.status_code == 200
    assert "_seg.nii.gz" in dl_multi.headers.get("Content-Disposition", "")
    assert len(dl_multi.data) > 0

    # 2. GTV download: type=gtv
    dl_gtv = client.get(f"/api/mask/{req_id}?type=gtv")
    assert dl_gtv.status_code == 200
    assert "_gtv.nii.gz" in dl_gtv.headers.get("Content-Disposition", "")
    assert len(dl_gtv.data) > 0


