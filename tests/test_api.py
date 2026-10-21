"""
Test Suite for Backend REST API.
================================

Verifies all API endpoints using Flask TestClient with mocked and synthetic inputs:
  - GET  /api/health
  - POST /api/upload (valid, unsupported, corrupt)
  - POST /api/segment (Glioma, Meningioma, missing modalities, invalid types)
  - GET  /api/results/<request_id>
  - GET  /api/mask/<request_id>
  - Error handling and safe exception responses
"""

import io
import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any
from unittest.mock import MagicMock

import numpy as np
import nibabel as nib
import pytest
import SimpleITK as sitk

from api.app import create_app, API_RUNS_DIR
from integration.result_schema import (
    TumorType,
    PipelineStatus,
    RequestMetadata,
    InputMetadata,
    VolumetricSummary,
    GliomaMeasurements,
    GliomaSubregionVolumes,
    GliomaSubregionMasks,
    MaskReference,
    MeningiomaMeasurements,
    MeningiomaMultimodalMeasurements,
    MeningiomaMultimodalSubregionVolumes,
    InferenceQualityStatus,
    SegmentationResult,
)


def create_synthetic_nifti_bytes(shape=(16, 16, 16), spacing=(1.0, 1.0, 1.0)) -> bytes:
    """Creates an in-memory gzipped NIfTI image for testing."""
    arr = np.random.randn(*shape).astype(np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)

    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        sitk.WriteImage(img, str(tmp_path))
        with open(tmp_path, "rb") as f:
            data = f.read()
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return data


def create_synthetic_nifti_bytes_with_affine(
    shape=(16, 16, 16),
    affine=None,
    spacing=(1.0, 1.0, 1.0),
) -> bytes:
    """Creates an in-memory gzipped NIfTI image with explicit affine, shape, and zooms."""
    arr = np.random.randn(*shape).astype(np.float32)
    if affine is None:
        affine = np.array([
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)
    nii = nib.Nifti1Image(arr, affine=affine)
    nii.header.set_zooms(tuple(float(z) for z in spacing[:3]))

    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        nib.save(nii, str(tmp_path))
        with open(tmp_path, "rb") as f:
            data = f.read()
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return data


def create_mock_glioma_result(request_id: str) -> SegmentationResult:
    return SegmentationResult(
        request=RequestMetadata(request_id=request_id, tumor_type=TumorType.GLIOMA),
        input_metadata=InputMetadata(
            modality_names=["T1", "T1c", "T2", "FLAIR"],
            original_shape=[16, 16, 16],
            original_spacing=[1.0, 1.0, 1.0],
            voxel_volume_mm3=1.0,
            resampled_shape=[16, 16, 16],
            resampled_spacing=[1.0, 1.0, 1.0],
        ),
        volumetrics=VolumetricSummary(
            total_tumor_voxels=500,
            total_tumor_volume_mm3=500.0,
            total_tumor_volume_cm3=0.5,
            voxel_dimensions_mm=[1.0, 1.0, 1.0],
        ),
        quality_status=InferenceQualityStatus(
            status=PipelineStatus.SUCCESS,
            inference_success=True,
            pipeline_frozen=True,
            checkpoint_used="outputs/glioma_training/glioma_best_medium_experiment.pth",
            threshold_applied=0.60,
            min_component_size_voxels=1000,
            qc_verdict="PASS WITH DIFFICULT CASES",
        ),
        glioma=GliomaMeasurements(
            subregions=GliomaSubregionVolumes(
                wt_voxels=500,
                wt_volume_mm3=500.0,
                wt_volume_cm3=0.5,
                tc_voxels=200,
                tc_volume_mm3=200.0,
                tc_volume_cm3=0.2,
                et_voxels=100,
                et_volume_mm3=100.0,
                et_volume_cm3=0.1,
                ncr_net_voxels=100,
                ncr_net_volume_mm3=100.0,
                ncr_net_volume_cm3=0.1,
                edema_voxels=300,
                edema_volume_mm3=300.0,
                edema_volume_cm3=0.3,
            ),
            masks=GliomaSubregionMasks(
                multiclass_mask=MaskReference(
                    mask_path="dummy_path.nii.gz",
                    mask_format="nifti",
                    is_original_space=True,
                    shape=[16, 16, 16],
                    non_zero_voxels=500,
                )
            ),
        ),
        meningioma=None,
    )


def create_mock_meningioma_result(request_id: str) -> SegmentationResult:
    return SegmentationResult(
        request=RequestMetadata(request_id=request_id, tumor_type=TumorType.MENINGIOMA),
        input_metadata=InputMetadata(
            modality_names=["t1n", "t1c", "t2w", "t2f"],
            original_shape=[16, 16, 16],
            original_spacing=[1.0, 1.0, 1.0],
            voxel_volume_mm3=1.0,
            resampled_shape=[16, 16, 16],
            resampled_spacing=[1.0, 1.0, 1.0],
        ),
        volumetrics=VolumetricSummary(
            total_tumor_voxels=800,
            total_tumor_volume_mm3=800.0,
            total_tumor_volume_cm3=0.8,
            voxel_dimensions_mm=[1.0, 1.0, 1.0],
        ),
        quality_status=InferenceQualityStatus(
            status=PipelineStatus.SUCCESS,
            inference_success=True,
            pipeline_frozen=True,
            checkpoint_used="outputs/meningioma_multimodal_experiment/full800_brainaware_20261014/checkpoints/meningioma_multimodal_best.pth",
            threshold_applied=0.0,
            min_component_size_voxels=0,
            qc_verdict="PASS WITH VALIDATION BENCHMARK (0.7991 Macro WT Dice / 0.9006 Median WT Dice)",
        ),
        glioma=None,
        meningioma=MeningiomaMeasurements(
            gtv_voxels=800,
            gtv_volume_mm3=800.0,
            gtv_volume_cm3=0.8,
            gtv_mask=MaskReference(
                mask_path="dummy_men_gtv_path.nii.gz",
                mask_format="nifti",
                is_original_space=True,
                shape=[16, 16, 16],
                non_zero_voxels=800,
            ),
        ),
        meningioma_multimodal=MeningiomaMultimodalMeasurements(
            subregions=MeningiomaMultimodalSubregionVolumes(
                wt_voxels=800,
                wt_volume_mm3=800.0,
                wt_volume_cm3=0.8,
                tc_voxels=400,
                tc_volume_mm3=400.0,
                tc_volume_cm3=0.4,
                et_voxels=200,
                et_volume_mm3=200.0,
                et_volume_cm3=0.2,
            ),
            masks=GliomaSubregionMasks(
                multiclass_mask=MaskReference(
                    mask_path="dummy_men_multiclass_path.nii.gz",
                    mask_format="nifti",
                    is_original_space=True,
                    shape=[16, 16, 16],
                    non_zero_voxels=800,
                )
            ),
        ),
    )


@pytest.fixture
def mock_router():
    """Provides a mock inference router that simulates fast dispatch without loading GPU models."""
    router = MagicMock()

    def route_side_effect(tumor_type, input_data, request_id, save_mask_path=None, **kwargs):
        if save_mask_path:
            p = Path(save_mask_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            # Write a dummy NIfTI mask with classes 0, 2, 3
            dummy_arr = np.zeros((16, 16, 16), dtype=np.uint8)
            dummy_arr[4:12, 4:12, 4:12] = 2
            dummy_arr[6:10, 6:10, 6:10] = 3
            dummy_img = sitk.GetImageFromArray(dummy_arr)
            sitk.WriteImage(dummy_img, str(p))

        if tumor_type == TumorType.GLIOMA:
            return create_mock_glioma_result(request_id)
        elif tumor_type in (TumorType.MENINGIOMA, TumorType.MENINGIOMA_MULTIMODAL):
            return create_mock_meningioma_result(request_id)
        elif tumor_type in (TumorType.AUTO, "auto"):
            cs = getattr(router, "classifier_service", None)
            if cs:
                eval_res = cs.classify(input_data if isinstance(input_data, dict) else {})
                if eval_res.get("status") == "automatic_classification_not_available":
                    from integration.inference_router import AutomaticClassificationUnavailableError
                    raise AutomaticClassificationUnavailableError(
                        message=eval_res.get("reason", "Automatic classification not available."),
                        details=eval_res,
                    )
            return create_mock_glioma_result(request_id)
        raise ValueError(f"Unknown tumor type: {tumor_type}")

    def normalize_side_effect(t):
        if isinstance(t, TumorType):
            return t
        if isinstance(t, str):
            clean = t.strip().lower()
            if clean in ["glioma", "meningioma", "meningioma_multimodal", "auto"]:
                return TumorType(clean)
        raise ValueError(f"Unsupported tumor type: '{t}'")

    router.normalize_tumor_type.side_effect = normalize_side_effect
    router.route.side_effect = route_side_effect
    return router


@pytest.fixture
def client(mock_router):
    """Provides a Flask test client configured with the mock router."""
    app = create_app(router=mock_router)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0"
    types_lower = [t.lower() for t in data["supported_tumor_types"]]
    assert "glioma" in types_lower
    assert "meningioma" in types_lower


def test_upload_valid_nifti(client):
    nii_bytes = create_synthetic_nifti_bytes()
    data = {
        "files": (io.BytesIO(nii_bytes), "case_t1c.nii.gz")
    }
    response = client.post("/api/upload", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    res_data = response.get_json()
    assert res_data["status"] == "uploaded"
    assert res_data["files_count"] == 1
    assert "T1c" in res_data["detected_modalities"]
    assert "request_id" in res_data


def test_upload_unsupported_file(client):
    data = {
        "files": (io.BytesIO(b"fake-png-bytes"), "image.png")
    }
    response = client.post("/api/upload", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert "unsupported format" in response.get_json()["detail"].lower()


def test_upload_corrupt_nifti(client):
    data = {
        "files": (io.BytesIO(b"corrupted-header-data-12345"), "corrupt.nii.gz")
    }
    response = client.post("/api/upload", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert "corrupted" in response.get_json()["detail"].lower()


def test_segment_glioma_dispatch(client):
    # 1. Upload 4 modalities
    nii_bytes = create_synthetic_nifti_bytes()
    data = {
        "files": [
            (io.BytesIO(nii_bytes), "case_t1.nii.gz"),
            (io.BytesIO(nii_bytes), "case_t1c.nii.gz"),
            (io.BytesIO(nii_bytes), "case_t2.nii.gz"),
            (io.BytesIO(nii_bytes), "case_flair.nii.gz"),
        ]
    }
    upload_res = client.post("/api/upload", data=data, content_type="multipart/form-data")
    assert upload_res.status_code == 200
    req_id = upload_res.get_json()["request_id"]

    # 2. Trigger segmentation
    seg_res = client.post("/api/segment", json={"request_id": req_id, "tumor_type": "GLIOMA"})
    assert seg_res.status_code == 200
    res_data = seg_res.get_json()
    assert res_data["request"]["tumor_type"] == "glioma"
    assert res_data["glioma"] is not None
    assert res_data["volumetrics"]["total_tumor_volume_cm3"] == 0.5


def test_segment_meningioma_dispatch(client):
    # 1. Upload all 4 modalities
    nii_bytes = create_synthetic_nifti_bytes()
    data = {
        "files": [
            (io.BytesIO(nii_bytes), "BraTS-MEN-0001_t1n.nii.gz"),
            (io.BytesIO(nii_bytes), "BraTS-MEN-0001_t1c.nii.gz"),
            (io.BytesIO(nii_bytes), "BraTS-MEN-0001_t2w.nii.gz"),
            (io.BytesIO(nii_bytes), "BraTS-MEN-0001_t2f.nii.gz"),
        ]
    }
    upload_res = client.post("/api/upload", data=data, content_type="multipart/form-data")
    assert upload_res.status_code == 200
    req_id = upload_res.get_json()["request_id"]

    # 2. Trigger segmentation
    seg_res = client.post("/api/segment", json={"request_id": req_id, "tumor_type": "MENINGIOMA"})
    assert seg_res.status_code == 200
    res_data = seg_res.get_json()
    assert res_data["request"]["tumor_type"] == "meningioma"
    assert res_data["meningioma"] is not None
    assert res_data["meningioma_multimodal"] is not None
    assert res_data["volumetrics"]["total_tumor_volume_cm3"] == 0.8


def test_segment_missing_modality_meningioma(client):
    # Upload only 1 modality and request Meningioma
    nii_bytes = create_synthetic_nifti_bytes()
    data = {
        "files": [
            (io.BytesIO(nii_bytes), "BraTS-MEN-0001_t1c.nii.gz"),
        ]
    }
    upload_res = client.post("/api/upload", data=data, content_type="multipart/form-data")
    req_id = upload_res.get_json()["request_id"]

    seg_res = client.post("/api/segment", json={"request_id": req_id, "tumor_type": "MENINGIOMA"})
    assert seg_res.status_code == 400
    assert "requires 4 modalities" in seg_res.get_json()["detail"].lower()


def test_segment_missing_modality_glioma(client):
    # Upload only 1 modality and request Glioma
    nii_bytes = create_synthetic_nifti_bytes()
    data = {
        "files": [
            (io.BytesIO(nii_bytes), "case_t1c.nii.gz"),
        ]
    }
    upload_res = client.post("/api/upload", data=data, content_type="multipart/form-data")
    req_id = upload_res.get_json()["request_id"]

    seg_res = client.post("/api/segment", json={"request_id": req_id, "tumor_type": "GLIOMA"})
    assert seg_res.status_code == 400
    assert "requires 4 modalities" in seg_res.get_json()["detail"].lower()


def test_segment_invalid_tumor_type(client):
    nii_bytes = create_synthetic_nifti_bytes()
    data = {
        "files": [
            (io.BytesIO(nii_bytes), "case_t1c.nii.gz"),
        ]
    }
    upload_res = client.post("/api/upload", data=data, content_type="multipart/form-data")
    req_id = upload_res.get_json()["request_id"]

    seg_res = client.post("/api/segment", json={"request_id": req_id, "tumor_type": "UNKNOWN_TUMOR"})
    assert seg_res.status_code == 400
    assert "unsupported tumor type" in seg_res.get_json()["detail"].lower()


def test_segment_unknown_request_id(client):
    seg_res = client.post("/api/segment", json={"request_id": "NON_EXISTENT_ID", "tumor_type": "GLIOMA"})
    assert seg_res.status_code == 404
    assert "not found" in seg_res.get_json()["detail"].lower()


def test_get_results_and_download_mask(client):
    # Upload, segment, retrieve result and mask
    nii_bytes = create_synthetic_nifti_bytes()
    data = {
        "files": [
            (io.BytesIO(nii_bytes), "scan_t1n.nii.gz"),
            (io.BytesIO(nii_bytes), "scan_t1c.nii.gz"),
            (io.BytesIO(nii_bytes), "scan_t2w.nii.gz"),
            (io.BytesIO(nii_bytes), "scan_t2f.nii.gz"),
        ]
    }
    upload_res = client.post("/api/upload", data=data, content_type="multipart/form-data")
    req_id = upload_res.get_json()["request_id"]

    client.post("/api/segment", json={"request_id": req_id, "tumor_type": "MENINGIOMA"})

    # Get results
    res_get = client.get(f"/api/results/{req_id}")
    assert res_get.status_code == 200
    assert res_get.get_json()["request"]["request_id"] == req_id

    # Download mask
    mask_res = client.get(f"/api/mask/{req_id}")
    assert mask_res.status_code == 200
    assert len(mask_res.data) > 0


def test_mocked_inference_failure(client, mock_router):
    # Configure mock to raise an error
    mock_router.route.side_effect = RuntimeError("GPU Out of Memory or Internal Inference Fault")

    nii_bytes = create_synthetic_nifti_bytes()
    data = {
        "files": [
            (io.BytesIO(nii_bytes), "scan_t1n.nii.gz"),
            (io.BytesIO(nii_bytes), "scan_t1c.nii.gz"),
            (io.BytesIO(nii_bytes), "scan_t2w.nii.gz"),
            (io.BytesIO(nii_bytes), "scan_t2f.nii.gz"),
        ]
    }
    upload_res = client.post("/api/upload", data=data, content_type="multipart/form-data")
    req_id = upload_res.get_json()["request_id"]

    seg_res = client.post("/api/segment", json={"request_id": req_id, "tumor_type": "MENINGIOMA"})
    assert seg_res.status_code == 500
    assert "inference execution failed" in seg_res.get_json()["detail"].lower()


def test_upload_with_traversal_or_malformed_request_id(client):
    nii_bytes = create_synthetic_nifti_bytes()
    # 1. Path traversal attempt
    data_traversal = {
        "files": (io.BytesIO(nii_bytes), "scan_t1c.nii.gz"),
        "request_id": "../traversal_attack",
    }
    res_trav = client.post("/api/upload", data=data_traversal, content_type="multipart/form-data")
    assert res_trav.status_code == 400
    assert "invalid request id" in res_trav.get_json()["detail"].lower()

    # 2. Malformed characters
    data_malformed = {
        "files": (io.BytesIO(nii_bytes), "scan_t1c.nii.gz"),
        "request_id": "REQ@bad#id$",
    }
    res_mal = client.post("/api/upload", data=data_malformed, content_type="multipart/form-data")
    assert res_mal.status_code == 400
    assert "invalid request id" in res_mal.get_json()["detail"].lower()

    # 3. Oversized ID (>64 chars)
    data_oversized = {
        "files": (io.BytesIO(nii_bytes), "scan_t1c.nii.gz"),
        "request_id": "A" * 65,
    }
    res_over = client.post("/api/upload", data=data_oversized, content_type="multipart/form-data")
    assert res_over.status_code == 400
    assert "invalid request id" in res_over.get_json()["detail"].lower()


def test_segment_with_traversal_or_malformed_request_id(client):
    # 1. Path traversal in JSON
    res_trav = client.post("/api/segment", json={"request_id": "../../etc/passwd", "tumor_type": "GLIOMA"})
    assert res_trav.status_code == 400
    assert "invalid request id" in res_trav.get_json()["detail"].lower()

    # 2. Empty string
    res_empty = client.post("/api/segment", json={"request_id": "   ", "tumor_type": "GLIOMA"})
    assert res_empty.status_code == 400
    assert "request id cannot be empty" in res_empty.get_json()["detail"].lower()

    # 3. Special characters
    res_spec = client.post("/api/segment", json={"request_id": "REQ/SUB_DIR", "tumor_type": "MENINGIOMA"})
    assert res_spec.status_code == 400
    assert "invalid request id" in res_spec.get_json()["detail"].lower()


def test_get_results_path_traversal_and_malformed_id(client):
    # Malformed ID -> 400
    res_mal = client.get("/api/results/bad@id!123")
    assert res_mal.status_code == 400
    assert "invalid request id" in res_mal.get_json()["detail"].lower()

    # Well-formed but non-existent ID -> 404
    res_404 = client.get("/api/results/REQ-WELLFORMED-BUT-NONEXISTENT")
    assert res_404.status_code == 404
    assert "not found" in res_404.get_json()["detail"].lower()


def test_download_mask_path_traversal_and_malformed_id(client):
    # Malformed ID -> 400
    res_mal = client.get("/api/mask/bad@id!123")
    assert res_mal.status_code == 400
    assert "invalid request id" in res_mal.get_json()["detail"].lower()

    # Well-formed but non-existent ID -> 404
    res_404 = client.get("/api/mask/REQ-WELLFORMED-BUT-NONEXISTENT")
    assert res_404.status_code == 404
    assert "not found" in res_404.get_json()["detail"].lower()


def test_valid_custom_request_id_workflow(client):
    nii_bytes = create_synthetic_nifti_bytes()
    custom_id = "REQ-CUSTOM_2026-TEST_01"
    data = {
        "files": [
            (io.BytesIO(nii_bytes), "case_t1n.nii.gz"),
            (io.BytesIO(nii_bytes), "case_t1c.nii.gz"),
            (io.BytesIO(nii_bytes), "case_t2w.nii.gz"),
            (io.BytesIO(nii_bytes), "case_t2f.nii.gz"),
        ],
        "request_id": custom_id,
    }
    upload_res = client.post("/api/upload", data=data, content_type="multipart/form-data")
    assert upload_res.status_code == 200
    assert upload_res.get_json()["request_id"] == custom_id

    seg_res = client.post("/api/segment", json={"request_id": custom_id, "tumor_type": "MENINGIOMA"})
    assert seg_res.status_code == 200
    assert seg_res.get_json()["request"]["request_id"] == custom_id

    res_get = client.get(f"/api/results/{custom_id}")
    assert res_get.status_code == 200
    assert res_get.get_json()["request"]["request_id"] == custom_id


def test_download_mask_gtv_semantics_for_glioma_and_meningioma(client, tmp_path):
    """
    Verifies that:
    1. Default /api/mask/<req_id> returns the multiclass mask (_seg.nii.gz).
    2. /api/mask/<req_id>?type=gtv returns a strictly binary GTV mask (labels 0 and 1 only) matching multiclass > 0.
    3. /api/mask/<req_id>?type=invalid returns 400 Bad Request with an informative error.
    """
    nii_bytes = create_synthetic_nifti_bytes()
    data = {
        "files": [
            (io.BytesIO(nii_bytes), "case_t1.nii.gz"),
            (io.BytesIO(nii_bytes), "case_t1c.nii.gz"),
            (io.BytesIO(nii_bytes), "case_t2.nii.gz"),
            (io.BytesIO(nii_bytes), "case_flair.nii.gz"),
        ]
    }
    upload_res = client.post("/api/upload", data=data, content_type="multipart/form-data")
    assert upload_res.status_code == 200
    req_id = upload_res.get_json()["request_id"]

    seg_res = client.post("/api/segment", json={"request_id": req_id, "tumor_type": "GLIOMA"})
    assert seg_res.status_code == 200

    # 1. Default request -> multiclass
    res_multi = client.get(f"/api/mask/{req_id}")
    assert res_multi.status_code == 200
    assert "_glioma_seg.nii.gz" in res_multi.headers.get("Content-Disposition", "")
    temp_multi = tmp_path / "downloaded_multi.nii.gz"
    temp_multi.write_bytes(res_multi.data)
    img_multi = sitk.ReadImage(str(temp_multi))
    arr_multi = sitk.GetArrayFromImage(img_multi)
    assert set(np.unique(arr_multi)).issubset({0, 1, 2, 3})
    assert any(c in np.unique(arr_multi) for c in [2, 3])

    # 2. type=gtv -> strictly binary (labels 0 and 1 only)
    res_gtv = client.get(f"/api/mask/{req_id}?type=gtv")
    assert res_gtv.status_code == 200
    assert "_glioma_gtv.nii.gz" in res_gtv.headers.get("Content-Disposition", "")
    temp_gtv = tmp_path / "downloaded_gtv.nii.gz"
    temp_gtv.write_bytes(res_gtv.data)
    img_gtv = sitk.ReadImage(str(temp_gtv))
    arr_gtv = sitk.GetArrayFromImage(img_gtv)
    assert set(np.unique(arr_gtv)).issubset({0, 1})
    assert 1 in np.unique(arr_gtv)
    # Check binary equivalence
    assert np.array_equal((arr_multi > 0).astype(np.uint8), arr_gtv)

    # 3. type=invalid -> 400
    res_invalid = client.get(f"/api/mask/{req_id}?type=unknown_type")
    assert res_invalid.status_code == 400
    assert "invalid mask type" in res_invalid.get_json()["detail"].lower()


def test_segment_duplicate_modality_detection_rejection(client):
    """Test: API rejects requests where multiple uploaded files resolve to the same modality."""
    nii_bytes = create_synthetic_nifti_bytes()
    # Upload 2 files resolving to T1 for Glioma
    data_gli = {
        "files": [
            (io.BytesIO(nii_bytes), "patient_t1_v1.nii.gz"),
            (io.BytesIO(nii_bytes), "patient_t1_v2.nii.gz"),
            (io.BytesIO(nii_bytes), "patient_t1c.nii.gz"),
            (io.BytesIO(nii_bytes), "patient_t2.nii.gz"),
            (io.BytesIO(nii_bytes), "patient_flair.nii.gz"),
        ]
    }
    upload_res = client.post("/api/upload", data=data_gli, content_type="multipart/form-data")
    assert upload_res.status_code == 200
    req_id = upload_res.get_json()["request_id"]

    seg_res = client.post("/api/segment", json={"request_id": req_id, "tumor_type": "glioma"})
    assert seg_res.status_code == 400
    assert "duplicate modality" in seg_res.get_json()["detail"].lower()

    # Upload 2 files resolving to t1n for Meningioma
    data_men = {
        "files": [
            (io.BytesIO(nii_bytes), "scan_t1n_a.nii.gz"),
            (io.BytesIO(nii_bytes), "scan_t1n_b.nii.gz"),
            (io.BytesIO(nii_bytes), "scan_t1c.nii.gz"),
            (io.BytesIO(nii_bytes), "scan_t2w.nii.gz"),
            (io.BytesIO(nii_bytes), "scan_t2f.nii.gz"),
        ]
    }
    upload_res2 = client.post("/api/upload", data=data_men, content_type="multipart/form-data")
    assert upload_res2.status_code == 200
    req_id2 = upload_res2.get_json()["request_id"]

    seg_res2 = client.post("/api/segment", json={"request_id": req_id2, "tumor_type": "meningioma"})
    assert seg_res2.status_code == 400
    assert "duplicate modality" in seg_res2.get_json()["detail"].lower()


def test_segment_unexpected_modality_and_unrecognized_file_rejection(client):
    """Test: API rejects unrecognized files in auto-detect and unexpected modalities in modality_map."""
    nii_bytes = create_synthetic_nifti_bytes()
    # 1. Unrecognized file in auto-detect
    data_unrec = {
        "files": [
            (io.BytesIO(nii_bytes), "unknown_named_scan.nii.gz"),
            (io.BytesIO(nii_bytes), "case_t1c.nii.gz"),
            (io.BytesIO(nii_bytes), "case_t2.nii.gz"),
            (io.BytesIO(nii_bytes), "case_flair.nii.gz"),
        ]
    }
    upload_res = client.post("/api/upload", data=data_unrec, content_type="multipart/form-data")
    assert upload_res.status_code == 200
    req_id = upload_res.get_json()["request_id"]

    seg_res = client.post("/api/segment", json={"request_id": req_id, "tumor_type": "glioma"})
    assert seg_res.status_code == 400
    assert "unrecognized file" in seg_res.get_json()["detail"].lower()

    # 2. Unexpected modality in modality_map
    data_valid = {
        "files": [
            (io.BytesIO(nii_bytes), "s1.nii.gz"),
            (io.BytesIO(nii_bytes), "s2.nii.gz"),
            (io.BytesIO(nii_bytes), "s3.nii.gz"),
            (io.BytesIO(nii_bytes), "s4.nii.gz"),
        ]
    }
    upload_res2 = client.post("/api/upload", data=data_valid, content_type="multipart/form-data")
    assert upload_res2.status_code == 200
    req_id2 = upload_res2.get_json()["request_id"]

    seg_res2 = client.post("/api/segment", json={
        "request_id": req_id2,
        "tumor_type": "glioma",
        "modality_map": {
            "T1": "s1.nii.gz",
            "T1c": "s2.nii.gz",
            "T2": "s3.nii.gz",
            "FLAIR": "s4.nii.gz",
            "DWI": "s4.nii.gz",  # unexpected
        }
    })
    assert seg_res2.status_code == 400
    assert "unexpected modality" in seg_res2.get_json()["detail"].lower()


def test_segment_malformed_request_body_rejection(client):
    """Test: API rejects malformed request bodies and parameter structures."""
    # 1. Non-dict JSON payload
    res_list = client.post("/api/segment", json=["not", "a", "dict"])
    assert res_list.status_code == 400
    assert "must be a valid json object" in res_list.get_json()["detail"].lower()

    # 2. Missing tumor_type
    res_no_tt = client.post("/api/segment", json={"request_id": "REQ-12345678"})
    assert res_no_tt.status_code == 400
    assert "missing required field 'tumor_type'" in res_no_tt.get_json()["detail"].lower()

    # 3. Invalid modality_map type
    res_bad_map = client.post("/api/segment", json={
        "request_id": "REQ-12345678",
        "tumor_type": "glioma",
        "modality_map": "not_a_dict",
    })
    assert res_bad_map.status_code == 400
    assert "must be a json dictionary" in res_bad_map.get_json()["detail"].lower()

    # 4. Nonexistent file in modality_map
    nii_bytes = create_synthetic_nifti_bytes()
    upload_res = client.post("/api/upload", data={"files": [(io.BytesIO(nii_bytes), "scan_t1.nii.gz")]}, content_type="multipart/form-data")
    req_id = upload_res.get_json()["request_id"]

    res_missing_file = client.post("/api/segment", json={
        "request_id": req_id,
        "tumor_type": "glioma",
        "modality_map": {
            "T1": "nonexistent_file.nii.gz",
            "T1c": "scan_t1.nii.gz",
            "T2": "scan_t1.nii.gz",
            "FLAIR": "scan_t1.nii.gz",
        }
    })
    assert res_missing_file.status_code == 400
    assert "was not found in uploads" in res_missing_file.get_json()["detail"].lower()


def test_segment_model_selection_isolation(client):
    """Test: Verifies that Glioma requests route only to Glioma model and Meningioma only to Meningioma model."""
    nii_bytes = create_synthetic_nifti_bytes()

    # 1. Test Glioma selection
    gli_files = {
        "files": [
            (io.BytesIO(nii_bytes), "gli_t1.nii.gz"),
            (io.BytesIO(nii_bytes), "gli_t1c.nii.gz"),
            (io.BytesIO(nii_bytes), "gli_t2.nii.gz"),
            (io.BytesIO(nii_bytes), "gli_flair.nii.gz"),
        ]
    }
    res_up_gli = client.post("/api/upload", data=gli_files, content_type="multipart/form-data")
    req_gli = res_up_gli.get_json()["request_id"]

    res_seg_gli = client.post("/api/segment", json={"request_id": req_gli, "tumor_type": "glioma"})
    assert res_seg_gli.status_code == 200
    data_gli = res_seg_gli.get_json()
    assert data_gli["request"]["tumor_type"] == "glioma"
    assert data_gli["glioma"] is not None
    assert data_gli["meningioma"] is None
    assert data_gli["meningioma_multimodal"] is None
    assert "glioma_best_medium_experiment.pth" in data_gli["quality_status"]["checkpoint_used"]

    # 2. Test Meningioma selection
    men_files = {
        "files": [
            (io.BytesIO(nii_bytes), "men_t1n.nii.gz"),
            (io.BytesIO(nii_bytes), "men_t1c.nii.gz"),
            (io.BytesIO(nii_bytes), "men_t2w.nii.gz"),
            (io.BytesIO(nii_bytes), "men_t2f.nii.gz"),
        ]
    }
    res_up_men = client.post("/api/upload", data=men_files, content_type="multipart/form-data")
    req_men = res_up_men.get_json()["request_id"]

    res_seg_men = client.post("/api/segment", json={"request_id": req_men, "tumor_type": "meningioma"})
    assert res_seg_men.status_code == 200
    data_men = res_seg_men.get_json()
    assert data_men["request"]["tumor_type"] == "meningioma"
    assert data_men["meningioma"] is not None
    assert data_men["meningioma_multimodal"] is not None
    assert data_men["glioma"] is None
    assert "meningioma_multimodal_best.pth" in data_men["quality_status"]["checkpoint_used"]


def test_segment_backward_compatibility_multimodal_endpoint(client):
    """Test: Dedicated /api/segment/meningioma-multimodal remains backwards compatible."""
    nii_bytes = create_synthetic_nifti_bytes()
    men_files = {
        "files": [
            (io.BytesIO(nii_bytes), "legacy_t1n.nii.gz"),
            (io.BytesIO(nii_bytes), "legacy_t1c.nii.gz"),
            (io.BytesIO(nii_bytes), "legacy_t2w.nii.gz"),
            (io.BytesIO(nii_bytes), "legacy_t2f.nii.gz"),
        ]
    }
    res_up = client.post("/api/upload", data=men_files, content_type="multipart/form-data")
    req_id = res_up.get_json()["request_id"]

    res_seg = client.post("/api/segment/meningioma-multimodal", json={"request_id": req_id})
    assert res_seg.status_code == 200
    data = res_seg.get_json()
    assert data["meningioma"] is not None
    assert data["meningioma_multimodal"] is not None
    assert "meningioma_multimodal_best.pth" in data["quality_status"]["checkpoint_used"]


def test_segment_geometry_validation_shape_mismatch_rejected(client, mock_router):
    """Test: Reject segmentation if uploaded modalities have discordant spatial shapes."""
    mock_router.route.reset_mock()
    std_bytes = create_synthetic_nifti_bytes(shape=(16, 16, 16))
    bad_bytes = create_synthetic_nifti_bytes(shape=(16, 16, 12))

    files = {
        "files": [
            (io.BytesIO(std_bytes), "case_t1.nii.gz"),
            (io.BytesIO(std_bytes), "case_t1c.nii.gz"),
            (io.BytesIO(std_bytes), "case_t2.nii.gz"),
            (io.BytesIO(bad_bytes), "case_flair.nii.gz"),
        ]
    }
    up_res = client.post("/api/upload", data=files, content_type="multipart/form-data")
    assert up_res.status_code == 200
    req_id = up_res.get_json()["request_id"]

    seg_res = client.post("/api/segment", json={"request_id": req_id, "tumor_type": "glioma"})
    assert seg_res.status_code == 400
    detail = seg_res.get_json()["detail"]
    assert "MRI geometry validation failed" in detail
    assert "Spatial shape mismatch for modality 'FLAIR'" in detail

    # Verify inference router was NOT invoked
    mock_router.route.assert_not_called()


def test_segment_geometry_validation_spacing_mismatch_rejected(client, mock_router):
    """Test: Reject segmentation if uploaded modalities have discordant voxel resolutions."""
    mock_router.route.reset_mock()
    std_bytes = create_synthetic_nifti_bytes_with_affine(spacing=(1.0, 1.0, 1.0))
    bad_bytes = create_synthetic_nifti_bytes_with_affine(spacing=(1.0, 1.0, 2.5))

    files = {
        "files": [
            (io.BytesIO(std_bytes), "case_t1.nii.gz"),
            (io.BytesIO(std_bytes), "case_t1c.nii.gz"),
            (io.BytesIO(bad_bytes), "case_t2.nii.gz"),
            (io.BytesIO(std_bytes), "case_flair.nii.gz"),
        ]
    }
    up_res = client.post("/api/upload", data=files, content_type="multipart/form-data")
    assert up_res.status_code == 200
    req_id = up_res.get_json()["request_id"]

    seg_res = client.post("/api/segment", json={"request_id": req_id, "tumor_type": "glioma"})
    assert seg_res.status_code == 400
    detail = seg_res.get_json()["detail"]
    assert "MRI geometry validation failed" in detail
    assert "Voxel spacing mismatch for modality 'T2'" in detail

    # Verify inference router was NOT invoked
    mock_router.route.assert_not_called()


def test_segment_geometry_validation_affine_mismatch_rejected(client, mock_router):
    """Test: Reject segmentation if uploaded modalities have discordant coordinate affines."""
    mock_router.route.reset_mock()
    std_aff = np.array([
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)
    bad_aff = std_aff.copy()
    bad_aff[0, 3] = 20.0  # Physical origin shift of 20 mm

    std_bytes = create_synthetic_nifti_bytes_with_affine(affine=std_aff)
    bad_bytes = create_synthetic_nifti_bytes_with_affine(affine=bad_aff)

    files = {
        "files": [
            (io.BytesIO(std_bytes), "men_t1n.nii.gz"),
            (io.BytesIO(std_bytes), "men_t1c.nii.gz"),
            (io.BytesIO(std_bytes), "men_t2w.nii.gz"),
            (io.BytesIO(bad_bytes), "men_t2f.nii.gz"),
        ]
    }
    up_res = client.post("/api/upload", data=files, content_type="multipart/form-data")
    assert up_res.status_code == 200
    req_id = up_res.get_json()["request_id"]

    seg_res = client.post("/api/segment", json={"request_id": req_id, "tumor_type": "meningioma"})
    assert seg_res.status_code == 400
    detail = seg_res.get_json()["detail"]
    assert "MRI geometry validation failed" in detail
    assert "Spatial affine mismatch for modality 't2f'" in detail

    # Verify inference router was NOT invoked
    mock_router.route.assert_not_called()


def test_segment_geometry_validation_non_3d_volume_rejected(client, mock_router):
    """Test: Reject segmentation if an uploaded file is not a 3D spatial volume."""
    mock_router.route.reset_mock()
    std_bytes = create_synthetic_nifti_bytes()

    files = {
        "files": [
            (io.BytesIO(std_bytes), "vol_t1.nii.gz"),
            (io.BytesIO(std_bytes), "vol_t1c.nii.gz"),
            (io.BytesIO(std_bytes), "vol_t2.nii.gz"),
            (io.BytesIO(std_bytes), "vol_flair.nii.gz"),
        ]
    }
    up_res = client.post("/api/upload", data=files, content_type="multipart/form-data")
    assert up_res.status_code == 200
    req_id = up_res.get_json()["request_id"]

    # Overwrite one modality in the upload directory with a 4D volume
    target_file = API_RUNS_DIR / req_id / "uploads" / "vol_flair.nii.gz"
    arr_4d = np.zeros((2, 16, 16, 16), dtype=np.float32)
    nib.save(nib.Nifti1Image(arr_4d, np.eye(4)), str(target_file))

    seg_res = client.post("/api/segment", json={"request_id": req_id, "tumor_type": "glioma"})
    assert seg_res.status_code == 400
    detail = seg_res.get_json()["detail"]
    assert "must be a 3D volume" in detail

    # Verify inference router was NOT invoked
    mock_router.route.assert_not_called()


def test_segment_geometry_validation_corrupted_file_rejected(client, mock_router):
    """Test: Reject segmentation if a file becomes unreadable or corrupt."""
    mock_router.route.reset_mock()
    std_bytes = create_synthetic_nifti_bytes()

    files = {
        "files": [
            (io.BytesIO(std_bytes), "men_corrupt_t1n.nii.gz"),
            (io.BytesIO(std_bytes), "men_corrupt_t1c.nii.gz"),
            (io.BytesIO(std_bytes), "men_corrupt_t2w.nii.gz"),
            (io.BytesIO(std_bytes), "men_corrupt_t2f.nii.gz"),
        ]
    }
    up_res = client.post("/api/upload", data=files, content_type="multipart/form-data")
    assert up_res.status_code == 200
    req_id = up_res.get_json()["request_id"]

    # Corrupt one modality on disk
    target_file = API_RUNS_DIR / req_id / "uploads" / "men_corrupt_t2w.nii.gz"
    with open(target_file, "wb") as f:
        f.write(b"CORRUPTED_RAW_BYTES_NOT_VALID_NIFTI")

    seg_res = client.post("/api/segment", json={"request_id": req_id, "tumor_type": "meningioma"})
    assert seg_res.status_code == 400
    detail = seg_res.get_json()["detail"]
    assert "corrupted or not a valid NIfTI image" in detail

    # Verify inference router was NOT invoked
    mock_router.route.assert_not_called()


def test_segment_geometry_validation_aligned_modalities_invokes_router(client, mock_router):
    """Test: Valid aligned modalities for Glioma and Meningioma pass geometry validation and invoke router."""
    mock_router.route.reset_mock()
    std_bytes = create_synthetic_nifti_bytes()

    # 1. Glioma aligned test
    gli_files = {
        "files": [
            (io.BytesIO(std_bytes), "valid_aligned_t1.nii.gz"),
            (io.BytesIO(std_bytes), "valid_aligned_t1c.nii.gz"),
            (io.BytesIO(std_bytes), "valid_aligned_t2.nii.gz"),
            (io.BytesIO(std_bytes), "valid_aligned_flair.nii.gz"),
        ]
    }
    up_res_gli = client.post("/api/upload", data=gli_files, content_type="multipart/form-data")
    assert up_res_gli.status_code == 200
    req_gli = up_res_gli.get_json()["request_id"]

    seg_res_gli = client.post("/api/segment", json={"request_id": req_gli, "tumor_type": "glioma"})
    assert seg_res_gli.status_code == 200
    assert seg_res_gli.get_json()["request"]["tumor_type"] == "glioma"
    assert mock_router.route.call_count == 1

    # 2. Meningioma aligned test
    men_files = {
        "files": [
            (io.BytesIO(std_bytes), "valid_aligned_t1n.nii.gz"),
            (io.BytesIO(std_bytes), "valid_aligned_t1c.nii.gz"),
            (io.BytesIO(std_bytes), "valid_aligned_t2w.nii.gz"),
            (io.BytesIO(std_bytes), "valid_aligned_t2f.nii.gz"),
        ]
    }
    up_res_men = client.post("/api/upload", data=men_files, content_type="multipart/form-data")
    assert up_res_men.status_code == 200
    req_men = up_res_men.get_json()["request_id"]

    seg_res_men = client.post("/api/segment", json={"request_id": req_men, "tumor_type": "meningioma"})
    assert seg_res_men.status_code == 200
    assert seg_res_men.get_json()["request"]["tumor_type"] == "meningioma"
    assert mock_router.route.call_count == 2
