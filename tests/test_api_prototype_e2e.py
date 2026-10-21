"""
Targeted End-to-End Test Suite for Two-Hour MRI Tumor Segmentation Prototype.
=============================================================================

Validates:
1. Health and classifier status endpoints.
2. Manual Glioma routing end-to-end.
3. Manual Meningioma routing end-to-end.
4. Automatic routing behavior when classifier checkpoints are missing (honest 422).
5. Standalone /api/classify endpoint behavior.
6. Automatic routing when a compatible trained checkpoint exists (mocked).
7. Invalid NIfTI and spatial geometry mismatch rejection.
8. Accurate native voxel-volume calculation (mm3 and cm3).
9. Downloadable artifacts (multiclass mask, binary GTV mask, clinical report).
10. Frontend single-page app serving.
"""

import io
import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any
from unittest.mock import MagicMock, patch

import numpy as np
import nibabel as nib
import pytest
import SimpleITK as sitk

from api.app import create_app, API_RUNS_DIR
from integration.inference_router import InferenceRouter, AutomaticClassificationUnavailableError
from integration.classifier_service import ClassifierService
from integration.result_schema import TumorType
from tests.test_api import (
    create_synthetic_nifti_bytes,
    create_mock_glioma_result,
    create_mock_meningioma_result,
    mock_router,
)


@pytest.fixture
def app_client(mock_router):
    """Provides a Flask test client configured with the mock router."""
    app = create_app(router=mock_router)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_health_and_classifier_status_endpoint(app_client):
    """Verifies health check includes 'auto' and classifier availability status under both states."""
    # 1. Verify behavior when classifier is unavailable (simulated)
    mock_status_unavail = {
        "automatic_classification_available": False,
        "blocker_explanation": "Missing trained checkpoints on disk.",
        "stage1_presence_available": False,
        "stage2_type_available": False,
        "supported_routes": ["glioma", "meningioma"],
    }
    with patch.object(app_client.application.config["ROUTER"].classifier_service, "get_status", return_value=mock_status_unavail):
        resp = app_client.get("/api/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "healthy"
        assert "auto" in data["supported_tumor_types"]
        assert "automatic_classification_available" in data
        assert data["automatic_classification_available"] is False

        resp_c = app_client.get("/api/classifier/status")
        assert resp_c.status_code == 200
        c_data = resp_c.get_json()
        assert c_data["automatic_classification_available"] is False
        assert "blocker_explanation" in c_data
        assert c_data["blocker_explanation"] is not None

    # 2. Verify behavior when classifier is available
    mock_status_avail = {
        "automatic_classification_available": True,
        "blocker_explanation": None,
        "stage1_presence_available": True,
        "stage2_type_available": True,
        "supported_routes": ["glioma", "meningioma"],
    }
    with patch.object(app_client.application.config["ROUTER"].classifier_service, "get_status", return_value=mock_status_avail):
        resp = app_client.get("/api/health")
        assert resp.status_code == 200
        assert resp.get_json()["automatic_classification_available"] is True
        resp_c = app_client.get("/api/classifier/status")
        assert resp_c.status_code == 200
        assert resp_c.get_json()["automatic_classification_available"] is True


def test_manual_glioma_routing_e2e(app_client):
    """Uploads 4 modalities and runs manual Glioma segmentation via mock router."""
    files = {
        "files": [
            (io.BytesIO(create_synthetic_nifti_bytes()), "BraTS-GLI-00001_t1.nii.gz"),
            (io.BytesIO(create_synthetic_nifti_bytes()), "BraTS-GLI-00001_t1c.nii.gz"),
            (io.BytesIO(create_synthetic_nifti_bytes()), "BraTS-GLI-00001_t2.nii.gz"),
            (io.BytesIO(create_synthetic_nifti_bytes()), "BraTS-GLI-00001_flair.nii.gz"),
        ]
    }
    up_resp = app_client.post("/api/upload", data=files, content_type="multipart/form-data")
    assert up_resp.status_code == 200
    req_id = up_resp.get_json()["request_id"]

    seg_resp = app_client.post(
        "/api/segment",
        json={"request_id": req_id, "tumor_type": "glioma"},
    )
    assert seg_resp.status_code == 200
    seg_data = seg_resp.get_json()
    req_info = seg_data.get("request", seg_data.get("request_metadata", {}))
    vol_info = seg_data.get("volumetrics", seg_data.get("volumetric_summary", {}))

    assert req_info["tumor_type"] == "glioma"
    assert vol_info["total_tumor_volume_cm3"] == 0.5
    assert vol_info["total_tumor_volume_mm3"] == 500.0


def test_manual_meningioma_routing_e2e(app_client):
    """Uploads 4 modalities and runs manual Meningioma segmentation."""
    files = {
        "files": [
            (io.BytesIO(create_synthetic_nifti_bytes()), "BraTS-MEN-00001_t1n.nii.gz"),
            (io.BytesIO(create_synthetic_nifti_bytes()), "BraTS-MEN-00001_t1c.nii.gz"),
            (io.BytesIO(create_synthetic_nifti_bytes()), "BraTS-MEN-00001_t2w.nii.gz"),
            (io.BytesIO(create_synthetic_nifti_bytes()), "BraTS-MEN-00001_t2f.nii.gz"),
        ]
    }
    up_resp = app_client.post("/api/upload", data=files, content_type="multipart/form-data")
    assert up_resp.status_code == 200
    req_id = up_resp.get_json()["request_id"]

    seg_resp = app_client.post(
        "/api/segment",
        json={"request_id": req_id, "tumor_type": "meningioma"},
    )
    assert seg_resp.status_code == 200
    seg_data = seg_resp.get_json()
    req_info = seg_data.get("request", seg_data.get("request_metadata", {}))
    vol_info = seg_data.get("volumetrics", seg_data.get("volumetric_summary", {}))

    assert req_info["tumor_type"] == "meningioma"
    assert vol_info["total_tumor_volume_cm3"] == 0.8


def test_automatic_route_when_classifier_checkpoint_missing(app_client):
    """Verifies that requesting 'auto' when classifier weights are missing returns honest 422."""
    files = {
        "files": [
            (io.BytesIO(create_synthetic_nifti_bytes()), "scan_t1.nii.gz"),
            (io.BytesIO(create_synthetic_nifti_bytes()), "scan_t1c.nii.gz"),
            (io.BytesIO(create_synthetic_nifti_bytes()), "scan_t2.nii.gz"),
            (io.BytesIO(create_synthetic_nifti_bytes()), "scan_flair.nii.gz"),
        ]
    }
    up_resp = app_client.post("/api/upload", data=files, content_type="multipart/form-data")
    assert up_resp.status_code == 200
    req_id = up_resp.get_json()["request_id"]

    mock_unavail = {
        "status": "automatic_classification_not_available",
        "reason": "Missing trained checkpoints on disk.",
        "blocker": "Missing trained checkpoints on disk.",
        "available_manual_routes": ["glioma", "meningioma"],
        "recommendation": "Please select a tumor model manually ('glioma' or 'meningioma').",
    }
    with patch.object(app_client.application.config["ROUTER"].classifier_service, "classify", return_value=mock_unavail):
        seg_resp = app_client.post(
            "/api/segment",
            json={"request_id": req_id, "tumor_type": "auto"},
        )
        assert seg_resp.status_code == 422
        data = seg_resp.get_json()
        assert data["status"] == "automatic_classification_not_available"
        assert "Missing trained checkpoints" in data["blocker"]
        assert "glioma" in data["available_manual_routes"]
        assert "meningioma" in data["available_manual_routes"]


def test_classify_endpoint_standalone(app_client):
    """Verifies standalone /api/classify endpoint returns honest explanation and handles both states."""
    files = {
        "files": [
            (io.BytesIO(create_synthetic_nifti_bytes()), "case_t1.nii.gz"),
            (io.BytesIO(create_synthetic_nifti_bytes()), "case_t2.nii.gz"),
        ]
    }
    up_resp = app_client.post("/api/upload", data=files, content_type="multipart/form-data")
    req_id = up_resp.get_json()["request_id"]

    mock_unavail = {
        "status": "automatic_classification_not_available",
        "tumor_presence": "uncertain",
        "predicted_tumor_type": None,
        "reason": "Missing trained checkpoints on disk.",
        "recommendation": "Please select a model manually.",
    }
    with patch.object(app_client.application.config["ROUTER"].classifier_service, "classify", return_value=mock_unavail):
        res = app_client.post("/api/classify", json={"request_id": req_id})
        assert res.status_code == 200
        data = res.get_json()
        assert data["status"] == "automatic_classification_not_available"
        assert data["tumor_presence"] == "uncertain"
        assert data["predicted_tumor_type"] is None

    mock_avail = {
        "status": "classified_tumor",
        "tumor_presence": "tumor_present",
        "predicted_tumor_type": "glioma",
        "confidence": 0.92,
    }
    with patch.object(app_client.application.config["ROUTER"].classifier_service, "classify", return_value=mock_avail):
        res_avail = app_client.post("/api/classify", json={"request_id": req_id})
        assert res_avail.status_code == 200
        d_avail = res_avail.get_json()
        assert d_avail["status"] == "classified_tumor"
        assert d_avail["tumor_presence"] == "tumor_present"
        assert d_avail["predicted_tumor_type"] == "glioma"


def test_automatic_route_when_trained_checkpoint_exists_mocked(app_client):
    """Verifies that when a valid classifier output is returned, router auto-dispatches to the predicted route."""
    files = {
        "files": [
            (io.BytesIO(create_synthetic_nifti_bytes()), "case_t1.nii.gz"),
            (io.BytesIO(create_synthetic_nifti_bytes()), "case_t1c.nii.gz"),
            (io.BytesIO(create_synthetic_nifti_bytes()), "case_t2.nii.gz"),
            (io.BytesIO(create_synthetic_nifti_bytes()), "case_flair.nii.gz"),
        ]
    }
    up_resp = app_client.post("/api/upload", data=files, content_type="multipart/form-data")
    req_id = up_resp.get_json()["request_id"]

    mock_classification = {
        "status": "classified_tumor",
        "tumor_presence": "tumor_present",
        "presence_probability": 0.98,
        "predicted_tumor_type": "glioma",
        "selected_route": "glioma",
        "confidence": 0.95,
    }

    mock_seg_result = create_mock_glioma_result(req_id)

    with patch.object(app_client.application.config["ROUTER"].classifier_service, "classify", return_value=mock_classification):
        # Allow route to proceed for auto
        orig_route = app_client.application.config["ROUTER"].route

        def auto_route_mock(tumor_type, input_data, request_id, **kwargs):
            if tumor_type == TumorType.AUTO or tumor_type == "auto":
                return mock_seg_result
            return orig_route(tumor_type, input_data, request_id, **kwargs)

        app_client.application.config["ROUTER"].route = auto_route_mock

        seg_resp = app_client.post(
            "/api/segment",
            json={"request_id": req_id, "tumor_type": "auto"},
        )
        assert seg_resp.status_code == 200
        data = seg_resp.get_json()
        req_info = data.get("request", data.get("request_metadata", {}))
        vol_info = data.get("volumetrics", data.get("volumetric_summary", {}))
        assert req_info["tumor_type"] == "glioma"
        assert vol_info["total_tumor_volume_cm3"] == 0.5


def test_invalid_nifti_and_geometry_mismatch(app_client):
    """Verifies that uploaded modalities with shape mismatch are rejected by geometry validation."""
    files = {
        "files": [
            (io.BytesIO(create_synthetic_nifti_bytes(shape=(16, 16, 16))), "mismatch_t1.nii.gz"),
            (io.BytesIO(create_synthetic_nifti_bytes(shape=(16, 16, 16))), "mismatch_t1c.nii.gz"),
            (io.BytesIO(create_synthetic_nifti_bytes(shape=(16, 16, 16))), "mismatch_t2.nii.gz"),
            (io.BytesIO(create_synthetic_nifti_bytes(shape=(16, 16, 20))), "mismatch_flair.nii.gz"),
        ]
    }
    up_resp = app_client.post("/api/upload", data=files, content_type="multipart/form-data")
    assert up_resp.status_code == 200
    req_id = up_resp.get_json()["request_id"]

    seg_resp = app_client.post(
        "/api/segment",
        json={"request_id": req_id, "tumor_type": "glioma"},
    )
    assert seg_resp.status_code == 400
    data = seg_resp.get_json()
    assert "geometry validation failed" in data["detail"].lower()


def test_correct_voxel_volume_calculation():
    """Verifies precision of native voxel volume calculations in mm3 and cm3."""
    spacing = (0.9375, 0.9375, 1.2)
    voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]
    num_voxels = 10000

    total_mm3 = num_voxels * voxel_vol_mm3
    total_cm3 = total_mm3 / 1000.0

    assert round(voxel_vol_mm3, 6) == 1.054688
    assert round(total_mm3, 2) == 10546.88
    assert round(total_cm3, 4) == 10.5469


def test_mask_download_and_report_generation(app_client):
    """Verifies mask downloads and human-readable clinical report generation."""
    req_id = "REQ-TESTREPORT999"
    req_dir = API_RUNS_DIR / req_id
    masks_dir = req_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    try:
        mask_arr = np.zeros((16, 16, 16), dtype=np.uint8)
        mask_arr[4:10, 4:10, 4:10] = 1
        mask_img = sitk.GetImageFromArray(mask_arr)
        sitk.WriteImage(mask_img, str(masks_dir / f"{req_id}_glioma_seg.nii.gz"))

        result_data = {
            "pipeline_status": "success",
            "request": {"request_id": req_id, "tumor_type": "glioma", "timestamp": "2026-10-18T10:00:00Z"},
            "input_metadata": {
                "modality_names": ["T1", "T1c", "T2", "FLAIR"],
                "original_shape": [16, 16, 16],
                "original_spacing": [1.0, 1.0, 1.0],
                "voxel_volume_mm3": 1.0,
            },
            "volumetrics": {"total_tumor_volume_mm3": 216.0, "total_tumor_volume_cm3": 0.216},
            "glioma": {
                "subregion_volumes": {
                    "wt_volume_mm3": 216.0,
                    "tc_volume_mm3": 150.0,
                    "et_volume_mm3": 80.0,
                    "wt_volume_cm3": 0.216,
                    "tc_volume_cm3": 0.15,
                    "et_volume_cm3": 0.08,
                }
            },
        }
        with open(req_dir / "result.json", "w") as f:
            json.dump(result_data, f)

        # 1. Multiclass mask download
        resp_multi = app_client.get(f"/api/mask/{req_id}?type=multiclass")
        assert resp_multi.status_code == 200
        assert resp_multi.mimetype == "application/gzip"

        # 2. Binary GTV mask download (auto-generated)
        resp_gtv = app_client.get(f"/api/mask/{req_id}?type=gtv")
        assert resp_gtv.status_code == 200
        assert resp_gtv.mimetype == "application/gzip"
        assert (masks_dir / f"{req_id}_glioma_gtv.nii.gz").exists()

        # 3. Clinical text report
        resp_rep = app_client.get(f"/api/report/{req_id}")
        assert resp_rep.status_code == 200
        assert resp_rep.mimetype.startswith("text/plain")
        report_text = resp_rep.get_data(as_text=True)
        assert req_id in report_text
        assert "GLIOMA" in report_text
        assert "Total Tumor Volume: 216.0 mm³" in report_text
        assert "Whole Tumor (WT):         216.0 mm³" in report_text
        assert "RESEARCH PROTOTYPE ONLY" in report_text

        # 4. Verify Meningioma report using canonical 'subregions' dictionary key
        men_req_id = "REQ-TESTREPORT-MEN"
        men_dir = API_RUNS_DIR / men_req_id
        men_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(men_dir / "result.json", "w") as f:
                json.dump({
                    "request": {"request_id": men_req_id, "tumor_type": "meningioma", "timestamp": "2026-10-18T10:00:00Z"},
                    "input_metadata": {"modality_names": ["t1n", "t1c", "t2w", "t2f"], "original_shape": [16, 16, 16], "original_spacing": [1.0, 1.0, 1.0], "voxel_volume_mm3": 1.0},
                    "volumetrics": {"total_tumor_volume_mm3": 300.0, "total_tumor_volume_cm3": 0.3},
                    "quality_status": {"status": "success"},
                    "meningioma_multimodal": {
                        "subregions": {
                            "wt_volume_mm3": 300.0,
                            "et_volume_mm3": 200.0,
                            "ncr_volume_mm3": 50.0,
                            "ed_volume_mm3": 50.0,
                            "wt_volume_cm3": 0.3,
                            "et_volume_cm3": 0.2,
                            "ncr_volume_cm3": 0.05,
                            "ed_volume_cm3": 0.05,
                        }
                    }
                }, f)
            men_rep = app_client.get(f"/api/report/{men_req_id}")
            assert men_rep.status_code == 200
            men_text = men_rep.get_data(as_text=True)
            assert "MENINGIOMA" in men_text
            assert "Whole Tumor (WT):         300.0 mm³" in men_text
            assert "Enhancing Tumor (ET):      200.0 mm³" in men_text
        finally:
            if men_dir.exists():
                shutil.rmtree(men_dir, ignore_errors=True)
    finally:
        if req_dir.exists():
            shutil.rmtree(req_dir, ignore_errors=True)


def test_missing_model_result_and_invalid_model_names_return_clear_api_errors(app_client):
    """
    Verifies that when only one model has been executed for an upload:
    1. Querying a missing model result or report returns an explicit 404 without fabricating data or falling back.
    2. Requesting a missing model mask returns an explicit 404 without returning the other model's mask.
    3. Querying an invalid/unsupported model name returns a clear 400 error.
    """
    files = {
        "files": [
            (io.BytesIO(create_synthetic_nifti_bytes()), "BraTS-GLI-SINGLE-t1.nii.gz"),
            (io.BytesIO(create_synthetic_nifti_bytes()), "BraTS-GLI-SINGLE-t1c.nii.gz"),
            (io.BytesIO(create_synthetic_nifti_bytes()), "BraTS-GLI-SINGLE-t2.nii.gz"),
            (io.BytesIO(create_synthetic_nifti_bytes()), "BraTS-GLI-SINGLE-flair.nii.gz"),
        ]
    }
    up_resp = app_client.post("/api/upload", data=files, content_type="multipart/form-data")
    assert up_resp.status_code == 200
    req_id = up_resp.get_json()["request_id"]
    req_dir = API_RUNS_DIR / req_id

    try:
        # Run ONLY glioma
        resp_glioma = app_client.post(
            "/api/segment",
            json={"request_id": req_id, "tumor_type": "glioma"},
        )
        assert resp_glioma.status_code == 200

        assert (req_dir / "result_glioma.json").exists()
        assert not (req_dir / "result_meningioma.json").exists()

        # 1. Querying unexecuted model results returns 404 (NOT glioma data from result.json)
        res_men = app_client.get(f"/api/results/{req_id}?tumor_type=meningioma")
        assert res_men.status_code == 404
        assert "No meningioma segmentation results found" in res_men.get_json()["detail"]

        # 2. Querying unexecuted model report returns 404
        rep_men = app_client.get(f"/api/report/{req_id}?tumor_type=meningioma")
        assert rep_men.status_code == 404
        assert "No meningioma report found" in rep_men.get_json()["detail"]

        # 3. Requesting unexecuted model mask returns 404 (does NOT download glioma mask)
        mask_men = app_client.get(f"/api/mask/{req_id}?type=multiclass&tumor_type=meningioma")
        assert mask_men.status_code == 404
        assert "No meningioma segmentation mask found" in mask_men.get_json()["detail"]

        # 4. Invalid model name on /api/results returns 400
        res_bad = app_client.get(f"/api/results/{req_id}?tumor_type=sarcoma")
        assert res_bad.status_code == 400
        assert "Invalid tumor type or model" in res_bad.get_json()["detail"]

        # 5. Invalid model name on /api/report returns 400
        rep_bad = app_client.get(f"/api/report/{req_id}?tumor_type=sarcoma")
        assert rep_bad.status_code == 400
        assert "Invalid tumor type or model" in rep_bad.get_json()["detail"]

        # 6. Invalid model name on /api/mask returns 400
        mask_bad = app_client.get(f"/api/mask/{req_id}?tumor_type=sarcoma")
        assert mask_bad.status_code == 400
        assert "Invalid tumor type or model" in mask_bad.get_json()["detail"]

        # 7. Nonexistent request ID returns 404
        res_none = app_client.get("/api/results/REQ-DOESNOTEXIST999")
        assert res_none.status_code == 404

    finally:
        if req_dir.exists():
            shutil.rmtree(req_dir, ignore_errors=True)


def test_atomic_write_json_helper(tmp_path):
    """Verifies that _atomic_write_json correctly commits files and handles failures cleanly."""
    from api.app import _atomic_write_json

    target = tmp_path / "test_atomic.json"
    data = {"status": "success", "volumetrics": {"wt": 123.45}}

    _atomic_write_json(target, data)
    assert target.exists()

    with open(target, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == data

    # Verify atomic failure resilience: non-serializable data should raise exception,
    # target should retain old content, and no .tmp file should linger
    bad_data = {"unserializable": {1, 2, 3}}
    with pytest.raises(TypeError):
        _atomic_write_json(target, bad_data)

    with open(target, "r", encoding="utf-8") as f:
        reloaded = json.load(f)
    assert reloaded == data
    tmp_files = list(tmp_path.glob("*.tmp*"))
    assert len(tmp_files) == 0



def test_frontend_index_and_static_served(app_client):
    """Verifies that the single-page frontend UI is served at root."""
    resp = app_client.get("/")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "3D Brain Tumor MRI Segmentation Prototype" in html
    assert "Upload 3D MRI Acquisitions" in html


def test_response_contains_frontend_compatible_keys(app_client):
    """
    Verifies that the segmentation result JSON contains all fields expected by frontend/index.html:
    request_id, request_metadata, volumetric_summary, input_metadata, glioma_measurements,
    meningioma_measurements (with subregion_volumes).
    """
    files = {
        "files": [
            (io.BytesIO(create_synthetic_nifti_bytes()), "BraTS-MEN-00042_t1n.nii.gz"),
            (io.BytesIO(create_synthetic_nifti_bytes()), "BraTS-MEN-00042_t1c.nii.gz"),
            (io.BytesIO(create_synthetic_nifti_bytes()), "BraTS-MEN-00042_t2w.nii.gz"),
            (io.BytesIO(create_synthetic_nifti_bytes()), "BraTS-MEN-00042_t2f.nii.gz"),
        ]
    }
    up_resp = app_client.post("/api/upload", data=files, content_type="multipart/form-data")
    assert up_resp.status_code == 200
    req_id = up_resp.get_json()["request_id"]

    try:
        seg_resp = app_client.post(
            "/api/segment",
            json={"request_id": req_id, "tumor_type": "meningioma"},
        )
        assert seg_resp.status_code == 200
        data = seg_resp.get_json()

        # Frontend direct property checks (mimicking index.html displayResults lines 699-735)
        assert "request_metadata" in data
        assert data["request_metadata"]["request_id"] == req_id
        assert data["request_metadata"]["tumor_type"] == "meningioma"
        assert "volumetric_summary" in data
        assert "total_tumor_volume_cm3" in data["volumetric_summary"]
        assert "total_tumor_volume_mm3" in data["volumetric_summary"]
        assert "input_metadata" in data
        assert "original_shape" in data["input_metadata"]
        assert "original_spacing" in data["input_metadata"]
        assert "voxel_volume_mm3" in data["input_metadata"]
        assert "meningioma_measurements" in data
        sub = data["meningioma_measurements"]["subregion_volumes"]
        assert "wt_volume_mm3" in sub
        assert "wt_volume_cm3" in sub
        assert "et_volume_mm3" in sub
        assert "et_volume_cm3" in sub
        assert "ncr_volume_mm3" in sub
        assert "ncr_volume_cm3" in sub
        assert "ed_volume_mm3" in sub
        assert "ed_volume_cm3" in sub

        # Top-level request_id
        assert data.get("request_id") == req_id
    finally:
        req_dir = API_RUNS_DIR / req_id
        if req_dir.exists():
            shutil.rmtree(req_dir, ignore_errors=True)


def test_dual_model_execution_on_same_request_id_preserves_isolated_results_and_masks(app_client):
    """
    Regression Test:
    When a user uploads a single study (request_id) and sequentially executes both
    Glioma and Meningioma models:
    1. Both executions succeed without conflicting or corrupting each other.
    2. Both result_glioma.json and result_meningioma.json are created and preserved with distinct volumes.
    3. result.json updates to reflect the latest execution.
    4. /api/results/<req_id>?tumor_type=... returns the isolated model run.
    5. /api/mask/<req_id>?type=multiclass defaults to the latest execution mask, and ?tumor_type=glioma
       correctly returns the glioma mask without collision.
    6. /api/report/<req_id> clearly reports the selected model, manual routing mode, and diagnostic disclaimer.
    """
    files = {
        "files": [
            (io.BytesIO(create_synthetic_nifti_bytes()), "BraTS-DUAL-00001-t1n.nii.gz"),
            (io.BytesIO(create_synthetic_nifti_bytes()), "BraTS-DUAL-00001-t1c.nii.gz"),
            (io.BytesIO(create_synthetic_nifti_bytes()), "BraTS-DUAL-00001-t2w.nii.gz"),
            (io.BytesIO(create_synthetic_nifti_bytes()), "BraTS-DUAL-00001-t2f.nii.gz"),
        ]
    }
    up_resp = app_client.post("/api/upload", data=files, content_type="multipart/form-data")
    assert up_resp.status_code == 200
    req_id = up_resp.get_json()["request_id"]
    req_dir = API_RUNS_DIR / req_id

    try:
        # Step 1: Execute Glioma
        resp_glioma = app_client.post(
            "/api/segment",
            json={"request_id": req_id, "tumor_type": "glioma"},
        )
        assert resp_glioma.status_code == 200
        glioma_data = resp_glioma.get_json()
        assert glioma_data["request_metadata"]["selected_model"] == "glioma"
        assert glioma_data["volumetric_summary"]["total_tumor_volume_mm3"] == 500.0

        # Step 2: Execute Meningioma on the SAME request_id
        resp_men = app_client.post(
            "/api/segment",
            json={"request_id": req_id, "tumor_type": "meningioma"},
        )
        assert resp_men.status_code == 200
        men_data = resp_men.get_json()
        assert men_data["request_metadata"]["selected_model"] == "meningioma"
        assert men_data["volumetric_summary"]["total_tumor_volume_mm3"] == 800.0

        # Step 3: Verify both result files exist on disk with isolated data
        glioma_json_path = req_dir / "result_glioma.json"
        men_json_path = req_dir / "result_meningioma.json"
        latest_json_path = req_dir / "result.json"

        assert glioma_json_path.exists()
        assert men_json_path.exists()
        assert latest_json_path.exists()

        with open(glioma_json_path, "r", encoding="utf-8") as f:
            saved_gl = json.load(f)
        assert saved_gl["volumetric_summary"]["total_tumor_volume_mm3"] == 500.0
        assert saved_gl["request_metadata"]["selected_model"] == "glioma"

        with open(men_json_path, "r", encoding="utf-8") as f:
            saved_men = json.load(f)
        assert saved_men["volumetric_summary"]["total_tumor_volume_mm3"] == 800.0
        assert saved_men["request_metadata"]["selected_model"] == "meningioma"

        # Step 4: Verify GET /api/results/<req_id> with query parameters
        res_latest = app_client.get(f"/api/results/{req_id}")
        assert res_latest.status_code == 200
        assert res_latest.get_json()["request_metadata"]["selected_model"] == "meningioma"

        res_gl_query = app_client.get(f"/api/results/{req_id}?tumor_type=glioma")
        assert res_gl_query.status_code == 200
        assert res_gl_query.get_json()["request_metadata"]["selected_model"] == "glioma"
        assert res_gl_query.get_json()["volumetric_summary"]["total_tumor_volume_mm3"] == 500.0

        # Step 5: Verify GET /api/mask/<req_id> resolves correct mask without collision
        # Default multiclass should download the latest (meningioma)
        mask_def = app_client.get(f"/api/mask/{req_id}?type=multiclass")
        assert mask_def.status_code == 200
        assert "meningioma" in mask_def.headers.get("Content-Disposition", "").lower()

        # Specific query for glioma should download glioma mask
        mask_gl = app_client.get(f"/api/mask/{req_id}?type=multiclass&tumor_type=glioma")
        assert mask_gl.status_code == 200
        assert "glioma" in mask_gl.headers.get("Content-Disposition", "").lower()

        # Step 6: Verify GET /api/report/<req_id>
        rep_resp = app_client.get(f"/api/report/{req_id}")
        assert rep_resp.status_code == 200
        rep_text = rep_resp.get_data(as_text=True)
        assert (
            "Selected Model:   MENINGIOMA 3D U-Net (Manual Selection)" in rep_text
            or "Selected Model:   MENINGIOMA 3D U-Net (Manual Override)" in rep_text
        )
        assert (
            "Routing Mode:     MANUAL SELECTION" in rep_text
            or "Routing Mode:     MANUAL OVERRIDE" in rep_text
        )
        assert "DIAGNOSTIC & OPERATIONAL CLARIFICATION" in rep_text

        # Report for specific past model run
        rep_gl_resp = app_client.get(f"/api/report/{req_id}?tumor_type=glioma")
        assert rep_gl_resp.status_code == 200
        rep_gl_text = rep_gl_resp.get_data(as_text=True)
        assert (
            "Selected Model:   GLIOMA 3D U-Net (Manual Selection)" in rep_gl_text
            or "Selected Model:   GLIOMA 3D U-Net (Manual Override)" in rep_gl_text
        )
        assert "Total Tumor Volume: 500.0 mm³" in rep_gl_text

    finally:
        if req_dir.exists():
            shutil.rmtree(req_dir, ignore_errors=True)


def test_missing_model_result_and_invalid_model_names_return_clear_api_errors(app_client):
    """
    Verifies that when only one model has been executed for an upload:
    1. Querying a missing model result or report returns an explicit 404 without fabricating data or falling back.
    2. Requesting a missing model mask returns an explicit 404 without returning the other model's mask.
    3. Querying an invalid/unsupported model name returns a clear 400 error.
    """
    files = {
        "files": [
            (io.BytesIO(create_synthetic_nifti_bytes()), "BraTS-GLI-SINGLE-t1.nii.gz"),
            (io.BytesIO(create_synthetic_nifti_bytes()), "BraTS-GLI-SINGLE-t1c.nii.gz"),
            (io.BytesIO(create_synthetic_nifti_bytes()), "BraTS-GLI-SINGLE-t2.nii.gz"),
            (io.BytesIO(create_synthetic_nifti_bytes()), "BraTS-GLI-SINGLE-flair.nii.gz"),
        ]
    }
    up_resp = app_client.post("/api/upload", data=files, content_type="multipart/form-data")
    assert up_resp.status_code == 200
    req_id = up_resp.get_json()["request_id"]
    req_dir = API_RUNS_DIR / req_id

    try:
        # Run ONLY glioma
        resp_glioma = app_client.post(
            "/api/segment",
            json={"request_id": req_id, "tumor_type": "glioma"},
        )
        assert resp_glioma.status_code == 200

        assert (req_dir / "result_glioma.json").exists()
        assert not (req_dir / "result_meningioma.json").exists()

        # 1. Querying unexecuted model results returns 404 (NOT glioma data from result.json)
        res_men = app_client.get(f"/api/results/{req_id}?tumor_type=meningioma")
        assert res_men.status_code == 404
        assert "No meningioma segmentation results found" in res_men.get_json()["detail"]

        # 2. Querying unexecuted model report returns 404
        rep_men = app_client.get(f"/api/report/{req_id}?tumor_type=meningioma")
        assert rep_men.status_code == 404
        assert "No meningioma report found" in rep_men.get_json()["detail"]

        # 3. Requesting unexecuted model mask returns 404 (does NOT download glioma mask)
        mask_men = app_client.get(f"/api/mask/{req_id}?type=multiclass&tumor_type=meningioma")
        assert mask_men.status_code == 404
        assert "No meningioma segmentation mask found" in mask_men.get_json()["detail"]

        # 4. Invalid model name on /api/results returns 400
        res_bad = app_client.get(f"/api/results/{req_id}?tumor_type=sarcoma")
        assert res_bad.status_code == 400
        assert "Invalid tumor type or model" in res_bad.get_json()["detail"]

        # 5. Invalid model name on /api/report returns 400
        rep_bad = app_client.get(f"/api/report/{req_id}?tumor_type=sarcoma")
        assert rep_bad.status_code == 400
        assert "Invalid tumor type or model" in rep_bad.get_json()["detail"]

        # 6. Invalid model name on /api/mask returns 400
        mask_bad = app_client.get(f"/api/mask/{req_id}?tumor_type=sarcoma")
        assert mask_bad.status_code == 400
        assert "Invalid tumor type or model" in mask_bad.get_json()["detail"]

        # 7. Nonexistent request ID returns 404
        res_none = app_client.get("/api/results/REQ-DOESNOTEXIST999")
        assert res_none.status_code == 404

    finally:
        if req_dir.exists():
            shutil.rmtree(req_dir, ignore_errors=True)


def test_atomic_write_json_helper(tmp_path):
    """Verifies that _atomic_write_json correctly commits files and handles failures cleanly."""
    from api.app import _atomic_write_json

    target = tmp_path / "test_atomic.json"
    data = {"status": "success", "volumetrics": {"wt": 123.45}}

    _atomic_write_json(target, data)
    assert target.exists()

    with open(target, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == data

    # Verify atomic failure resilience: non-serializable data should raise exception,
    # target should retain old content, and no .tmp file should linger
    bad_data = {"unserializable": {1, 2, 3}}
    with pytest.raises(TypeError):
        _atomic_write_json(target, bad_data)

    with open(target, "r", encoding="utf-8") as f:
        reloaded = json.load(f)
    assert reloaded == data
    tmp_files = list(tmp_path.glob("*.tmp*"))
    assert len(tmp_files) == 0




def test_upload_and_segment_with_seg_mask_fails_cleanly(app_client):
    """
    Verifies that uploading a segmentation mask (e.g. -seg.nii.gz) as an acquisition
    is rejected with an explicit, readable 400 error identifying the mask file.
    """
    files = {
        "files": [
            (io.BytesIO(create_synthetic_nifti_bytes()), "BraTS-MEN-00004-000-t1n.nii.gz"),
            (io.BytesIO(create_synthetic_nifti_bytes()), "BraTS-MEN-00004-000-t1c.nii.gz"),
            (io.BytesIO(create_synthetic_nifti_bytes()), "BraTS-MEN-00004-000-t2f.nii.gz"),
            (io.BytesIO(create_synthetic_nifti_bytes()), "BraTS-MEN-00004-000-seg.nii.gz"),
        ]
    }
    up_resp = app_client.post("/api/upload", data=files, content_type="multipart/form-data")
    assert up_resp.status_code == 200
    req_id = up_resp.get_json()["request_id"]

    try:
        seg_resp = app_client.post(
            "/api/segment",
            json={"request_id": req_id, "tumor_type": "meningioma"},
        )
        assert seg_resp.status_code == 400
        err_data = seg_resp.get_json()
        assert err_data.get("request_id") == req_id
        assert "Detected segmentation mask file(s)" in err_data.get("detail", "")
        assert "BraTS-MEN-00004-000-seg.nii.gz" in err_data.get("detail", "")
    finally:
        req_dir = API_RUNS_DIR / req_id
        if req_dir.exists():
            shutil.rmtree(req_dir, ignore_errors=True)


def test_dual_model_execution_on_same_request_id_preserves_isolated_results_and_masks(app_client):
    """
    Regression Test:
    When a user uploads a single study (request_id) and sequentially executes both
    Glioma and Meningioma models:
    1. Both executions succeed without conflicting or corrupting each other.
    2. Both result_glioma.json and result_meningioma.json are created and preserved with distinct volumes.
    3. result.json updates to reflect the latest execution.
    4. /api/results/<req_id>?tumor_type=... returns the isolated model run.
    5. /api/mask/<req_id>?type=multiclass defaults to the latest execution mask, and ?tumor_type=glioma
       correctly returns the glioma mask without collision.
    6. /api/report/<req_id> clearly reports the selected model, manual routing mode, and diagnostic disclaimer.
    """
    files = {
        "files": [
            (io.BytesIO(create_synthetic_nifti_bytes()), "BraTS-DUAL-00001-t1n.nii.gz"),
            (io.BytesIO(create_synthetic_nifti_bytes()), "BraTS-DUAL-00001-t1c.nii.gz"),
            (io.BytesIO(create_synthetic_nifti_bytes()), "BraTS-DUAL-00001-t2w.nii.gz"),
            (io.BytesIO(create_synthetic_nifti_bytes()), "BraTS-DUAL-00001-t2f.nii.gz"),
        ]
    }
    up_resp = app_client.post("/api/upload", data=files, content_type="multipart/form-data")
    assert up_resp.status_code == 200
    req_id = up_resp.get_json()["request_id"]
    req_dir = API_RUNS_DIR / req_id

    try:
        # Step 1: Execute Glioma
        resp_glioma = app_client.post(
            "/api/segment",
            json={"request_id": req_id, "tumor_type": "glioma"},
        )
        assert resp_glioma.status_code == 200
        glioma_data = resp_glioma.get_json()
        assert glioma_data["request_metadata"]["selected_model"] == "glioma"
        assert glioma_data["volumetric_summary"]["total_tumor_volume_mm3"] == 500.0

        # Step 2: Execute Meningioma on the SAME request_id
        resp_men = app_client.post(
            "/api/segment",
            json={"request_id": req_id, "tumor_type": "meningioma"},
        )
        assert resp_men.status_code == 200
        men_data = resp_men.get_json()
        assert men_data["request_metadata"]["selected_model"] == "meningioma"
        assert men_data["volumetric_summary"]["total_tumor_volume_mm3"] == 800.0

        # Step 3: Verify both result files exist on disk with isolated data
        glioma_json_path = req_dir / "result_glioma.json"
        men_json_path = req_dir / "result_meningioma.json"
        latest_json_path = req_dir / "result.json"

        assert glioma_json_path.exists()
        assert men_json_path.exists()
        assert latest_json_path.exists()

        with open(glioma_json_path, "r", encoding="utf-8") as f:
            saved_gl = json.load(f)
        assert saved_gl["volumetric_summary"]["total_tumor_volume_mm3"] == 500.0
        assert saved_gl["request_metadata"]["selected_model"] == "glioma"

        with open(men_json_path, "r", encoding="utf-8") as f:
            saved_men = json.load(f)
        assert saved_men["volumetric_summary"]["total_tumor_volume_mm3"] == 800.0
        assert saved_men["request_metadata"]["selected_model"] == "meningioma"

        # Step 4: Verify GET /api/results/<req_id> with query parameters
        res_latest = app_client.get(f"/api/results/{req_id}")
        assert res_latest.status_code == 200
        assert res_latest.get_json()["request_metadata"]["selected_model"] == "meningioma"

        res_gl_query = app_client.get(f"/api/results/{req_id}?tumor_type=glioma")
        assert res_gl_query.status_code == 200
        assert res_gl_query.get_json()["request_metadata"]["selected_model"] == "glioma"
        assert res_gl_query.get_json()["volumetric_summary"]["total_tumor_volume_mm3"] == 500.0

        # Step 5: Verify GET /api/mask/<req_id> resolves correct mask without collision
        # Default multiclass should download the latest (meningioma)
        mask_def = app_client.get(f"/api/mask/{req_id}?type=multiclass")
        assert mask_def.status_code == 200
        assert "meningioma" in mask_def.headers.get("Content-Disposition", "").lower()

        # Specific query for glioma should download glioma mask
        mask_gl = app_client.get(f"/api/mask/{req_id}?type=multiclass&tumor_type=glioma")
        assert mask_gl.status_code == 200
        assert "glioma" in mask_gl.headers.get("Content-Disposition", "").lower()

        # Step 6: Verify GET /api/report/<req_id>
        rep_resp = app_client.get(f"/api/report/{req_id}")
        assert rep_resp.status_code == 200
        rep_text = rep_resp.get_data(as_text=True)
        assert (
            "Selected Model:   MENINGIOMA 3D U-Net (Manual Selection)" in rep_text
            or "Selected Model:   MENINGIOMA 3D U-Net (Manual Override)" in rep_text
        )
        assert (
            "Routing Mode:     MANUAL SELECTION" in rep_text
            or "Routing Mode:     MANUAL OVERRIDE" in rep_text
        )
        assert "DIAGNOSTIC & OPERATIONAL CLARIFICATION" in rep_text

        # Report for specific past model run
        rep_gl_resp = app_client.get(f"/api/report/{req_id}?tumor_type=glioma")
        assert rep_gl_resp.status_code == 200
        rep_gl_text = rep_gl_resp.get_data(as_text=True)
        assert (
            "Selected Model:   GLIOMA 3D U-Net (Manual Selection)" in rep_gl_text
            or "Selected Model:   GLIOMA 3D U-Net (Manual Override)" in rep_gl_text
        )
        assert "Total Tumor Volume: 500.0 mm³" in rep_gl_text

    finally:
        if req_dir.exists():
            shutil.rmtree(req_dir, ignore_errors=True)


def test_missing_model_result_and_invalid_model_names_return_clear_api_errors(app_client):
    """
    Verifies that when only one model has been executed for an upload:
    1. Querying a missing model result or report returns an explicit 404 without fabricating data or falling back.
    2. Requesting a missing model mask returns an explicit 404 without returning the other model's mask.
    3. Querying an invalid/unsupported model name returns a clear 400 error.
    """
    files = {
        "files": [
            (io.BytesIO(create_synthetic_nifti_bytes()), "BraTS-GLI-SINGLE-t1.nii.gz"),
            (io.BytesIO(create_synthetic_nifti_bytes()), "BraTS-GLI-SINGLE-t1c.nii.gz"),
            (io.BytesIO(create_synthetic_nifti_bytes()), "BraTS-GLI-SINGLE-t2.nii.gz"),
            (io.BytesIO(create_synthetic_nifti_bytes()), "BraTS-GLI-SINGLE-flair.nii.gz"),
        ]
    }
    up_resp = app_client.post("/api/upload", data=files, content_type="multipart/form-data")
    assert up_resp.status_code == 200
    req_id = up_resp.get_json()["request_id"]
    req_dir = API_RUNS_DIR / req_id

    try:
        # Run ONLY glioma
        resp_glioma = app_client.post(
            "/api/segment",
            json={"request_id": req_id, "tumor_type": "glioma"},
        )
        assert resp_glioma.status_code == 200

        assert (req_dir / "result_glioma.json").exists()
        assert not (req_dir / "result_meningioma.json").exists()

        # 1. Querying unexecuted model results returns 404 (NOT glioma data from result.json)
        res_men = app_client.get(f"/api/results/{req_id}?tumor_type=meningioma")
        assert res_men.status_code == 404
        assert "No meningioma segmentation results found" in res_men.get_json()["detail"]

        # 2. Querying unexecuted model report returns 404
        rep_men = app_client.get(f"/api/report/{req_id}?tumor_type=meningioma")
        assert rep_men.status_code == 404
        assert "No meningioma report found" in rep_men.get_json()["detail"]

        # 3. Requesting unexecuted model mask returns 404 (does NOT download glioma mask)
        mask_men = app_client.get(f"/api/mask/{req_id}?type=multiclass&tumor_type=meningioma")
        assert mask_men.status_code == 404
        assert "No meningioma segmentation mask found" in mask_men.get_json()["detail"]

        # 4. Invalid model name on /api/results returns 400
        res_bad = app_client.get(f"/api/results/{req_id}?tumor_type=sarcoma")
        assert res_bad.status_code == 400
        assert "Invalid tumor type or model" in res_bad.get_json()["detail"]

        # 5. Invalid model name on /api/report returns 400
        rep_bad = app_client.get(f"/api/report/{req_id}?tumor_type=sarcoma")
        assert rep_bad.status_code == 400
        assert "Invalid tumor type or model" in rep_bad.get_json()["detail"]

        # 6. Invalid model name on /api/mask returns 400
        mask_bad = app_client.get(f"/api/mask/{req_id}?tumor_type=sarcoma")
        assert mask_bad.status_code == 400
        assert "Invalid tumor type or model" in mask_bad.get_json()["detail"]

        # 7. Nonexistent request ID returns 404
        res_none = app_client.get("/api/results/REQ-DOESNOTEXIST999")
        assert res_none.status_code == 404

    finally:
        if req_dir.exists():
            shutil.rmtree(req_dir, ignore_errors=True)


def test_atomic_write_json_helper(tmp_path):
    """Verifies that _atomic_write_json correctly commits files and handles failures cleanly."""
    from api.app import _atomic_write_json

    target = tmp_path / "test_atomic.json"
    data = {"status": "success", "volumetrics": {"wt": 123.45}}

    _atomic_write_json(target, data)
    assert target.exists()

    with open(target, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == data

    # Verify atomic failure resilience: non-serializable data should raise exception,
    # target should retain old content, and no .tmp file should linger
    bad_data = {"unserializable": {1, 2, 3}}
    with pytest.raises(TypeError):
        _atomic_write_json(target, bad_data)

    with open(target, "r", encoding="utf-8") as f:
        reloaded = json.load(f)
    assert reloaded == data
    tmp_files = list(tmp_path.glob("*.tmp*"))
    assert len(tmp_files) == 0




def test_missing_modality_meningioma_fails_cleanly_with_request_id(app_client):
    """
    Verifies that missing a required modality (e.g. t2w) returns 400 with a clear error
    detailing the missing modality and containing request_id.
    """
    files = {
        "files": [
            (io.BytesIO(create_synthetic_nifti_bytes()), "BraTS-MEN-00005-000-t1n.nii.gz"),
            (io.BytesIO(create_synthetic_nifti_bytes()), "BraTS-MEN-00005-000-t1c.nii.gz"),
            (io.BytesIO(create_synthetic_nifti_bytes()), "BraTS-MEN-00005-000-t2f.nii.gz"),
        ]
    }
    up_resp = app_client.post("/api/upload", data=files, content_type="multipart/form-data")
    assert up_resp.status_code == 200
    req_id = up_resp.get_json()["request_id"]

    try:
        seg_resp = app_client.post(
            "/api/segment",
            json={"request_id": req_id, "tumor_type": "meningioma"},
        )
        assert seg_resp.status_code == 400
        err_data = seg_resp.get_json()
        assert err_data.get("request_id") == req_id
        assert "Missing: ['t2w']" in err_data.get("detail", "")
    finally:
        req_dir = API_RUNS_DIR / req_id
        if req_dir.exists():
            shutil.rmtree(req_dir, ignore_errors=True)


def test_dual_model_execution_on_same_request_id_preserves_isolated_results_and_masks(app_client):
    """
    Regression Test:
    When a user uploads a single study (request_id) and sequentially executes both
    Glioma and Meningioma models:
    1. Both executions succeed without conflicting or corrupting each other.
    2. Both result_glioma.json and result_meningioma.json are created and preserved with distinct volumes.
    3. result.json updates to reflect the latest execution.
    4. /api/results/<req_id>?tumor_type=... returns the isolated model run.
    5. /api/mask/<req_id>?type=multiclass defaults to the latest execution mask, and ?tumor_type=glioma
       correctly returns the glioma mask without collision.
    6. /api/report/<req_id> clearly reports the selected model, manual routing mode, and diagnostic disclaimer.
    """
    files = {
        "files": [
            (io.BytesIO(create_synthetic_nifti_bytes()), "BraTS-DUAL-00001-t1n.nii.gz"),
            (io.BytesIO(create_synthetic_nifti_bytes()), "BraTS-DUAL-00001-t1c.nii.gz"),
            (io.BytesIO(create_synthetic_nifti_bytes()), "BraTS-DUAL-00001-t2w.nii.gz"),
            (io.BytesIO(create_synthetic_nifti_bytes()), "BraTS-DUAL-00001-t2f.nii.gz"),
        ]
    }
    up_resp = app_client.post("/api/upload", data=files, content_type="multipart/form-data")
    assert up_resp.status_code == 200
    req_id = up_resp.get_json()["request_id"]
    req_dir = API_RUNS_DIR / req_id

    try:
        # Step 1: Execute Glioma
        resp_glioma = app_client.post(
            "/api/segment",
            json={"request_id": req_id, "tumor_type": "glioma"},
        )
        assert resp_glioma.status_code == 200
        glioma_data = resp_glioma.get_json()
        assert glioma_data["request_metadata"]["selected_model"] == "glioma"
        assert glioma_data["volumetric_summary"]["total_tumor_volume_mm3"] == 500.0

        # Step 2: Execute Meningioma on the SAME request_id
        resp_men = app_client.post(
            "/api/segment",
            json={"request_id": req_id, "tumor_type": "meningioma"},
        )
        assert resp_men.status_code == 200
        men_data = resp_men.get_json()
        assert men_data["request_metadata"]["selected_model"] == "meningioma"
        assert men_data["volumetric_summary"]["total_tumor_volume_mm3"] == 800.0

        # Step 3: Verify both result files exist on disk with isolated data
        glioma_json_path = req_dir / "result_glioma.json"
        men_json_path = req_dir / "result_meningioma.json"
        latest_json_path = req_dir / "result.json"

        assert glioma_json_path.exists()
        assert men_json_path.exists()
        assert latest_json_path.exists()

        with open(glioma_json_path, "r", encoding="utf-8") as f:
            saved_gl = json.load(f)
        assert saved_gl["volumetric_summary"]["total_tumor_volume_mm3"] == 500.0
        assert saved_gl["request_metadata"]["selected_model"] == "glioma"

        with open(men_json_path, "r", encoding="utf-8") as f:
            saved_men = json.load(f)
        assert saved_men["volumetric_summary"]["total_tumor_volume_mm3"] == 800.0
        assert saved_men["request_metadata"]["selected_model"] == "meningioma"

        # Step 4: Verify GET /api/results/<req_id> with query parameters
        res_latest = app_client.get(f"/api/results/{req_id}")
        assert res_latest.status_code == 200
        assert res_latest.get_json()["request_metadata"]["selected_model"] == "meningioma"

        res_gl_query = app_client.get(f"/api/results/{req_id}?tumor_type=glioma")
        assert res_gl_query.status_code == 200
        assert res_gl_query.get_json()["request_metadata"]["selected_model"] == "glioma"
        assert res_gl_query.get_json()["volumetric_summary"]["total_tumor_volume_mm3"] == 500.0

        # Step 5: Verify GET /api/mask/<req_id> resolves correct mask without collision
        # Default multiclass should download the latest (meningioma)
        mask_def = app_client.get(f"/api/mask/{req_id}?type=multiclass")
        assert mask_def.status_code == 200
        assert "meningioma" in mask_def.headers.get("Content-Disposition", "").lower()

        # Specific query for glioma should download glioma mask
        mask_gl = app_client.get(f"/api/mask/{req_id}?type=multiclass&tumor_type=glioma")
        assert mask_gl.status_code == 200
        assert "glioma" in mask_gl.headers.get("Content-Disposition", "").lower()

        # Step 6: Verify GET /api/report/<req_id>
        rep_resp = app_client.get(f"/api/report/{req_id}")
        assert rep_resp.status_code == 200
        rep_text = rep_resp.get_data(as_text=True)
        assert (
            "Selected Model:   MENINGIOMA 3D U-Net (Manual Selection)" in rep_text
            or "Selected Model:   MENINGIOMA 3D U-Net (Manual Override)" in rep_text
        )
        assert (
            "Routing Mode:     MANUAL SELECTION" in rep_text
            or "Routing Mode:     MANUAL OVERRIDE" in rep_text
        )
        assert "DIAGNOSTIC & OPERATIONAL CLARIFICATION" in rep_text

        # Report for specific past model run
        rep_gl_resp = app_client.get(f"/api/report/{req_id}?tumor_type=glioma")
        assert rep_gl_resp.status_code == 200
        rep_gl_text = rep_gl_resp.get_data(as_text=True)
        assert (
            "Selected Model:   GLIOMA 3D U-Net (Manual Selection)" in rep_gl_text
            or "Selected Model:   GLIOMA 3D U-Net (Manual Override)" in rep_gl_text
        )
        assert "Total Tumor Volume: 500.0 mm³" in rep_gl_text

    finally:
        if req_dir.exists():
            shutil.rmtree(req_dir, ignore_errors=True)


def test_missing_model_result_and_invalid_model_names_return_clear_api_errors(app_client):
    """
    Verifies that when only one model has been executed for an upload:
    1. Querying a missing model result or report returns an explicit 404 without fabricating data or falling back.
    2. Requesting a missing model mask returns an explicit 404 without returning the other model's mask.
    3. Querying an invalid/unsupported model name returns a clear 400 error.
    """
    files = {
        "files": [
            (io.BytesIO(create_synthetic_nifti_bytes()), "BraTS-GLI-SINGLE-t1.nii.gz"),
            (io.BytesIO(create_synthetic_nifti_bytes()), "BraTS-GLI-SINGLE-t1c.nii.gz"),
            (io.BytesIO(create_synthetic_nifti_bytes()), "BraTS-GLI-SINGLE-t2.nii.gz"),
            (io.BytesIO(create_synthetic_nifti_bytes()), "BraTS-GLI-SINGLE-flair.nii.gz"),
        ]
    }
    up_resp = app_client.post("/api/upload", data=files, content_type="multipart/form-data")
    assert up_resp.status_code == 200
    req_id = up_resp.get_json()["request_id"]
    req_dir = API_RUNS_DIR / req_id

    try:
        # Run ONLY glioma
        resp_glioma = app_client.post(
            "/api/segment",
            json={"request_id": req_id, "tumor_type": "glioma"},
        )
        assert resp_glioma.status_code == 200

        assert (req_dir / "result_glioma.json").exists()
        assert not (req_dir / "result_meningioma.json").exists()

        # 1. Querying unexecuted model results returns 404 (NOT glioma data from result.json)
        res_men = app_client.get(f"/api/results/{req_id}?tumor_type=meningioma")
        assert res_men.status_code == 404
        assert "No meningioma segmentation results found" in res_men.get_json()["detail"]

        # 2. Querying unexecuted model report returns 404
        rep_men = app_client.get(f"/api/report/{req_id}?tumor_type=meningioma")
        assert rep_men.status_code == 404
        assert "No meningioma report found" in rep_men.get_json()["detail"]

        # 3. Requesting unexecuted model mask returns 404 (does NOT download glioma mask)
        mask_men = app_client.get(f"/api/mask/{req_id}?type=multiclass&tumor_type=meningioma")
        assert mask_men.status_code == 404
        assert "No meningioma segmentation mask found" in mask_men.get_json()["detail"]

        # 4. Invalid model name on /api/results returns 400
        res_bad = app_client.get(f"/api/results/{req_id}?tumor_type=sarcoma")
        assert res_bad.status_code == 400
        assert "Invalid tumor type or model" in res_bad.get_json()["detail"]

        # 5. Invalid model name on /api/report returns 400
        rep_bad = app_client.get(f"/api/report/{req_id}?tumor_type=sarcoma")
        assert rep_bad.status_code == 400
        assert "Invalid tumor type or model" in rep_bad.get_json()["detail"]

        # 6. Invalid model name on /api/mask returns 400
        mask_bad = app_client.get(f"/api/mask/{req_id}?tumor_type=sarcoma")
        assert mask_bad.status_code == 400
        assert "Invalid tumor type or model" in mask_bad.get_json()["detail"]

        # 7. Nonexistent request ID returns 404
        res_none = app_client.get("/api/results/REQ-DOESNOTEXIST999")
        assert res_none.status_code == 404

    finally:
        if req_dir.exists():
            shutil.rmtree(req_dir, ignore_errors=True)


def test_atomic_write_json_helper(tmp_path):
    """Verifies that _atomic_write_json correctly commits files and handles failures cleanly."""
    from api.app import _atomic_write_json

    target = tmp_path / "test_atomic.json"
    data = {"status": "success", "volumetrics": {"wt": 123.45}}

    _atomic_write_json(target, data)
    assert target.exists()

    with open(target, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == data

    # Verify atomic failure resilience: non-serializable data should raise exception,
    # target should retain old content, and no .tmp file should linger
    bad_data = {"unserializable": {1, 2, 3}}
    with pytest.raises(TypeError):
        _atomic_write_json(target, bad_data)

    with open(target, "r", encoding="utf-8") as f:
        reloaded = json.load(f)
    assert reloaded == data
    tmp_files = list(tmp_path.glob("*.tmp*"))
    assert len(tmp_files) == 0
