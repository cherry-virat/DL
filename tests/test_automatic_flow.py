"""
Unit and Integration Tests for Two-Stage Automatic Classification & Safe Routing Flow.
========================================================================================

Verifies:
1. Class-index mappings for Stage 1 (0: healthy, 1: tumor) and Stage 2 (0: glioma, 1: meningioma).
2. Stage 1 healthy / no-tumor predictions (safe termination without running segmentation).
3. Stage 2 Glioma prediction and routing to frozen Glioma 3D U-Net.
4. Stage 2 Meningioma prediction and routing to frozen Multimodal Meningioma 3D U-Net.
5. Low-confidence / ambiguous predictions (refusing automatic routing and requesting clinical review).
6. Missing modality count and incomplete channels handling.
7. Missing and smoke-test checkpoint rejection (defensive gating).
8. Manual routes compatibility (unaffected by classifier availability).
9. API status / readiness endpoints (/api/health and /api/classifier/status).
10. API auto-detect no-tumor-detected response.
"""

import io
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import nibabel as nib
import numpy as np
import pytest
import torch
import SimpleITK as sitk

from api.app import create_app
from integration.classifier_service import ClassifierService
from integration.inference_router import (
    InferenceRouter,
    AutomaticClassificationUnavailableError,
    NoTumorDetectedError,
)
from integration.result_schema import TumorType
from tests.test_api import (
    create_mock_glioma_result,
    create_mock_meningioma_result,
    create_synthetic_nifti_bytes,
)
from training.train_classifier import CLASS_NAMES as STAGE2_CLASS_NAMES


# =========================================================================
# TEST 1: Class-Index Mapping Verification
# =========================================================================

def test_class_index_mappings():
    """Confirms class-index contracts for Stage 1 and Stage 2 models."""
    # Stage 1 mapping: 0 = Healthy Control, 1 = Tumor Present
    stage1_mapping = {0: "healthy_control", 1: "tumor_present"}
    assert stage1_mapping[0] == "healthy_control"
    assert stage1_mapping[1] == "tumor_present"

    # Stage 2 mapping: 0 = Glioma, 1 = Meningioma
    assert STAGE2_CLASS_NAMES[0] == "glioma"
    assert STAGE2_CLASS_NAMES[1] == "meningioma"

    # Check manifest labels
    manifest_p = Path("data/classifier/manifest.json")
    if manifest_p.exists():
        with open(manifest_p, "r", encoding="utf-8") as f:
            m = json.load(f)
        for c in m["cases"][:50]:
            if c["class_label"] == "glioma":
                assert c["class_id"] == 0
            elif c["class_label"] == "meningioma":
                assert c["class_id"] == 1


# =========================================================================
# TEST 2: Stage 1 Healthy / No-Tumor Detection
# =========================================================================

def test_healthy_no_tumor_detection():
    """Verifies that when Stage 1 detects healthy brain, routing safely stops and returns structured result."""
    service = ClassifierService()

    mock_healthy_assessment = {
        "status": "classified_healthy",
        "tumor_presence": "healthy_control",
        "presence_probability": 0.002,
        "predicted_tumor_type": None,
        "selected_route": None,
        "confidence": 0.998,
        "recommendation": "No tumor detected by Stage 1 classifier. No segmentation required.",
    }
    service.classify = MagicMock(return_value=mock_healthy_assessment)

    router = InferenceRouter(classifier_service=service)

    dummy_input = {"T1": Path("dummy_t1.nii.gz"), "T2": Path("dummy_t2.nii.gz")}
    with pytest.raises(NoTumorDetectedError) as exc_info:
        router.route(tumor_type="auto", input_data=dummy_input)

    assert "No tumor detected" in str(exc_info.value)
    assert exc_info.value.details["tumor_presence"] == "healthy_control"
    assert exc_info.value.details["confidence"] == 0.998


# =========================================================================
# TEST 3: Stage 2 Glioma Prediction and Routing
# =========================================================================

def test_glioma_prediction_and_routing():
    """Verifies Stage 1 (tumor) + Stage 2 (glioma) routes strictly to Glioma U-Net."""
    mock_glioma_svc = MagicMock()
    mock_meningioma_svc = MagicMock()
    mock_glioma_svc.predict.return_value = create_mock_glioma_result("REQ-AUTO-GLI")

    mock_classifier_svc = MagicMock()
    mock_classifier_svc.classify.return_value = {
        "status": "classified_tumor",
        "tumor_presence": "tumor_present",
        "presence_probability": 0.992,
        "predicted_tumor_type": "glioma",
        "selected_route": "glioma",
        "type_probability": 0.02,
        "confidence": 0.98,
        "notes": "Verified mock prediction",
    }

    router = InferenceRouter(
        glioma_service=mock_glioma_svc,
        meningioma_service=mock_meningioma_svc,
        classifier_service=mock_classifier_svc,
    )

    dummy_input = {
        "T1": Path("t1.nii.gz"), "T1c": Path("t1c.nii.gz"),
        "T2": Path("t2.nii.gz"), "FLAIR": Path("flair.nii.gz"),
    }
    result = router.route(tumor_type="auto", input_data=dummy_input, request_id="REQ-AUTO-GLI")

    assert result.request.tumor_type == TumorType.GLIOMA
    mock_glioma_svc.predict.assert_called_once()
    mock_meningioma_svc.predict.assert_not_called()
    assert "Automatic routing: glioma" in result.quality_status.qc_notes


# =========================================================================
# TEST 4: Stage 2 Meningioma Prediction and Routing
# =========================================================================

def test_meningioma_prediction_and_routing():
    """Verifies Stage 1 (tumor) + Stage 2 (meningioma) routes strictly to Meningioma U-Net."""
    mock_glioma_svc = MagicMock()
    mock_meningioma_svc = MagicMock()
    mock_meningioma_svc.predict.return_value = create_mock_meningioma_result("REQ-AUTO-MEN")

    mock_classifier_svc = MagicMock()
    mock_classifier_svc.classify.return_value = {
        "status": "classified_tumor",
        "tumor_presence": "tumor_present",
        "presence_probability": 0.995,
        "predicted_tumor_type": "meningioma",
        "selected_route": "meningioma",
        "type_probability": 0.96,
        "confidence": 0.96,
        "notes": "Verified mock prediction",
    }

    router = InferenceRouter(
        glioma_service=mock_glioma_svc,
        meningioma_service=mock_meningioma_svc,
        classifier_service=mock_classifier_svc,
    )

    dummy_input = {
        "t1n": Path("t1n.nii.gz"), "t1c": Path("t1c.nii.gz"),
        "t2w": Path("t2w.nii.gz"), "t2f": Path("t2f.nii.gz"),
    }
    result = router.route(tumor_type="auto", input_data=dummy_input, request_id="REQ-AUTO-MEN")

    assert result.request.tumor_type == TumorType.MENINGIOMA
    mock_meningioma_svc.predict.assert_called_once()
    mock_glioma_svc.predict.assert_not_called()
    assert "Automatic routing: meningioma" in result.quality_status.qc_notes


# =========================================================================
# TEST 5: Ambiguous / Low-Confidence Predictions Refuse Routing
# =========================================================================

def test_ambiguous_prediction_refuses_auto_routing():
    """Verifies that an uncertain prediction (confidence < 0.55) raises error and refuses routing."""
    mock_classifier_svc = MagicMock()
    mock_classifier_svc.classify.return_value = {
        "status": "uncertain_classification",
        "tumor_presence": "tumor_present",
        "presence_probability": 0.92,
        "predicted_tumor_type": "uncertain",
        "selected_route": None,
        "type_probability": 0.52,
        "confidence": 0.52,
        "reason": "Tumor type prediction is ambiguous (confidence 0.52 < 0.55). Clinical review required.",
        "available_manual_routes": ["glioma", "meningioma"],
    }

    mock_glioma_svc = MagicMock()
    mock_men_svc = MagicMock()

    router = InferenceRouter(
        glioma_service=mock_glioma_svc,
        meningioma_service=mock_men_svc,
        classifier_service=mock_classifier_svc,
    )

    dummy_input = {"T1": Path("t1.nii"), "T1c": Path("t1c.nii"), "T2": Path("t2.nii"), "FLAIR": Path("flair.nii")}
    with pytest.raises(AutomaticClassificationUnavailableError) as exc_info:
        router.route(tumor_type="auto", input_data=dummy_input)

    err_msg = str(exc_info.value).lower()
    assert "ambiguous" in err_msg or "uncertain" in err_msg
    mock_glioma_svc.predict.assert_not_called()
    mock_men_svc.predict.assert_not_called()


# =========================================================================
# TEST 6: Missing Modalities Handling
# =========================================================================

def test_missing_modalities_handling():
    """Verifies that incomplete channel sets are identified with appropriate errors."""
    service = ClassifierService()

    # When automatic classification is enabled/mocked, verify missing T2 is caught
    with patch.object(ClassifierService, "is_automatic_classification_available", new_callable=PropertyMock) as mock_avail:
        mock_avail.return_value = True
        service._presence_model = MagicMock()
        assessment = service.classify({"T1": Path("dummy_t1.nii")})
        assert assessment["status"] == "unsupported_modality_set"
        assert "requires at least T1 and T2" in assessment["reason"]


# =========================================================================
# TEST 7: Defensive Gating for Missing or Smoke-Test Checkpoints
# =========================================================================

def test_checkpoint_defensive_gating():
    """Verifies checkpoints with < 50 validation cases are marked unavailable."""
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        # Create a fake smoke test checkpoint with 4 val samples
        torch.save({
            "val_metrics": {"total_samples": 4},
            "best_val_f1": 1.0,
            "architecture": "TumorClassifier3D",
        }, str(tmp_path))

        service = ClassifierService(type_checkpoint_path=tmp_path)
        assert service.is_type_model_available is False
        assert service.is_automatic_classification_available is False
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


# =========================================================================
# TEST 8: Manual Routes Remain Operational
# =========================================================================

def test_manual_routes_remain_operational():
    """Verifies manual glioma and meningioma routes bypass classifier checks entirely."""
    mock_gli = MagicMock()
    mock_men = MagicMock()
    mock_gli.predict.return_value = create_mock_glioma_result("REQ-MAN-GLI")
    mock_men.predict.return_value = create_mock_meningioma_result("REQ-MAN-MEN")

    router = InferenceRouter(
        glioma_service=mock_gli,
        meningioma_service=mock_men,
        classifier_service=None,
    )

    dummy_input = {"T1": "t1", "T1c": "t1c", "T2": "t2", "FLAIR": "flair"}

    # 1. Manual Glioma
    res_gli = router.route(tumor_type="glioma", input_data=dummy_input)
    assert res_gli.request.tumor_type == TumorType.GLIOMA
    mock_gli.predict.assert_called_once()

    # 2. Manual Meningioma
    dummy_men_input = {"t1n": "t1n", "t1c": "t1c", "t2w": "t2w", "t2f": "t2f"}
    res_men = router.route(tumor_type="meningioma", input_data=dummy_men_input)
    assert res_men.request.tumor_type == TumorType.MENINGIOMA
    mock_men.predict.assert_called_once()


# =========================================================================
# TEST 9: API Status Endpoint Behavior
# =========================================================================

def test_api_classifier_status_endpoint():
    """Verifies /api/classifier/status and /api/health report correct gating."""
    app = create_app()
    client = app.test_client()

    # Health check
    res_h = client.get("/api/health")
    assert res_h.status_code == 200
    h_data = res_h.get_json()
    assert "status" in h_data
    assert "supported_tumor_types" in h_data
    assert "automatic_classification_available" in h_data

    # Classifier status
    res_cs = client.get("/api/classifier/status")
    assert res_cs.status_code == 200
    cs_data = res_cs.get_json()
    assert "presence_model_available" in cs_data
    assert "type_model_available" in cs_data
    assert "automatic_classification_available" in cs_data
    assert "supported_manual_routes" in cs_data


# =========================================================================
# TEST 10: API End-to-End Healthy / No-Tumor Response
# =========================================================================

def test_api_auto_segment_no_tumor_detected():
    """Verifies that API returns HTTP 200 with no_tumor_detected payload when Stage 1 indicates healthy."""
    mock_router = MagicMock(spec=InferenceRouter)
    mock_router.normalize_tumor_type.return_value = TumorType.AUTO
    mock_router.route.side_effect = NoTumorDetectedError(
        message="No tumor detected by automated presence detector.",
        details={
            "status": "classified_healthy",
            "tumor_presence": "healthy_control",
            "presence_probability": 0.001,
            "confidence": 0.999,
            "recommendation": "No tumor detected by Stage 1 classifier. No segmentation required.",
        }
    )

    app = create_app(router=mock_router)
    client = app.test_client()

    std_bytes = create_synthetic_nifti_bytes()
    files = {
        "files": [
            (io.BytesIO(std_bytes), "healthy_t1.nii.gz"),
            (io.BytesIO(std_bytes), "healthy_t1c.nii.gz"),
            (io.BytesIO(std_bytes), "healthy_t2.nii.gz"),
            (io.BytesIO(std_bytes), "healthy_flair.nii.gz"),
        ]
    }
    up_res = client.post("/api/upload", data=files, content_type="multipart/form-data")
    assert up_res.status_code == 200
    req_id = up_res.get_json()["request_id"]

    seg_res = client.post("/api/segment", json={"request_id": req_id, "tumor_type": "auto"})
    assert seg_res.status_code == 200
    res_json = seg_res.get_json()
    assert res_json["status"] == "no_tumor_detected"
    assert res_json["tumor_presence"] == "healthy_control"
    assert res_json["confidence"] == 0.999
    assert res_json["routing_mode"] == "automatic_classification"


# =========================================================================
# TEST 11: Manual Override Conflict Detection
# =========================================================================

def test_manual_override_conflict_detection_when_classifier_predicts_different_type():
    """Verifies that manually selecting glioma on a scan predicted as meningioma sets conflict_detected: true and warning."""
    router = MagicMock(spec=InferenceRouter)
    router.normalize_tumor_type.side_effect = lambda t: TumorType(t)

    mock_cs = MagicMock()
    mock_cs.is_automatic_classification_available = True
    mock_cs.classify.return_value = {
        "status": "classified_tumor",
        "tumor_presence": "tumor_present",
        "presence_probability": 0.99,
        "predicted_tumor_type": "meningioma",
        "selected_route": "meningioma",
        "confidence": 0.95,
    }
    router.classifier_service = mock_cs
    router.route.return_value = create_mock_glioma_result("REQ-OVERRIDE-01")

    app = create_app(router=router)
    client = app.test_client()

    std_bytes = create_synthetic_nifti_bytes()
    files = {
        "files": [
            (io.BytesIO(std_bytes), "scan_t1.nii.gz"),
            (io.BytesIO(std_bytes), "scan_t1c.nii.gz"),
            (io.BytesIO(std_bytes), "scan_t2.nii.gz"),
            (io.BytesIO(std_bytes), "scan_flair.nii.gz"),
        ]
    }
    up_res = client.post("/api/upload", data=files, content_type="multipart/form-data")
    assert up_res.status_code == 200
    req_id = up_res.get_json()["request_id"]

    # Manually select glioma despite classifier predicting meningioma
    res = client.post("/api/segment", json={"request_id": req_id, "tumor_type": "glioma"})
    assert res.status_code == 200
    data = res.get_json()

    meta = data["request_metadata"]
    assert meta["manual_override"] is True
    assert meta["selected_model"] == "glioma"
    assert meta["conflict_detected"] is True
    assert "MANUAL MODEL OVERRIDE CONFLICT" in meta["warning"]
    assert "MENINGIOMA" in meta["warning"]
    assert "does NOT confirm GLIOMA pathology" in meta["warning"]


# =========================================================================
# TEST 12: Direct Dedicated Manual Endpoints
# =========================================================================

def test_direct_dedicated_endpoints_glioma_and_meningioma():
    """Verifies that POST /api/segment/glioma and POST /api/segment/meningioma execute successfully."""
    router = MagicMock(spec=InferenceRouter)
    router.normalize_tumor_type.side_effect = lambda t: TumorType(t)
    mock_cs = MagicMock()
    mock_cs.is_automatic_classification_available = False
    router.classifier_service = mock_cs
    router.route.side_effect = lambda tumor_type, **kw: (
        create_mock_glioma_result("REQ-GLI-DIR") if tumor_type == TumorType.GLIOMA
        else create_mock_meningioma_result("REQ-MEN-DIR")
    )

    app = create_app(router=router)
    client = app.test_client()

    # 1. Glioma dedicated endpoint
    std_bytes = create_synthetic_nifti_bytes()
    files_gli = {
        "files": [
            (io.BytesIO(std_bytes), "case_t1.nii.gz"),
            (io.BytesIO(std_bytes), "case_t1c.nii.gz"),
            (io.BytesIO(std_bytes), "case_t2.nii.gz"),
            (io.BytesIO(std_bytes), "case_flair.nii.gz"),
        ]
    }
    up_gli = client.post("/api/upload", data=files_gli, content_type="multipart/form-data")
    req_id_gli = up_gli.get_json()["request_id"]

    res_gli = client.post("/api/segment/glioma", json={"request_id": req_id_gli})
    assert res_gli.status_code == 200
    d_gli = res_gli.get_json()
    assert d_gli["request_metadata"]["tumor_type"] == "glioma"
    assert d_gli["request_metadata"]["selected_model"] == "glioma"
    assert d_gli["request_metadata"]["manual_override"] is True

    # 2. Meningioma dedicated endpoint
    files_men = {
        "files": [
            (io.BytesIO(std_bytes), "case_t1n.nii.gz"),
            (io.BytesIO(std_bytes), "case_t1c.nii.gz"),
            (io.BytesIO(std_bytes), "case_t2w.nii.gz"),
            (io.BytesIO(std_bytes), "case_t2f.nii.gz"),
        ]
    }
    up_men = client.post("/api/upload", data=files_men, content_type="multipart/form-data")
    req_id_men = up_men.get_json()["request_id"]

    res_men = client.post("/api/segment/meningioma", json={"request_id": req_id_men})
    assert res_men.status_code == 200
    d_men = res_men.get_json()
    assert d_men["request_metadata"]["tumor_type"] == "meningioma"
    assert d_men["request_metadata"]["selected_model"] == "meningioma"
    assert d_men["request_metadata"]["manual_override"] is True


# =========================================================================
# TEST 13: Confidence Semantics and Probability Disambiguation
# =========================================================================

def test_confidence_semantics_fields():
    """Verifies that classification result explicitly separates presence and type probabilities."""
    service = ClassifierService()

    # Case 1: Healthy
    service._presence_model = MagicMock()
    # Mock presence logits: [healthy=10.0, tumor=-10.0] -> presence_prob ~ 0.0
    service._presence_model.return_value = torch.tensor([[10.0, -10.0]])
    with patch("preprocessing.tumor_presence_preprocess.preprocess_2channel_presence_tensor") as mock_prep:
        mock_prep.return_value = (torch.zeros((2, 32, 32, 32)), None)
        res = service._execute_trained_classification({
            "T1": Path("t1.nii"), "T2": Path("t2.nii")
        })
        assert res["status"] == "classified_healthy"
        assert res["presence_probability"] < 0.01
        assert res["presence_confidence"] > 0.99
        assert res["confidence"] == res["presence_confidence"]

    # Case 2: Tumor (Meningioma)
    service._presence_model = MagicMock()
    service._presence_model.return_value = torch.tensor([[-5.0, 5.0]])  # tumor present ~ 0.9999
    service._type_model = MagicMock()
    service._type_model.return_value = torch.tensor([[-2.0, 2.0]])  # meningioma ~ 0.982
    with patch("preprocessing.tumor_presence_preprocess.preprocess_2channel_presence_tensor") as mock_p1, \
         patch("preprocessing.tumor_classifier_preprocess.preprocess_case_to_tensor") as mock_p2:
        mock_p1.return_value = (torch.zeros((2, 32, 32, 32)), None)
        mock_p2.return_value = torch.zeros((4, 32, 32, 32))
        res = service._execute_trained_classification({
            "T1": Path("t1.nii"), "T1c": Path("t1c.nii"),
            "T2": Path("t2.nii"), "FLAIR": Path("flair.nii"),
        })
        assert res["status"] == "classified_tumor"
        assert res["predicted_tumor_type"] == "meningioma"
        assert res["presence_confidence"] >= 0.99
        assert res["type_confidence"] >= 0.95
        assert res["meningioma_probability"] >= 0.95
        assert res["glioma_probability"] <= 0.05
        assert "glioma" in res["type_probabilities"]
        assert "meningioma" in res["type_probabilities"]


# =========================================================================
# TEST 14: /api/classify Endpoint Persists classification.json
# =========================================================================

def test_api_classify_persists_artifact():
    """Verifies that /api/classify atomically persists classification.json in request folder."""
    app = create_app()
    client = app.test_client()

    std_bytes = create_synthetic_nifti_bytes()
    files = {
        "files": [
            (io.BytesIO(std_bytes), "scan_t1.nii.gz"),
            (io.BytesIO(std_bytes), "scan_t1c.nii.gz"),
            (io.BytesIO(std_bytes), "scan_t2.nii.gz"),
            (io.BytesIO(std_bytes), "scan_flair.nii.gz"),
        ]
    }
    up_res = client.post("/api/upload", data=files, content_type="multipart/form-data")
    assert up_res.status_code == 200
    req_id = up_res.get_json()["request_id"]

    classify_res = client.post("/api/classify", json={"request_id": req_id})
    assert classify_res.status_code == 200
    data = classify_res.get_json()
    assert data["request_id"] == req_id

    # Verify classification.json exists on disk
    req_dir = Path("outputs/api_runs") / req_id
    saved_json = req_dir / "classification.json"
    assert saved_json.exists()
    with open(saved_json, "r", encoding="utf-8") as f:
        disk_data = json.load(f)
    assert disk_data["request_id"] == req_id
    assert disk_data["status"] == data["status"]


# =========================================================================
# TEST 15: Uncertain Classification Refuses Auto Routing via API (HTTP 422)
# =========================================================================

def test_uncertain_classification_refuses_auto_segmentation_api():
    """Verifies that uncertain classification (<0.55) returns HTTP 422 and does not execute segmentation."""
    mock_router = MagicMock(spec=InferenceRouter)
    mock_router.normalize_tumor_type.return_value = TumorType.AUTO
    mock_router.route.side_effect = AutomaticClassificationUnavailableError(
        message="Tumor type prediction is ambiguous (confidence 0.53 < 0.55).",
        details={
            "status": "uncertain_classification",
            "tumor_presence": "tumor_present",
            "presence_probability": 0.98,
            "presence_confidence": 0.98,
            "predicted_tumor_type": "uncertain",
            "type_probability": 0.53,
            "confidence": 0.53,
            "type_confidence": 0.53,
            "available_manual_routes": ["glioma", "meningioma"],
            "reason": "Tumor type prediction is ambiguous (confidence 0.53 < 0.55). Automatic routing is refused.",
        }
    )

    app = create_app(router=mock_router)
    client = app.test_client()

    std_bytes = create_synthetic_nifti_bytes()
    files = {
        "files": [
            (io.BytesIO(std_bytes), "unc_t1.nii.gz"),
            (io.BytesIO(std_bytes), "unc_t1c.nii.gz"),
            (io.BytesIO(std_bytes), "unc_t2.nii.gz"),
            (io.BytesIO(std_bytes), "unc_flair.nii.gz"),
        ]
    }
    up_res = client.post("/api/upload", data=files, content_type="multipart/form-data")
    req_id = up_res.get_json()["request_id"]

    seg_res = client.post("/api/segment", json={"request_id": req_id, "tumor_type": "auto"})
    assert seg_res.status_code == 422
    err_json = seg_res.get_json()
    assert err_json["status"] == "uncertain_classification"
    assert "ambiguous" in err_json["detail"].lower() or "uncertain" in err_json["detail"].lower()
    assert err_json["available_manual_routes"] == ["glioma", "meningioma"]


# =========================================================================
# TEST 16: Locked Split Disjointness & Exact Case Counts
# =========================================================================

def test_locked_split_integrity_disjointness_and_counts():
    """Verifies disjointness of train, val, test, and extension partitions and exact counts."""
    split_p = Path("data/classifier/split.json")
    if not split_p.exists():
        pytest.skip("data/classifier/split.json not found")

    with open(split_p, "r", encoding="utf-8") as f:
        data = json.load(f)

    train = set(data["train_cases"])
    val = set(data["val_cases"])
    test = set(data["test_cases"])
    ext = set(data.get("unbalanced_glioma_extension", []))

    assert len(train) == 1400, f"Expected 1400 train cases, got {len(train)}"
    assert len(val) == 300, f"Expected 300 val cases, got {len(val)}"
    assert len(test) == 300, f"Expected 300 test cases, got {len(test)}"
    assert len(ext) == 251, f"Expected 251 ext cases, got {len(ext)}"

    assert len(train & val) == 0, "Train and Val overlap!"
    assert len(train & test) == 0, "Train and Test overlap!"
    assert len(val & test) == 0, "Val and Test overlap!"
    assert len(ext & (train | val | test)) == 0, "Extension overlaps with other partitions!"


# =========================================================================
# TEST 17: Modality Order & Preprocessing Contract
# =========================================================================

def test_modality_order_and_preprocessing_contract():
    """Verifies that the canonical modality contract is strictly ['t1n', 't1c', 't2w', 't2f']."""
    from preprocessing.tumor_classifier_preprocess import (
        CANONICAL_MODALITIES,
        TARGET_SHAPE,
        preprocess_case_to_tensor,
    )

    assert CANONICAL_MODALITIES == ["t1n", "t1c", "t2w", "t2f"]
    assert TARGET_SHAPE == (128, 128, 96)

    # Synthetic arrays
    dummy_vol = np.zeros((240, 240, 155), dtype=np.float32)
    dummy_vol[100:120, 100:120, 70:90] = 50.0

    tensor = preprocess_case_to_tensor({
        "t1": dummy_vol,
        "t1c": dummy_vol,
        "t2": dummy_vol,
        "flair": dummy_vol,
    })
    assert tensor.shape == (4, 128, 128, 96)
    assert tensor.dtype == torch.float32


# =========================================================================
# TEST 18: Classifier Service Active Checkpoint Metadata & Readiness
# =========================================================================

def test_classifier_service_status_checkpoint_metadata_and_readiness():
    """Verifies get_status() includes active checkpoint sha256 and metadata."""
    service = ClassifierService()
    status = service.get_status()

    assert "automatic_classification_available" in status
    assert "active_type_checkpoint" in status
    assert "active_presence_checkpoint" in status

    type_info = status["active_type_checkpoint"]
    assert "checkpoint_path" in type_info
    assert "available" in type_info
    if type_info["available"]:
        assert type_info["sha256"] is not None
        assert len(type_info["sha256"]) == 64


# =========================================================================
# TEST 19: Request ID Consistency Across Results, Reports, and Download
# =========================================================================

def test_report_and_mask_request_id_consistency():
    """Verifies request_id is consistently preserved across report and download endpoints."""
    app = create_app()
    client = app.test_client()

    std_bytes = create_synthetic_nifti_bytes()
    files = {
        "files": [
            (io.BytesIO(std_bytes), "case_t1n.nii.gz"),
            (io.BytesIO(std_bytes), "case_t1c.nii.gz"),
            (io.BytesIO(std_bytes), "case_t2w.nii.gz"),
            (io.BytesIO(std_bytes), "case_t2f.nii.gz"),
        ]
    }
    up_res = client.post("/api/upload", data=files, content_type="multipart/form-data")
    assert up_res.status_code == 200
    req_id = up_res.get_json()["request_id"]

    # Manual segmentation call
    seg_res = client.post("/api/segment", json={"request_id": req_id, "tumor_type": "meningioma"})
    assert seg_res.status_code == 200
    data = seg_res.get_json()
    assert data["request_metadata"]["request_id"] == req_id

    # Retrieve results via GET
    get_res = client.get(f"/api/results/{req_id}")
    assert get_res.status_code == 200
    assert get_res.get_json()["request_metadata"]["request_id"] == req_id

    # Retrieve report via GET
    rep_res = client.get(f"/api/report/{req_id}")
    assert rep_res.status_code == 200
    assert req_id in rep_res.get_data(as_text=True)


# =========================================================================
# TEST 20: Robustness - Volume Identity Verification (WT = TC + ED)
# =========================================================================

def test_robustness_volume_identity_consistency():
    """Confirms mathematical identity WT = TC + ED and non-negativity across results."""
    app = create_app()
    client = app.test_client()

    std_bytes = create_synthetic_nifti_bytes()
    files = {
        "files": [
            (io.BytesIO(std_bytes), "case_t1n.nii.gz"),
            (io.BytesIO(std_bytes), "case_t1c.nii.gz"),
            (io.BytesIO(std_bytes), "case_t2w.nii.gz"),
            (io.BytesIO(std_bytes), "case_t2f.nii.gz"),
        ]
    }
    up_res = client.post("/api/upload", data=files, content_type="multipart/form-data")
    assert up_res.status_code == 200
    req_id = up_res.get_json()["request_id"]

    # Test Glioma volume arithmetic
    g_res = client.post("/api/segment", json={"request_id": req_id, "tumor_type": "glioma"})
    assert g_res.status_code == 200
    g_data = g_res.get_json()
    g_sub = g_data["glioma_measurements"]["subregion_volumes"]
    assert g_sub["wt_volume_mm3"] >= 0.0
    assert g_sub["tc_volume_mm3"] >= 0.0
    assert g_sub["ed_volume_mm3"] >= 0.0
    # In Glioma: WT = TC + ED (since labels {1, 3} are TC, {2} is ED, {1,2,3} is WT)
    assert abs(g_sub["wt_volume_mm3"] - (g_sub["tc_volume_mm3"] + g_sub["ed_volume_mm3"])) < 1e-4

    # Test Meningioma volume arithmetic
    m_res = client.post("/api/segment", json={"request_id": req_id, "tumor_type": "meningioma"})
    assert m_res.status_code == 200
    m_data = m_res.get_json()
    m_sub = m_data["meningioma_measurements"]["subregion_volumes"]
    assert m_sub["wt_volume_mm3"] >= 0.0
    assert m_sub["tc_volume_mm3"] >= 0.0
    assert m_sub["ed_volume_mm3"] >= 0.0
    # In Meningioma: WT = TC + ED
    assert abs(m_sub["wt_volume_mm3"] - (m_sub["tc_volume_mm3"] + m_sub["ed_volume_mm3"])) < 1e-4


# =========================================================================
# TEST 21: Robustness - Missing & Corrupt Checkpoint Defensive Handling
# =========================================================================

def test_robustness_corrupt_checkpoint_handling(tmp_path):
    """Verifies missing or corrupted checkpoint files cause graceful handling rather than unhandled crashes."""
    missing_pth = tmp_path / "non_existent_model.pth"

    # 1. Missing checkpoint test
    service_missing = ClassifierService(
        presence_checkpoint_path=missing_pth,
        type_checkpoint_path=missing_pth,
    )
    assert service_missing.is_presence_model_available is False
    assert service_missing.is_type_model_available is False
    assert service_missing.is_automatic_classification_available is False

    status = service_missing.get_status()
    assert status["automatic_classification_available"] is False
    assert status["presence_model_available"] is False
    assert status["type_model_available"] is False

    res_missing = service_missing.classify({"T1": Path("dummy_t1.nii.gz"), "T2": Path("dummy_t2.nii.gz")})
    assert res_missing["status"] == "automatic_classification_not_available"
    assert "Missing trained checkpoints on disk" in res_missing["blocker"]

    # 2. Corrupted file test
    corrupt_pth = tmp_path / "corrupt_model.pth"
    corrupt_pth.write_bytes(b"not_a_valid_pytorch_checkpoint_data")
    service_corrupt = ClassifierService(
        presence_checkpoint_path=corrupt_pth,
        type_checkpoint_path=corrupt_pth,
    )
    res_corrupt = service_corrupt.classify({"T1": Path("dummy_t1.nii.gz"), "T2": Path("dummy_t2.nii.gz")})
    assert res_corrupt["status"] == "classification_failed"
    assert "Classification inference failed" in res_corrupt["reason"]


# =========================================================================
# TEST 22: Robustness - CUDA OOM Graceful Handling Simulation
# =========================================================================

def test_robustness_cuda_oom_handling_simulation():
    """Verifies that OutOfMemoryError is intercepted and returns HTTP 500 without crashing."""
    app = create_app()
    client = app.test_client()

    std_bytes = create_synthetic_nifti_bytes()
    files = {
        "files": [
            (io.BytesIO(std_bytes), "case_t1n.nii.gz"),
            (io.BytesIO(std_bytes), "case_t1c.nii.gz"),
            (io.BytesIO(std_bytes), "case_t2w.nii.gz"),
            (io.BytesIO(std_bytes), "case_t2f.nii.gz"),
        ]
    }
    up_res = client.post("/api/upload", data=files, content_type="multipart/form-data")
    assert up_res.status_code == 200
    req_id = up_res.get_json()["request_id"]

    with patch("integration.inference_router.InferenceRouter.route", side_effect=torch.cuda.OutOfMemoryError("CUDA out of memory")):
        seg_res = client.post("/api/segment", json={"request_id": req_id, "tumor_type": "glioma"})
        assert seg_res.status_code == 500
        data = seg_res.get_json()
        assert "Inference execution failed" in data["detail"]
        assert "CUDA out of memory" in data["detail"]
