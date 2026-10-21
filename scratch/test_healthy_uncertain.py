import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

import json
from unittest.mock import MagicMock
from api.app import create_app
from integration.inference_router import InferenceRouter, NoTumorDetectedError, AutomaticClassificationUnavailableError
from integration.result_schema import TumorType

# Test 4: Healthy Early Exit
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

from tests.test_api import create_synthetic_nifti_bytes
import io

nii = create_synthetic_nifti_bytes()
up = client.post("/api/upload", data={
    "files": [
        (io.BytesIO(nii), "scan_t1.nii.gz"),
        (io.BytesIO(nii), "scan_t1c.nii.gz"),
        (io.BytesIO(nii), "scan_t2.nii.gz"),
        (io.BytesIO(nii), "scan_flair.nii.gz"),
    ]
}, content_type="multipart/form-data")
req_h = up.get_json()["request_id"]

seg_h = client.post("/api/segment", json={"request_id": req_h, "tumor_type": "auto"})
print(f"Healthy early exit status code: {seg_h.status_code}")
print(f"Healthy payload: {seg_h.get_json()}")
rep_h = client.get(f"/api/report/{req_h}").get_data(as_text=True)
print("Healthy Report:\n" + rep_h)

assert seg_h.status_code == 200
assert "Selected Model:   None (Halted at Stage 1)" in rep_h
assert "Routing Mode:     AUTOMATIC CLASSIFICATION (Normative Scan)" in rep_h
assert "Pipeline Status:  NO TUMOR DETECTED (200 OK)" in rep_h

# Test 5: Uncertainty Rejection
mock_router.route.side_effect = AutomaticClassificationUnavailableError(
    message="Tumor type prediction is ambiguous (confidence 0.52 < 0.55).",
    details={
        "status": "uncertain_classification",
        "tumor_presence": "tumor_present",
        "presence_probability": 0.99,
        "confidence": 0.52,
        "available_manual_routes": ["glioma", "meningioma"],
    }
)

up_u = client.post("/api/upload", data={
    "files": [
        (io.BytesIO(nii), "scan_t1.nii.gz"),
        (io.BytesIO(nii), "scan_t1c.nii.gz"),
        (io.BytesIO(nii), "scan_t2.nii.gz"),
        (io.BytesIO(nii), "scan_flair.nii.gz"),
    ]
}, content_type="multipart/form-data")
req_u = up_u.get_json()["request_id"]

seg_u = client.post("/api/segment", json={"request_id": req_u, "tumor_type": "auto"})
print(f"\nUncertainty rejection status code: {seg_u.status_code}")
print(f"Uncertainty payload: {seg_u.get_json()}")
rep_u = client.get(f"/api/report/{req_u}").get_data(as_text=True)
print("Uncertainty Report:\n" + rep_u)

assert seg_u.status_code == 422
assert "Selected Model:   None (Refused - Ambiguous)" in rep_u
assert "Routing Mode:     AUTOMATIC CLASSIFICATION (Uncertainty Refusal)" in rep_u
assert "Pipeline Status:  UNCERTAIN CLASSIFICATION (422 UNPROCESSABLE ENTITY)" in rep_u

print("\n--- ALL HEALTHY & UNCERTAINTY REPORT FORMATS VERIFIED 100% ---")
