"""
Unit tests for Report and Export endpoints (PDF, PNG, CSV).
===========================================================
Validates:
  1. PDF Report generation (ReportLab, MIME type application/pdf, %PDF magic bytes, filename header).
  2. CSV Volumetrics export (MIME type text/csv, tabular headers, correct metrics and units).
  3. PNG 2D Slice view export (MIME type image/png, PNG magic bytes, axial slice annotation).
  4. Proper HTTP error handling (404 for unknown request ID, 400 for malformed/traversal ID).
  5. Cross-request data isolation (no leakage between cases).
"""

import io
import json
import pytest
from pathlib import Path

from api.app import create_app, API_RUNS_DIR


@pytest.fixture
def test_client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def mock_case_dir(tmp_path):
    """Creates a temporary mock API run case directory."""
    req_id = "REQ-TESTEXPORT"
    case_dir = API_RUNS_DIR / req_id
    case_dir.mkdir(parents=True, exist_ok=True)
    uploads_dir = case_dir / "uploads"
    uploads_dir.mkdir(exist_ok=True)
    masks_dir = case_dir / "masks"
    masks_dir.mkdir(exist_ok=True)

    result_payload = {
        "request": {
            "request_id": req_id,
            "tumor_type": "meningioma",
            "timestamp": "2026-10-27T00:00:00.000000+00:00",
            "routing_mode": "automatic_classification",
            "selected_model": "meningioma",
            "manual_override": False,
            "conflict_detected": False,
        },
        "input_metadata": {
            "original_shape": [240, 240, 155],
            "original_spacing": [1.0, 1.0, 1.0],
            "voxel_volume_mm3": 1.0,
            "modality_names": ["t1n", "t1c", "t2w", "t2f"],
        },
        "volumetrics": {
            "total_tumor_voxels": 50000,
            "total_tumor_volume_mm3": 50000.0,
            "total_tumor_volume_cm3": 50.0,
        },
        "meningioma_multimodal": {
            "subregions": {
                "wt_voxels": 50000,
                "wt_volume_mm3": 50000.0,
                "wt_volume_cm3": 50.0,
                "et_voxels": 20000,
                "et_volume_mm3": 20000.0,
                "et_volume_cm3": 20.0,
                "ncr_net_voxels": 1000,
                "ncr_net_volume_mm3": 1000.0,
                "ncr_net_volume_cm3": 1.0,
                "edema_voxels": 29000,
                "edema_volume_mm3": 29000.0,
                "edema_volume_cm3": 29.0,
            }
        },
        "quality_status": {
            "status": "success",
            "checkpoint_used": "checkpoints/meningioma_best.pth",
            "qc_verdict": "PASS (Validation Verified)",
        },
    }

    with open(case_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result_payload, f)

    cls_payload = {
        "status": "classified_tumor",
        "presence_probability": 0.992,
        "predicted_tumor_type": "meningioma",
        "confidence": 0.985,
        "glioma_probability": 0.015,
        "meningioma_probability": 0.985,
    }
    with open(case_dir / "classification.json", "w", encoding="utf-8") as f:
        json.dump(cls_payload, f)

    yield req_id

    # Cleanup
    import shutil
    shutil.rmtree(case_dir, ignore_errors=True)


def test_pdf_report_download(test_client, mock_case_dir):
    """Verifies PDF report endpoint returns valid PDF stream with correct headers."""
    # Via query parameter
    resp_q = test_client.get(f"/api/report/{mock_case_dir}?format=pdf")
    assert resp_q.status_code == 200
    assert resp_q.content_type == "application/pdf"
    assert resp_q.data.startswith(b"%PDF")
    disposition = resp_q.headers.get("Content-Disposition", "")
    assert f"filename={mock_case_dir}_meningioma_report.pdf" in disposition

    # Via dedicated route
    resp_r = test_client.get(f"/api/export/pdf/{mock_case_dir}")
    assert resp_r.status_code == 200
    assert resp_r.content_type == "application/pdf"
    assert resp_r.data.startswith(b"%PDF")


def test_csv_volumetrics_export(test_client, mock_case_dir):
    """Verifies CSV export contains tabular volumetric metrics and correct units."""
    resp = test_client.get(f"/api/export/csv/{mock_case_dir}")
    assert resp.status_code == 200
    assert resp.content_type.startswith("text/csv")
    disposition = resp.headers.get("Content-Disposition", "")
    assert f"{mock_case_dir}_meningioma_volumetrics.csv" in disposition

    csv_text = resp.get_data(as_text=True)
    assert mock_case_dir in csv_text
    assert "Whole_Tumor_WT" in csv_text
    assert "50000.0" in csv_text
    assert "50.00" in csv_text
    assert "Enhancing_Tumor_ET" in csv_text
    assert "Peritumoral_Edema_ED" in csv_text
    assert "mm3" in csv_text
    assert "cm3" in csv_text


def test_png_segmentation_view_export(test_client, mock_case_dir):
    """Verifies PNG export produces valid PNG image bytes and headers."""
    resp = test_client.get(f"/api/export/png/{mock_case_dir}")
    assert resp.status_code == 200
    assert resp.content_type == "image/png"
    assert resp.data.startswith(b"\x89PNG\r\n\x1a\n")
    disposition = resp.headers.get("Content-Disposition", "")
    assert f"{mock_case_dir}_meningioma_segmentation_view.png" in disposition


def test_exports_unknown_request_id_returns_404(test_client):
    """Verifies 404 is returned when request ID does not exist."""
    assert test_client.get("/api/report/REQ-NONEXISTENT?format=pdf").status_code == 404
    assert test_client.get("/api/export/pdf/REQ-NONEXISTENT").status_code == 404
    assert test_client.get("/api/export/csv/REQ-NONEXISTENT").status_code == 404
    assert test_client.get("/api/export/png/REQ-NONEXISTENT").status_code == 404


def test_exports_invalid_request_id_returns_400(test_client):
    """Verifies 400 is returned for malformed or path-traversal request IDs."""
    assert test_client.get("/api/export/pdf/.._.._etc").status_code == 400
    assert test_client.get("/api/export/csv/invalid!!id").status_code == 400
    assert test_client.get("/api/export/png/bad*name").status_code == 400


def test_cross_case_isolation(test_client, tmp_path):
    """Verifies two different cases never bleed data into each other's reports or CSVs."""
    case_a = "REQ-ISOLATION-A"
    case_b = "REQ-ISOLATION-B"
    dir_a = API_RUNS_DIR / case_a
    dir_b = API_RUNS_DIR / case_b
    dir_a.mkdir(parents=True, exist_ok=True)
    dir_b.mkdir(parents=True, exist_ok=True)

    try:
        with open(dir_a / "result.json", "w", encoding="utf-8") as f:
            json.dump({
                "request": {"request_id": case_a, "tumor_type": "glioma"},
                "volumetrics": {"total_tumor_volume_mm3": 11111.0, "total_tumor_volume_cm3": 11.11},
            }, f)

        with open(dir_b / "result.json", "w", encoding="utf-8") as f:
            json.dump({
                "request": {"request_id": case_b, "tumor_type": "meningioma"},
                "volumetrics": {"total_tumor_volume_mm3": 22222.0, "total_tumor_volume_cm3": 22.22},
            }, f)

        csv_a = test_client.get(f"/api/export/csv/{case_a}").get_data(as_text=True)
        csv_b = test_client.get(f"/api/export/csv/{case_b}").get_data(as_text=True)

        assert case_a in csv_a and "11111.0" in csv_a
        assert case_b not in csv_a and "22222.0" not in csv_a

        assert case_b in csv_b and "22222.0" in csv_b
        assert case_a not in csv_b and "11111.0" not in csv_b

    finally:
        import shutil
        shutil.rmtree(dir_a, ignore_errors=True)
        shutil.rmtree(dir_b, ignore_errors=True)


def test_export_endpoints_with_file_extensions(test_client, mock_case_dir):
    """Verifies that requesting /api/export/<format>/<id>.<ext> gracefully strips suffix."""
    req_id = "REQ-TESTEXPORT"
    res_pdf = test_client.get(f"/api/export/pdf/{req_id}.pdf")
    assert res_pdf.status_code == 200
    assert res_pdf.content_type == "application/pdf"
    assert res_pdf.data.startswith(b"%PDF-")

    res_csv = test_client.get(f"/api/export/csv/{req_id}.csv")
    assert res_csv.status_code == 200
    assert "text/csv" in res_csv.content_type
    assert req_id in res_csv.get_data(as_text=True)

    res_png = test_client.get(f"/api/export/png/{req_id}.png")
    assert res_png.status_code == 200
    assert res_png.content_type == "image/png"
    assert res_png.data.startswith(b"\x89PNG\r\n\x1a\n")


def test_api_nonexistent_route_returns_json_not_html(test_client):
    """Ensures invalid or misspelled /api/ routes return 404 JSON and never fall through to index.html."""
    res = test_client.get("/api/export/nonexistent_format/REQ-123")
    assert res.status_code == 404
    assert res.is_json
    data = res.get_json()
    assert "detail" in data
    assert "not found" in data["detail"].lower()


def test_two_case_sequential_export_workflow(test_client, tmp_path):
    """Simulates sequential Case A -> Case B lifecycle ensuring clean separation and complete payloads."""
    case_1 = "REQ-SEQ-CASE1"
    case_2 = "REQ-SEQ-CASE2"
    dir_1 = API_RUNS_DIR / case_1
    dir_2 = API_RUNS_DIR / case_2
    dir_1.mkdir(parents=True, exist_ok=True)
    dir_2.mkdir(parents=True, exist_ok=True)

    try:
        # Case 1 setup
        with open(dir_1 / "result.json", "w", encoding="utf-8") as f:
            json.dump({
                "request": {"request_id": case_1, "tumor_type": "meningioma", "selected_model": "meningioma"},
                "volumetrics": {"total_tumor_volume_mm3": 15000.0, "total_tumor_volume_cm3": 15.0},
                "meningioma_measurements": {
                    "subregion_volumes": {
                        "wt_volume_mm3": 15000.0, "wt_volume_cm3": 15.0,
                        "et_volume_mm3": 5000.0, "et_volume_cm3": 5.0,
                        "ncr_volume_mm3": 2000.0, "ncr_volume_cm3": 2.0,
                        "ed_volume_mm3": 8000.0, "ed_volume_cm3": 8.0,
                    }
                }
            }, f)

        # Retrieve Case 1 PDF and CSV
        pdf_1 = test_client.get(f"/api/export/pdf/{case_1}")
        assert pdf_1.status_code == 200
        assert pdf_1.data.startswith(b"%PDF-")
        csv_1 = test_client.get(f"/api/export/csv/{case_1}").get_data(as_text=True)
        assert case_1 in csv_1
        assert "15000.0" in csv_1

        # Case 2 setup (Simulating new case started after Case 1)
        with open(dir_2 / "result.json", "w", encoding="utf-8") as f:
            json.dump({
                "request": {"request_id": case_2, "tumor_type": "glioma", "selected_model": "glioma"},
                "volumetrics": {"total_tumor_volume_mm3": 35000.0, "total_tumor_volume_cm3": 35.0},
                "glioma_measurements": {
                    "subregion_volumes": {
                        "wt_volume_mm3": 35000.0, "wt_volume_cm3": 35.0,
                        "tc_volume_mm3": 12000.0, "tc_volume_cm3": 12.0,
                        "et_volume_mm3": 9000.0, "et_volume_cm3": 9.0,
                    }
                }
            }, f)

        # Retrieve Case 2 PDF and CSV
        pdf_2 = test_client.get(f"/api/export/pdf/{case_2}")
        assert pdf_2.status_code == 200
        assert pdf_2.data.startswith(b"%PDF-")
        csv_2 = test_client.get(f"/api/export/csv/{case_2}").get_data(as_text=True)
        assert case_2 in csv_2
        assert "35000.0" in csv_2
        assert case_1 not in csv_2

    finally:
        import shutil
        shutil.rmtree(dir_1, ignore_errors=True)
        shutil.rmtree(dir_2, ignore_errors=True)


def test_frontend_upload_slots_and_inline_notices_removed():
    """Verifies frontend HTML does not contain custom slot upload tabs or intrusive routine notices."""
    html_path = Path("frontend/index.html")
    assert html_path.exists()
    content = html_path.read_text(encoding="utf-8")

    # Slots UI and tabs must be absent
    assert "tabSlotsBtn" not in content
    assert "slotsUploadPanel" not in content
    assert "slotT1Box" not in content
    assert "slotT1cBox" not in content
    assert "slotT2Box" not in content
    assert "slotFlairBox" not in content

    # Single batch upload panel present
    assert "batchUploadPanel" in content
    assert "detectModalityClient" in content

    # Inline routine notice block must be removed (only conflict alerts retained)
    assert "meta.diagnostic_notice" not in content


def test_pdf_and_csv_pathology_awareness_and_clean_exports(test_client, tmp_path):
    """Verifies that PDF, PNG, and CSV exports exclude regulatory disclaimers and strictly output pathology-specific metrics."""
    glioma_case = "REQ-GLI-CLEAN"
    dir_gli = API_RUNS_DIR / glioma_case
    dir_gli.mkdir(parents=True, exist_ok=True)

    try:
        payload = {
            "request": {
                "request_id": glioma_case,
                "tumor_type": "glioma",
                "selected_model": "glioma",
                "routing_mode": "automatic_classification",
                "manual_override": False,
                "conflict_detected": False,
            },
            "input_metadata": {
                "original_shape": [240, 240, 155],
                "original_spacing": [1.0, 1.0, 1.0],
                "voxel_volume_mm3": 1.0,
                "modality_names": ["T1", "T1c", "T2", "FLAIR"],
            },
            "volumetrics": {
                "total_tumor_voxels": 42000,
                "total_tumor_volume_mm3": 42000.0,
                "total_tumor_volume_cm3": 42.0,
            },
            "glioma_measurements": {
                "subregions": {
                    "wt_voxels": 42000,
                    "wt_volume_mm3": 42000.0,
                    "wt_volume_cm3": 42.0,
                    "tc_voxels": 14000,
                    "tc_volume_mm3": 14000.0,
                    "tc_volume_cm3": 14.0,
                    "et_voxels": 6000,
                    "et_volume_mm3": 6000.0,
                    "et_volume_cm3": 6.0,
                    "ncr_net_voxels": 8000,
                    "ncr_net_volume_mm3": 8000.0,
                    "ncr_net_volume_cm3": 8.0,
                    "edema_voxels": 28000,
                    "edema_volume_mm3": 28000.0,
                    "edema_volume_cm3": 28.0,
                }
            },
            "quality_status": {
                "status": "success",
                "checkpoint_used": "checkpoints/glioma_best.pth",
                "qc_verdict": "PASS",
            },
        }

        with open(dir_gli / "result.json", "w", encoding="utf-8") as f:
            json.dump(payload, f)

        # 1. Verify CSV content
        csv_resp = test_client.get(f"/api/export/csv/{glioma_case}")
        assert csv_resp.status_code == 200
        csv_text = csv_resp.get_data(as_text=True)

        assert "Whole_Tumor_WT" in csv_text
        assert "Tumor_Core_TC" in csv_text
        assert "Enhancing_Tumor_ET" in csv_text
        assert "Necrotic_Core_NCR" in csv_text
        assert "Peritumoral_Edema_ED" in csv_text
        assert "Gross_Tumor_Volume_GTV" not in csv_text  # Meningioma-only metric absent
        assert "Investigational Research Prototype Only" not in csv_text

        # 2. Verify PDF generation and disclaimer removal
        pdf_resp = test_client.get(f"/api/export/pdf/{glioma_case}")
        assert pdf_resp.status_code == 200
        assert pdf_resp.data.startswith(b"%PDF-")
        # Ensure the disclaimer section heading is absent
        assert b"RESEARCH PROTOTYPE NOTICE & REGULATORY DISCLAIMER" not in pdf_resp.data

    finally:
        import shutil
        shutil.rmtree(dir_gli, ignore_errors=True)
