import io
from pathlib import Path
import sys
import unittest

# Ensure backend module can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app import app
from services.report_service import generate_report, get_report_file_path


class ReportApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.created_reports = []
        self.created_uploads = []

    def tearDown(self):
        # Clean up any test PDF reports created during test
        for report_id in self.created_reports:
            path = get_report_file_path(report_id)
            if path and path.is_file():
                try:
                    path.unlink()
                except OSError:
                    pass

        # Clean up any test uploaded MRI files
        for upload_path in self.created_uploads:
            if upload_path and upload_path.is_file():
                try:
                    upload_path.unlink()
                except OSError:
                    pass

    def test_health_endpoint(self):
        """Confirm /health continues to operate."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"status": "healthy"})

    def test_create_report_with_prediction_data(self):
        """Confirm POST /report succeeds when structured prediction data is provided."""
        payload = {
            "prediction": {
                "mode": "mock",
                "tumor_present": True,
                "tumor_type": "Meningioma",
                "confidence": 0.94,
                "disclaimer": "Demo mock prediction only."
            },
            "filename": "test_mri_scan.nii.gz"
        }
        response = self.client.post("/report", json=payload)
        self.assertEqual(response.status_code, 201)
        data = response.get_json()

        self.assertTrue(data["success"])
        self.assertEqual(data["mode"], "mock")
        self.assertIn("report_id", data)
        self.assertIn("download_url", data)
        self.created_reports.append(data["report_id"])

    def test_create_report_with_file_id(self):
        """Confirm POST /report succeeds when file_id of an uploaded scan is provided."""
        # First upload a test scan
        fake_nii = io.BytesIO(b"fake nifti scan data for report test")
        upload_resp = self.client.post(
            "/upload",
            data={"file": (fake_nii, "report_test_scan.nii.gz")},
            content_type="multipart/form-data"
        )
        self.assertEqual(upload_resp.status_code, 201)
        upload_data = upload_resp.get_json()
        file_id = upload_data["file_id"]

        # Track file for cleanup
        from services.file_service import get_uploaded_file
        uploaded_info = get_uploaded_file(file_id)
        if uploaded_info:
            self.created_uploads.append(Path(uploaded_info["path"]))

        # Now generate report by file_id
        response = self.client.post("/report", json={"file_id": file_id})
        self.assertEqual(response.status_code, 201)
        data = response.get_json()

        self.assertTrue(data["success"])
        self.assertEqual(data["mode"], "mock")
        self.assertTrue(data["report_id"].startswith("report_"))
        self.created_reports.append(data["report_id"])

    def test_download_report_success(self):
        """Confirm GET /report/download/<report_id> returns PDF file."""
        payload = {
            "prediction": {
                "mode": "mock",
                "tumor_present": True,
                "tumor_type": "Meningioma",
                "confidence": 0.94
            }
        }
        create_resp = self.client.post("/report", json=payload)
        self.assertEqual(create_resp.status_code, 201)
        report_id = create_resp.get_json()["report_id"]
        self.created_reports.append(report_id)

        # Download the report
        download_resp = self.client.get(f"/report/download/{report_id}")
        self.assertEqual(download_resp.status_code, 200)
        self.assertEqual(download_resp.content_type, "application/pdf")
        # PDF files begin with %PDF header
        self.assertTrue(download_resp.data.startswith(b"%PDF"))
        download_resp.close()

    def test_download_report_not_found(self):
        """Confirm download endpoint returns 404 for unknown report IDs."""
        response = self.client.get("/report/download/report_00000000000000000000000000000000")
        self.assertEqual(response.status_code, 404)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertEqual(data["error"], "Report not found.")

    def test_download_report_invalid_id(self):
        """Confirm download endpoint safely rejects path traversal attempts."""
        response = self.client.get("/report/download/../../etc/passwd")
        self.assertEqual(response.status_code, 404)

    def test_create_report_invalid_json(self):
        """Confirm POST /report returns 400 for non-JSON body."""
        response = self.client.post("/report", data="not json")
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertEqual(data["error"], "Request body must contain JSON.")

    def test_create_report_empty_body(self):
        """Confirm POST /report returns 400 when neither file_id nor prediction is supplied."""
        response = self.client.post("/report", json={})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertEqual(data["error"], "Either 'file_id' or 'prediction' is required.")

    def test_create_report_unknown_file_id(self):
        """Confirm POST /report returns 404 when file_id does not exist."""
        response = self.client.post("/report", json={"file_id": "non_existent_scan_id_999"})
        self.assertEqual(response.status_code, 404)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertEqual(data["error"], "Uploaded MRI file not found.")

    def test_create_report_invalid_prediction_type(self):
        """Confirm POST /report returns 400 when prediction is not an object."""
        response = self.client.post("/report", json={"prediction": "invalid_string"})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertEqual(data["error"], "'prediction' must be a JSON object.")

    def test_mock_result_labelling_in_service(self):
        """Verify report_service explicitly labels mock analysis and handles future formats."""
        # 1. Mock payload
        mock_result = {
            "file_id": "mock_id",
            "filename": "scan.nii.gz",
            "mode": "mock",
            "prediction": {
                "tumor_present": True,
                "tumor_type": "Meningioma",
                "confidence": 0.94,
                "mode": "mock"
            }
        }
        res_mock = generate_report(mock_result)
        self.assertEqual(res_mock["mode"], "mock")
        self.created_reports.append(res_mock["report_id"])

        pdf_path = get_report_file_path(res_mock["report_id"])
        self.assertIsNotNone(pdf_path)
        self.assertTrue(pdf_path.is_file())

        # 2. Future real inference format with segmentation and quantitative metrics
        real_result = {
            "file_id": "real_id",
            "filename": "real_scan.nii.gz",
            "mode": "production",
            "prediction": {
                "tumor_present": True,
                "tumor_type": "Glioma",
                "confidence": 0.98,
                "mode": "production"
            },
            "segmentation": {
                "status": "Completed",
                "mask_filename": "mask_real_scan.nii.gz",
                "sub_regions": ["enhancing_core", "edema", "necrosis"]
            },
            "quantitative_analysis": {
                "total_volume_mm3": 24150.5,
                "enhancing_volume_mm3": 11200.2,
                "edema_volume_mm3": 9500.3,
                "centroid_voxel": [120, 115, 78]
            },
            "limitations": [
                "Evaluated using 3D U-Net trained on BraTS-2023 dataset."
            ]
        }
        res_real = generate_report(real_result)
        self.assertEqual(res_real["mode"], "production")
        self.created_reports.append(res_real["report_id"])

        real_pdf_path = get_report_file_path(res_real["report_id"])
        self.assertIsNotNone(real_pdf_path)
        self.assertTrue(real_pdf_path.is_file())


if __name__ == "__main__":
    unittest.main()
