"""
Real-Data End-to-End Smoke Test for 3D Brain Tumor MRI Prototype.
=================================================================

Executes the real Flask REST API client against genuine BraTS MRI scans
on the local NVIDIA RTX 3050 GPU, verifying:
1. System health and classifier availability reporting.
2. Real Glioma case upload, geometry check, honest auto-route 422, manual segmentation,
   volumetric quantification, mask downloads, and clinical report.
3. Real Meningioma case upload, geometry check, manual segmentation,
   volumetric quantification, mask downloads, and clinical report.
4. Unsupported / ambiguous input handling and error responses.
"""

import sys
import time
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import SimpleITK as sitk
import torch

from api.app import create_app, API_RUNS_DIR


def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def main():
    print_banner("INITIALIZING REAL-DATA SMOKE TEST")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    app = create_app()
    client = app.test_client()

    # 1. Health and classifier status
    print_banner("1. CHECKING HEALTH & CLASSIFIER STATUS ENDPOINTS")
    health_resp = client.get("/api/health")
    assert health_resp.status_code == 200, f"Health check failed: {health_resp.status_code}"
    health_data = health_resp.get_json()
    print("Health response:", json.dumps(health_data, indent=2))
    assert "auto" in health_data["supported_tumor_types"]
    assert health_data["automatic_classification_available"] is False

    clf_resp = client.get("/api/classifier/status")
    assert clf_resp.status_code == 200
    clf_data = clf_resp.get_json()
    print("Classifier status response:", json.dumps(clf_data, indent=2))
    assert clf_data["automatic_classification_available"] is False

    # 2. Real Glioma case
    print_banner("2. EXECUTING REAL GLIOMA CASE (BraTS-GLI-00000-000)")
    gli_dir = Path("C:/3D_DATASET/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00000-000")
    gli_files = {
        "t1": gli_dir / "BraTS-GLI-00000-000-t1n.nii.gz",
        "t1c": gli_dir / "BraTS-GLI-00000-000-t1c.nii.gz",
        "t2": gli_dir / "BraTS-GLI-00000-000-t2w.nii.gz",
        "flair": gli_dir / "BraTS-GLI-00000-000-t2f.nii.gz",
    }
    for k, p in gli_files.items():
        assert p.exists(), f"Glioma file {k} not found at {p}"
        print(f"Found {k}: {p.name} ({p.stat().st_size / (1024*1024):.2f} MB)")

    print("\nUploading 4 modalities to /api/upload...")
    t0_up = time.perf_counter()
    with open(gli_files["t1"], "rb") as f_t1, \
         open(gli_files["t1c"], "rb") as f_t1c, \
         open(gli_files["t2"], "rb") as f_t2, \
         open(gli_files["flair"], "rb") as f_flair:
        upload_data = {
            "files": [
                (f_t1, gli_files["t1"].name),
                (f_t1c, gli_files["t1c"].name),
                (f_t2, gli_files["t2"].name),
                (f_flair, gli_files["flair"].name),
            ]
        }
        up_resp = client.post("/api/upload", data=upload_data, content_type="multipart/form-data")
    dt_up = time.perf_counter() - t0_up
    assert up_resp.status_code == 200, f"Upload failed: {up_resp.get_json()}"
    up_res = up_resp.get_json()
    gli_req_id = up_res["request_id"]
    print(f"Upload complete in {dt_up:.2f}s! Assigned Request ID: {gli_req_id}")
    print(f"Detected modalities: {up_res['detected_modalities']}")

    # Check standalone /api/classify on this case
    print("\nTesting /api/classify on uploaded Glioma scan...")
    classify_resp = client.post("/api/classify", json={"request_id": gli_req_id})
    assert classify_resp.status_code == 200
    classify_data = classify_resp.get_json()
    print("Classify response:", json.dumps(classify_data, indent=2))
    assert classify_data["status"] == "automatic_classification_not_available"

    # Test auto routing on this case (should return 422 honest explanation)
    print("\nTesting /api/segment with tumor_type='auto'...")
    auto_resp = client.post("/api/segment", json={"request_id": gli_req_id, "tumor_type": "auto"})
    assert auto_resp.status_code == 422, f"Expected 422 for auto route, got {auto_resp.status_code}"
    auto_data = auto_resp.get_json()
    print("Auto-route 422 response:", json.dumps(auto_data, indent=2))
    assert auto_data["status"] == "automatic_classification_not_available"
    assert "Missing trained checkpoints" in auto_data["blocker"]

    # Now execute manual Glioma segmentation
    print("\nExecuting manual Glioma segmentation via /api/segment...")
    t0_seg = time.perf_counter()
    seg_resp = client.post("/api/segment", json={"request_id": gli_req_id, "tumor_type": "glioma"})
    dt_seg = time.perf_counter() - t0_seg
    assert seg_resp.status_code == 200, f"Glioma segmentation failed: {seg_resp.get_json()}"
    seg_data = seg_resp.get_json()
    print(f"Glioma segmentation inference completed in {dt_seg:.2f}s!")

    vol_info = seg_data.get("volumetrics", {})
    sub_info = seg_data.get("glioma", {}).get("subregions", {})
    print(f"Total Tumor Volume: {vol_info.get('total_tumor_volume_mm3', 0.0):,.1f} mm³ ({vol_info.get('total_tumor_volume_cm3', 0.0):.2f} cm³)")
    print(f"  - Whole Tumor (WT):    {sub_info.get('wt_volume_mm3', 0.0):,.1f} mm³ ({sub_info.get('wt_volume_cm3', 0.0):.2f} cm³)")
    print(f"  - Tumor Core (TC):     {sub_info.get('tc_volume_mm3', 0.0):,.1f} mm³ ({sub_info.get('tc_volume_cm3', 0.0):.2f} cm³)")
    print(f"  - Enhancing Tumor (ET): {sub_info.get('et_volume_mm3', 0.0):,.1f} mm³ ({sub_info.get('et_volume_cm3', 0.0):.2f} cm³)")

    # Download masks and clinical report
    print("\nTesting mask downloads and clinical report...")
    dl_multi = client.get(f"/api/mask/{gli_req_id}?type=multiclass")
    assert dl_multi.status_code == 200
    assert dl_multi.mimetype == "application/gzip"
    print(f"Multiclass mask download: {len(dl_multi.data)} bytes")

    dl_gtv = client.get(f"/api/mask/{gli_req_id}?type=gtv")
    assert dl_gtv.status_code == 200
    assert dl_gtv.mimetype == "application/gzip"
    print(f"Binary GTV mask download: {len(dl_gtv.data)} bytes")

    rep_resp = client.get(f"/api/report/{gli_req_id}")
    assert rep_resp.status_code == 200
    gli_report_text = rep_resp.get_data(as_text=True)
    assert "Whole Tumor (WT):         90,908.0 mm³" in gli_report_text
    assert "Tumor Core (TC):          49,044.0 mm³" in gli_report_text
    print("Clinical report preview:")
    print("-" * 50)
    for line in gli_report_text.splitlines()[:22]:
        print(" ", line)
    print("  ...")
    print("-" * 50)

    # 3. Real Meningioma case
    print_banner("3. EXECUTING REAL MENINGIOMA CASE (BraTS-MEN-00004-000)")
    men_dir = Path("C:/3D_DATASET/BraTS2023_MEN/BraTS-MEN-Train/BraTS-MEN-00004-000")
    men_files = {
        "t1n": men_dir / "BraTS-MEN-00004-000-t1n.nii.gz",
        "t1c": men_dir / "BraTS-MEN-00004-000-t1c.nii.gz",
        "t2w": men_dir / "BraTS-MEN-00004-000-t2w.nii.gz",
        "t2f": men_dir / "BraTS-MEN-00004-000-t2f.nii.gz",
    }
    for k, p in men_files.items():
        assert p.exists(), f"Meningioma file {k} not found at {p}"
        print(f"Found {k}: {p.name} ({p.stat().st_size / (1024*1024):.2f} MB)")

    print("\nUploading 4 modalities to /api/upload...")
    t0_up_m = time.perf_counter()
    with open(men_files["t1n"], "rb") as f_t1n, \
         open(men_files["t1c"], "rb") as f_t1c, \
         open(men_files["t2w"], "rb") as f_t2w, \
         open(men_files["t2f"], "rb") as f_t2f:
        upload_data_m = {
            "files": [
                (f_t1n, men_files["t1n"].name),
                (f_t1c, men_files["t1c"].name),
                (f_t2w, men_files["t2w"].name),
                (f_t2f, men_files["t2f"].name),
            ]
        }
        up_resp_m = client.post("/api/upload", data=upload_data_m, content_type="multipart/form-data")
    dt_up_m = time.perf_counter() - t0_up_m
    assert up_resp_m.status_code == 200, f"Meningioma upload failed: {up_resp_m.get_json()}"
    up_res_m = up_resp_m.get_json()
    men_req_id = up_res_m["request_id"]
    print(f"Upload complete in {dt_up_m:.2f}s! Assigned Request ID: {men_req_id}")

    print("\nExecuting manual Meningioma segmentation via /api/segment...")
    t0_seg_m = time.perf_counter()
    seg_resp_m = client.post("/api/segment", json={"request_id": men_req_id, "tumor_type": "meningioma"})
    dt_seg_m = time.perf_counter() - t0_seg_m
    assert seg_resp_m.status_code == 200, f"Meningioma segmentation failed: {seg_resp_m.get_json()}"
    seg_data_m = seg_resp_m.get_json()
    print(f"Meningioma segmentation inference completed in {dt_seg_m:.2f}s!")

    vol_m = seg_data_m.get("volumetrics", {})
    sub_m = seg_data_m.get("meningioma_multimodal", {}).get("subregions", {})
    print(f"Total Tumor Volume: {vol_m.get('total_tumor_volume_mm3', 0.0):,.1f} mm³ ({vol_m.get('total_tumor_volume_cm3', 0.0):.2f} cm³)")
    print(f"  - Whole Tumor (WT):         {sub_m.get('wt_volume_mm3', 0.0):,.1f} mm³ ({sub_m.get('wt_volume_cm3', 0.0):.2f} cm³)")
    print(f"  - Enhancing Tumor (ET):      {sub_m.get('et_volume_mm3', 0.0):,.1f} mm³ ({sub_m.get('et_volume_cm3', 0.0):.2f} cm³)")
    print(f"  - Non-Enhancing Core (NCR): {sub_m.get('ncr_volume_mm3', 0.0):,.1f} mm³ ({sub_m.get('ncr_volume_cm3', 0.0):.2f} cm³)")
    print(f"  - Edema (ED):                {sub_m.get('ed_volume_mm3', 0.0):,.1f} mm³ ({sub_m.get('ed_volume_cm3', 0.0):.2f} cm³)")

    # Download masks and report for Meningioma
    dl_multi_m = client.get(f"/api/mask/{men_req_id}?type=multiclass")
    assert dl_multi_m.status_code == 200
    dl_gtv_m = client.get(f"/api/mask/{men_req_id}?type=gtv")
    assert dl_gtv_m.status_code == 200
    rep_m = client.get(f"/api/report/{men_req_id}")
    assert rep_m.status_code == 200
    men_report_text = rep_m.get_data(as_text=True)
    assert "Whole Tumor (WT):         2,465.0 mm³" in men_report_text
    assert "Enhancing Tumor (ET):      2,025.0 mm³" in men_report_text
    print(f"Meningioma masks & report verified successfully.")

    # 4. Negative / Unsupported tests
    print_banner("4. TESTING UNSUPPORTED & INVALID INPUTS")
    # Geometry mismatch: combine Glioma T1 with Meningioma T2
    with open(gli_files["t1"], "rb") as f1, open(men_files["t2w"], "rb") as f2:
        mismatch_up = client.post("/api/upload", data={"files": [(f1, "mismatch_t1.nii.gz"), (f2, "mismatch_t2.nii.gz")]}, content_type="multipart/form-data")
    assert mismatch_up.status_code == 200
    mismatch_req = mismatch_up.get_json()["request_id"]
    mismatch_seg = client.post("/api/segment", json={"request_id": mismatch_req, "tumor_type": "glioma"})
    assert mismatch_seg.status_code == 400
    print("Mismatched modalities rejection:", mismatch_seg.get_json()["detail"])

    # Invalid tumor type
    inv_type = client.post("/api/segment", json={"request_id": gli_req_id, "tumor_type": "carcinoma"})
    assert inv_type.status_code == 400
    print("Invalid tumor type rejection:", inv_type.get_json()["detail"])

    # Non-existent request ID
    non_exist = client.get("/api/results/REQ-DOESNOTEXIST999")
    assert non_exist.status_code == 404
    print("Non-existent request rejection:", non_exist.get_json()["detail"])

    print_banner("ALL REAL-DATA PIPELINE TESTS COMPLETED SUCCESSFULLY!")
    print(f"Glioma Upload Time: {dt_up:.2f}s | Inference Time: {dt_seg:.2f}s")
    print(f"Meningioma Upload Time: {dt_up_m:.2f}s | Inference Time: {dt_seg_m:.2f}s")


if __name__ == "__main__":
    main()
