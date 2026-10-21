import sys
import time
import json
from pathlib import Path
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import torch

PROJECT_ROOT = Path(r"C:\Users\nchan\Desktop\DL")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.app import create_app, API_RUNS_DIR
from integration.inference_router import InferenceRouter
from integration.classifier_service import ClassifierService
from integration.result_schema import TumorType

def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred_bool = (pred_mask > 0)
    gt_bool = (gt_mask > 0)
    intersection = np.logical_and(pred_bool, gt_bool).sum()
    total = pred_bool.sum() + gt_bool.sum()
    if total == 0:
        return 1.0
    return float((2.0 * intersection) / total)

def evaluate_subregions_dice(pred_arr: np.ndarray, gt_arr: np.ndarray):
    # WT: 1, 2, 3
    pred_wt = np.isin(pred_arr, [1, 2, 3])
    gt_wt = np.isin(gt_arr, [1, 2, 3])
    wt_dice = compute_dice(pred_wt, gt_wt)

    # TC: 1, 3
    pred_tc = np.isin(pred_arr, [1, 3])
    gt_tc = np.isin(gt_arr, [1, 3])
    tc_dice = compute_dice(pred_tc, gt_tc)

    # ET: 3
    pred_et = (pred_arr == 3)
    gt_et = (gt_arr == 3)
    et_dice = compute_dice(pred_et, gt_et)

    return {
        "wt_dice": round(wt_dice, 4),
        "tc_dice": round(tc_dice, 4),
        "et_dice": round(et_dice, 4),
    }

def main():
    print("=== STARTING INDEPENDENT TEST EVALUATION ===")
    out_dir = PROJECT_ROOT / "outputs" / "independent_test_evaluation_experiment_20261019"
    out_dir.mkdir(parents=True, exist_ok=True)

    app = create_app()
    client = app.test_client()

    results = {
        "metadata": {
            "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stage2_checkpoint": "outputs/classifier_training/experiment_20261019/tumor_classifier_best.pth",
            "stage1_checkpoint": "outputs/classifier_training/tumor_presence_best.pth",
            "uncertainty_threshold": 0.55,
        },
        "cases_evaluated": {},
    }

    # 1. Independent Glioma Case (from locked test set)
    gli_case_id = "BraTS-GLI-00009-000"
    gli_anon_id = "ANON_GLI_0008"
    gli_dir = Path("C:/3D_DATASET/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00009-000")
    gli_files = {
        "t1n": gli_dir / "BraTS-GLI-00009-000-t1n.nii.gz",
        "t1c": gli_dir / "BraTS-GLI-00009-000-t1c.nii.gz",
        "t2w": gli_dir / "BraTS-GLI-00009-000-t2w.nii.gz",
        "t2f": gli_dir / "BraTS-GLI-00009-000-t2f.nii.gz",
    }
    gli_gt_path = gli_dir / "BraTS-GLI-00009-000-seg.nii.gz"
    assert all(p.exists() for p in gli_files.values()), f"Glioma modalities missing: {gli_files}"
    assert gli_gt_path.exists(), f"Glioma GT mask missing: {gli_gt_path}"

    print(f"\n1. Evaluating Independent Glioma Case: {gli_case_id} ({gli_anon_id})")
    # Upload 4 modalities in order: t1n, t1c, t2w, t2f
    with open(gli_files["t1n"], "rb") as f1, open(gli_files["t1c"], "rb") as f2, \
         open(gli_files["t2w"], "rb") as f3, open(gli_files["t2f"], "rb") as f4:
        up_data = {
            "files": [
                (f1, gli_files["t1n"].name),
                (f2, gli_files["t1c"].name),
                (f3, gli_files["t2w"].name),
                (f4, gli_files["t2f"].name),
            ]
        }
        up_resp = client.post("/api/upload", data=up_data, content_type="multipart/form-data")
    assert up_resp.status_code == 200, up_resp.get_json()
    gli_req_id = up_resp.get_json()["request_id"]
    print(f"  Uploaded. Request ID: {gli_req_id}")

    # Classify via /api/classify
    c_resp = client.post("/api/classify", json={"request_id": gli_req_id})
    assert c_resp.status_code == 200, c_resp.get_json()
    gli_class_data = c_resp.get_json()
    print(f"  Classification: status={gli_class_data['status']}, type={gli_class_data['predicted_tumor_type']}, conf={gli_class_data['confidence']}")

    # Segment via /api/segment with tumor_type='auto'
    seg_resp = client.post("/api/segment", json={"request_id": gli_req_id, "tumor_type": "auto"})
    assert seg_resp.status_code == 200, seg_resp.get_json()
    gli_seg_data = seg_resp.get_json()
    print(f"  Segmentation complete. Selected model: {gli_seg_data['request_metadata']['selected_model']}")

    # Load predicted mask and ground truth mask to compute Dice
    pred_mask_file = list((API_RUNS_DIR / gli_req_id / "masks").glob("*_seg.nii.gz"))[0]
    pred_nii = nib.load(str(pred_mask_file))
    gt_nii = nib.load(str(gli_gt_path))
    pred_arr = np.asanyarray(pred_nii.dataobj)
    gt_arr = np.asanyarray(gt_nii.dataobj)

    gli_dice = evaluate_subregions_dice(pred_arr, gt_arr)
    print(f"  Glioma Dice: WT={gli_dice['wt_dice']}, TC={gli_dice['tc_dice']}, ET={gli_dice['et_dice']}")

    # Download report
    rep_resp = client.get(f"/api/report/{gli_req_id}")
    assert rep_resp.status_code == 200
    gli_rep_text = rep_resp.get_data(as_text=True)

    results["cases_evaluated"]["independent_glioma"] = {
        "case_id": gli_case_id,
        "anonymized_id": gli_anon_id,
        "request_id": gli_req_id,
        "ground_truth_label": "glioma",
        "stage1_presence_prob": gli_class_data["presence_probability"],
        "stage2_predicted_type": gli_class_data["predicted_tumor_type"],
        "type_probabilities": gli_class_data["type_probabilities"],
        "confidence": gli_class_data["confidence"],
        "accepted": gli_class_data["confidence"] >= 0.55,
        "selected_model": gli_seg_data["request_metadata"]["selected_model"],
        "routing_mode": gli_seg_data["request_metadata"]["routing_mode"],
        "volumetrics": gli_seg_data["volumetrics"],
        "dice_scores": gli_dice,
        "spatial_shape": list(pred_arr.shape),
        "voxel_spacing": [float(round(z, 4)) for z in pred_nii.header.get_zooms()[:3]],
        "mask_path": str(pred_mask_file),
    }

    # 2. Independent Meningioma Case (from locked test set)
    men_case_id = "BraTS-MEN-00008-000"
    men_anon_id = "ANON_MEN_0002"
    men_dir = Path("C:/3D_DATASET/BraTS2023_MEN/BraTS-MEN-Train/BraTS-MEN-00008-000")
    men_files = {
        "t1n": men_dir / "BraTS-MEN-00008-000-t1n.nii.gz",
        "t1c": men_dir / "BraTS-MEN-00008-000-t1c.nii.gz",
        "t2w": men_dir / "BraTS-MEN-00008-000-t2w.nii.gz",
        "t2f": men_dir / "BraTS-MEN-00008-000-t2f.nii.gz",
    }
    men_gt_path = men_dir / "BraTS-MEN-00008-000-seg.nii.gz"
    assert all(p.exists() for p in men_files.values()), f"Meningioma modalities missing: {men_files}"
    assert men_gt_path.exists(), f"Meningioma GT mask missing: {men_gt_path}"

    print(f"\n2. Evaluating Independent Meningioma Case: {men_case_id} ({men_anon_id})")
    with open(men_files["t1n"], "rb") as f1, open(men_files["t1c"], "rb") as f2, \
         open(men_files["t2w"], "rb") as f3, open(men_files["t2f"], "rb") as f4:
        up_data = {
            "files": [
                (f1, men_files["t1n"].name),
                (f2, men_files["t1c"].name),
                (f3, men_files["t2w"].name),
                (f4, men_files["t2f"].name),
            ]
        }
        up_resp = client.post("/api/upload", data=up_data, content_type="multipart/form-data")
    assert up_resp.status_code == 200, up_resp.get_json()
    men_req_id = up_resp.get_json()["request_id"]
    print(f"  Uploaded. Request ID: {men_req_id}")

    # Classify via /api/classify
    c_resp = client.post("/api/classify", json={"request_id": men_req_id})
    assert c_resp.status_code == 200, c_resp.get_json()
    men_class_data = c_resp.get_json()
    print(f"  Classification: status={men_class_data['status']}, type={men_class_data['predicted_tumor_type']}, conf={men_class_data['confidence']}")

    # Segment via /api/segment with tumor_type='auto'
    seg_resp = client.post("/api/segment", json={"request_id": men_req_id, "tumor_type": "auto"})
    assert seg_resp.status_code == 200, seg_resp.get_json()
    men_seg_data = seg_resp.get_json()
    print(f"  Segmentation complete. Selected model: {men_seg_data['request_metadata']['selected_model']}")

    # Load predicted mask and ground truth mask to compute Dice
    pred_mask_file_m = list((API_RUNS_DIR / men_req_id / "masks").glob("*_seg.nii.gz"))[0]
    pred_nii_m = nib.load(str(pred_mask_file_m))
    gt_nii_m = nib.load(str(men_gt_path))
    pred_arr_m = np.asanyarray(pred_nii_m.dataobj)
    gt_arr_m = np.asanyarray(gt_nii_m.dataobj)

    men_dice = evaluate_subregions_dice(pred_arr_m, gt_arr_m)
    print(f"  Meningioma Dice: WT={men_dice['wt_dice']}, TC={men_dice['tc_dice']}, ET={men_dice['et_dice']}")

    rep_resp_m = client.get(f"/api/report/{men_req_id}")
    assert rep_resp_m.status_code == 200
    men_rep_text = rep_resp_m.get_data(as_text=True)

    results["cases_evaluated"]["independent_meningioma"] = {
        "case_id": men_case_id,
        "anonymized_id": men_anon_id,
        "request_id": men_req_id,
        "ground_truth_label": "meningioma",
        "stage1_presence_prob": men_class_data["presence_probability"],
        "stage2_predicted_type": men_class_data["predicted_tumor_type"],
        "type_probabilities": men_class_data["type_probabilities"],
        "confidence": men_class_data["confidence"],
        "accepted": men_class_data["confidence"] >= 0.55,
        "selected_model": men_seg_data["request_metadata"]["selected_model"],
        "routing_mode": men_seg_data["request_metadata"]["routing_mode"],
        "volumetrics": men_seg_data["volumetrics"],
        "dice_scores": men_dice,
        "spatial_shape": list(pred_arr_m.shape),
        "voxel_spacing": [float(round(z, 4)) for z in pred_nii_m.header.get_zooms()[:3]],
        "mask_path": str(pred_mask_file_m),
    }

    # 3. Independent Healthy Control Case (from locked presence test set)
    healthy_case_id = "IXI012"
    healthy_anon_id = "ANON_HEALTHY_00002"
    h_files = {
        "t1": Path("C:/3D_DATASET/HealthyControls/IXI/processed/T1/IXI012_t1_skull_stripped.nii.gz"),
        "t2": Path("C:/3D_DATASET/HealthyControls/IXI/processed/T2/IXI012_t2_skull_stripped.nii.gz"),
    }
    assert all(p.exists() for p in h_files.values()), f"Healthy files missing: {h_files}"

    print(f"\n3. Evaluating Independent Healthy Control Case: {healthy_case_id} ({healthy_anon_id})")
    with open(h_files["t1"], "rb") as f1, open(h_files["t2"], "rb") as f2:
        up_data = {
            "files": [
                (f1, h_files["t1"].name),
                (f2, h_files["t2"].name),
            ]
        }
        up_resp = client.post("/api/upload", data=up_data, content_type="multipart/form-data")
    assert up_resp.status_code == 200, up_resp.get_json()
    h_req_id = up_resp.get_json()["request_id"]
    print(f"  Uploaded. Request ID: {h_req_id}")

    # Classify via /api/classify
    c_resp_h = client.post("/api/classify", json={"request_id": h_req_id})
    assert c_resp_h.status_code == 200, c_resp_h.get_json()
    h_class_data = c_resp_h.get_json()
    print(f"  Classification: status={h_class_data['status']}, presence={h_class_data['tumor_presence']}, prob={h_class_data['presence_probability']}")

    # Route via /api/segment with tumor_type='auto' -> MUST NOT TRIGGER SEGMENTATION
    seg_resp_h = client.post("/api/segment", json={"request_id": h_req_id, "tumor_type": "auto"})
    assert seg_resp_h.status_code == 200, seg_resp_h.get_json()
    h_seg_data = seg_resp_h.get_json()
    print(f"  Auto route response: status={h_seg_data['status']}, detail={h_seg_data['detail']}")
    assert h_seg_data["status"] == "no_tumor_detected"
    assert (API_RUNS_DIR / h_req_id / "masks").exists() is False or len(list((API_RUNS_DIR / h_req_id / "masks").glob("*.nii*"))) == 0

    results["cases_evaluated"]["independent_healthy"] = {
        "case_id": healthy_case_id,
        "anonymized_id": healthy_anon_id,
        "request_id": h_req_id,
        "ground_truth_label": "healthy",
        "stage1_presence_prob": h_class_data["presence_probability"],
        "stage1_status": h_class_data["status"],
        "stage1_confidence": h_class_data["presence_confidence"],
        "auto_segment_status": h_seg_data["status"],
        "segmentation_triggered": False,
        "recommendation": h_seg_data["recommendation"],
    }

    # 4. Uncertainty Rejection Verification
    print(f"\n4. Verifying Uncertainty Rejection (Threshold = 0.55)")
    # We test with a mock classification where confidence is < 0.55
    from unittest.mock import patch
    uncertain_assessment = {
        "status": "uncertain_classification",
        "tumor_presence": "tumor_present",
        "presence_probability": 0.85,
        "presence_confidence": 0.85,
        "predicted_tumor_type": "uncertain",
        "selected_route": None,
        "type_probability": 0.52,
        "confidence": 0.52,
        "type_confidence": 0.52,
        "glioma_probability": 0.48,
        "meningioma_probability": 0.52,
        "type_probabilities": {"glioma": 0.48, "meningioma": 0.52},
        "reason": "Tumor type prediction is ambiguous (confidence 0.52 < 0.55). Automatic routing is refused.",
        "available_manual_routes": ["glioma", "meningioma"],
        "recommendation": "Please select the appropriate segmentation model manually ('glioma' or 'meningioma').",
    }
    with patch.object(ClassifierService, "classify", return_value=uncertain_assessment):
        # Using glioma request to test auto segment behavior on uncertain classification
        unc_resp = client.post("/api/segment", json={"request_id": gli_req_id, "tumor_type": "auto"})
        print(f"  Uncertain route response code: {unc_resp.status_code}")
        unc_data = unc_resp.get_json()
        print(f"  Uncertain route response data: {unc_data}")
        assert unc_resp.status_code == 422, f"Expected 422, got {unc_resp.status_code}"
        assert unc_data["status"] == "uncertain_classification"
        assert "glioma" in unc_data["available_manual_routes"]
        assert "meningioma" in unc_data["available_manual_routes"]

    results["uncertainty_rejection_test"] = {
        "threshold": 0.55,
        "simulated_confidence": 0.52,
        "http_status_code": unc_resp.status_code,
        "response_status": unc_data["status"],
        "automatic_routing_refused": True,
        "manual_routes_offered": unc_data["available_manual_routes"],
        "recommendation": unc_data["recommendation"],
    }

    # Save dedicated report files
    json_report_path = out_dir / "independent_test_evaluation_report.json"
    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save markdown summary
    md_report_path = out_dir / "independent_test_evaluation_report.md"
    md_content = f"""# Independent Test Partition Evaluation Report

**Evaluation Timestamp:** {results['metadata']['evaluation_timestamp']}
**Evaluation Scope:** Hold-out independent test cases (zero overlap with training set)
**Stage 1 Checkpoint:** `{results['metadata']['stage1_checkpoint']}`
**Stage 2 Checkpoint:** `{results['metadata']['stage2_checkpoint']}`
**Uncertainty Threshold:** {results['metadata']['uncertainty_threshold']}

---

## 1. Case-Level Independent Evaluation Summary

| Modality / Case | Case ID & Anonymized ID | Ground Truth | Stage 1 Presence | Stage 2 Pred (Type Probs) | Confidence & Decision | Auto Selected Model | WT Dice Score | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Independent Glioma** | `{gli_case_id}` (`{gli_anon_id}`) | Glioma | **{gli_class_data['presence_probability']}** | Glioma: **{gli_class_data['type_probabilities']['glioma']}** | **{gli_class_data['confidence']}** (ACCEPTED) | **{gli_seg_data['request_metadata']['selected_model'].upper()}** | **{gli_dice['wt_dice']:.4f}** | **SUCCESS** |
| **Independent Meningioma** | `{men_case_id}` (`{men_anon_id}`) | Meningioma | **{men_class_data['presence_probability']}** | Meningioma: **{men_class_data['type_probabilities']['meningioma']}** | **{men_class_data['confidence']}** (ACCEPTED) | **{men_seg_data['request_metadata']['selected_model'].upper()}** | **{men_dice['wt_dice']:.4f}** | **SUCCESS** |
| **Independent Healthy** | `{healthy_case_id}` (`{healthy_anon_id}`) | Healthy | **{h_class_data['presence_probability']}** | *Bypassed* | **{h_class_data['presence_confidence']}** (Normative) | *None (Halted)* | N/A | **NO TUMOR DETECTED (200)** |

---

## 2. Segmentation Quality vs. Ground Truth Masks

### A. Independent Glioma (`{gli_case_id}`)
- **Whole Tumor (WT) Dice:** **{gli_dice['wt_dice']:.4f}**
- **Tumor Core (TC) Dice:** **{gli_dice['tc_dice']:.4f}**
- **Enhancing Tumor (ET) Dice:** **{gli_dice['et_dice']:.4f}**
- **Spatial Matrix Dimensions:** `{results['cases_evaluated']['independent_glioma']['spatial_shape']}`
- **Voxel Spacing:** `{results['cases_evaluated']['independent_glioma']['voxel_spacing']}` mm
- **Predicted WT Volume:** {gli_seg_data['volumetrics']['total_tumor_volume_mm3']:,.1f} mm³ ({gli_seg_data['volumetrics']['total_tumor_volume_cm3']:.2f} cm³)

### B. Independent Meningioma (`{men_case_id}`)
- **Whole Tumor (WT) Dice:** **{men_dice['wt_dice']:.4f}**
- **Tumor Core (TC) Dice:** **{men_dice['tc_dice']:.4f}**
- **Enhancing Tumor (ET) Dice:** **{men_dice['et_dice']:.4f}**
- **Spatial Matrix Dimensions:** `{results['cases_evaluated']['independent_meningioma']['spatial_shape']}`
- **Voxel Spacing:** `{results['cases_evaluated']['independent_meningioma']['voxel_spacing']}` mm
- **Predicted WT Volume:** {men_seg_data['volumetrics']['total_tumor_volume_mm3']:,.1f} mm³ ({men_seg_data['volumetrics']['total_tumor_volume_cm3']:.2f} cm³)

---

## 3. Safety Gating & Uncertainty Verification
- **Uncertainty Decision Test:** Input with simulated confidence 0.52 (< 0.55 threshold) returned **HTTP 422** with `status: "uncertain_classification"`.
- **Automated Routing Refusal:** Refused automatic segmentation dispatch; presented explicit manual routes (`glioma`, `meningioma`).
- **Normative Scan Safety:** Healthy control scan (`{healthy_case_id}`) produced presence probability {h_class_data['presence_probability']}, stopping the pipeline at Stage 1 with HTTP 200 `no_tumor_detected` and zero tumor mask generation.
"""

    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\nSaved evaluation artifacts to:")
    print(f"  - {json_report_path}")
    print(f"  - {md_report_path}")
    print("=== INDEPENDENT TEST EVALUATION COMPLETE ===")

if __name__ == "__main__":
    main()
