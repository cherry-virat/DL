"""
Export Utilities for 3D Brain Tumor MRI System.
===============================================
Generates user-friendly, publication-grade export artifacts:
  - 2D Segmentation View (PNG): Max-tumor cross-section axial slice with color overlay & annotations
  - Volume Summary (CSV): Tabular volumetric measurements with units and metadata
"""

import io
import csv
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import nibabel as nib
from PIL import Image, ImageDraw


def generate_csv_report(res_data: Dict[str, Any], request_id: str, classification_data: Optional[Dict[str, Any]] = None) -> str:
    """
    Generates a tidy, readable CSV string of volumetric and classification results.
    """
    output = io.StringIO()
    writer = csv.writer(output)

    req_meta = res_data.get("request_metadata") or res_data.get("request", {})
    inp_meta = res_data.get("input_metadata", {})
    vol_summary = res_data.get("volumetrics", res_data.get("volumetric_summary", {}))
    gl_meas = res_data.get("glioma", res_data.get("glioma_measurements"))
    men_meas = res_data.get("meningioma_multimodal", res_data.get("meningioma", res_data.get("meningioma_measurements")))
    qc = res_data.get("quality_status", {})
    res_status = res_data.get("status")

    tumor_type = str(req_meta.get("tumor_type", "Unknown")).upper()
    routing_mode = req_meta.get("routing_mode", "manual_selection")
    selected_model = req_meta.get("selected_model", tumor_type)
    conflict = bool(req_meta.get("manual_override", False) and req_meta.get("conflict_detected", False))
    timestamp = req_meta.get("timestamp") or res_data.get("timestamp", "")
    pipeline_status = str(qc.get("status") or res_data.get("status") or "SUCCESS").upper()

    shape = inp_meta.get("original_shape", [240, 240, 155])
    spacing = inp_meta.get("original_spacing", [1.0, 1.0, 1.0])
    voxel_vol = inp_meta.get("voxel_volume_mm3", 1.0)

    # 1. Overview Table Section
    writer.writerow(["# 3D BRAIN MRI QUANTITATIVE EVALUATION SUMMARY"])
    writer.writerow(["Metric", "Value", "Unit", "Description"])
    writer.writerow(["Request_ID", request_id, "-", "Unique case run identifier"])
    writer.writerow(["Timestamp", timestamp, "ISO_8601", "Execution timestamp"])
    writer.writerow(["Target_Pathology", tumor_type, "-", "Pathology category evaluated"])
    writer.writerow(["Selected_Model", f"{selected_model.upper()} 3D U-Net", "-", "Inference model executed"])
    writer.writerow(["Routing_Mode", routing_mode, "-", "Workflow routing mode"])
    writer.writerow(["Manual_Override_Conflict", str(conflict), "boolean", "Classifier disagreement with manual route"])
    writer.writerow(["Pipeline_Status", pipeline_status, "-", "Overall pipeline completion status"])
    writer.writerow(["Matrix_Dimensions", f"{shape[0]}x{shape[1]}x{shape[2]}", "voxels", "3D acquisition matrix"])
    writer.writerow(["Voxel_Spacing", f"{spacing[0]:.2f}x{spacing[1]:.2f}x{spacing[2]:.2f}", "mm", "Physical voxel resolution"])
    writer.writerow(["Unit_Voxel_Volume", f"{voxel_vol:.4f}", "mm3", "Physical volume of one voxel"])

    # Classification telemetry if present
    cls = classification_data or req_meta.get("classifier_assessment") or {}
    if cls:
        if "presence_probability" in cls:
            writer.writerow(["Stage1_Tumor_Presence_Prob", f"{cls['presence_probability']:.4f}", "probability", "Stage 1 tumor vs healthy probability"])
        if "predicted_tumor_type" in cls:
            writer.writerow(["Stage2_Predicted_Type", str(cls["predicted_tumor_type"]).upper(), "-", "Stage 2 classifier predicted type"])
        if "confidence" in cls or "type_probability" in cls:
            conf = cls.get("confidence") or cls.get("type_probability", 0.0)
            writer.writerow(["Stage2_Classification_Confidence", f"{conf:.4f}", "probability", "Stage 2 prediction confidence"])

    writer.writerow([])

    # 2. Subregion Volumetrics Section
    writer.writerow(["# SUBREGION VOLUMETRIC BREAKDOWN"])
    writer.writerow(["Subregion_Name", "Voxel_Count", "Volume_mm3", "Volume_cm3", "Percent_Of_Whole_Tumor"])

    tot_mm3 = vol_summary.get("total_tumor_volume_mm3", 0.0)
    tot_vox = vol_summary.get("total_tumor_voxels", int(tot_mm3 / voxel_vol if voxel_vol else 0))

    if res_status in ("no_tumor_detected", "uncertain_classification"):
        writer.writerow(["Whole_Tumor_WT", 0, "0.0", "0.00", "0.0%"])
    elif tumor_type == "GLIOMA" and gl_meas:
        sub = gl_meas.get("subregions") or gl_meas.get("subregion_volumes") or {}
        wt_mm3 = sub.get("wt_volume_mm3", tot_mm3)
        tc_mm3 = sub.get("tc_volume_mm3", 0.0)
        et_mm3 = sub.get("et_volume_mm3", 0.0)

        wt_vox = int(wt_mm3 / voxel_vol) if voxel_vol else tot_vox
        tc_vox = int(tc_mm3 / voxel_vol) if voxel_vol else 0
        et_vox = int(et_mm3 / voxel_vol) if voxel_vol else 0

        writer.writerow(["Whole_Tumor_WT", wt_vox, f"{wt_mm3:.1f}", f"{wt_mm3/1000.0:.2f}", "100.0%"])
        writer.writerow(["Tumor_Core_TC", tc_vox, f"{tc_mm3:.1f}", f"{tc_mm3/1000.0:.2f}", f"{(tc_mm3/wt_mm3*100.0):.1f}%" if wt_mm3 else "0.0%"])
        writer.writerow(["Enhancing_Tumor_ET", et_vox, f"{et_mm3:.1f}", f"{et_mm3/1000.0:.2f}", f"{(et_mm3/wt_mm3*100.0):.1f}%" if wt_mm3 else "0.0%"])
    elif men_meas:
        sub = men_meas.get("subregions") or men_meas.get("subregion_volumes") or {}
        wt_mm3 = sub.get("wt_volume_mm3", tot_mm3)
        et_mm3 = sub.get("et_volume_mm3", 0.0)
        ncr_mm3 = sub.get("ncr_net_volume_mm3") or sub.get("ncr_volume_mm3", 0.0)
        ed_mm3 = sub.get("edema_volume_mm3") or sub.get("ed_volume_mm3", 0.0)

        wt_vox = sub.get("wt_voxels", int(wt_mm3 / voxel_vol) if voxel_vol else tot_vox)
        et_vox = sub.get("et_voxels", int(et_mm3 / voxel_vol) if voxel_vol else 0)
        ncr_vox = sub.get("ncr_net_voxels") or sub.get("ncr_voxels", int(ncr_mm3 / voxel_vol) if voxel_vol else 0)
        ed_vox = sub.get("edema_voxels") or sub.get("ed_voxels", int(ed_mm3 / voxel_vol) if voxel_vol else 0)

        writer.writerow(["Whole_Tumor_WT", wt_vox, f"{wt_mm3:.1f}", f"{wt_mm3/1000.0:.2f}", "100.0%"])
        writer.writerow(["Enhancing_Tumor_ET", et_vox, f"{et_mm3:.1f}", f"{et_mm3/1000.0:.2f}", f"{(et_mm3/wt_mm3*100.0):.1f}%" if wt_mm3 else "0.0%"])
        writer.writerow(["Non_Enhancing_Core_NCR", ncr_vox, f"{ncr_mm3:.1f}", f"{ncr_mm3/1000.0:.2f}", f"{(ncr_mm3/wt_mm3*100.0):.1f}%" if wt_mm3 else "0.0%"])
        writer.writerow(["Peritumoral_Edema_ED", ed_vox, f"{ed_mm3:.1f}", f"{ed_mm3/1000.0:.2f}", f"{(ed_mm3/wt_mm3*100.0):.1f}%" if wt_mm3 else "0.0%"])
    else:
        writer.writerow(["Total_Tumor_Mass", tot_vox, f"{tot_mm3:.1f}", f"{tot_mm3/1000.0:.2f}", "100.0%"])

    writer.writerow([])
    writer.writerow(["# REGULATORY NOTICE"])
    writer.writerow(["Disclaimer", "Investigational Research Prototype Only. Not evaluated or approved by FDA or CE. Not for primary diagnosis or clinical treatment planning.", "-", "-"])

    return output.getvalue()


def generate_segmentation_png(res_data: Dict[str, Any], request_id: str, api_runs_dir: Path) -> bytes:
    """
    Renders an axial slice 2D segmentation overlay PNG from the actual NIfTI volumes.
    Identifies the slice with maximal tumor cross-sectional area.
    """
    req_dir = api_runs_dir / request_id
    upload_dir = req_dir / "uploads"
    mask_dir = req_dir / "masks"

    # 1. Locate background MRI scan (prefer T1c, then FLAIR, T1, T2)
    mri_path = None
    for pref in ["*-t1c.nii.gz", "*-t2f.nii.gz", "*-t1n.nii.gz", "*-t2w.nii.gz", "*.nii.gz", "*.nii"]:
        matches = list(upload_dir.glob(pref))
        if matches:
            mri_path = matches[0]
            break

    # 2. Locate segmentation mask
    mask_path = None
    for pref in ["*_seg.nii.gz", "*_gtv.nii.gz", "*.nii.gz", "*.nii"]:
        matches = list(mask_dir.glob(pref))
        if matches:
            mask_path = matches[0]
            break

    if not mri_path or not mri_path.exists():
        # Fallback empty placeholder image if MRI not available
        canvas = Image.new("RGB", (480, 520), color=(15, 23, 42))
        draw = ImageDraw.Draw(canvas)
        draw.text((20, 20), f"Case: {request_id}", fill=(248, 250, 252))
        draw.text((20, 50), "Notice: MRI volume data not found on disk.", fill=(248, 113, 113))
        buf = io.BytesIO()
        canvas.save(buf, format="PNG")
        return buf.getvalue()

    mri_nii = nib.load(str(mri_path))
    mri_vol = mri_nii.get_fdata()

    if mask_path and mask_path.exists():
        mask_nii = nib.load(str(mask_path))
        mask_vol = mask_nii.get_fdata()
    else:
        mask_vol = np.zeros_like(mri_vol, dtype=np.uint8)

    num_slices = mri_vol.shape[2]
    # Find slice with maximum tumor burden
    if np.any(mask_vol > 0):
        slice_counts = [int(np.sum(mask_vol[:, :, z] > 0)) for z in range(num_slices)]
        best_z = int(np.argmax(slice_counts))
        slice_voxels = slice_counts[best_z]
    else:
        best_z = num_slices // 2
        slice_voxels = 0

    mri_slice = mri_vol[:, :, best_z]
    mask_slice = mask_vol[:, :, best_z] if best_z < mask_vol.shape[2] else np.zeros_like(mri_slice)

    # Standard radiological rotation (axial: anterior at top, right on left)
    mri_slice = np.rot90(mri_slice)
    mask_slice = np.rot90(mask_slice)

    # Normalize MRI intensity with percentile clipping for clear contrast
    valid_voxels = mri_slice[mri_slice > 0]
    if len(valid_voxels) > 0:
        p1, p99 = np.percentile(valid_voxels, 1), np.percentile(valid_voxels, 99)
        if p99 > p1:
            norm_mri = np.clip((mri_slice - p1) / (p99 - p1) * 255.0, 0, 255).astype(np.uint8)
        else:
            norm_mri = np.zeros_like(mri_slice, dtype=np.uint8)
    else:
        norm_mri = np.zeros_like(mri_slice, dtype=np.uint8)

    rgb_img = np.stack([norm_mri, norm_mri, norm_mri], axis=-1)

    # Alpha overlay: 1=NCR (Red), 2=Edema (Yellow), 3=Enhancing (Green)
    alpha = 0.55
    m1 = (mask_slice == 1)
    if np.any(m1):
        rgb_img[m1, 0] = (rgb_img[m1, 0] * (1 - alpha) + 239 * alpha).astype(np.uint8)
        rgb_img[m1, 1] = (rgb_img[m1, 1] * (1 - alpha) + 68 * alpha).astype(np.uint8)
        rgb_img[m1, 2] = (rgb_img[m1, 2] * (1 - alpha) + 68 * alpha).astype(np.uint8)

    m2 = (mask_slice == 2)
    if np.any(m2):
        rgb_img[m2, 0] = (rgb_img[m2, 0] * (1 - alpha) + 234 * alpha).astype(np.uint8)
        rgb_img[m2, 1] = (rgb_img[m2, 1] * (1 - alpha) + 179 * alpha).astype(np.uint8)
        rgb_img[m2, 2] = (rgb_img[m2, 2] * (1 - alpha) + 8 * alpha).astype(np.uint8)

    m3 = (mask_slice == 3)
    if np.any(m3):
        rgb_img[m3, 0] = (rgb_img[m3, 0] * (1 - alpha) + 34 * alpha).astype(np.uint8)
        rgb_img[m3, 1] = (rgb_img[m3, 1] * (1 - alpha) + 197 * alpha).astype(np.uint8)
        rgb_img[m3, 2] = (rgb_img[m3, 2] * (1 - alpha) + 94 * alpha).astype(np.uint8)

    # Resize slice canvas to crisp viewing resolution (480x480)
    slice_img = Image.fromarray(rgb_img).resize((480, 480), Image.Resampling.NEAREST)

    # Canvas with informative medical header & footer banners
    header_h = 56
    footer_h = 48
    canvas = Image.new("RGB", (480, 480 + header_h + footer_h), color=(15, 23, 42))
    canvas.paste(slice_img, (0, header_h))

    draw = ImageDraw.Draw(canvas)

    # Header text
    req_meta = res_data.get("request_metadata") or res_data.get("request", {})
    tumor_type = str(req_meta.get("tumor_type", "Tumor")).upper()
    draw.text((14, 10), f"Case: {request_id}  |  Pathology: {tumor_type}", fill=(248, 250, 252))
    draw.text((14, 28), f"Orientation: Axial Plane  |  Slice Z={best_z}/{num_slices} ({slice_voxels:,} tumor voxels)", fill=(56, 189, 248))

    # Divider lines
    draw.line([(0, header_h - 1), (480, header_h - 1)], fill=(51, 65, 85), width=1)
    draw.line([(0, 480 + header_h), (480, 480 + header_h)], fill=(51, 65, 85), width=1)

    # Footer legend & disclaimer
    draw.text((14, 480 + header_h + 8), "Legend:  Red: NCR (1)  |  Yellow: Edema (2)  |  Green: Enhancing (3)", fill=(226, 232, 240))
    draw.text((14, 480 + header_h + 26), "2D Cross-Section View  •  Investigational Research  •  Not for Diagnosis", fill=(148, 163, 184))

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()
