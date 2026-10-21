"""
HD-BET Pilot Execution & Multi-Case Validation Pipeline (Phase 3D-1).
=====================================================================

Processes representative scans:
1. Healthy Volunteer: IXI002-Guys-0828
2. Glioma Reference: BraTS-GLI-00000-000
3. Meningioma Reference: BraTS-MEN-00004-000

Steps per case:
- Native geometry & intensity profiling
- Rigid T2-to-T1 coregistration (SimpleITK physical shift scaling)
- HD-BET brain extraction on T1 using isolated .venv_hdbet environment
- Dual-channel masking (T1_stripped = T1 * mask, T2_stripped = T2_coreg * mask)
- Quantitative plausibility audit (intracranial volume, boundary leak, geometry preservation)
- Orthogonal-slice visual overlay generation (axial, sagittal, coronal)
- Complete failure logging & JSON report generation
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

PROJECT_ROOT = Path(r"C:\Users\nchan\Desktop\DL")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import SimpleITK as sitk

from preprocessing.tumor_presence_preprocess import (
    align_and_resample_t2_to_t1,
    validate_brain_mask_plausibility,
)

HDBET_EXE = PROJECT_ROOT / ".venv_hdbet" / "Scripts" / "hd-bet.exe"
PILOT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "pilot_hdbet"


def inspect_nifti_geometry(path: Path) -> Dict[str, Any]:
    """Extracts geometric metadata, orientation, and voxel intensity statistics."""
    nii = nib.load(str(path))
    hdr = nii.header
    arr = np.asanyarray(nii.dataobj, dtype=np.float32)
    zooms = tuple(float(z) for z in hdr.get_zooms()[:3])
    orient = tuple(nib.aff2axcodes(nii.affine))
    nonzeros = arr[arr > 0]

    return {
        "path": str(path),
        "shape": list(arr.shape),
        "zooms": list(zooms),
        "orientation": list(orient),
        "affine": nii.affine.tolist(),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean_nonzero": float(nonzeros.mean()) if len(nonzeros) > 0 else 0.0,
        "std_nonzero": float(nonzeros.std()) if len(nonzeros) > 0 else 0.0,
        "nonzero_voxels": int(len(nonzeros)),
        "voxel_volume_mm3": float(zooms[0] * zooms[1] * zooms[2]),
    }


def save_nifti_like(arr: np.ndarray, ref_nii_path: Path, out_path: Path, dtype=None):
    """Saves array using reference NIfTI coordinate system, origin, spacing, and header."""
    ref_nii = nib.load(str(ref_nii_path))
    if dtype is not None:
        arr = arr.astype(dtype)
    new_nii = nib.Nifti1Image(arr, ref_nii.affine, ref_nii.header)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(new_nii, str(out_path))


def run_hdbet_subprocess(
    input_path: Path,
    output_path: Path,
    device: str = "cuda",
    disable_tta: bool = True,
) -> Tuple[bool, float, str]:
    """
    Executes HD-BET via isolated .venv_hdbet CLI executable.
    Returns (success, elapsed_seconds, stderr_or_stdout).
    """
    if not HDBET_EXE.exists():
        raise FileNotFoundError(f"HD-BET executable not found at: {HDBET_EXE}")

    cmd = [
        str(HDBET_EXE),
        "-i", str(input_path),
        "-o", str(output_path),
        "-device", device,
        "--save_bet_mask",
    ]
    if disable_tta:
        cmd.append("--disable_tta")

    t0 = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    if res.returncode != 0:
        return False, elapsed, res.stderr + "\n" + res.stdout
    return True, elapsed, res.stdout


def generate_orthogonal_overlays(
    t1_arr: np.ndarray,
    t2_arr: np.ndarray,
    mask_arr: np.ndarray,
    case_id: str,
    out_png: Path,
):
    """
    Creates multi-panel orthogonal slice plots (axial, sagittal, coronal)
    with brain mask contours overlaid in red on T1 and T2.
    """
    # Find center of mass of the brain mask for optimal slice selection
    coords = np.argwhere(mask_arr > 0)
    if len(coords) > 0:
        cz, cy, cx = np.median(coords, axis=0).astype(int)
    else:
        cz, cy, cx = [s // 2 for s in mask_arr.shape]

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f"HD-BET Brain Extraction Pilot: {case_id}", fontsize=16, fontweight="bold")

    slices = [
        ("Axial", 0, cz, lambda a, idx: a[idx, :, :]),
        ("Coronal", 1, cy, lambda a, idx: a[:, idx, :]),
        ("Sagittal", 2, cx, lambda a, idx: a[:, :, idx]),
    ]

    for col_idx, (view_name, ax_idx, slice_num, slice_fn) in enumerate(slices):
        # Row 1: T1 with red mask contour
        t1_sl = slice_fn(t1_arr, slice_num)
        m_sl = slice_fn(mask_arr, slice_num)
        ax = axes[0, col_idx]
        ax.imshow(t1_sl, cmap="gray", origin="lower")
        if m_sl.max() > 0:
            ax.contour(m_sl, levels=[0.5], colors="red", linewidths=1.5)
        ax.set_title(f"T1 {view_name} (slice {slice_num})")
        ax.axis("off")

        # Row 2: Registered T2 with red mask contour
        t2_sl = slice_fn(t2_arr, slice_num)
        ax = axes[1, col_idx]
        ax.imshow(t2_sl, cmap="gray", origin="lower")
        if m_sl.max() > 0:
            ax.contour(m_sl, levels=[0.5], colors="red", linewidths=1.5)
        ax.set_title(f"Registered T2 {view_name} (slice {slice_num})")
        ax.axis("off")

        # Row 3: Skull-stripped T1 * mask
        t1_strip_sl = t1_sl * m_sl
        ax = axes[2, col_idx]
        ax.imshow(t1_strip_sl, cmap="gray", origin="lower")
        ax.set_title(f"Skull-Stripped T1 {view_name}")
        ax.axis("off")

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_png), dpi=150, bbox_inches="tight")
    plt.close()


def process_pilot_case(
    case_id: str,
    t1_path: Path,
    t2_path: Path,
    is_pre_skull_stripped: bool = False,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Runs the complete end-to-end pilot pipeline for a single subject."""
    case_out_dir = PILOT_OUTPUT_ROOT / case_id
    case_out_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {"case_id": case_id, "is_pre_skull_stripped": is_pre_skull_stripped}

    # 1. Native Geometry Profile
    summary["t1_native"] = inspect_nifti_geometry(t1_path)
    summary["t2_native"] = inspect_nifti_geometry(t2_path)

    # 2. Registration: T2 to T1 grid
    t0_reg = time.perf_counter()
    t1_nii = nib.load(str(t1_path))
    t2_nii = nib.load(str(t2_path))

    # Check if shapes and affines already match (like in BraTS)
    if is_pre_skull_stripped and t1_nii.shape == t2_nii.shape and np.allclose(t1_nii.affine, t2_nii.affine, atol=1e-3):
        t2_reg_arr = np.asanyarray(t2_nii.dataobj, dtype=np.float32)
        summary["registration_needed"] = False
        reg_elapsed = 0.0
    else:
        summary["registration_needed"] = True
        t2_reg_arr = align_and_resample_t2_to_t1(
            fixed_src=t1_path,
            moving_src=t2_path,
            fixed_affine=t1_nii.affine,
            moving_affine=t2_nii.affine,
        )
        reg_elapsed = time.perf_counter() - t0_reg

    summary["registration_time_seconds"] = round(reg_elapsed, 3)

    # Save registered T2
    t2_reg_path = case_out_dir / f"{case_id}_t2_registered.nii.gz"
    save_nifti_like(t2_reg_arr, t1_path, t2_reg_path, dtype=np.float32)
    summary["t2_registered_path"] = str(t2_reg_path)

    # 3. Brain Extraction on T1 via HD-BET
    t1_strip_target = case_out_dir / f"{case_id}_t1_stripped.nii.gz"
    expected_mask_path = case_out_dir / f"{case_id}_t1_stripped_bet.nii.gz"

    success, bet_elapsed, log_output = run_hdbet_subprocess(
        input_path=t1_path,
        output_path=t1_strip_target,
        device=device,
        disable_tta=True,
    )
    summary["hdbet_success"] = success
    summary["hdbet_time_seconds"] = round(bet_elapsed, 3)
    summary["hdbet_log"] = log_output[:500] if log_output else ""

    if not success or not expected_mask_path.exists():
        summary["status"] = "FAILED"
        summary["error"] = f"HD-BET failed or mask not created: {expected_mask_path}"
        return summary

    # Load HD-BET mask
    mask_nii = nib.load(str(expected_mask_path))
    mask_arr = np.asanyarray(mask_nii.dataobj, dtype=np.uint8)
    summary["mask_path"] = str(expected_mask_path)
    summary["mask_shape"] = list(mask_arr.shape)

    # 4. Dual-Channel Mask Application
    t1_native_arr = np.asanyarray(t1_nii.dataobj, dtype=np.float32)
    # Re-mask T1 and T2 strictly using the binary mask
    t1_masked = (t1_native_arr * (mask_arr > 0)).astype(np.float32)
    t2_masked = (t2_reg_arr * (mask_arr > 0)).astype(np.float32)

    t1_final_path = case_out_dir / f"{case_id}_t1_skull_stripped.nii.gz"
    t2_final_path = case_out_dir / f"{case_id}_t2_skull_stripped.nii.gz"

    save_nifti_like(t1_masked, t1_path, t1_final_path, dtype=np.float32)
    save_nifti_like(t2_masked, t1_path, t2_final_path, dtype=np.float32)

    summary["t1_final_path"] = str(t1_final_path)
    summary["t2_final_path"] = str(t2_final_path)

    # 5. Quantitative Verification & Plausibility Audit
    voxel_zooms = tuple(float(z) for z in t1_nii.header.get_zooms()[:3])
    plausibility = validate_brain_mask_plausibility(
        volume=mask_arr,
        voxel_spacing=voxel_zooms,
        min_volume_cc=700.0,
        max_volume_cc=2000.0,
        max_boundary_voxels=500,
    )
    summary["plausibility_audit"] = plausibility

    # Check exact shape and affine preservation
    summary["geometry_preserved"] = bool(
        mask_arr.shape == t1_native_arr.shape
        and t2_reg_arr.shape == t1_native_arr.shape
        and np.allclose(mask_nii.affine, t1_nii.affine, atol=1e-3)
    )

    # 6. Generate Orthogonal Overlays
    overlay_path = case_out_dir / f"{case_id}_orthogonal_overlays.png"
    generate_orthogonal_overlays(
        t1_arr=t1_native_arr,
        t2_arr=t2_reg_arr,
        mask_arr=mask_arr,
        case_id=case_id,
        out_png=overlay_path,
    )
    summary["overlay_path"] = str(overlay_path)

    # Pass / Fail Decision
    is_pass = (
        success
        and summary["geometry_preserved"]
        and plausibility["is_skull_stripped_plausible"]
    )
    summary["status"] = "PASSED" if is_pass else "FLAGGED"

    # Save case JSON report
    report_json_path = case_out_dir / f"{case_id}_pilot_report.json"
    with open(report_json_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    print("=" * 70)
    print("STARTING PHASE 3D-1 PILOT EXECUTION")
    print("=" * 70)

    pilot_cases = [
        {
            "case_id": "IXI002",
            "t1": Path(r"C:\3D_DATASET\HealthyControls\IXI\raw\T1\IXI002-Guys-0828-T1.nii.gz"),
            "t2": Path(r"C:\3D_DATASET\HealthyControls\IXI\raw\T2\IXI002-Guys-0828-T2.nii.gz"),
            "pre_stripped": False,
        },
        {
            "case_id": "BraTS-GLI-00000-000",
            "t1": Path(r"C:\3D_DATASET\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData\BraTS-GLI-00000-000\BraTS-GLI-00000-000-t1n.nii.gz"),
            "t2": Path(r"C:\3D_DATASET\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData\BraTS-GLI-00000-000\BraTS-GLI-00000-000-t2w.nii.gz"),
            "pre_stripped": True,
        },
        {
            "case_id": "BraTS-MEN-00004-000",
            "t1": Path(r"C:\3D_DATASET\BraTS2023_MEN\BraTS-MEN-Train\BraTS-MEN-00004-000\BraTS-MEN-00004-000-t1n.nii.gz"),
            "t2": Path(r"C:\3D_DATASET\BraTS2023_MEN\BraTS-MEN-Train\BraTS-MEN-00004-000\BraTS-MEN-00004-000-t2w.nii.gz"),
            "pre_stripped": True,
        },
    ]

    all_results = {}
    for case in pilot_cases:
        cid = case["case_id"]
        print(f"\n---> Processing Pilot Case: {cid} (pre_stripped={case['pre_stripped']})")
        res = process_pilot_case(
            case_id=cid,
            t1_path=case["t1"],
            t2_path=case["t2"],
            is_pre_skull_stripped=case["pre_stripped"],
            device="cuda",
        )
        all_results[cid] = res
        print(f"Status: {res.get('status')}")
        if "plausibility_audit" in res:
            p = res["plausibility_audit"]
            print(f"  Volume: {p['nonzero_volume_cc']} cc (plausible={p['is_skull_stripped_plausible']})")
            print(f"  Boundary voxels: {p['outer_boundary_voxels']}")
            print(f"  Registration time: {res.get('registration_time_seconds')}s")
            print(f"  HD-BET time: {res.get('hdbet_time_seconds')}s")
            print(f"  Overlay saved to: {res.get('overlay_path')}")

    # Save overall summary
    summary_path = PILOT_OUTPUT_ROOT / "pilot_summary_all.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll pilot cases completed. Global summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
