"""
Batch Brain Extraction & Dual-Channel Preprocessing Pipeline for IXI Healthy Controls.
======================================================================================
Phase 3D-2: High-throughput rigid registration, HD-BET skull stripping, dual-channel
masking, and quality-assurance auditing for the normative healthy cohort.

Strict Safeguards:
- Raw inputs are strictly READ-ONLY.
- Outputs written to:
    C:\\3D_DATASET\\HealthyControls\\IXI\\processed\\T1
    C:\\3D_DATASET\\HealthyControls\\IXI\\processed\\T2
    C:\\3D_DATASET\\HealthyControls\\IXI\\processed\\masks
    C:\\3D_DATASET\\HealthyControls\\IXI\\processed\\metadata
- Non-destructive resume support: existing valid cases are verified and skipped.
- Continuous disk-space guarding (aborts if free space < 2.0 GB).
- Comprehensive per-case and cohort-level JSON auditing.
"""

import argparse
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(r"C:\Users\nchan\Desktop\DL")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import nibabel as nib
import numpy as np
import torch
import SimpleITK as sitk

from preprocessing.tumor_presence_preprocess import (
    align_and_resample_t2_to_t1,
    validate_brain_mask_plausibility,
)
try:
    from HD_BET.hd_bet_prediction import get_hdbet_predictor
except ImportError:
    get_hdbet_predictor = None

# Default paths
DEFAULT_RAW_T1_DIR = Path(r"C:\3D_DATASET\HealthyControls\IXI\raw\T1")
DEFAULT_RAW_T2_DIR = Path(r"C:\3D_DATASET\HealthyControls\IXI\raw\T2")
DEFAULT_PROCESSED_ROOT = Path(r"C:\3D_DATASET\HealthyControls\IXI\processed")

PARTICIPANT_ID_PATTERN = re.compile(r"^(IXI\d{3})")
MIN_FREE_DISK_GB = 2.0


def check_free_disk_space_gb(path: Path) -> float:
    """Returns free space in GB on the drive containing path."""
    total, used, free = shutil.disk_usage(str(path.anchor if path.anchor else "C:"))
    return round(free / (1024 ** 3), 2)


def discover_and_match_ixi_cohort(
    t1_dir: Path,
    t2_dir: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Scans raw directories, matches T1 and T2 pairs by IXI participant ID,
    identifies missing pairs, and filters out known thick-slice cases (IXI014).
    """
    t1_files = sorted(list(t1_dir.glob("*.nii.gz")))
    t2_files = sorted(list(t2_dir.glob("*.nii.gz")))

    t1_map: Dict[str, Path] = {}
    for f in t1_files:
        m = PARTICIPANT_ID_PATTERN.match(f.name)
        if m:
            t1_map[m.group(1)] = f

    t2_map: Dict[str, Path] = {}
    for f in t2_files:
        m = PARTICIPANT_ID_PATTERN.match(f.name)
        if m:
            t2_map[m.group(1)] = f

    all_ids = sorted(list(set(t1_map.keys()) | set(t2_map.keys())))
    paired_ids = sorted(list(set(t1_map.keys()) & set(t2_map.keys())))
    t1_only = sorted(list(set(t1_map.keys()) - set(t2_map.keys())))
    t2_only = sorted(list(set(t2_map.keys()) - set(t1_map.keys())))

    eligible_cases: List[Dict[str, Any]] = []
    excluded_cases: List[Dict[str, Any]] = []

    for sub_id in paired_ids:
        t1_p = t1_map[sub_id]
        t2_p = t2_map[sub_id]

        # IXI014 is a known non-volumetric acquisition (28 slices, 4.98mm slice thickness)
        if sub_id == "IXI014":
            excluded_cases.append({
                "participant_id": sub_id,
                "t1_path": str(t1_p),
                "t2_path": str(t2_p),
                "reason": "Thick-slice 2D acquisition (28 slices, ~5mm slice thickness)",
            })
            continue

        eligible_cases.append({
            "participant_id": sub_id,
            "t1_path": t1_p,
            "t2_path": t2_p,
            "t1_filename": t1_p.name,
            "t2_filename": t2_p.name,
        })

    audit_summary = {
        "raw_t1_files": len(t1_files),
        "raw_t2_files": len(t2_files),
        "total_unique_participants": len(all_ids),
        "paired_participants": len(paired_ids),
        "t1_only_unpaired": t1_only,
        "t2_only_unpaired": t2_only,
        "excluded_thick_slice": excluded_cases,
        "eligible_3d_pairs_count": len(eligible_cases),
    }

    return eligible_cases, audit_summary


def is_case_already_valid(
    participant_id: str,
    output_root: Path,
) -> Tuple[bool, str]:
    """
    Validates whether an existing processed case is complete, uncorrupted,
    and passed quality checks.
    """
    t1_out = output_root / "T1" / f"{participant_id}_t1_skull_stripped.nii.gz"
    t2_out = output_root / "T2" / f"{participant_id}_t2_skull_stripped.nii.gz"
    mask_out = output_root / "masks" / f"{participant_id}_mask.nii.gz"
    meta_out = output_root / "metadata" / f"{participant_id}_meta.json"

    for p in (t1_out, t2_out, mask_out, meta_out):
        if not p.exists():
            return False, f"Missing {p.name}"

    try:
        with open(meta_out, "r") as f:
            meta = json.load(f)
        if meta.get("status") != "PASSED":
            return False, f"Prior status was {meta.get('status')}"

        # Fast header inspection to verify geometric alignment & uncorrupted gzip stream
        t1_hdr = nib.load(str(t1_out)).header
        t2_hdr = nib.load(str(t2_out)).header
        mask_hdr = nib.load(str(mask_out)).header

        s1 = tuple(int(x) for x in t1_hdr.get_data_shape()[:3])
        s2 = tuple(int(x) for x in t2_hdr.get_data_shape()[:3])
        sm = tuple(int(x) for x in mask_hdr.get_data_shape()[:3])

        if not (s1 == s2 == sm):
            return False, f"Shape mismatch: T1 {s1}, T2 {s2}, mask {sm}"

        return True, "Valid processed case"
    except Exception as e:
        return False, f"Validation read failed: {e}"


def save_nifti_like(arr: np.ndarray, ref_nii: nib.Nifti1Image, out_path: Path, dtype=None):
    """Saves array using reference NIfTI coordinate system, origin, spacing, and header."""
    if dtype is not None:
        arr = arr.astype(dtype)
    new_nii = nib.Nifti1Image(arr, ref_nii.affine, ref_nii.header)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(new_nii, str(out_path))


def process_single_ixi_case(
    case_info: Dict[str, Any],
    predictor: Any,
    output_root: Path,
) -> Dict[str, Any]:
    """
    Executes rigid coregistration of T2 to T1, HD-BET brain extraction,
    dual-channel masking, and quality assurance audit for a single participant.
    """
    sub_id = case_info["participant_id"]
    t1_path: Path = case_info["t1_path"]
    t2_path: Path = case_info["t2_path"]

    t0_case = time.perf_counter()
    report: Dict[str, Any] = {
        "participant_id": sub_id,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "t1_raw_path": str(t1_path),
        "t2_raw_path": str(t2_path),
    }

    # 1. Load and inspect native volumes
    t1_nii = nib.load(str(t1_path))
    t2_nii = nib.load(str(t2_path))

    t1_raw_arr = np.asanyarray(t1_nii.dataobj, dtype=np.float32)
    t2_raw_arr = np.asanyarray(t2_nii.dataobj, dtype=np.float32)

    # Validate finite values
    if not (np.all(np.isfinite(t1_raw_arr)) and np.all(np.isfinite(t2_raw_arr))):
        raise ValueError("Non-finite intensities (NaN or Inf) detected in raw inputs")

    t1_shape = list(t1_raw_arr.shape)
    t2_shape = list(t2_raw_arr.shape)
    t1_zooms = [float(z) for z in t1_nii.header.get_zooms()[:3]]
    t2_zooms = [float(z) for z in t2_nii.header.get_zooms()[:3]]

    report["t1_native_shape"] = t1_shape
    report["t2_native_shape"] = t2_shape
    report["t1_native_zooms"] = t1_zooms
    report["t2_native_zooms"] = t2_zooms

    # 2. Rigid registration: T2 -> T1 grid
    t0_reg = time.perf_counter()
    t2_reg_arr = align_and_resample_t2_to_t1(
        fixed_src=t1_path,
        moving_src=t2_path,
        fixed_affine=t1_nii.affine,
        moving_affine=t2_nii.affine,
    )
    reg_elapsed = time.perf_counter() - t0_reg
    report["registration_time_seconds"] = round(reg_elapsed, 3)

    if not np.all(np.isfinite(t2_reg_arr)):
        raise ValueError("Registration yielded non-finite intensities in resampled T2")

    # 3. HD-BET brain mask inference on T1
    mask_out_dir = output_root / "masks"
    mask_out_dir.mkdir(parents=True, exist_ok=True)
    mask_file_path = mask_out_dir / f"{sub_id}_mask.nii.gz"

    t0_bet = time.perf_counter()
    predictor.predict_from_files(
        [[str(t1_path)]],
        [str(mask_file_path)],
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0,
    )
    bet_elapsed = time.perf_counter() - t0_bet
    report["hdbet_time_seconds"] = round(bet_elapsed, 3)

    # Clean up nnUNet temporary json dump files in masks directory
    for extra_name in ["dataset.json", "plans.json", "predict_from_raw_data_args.json"]:
        extra_p = mask_out_dir / extra_name
        if extra_p.exists():
            try:
                extra_p.unlink()
            except OSError:
                pass

    if not mask_file_path.exists():
        raise FileNotFoundError(f"HD-BET failed to create mask: {mask_file_path}")

    # 4. Load & validate mask
    mask_nii = nib.load(str(mask_file_path))
    mask_arr = np.asanyarray(mask_nii.dataobj, dtype=np.uint8)

    # Verify binary values
    unique_vals = set(np.unique(mask_arr))
    if not unique_vals.issubset({0, 1}):
        raise ValueError(f"Brain mask contains non-binary values: {unique_vals}")

    # Plausibility checks
    voxel_zooms = tuple(float(z) for z in t1_nii.header.get_zooms()[:3])
    plausibility = validate_brain_mask_plausibility(
        volume=mask_arr,
        voxel_spacing=voxel_zooms,
        min_volume_cc=700.0,
        max_volume_cc=2000.0,
        max_boundary_voxels=500,
    )
    report["plausibility_audit"] = plausibility

    # 5. Dual-channel mask application
    binary_mask = (mask_arr > 0)
    t1_masked = (t1_raw_arr * binary_mask).astype(np.float32)
    t2_masked = (t2_reg_arr * binary_mask).astype(np.float32)

    # 6. Save final processed NIfTI files
    t1_out_path = output_root / "T1" / f"{sub_id}_t1_skull_stripped.nii.gz"
    t2_out_path = output_root / "T2" / f"{sub_id}_t2_skull_stripped.nii.gz"

    save_nifti_like(t1_masked, t1_nii, t1_out_path, dtype=np.float32)
    save_nifti_like(t2_masked, t1_nii, t2_out_path, dtype=np.float32)

    report["t1_processed_path"] = str(t1_out_path)
    report["t2_processed_path"] = str(t2_out_path)
    report["mask_processed_path"] = str(mask_file_path)

    # Verify geometry preservation
    geometry_preserved = bool(
        t1_masked.shape == tuple(t1_shape)
        and t2_masked.shape == tuple(t1_shape)
        and mask_arr.shape == tuple(t1_shape)
        and np.allclose(mask_nii.affine, t1_nii.affine, atol=1e-3)
    )
    report["geometry_preserved"] = geometry_preserved

    total_elapsed = time.perf_counter() - t0_case
    report["total_case_time_seconds"] = round(total_elapsed, 3)

    is_passed = (
        geometry_preserved
        and plausibility["is_skull_stripped_plausible"]
    )
    report["status"] = "PASSED" if is_passed else "FLAGGED"

    # Save per-case metadata
    meta_out_dir = output_root / "metadata"
    meta_out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_out_dir / f"{sub_id}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


def select_batch_cases(
    eligible_cases: List[Dict[str, Any]],
    output_root: Path,
    max_cases: Optional[int] = None,
    start_index: int = 0,
    case_ids: Optional[str] = None,
    no_resume: bool = False,
    skip_flagged: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    """
    Selects pending cases to process in the current run based on resume status.

    Parameters:
        eligible_cases: Deterministically sorted list of all eligible cohort cases.
        output_root: Root path for processed outputs.
        max_cases: Maximum number of pending cases to process in this run.
        start_index: Offset into the eligible cases pool before evaluation.
        case_ids: Optional comma-separated participant IDs to filter by.
        no_resume: If True, bypasses resume validation and reprocesses all selected cases.
        skip_flagged: If True, treats cases with prior 'FLAGGED' status as completed/skipped.

    Returns:
        cases_to_process: List of pending cases targeted for execution in this run (up to max_cases).
        already_valid_cases: List of cases skipped as already processed/valid.
        deferred_count: Number of pending cases deferred for subsequent runs.
    """
    cases_pool = list(eligible_cases)
    if case_ids:
        requested_ids = set(cid.strip() for cid in case_ids.split(",") if cid.strip())
        cases_pool = [c for c in cases_pool if c["participant_id"] in requested_ids]

    if start_index > 0:
        cases_pool = cases_pool[start_index:]

    already_valid_cases: List[Dict[str, Any]] = []
    pending_cases: List[Dict[str, Any]] = []

    for c in cases_pool:
        sub_id = c["participant_id"]
        if not no_resume:
            is_valid, reason = is_case_already_valid(sub_id, output_root)
            if is_valid:
                already_valid_cases.append(c)
                continue
            if skip_flagged and "Prior status was FLAGGED" in reason:
                already_valid_cases.append(c)
                continue

        pending_cases.append(c)

    if max_cases is not None and max_cases > 0:
        cases_to_process = pending_cases[:max_cases]
        deferred_count = len(pending_cases) - len(cases_to_process)
    else:
        cases_to_process = pending_cases
        deferred_count = 0

    return cases_to_process, already_valid_cases, deferred_count


def main():
    parser = argparse.ArgumentParser(description="IXI HD-BET Batch Preprocessing Pipeline")
    parser.add_argument("--raw-t1-dir", type=Path, default=DEFAULT_RAW_T1_DIR, help="Raw T1 directory")
    parser.add_argument("--raw-t2-dir", type=Path, default=DEFAULT_RAW_T2_DIR, help="Raw T2 directory")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_PROCESSED_ROOT, help="Processed output root")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device ('cuda' or 'cpu')")
    parser.add_argument("--max-cases", type=int, default=None, help="Maximum number of cases to process in this run")
    parser.add_argument("--start-index", type=int, default=0, help="Starting index in eligible cohort")
    parser.add_argument("--case-ids", type=str, default=None, help="Comma-separated participant IDs to process")
    parser.add_argument("--no-resume", action="store_true", help="Force reprocessing even if valid output exists")
    parser.add_argument("--skip-flagged", action="store_true", default=False, help="Skip cases with prior FLAGGED status")
    parser.add_argument("--disable-tta", action="store_true", default=True, help="Disable test-time augmentation")
    args = parser.parse_args()

    print("=" * 70)
    print("IXI Healthy-Control Batch Preprocessing Pipeline (Phase 3D-2)")
    print("=" * 70)
    print(f"Timestamp:        {datetime.now(timezone.utc).isoformat()}")
    print(f"Raw T1 Root:      {args.raw_t1_dir}")
    print(f"Raw T2 Root:      {args.raw_t2_dir}")
    print(f"Output Root:      {args.output_root}")
    print(f"Device:           {args.device}")
    print(f"Resume Enabled:   {not args.no_resume}")

    # Check initial disk space
    free_gb = check_free_disk_space_gb(args.output_root)
    print(f"Free Disk Space:  {free_gb} GB")
    if free_gb < MIN_FREE_DISK_GB:
        print(f"FATAL: Insufficient disk space ({free_gb} GB < {MIN_FREE_DISK_GB} GB). Aborting.")
        sys.exit(1)

    # 1. Scan and Match Cohort
    eligible_cases, audit_summary = discover_and_match_ixi_cohort(args.raw_t1_dir, args.raw_t2_dir)
    print(f"\nCohort Inventory:")
    print(f"  Raw T1 files:             {audit_summary['raw_t1_files']}")
    print(f"  Raw T2 files:             {audit_summary['raw_t2_files']}")
    print(f"  Paired participants:      {audit_summary['paired_participants']}")
    print(f"  T1-only (missing T2):     {audit_summary['t1_only_unpaired']}")
    print(f"  T2-only (missing T1):     {audit_summary['t2_only_unpaired']}")
    print(f"  Excluded thick-slice:     {[c['participant_id'] for c in audit_summary['excluded_thick_slice']]}")
    print(f"  Eligible 3D pairs:        {len(eligible_cases)}")

    # Ensure output directories exist
    for sub in ["T1", "T2", "masks", "metadata"]:
        (args.output_root / sub).mkdir(parents=True, exist_ok=True)

    # Save cohort audit summary
    with open(args.output_root / "cohort_inventory_audit.json", "w") as f:
        json.dump(audit_summary, f, indent=2)

    # 2. Select batch cases with resume-aware pending queue
    cases_to_process, already_valid_cases, deferred_count = select_batch_cases(
        eligible_cases=eligible_cases,
        output_root=args.output_root,
        max_cases=args.max_cases,
        start_index=args.start_index,
        case_ids=args.case_ids,
        no_resume=args.no_resume,
        skip_flagged=args.skip_flagged,
    )
    skipped_count = len(already_valid_cases)

    print(f"\nResume Status Check:")
    print(f"  Total eligible in cohort:            {len(eligible_cases)}")
    print(f"  Already processed & valid (skipped): {skipped_count}")
    print(f"  Targeted for this run:               {len(cases_to_process)}")
    print(f"  Remaining pending (deferred):        {deferred_count}")

    if not cases_to_process:
        print("\nAll requested cases are already processed and validated. Nothing to do!")
        sys.exit(0)

    # 3. Initialize HD-BET predictor
    device_obj = torch.device(args.device if (torch.cuda.is_available() and args.device == "cuda") else "cpu")
    print(f"\nInitializing HD-BET predictor on {device_obj}...")
    if get_hdbet_predictor is None:
        raise ImportError(
            "HD_BET package not found in current environment. "
            "Batch execution requires .venv_hdbet (e.g. .venv_hdbet\\Scripts\\python.exe scripts\\run_hdbet_batch.py)"
        )
    t0_model = time.perf_counter()
    predictor = get_hdbet_predictor(use_tta=not args.disable_tta, device=device_obj, verbose=False)
    print(f"Predictor ready in {time.perf_counter() - t0_model:.2f}s.")

    # 4. Process Batch
    passed_count = 0
    flagged_count = 0
    failed_count = 0
    runtimes: List[float] = []

    t_batch_start = time.perf_counter()

    for idx, c in enumerate(cases_to_process, 1):
        sub_id = c["participant_id"]
        print(f"\n[{idx}/{len(cases_to_process)}] Processing {sub_id} ({c['t1_filename']})...")

        # Disk space safeguard check before each case
        cur_free_gb = check_free_disk_space_gb(args.output_root)
        if cur_free_gb < MIN_FREE_DISK_GB:
            print(f"CRITICAL STOP: Drive C free space reached {cur_free_gb} GB (< {MIN_FREE_DISK_GB} GB). Stopping early.")
            break

        try:
            res = process_single_ixi_case(c, predictor, args.output_root)
            status = res["status"]
            vol_cc = res["plausibility_audit"]["nonzero_volume_cc"]
            boundary_vox = res["plausibility_audit"]["outer_boundary_voxels"]
            reg_sec = res["registration_time_seconds"]
            bet_sec = res["hdbet_time_seconds"]
            tot_sec = res["total_case_time_seconds"]
            runtimes.append(tot_sec)

            if status == "PASSED":
                passed_count += 1
                print(f"  -> PASSED in {tot_sec:.1f}s (reg: {reg_sec:.1f}s, bet: {bet_sec:.1f}s | Vol: {vol_cc}cc, Boundary: {boundary_vox} voxels)")
            else:
                flagged_count += 1
                reasons = res["plausibility_audit"].get("failure_reasons", [])
                print(f"  -> FLAGGED in {tot_sec:.1f}s | Reasons: {reasons}")

        except Exception as err:
            failed_count += 1
            print(f"  -> FAILED: {err}")
            # Record failed case metadata
            fail_report = {
                "participant_id": sub_id,
                "status": "FAILED",
                "error": str(err),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            with open(args.output_root / "metadata" / f"{sub_id}_meta.json", "w") as f:
                json.dump(fail_report, f, indent=2)

        # Free CUDA cache and trigger garbage collection between cases
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

        # Incremental batch progress file update
        cur_elapsed_sec = time.perf_counter() - t_batch_start
        cases_done = passed_count + flagged_count + failed_count
        avg_sec = (cur_elapsed_sec / cases_done) if cases_done > 0 else 0.0
        remaining_cases = len(cases_to_process) - cases_done
        eta_minutes = round((remaining_cases * avg_sec) / 60, 1)

        progress_data = {
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "last_case_processed": sub_id,
            "cohort_total_eligible": len(eligible_cases),
            "batch_total_targets": len(cases_to_process),
            "batch_processed_so_far": cases_done,
            "overall_completed_including_skipped": skipped_count + cases_done,
            "percent_of_cohort_completed": round(((skipped_count + cases_done) / len(eligible_cases)) * 100, 1) if len(eligible_cases) > 0 else 0.0,
            "batch_passed": passed_count,
            "batch_flagged": flagged_count,
            "batch_failed": failed_count,
            "batch_skipped_prior": skipped_count,
            "elapsed_minutes": round(cur_elapsed_sec / 60, 2),
            "avg_seconds_per_case": round(avg_sec, 2),
            "estimated_remaining_minutes": eta_minutes,
            "current_free_disk_gb": check_free_disk_space_gb(args.output_root),
        }
        with open(args.output_root / "batch_progress.json", "w") as pf:
            json.dump(progress_data, pf, indent=2)

    total_batch_time = time.perf_counter() - t_batch_start
    final_free_gb = check_free_disk_space_gb(args.output_root)

    # 5. Cohort Summary
    summary = {
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_cohort_eligible": len(eligible_cases),
        "requested_batch_size": len(cases_to_process),
        "processed_count": passed_count + flagged_count + failed_count,
        "passed_count": passed_count,
        "flagged_count": flagged_count,
        "failed_count": failed_count,
        "skipped_valid_count": skipped_count,
        "deferred_pending_count": deferred_count,
        "total_batch_seconds": round(total_batch_time, 2),
        "mean_case_seconds": round(float(np.mean(runtimes)), 2) if runtimes else 0.0,
        "min_case_seconds": round(float(np.min(runtimes)), 2) if runtimes else 0.0,
        "max_case_seconds": round(float(np.max(runtimes)), 2) if runtimes else 0.0,
        "disk_free_gb_before": free_gb,
        "disk_free_gb_after": final_free_gb,
    }

    summary_file = args.output_root / "cohort_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("Batch Preprocessing Execution Finished")
    print("=" * 70)
    print(f"Passed:           {passed_count}")
    print(f"Flagged:          {flagged_count}")
    print(f"Failed:           {failed_count}")
    print(f"Skipped (Resume): {skipped_count}")
    print(f"Total Time:       {total_batch_time / 60:.2f} minutes")
    if runtimes:
        print(f"Average Time:     {np.mean(runtimes):.1f} s/case")
    print(f"Free Disk Space:  {final_free_gb} GB (delta: {round(final_free_gb - free_gb, 2)} GB)")
    print(f"Summary Saved:    {summary_file}")


if __name__ == "__main__":
    main()
