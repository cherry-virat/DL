"""
Focused Tests for Resumable IXI HD-BET Batch Processing (Phase 3D-3).
======================================================================

Validates:
1. Exact Bug Reproduction & Fix:
   After 100 valid subjects, requesting --max-cases 100 must select the next
   100 pending subjects rather than exiting with zero work.
2. Interruption Recovery:
   Interrupted cases (missing masks, partial files, missing metadata) are
   detected as non-valid and re-queued for processing.
3. Flagged & Failed Case Handling:
   Cases with prior status FLAGGED or FAILED are correctly differentiated from
   valid cases.
4. Deterministic Ordering:
   Pending case selection preserves exact alphabetical/numerical cohort sorting.
5. Real Disk Verification:
   Read-only test against the real C:\\3D_DATASET\\HealthyControls\\IXI directory
   confirms that next chunk selects pending cases 101..200 without modifying any files.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
import tempfile
import shutil

import nibabel as nib
import numpy as np
import pytest

from scripts.run_hdbet_batch import (
    is_case_already_valid,
    select_batch_cases,
    discover_and_match_ixi_cohort,
    DEFAULT_RAW_T1_DIR,
    DEFAULT_RAW_T2_DIR,
    DEFAULT_PROCESSED_ROOT,
)


def _create_mock_nifti(path: Path, shape=(20, 20, 20), affine=None):
    """Helper creating a minimal valid NIfTI file for fast testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if affine is None:
        affine = np.eye(4)
    data = np.ones(shape, dtype=np.float32)
    img = nib.Nifti1Image(data, affine)
    nib.save(img, str(path))


def _create_mock_processed_case(output_root: Path, sub_id: str, status: str = "PASSED", shape=(20, 20, 20)):
    """Creates a complete mock processed case with 4 files."""
    t1_p = output_root / "T1" / f"{sub_id}_t1_skull_stripped.nii.gz"
    t2_p = output_root / "T2" / f"{sub_id}_t2_skull_stripped.nii.gz"
    mask_p = output_root / "masks" / f"{sub_id}_mask.nii.gz"
    meta_p = output_root / "metadata" / f"{sub_id}_meta.json"

    _create_mock_nifti(t1_p, shape=shape)
    _create_mock_nifti(t2_p, shape=shape)
    # Binary mask
    mask_p.parent.mkdir(parents=True, exist_ok=True)
    mask_arr = np.ones(shape, dtype=np.uint8)
    nib.save(nib.Nifti1Image(mask_arr, np.eye(4)), str(mask_p))

    meta = {
        "participant_id": sub_id,
        "status": status,
        "plausibility_audit": {
            "is_skull_stripped_plausible": (status == "PASSED"),
            "nonzero_volume_cc": 1400.0 if status == "PASSED" else 400.0,
            "outer_boundary_voxels": 0 if status == "PASSED" else 800,
            "failure_reasons": [] if status == "PASSED" else ["Boundary leakage"],
        },
        "registration_time_seconds": 1.0,
        "hdbet_time_seconds": 15.0,
        "total_case_time_seconds": 16.0,
    }
    meta_p.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_p, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


# =====================================================================
# Unit Tests for select_batch_cases & is_case_already_valid
# =====================================================================

def test_select_batch_cases_advances_after_completed_prefix(tmp_path):
    """
    Exact Bug Verification (Task 5):
    Given 576 eligible cases where the first 100 are already valid:
    --max-cases 100 must select pending cases 101..200 (not cases 1..100)
    and must NOT exit with 0 cases to process.
    """
    output_root = tmp_path / "processed"

    # Create synthetic cohort of 576 cases
    cohort: List[Dict[str, Any]] = [
        {"participant_id": f"IXI{i:03d}", "t1_path": Path(f"t1_{i}"), "t2_path": Path(f"t2_{i}")}
        for i in range(1, 577)
    ]

    # Mark first 100 cases as valid
    for c in cohort[:100]:
        _create_mock_processed_case(output_root, c["participant_id"], status="PASSED")

    # Run selection with max_cases=100
    cases_to_process, already_valid, deferred_count = select_batch_cases(
        eligible_cases=cohort,
        output_root=output_root,
        max_cases=100,
    )

    # Assertions
    assert len(already_valid) == 100, f"Expected 100 skipped valid cases, got {len(already_valid)}"
    assert len(cases_to_process) == 100, f"Expected 100 cases to process, got {len(cases_to_process)}"
    assert deferred_count == 376, f"Expected 376 deferred cases, got {deferred_count}"

    # Verify that cases_to_process starts at case 101 (IXI101) and ends at case 200 (IXI200)
    expected_ids = [f"IXI{i:03d}" for i in range(101, 201)]
    actual_ids = [c["participant_id"] for c in cases_to_process]
    assert actual_ids == expected_ids

    # Verify zero overlap between already_valid and cases_to_process
    valid_ids = {c["participant_id"] for c in already_valid}
    process_ids = {c["participant_id"] for c in cases_to_process}
    assert len(valid_ids & process_ids) == 0


def test_select_batch_cases_all_completed_returns_zero_pending(tmp_path):
    """When all cases in cohort are valid, returns 0 pending cases without error."""
    output_root = tmp_path / "processed"
    cohort = [
        {"participant_id": f"IXI{i:03d}", "t1_path": Path(f"t1_{i}"), "t2_path": Path(f"t2_{i}")}
        for i in range(1, 11)
    ]
    for c in cohort:
        _create_mock_processed_case(output_root, c["participant_id"], status="PASSED")

    cases_to_process, already_valid, deferred_count = select_batch_cases(
        eligible_cases=cohort,
        output_root=output_root,
        max_cases=5,
    )

    assert len(already_valid) == 10
    assert len(cases_to_process) == 0
    assert deferred_count == 0


def test_select_batch_cases_interrupted_case_resumption(tmp_path):
    """
    Task 6: Test behavior when a chunk is interrupted.
    Cases 1..4 valid, Case 5 interrupted (missing mask), Case 6..10 pending.
    Case 5 must be detected as incomplete and re-queued at the front of cases_to_process.
    """
    output_root = tmp_path / "processed"
    cohort = [
        {"participant_id": f"IXI{i:03d}", "t1_path": Path(f"t1_{i}"), "t2_path": Path(f"t2_{i}")}
        for i in range(1, 11)
    ]

    # Cases 1..4 complete
    for c in cohort[:4]:
        _create_mock_processed_case(output_root, c["participant_id"], status="PASSED")

    # Case 5: partial/interrupted (T1 exists, but mask is missing)
    t1_5 = output_root / "T1" / "IXI005_t1_skull_stripped.nii.gz"
    _create_mock_nifti(t1_5)

    is_valid, reason = is_case_already_valid("IXI005", output_root)
    assert is_valid is False
    assert "Missing" in reason

    # Select batch with max_cases=3
    cases_to_process, already_valid, deferred_count = select_batch_cases(
        eligible_cases=cohort,
        output_root=output_root,
        max_cases=3,
    )

    assert len(already_valid) == 4
    assert len(cases_to_process) == 3
    # First pending case must be the interrupted case IXI005
    assert cases_to_process[0]["participant_id"] == "IXI005"
    assert cases_to_process[1]["participant_id"] == "IXI006"
    assert cases_to_process[2]["participant_id"] == "IXI007"
    assert deferred_count == 3  # IXI008, IXI009, IXI010 deferred


def test_select_batch_cases_flagged_and_failed_cases(tmp_path):
    """
    Task 6: Test behavior with FLAGGED and FAILED cases.
    - Status FLAGGED: not valid; re-queued by default, or skipped if skip_flagged=True.
    - Status FAILED: not valid; re-queued for retry.
    """
    output_root = tmp_path / "processed"
    cohort = [
        {"participant_id": "IXI001", "t1_path": Path("t1_1"), "t2_path": Path("t2_1")},
        {"participant_id": "IXI002", "t1_path": Path("t1_2"), "t2_path": Path("t2_2")},
        {"participant_id": "IXI003", "t1_path": Path("t1_3"), "t2_path": Path("t2_3")},
        {"participant_id": "IXI004", "t1_path": Path("t1_4"), "t2_path": Path("t2_4")},
    ]

    _create_mock_processed_case(output_root, "IXI001", status="PASSED")
    _create_mock_processed_case(output_root, "IXI002", status="FLAGGED")
    _create_mock_processed_case(output_root, "IXI003", status="FAILED")
    # IXI004: completely unprocessed

    is_v1, _ = is_case_already_valid("IXI001", output_root)
    is_v2, r2 = is_case_already_valid("IXI002", output_root)
    is_v3, r3 = is_case_already_valid("IXI003", output_root)
    is_v4, _ = is_case_already_valid("IXI004", output_root)

    assert is_v1 is True
    assert is_v2 is False and "FLAGGED" in r2
    assert is_v3 is False and "FAILED" in r3
    assert is_v4 is False

    # Default: FLAGGED and FAILED are queued for re-evaluation/processing
    to_proc_def, valid_def, _ = select_batch_cases(cohort, output_root)
    assert len(valid_def) == 1  # only IXI001
    assert [c["participant_id"] for c in to_proc_def] == ["IXI002", "IXI003", "IXI004"]

    # With skip_flagged=True: FLAGGED is bypassed, only FAILED and PENDING are queued
    to_proc_skip, valid_skip, _ = select_batch_cases(cohort, output_root, skip_flagged=True)
    assert len(valid_skip) == 2  # IXI001 (passed) and IXI002 (flagged skipped)
    assert [c["participant_id"] for c in to_proc_skip] == ["IXI003", "IXI004"]


def test_select_batch_cases_deterministic_ordering(tmp_path):
    """Verifies that pending cases strictly maintain sorted alphabetical order."""
    output_root = tmp_path / "processed"
    cohort = [
        {"participant_id": f"IXI{i:03d}", "t1_path": Path(f"t1_{i}"), "t2_path": Path(f"t2_{i}")}
        for i in [10, 5, 2, 8, 1, 9, 3, 7, 4, 6]
    ]
    # Sort cohort deterministically as discover_and_match_ixi_cohort does
    cohort = sorted(cohort, key=lambda x: x["participant_id"])

    # Mark IXI001 and IXI002 as valid
    _create_mock_processed_case(output_root, "IXI001", status="PASSED")
    _create_mock_processed_case(output_root, "IXI002", status="PASSED")

    cases_to_process, already_valid, _ = select_batch_cases(cohort, output_root, max_cases=4)
    proc_ids = [c["participant_id"] for c in cases_to_process]
    assert proc_ids == ["IXI003", "IXI004", "IXI005", "IXI006"]


def test_existing_valid_files_never_deleted_or_overwritten(tmp_path):
    """Task 4: Confirms select_batch_cases performs strictly read-only inspection."""
    output_root = tmp_path / "processed"
    cohort = [{"participant_id": "IXI001", "t1_path": Path("t1_1"), "t2_path": Path("t2_1")}]
    _create_mock_processed_case(output_root, "IXI001", status="PASSED")

    meta_file = output_root / "metadata" / "IXI001_meta.json"
    mtime_before = meta_file.stat().st_mtime_ns

    cases_to_process, already_valid, _ = select_batch_cases(cohort, output_root, max_cases=10)
    assert len(already_valid) == 1
    assert len(cases_to_process) == 0

    mtime_after = meta_file.stat().st_mtime_ns
    assert mtime_before == mtime_after, "File modification time changed during selection check!"


# =====================================================================
# Real Disk Smoke Test (Read-Only)
# =====================================================================

def test_real_disk_ixi_batch_resumption_selects_next_100():
    """
    Phase 3D-3 Real Data Verification:
    Executes select_batch_cases against the actual C:\\3D_DATASET\\HealthyControls\\IXI directories.
    Strictly READ-ONLY (no files created, modified, or deleted).

    Verifies:
    1. All 576 eligible 3D pairs are discovered.
    2. Completed valid cases on disk are recognized without re-processing (at least 301).
    3. Pending cases are dynamically selected up to max_cases without overlap.
    4. Partition arithmetic: valid + to_process + deferred equals 576 exactly.
    5. IXI232 is confirmed present in the completed set.
    6. Zero intersection between valid and to-process sets.
    """
    if not (DEFAULT_RAW_T1_DIR.exists() and DEFAULT_PROCESSED_ROOT.exists()):
        pytest.skip("Real IXI directories not accessible on this machine.")

    eligible_cases, audit_summary = discover_and_match_ixi_cohort(DEFAULT_RAW_T1_DIR, DEFAULT_RAW_T2_DIR)
    assert len(eligible_cases) == 576

    valid_count_on_disk = 0
    for mf in (DEFAULT_PROCESSED_ROOT / "metadata").glob("*.json"):
        try:
            with open(mf, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            if data.get("status") == "PASSED" and data.get("plausibility_audit", {}).get("is_skull_stripped_plausible", False):
                valid_count_on_disk += 1
        except Exception:
            pass

    assert valid_count_on_disk >= 301, f"Expected at least 301 valid cases on disk, found {valid_count_on_disk}"

    cases_to_process, already_valid, deferred_count = select_batch_cases(
        eligible_cases=eligible_cases,
        output_root=DEFAULT_PROCESSED_ROOT,
        max_cases=100,
    )

    # 1. Valid count must match QC-passed physical cases on disk
    assert len(already_valid) == valid_count_on_disk
    assert already_valid[0]["participant_id"] == "IXI002"
    valid_ids = {c["participant_id"] for c in already_valid}
    assert "IXI232" in valid_ids, "IXI232 must be in valid completed set!"

    # 2. Partition arithmetic: valid + to_process + deferred must equal 576 exactly
    total_pending = 576 - len(already_valid)
    expected_chunk = min(100, total_pending)
    expected_deferred = total_pending - expected_chunk

    assert len(cases_to_process) == expected_chunk
    assert deferred_count == expected_deferred
    assert len(already_valid) + len(cases_to_process) + deferred_count == 576

    # 3. If there are pending cases, verify the first pending case
    if expected_chunk > 0:
        first_pending_id = cases_to_process[0]["participant_id"]
        assert first_pending_id not in valid_ids, f"First pending case {first_pending_id} is already in valid set!"
        proc_ids = {c["participant_id"] for c in cases_to_process}
        assert len(valid_ids & proc_ids) == 0
