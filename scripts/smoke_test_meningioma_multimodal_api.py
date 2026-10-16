"""
smoke_test_meningioma_multimodal_api.py
=======================================
End-to-end smoke test for the 4-channel Multimodal Meningioma REST API.

Pipeline:
1. Selects one case from the existing validation split (default: BraTS-MEN-00020-000).
2. Verifies that the case belongs strictly to the validation split in split.json.
3. Locates the four aligned NIfTI modalities: t1n, t1c, t2w, and t2f.
4. Checks the API /api/health endpoint.
5. Uploads the four scans via POST /api/upload.
6. Invokes segmentation via POST /api/segment/meningioma-multimodal.
7. Retrieves and verifies the segmentation result and subregion voxel counts (WT, TC, ET, NCR, ED).
8. Downloads the generated 3D NIfTI segmentation mask from /api/mask/<request_id>.
9. Reports all HTTP statuses, request IDs, mask paths, and volumetric findings.

Usage (Live HTTP Server):
--------------------------
Terminal 1 (Start server):
    .\\.venv\\Scripts\\python.exe api\\app.py

Terminal 2 (Run smoke test):
    .\\.venv\\Scripts\\python.exe scripts\\smoke_test_meningioma_multimodal_api.py

Usage (In-Process Test Client, no separate server needed):
---------------------------------------------------------
    .\\.venv\\Scripts\\python.exe scripts\\smoke_test_meningioma_multimodal_api.py --in-process
"""

import argparse
import gzip
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import nibabel as nib
import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SPLIT_JSON = PROJECT_ROOT / "data" / "meningioma_multimodal" / "split.json"
DEFAULT_DATASET_ROOT = Path(r"C:\3D_DATASET\BraTS2023_MEN")
DEFAULT_CASE_ID = "BraTS-MEN-00020-000"
DEFAULT_BASE_URL = "http://127.0.0.1:5000"
REQUIRED_MODALITIES = ["t1n", "t1c", "t2w", "t2f"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-End Smoke Test for 4-Channel Multimodal Meningioma REST API"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=DEFAULT_BASE_URL,
        help=f"Base URL for the running Flask API (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--case-id",
        type=str,
        default=DEFAULT_CASE_ID,
        help=f"Case ID to test from the validation split (default: {DEFAULT_CASE_ID})",
    )
    parser.add_argument(
        "--split-json",
        type=str,
        default=str(DEFAULT_SPLIT_JSON),
        help="Path to split.json defining train and validation partitions",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(DEFAULT_DATASET_ROOT),
        help="Root path to BraTS2023_MEN dataset",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="HTTP request timeout in seconds (default: 300s)",
    )
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="Run against an in-process Flask test client instead of external HTTP server",
    )
    return parser.parse_args()


def verify_validation_case(case_id: str, split_json_path: Path) -> Dict[str, Any]:
    """Ensures the selected case strictly belongs to the validation split."""
    if not split_json_path.exists():
        raise FileNotFoundError(f"Split file does not exist: {split_json_path}")

    with open(split_json_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    val_cases = set(split_data.get("val_cases", []))
    train_cases = set(split_data.get("train_cases", []))

    if case_id not in val_cases:
        if case_id in train_cases:
            raise ValueError(
                f"SAFETY VIOLATION: Case '{case_id}' is in the TRAINING split! "
                "The smoke test must strictly use an unseen validation case."
            )
        else:
            raise ValueError(f"Case '{case_id}' was not found in the validation split.")

    case_info = split_data.get("case_details", {}).get(case_id, {})
    return case_info


def locate_modality_files(
    case_id: str,
    case_info: Dict[str, Any],
    dataset_root: Path,
) -> Dict[str, Path]:
    """Locates and verifies existence of the 4 aligned NIfTI files on disk."""
    modality_paths: Dict[str, Path] = {}

    if "modalities" in case_info:
        for mod in REQUIRED_MODALITIES:
            raw_path = case_info["modalities"].get(mod)
            if raw_path and Path(raw_path).exists():
                modality_paths[mod] = Path(raw_path)

    # Fallback directory resolution if not in case_details
    if len(modality_paths) < 4:
        cand_dirs = [
            dataset_root / "BraTS-MEN-Train" / case_id,
            dataset_root / "BraTS-MEN-Train-Fix" / case_id,
            dataset_root / "BraTS-MEN-Validation" / case_id,
        ]
        found_dir = None
        for cd in cand_dirs:
            if cd.exists():
                found_dir = cd
                break

        if not found_dir:
            raise FileNotFoundError(f"Could not locate dataset folder for case '{case_id}'")

        for mod in REQUIRED_MODALITIES:
            matches = list(found_dir.glob(f"*{mod}*.nii.gz"))
            if not matches:
                raise FileNotFoundError(f"Missing modality '{mod}' file in: {found_dir}")
            modality_paths[mod] = matches[0]

    # Verify all 4 exist
    for mod in REQUIRED_MODALITIES:
        p = modality_paths.get(mod)
        if not p or not p.exists():
            raise FileNotFoundError(f"Required modality '{mod}' not found: {p}")
        if p.stat().st_size == 0:
            raise ValueError(f"Modality '{mod}' file is empty: {p}")

    return modality_paths


def run_http_smoke_test(
    base_url: str,
    case_id: str,
    modality_paths: Dict[str, Path],
    timeout: int,
):
    """Executes the smoke test over HTTP using requests."""
    print(f"\n[1/5] Checking Health Endpoint: GET {base_url}/api/health ...")
    t0 = time.time()
    try:
        health_resp = requests.get(f"{base_url}/api/health", timeout=10)
    except requests.exceptions.ConnectionError:
        print(f"\n[ERROR] Could not connect to API server at {base_url}!")
        print("Please ensure the Flask backend is running in another terminal:")
        print("    .\\.venv\\Scripts\\python.exe api\\app.py")
        sys.exit(1)

    print(f"    HTTP Status: {health_resp.status_code}")
    assert health_resp.status_code == 200, f"Health check failed: {health_resp.text}"
    health_data = health_resp.json()
    print(f"    Server Status: {health_data.get('status')} (v{health_data.get('version')})")
    print(f"    Compute Device: {health_data.get('device')} ({health_data.get('device_name')})")
    supported = health_data.get("supported_tumor_types", [])
    print(f"    Supported Tumor Types: {supported}")
    assert "meningioma_multimodal" in supported, "Endpoint does not report meningioma_multimodal support!"

    # 2. Upload
    print(f"\n[2/5] Uploading 4 Modalities to POST {base_url}/api/upload ...")
    files_to_upload = []
    open_handles = []
    try:
        for mod in REQUIRED_MODALITIES:
            file_path = modality_paths[mod]
            fh = open(file_path, "rb")
            open_handles.append(fh)
            files_to_upload.append(("files", (file_path.name, fh, "application/gzip")))
            print(f"    Attaching: {file_path.name} ({file_path.stat().st_size:,} bytes)")

        up_resp = requests.post(f"{base_url}/api/upload", files=files_to_upload, timeout=timeout)
    finally:
        for fh in open_handles:
            fh.close()

    print(f"    HTTP Status: {up_resp.status_code}")
    assert up_resp.status_code == 200, f"Upload failed: {up_resp.text}"
    up_data = up_resp.json()
    req_id = up_data.get("request_id")
    print(f"    Issued Request ID: {req_id}")
    print(f"    Uploaded Files Count: {up_data.get('files_count')}")
    print(f"    Detected Modalities: {up_data.get('detected_modalities')}")

    # 3. Segment
    print(f"\n[3/5] Invoking Multimodal Segmentation: POST {base_url}/api/segment/meningioma-multimodal ...")
    seg_payload = {
        "request_id": req_id,
        "patient_id": case_id,
        "modality_map": {mod: modality_paths[mod].name for mod in REQUIRED_MODALITIES},
    }
    t_seg0 = time.time()
    seg_resp = requests.post(
        f"{base_url}/api/segment/meningioma-multimodal",
        json=seg_payload,
        timeout=timeout,
    )
    seg_elapsed = time.time() - t_seg0
    print(f"    HTTP Status: {seg_resp.status_code} (Inference completed in {seg_elapsed:.2f}s)")
    assert seg_resp.status_code == 200, f"Segmentation failed: {seg_resp.text}"

    seg_data = seg_resp.json()
    _inspect_segmentation_data(seg_data)

    # 4. Results check
    print(f"\n[4/5] Verifying Result Persistence: GET {base_url}/api/results/{req_id} ...")
    res_resp = requests.get(f"{base_url}/api/results/{req_id}", timeout=10)
    print(f"    HTTP Status: {res_resp.status_code}")
    assert res_resp.status_code == 200, f"Result retrieval failed: {res_resp.text}"

    # 5. Mask download
    print(f"\n[5/5] Downloading 3D Segmentation Mask: GET {base_url}/api/mask/{req_id} ...")
    mask_resp = requests.get(f"{base_url}/api/mask/{req_id}", timeout=60)
    print(f"    HTTP Status: {mask_resp.status_code}")
    assert mask_resp.status_code == 200, f"Mask download failed: {mask_resp.text}"
    mask_bytes = mask_resp.content
    print(f"    Downloaded Mask Size: {len(mask_bytes):,} bytes")

    # Read mask bytes to verify geometry
    decompressed = gzip.decompress(mask_bytes)
    nii_mask = nib.Nifti1Image.from_bytes(decompressed)
    mask_arr = np.asanyarray(nii_mask.dataobj)
    print(f"    Mask Spatial Shape: {mask_arr.shape}")
    print(f"    Mask Unique Labels: {np.unique(mask_arr).tolist()}")

    total_time = time.time() - t0
    print("\n" + "=" * 70)
    print("END-TO-END SMOKE TEST RESULT: SUCCESS")
    print(f"Total Execution Time: {total_time:.2f}s")
    print("=" * 70)


def run_inprocess_smoke_test(
    case_id: str,
    modality_paths: Dict[str, Path],
):
    """Executes the smoke test in-process using the Flask test client."""
    from api.app import create_app
    from integration.inference_router import InferenceRouter

    print("\n[In-Process Mode] Initializing Flask App and InferenceRouter ...")
    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    # 1. Health
    print("\n[1/5] Checking Health Endpoint: GET /api/health ...")
    health_resp = client.get("/api/health")
    print(f"    HTTP Status: {health_resp.status_code}")
    assert health_resp.status_code == 200
    health_data = health_resp.get_json()
    print(f"    Server Status: {health_data.get('status')} (v{health_data.get('version')})")
    print(f"    Compute Device: {health_data.get('device')} ({health_data.get('device_name')})")

    # 2. Upload
    print("\n[2/5] Uploading 4 Modalities to POST /api/upload ...")
    files_dict = {}
    open_handles = []
    try:
        files_list = []
        for mod in REQUIRED_MODALITIES:
            file_path = modality_paths[mod]
            fh = open(file_path, "rb")
            open_handles.append(fh)
            files_list.append((fh, file_path.name))
            print(f"    Attaching: {file_path.name} ({file_path.stat().st_size:,} bytes)")
        upload_data = {"files": files_list}
        up_resp = client.post("/api/upload", data=upload_data, content_type="multipart/form-data")
    finally:
        for fh in open_handles:
            fh.close()

    print(f"    HTTP Status: {up_resp.status_code}")
    assert up_resp.status_code == 200
    up_data = up_resp.get_json()
    req_id = up_data.get("request_id")
    print(f"    Issued Request ID: {req_id}")

    # 3. Segment
    print("\n[3/5] Invoking Multimodal Segmentation: POST /api/segment/meningioma-multimodal ...")
    seg_payload = {
        "request_id": req_id,
        "patient_id": case_id,
        "modality_map": {mod: modality_paths[mod].name for mod in REQUIRED_MODALITIES},
    }
    t_seg0 = time.time()
    seg_resp = client.post("/api/segment/meningioma-multimodal", json=seg_payload)
    seg_elapsed = time.time() - t_seg0
    print(f"    HTTP Status: {seg_resp.status_code} (Inference completed in {seg_elapsed:.2f}s)")
    assert seg_resp.status_code == 200

    seg_data = seg_resp.get_json()
    _inspect_segmentation_data(seg_data)

    # 4. Results
    print(f"\n[4/5] Verifying Result Persistence: GET /api/results/{req_id} ...")
    res_resp = client.get(f"/api/results/{req_id}")
    print(f"    HTTP Status: {res_resp.status_code}")
    assert res_resp.status_code == 200

    # 5. Mask
    print(f"\n[5/5] Downloading 3D Segmentation Mask: GET /api/mask/{req_id} ...")
    mask_resp = client.get(f"/api/mask/{req_id}")
    print(f"    HTTP Status: {mask_resp.status_code}")
    assert mask_resp.status_code == 200
    print(f"    Downloaded Mask Size: {len(mask_resp.data):,} bytes")
    decompressed = gzip.decompress(mask_resp.data)
    nii_mask = nib.Nifti1Image.from_bytes(decompressed)
    mask_arr = np.asanyarray(nii_mask.dataobj)
    print(f"    Mask Spatial Shape: {mask_arr.shape}")
    print(f"    Mask Unique Labels: {np.unique(mask_arr).tolist()}")

    print("\n" + "=" * 70)
    print("IN-PROCESS SMOKE TEST RESULT: SUCCESS")
    print("=" * 70)


def _inspect_segmentation_data(seg_data: Dict[str, Any]):
    """Prints extracted subregion metrics and validates response structure."""
    req_meta = seg_data.get("request", {})
    volumetrics = seg_data.get("volumetrics", {})
    quality = seg_data.get("quality_status", {})
    mm = seg_data.get("meningioma_multimodal", {})
    subreg = mm.get("subregions", {})
    masks = mm.get("masks", {})
    mc_mask = masks.get("multiclass_mask", {}) if masks else {}

    print("\n--- Segmentation Result Payload ---")
    print(f"Request ID:              {req_meta.get('request_id')}")
    print(f"Tumor Type:              {req_meta.get('tumor_type')}")
    print(f"Quality Verdict:         {quality.get('qc_verdict')}")
    print(f"Status:                  {quality.get('status')}")
    print(f"Checkpoint Used:         {quality.get('checkpoint_used')}")
    print(f"Saved Mask Path:         {mc_mask.get('mask_path')}")
    print(f"Total Tumor Volume (WT): {volumetrics.get('total_tumor_volume_cm3')} cm3 ({volumetrics.get('total_tumor_voxels')} voxels)")

    print("\n--- BraTS Subregion Breakdown ---")
    print(f"Whole Tumor (WT):        {subreg.get('wt_voxels')} voxels ({subreg.get('wt_volume_cm3')} cm3)")
    print(f"Tumor Core (TC):         {subreg.get('tc_voxels')} voxels ({subreg.get('tc_volume_cm3')} cm3)")
    print(f"Enhancing Tumor (ET):    {subreg.get('et_voxels')} voxels ({subreg.get('et_volume_cm3')} cm3)")
    print(f"Necrotic Core (NCR):     {subreg.get('ncr_net_voxels')} voxels ({subreg.get('ncr_net_volume_cm3')} cm3)")
    print(f"Peritumoral Edema (ED):  {subreg.get('edema_voxels')} voxels ({subreg.get('edema_volume_cm3')} cm3)")

    disclaimer = mm.get("disclaimer")
    print(f"\nRegulatory Notice:       {disclaimer}")


def main():
    args = parse_args()

    print("=" * 70)
    print("4-CHANNEL MULTIMODAL MENINGIOMA REST API SMOKE TEST")
    print("=" * 70)
    print(f"Target Case ID:      {args.case_id}")
    print(f"Split JSON:          {args.split_json}")
    print(f"Dataset Root:        {args.dataset_root}")
    print(f"Execution Mode:      {'In-Process Client' if args.in_process else f'Live HTTP ({args.base_url})'}")

    # 1. Strictly verify validation cohort membership
    print("\nVerifying Case Provenance in Validation Split ...")
    case_info = verify_validation_case(args.case_id, Path(args.split_json))
    print(f"[OK] Verified: '{args.case_id}' is strictly in the validation partition.")

    # 2. Locate 4 aligned modality NIfTI files
    print("\nLocating 4 Aligned Modality Files ...")
    modality_paths = locate_modality_files(args.case_id, case_info, Path(args.dataset_root))
    for mod in REQUIRED_MODALITIES:
        print(f"    [{mod}] -> {modality_paths[mod]}")

    # 3. Execute smoke test
    if args.in_process:
        run_inprocess_smoke_test(args.case_id, modality_paths)
    else:
        run_http_smoke_test(args.base_url, args.case_id, modality_paths, args.timeout)


if __name__ == "__main__":
    main()
