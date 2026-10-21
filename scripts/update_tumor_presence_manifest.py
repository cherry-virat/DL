"""
Update Healthy-Control Entries in Tumor Presence Manifest after HD-BET QC.
===========================================================================

Safeguards:
1. Only includes IXI subjects that pass rigorous HD-BET and brain-mask QC checks.
2. Verifies physical existence, header geometry, and non-zero size for processed T1, T2, and brain masks.
3. Preserves all subject IDs, anonymized IDs, group keys, and locked split assignments.
4. Preserves locked test set membership and checksums in split.json.
5. Strictly ensures tumor cases remain class 1 (tumor_present).
"""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(r"C:\Users\nchan\Desktop\DL")
MANIFEST_PATH = PROJECT_ROOT / "data" / "tumor_presence" / "manifest.json"
SPLIT_PATH = PROJECT_ROOT / "data" / "tumor_presence" / "split.json"
PROCESSED_DIR = Path(r"C:\3D_DATASET\HealthyControls\IXI\processed")


def inspect_nifti_header(p: Path) -> Dict[str, Any]:
    """Inspects NIfTI header and returns standardized metadata dictionary."""
    img = nib.load(str(p))
    hdr = img.header
    shape = [int(x) for x in hdr.get_data_shape()[:3]]
    zooms = [float(round(z, 4)) for z in hdr.get_zooms()[:3]]
    aff = [[float(round(c, 6)) for c in row] for row in np.array(img.affine, dtype=np.float64)]
    return {
        "file_path": str(p),
        "file_size_bytes": p.stat().st_size,
        "readable": True,
        "shape": shape,
        "voxel_spacing_mm": zooms,
        "affine": aff,
        "sha256": None,
        "error": None,
    }


def audit_processed_ixi_cohort() -> Tuple[Dict[str, Dict[str, Any]], List[str], List[str]]:
    """
    Audits all metadata and NIfTI files in C:\\3D_DATASET\\HealthyControls\\IXI\\processed.
    Returns:
        valid_cases: Dict mapping participant_id -> audit record
        flagged_cases: List of participant IDs flagged by plausibility check
        failed_cases: List of participant IDs failing geometry or missing files
    """
    meta_dir = PROCESSED_DIR / "metadata"
    t1_dir = PROCESSED_DIR / "T1"
    t2_dir = PROCESSED_DIR / "T2"
    mask_dir = PROCESSED_DIR / "masks"

    valid_cases: Dict[str, Dict[str, Any]] = {}
    flagged_cases: Dict[str, Dict[str, Any]] = {}
    failed_cases: Dict[str, Dict[str, Any]] = {}

    if not meta_dir.exists():
        return valid_cases, flagged_cases, failed_cases

    meta_files = sorted(list(meta_dir.glob("*.json")))
    for mf in meta_files:
        try:
            with open(mf, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            failed_cases[mf.name] = {"error": f"JSON parse error: {e}"}
            continue

        sub_id = meta.get("participant_id")
        if not sub_id:
            failed_cases[mf.name] = {"error": "Missing participant_id"}
            continue

        status = meta.get("status")
        plaus = meta.get("plausibility_audit", {})
        is_plausible = plaus.get("is_skull_stripped_plausible", False)

        if status == "FAILED":
            failed_cases[sub_id] = {
                "status": "FAILED",
                "error": meta.get("error", "Unknown error"),
                "timestamp": meta.get("timestamp"),
            }
            continue

        t1_p = t1_dir / f"{sub_id}_t1_skull_stripped.nii.gz"
        t2_p = t2_dir / f"{sub_id}_t2_skull_stripped.nii.gz"
        mask_p = mask_dir / f"{sub_id}_mask.nii.gz"

        # Check file existence and non-zero size
        if not (t1_p.exists() and t2_p.exists() and mask_p.exists()):
            failed_cases[sub_id] = {"error": "Missing processed NIfTI files on disk"}
            continue
        if min(t1_p.stat().st_size, t2_p.stat().st_size, mask_p.stat().st_size) == 0:
            failed_cases[sub_id] = {"error": "Zero-byte NIfTI file detected"}
            continue

        # Check geometry alignment across all three NIfTI files
        try:
            nii_t1 = nib.load(str(t1_p))
            nii_t2 = nib.load(str(t2_p))
            nii_m = nib.load(str(mask_p))

            s1 = tuple(int(x) for x in nii_t1.header.get_data_shape()[:3])
            s2 = tuple(int(x) for x in nii_t2.header.get_data_shape()[:3])
            sm = tuple(int(x) for x in nii_m.header.get_data_shape()[:3])
            if not (s1 == s2 == sm):
                failed_cases[sub_id] = {"error": f"Shape mismatch: {s1} vs {s2} vs {sm}"}
                continue

            if not (np.allclose(nii_t1.affine, nii_t2.affine, atol=1e-3) and
                    np.allclose(nii_t1.affine, nii_m.affine, atol=1e-3)):
                failed_cases[sub_id] = {"error": "Affine alignment mismatch"}
                continue

            # Check binary mask values
            m_arr = np.asanyarray(nii_m.dataobj, dtype=np.uint8)
            u_vals = set(np.unique(m_arr))
            if not u_vals.issubset({0, 1}):
                failed_cases[sub_id] = {"error": f"Non-binary mask values: {u_vals}"}
                continue

        except Exception as exc:
            failed_cases[sub_id] = {"error": f"Failed reading NIfTI: {exc}"}
            continue

        if status == "PASSED" and is_plausible:
            valid_cases[sub_id] = {
                "participant_id": sub_id,
                "t1_path": t1_p,
                "t2_path": t2_p,
                "mask_path": mask_p,
                "meta": meta,
                "t1_header": inspect_nifti_header(t1_p),
                "t2_header": inspect_nifti_header(t2_p),
            }
        elif status == "FLAGGED" or not is_plausible:
            flagged_cases[sub_id] = {
                "status": "FLAGGED",
                "reasons": plaus.get("failure_reasons", []),
                "meta": meta,
            }
        else:
            failed_cases[sub_id] = {
                "status": status,
                "error": meta.get("error", f"Unrecognized status: {status}"),
            }

    return valid_cases, flagged_cases, failed_cases


def update_manifest():
    print("[*] Auditing processed IXI cohort...")
    valid_ixi, flagged_ixi, failed_ixi = audit_processed_ixi_cohort()
    print(f"    - Eligible QC-passed processed cases: {len(valid_ixi)}")
    print(f"    - Flagged cases: {len(flagged_ixi)}")
    print(f"    - Failed cases: {len(failed_ixi)}")

    # Load existing manifest
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Load existing split
    with open(SPLIT_PATH, "r", encoding="utf-8") as f:
        split = json.load(f)

    original_total = len(manifest["cases"])
    updated_healthy_count = 0
    pending_healthy_count = 0
    flagged_healthy_count = 0
    failed_healthy_count = 0
    excluded_healthy_count = 0
    tumor_count = 0

    for case in manifest["cases"]:
        source = case.get("source_cohort")
        group_key = case.get("subject_group_key")

        if source == "ixi":
            # Safeguard: Ensure healthy cases are strictly class 0
            assert case["class_id"] == 0, f"Healthy case {group_key} has invalid class_id {case['class_id']}"
            assert case["class_label"] == "healthy_control"

            if group_key in valid_ixi:
                v = valid_ixi[group_key]
                case["modalities"]["t1"] = str(v["t1_path"])
                case["modalities"]["t2"] = str(v["t2_path"])
                case["mask_path"] = str(v["mask_path"])
                case["geometry_status"] = "aligned"
                case["is_skull_stripped"] = True
                case["qc_status"] = "PASSED"
                case["qc_metadata"] = {
                    "status": "PASSED",
                    "nonzero_volume_cc": v["meta"].get("plausibility_audit", {}).get("nonzero_volume_cc"),
                    "outer_boundary_voxels": v["meta"].get("plausibility_audit", {}).get("outer_boundary_voxels"),
                    "registration_time_seconds": v["meta"].get("registration_time_seconds"),
                    "hdbet_time_seconds": v["meta"].get("hdbet_time_seconds"),
                    "processed_at": v["meta"].get("processed_at"),
                }
                case["modalities_metadata"]["t1"] = v["t1_header"]
                case["modalities_metadata"]["t2"] = v["t2_header"]
                updated_healthy_count += 1
            elif group_key in flagged_ixi:
                case["geometry_status"] = "flagged_qc"
                case["is_skull_stripped"] = False
                case["qc_status"] = "FLAGGED"
                case["qc_metadata"] = flagged_ixi[group_key]
                flagged_healthy_count += 1
            elif group_key in failed_ixi:
                case["geometry_status"] = "failed_qc"
                case["is_skull_stripped"] = False
                case["qc_status"] = "FAILED"
                case["qc_metadata"] = failed_ixi[group_key]
                failed_healthy_count += 1
            else:
                if not case.get("is_valid", False):
                    case["qc_status"] = "EXCLUDED"
                    case["is_skull_stripped"] = False
                    excluded_healthy_count += 1
                else:
                    case["qc_status"] = "PENDING_PREPROCESSING"
                    case["is_skull_stripped"] = False
                    pending_healthy_count += 1
        elif source in ("brats_glioma", "brats_meningioma"):
            # Safeguard: Tumor cases must strictly remain class 1
            assert case["class_id"] == 1, f"Tumor case {group_key} must be class 1"
            assert case["class_label"] == "tumor_present"
            case["qc_status"] = "PASSED"
            case["is_skull_stripped"] = True
            tumor_count += 1

    print(f"[*] Manifest update summary:")
    print(f"    - Processed QC-passed healthy cases updated: {updated_healthy_count}")
    print(f"    - Flagged healthy cases: {flagged_healthy_count}")
    print(f"    - Failed healthy cases: {failed_healthy_count}")
    print(f"    - Pending healthy cases preserved: {pending_healthy_count}")
    print(f"    - Excluded healthy cases preserved: {excluded_healthy_count}")
    print(f"    - Tumor cases preserved (class 1): {tumor_count}")
    print(f"    - Total manifest cases: {len(manifest['cases'])} (original: {original_total})")

    # Update manifest metadata
    meta = manifest["metadata"]
    meta["healthy_qc_passed_cases"] = updated_healthy_count
    meta["healthy_flagged_cases"] = flagged_healthy_count
    meta["healthy_failed_cases"] = failed_healthy_count
    meta["healthy_pending_preprocessing_cases"] = pending_healthy_count
    meta["healthy_excluded_cases"] = excluded_healthy_count
    meta["last_manifest_qc_update"] = datetime.now(timezone.utc).isoformat()

    # Create backup before writing
    backup_path = MANIFEST_PATH.with_suffix(".json.bak")
    shutil.copy2(MANIFEST_PATH, backup_path)
    print(f"[*] Created manifest backup at: {backup_path}")

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[*] Successfully saved updated manifest to: {MANIFEST_PATH}")

    # Verify split integrity and non-overlap
    print("[*] Verifying split integrity against updated manifest...")
    all_anon_ids = {c["anonymized_id"]: c for c in manifest["cases"]}
    all_group_keys = {c["anonymized_id"]: c["subject_group_key"] for c in manifest["cases"]}

    train_ids = split["train_cases"]
    val_ids = split["val_cases"]
    test_ids = split["test_cases"]

    for cid in train_ids + val_ids + test_ids:
        assert cid in all_anon_ids, f"Split ID {cid} missing from manifest!"

    s_tr, s_va, s_te = set(train_ids), set(val_ids), set(test_ids)
    assert len(s_tr & s_va) == 0, "Train and Val overlap detected!"
    assert len(s_tr & s_te) == 0, "Train and Test overlap detected!"
    assert len(s_va & s_te) == 0, "Val and Test overlap detected!"

    # Verify zero subject_group_key leakage
    keys_tr = {all_group_keys[cid] for cid in train_ids}
    keys_va = {all_group_keys[cid] for cid in val_ids}
    keys_te = {all_group_keys[cid] for cid in test_ids}
    assert len(keys_tr & keys_va) == 0, "Subject group key overlap between Train and Val!"
    assert len(keys_tr & keys_te) == 0, "Subject group key overlap between Train and Test!"
    assert len(keys_va & keys_te) == 0, "Subject group key overlap between Val and Test!"

    # Verify split checksums remain identical
    import hashlib
    def list_hash(lst: List[str]) -> str:
        return hashlib.sha256("\n".join(sorted(lst)).encode("utf-8")).hexdigest()

    saved_chk = split["metadata"]["split_checksums"]
    assert list_hash(train_ids) == saved_chk["train_cases_sha256"], "Train split checksum mismatch!"
    assert list_hash(val_ids) == saved_chk["val_cases_sha256"], "Val split checksum mismatch!"
    assert list_hash(test_ids) == saved_chk["test_cases_sha256"], "Locked test split checksum mismatch!"

    print("[*] All split integrity assertions verified: 100% disjoint, zero leakage, locked test set intact.")


if __name__ == "__main__":
    update_manifest()
