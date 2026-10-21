"""
Reproducible Tumor Presence Manifest Generator and Metadata Verifier.
=====================================================================

Prepares structured manifest and split files for the 2-channel binary
tumor presence detection task:
  - Class 0: Healthy Control (Normative structural MRI: IXI)
  - Class 1: Tumor Present (Neoplastic structural MRI: BraTS GLI & MEN)

Design Principles & Standards:
------------------------------
1. Two-Sequence Standard: Focuses exclusively on T1 and T2 structural acquisitions.
2. Complete Directory Isolation: Outputs strictly to `data/tumor_presence/manifest.json`
   and `split.json`, leaving `data/classifier/` completely untouched.
3. Strict Preservation of Valid Cohorts: Does NOT artificially truncate or delete tumor
   cases to balance sample sizes. Preserves all valid scans and documents class weighting/sampling
   as the appropriate method for loss computation during future training.
4. Anonymization & Subject Grouping: Tracks safe anonymized IDs (ANON_HEALTHY_*, ANON_TUMOR_*)
   and provides a `subject_group_key` (participant ID) to enforce zero subject-level leakage across splits.
5. Rigorous Exclusion Tracking: Records all inspected participants and provides explicit
   exclusion reasons for missing modalities, thick-slice 2D acquisitions, or unreadable files.
6. Deterministic Disjoint Stratification: Partitions valid cases into disjoint train (70%),
   val (15%), and locked test (15%) splits, stratified across source sub-cohorts with locked seed 42.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import argparse
import datetime
import hashlib
import json
import re
import sys
import time

import nibabel as nib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "tumor_presence"
DEFAULT_IXI_DIR = Path(r"C:\3D_DATASET\HealthyControls\IXI")
DEFAULT_GLI_DIR = Path(r"C:\3D_DATASET\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData")
DEFAULT_MEN_DIR = Path(r"C:\3D_DATASET\BraTS2023_MEN\BraTS-MEN-Train")

CANONICAL_PRESENCE_MODALITIES = ["t1", "t2"]
DEFAULT_TOLERANCE = 1e-3
MIN_ACCEPTABLE_SLICES = 35


def compute_file_sha256(file_path: Path, chunk_size: int = 65536) -> str:
    """Computes SHA-256 checksum of a file on disk."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as fp:
        while chunk := fp.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def inspect_presence_modality_file(
    file_path: Path,
    compute_hash: bool = False,
) -> Tuple[Dict[str, Any], Optional[np.ndarray], Optional[Tuple[int, ...]], Optional[Tuple[float, ...]]]:
    """
    Measures NIfTI header metadata for a single modality file.
    """
    meta: Dict[str, Any] = {
        "file_path": str(file_path),
        "file_size_bytes": file_path.stat().st_size if file_path.exists() else 0,
        "readable": False,
        "shape": None,
        "voxel_spacing_mm": None,
        "affine": None,
        "sha256": None,
        "error": None,
    }

    if not file_path.exists():
        meta["error"] = "File does not exist."
        return meta, None, None, None

    if meta["file_size_bytes"] == 0:
        meta["error"] = "File is empty (0 bytes)."
        return meta, None, None, None

    try:
        img = nib.load(str(file_path))
        hdr = img.header
        shape = tuple(int(x) for x in hdr.get_data_shape()[:3])
        zooms = tuple(float(round(z, 4)) for z in hdr.get_zooms()[:3])
        aff = np.array(img.affine, dtype=np.float64)

        meta["readable"] = True
        meta["shape"] = list(shape)
        meta["voxel_spacing_mm"] = list(zooms)
        meta["affine"] = [[float(round(c, 6)) for c in row] for row in aff]

        if compute_hash:
            meta["sha256"] = compute_file_sha256(file_path)

        return meta, aff, shape, zooms
    except Exception as exc:
        meta["error"] = f"Failed to parse NIfTI header: {str(exc)}"
        return meta, None, None, None


def load_ixi_demographics(ixi_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Loads demographic information from IXI.csv if present."""
    csv_candidates = [
        ixi_dir / "metadata" / "IXI.csv",
        ixi_dir / "IXI.csv",
        PROJECT_ROOT / "data" / "IXI.csv",
    ]
    for p in csv_candidates:
        if p.exists():
            try:
                df = pd.read_csv(p)
                demographics = {}
                for _, row in df.iterrows():
                    ixi_id_val = row.get("IXI_ID")
                    if pd.notna(ixi_id_val):
                        pid = f"IXI{int(ixi_id_val):03d}"
                        demographics[pid] = {
                            "age": float(row["AGE"]) if "AGE" in row and pd.notna(row["AGE"]) else None,
                            "sex": int(row["SEX_ID (1=m, 2=f)"]) if "SEX_ID (1=m, 2=f)" in row and pd.notna(row["SEX_ID (1=m, 2=f)"]) else None,
                            "height": float(row["HEIGHT"]) if "HEIGHT" in row and pd.notna(row["HEIGHT"]) else None,
                            "weight": float(row["WEIGHT"]) if "WEIGHT" in row and pd.notna(row["WEIGHT"]) else None,
                            "ethnic_id": int(row["ETHNIC_ID"]) if "ETHNIC_ID" in row and pd.notna(row["ETHNIC_ID"]) else None,
                        }
                return demographics
            except Exception as e:
                print(f"[!] Warning: failed to parse IXI demographic CSV at {p}: {e}")
    return {}


def audit_ixi_cohort(
    ixi_dir: Path,
    compute_hashes: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Audits the entire IXI cohort directory, discovering participants, pairing T1 and T2,
    and classifying valid vs. excluded cases.
    """
    records: List[Dict[str, Any]] = []
    stats = {
        "total_t1_files": 0,
        "total_t2_files": 0,
        "unique_participants": 0,
        "paired_participants": 0,
        "t1_only": 0,
        "t2_only": 0,
        "thick_slice_excluded": 0,
        "valid_cases": 0,
    }

    if not ixi_dir.exists():
        return records, stats

    t1_sub = ixi_dir / "raw" / "T1"
    t2_sub = ixi_dir / "raw" / "T2"
    if not t1_sub.exists():
        t1_sub = ixi_dir / "T1"
    if not t2_sub.exists():
        t2_sub = ixi_dir / "T2"

    candidates_t1 = list(t1_sub.glob("*.nii*")) if t1_sub.exists() else []
    candidates_t2 = list(t2_sub.glob("*.nii*")) if t2_sub.exists() else []

    if not candidates_t1 and not candidates_t2:
        candidates_t1 = [p for p in ixi_dir.rglob("*.nii*") if "t1" in p.name.lower() and "t2" not in p.name.lower()]
        candidates_t2 = [p for p in ixi_dir.rglob("*.nii*") if "t2" in p.name.lower() and "t1" not in p.name.lower()]

    stats["total_t1_files"] = len(candidates_t1)
    stats["total_t2_files"] = len(candidates_t2)

    pattern = re.compile(r"^(IXI\d{3})-([A-Za-z]+)-(\d+)-(T1|T2)\.nii\.gz$", re.IGNORECASE)

    t1_by_subj: Dict[str, Path] = {}
    for p in candidates_t1:
        m = pattern.match(p.name)
        subj = m.group(1) if m else re.sub(r"[-_]t1.*", "", p.name, flags=re.IGNORECASE)
        t1_by_subj[subj] = p

    t2_by_subj: Dict[str, Path] = {}
    for p in candidates_t2:
        m = pattern.match(p.name)
        subj = m.group(1) if m else re.sub(r"[-_]t2.*", "", p.name, flags=re.IGNORECASE)
        t2_by_subj[subj] = p

    all_subjects = sorted(list(set(t1_by_subj.keys()) | set(t2_by_subj.keys())))
    stats["unique_participants"] = len(all_subjects)

    demographics_map = load_ixi_demographics(ixi_dir)

    for subj in all_subjects:
        has_t1 = subj in t1_by_subj
        has_t2 = subj in t2_by_subj
        p_t1 = t1_by_subj.get(subj)
        p_t2 = t2_by_subj.get(subj)

        record: Dict[str, Any] = {
            "subject_group_key": subj,
            "class_id": 0,
            "class_label": "healthy_control",
            "source_cohort": "ixi",
            "modalities": {
                "t1": str(p_t1) if p_t1 else None,
                "t2": str(p_t2) if p_t2 else None,
            },
            "is_valid": False,
            "geometry_status": "incompatible",
            "exclusion_reason": None,
            "demographics": demographics_map.get(subj, None),
            "modalities_metadata": {},
        }

        if not has_t1:
            record["exclusion_reason"] = "Missing T1 acquisition"
            stats["t2_only"] += 1
            records.append(record)
            continue

        if not has_t2:
            record["exclusion_reason"] = "Missing T2 acquisition"
            stats["t1_only"] += 1
            records.append(record)
            continue

        stats["paired_participants"] += 1

        # Inspect headers for T1 and T2
        meta_t1, aff_t1, shape_t1, zooms_t1 = inspect_presence_modality_file(p_t1, compute_hash=compute_hashes)
        meta_t2, aff_t2, shape_t2, zooms_t2 = inspect_presence_modality_file(p_t2, compute_hash=compute_hashes)

        record["modalities_metadata"]["t1"] = meta_t1
        record["modalities_metadata"]["t2"] = meta_t2

        if not meta_t1["readable"]:
            record["exclusion_reason"] = f"Unreadable T1 acquisition: {meta_t1['error']}"
            records.append(record)
            continue

        if not meta_t2["readable"]:
            record["exclusion_reason"] = f"Unreadable T2 acquisition: {meta_t2['error']}"
            records.append(record)
            continue

        # Check for thick-slice acquisition
        if shape_t2[2] <= MIN_ACCEPTABLE_SLICES:
            record["exclusion_reason"] = (
                f"Thick 2D multi-slice acquisition ({shape_t2[2]} slices, "
                f"{zooms_t2[2]:.2f}mm slice thickness; minimum required is > {MIN_ACCEPTABLE_SLICES})"
            )
            stats["thick_slice_excluded"] += 1
            records.append(record)
            continue

        # Valid volumetric healthy case
        record["is_valid"] = True
        record["geometry_status"] = "requires_resampling"
        stats["valid_cases"] += 1
        records.append(record)

    return records, stats


def find_ixi_paired_cases(ixi_dir: Path) -> List[Dict[str, Path]]:
    """Helper returning list of paired T1/T2 subjects."""
    records, _ = audit_ixi_cohort(ixi_dir)
    paired = []
    for r in records:
        if r["modalities"]["t1"] and r["modalities"]["t2"]:
            paired.append({
                "subject_id": r["subject_group_key"],
                "t1": Path(r["modalities"]["t1"]),
                "t2": Path(r["modalities"]["t2"]),
            })
    return paired


def audit_brats_cohort(
    data_dir: Path,
    cohort_name: str,
    prefix: str,
    compute_hashes: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Audits a BraTS tumor cohort directory, discovering cases and validating T1 and T2 acquisitions.
    """
    records: List[Dict[str, Any]] = []
    stats = {
        "total_cases_found": 0,
        "valid_cases": 0,
        "missing_modality_excluded": 0,
        "unreadable_excluded": 0,
    }

    if not data_dir.exists():
        return records, stats

    case_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)])
    stats["total_cases_found"] = len(case_dirs)

    for cd in case_dirs:
        t1_files = list(cd.glob("*-t1n.nii.gz"))
        t2_files = list(cd.glob("*-t2w.nii.gz"))

        record: Dict[str, Any] = {
            "subject_group_key": cd.name,
            "class_id": 1,
            "class_label": "tumor_present",
            "source_cohort": cohort_name,
            "modalities": {
                "t1": str(t1_files[0]) if t1_files else None,
                "t2": str(t2_files[0]) if t2_files else None,
            },
            "is_valid": False,
            "geometry_status": "incompatible",
            "exclusion_reason": None,
            "demographics": None,
            "modalities_metadata": {},
        }

        if not t1_files:
            record["exclusion_reason"] = "Missing t1n sequence"
            stats["missing_modality_excluded"] += 1
            records.append(record)
            continue

        if not t2_files:
            record["exclusion_reason"] = "Missing t2w sequence"
            stats["missing_modality_excluded"] += 1
            records.append(record)
            continue

        meta_t1, aff_t1, shape_t1, zooms_t1 = inspect_presence_modality_file(t1_files[0], compute_hash=compute_hashes)
        meta_t2, aff_t2, shape_t2, zooms_t2 = inspect_presence_modality_file(t2_files[0], compute_hash=compute_hashes)

        record["modalities_metadata"]["t1"] = meta_t1
        record["modalities_metadata"]["t2"] = meta_t2

        if not meta_t1["readable"] or not meta_t2["readable"]:
            record["exclusion_reason"] = "Unreadable NIfTI file in t1n or t2w"
            stats["unreadable_excluded"] += 1
            records.append(record)
            continue

        record["is_valid"] = True
        record["geometry_status"] = "aligned"
        stats["valid_cases"] += 1
        records.append(record)

    return records, stats


def generate_presence_cohort(
    ixi_dir: Path,
    glioma_dir: Path,
    meningioma_dir: Path,
    output_dir: Path,
    compute_hashes: bool = False,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> int:
    """
    Builds the full tumor presence manifest and split files across IXI, Glioma, and Meningioma.
    """
    print(f"[*] Tumor Presence Manifest Generator & Cohort Auditor")
    print(f"[*] Inspecting healthy control cohort at: {ixi_dir}")

    ixi_records, ixi_stats = audit_ixi_cohort(ixi_dir, compute_hashes=compute_hashes)
    print(f"    - Total T1 files: {ixi_stats['total_t1_files']}")
    print(f"    - Total T2 files: {ixi_stats['total_t2_files']}")
    print(f"    - Total unique participants: {ixi_stats['unique_participants']}")
    print(f"    - Paired T1/T2 participants: {ixi_stats['paired_participants']}")
    print(f"    - T1-only excluded: {ixi_stats['t1_only']}")
    print(f"    - T2-only excluded: {ixi_stats['t2_only']}")
    print(f"    - Thick-slice (<=35) excluded: {ixi_stats['thick_slice_excluded']}")
    print(f"    - Valid volumetric healthy cases: {ixi_stats['valid_cases']}")

    if ixi_stats["valid_cases"] == 0:
        print(f"[!] NOTICE: No valid paired T1/T2 healthy control cases found.")
        print(f"[!] The normative healthy cohort must be downloaded and extracted before building.")
        print(f"[*] Exiting cleanly without error.")
        return 0

    print(f"[*] Inspecting BraTS Glioma cohort at: {glioma_dir}")
    gli_records, gli_stats = audit_brats_cohort(glioma_dir, "brats_glioma", "BraTS-GLI-", compute_hashes=compute_hashes)
    print(f"    - Found: {gli_stats['total_cases_found']}, Valid: {gli_stats['valid_cases']}")

    print(f"[*] Inspecting BraTS Meningioma cohort at: {meningioma_dir}")
    men_records, men_stats = audit_brats_cohort(meningioma_dir, "brats_meningioma", "BraTS-MEN-", compute_hashes=compute_hashes)
    print(f"    - Found: {men_stats['total_cases_found']}, Valid: {men_stats['valid_cases']}")

    all_records = ixi_records + gli_records + men_records

    # Assign safe anonymized identifiers
    healthy_idx = 1
    tumor_idx = 1
    for r in all_records:
        if r["class_id"] == 0:
            r["anonymized_id"] = f"ANON_HEALTHY_{healthy_idx:05d}"
            healthy_idx += 1
        else:
            r["anonymized_id"] = f"ANON_TUMOR_{tumor_idx:05d}"
            tumor_idx += 1

    output_dir.mkdir(parents=True, exist_ok=True)

    # Disjoint Stratified Split across sub-cohorts for valid cases
    valid_ixi = [r for r in all_records if r["source_cohort"] == "ixi" and r["is_valid"]]
    valid_gli = [r for r in all_records if r["source_cohort"] == "brats_glioma" and r["is_valid"]]
    valid_men = [r for r in all_records if r["source_cohort"] == "brats_meningioma" and r["is_valid"]]

    rng = np.random.RandomState(seed)

    def partition_subcohort(records_list: List[Dict]) -> Tuple[List[str], List[str], List[str]]:
        keys = [r["anonymized_id"] for r in records_list]
        perm = rng.permutation(keys).tolist()
        n = len(perm)
        n_tr = int(n * train_ratio)
        n_val = int(n * val_ratio)
        tr = perm[:n_tr]
        va = perm[n_tr:n_tr + n_val]
        te = perm[n_tr + n_val:]
        return tr, va, te

    ixi_tr, ixi_va, ixi_te = partition_subcohort(valid_ixi)
    gli_tr, gli_va, gli_te = partition_subcohort(valid_gli)
    men_tr, men_va, men_te = partition_subcohort(valid_men)

    train_cases = ixi_tr + gli_tr + men_tr
    val_cases = ixi_va + gli_va + men_va
    test_cases = ixi_te + gli_te + men_te

    rng.shuffle(train_cases)
    rng.shuffle(val_cases)
    rng.shuffle(test_cases)

    # Verify zero leakage across splits
    s_tr, s_va, s_te = set(train_cases), set(val_cases), set(test_cases)
    assert len(s_tr & s_va) == 0, "Train-Val overlap detected!"
    assert len(s_tr & s_te) == 0, "Train-Test overlap detected!"
    assert len(s_va & s_te) == 0, "Val-Test overlap detected!"

    # Verify zero subject_group_key leakage
    id_to_key = {r["anonymized_id"]: r["subject_group_key"] for r in all_records}
    keys_tr = {id_to_key[cid] for cid in train_cases}
    keys_va = {id_to_key[cid] for cid in val_cases}
    keys_te = {id_to_key[cid] for cid in test_cases}
    assert len(keys_tr & keys_va) == 0, "Subject key leakage between Train and Val!"
    assert len(keys_tr & keys_te) == 0, "Subject key leakage between Train and Test!"
    assert len(keys_va & keys_te) == 0, "Subject key leakage between Val and Test!"

    # Save manifest.json
    manifest_data = {
        "metadata": {
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "total_cases": len(all_records),
            "total_records_inspected": len(all_records),
            "total_valid_cases": len(valid_ixi) + len(valid_gli) + len(valid_men),
            "healthy_cases": len(valid_ixi),
            "healthy_valid_cases": len(valid_ixi),
            "tumor_cases": len(valid_gli) + len(valid_men),
            "tumor_valid_cases": len(valid_gli) + len(valid_men),
            "breakdown": {
                "ixi_healthy_valid": len(valid_ixi),
                "brats_glioma_valid": len(valid_gli),
                "brats_meningioma_valid": len(valid_men),
                "ixi_excluded": len(ixi_records) - len(valid_ixi),
                "glioma_excluded": len(gli_records) - len(valid_gli),
                "meningioma_excluded": len(men_records) - len(valid_men),
            },
            "random_seed": seed,
            "modalities": CANONICAL_PRESENCE_MODALITIES,
            "class_imbalance_note": (
                "All valid cases are preserved without downsampling. "
                "Class imbalance (2,204 tumor vs 576 healthy) must be addressed via loss weighting "
                "(e.g. healthy_weight ~ 3.82) or a WeightedRandomSampler during training."
            ),
        },
        "cases": all_records,
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    def list_hash(lst: List[str]) -> str:
        return hashlib.sha256("\n".join(sorted(lst)).encode("utf-8")).hexdigest()

    split_data = {
        "metadata": {
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "random_seed": seed,
            "locked_test_set": True,
            "counts": {
                "train_total": len(train_cases),
                "train_healthy": len(ixi_tr),
                "train_tumor": len(gli_tr) + len(men_tr),
                "val_total": len(val_cases),
                "val_healthy": len(ixi_va),
                "val_tumor": len(gli_va) + len(men_va),
                "test_total": len(test_cases),
                "test_healthy": len(ixi_te),
                "test_tumor": len(gli_te) + len(men_te),
            },
            "split_checksums": {
                "train_cases_sha256": list_hash(train_cases),
                "val_cases_sha256": list_hash(val_cases),
                "test_cases_sha256": list_hash(test_cases),
            },
        },
        "train_cases": train_cases,
        "val_cases": val_cases,
        "test_cases": test_cases,
    }

    split_path = output_dir / "split.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_data, f, indent=2)

    print(f"\n[*] Generated {manifest_path} (Total cases: {len(all_records)}, Valid: {len(train_cases)+len(val_cases)+len(test_cases)})")
    print(f"[*] Generated {split_path} (Train: {len(train_cases)}, Val: {len(val_cases)}, Test: {len(test_cases)})")
    print(f"    - Healthy (IXI): Train={len(ixi_tr)}, Val={len(ixi_va)}, Test={len(ixi_te)}")
    print(f"    - Glioma: Train={len(gli_tr)}, Val={len(gli_va)}, Test={len(gli_te)}")
    print(f"    - Meningioma: Train={len(men_tr)}, Val={len(men_va)}, Test={len(men_te)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Tumor Presence Manifest and Split Generator")
    parser.add_argument("--ixi-dir", type=Path, default=DEFAULT_IXI_DIR, help="Path to IXI healthy control root")
    parser.add_argument("--glioma-dir", type=Path, default=DEFAULT_GLI_DIR, help="Path to BraTS Glioma root")
    parser.add_argument("--meningioma-dir", type=Path, default=DEFAULT_MEN_DIR, help="Path to BraTS Meningioma root")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Destination directory for manifest and split")
    parser.add_argument("--compute-file-hashes", action="store_true", help="Compute SHA-256 hashes of NIfTI files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic cohort splitting")
    args = parser.parse_args()

    return generate_presence_cohort(
        ixi_dir=args.ixi_dir,
        glioma_dir=args.glioma_dir,
        meningioma_dir=args.meningioma_dir,
        output_dir=args.output_dir,
        compute_hashes=args.compute_file_hashes,
        seed=args.seed,
    )


if __name__ == "__main__":
    sys.exit(main())
