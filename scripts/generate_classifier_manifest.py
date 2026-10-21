"""
Reproducible Classifier Manifest Generator and NIfTI Metadata Verifier.
======================================================================

Reads raw MRI acquisitions defined in the classifier manifest, measures actual
NIfTI header dimensions, voxel spacing, and spatial affine transformation matrices,
and validates 4-sequence coregistration and file integrity.

Key Capabilities:
-----------------
1. Measures actual NIfTI headers (shape, voxel spacing zooms, affine matrix) per modality.
2. Assesses inter-modality spatial consistency (shape match, affine alignment within tolerance).
3. Distinguishes per-modality file sizes and readability status.
4. Preserves anonymized IDs (ANON_GLI_*, ANON_MEN_*) and split partition assignments.
5. Provides an optional `--compute-file-hashes` flag to calculate raw file SHA-256 on demand,
   avoiding ~22.5 GB of default disk I/O when not requested.
6. Clearly documents that checksums in `split.json` are split-list integrity hashes,
   not raw MRI file hashes.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import argparse
import datetime
import hashlib
import json
import sys
import time

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "classifier" / "manifest.json"
CANONICAL_MODALITIES = ["t1n", "t1c", "t2w", "t2f"]
DEFAULT_TOLERANCE = 1e-3


def compute_file_sha256(file_path: Path, chunk_size: int = 65536) -> str:
    """Computes SHA-256 checksum of a file on disk."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as fp:
        while chunk := fp.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def inspect_modality_file(
    file_path: Path,
    compute_hash: bool = False,
) -> Tuple[Dict[str, Any], Optional[np.ndarray], Optional[Tuple[int, ...]], Optional[Tuple[float, ...]]]:
    """
    Measures actual NIfTI header metadata for a single modality file.
    Returns:
        metadata_dict: Dict with shape, spacing, affine, size, readable status, and optional hash.
        affine: (4, 4) numpy array if readable.
        shape: Tuple of spatial dimensions if readable.
        zooms: Tuple of voxel spacing if readable.
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
        shape = tuple(int(x) for x in hdr.get_data_shape())
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


def verify_case_modalities(
    case_record: Dict[str, Any],
    compute_hash: bool = False,
    tolerance: float = DEFAULT_TOLERANCE,
) -> Dict[str, Any]:
    """
    Verifies all 4 modalities for a given case and compiles enriched metadata.
    """
    modalities = case_record["modalities"]
    modality_meta: Dict[str, Any] = {}

    ref_shape: Optional[Tuple[int, ...]] = None
    ref_aff: Optional[np.ndarray] = None
    ref_zooms: Optional[Tuple[float, ...]] = None
    ref_mod: Optional[str] = None

    shapes_match = True
    affines_match = True
    spacings_match = True
    all_readable = True
    max_aff_diff = 0.0
    max_spacing_diff = 0.0

    for mod in CANONICAL_MODALITIES:
        p = Path(modalities[mod])
        meta, aff, shape, zooms = inspect_modality_file(p, compute_hash=compute_hash)
        modality_meta[mod] = meta

        if not meta["readable"]:
            all_readable = False
            continue

        if ref_shape is None:
            ref_shape = shape
            ref_aff = aff
            ref_zooms = zooms
            ref_mod = mod
        else:
            if shape != ref_shape:
                shapes_match = False
            if aff is not None and ref_aff is not None:
                diff = float(np.max(np.abs(aff - ref_aff)))
                if diff > max_aff_diff:
                    max_aff_diff = diff
                if not np.allclose(aff, ref_aff, atol=tolerance):
                    affines_match = False
            if zooms is not None and ref_zooms is not None:
                s_diff = float(np.max(np.abs(np.array(zooms) - np.array(ref_zooms))))
                if s_diff > max_spacing_diff:
                    max_spacing_diff = s_diff
                if not np.allclose(zooms, ref_zooms, atol=tolerance):
                    spacings_match = False

    is_valid = all_readable and shapes_match and affines_match and spacings_match

    # Build updated case record
    updated_record = dict(case_record)
    updated_record["file_sizes_bytes"] = {
        m: modality_meta[m]["file_size_bytes"] for m in CANONICAL_MODALITIES
    }
    updated_record["spatial_shape"] = list(ref_shape) if ref_shape is not None else [240, 240, 155]
    updated_record["voxel_spacing_mm"] = list(ref_zooms) if ref_zooms is not None else [1.0, 1.0, 1.0]
    updated_record["modality_metadata"] = modality_meta
    updated_record["spatial_consistency"] = {
        "all_modalities_readable": all_readable,
        "shapes_consistent": shapes_match,
        "spacings_consistent": spacings_match,
        "affines_consistent": affines_match,
        "affine_max_diff": round(max_aff_diff, 6),
        "spacing_max_diff": round(max_spacing_diff, 6),
        "is_valid": is_valid,
    }

    return updated_record


def generate_verified_manifest(
    existing_manifest_path: Path = DEFAULT_MANIFEST_PATH,
    output_path: Optional[Path] = None,
    compute_hashes: bool = False,
    max_cases: Optional[int] = None,
    tolerance: float = DEFAULT_TOLERANCE,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Reads existing manifest cases, verifies all NIfTI headers, and outputs enriched manifest.
    """
    if not existing_manifest_path.exists():
        raise FileNotFoundError(f"Existing manifest not found at: {existing_manifest_path}")

    with open(existing_manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_cases = data["cases"]
    if max_cases is not None:
        raw_cases = raw_cases[:max_cases]

    total = len(raw_cases)
    print(f"Verifying NIfTI metadata for {total} cases (compute_file_hashes={compute_hashes})...")
    if compute_hashes:
        est_gb = (total * 4 * 2.5) / 1024.0
        print(f"[NOTE] Full SHA-256 calculation requested: estimating ~{est_gb:.1f} GB of sequential disk reads.")

    t0 = time.time()
    updated_cases: List[Dict[str, Any]] = []
    invalid_cases: List[str] = []

    for idx, c in enumerate(raw_cases, 1):
        updated = verify_case_modalities(c, compute_hash=compute_hashes, tolerance=tolerance)
        updated_cases.append(updated)

        if not updated["spatial_consistency"]["is_valid"]:
            invalid_cases.append(updated["anonymized_id"])

        if idx % 250 == 0 or idx == total:
            elapsed = time.time() - t0
            rate = idx / elapsed if elapsed > 0 else 0
            print(f"Processed {idx}/{total} cases ({rate:.1f} cases/s, elapsed: {elapsed:.1f}s)...")

    elapsed_total = time.time() - t0
    glioma_count = sum(1 for c in updated_cases if c["class_label"] == "glioma")
    meningioma_count = sum(1 for c in updated_cases if c["class_label"] == "meningioma")
    balanced_count = sum(1 for c in updated_cases if c.get("is_in_balanced_cohort") is True)
    extension_count = total - balanced_count

    print(f"\nVerification Complete in {elapsed_total:.2f}s:")
    print(f"  Total Cases: {total} (Glioma: {glioma_count}, Meningioma: {meningioma_count})")
    print(f"  Valid & Consistent: {total - len(invalid_cases)}/{total}")
    if invalid_cases:
        print(f"  [WARNING] Invalid / Inconsistent Cases: {len(invalid_cases)} ({invalid_cases[:5]}...)")

    output_manifest = {
        "metadata": {
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "generator_script": "scripts/generate_classifier_manifest.py",
            "random_seed": data.get("metadata", {}).get("random_seed", 42),
            "total_cases": total,
            "glioma_cases": glioma_count,
            "meningioma_cases": meningioma_count,
            "balanced_cohort_size": balanced_count,
            "glioma_extension_size": extension_count,
            "measured_properties": [
                "per_modality_shape",
                "per_modality_voxel_spacing",
                "per_modality_affine",
                "file_sizes_bytes",
                "inter_modality_spatial_consistency",
            ],
            "file_hashes_computed": compute_hashes,
            "file_hashes_note": (
                "Computed per-file SHA-256 hashes"
                if compute_hashes
                else "Per-file SHA-256 hashing skipped to avoid ~22.5GB disk I/O overhead. Use --compute-file-hashes to enable."
            ),
            "split_list_checksum_note": (
                "Checksums recorded in split.json are SHA-256 hashes of the case-list strings (partition membership), NOT raw MRI file data hashes."
            ),
        },
        "cases": updated_cases,
    }

    if not dry_run and output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_manifest, f, indent=2)
        print(f"Saved verified manifest to: {output_path}")

    return output_manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify and generate classifier manifest with measured NIfTI metadata.")
    parser.add_argument(
        "--existing-manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Path to existing manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Destination path for updated manifest.json",
    )
    parser.add_argument(
        "--compute-file-hashes",
        action="store_true",
        default=False,
        help="Compute SHA-256 for all individual raw MRI files (slow, ~22.5 GB I/O). Default is False.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Limit number of cases processed (for quick testing/dry-run).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help="Tolerance for affine and voxel spacing consistency (default: 1e-3).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Inspect and verify without overwriting output file.",
    )

    args = parser.parse_args()
    generate_verified_manifest(
        existing_manifest_path=args.existing_manifest,
        output_path=args.output,
        compute_hashes=args.compute_file_hashes,
        max_cases=args.max_cases,
        tolerance=args.tolerance,
        dry_run=args.dry_run,
    )
