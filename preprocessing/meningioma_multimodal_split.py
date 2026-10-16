"""
meningioma_multimodal_split.py
==============================
Deterministic, stratified patient-level 80/20 train/validation split generator
for the BraTS 2023 Multi-Modal Meningioma Dataset.

Source Data:
    - outputs/brats2023_men_audit/case_registry.csv (audited registry of all 1,000 cases)
    - C:\\3D_DATASET\\BraTS2023_MEN\\train.jsonl (manifest)

Target Split:
    - data/meningioma_multimodal/split.json
      * 800 training cases (80%)
      * 200 validation cases (20%)
      * Stratified by tumor volume (wt_voxels) and source provenance (BraTS-MEN-Train vs BraTS-MEN-Train-Fix)
      * Deterministic seed: 42
"""

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_REGISTRY_CSV = PROJECT_ROOT / "outputs" / "brats2023_men_audit" / "case_registry.csv"
DEFAULT_DATASET_ROOT = Path(r"C:\3D_DATASET\BraTS2023_MEN")
DEFAULT_OUTPUT_SPLIT_JSON = PROJECT_ROOT / "data" / "meningioma_multimodal" / "split.json"


def stratify_split_subset(
    df_subset: pd.DataFrame,
    n_val: int,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Deterministically partitions a subset into train and val using block-stratified
    sampling based on sorted tumor burden (wt_voxels).
    """
    df_sorted = df_subset.sort_values("wt_voxels").reset_index(drop=True)
    n = len(df_sorted)
    rng = np.random.RandomState(random_state)

    val_indices: List[int] = []
    # Block size 5 gives exactly 20% validation rate
    for i in range(0, n, 5):
        block = list(range(i, min(i + 5, n)))
        if len(block) == 5:
            val_idx = int(rng.choice(block))
            val_indices.append(val_idx)

    # Rebalance to match exact n_val count if tail blocks differed
    if len(val_indices) > n_val:
        val_indices = rng.choice(val_indices, size=n_val, replace=False).tolist()
    elif len(val_indices) < n_val:
        remaining = [idx for idx in range(n) if idx not in val_indices]
        add = rng.choice(remaining, size=(n_val - len(val_indices)), replace=False).tolist()
        val_indices.extend(add)

    val_indices_set = set(val_indices)
    val_set = df_sorted.iloc[list(val_indices_set)].copy()
    train_set = df_sorted.drop(index=list(val_indices_set)).copy()

    return train_set, val_set


def create_multimodal_split(
    registry_csv: Path = DEFAULT_REGISTRY_CSV,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    output_split_json: Path = DEFAULT_OUTPUT_SPLIT_JSON,
    random_state: int = 42,
) -> Dict[str, Any]:
    output_split_json.parent.mkdir(parents=True, exist_ok=True)

    if not registry_csv.exists():
        raise FileNotFoundError(f"Case registry not found: {registry_csv}")

    df = pd.read_csv(registry_csv)
    if len(df) != 1000:
        raise ValueError(f"Expected 1,000 cases in registry, got {len(df)}")

    # Separate by source folder provenance
    df_train_folder = df[df["source_folder"] == "BraTS-MEN-Train"].copy()
    df_fix_folder = df[df["source_folder"] == "BraTS-MEN-Train-Fix"].copy()

    # Partition 953 -> 762 train, 191 val
    tr_main, val_main = stratify_split_subset(df_train_folder, n_val=191, random_state=random_state)
    # Partition 47 -> 38 train, 9 val
    tr_fix, val_fix = stratify_split_subset(df_fix_folder, n_val=9, random_state=random_state)

    train_df = pd.concat([tr_main, tr_fix]).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    val_df = pd.concat([val_main, val_fix]).sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # Verification checks
    assert len(train_df) == 800, f"Expected 800 train cases, got {len(train_df)}"
    assert len(val_df) == 200, f"Expected 200 val cases, got {len(val_df)}"
    train_set = set(train_df["case_id"])
    val_set = set(val_df["case_id"])
    overlap = train_set.intersection(val_set)
    assert len(overlap) == 0, f"Train and validation sets overlap on {len(overlap)} cases!"
    assert len(train_set.union(val_set)) == 1000, "Missing cases in union of splits!"

    # Build per-case metadata map
    case_details = {}
    for _, row in pd.concat([train_df.assign(partition="train"), val_df.assign(partition="val")]).iterrows():
        cid = row["case_id"]
        s_folder = row["source_folder"]
        case_dir = dataset_root / s_folder / cid

        case_details[cid] = {
            "case_id": cid,
            "source_folder": s_folder,
            "partition": row["partition"],
            "modalities": {
                "t1n": str(case_dir / f"{cid}-t1n.nii.gz"),
                "t1c": str(case_dir / f"{cid}-t1c.nii.gz"),
                "t2w": str(case_dir / f"{cid}-t2w.nii.gz"),
                "t2f": str(case_dir / f"{cid}-t2f.nii.gz"),
            },
            "mask": str(case_dir / f"{cid}-seg.nii.gz"),
            "labels_present": row["labels_present"],
            "voxel_counts": {
                "wt": int(row["wt_voxels"]),
                "tc": int(row["tc_voxels"]),
                "et": int(row["et_voxels"]),
            },
        }

    # Summary statistics
    stats = {
        "train": {
            "total_cases": len(train_df),
            "source_folders": {
                "BraTS-MEN-Train": int((train_df["source_folder"] == "BraTS-MEN-Train").sum()),
                "BraTS-MEN-Train-Fix": int((train_df["source_folder"] == "BraTS-MEN-Train-Fix").sum()),
            },
            "wt_voxels": {
                "mean": float(train_df["wt_voxels"].mean()),
                "median": float(train_df["wt_voxels"].median()),
                "std": float(train_df["wt_voxels"].std()),
                "min": int(train_df["wt_voxels"].min()),
                "max": int(train_df["wt_voxels"].max()),
            },
            "labels_distribution": {str(k): int(v) for k, v in train_df["labels_present"].value_counts().items()},
        },
        "val": {
            "total_cases": len(val_df),
            "source_folders": {
                "BraTS-MEN-Train": int((val_df["source_folder"] == "BraTS-MEN-Train").sum()),
                "BraTS-MEN-Train-Fix": int((val_df["source_folder"] == "BraTS-MEN-Train-Fix").sum()),
            },
            "wt_voxels": {
                "mean": float(val_df["wt_voxels"].mean()),
                "median": float(val_df["wt_voxels"].median()),
                "std": float(val_df["wt_voxels"].std()),
                "min": int(val_df["wt_voxels"].min()),
                "max": int(val_df["wt_voxels"].max()),
            },
            "labels_distribution": {str(k): int(v) for k, v in val_df["labels_present"].value_counts().items()},
        },
    }

    split_payload = {
        "metadata": {
            "dataset": "BraTS2023_MEN",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "random_seed": random_state,
            "split_ratio": "80/20",
            "stratification_method": (
                "Block-stratified by tumor volume (wt_voxels) with separate proportional "
                "partitioning of BraTS-MEN-Train (762/191) and BraTS-MEN-Train-Fix (38/9)."
            ),
            "train_cases_count": 800,
            "val_cases_count": 200,
            "total_cases_count": 1000,
            "disclaimer": (
                "INTERNAL VALIDATION ONLY: The 200-case validation partition represents an internal development "
                "holdout for model selection and checkpoint monitoring. It must never be termed an independent test set."
            ),
        },
        "summary_statistics": stats,
        "train_cases": sorted(list(train_set)),
        "val_cases": sorted(list(val_set)),
        "case_details": case_details,
    }

    with open(output_split_json, "w", encoding="utf-8") as f:
        json.dump(split_payload, f, indent=2)

    print(f"Created multimodal split at: {output_split_json}")
    print(f"  - Train cases: {len(train_set)} (Train: {stats['train']['source_folders']['BraTS-MEN-Train']}, Fix: {stats['train']['source_folders']['BraTS-MEN-Train-Fix']})")
    print(f"  - Val cases: {len(val_set)} (Train: {stats['val']['source_folders']['BraTS-MEN-Train']}, Fix: {stats['val']['source_folders']['BraTS-MEN-Train-Fix']})")
    print(f"  - Overlap: {len(overlap)}")
    print(f"  - Train WT median: {stats['train']['wt_voxels']['median']:.1f} vs Val WT median: {stats['val']['wt_voxels']['median']:.1f}")

    return split_payload


if __name__ == "__main__":
    create_multimodal_split()

