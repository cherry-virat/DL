"""
Streaming PyTorch Dataset for 3D Brain Tumor Presence Detection.
================================================================

Provides on-demand, memory-efficient loading of 2-channel structural MRI acquisitions
(T1 and T2) for binary Healthy Control (Class 0) vs. Tumor Present (Class 1) classification.

Design Principles:
------------------
1. On-Demand Streaming: Preprocesses 3D volumes dynamically using the 2-channel preprocessing engine,
   avoiding massive intermediate disk caches.
2. Complete Anonymization: Downstream consumers observe only standardized (2, 128, 128, 96) tensors
   and integer labels. Raw paths and subject IDs are strictly excluded from tensors.
3. Isolated Workflow: Fully decoupled from the 4-channel tumor-type classifier and frozen segmentation.
4. Deterministic Partitioning: Split assignments are read from data/tumor_presence/split.json.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import json

import torch
from torch.utils.data import Dataset

from preprocessing.tumor_presence_preprocess import (
    preprocess_2channel_presence_tensor,
    CANONICAL_PRESENCE_MODALITIES,
    TARGET_SHAPE,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "tumor_presence" / "manifest.json"
DEFAULT_SPLIT_PATH = PROJECT_ROOT / "data" / "tumor_presence" / "split.json"


class TumorPresenceDataset(Dataset):
    """
    On-demand streaming PyTorch Dataset for Healthy Control vs. Tumor Present detection.
    """

    def __init__(
        self,
        split: str = "train",
        manifest_path: Union[str, Path] = DEFAULT_MANIFEST_PATH,
        split_path: Union[str, Path] = DEFAULT_SPLIT_PATH,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        auto_align: bool = True,
        filter_non_qc: bool = False,
    ):
        """
        Parameters:
            split: One of 'train', 'val', 'validation', 'test', or 'all'.
            manifest_path: Path to manifest.json.
            split_path: Path to split.json.
            transform: Optional tensor transform function.
            auto_align: Whether to automatically align multi-grid acquisitions (e.g. IXI).
            filter_non_qc: If True, strictly excludes cases marked FLAGGED, FAILED,
                           PENDING_PREPROCESSING, or EXCLUDED from training/evaluation.
        """
        self.split = split.lower().strip()
        self.manifest_path = Path(manifest_path)
        self.split_path = Path(split_path)
        self.transform = transform
        self.auto_align = auto_align
        self.filter_non_qc = filter_non_qc

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Tumor presence manifest not found at: {self.manifest_path}")
        if not self.split_path.exists():
            raise FileNotFoundError(f"Tumor presence split file not found at: {self.split_path}")

        # Load manifest
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        all_cases = {c["anonymized_id"]: c for c in manifest_data["cases"]}

        # Load split definitions
        with open(self.split_path, "r", encoding="utf-8") as f:
            split_data = json.load(f)

        if self.split == "train":
            target_ids = split_data.get("train_cases", [])
        elif self.split in ("val", "validation"):
            target_ids = split_data.get("val_cases", [])
        elif self.split == "test":
            target_ids = split_data.get("test_cases", [])
        elif self.split == "all":
            target_ids = sorted(list(all_cases.keys()))
        else:
            raise ValueError(
                f"Unsupported split '{split}'. Expected 'train', 'val', 'test', or 'all'."
            )

        self.case_records: List[Dict] = []
        for cid in target_ids:
            if cid in all_cases:
                record = all_cases[cid]
                if self.filter_non_qc:
                    # Enforce strict QC filter: cases with non-PASSED status or non-skull-stripped IXI scans are excluded
                    qc_status = record.get("qc_status")
                    if qc_status in ("FLAGGED", "FAILED", "PENDING_PREPROCESSING", "EXCLUDED"):
                        continue
                    if record.get("source_cohort") == "ixi" and not record.get("is_skull_stripped", False):
                        continue
                self.case_records.append(record)
            else:
                raise KeyError(f"Case ID '{cid}' from split not found in manifest.")

    def __len__(self) -> int:
        return len(self.case_records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Loads and preprocesses a single 2-channel volume on the fly.

        Returns:
            tensor_2d: PyTorch FloatTensor of shape (2, 128, 128, 96).
            label: Integer class ID (0 for Healthy Control, 1 for Tumor Present).
            anonymized_id: Anonymized tracking identifier (e.g. 'ANON_IXI_0042' or 'ANON_TUM_0012').
        """
        record = self.case_records[idx]
        modality_paths = record["modalities"]
        label = int(record["class_id"])
        anon_id = str(record["anonymized_id"])

        # Preprocess on the fly using 2-channel presence engine
        tensor_2d, _ = preprocess_2channel_presence_tensor(modality_paths, auto_align=self.auto_align)

        # Apply optional augmentation/transformation
        if self.transform is not None:
            tensor_2d = self.transform(tensor_2d)

        return tensor_2d, label, anon_id
