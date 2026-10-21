"""
Streaming PyTorch Dataset for 3D Brain Tumor Classification.
============================================================

Provides on-demand, memory-efficient loading of 4-channel MRI acquisitions
(t1n, t1c, t2w, t2f) for binary Glioma vs. Meningioma classification.

Design Principles:
------------------
1. On-Demand Streaming: Preprocesses volumes dynamically, avoiding multi-gigabyte disk caching.
2. Complete Anonymization: Model sees only standardized (4, 128, 128, 96) tensors and integer labels.
   Raw case identifiers and disk paths are excluded from input features.
3. Decoupled Architecture: Independent of frozen segmentation code.
4. Deterministic Partitioning: Loaded directly from data/classifier/split.json.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import json

import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing.tumor_classifier_preprocess import (
    preprocess_4channel_classifier_tensor,
    CANONICAL_MODALITIES,
    TARGET_SHAPE,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "classifier" / "manifest.json"
DEFAULT_SPLIT_PATH = PROJECT_ROOT / "data" / "classifier" / "split.json"


class BrainTumorClassifierDataset(Dataset):
    """
    On-demand streaming PyTorch Dataset for Glioma vs. Meningioma classification.
    """

    def __init__(
        self,
        split: str = "train",
        manifest_path: Union[str, Path] = DEFAULT_MANIFEST_PATH,
        split_path: Union[str, Path] = DEFAULT_SPLIT_PATH,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        """
        Parameters:
            split: One of 'train', 'val', 'test', 'unbalanced_glioma_extension', or 'all'.
            manifest_path: Path to manifest.json.
            split_path: Path to split.json.
            transform: Optional spatial or intensity transform function.
        """
        self.split = split.lower().strip()
        self.manifest_path = Path(manifest_path)
        self.split_path = Path(split_path)
        self.transform = transform

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Classifier manifest not found at: {self.manifest_path}")
        if not self.split_path.exists():
            raise FileNotFoundError(f"Classifier split file not found at: {self.split_path}")

        # Load manifest
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        all_cases = {c["anonymized_id"]: c for c in manifest_data["cases"]}

        # Load split definitions
        with open(self.split_path, "r", encoding="utf-8") as f:
            split_data = json.load(f)

        if self.split == "train":
            target_ids = split_data["train_cases"]
        elif self.split in ("val", "validation"):
            target_ids = split_data["val_cases"]
        elif self.split == "test":
            target_ids = split_data["test_cases"]
        elif self.split in ("extension", "unbalanced_glioma_extension"):
            target_ids = split_data.get("unbalanced_glioma_extension", [])
        elif self.split == "all":
            target_ids = sorted(list(all_cases.keys()))
        else:
            raise ValueError(
                f"Unsupported split '{split}'. Expected 'train', 'val', 'test', 'extension', or 'all'."
            )

        self.case_records: List[Dict] = []
        for cid in target_ids:
            if cid in all_cases:
                self.case_records.append(all_cases[cid])
            else:
                raise KeyError(f"Case ID '{cid}' from split not found in manifest.")

    def __len__(self) -> int:
        return len(self.case_records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Loads and preprocesses a single multimodal volume on the fly.

        Returns:
            tensor_4d: PyTorch FloatTensor of shape (4, 128, 128, 96).
            label: Integer class ID (0 for Glioma, 1 for Meningioma).
            anonymized_id: Safe internal tracking ID (e.g. 'ANON_GLI_0042').
        """
        record = self.case_records[idx]
        modality_paths = record["modalities"]
        label = int(record["class_id"])
        anon_id = str(record["anonymized_id"])

        # Preprocess on the fly
        tensor_4d, _ = preprocess_4channel_classifier_tensor(modality_paths)

        # Apply optional augmentation
        if self.transform is not None:
            tensor_4d = self.transform(tensor_4d)

        return tensor_4d, label, anon_id
