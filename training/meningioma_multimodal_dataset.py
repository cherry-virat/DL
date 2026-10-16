"""
meningioma_multimodal_dataset.py
================================
Streaming 4-Channel Multi-Modal PyTorch Dataset and Metric Utilities for
the BraTS 2023 Meningioma Dataset (BraTS2023_MEN).

Modality Channel Ordering:
    Channel 0: t1n (Native T1-weighted)
    Channel 1: t1c (Post-contrast T1-weighted)
    Channel 2: t2w (T2-weighted)
    Channel 3: t2f (T2-FLAIR)

Segmentation Target Labels:
    0: Background (BG)
    1: Necrotic Tumor Core (NCR)
    2: Peritumoral Edema (ED)
    3: Enhancing Tumor (ET)

Derived Evaluation Subregions:
    Whole Tumor (WT): labels 1, 2, or 3
    Tumor Core (TC):  labels 1 or 3
    Enhancing Tumor (ET): label 3

Design Constraints & Invariants:
    1. Pure Streaming: Reads NIfTI files on-the-fly without pre-caching 1,000 cases to disk.
    2. Zero Leakage: Non-zero z-score normalization computed strictly per-case and per-channel.
    3. Multiclass Targets: Retains discrete integer labels 0..3 without collapsing during loading.
    4. Exact Native Spatial Alignment: Preserves (240, 240, 155) 1.0mm isotropic orientation.
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SPLIT_JSON = PROJECT_ROOT / "data" / "meningioma_multimodal" / "split.json"
DEFAULT_DATASET_ROOT = Path(r"C:\3D_DATASET\BraTS2023_MEN")

MODALITY_KEYS = ["t1n", "t1c", "t2w", "t2f"]
NUM_CHANNELS = 4
NUM_CLASSES = 4

LABEL_DEFINITIONS = {
    0: "Background",
    1: "Necrotic Tumor Core (NCR)",
    2: "Peritumoral Edema (ED)",
    3: "Enhancing Tumor (ET)",
}


def normalize_nonzero_zscore(volume: np.ndarray) -> np.ndarray:
    """
    Performs standard z-score normalization on non-zero brain voxels.
    Background voxels (value == 0) remain strictly 0.0.
    Computed strictly on a single volume channel (no inter-patient data leakage).
    """
    vol = volume.astype(np.float32)
    non_zero_mask = vol != 0
    if not np.any(non_zero_mask):
        return vol

    mean = float(vol[non_zero_mask].mean())
    std = float(vol[non_zero_mask].std())

    if std < 1e-8:
        return np.zeros_like(vol, dtype=np.float32)

    normalized = np.zeros_like(vol, dtype=np.float32)
    normalized[non_zero_mask] = (vol[non_zero_mask] - mean) / std
    return normalized


def load_nifti_volume(file_path: Union[str, Path], is_mask: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads a single NIfTI file, returning its numpy array and 4x4 affine matrix.
    Uses dataobj conversion for minimal memory overhead.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"NIfTI file does not exist: {file_path}")

    nii = nib.load(str(file_path))
    affine = nii.affine

    if is_mask:
        data = np.asanyarray(nii.dataobj).astype(np.int64)
    else:
        data = np.asanyarray(nii.dataobj).astype(np.float32)

    return data, affine


class MeningiomaMultimodalDataset(Dataset):
    """
    Streaming 4-Channel PyTorch Dataset for BraTS2023 Meningioma.
    Loads modalities in order [t1n, t1c, t2w, t2f] and multiclass mask [0..3].
    """

    def __init__(
        self,
        split_json_path: Union[str, Path] = DEFAULT_SPLIT_JSON,
        partition: str = "train",
        dataset_root: Union[str, Path] = DEFAULT_DATASET_ROOT,
        patch_size: Optional[Tuple[int, int, int]] = None,
        patch_sampling: str = "tumor_weighted",
        transform: Optional[Any] = None,
        cache_num: int = 0,
    ):
        super().__init__()
        self.split_json_path = Path(split_json_path)
        self.partition = partition.lower()
        self.dataset_root = Path(dataset_root)
        self.patch_size = patch_size
        self.patch_sampling = patch_sampling
        self.transform = transform
        self.cache_num = max(0, cache_num)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._candidate_cache: Dict[str, Dict[str, torch.Tensor]] = {}

        if not self.split_json_path.exists():
            raise FileNotFoundError(f"Split file does not exist: {self.split_json_path}")

        with open(self.split_json_path, "r", encoding="utf-8") as f:
            self.split_data = json.load(f)

        if self.partition == "train":
            self.case_ids = sorted(self.split_data["train_cases"])
        elif self.partition in ("val", "validation"):
            self.case_ids = sorted(self.split_data["val_cases"])
        elif self.partition in ("all", "full"):
            self.case_ids = sorted(self.split_data["train_cases"] + self.split_data["val_cases"])
        else:
            raise ValueError(f"Unknown partition '{partition}'. Must be 'train', 'val', or 'all'.")

        self.case_details = self.split_data.get("case_details", {})

    def __len__(self) -> int:
        return len(self.case_ids)

    def _resolve_case_paths(self, case_id: str) -> Tuple[Dict[str, Path], Path, str]:
        """Resolves full disk paths for the 4 modalities and segmentation mask."""
        if case_id in self.case_details:
            info = self.case_details[case_id]
            source_folder = info["source_folder"]
            modality_paths = {k: Path(p) for k, p in info["modalities"].items()}
            mask_path = Path(info["mask"])
        else:
            # Fallback path construction if details dictionary not present
            source_folder = "BraTS-MEN-Train"
            c_dir = self.dataset_root / source_folder / case_id
            if not c_dir.exists():
                source_folder = "BraTS-MEN-Train-Fix"
                c_dir = self.dataset_root / source_folder / case_id
            modality_paths = {k: c_dir / f"{case_id}-{k}.nii.gz" for k in MODALITY_KEYS}
            mask_path = c_dir / f"{case_id}-seg.nii.gz"

        return modality_paths, mask_path, source_folder

    def _load_and_validate_case(self, case_id: str) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Reads, normalizes, validates, and tensors a full 4-channel MRI case."""
        if case_id in self._cache:
            item = self._cache[case_id]
            return item["image"], item["mask"], item["source_folder"]

        modality_paths, mask_path, source_folder = self._resolve_case_paths(case_id)

        # 1. Load mask
        mask_arr, mask_affine = load_nifti_volume(mask_path, is_mask=True)
        expected_shape = mask_arr.shape

        # Validate mask labels in range 0..3
        unique_labels = np.unique(mask_arr)
        if any(lbl < 0 or lbl > 3 for lbl in unique_labels):
            raise ValueError(f"Case {case_id} mask contains unexpected labels: {unique_labels}")

        # 2. Load and normalize each modality in strict order: [t1n, t1c, t2w, t2f]
        channels: List[np.ndarray] = []
        for mod_key in MODALITY_KEYS:
            mod_path = modality_paths[mod_key]
            vol_arr, vol_affine = load_nifti_volume(mod_path, is_mask=False)

            # Spatial consistency checks
            if vol_arr.shape != expected_shape:
                raise ValueError(
                    f"Shape mismatch in case {case_id} for {mod_key}: "
                    f"{vol_arr.shape} vs mask shape {expected_shape}"
                )
            if not np.allclose(vol_affine, mask_affine, atol=1e-3):
                raise ValueError(f"Affine mismatch in case {case_id} between {mod_key} and mask!")

            # Finite check
            if not np.isfinite(vol_arr).all():
                raise ValueError(f"Non-finite values detected in {mod_path} for case {case_id}")

            # Non-zero z-score normalization
            norm_vol = normalize_nonzero_zscore(vol_arr)
            channels.append(norm_vol)

        # Stack into 4D array: [4, D, H, W]
        image_4d = np.stack(channels, axis=0)  # float32

        image_tensor = torch.from_numpy(image_4d).float()
        mask_tensor = torch.from_numpy(mask_arr).long()

        if len(self._cache) < self.cache_num:
            self._cache[case_id] = {
                "image": image_tensor,
                "mask": mask_tensor,
                "source_folder": source_folder,
            }

        return image_tensor, mask_tensor, source_folder

    def _extract_patch(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        patch_size: Tuple[int, int, int],
        case_id: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts a 3D subvolume patch from the full volume.

        Supported patch_sampling modes:
        1. 'tumor_weighted' (default):
           - 70% probability: patch centered on a random tumor voxel (mask > 0).
           - 30% probability: uniform random crop across the full volume.
        2. 'brain_aware_weighted':
           - 60% probability: patch centered on a random tumor voxel (mask > 0).
           - 30% probability: patch centered on a random healthy brain voxel
             (any normalized modality != 0 and mask == 0).
           - 10% probability: uniform random crop across the full volume.
           Documented fallbacks:
           - If tumor center requested but no tumor candidates exist: falls back to healthy brain, then uniform.
           - If healthy brain center requested but no healthy candidates exist: falls back to tumor, then uniform.
        """
        _, d, h, w = image.shape
        pd, ph, pw = patch_size

        if d < pd or h < ph or w < pw:
            # Pad if volume is smaller than patch
            pad_d = max(0, pd - d)
            pad_h = max(0, ph - h)
            pad_w = max(0, pw - w)
            image = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h, 0, pad_d))
            mask = torch.nn.functional.pad(mask, (0, pad_w, 0, pad_h, 0, pad_d))
            _, d, h, w = image.shape

        # Retrieve or compute candidate coordinates
        if case_id is not None and case_id in self._candidate_cache:
            tumor_indices = self._candidate_cache[case_id]["tumor"]
            healthy_indices = self._candidate_cache[case_id]["healthy"]
        else:
            tumor_indices = torch.nonzero(mask > 0, as_tuple=False).to(torch.int16)
            is_brain = torch.any(image != 0, dim=0)
            healthy_mask = is_brain & (mask == 0)
            healthy_indices = torch.nonzero(healthy_mask, as_tuple=False).to(torch.int16)

            # Cap healthy candidate cache size to 50,000 to prevent excessive memory duplication
            if healthy_indices.shape[0] > 50000:
                perm = torch.randperm(healthy_indices.shape[0])[:50000]
                healthy_indices = healthy_indices[perm]

            if case_id is not None:
                self._candidate_cache[case_id] = {
                    "tumor": tumor_indices,
                    "healthy": healthy_indices,
                }

        r = float(np.random.rand())
        center = None

        if self.patch_sampling == "brain_aware_weighted":
            if r < 0.60:
                # 60% tumor-centered
                if tumor_indices.shape[0] > 0:
                    idx = int(np.random.randint(0, tumor_indices.shape[0]))
                    center = tumor_indices[idx].tolist()
                elif healthy_indices.shape[0] > 0:
                    # Fallback to healthy brain if no tumor candidates exist
                    idx = int(np.random.randint(0, healthy_indices.shape[0]))
                    center = healthy_indices[idx].tolist()
            elif r < 0.90:
                # 30% healthy brain-centered
                if healthy_indices.shape[0] > 0:
                    idx = int(np.random.randint(0, healthy_indices.shape[0]))
                    center = healthy_indices[idx].tolist()
                elif tumor_indices.shape[0] > 0:
                    # Fallback to tumor if no healthy brain candidates exist
                    idx = int(np.random.randint(0, tumor_indices.shape[0]))
                    center = tumor_indices[idx].tolist()
            # remaining 10% (r >= 0.90): center is None -> uniform crop
        elif self.patch_sampling == "tumor_weighted":
            if r < 0.70 and tumor_indices.shape[0] > 0:
                idx = int(np.random.randint(0, tumor_indices.shape[0]))
                center = tumor_indices[idx].tolist()
            # remaining 30% (or no tumor): center is None -> uniform crop

        if center is not None:
            cz, cy, cx = int(center[0]), int(center[1]), int(center[2])
            sz = max(0, min(cz - pd // 2, d - pd))
            sy = max(0, min(cy - ph // 2, h - ph))
            sx = max(0, min(cx - pw // 2, w - pw))
        else:
            sz = int(np.random.randint(0, d - pd + 1)) if d > pd else 0
            sy = int(np.random.randint(0, h - ph + 1)) if h > ph else 0
            sx = int(np.random.randint(0, w - pw + 1)) if w > pw else 0

        image_patch = image[:, sz : sz + pd, sy : sy + ph, sx : sx + pw]
        mask_patch = mask[sz : sz + pd, sy : sy + ph, sx : sx + pw]

        return image_patch, mask_patch

    def __getitem__(self, index: int) -> Dict[str, Any]:
        case_id = self.case_ids[index]
        image, mask, source_folder = self._load_and_validate_case(case_id)

        if self.patch_size is not None:
            image_out, mask_out = self._extract_patch(image, mask, self.patch_size, case_id=case_id)
        else:
            image_out, mask_out = image, mask

        if self.transform is not None:
            image_out, mask_out = self.transform(image_out, mask_out)

        return {
            "image": image_out,
            "mask": mask_out,
            "case_id": case_id,
            "source_folder": source_folder,
        }


# ============================================================
# METRIC EVALUATION UTILITIES
# ============================================================

def calculate_multimodal_metrics(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
) -> Dict[str, Any]:
    """
    Computes standard BraTS multimodal metrics for:
      - Whole Tumor (WT): labels 1, 2, 3
      - Tumor Core (TC): labels 1, 3
      - Enhancing Tumor (ET): label 3
      - Discrete class Dice for labels 0, 1, 2, 3

    Args:
        prediction: Tensor of predicted class integers [D, H, W] or [B, D, H, W].
        ground_truth: Tensor of ground truth class integers [D, H, W] or [B, D, H, W].
    """
    pred = prediction.detach().cpu().long()
    gt = ground_truth.detach().cpu().long()

    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch in metric calculation: {pred.shape} vs {gt.shape}")

    subregions = {
        "wt": {"pred": (pred >= 1) & (pred <= 3), "gt": (gt >= 1) & (gt <= 3)},
        "tc": {"pred": (pred == 1) | (pred == 3), "gt": (gt == 1) | (gt == 3)},
        "et": {"pred": (pred == 3), "gt": (gt == 3)},
    }

    metrics: Dict[str, Any] = {}

    for region_name, masks in subregions.items():
        p_mask = masks["pred"]
        g_mask = masks["gt"]

        tp = int((p_mask & g_mask).sum().item())
        fp = int((p_mask & (~g_mask)).sum().item())
        fn = int(((~p_mask) & g_mask).sum().item())
        tn = int(((~p_mask) & (~g_mask)).sum().item())

        denom_dice = 2 * tp + fp + fn
        dice = (2.0 * tp / denom_dice) if denom_dice > 0 else (1.0 if (tp + fp + fn == 0) else 0.0)

        denom_prec = tp + fp
        precision = (tp / denom_prec) if denom_prec > 0 else (1.0 if fp == 0 else 0.0)

        denom_rec = tp + fn
        recall = (tp / denom_rec) if denom_rec > 0 else (1.0 if fn == 0 else 0.0)

        denom_iou = tp + fp + fn
        iou = (tp / denom_iou) if denom_iou > 0 else (1.0 if (tp + fp + fn == 0) else 0.0)

        metrics[region_name] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "dice": float(dice),
            "precision": float(precision),
            "recall": float(recall),
            "iou": float(iou),
        }

    # Per-class Dice scores (0, 1, 2, 3)
    per_class_dice: Dict[int, float] = {}
    for cls_idx in range(NUM_CLASSES):
        p_cls = pred == cls_idx
        g_cls = gt == cls_idx
        tp_cls = int((p_cls & g_cls).sum().item())
        denom_cls = int(p_cls.sum().item()) + int(g_cls.sum().item())
        if denom_cls > 0:
            c_dice = 2.0 * tp_cls / denom_cls
        else:
            c_dice = 1.0
        per_class_dice[cls_idx] = float(c_dice)

    metrics["per_class_dice"] = per_class_dice
    return metrics


def get_multimodal_dataloader(
    split_json_path: Union[str, Path] = DEFAULT_SPLIT_JSON,
    partition: str = "train",
    batch_size: int = 1,
    shuffle: bool = False,
    patch_size: Optional[Tuple[int, int, int]] = None,
    patch_sampling: str = "tumor_weighted",
    num_workers: int = 0,
) -> DataLoader:
    """Helper factory function to create a PyTorch DataLoader for the multimodal dataset."""
    dataset = MeningiomaMultimodalDataset(
        split_json_path=split_json_path,
        partition=partition,
        patch_size=patch_size,
        patch_sampling=patch_sampling,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

