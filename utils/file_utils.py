"""
utils/file_utils.py
===================
Lightweight file, path, and naming utilities for 3D MRI data.
Independent of heavy deep learning frameworks (PyTorch, MONAI).
"""


def is_segmentation_mask_filename(filename: str) -> bool:
    """
    Checks whether a filename represents a segmentation mask or ground-truth label volume
    rather than a raw structural MRI acquisition sequence.
    """
    fn_lower = filename.lower()
    base = fn_lower
    for ext in (".gz", ".nii", ".tar", ".zip"):
        if base.endswith(ext):
            base = base[:-len(ext)]

    if base in ("seg", "mask", "gtv", "label", "labels"):
        return True

    for tok in ("-seg", "_seg", ".seg", "-mask", "_mask", ".mask", "-gtv", "_gtv", ".gtv"):
        if base.endswith(tok):
            return True

    for tok in ("seg", "mask", "gtv"):
        if f"-{tok}-" in fn_lower or f"_{tok}_" in fn_lower or f".{tok}." in fn_lower:
            return True
        if fn_lower.endswith(f"-{tok}.nii.gz") or fn_lower.endswith(f"_{tok}.nii.gz"):
            return True
        if fn_lower.endswith(f"-{tok}.nii") or fn_lower.endswith(f"_{tok}.nii"):
            return True

    return False

