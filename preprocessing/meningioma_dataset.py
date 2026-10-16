from pathlib import Path
import json
import random

import numpy as np
import torch
from torch.utils.data import Dataset


# ============================================================
# PATHS
# ============================================================

PREPROCESSED_ROOT = Path(
    r"C:\Users\nchan\Desktop\DL\data\meningioma_preprocessed"
)

SPLIT_FILE = Path(
    r"C:\Users\nchan\Desktop\DL\data\meningioma_split.json"
)


# ============================================================
# DATASET SETTINGS
# ============================================================

PATCH_SIZE = (
    64,
    64,
    64
)

# Number of patches generated from each case
PATCHES_PER_CASE = 2

# Probability of selecting a tumor-containing patch
# Used only for TRAIN split.
POSITIVE_PROBABILITY = 0.5

# Reproducibility
SEED = 42


# ============================================================
# MENINGIOMA DATASET
# ============================================================

class MeningiomaPatchDataset(Dataset):

    def __init__(
        self,
        split="train",
        patch_size=PATCH_SIZE,
        patches_per_case=PATCHES_PER_CASE,
        positive_probability=POSITIVE_PROBABILITY,
        seed=SEED
    ):

        if split not in {
            "train",
            "dev"
        }:

            raise ValueError(
                "split must be 'train' or 'dev'"
            )

        self.split = split

        self.patch_size = tuple(
            patch_size
        )

        self.patches_per_case = (
            int(patches_per_case)
        )

        self.positive_probability = (
            float(positive_probability)
        )

        self.seed = int(seed)

        # ----------------------------------------------------
        # Load split
        # ----------------------------------------------------

        if not SPLIT_FILE.exists():

            raise FileNotFoundError(
                f"Split file not found:\n"
                f"{SPLIT_FILE}"
            )

        with open(
            SPLIT_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            split_data = json.load(f)

        if split == "train":

            self.case_names = list(
                split_data["train_cases"]
            )

        else:

            self.case_names = list(
                split_data["dev_cases"]
            )

        # ----------------------------------------------------
        # Verify cases
        # ----------------------------------------------------

        missing_cases = []

        for case_name in self.case_names:

            case_dir = (
                PREPROCESSED_ROOT
                / case_name
            )

            image_file = (
                case_dir / "t1c.npy"
            )

            mask_file = (
                case_dir / "gtv.npy"
            )

            if (
                not case_dir.exists()
                or not image_file.exists()
                or not mask_file.exists()
            ):

                missing_cases.append(
                    case_name
                )

        if missing_cases:

            raise RuntimeError(
                "Missing preprocessed cases:\n"
                + "\n".join(
                    missing_cases[:20]
                )
            )

        # ----------------------------------------------------
        # Dataset length
        # ----------------------------------------------------

        self.total_patches = (
            len(self.case_names)
            * self.patches_per_case
        )

        print(
            f"\nMeningiomaPatchDataset"
        )

        print(
            f"Split:              {self.split}"
        )

        print(
            f"Cases:              {len(self.case_names)}"
        )

        print(
            f"Patch size:         {self.patch_size}"
        )

        print(
            f"Patches per case:   "
            f"{self.patches_per_case}"
        )

        print(
            f"Total patches:      "
            f"{self.total_patches}"
        )

        if self.split == "train":

            print(
                f"Positive probability:"
                f" {self.positive_probability}"
            )

        else:

            print(
                "Validation sampling: "
                "tumor-positive deterministic patches"
            )

    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):

        return self.total_patches

    # ========================================================
    # LOAD CASE
    # ========================================================

    def _load_case(self, case_name):

        case_dir = (
            PREPROCESSED_ROOT
            / case_name
        )

        image = np.load(
            case_dir / "t1c.npy"
        ).astype(
            np.float32
        )

        mask = np.load(
            case_dir / "gtv.npy"
        ).astype(
            np.uint8
        )

        return image, mask

    # ========================================================
    # PAD CASE
    # ========================================================

    def _pad_if_needed(
        self,
        image,
        mask
    ):

        pads = []

        for dimension, patch_dimension in zip(
            image.shape,
            self.patch_size
        ):

            if dimension < patch_dimension:

                total_pad = (
                    patch_dimension
                    - dimension
                )

                before = total_pad // 2

                after = (
                    total_pad
                    - before
                )

            else:

                before = 0
                after = 0

            pads.append(
                (before, after)
            )

        if any(
            before > 0 or after > 0
            for before, after in pads
        ):

            image = np.pad(
                image,
                pads,
                mode="constant",
                constant_values=0
            )

            mask = np.pad(
                mask,
                pads,
                mode="constant",
                constant_values=0
            )

        return image, mask

    # ========================================================
    # RANDOM PATCH START
    # ========================================================

    def _random_start(
        self,
        image_shape,
        rng
    ):

        starts = []

        for dimension, patch_dimension in zip(
            image_shape,
            self.patch_size
        ):

            maximum_start = (
                dimension
                - patch_dimension
            )

            if maximum_start <= 0:

                start = 0

            else:

                start = rng.randint(
                    0,
                    maximum_start
                )

            starts.append(
                start
            )

        return tuple(starts)

    # ========================================================
    # POSITIVE PATCH
    # ========================================================

    def _positive_patch(
        self,
        image,
        mask,
        rng
    ):

        tumor_coordinates = np.argwhere(
            mask > 0
        )

        if len(tumor_coordinates) == 0:

            raise RuntimeError(
                "Cannot create a positive patch "
                "because this case contains no tumor voxels."
            )

        # ----------------------------------------------------
        # Select random tumor voxel
        # ----------------------------------------------------

        center = tumor_coordinates[
            rng.randint(
                0,
                len(tumor_coordinates)
            )
        ]

        starts = []

        for axis in range(3):

            center_position = int(
                center[axis]
            )

            patch_dimension = (
                self.patch_size[axis]
            )

            dimension = (
                image.shape[axis]
            )

            start = (
                center_position
                - patch_dimension // 2
            )

            maximum_start = (
                dimension
                - patch_dimension
            )

            start = max(
                0,
                min(
                    start,
                    maximum_start
                )
            )

            starts.append(
                start
            )

        image_patch, mask_patch = (
            self._extract_patch(
                image,
                mask,
                starts
            )
        )

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if np.sum(mask_patch) == 0:

            raise RuntimeError(
                "Positive patch sampling failed: "
                "selected patch contains zero tumor voxels."
            )

        return (
            image_patch,
            mask_patch
        )

    # ========================================================
    # NEGATIVE PATCH
    # ========================================================

    def _negative_patch(
        self,
        image,
        mask,
        rng
    ):

        # Try several random patches to find
        # one without tumor.

        image_patch = None
        mask_patch = None

        for _ in range(20):

            starts = self._random_start(
                image.shape,
                rng
            )

            image_patch, mask_patch = (
                self._extract_patch(
                    image,
                    mask,
                    starts
                )
            )

            if np.sum(mask_patch) == 0:

                return (
                    image_patch,
                    mask_patch
                )

        # If no completely negative patch
        # was found, return the last patch.

        return (
            image_patch,
            mask_patch
        )

    # ========================================================
    # EXTRACT PATCH
    # ========================================================

    def _extract_patch(
        self,
        image,
        mask,
        starts
    ):

        slices = tuple(
            slice(
                start,
                start + patch_dimension
            )
            for start, patch_dimension
            in zip(
                starts,
                self.patch_size
            )
        )

        image_patch = image[
            slices
        ]

        mask_patch = mask[
            slices
        ]

        return (
            image_patch,
            mask_patch
        )

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(self, index):

        # ----------------------------------------------------
        # Determine case
        # ----------------------------------------------------

        case_index = (
            index
            // self.patches_per_case
        )

        patch_index = (
            index
            % self.patches_per_case
        )

        case_name = self.case_names[
            case_index
        ]

        # ----------------------------------------------------
        # Load case
        # ----------------------------------------------------

        image, mask = (
            self._load_case(
                case_name
            )
        )

        # ----------------------------------------------------
        # Pad
        # ----------------------------------------------------

        image, mask = (
            self._pad_if_needed(
                image,
                mask
            )
        )

        # ----------------------------------------------------
        # Random generator
        # ----------------------------------------------------

        rng = random.Random(
            self.seed
            + index
        )

        # ----------------------------------------------------
        # TRAIN SAMPLING
        # ----------------------------------------------------

        if self.split == "train":

            if (
                rng.random()
                < self.positive_probability
            ):

                image_patch, mask_patch = (
                    self._positive_patch(
                        image,
                        mask,
                        rng
                    )
                )

            else:

                image_patch, mask_patch = (
                    self._negative_patch(
                        image,
                        mask,
                        rng
                    )
                )

        # ----------------------------------------------------
        # DEV SAMPLING
        # ----------------------------------------------------

        else:

            # Every validation patch contains tumor.
            #
            # This prevents the validation metric from
            # becoming dominated by empty patches.

            image_patch, mask_patch = (
                self._positive_patch(
                    image,
                    mask,
                    rng
                )
            )

        # ----------------------------------------------------
        # Convert to tensors
        # ----------------------------------------------------

        image_patch = np.ascontiguousarray(
            image_patch,
            dtype=np.float32
        )

        mask_patch = np.ascontiguousarray(
            mask_patch,
            dtype=np.int64
        )

        image_tensor = torch.from_numpy(
            image_patch
        ).unsqueeze(0)

        mask_tensor = torch.from_numpy(
            mask_patch
        ).unsqueeze(0)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "case_name": case_name,
            "patch_index": patch_index
        }


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 100)
    print("MENINGIOMA DATASET TEST")
    print("=" * 100)

    for split in ["train", "dev"]:

        print(
            f"\nTesting split: {split}"
        )

        dataset = MeningiomaPatchDataset(
            split=split
        )

        print(
            "\nDataset length:",
            len(dataset)
        )

        # Test first few samples

        for index in range(
            min(5, len(dataset))
        ):

            sample = dataset[index]

            image = sample["image"]
            mask = sample["mask"]

            print(
                f"\nSample {index + 1}"
            )

            print(
                "  Case:",
                sample["case_name"]
            )

            print(
                "  Image shape:",
                tuple(image.shape)
            )

            print(
                "  Mask shape:",
                tuple(mask.shape)
            )

            print(
                "  Image dtype:",
                image.dtype
            )

            print(
                "  Mask dtype:",
                mask.dtype
            )

            print(
                "  Image min:",
                float(image.min())
            )

            print(
                "  Image max:",
                float(image.max())
            )

            print(
                "  Tumor voxels:",
                int(torch.sum(mask > 0))
            )

    print("\n" + "=" * 100)
    print("DATASET TEST COMPLETE")
    print("=" * 100)