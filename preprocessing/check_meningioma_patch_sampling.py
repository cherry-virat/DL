import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from preprocessing.meningioma_dataset import MeningiomaPatchDataset


def check_split(split):

    print("=" * 100)
    print(f"CHECKING SPLIT: {split.upper()}")
    print("=" * 100)

    dataset = MeningiomaPatchDataset(
        split=split
    )

    positive_samples = 0
    negative_samples = 0

    tumor_voxel_counts = []

    for index in range(len(dataset)):

        sample = dataset[index]

        mask = sample["mask"]

        tumor_voxels = int(
            torch.sum(mask > 0).item()
        )

        tumor_voxel_counts.append(
            tumor_voxels
        )

        if tumor_voxels > 0:
            positive_samples += 1
        else:
            negative_samples += 1

    print(
        "\nTotal samples:",
        len(dataset)
    )

    print(
        "Positive samples:",
        positive_samples
    )

    print(
        "Negative samples:",
        negative_samples
    )

    print(
        "Positive percentage:",
        f"{100.0 * positive_samples / len(dataset):.2f}%"
    )

    print(
        "\nMinimum tumor voxels:",
        min(tumor_voxel_counts)
    )

    print(
        "Maximum tumor voxels:",
        max(tumor_voxel_counts)
    )

    print(
        "Total tumor voxels:",
        sum(tumor_voxel_counts)
    )

    print()


def main():

    print("\nMENINGIOMA PATCH SAMPLING DIAGNOSTIC\n")

    check_split("train")

    check_split("dev")


if __name__ == "__main__":
    main()