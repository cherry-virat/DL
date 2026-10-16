"""
Glioma Train / Validation Split
================================

Splits Glioma cases at the CASE level.

Important:
We split patients/cases BEFORE patch extraction.

This prevents patches from the same case appearing
in both training and validation sets.
"""

from pathlib import Path

from preprocessing.glioma_dataset import (
    DATASET_ROOT,
    get_cases,
)


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_RATIO = 0.90

RANDOM_SEED = 42


# ============================================================
# CREATE SPLIT
# ============================================================

def create_train_val_split(
    cases,
    train_ratio=TRAIN_RATIO,
    seed=RANDOM_SEED,
):
    """
    Split cases into training and validation sets.
    """

    if len(cases) == 0:

        raise ValueError(
            "No Glioma cases were found."
        )

    # --------------------------------------------------------
    # Copy list so original list is not modified
    # --------------------------------------------------------

    cases = list(cases)

    # --------------------------------------------------------
    # Deterministic shuffle
    # --------------------------------------------------------

    import random

    rng = random.Random(
        seed
    )

    rng.shuffle(
        cases
    )

    # --------------------------------------------------------
    # Calculate training size
    # --------------------------------------------------------

    train_size = int(
        len(cases) * train_ratio
    )

    train_cases = cases[
        :train_size
    ]

    val_cases = cases[
        train_size:
    ]

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if len(train_cases) == 0:

        raise ValueError(
            "Training set is empty."
        )

    if len(val_cases) == 0:

        raise ValueError(
            "Validation set is empty."
        )

    return (
        train_cases,
        val_cases,
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("GLIOMA TRAIN / VALIDATION SPLIT")
    print("=" * 70)

    print(
        "\nDataset:",
        DATASET_ROOT
    )

    # --------------------------------------------------------
    # Find cases
    # --------------------------------------------------------

    cases = get_cases(
        DATASET_ROOT
    )

    print(
        "Total cases:",
        len(cases)
    )

    # --------------------------------------------------------
    # Create split
    # --------------------------------------------------------

    train_cases, val_cases = (
        create_train_val_split(
            cases
        )
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        "\nTraining cases:",
        len(train_cases)
    )

    print(
        "Validation cases:",
        len(val_cases)
    )

    print(
        "\nTraining percentage:",
        f"{len(train_cases) / len(cases) * 100:.2f}%"
    )

    print(
        "Validation percentage:",
        f"{len(val_cases) / len(cases) * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Show examples
    # --------------------------------------------------------

    print(
        "\nFirst 5 training cases:"
    )

    for case in train_cases[:5]:

        print(
            " ",
            case.name
        )

    print(
        "\nFirst 5 validation cases:"
    )

    for case in val_cases[:5]:

        print(
            " ",
            case.name
        )

    # --------------------------------------------------------
    # Check no overlap
    # --------------------------------------------------------

    train_names = {
        case.name
        for case in train_cases
    }

    val_names = {
        case.name
        for case in val_cases
    }

    overlap = (
        train_names
        &
        val_names
    )

    if overlap:

        raise RuntimeError(
            "Training/validation overlap detected: "
            f"{overlap}"
        )

    print(
        "\n✓ No case overlap."
    )

    print(
        "\nTRAIN / VALIDATION SPLIT PASSED."
    )