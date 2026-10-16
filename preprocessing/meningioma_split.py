from pathlib import Path
import json
import random


# ============================================================
# PATHS
# ============================================================

PREPROCESSED_ROOT = Path(
    r"C:\Users\nchan\Desktop\DL\data\meningioma_preprocessed"
)

SPLIT_OUTPUT = Path(
    r"C:\Users\nchan\Desktop\DL\data\meningioma_split.json"
)


# ============================================================
# SPLIT SETTINGS
# ============================================================

SEED = 42

TRAIN_RATIO = 0.80


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("MENINGIOMA TRAIN / DEVELOPMENT SPLIT")
    print("=" * 100)

    print("\nPreprocessed root:")
    print(PREPROCESSED_ROOT)

    if not PREPROCESSED_ROOT.exists():

        print("\nERROR:")
        print("Preprocessed dataset does not exist.")

        return

    # --------------------------------------------------------
    # Find all cases
    # --------------------------------------------------------

    case_dirs = sorted(
        [
            path
            for path in PREPROCESSED_ROOT.iterdir()
            if path.is_dir()
        ]
    )

    case_names = [
        path.name
        for path in case_dirs
    ]

    print(
        f"\nTotal cases found: {len(case_names)}"
    )

    # --------------------------------------------------------
    # Validate expected number
    # --------------------------------------------------------

    if len(case_names) != 500:

        print(
            "\nWARNING:"
            f" Expected 500 cases, found {len(case_names)}."
        )

    # --------------------------------------------------------
    # Shuffle reproducibly
    # --------------------------------------------------------

    rng = random.Random(SEED)

    shuffled_cases = case_names.copy()

    rng.shuffle(
        shuffled_cases
    )

    # --------------------------------------------------------
    # Calculate split
    # --------------------------------------------------------

    train_count = int(
        len(shuffled_cases)
        * TRAIN_RATIO
    )

    train_cases = sorted(
        shuffled_cases[:train_count]
    )

    dev_cases = sorted(
        shuffled_cases[train_count:]
    )

    # --------------------------------------------------------
    # Check leakage
    # --------------------------------------------------------

    train_set = set(train_cases)
    dev_set = set(dev_cases)

    overlap = train_set.intersection(
        dev_set
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print("\n" + "=" * 100)
    print("SPLIT SUMMARY")
    print("=" * 100)

    print(
        f"\nRandom seed:       {SEED}"
    )

    print(
        f"Training cases:    {len(train_cases)}"
    )

    print(
        f"Development cases: {len(dev_cases)}"
    )

    print(
        f"Total:             "
        f"{len(train_cases) + len(dev_cases)}"
    )

    print(
        f"Overlap:           {len(overlap)}"
    )

    # --------------------------------------------------------
    # Verify split
    # --------------------------------------------------------

    if len(overlap) == 0:

        print(
            "\n✅ No train/development case overlap."
        )

    else:

        print(
            "\n❌ DATA LEAKAGE DETECTED!"
        )

        for case in sorted(overlap):

            print(
                "  ",
                case
            )

        return

    if (
        len(train_cases)
        + len(dev_cases)
        != len(case_names)
    ):

        print(
            "\n❌ Split does not contain all cases."
        )

        return

    print(
        "\n✅ Every case belongs to exactly one split."
    )

    # --------------------------------------------------------
    # Show examples
    # --------------------------------------------------------

    print("\nFirst 10 training cases:")

    for case in train_cases[:10]:

        print(
            "  ",
            case
        )

    print("\nFirst 10 development cases:")

    for case in dev_cases[:10]:

        print(
            "  ",
            case
        )

    # --------------------------------------------------------
    # Save split
    # --------------------------------------------------------

    split_data = {
        "seed": SEED,
        "train_ratio": TRAIN_RATIO,
        "train_count": len(train_cases),
        "dev_count": len(dev_cases),
        "train_cases": train_cases,
        "dev_cases": dev_cases
    }

    SPLIT_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        SPLIT_OUTPUT,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            split_data,
            f,
            indent=4
        )

    print(
        "\nSplit file saved:"
    )

    print(
        SPLIT_OUTPUT.resolve()
    )

    print("\n" + "=" * 100)
    print("SPLIT CREATION COMPLETE") n       
    print("=" * 100)


if __name__ == "__main__":
    main()
