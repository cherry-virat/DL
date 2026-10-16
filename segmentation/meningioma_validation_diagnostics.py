from pathlib import Path
import json

import numpy as np
from scipy import ndimage


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\nchan\Desktop\DL"
)

INPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "meningioma_validation_inference"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "meningioma_validation_diagnostics"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

THRESHOLD = 0.125
MIN_COMPONENT_SIZE = 2500

STRUCTURE = np.ones(
    (3, 3, 3),
    dtype=np.uint8
)


# ============================================================
# HELPERS
# ============================================================

def get_case_directories():
    """
    Find validation case directories containing
    saved inference outputs.
    """

    case_dirs = []

    for path in sorted(INPUT_ROOT.iterdir()):

        if not path.is_dir():
            continue

        probability_file = (
            path / "tumor_probability.npy"
        )

        if probability_file.exists():
            case_dirs.append(path)

    return case_dirs


def analyze_probability(probability):
    """
    Analyze raw probability map before and after
    threshold + connected-component filtering.
    """

    probability = probability.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Basic probability statistics
    # --------------------------------------------------------

    max_probability = float(
        probability.max()
    )

    mean_probability_all = float(
        probability.mean()
    )

    positive_probability_voxels = (
        probability >= THRESHOLD
    )

    threshold_voxels = int(
        positive_probability_voxels.sum()
    )

    # --------------------------------------------------------
    # Connected components after threshold
    # --------------------------------------------------------

    labeled, num_components = (
        ndimage.label(
            positive_probability_voxels,
            structure=STRUCTURE
        )
    )

    component_sizes = np.bincount(
        labeled.ravel()
    )

    if len(component_sizes) > 1:

        component_sizes_without_background = (
            component_sizes[1:]
        )

        largest_component = int(
            component_sizes_without_background.max()
        )

        components_before_filter = int(
            len(component_sizes_without_background)
        )

    else:

        largest_component = 0
        components_before_filter = 0

    # --------------------------------------------------------
    # Apply minimum component size
    # --------------------------------------------------------

    final_mask = np.zeros(
        probability.shape,
        dtype=bool
    )

    kept_components = 0

    for component_id in range(
        1,
        len(component_sizes)
    ):

        size = int(
            component_sizes[component_id]
        )

        if size >= MIN_COMPONENT_SIZE:

            final_mask[
                labeled == component_id
            ] = True

            kept_components += 1

    predicted_voxels = int(
        final_mask.sum()
    )

    predicted_volume_cm3 = (
        predicted_voxels / 1000.0
    )

    return {
        "max_probability": max_probability,
        "mean_probability_all": mean_probability_all,
        "threshold_voxels": threshold_voxels,
        "components_before_filter": components_before_filter,
        "largest_component_voxels": largest_component,
        "components_kept": kept_components,
        "predicted_voxels": predicted_voxels,
        "predicted_volume_cm3": predicted_volume_cm3,
        "tumor_predicted": predicted_voxels > 0
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print(
        "MENINGIOMA VALIDATION PREDICTION DIAGNOSTICS"
    )
    print("=" * 100)

    print("\nInput:")
    print(INPUT_ROOT)

    print("\nThreshold:")
    print(THRESHOLD)

    print("\nMinimum component size:")
    print(MIN_COMPONENT_SIZE)

    case_dirs = get_case_directories()

    print(
        f"\nValidation cases found: "
        f"{len(case_dirs)}"
    )

    if not case_dirs:

        print(
            "\nERROR:"
        )

        print(
            "No validation probability maps found."
        )

        return

    results = []

    for index, case_dir in enumerate(
        case_dirs,
        start=1
    ):

        probability_file = (
            case_dir
            / "tumor_probability.npy"
        )

        probability = np.load(
            probability_file
        )

        analysis = analyze_probability(
            probability
        )

        record = {
            "case": case_dir.name,
            **analysis
        }

        results.append(record)

        print(
            f"\rAnalyzed "
            f"{index}/{len(case_dirs)}: "
            f"{case_dir.name}",
            end=""
        )

    print("\n")

    # ========================================================
    # SUMMARY
    # ========================================================

    positive_cases = [
        r for r in results
        if r["tumor_predicted"]
    ]

    negative_cases = [
        r for r in results
        if not r["tumor_predicted"]
    ]

    print("=" * 100)
    print("VALIDATION DIAGNOSTIC SUMMARY")
    print("=" * 100)

    print(
        f"\nTotal cases: "
        f"{len(results)}"
    )

    print(
        f"Predicted positive: "
        f"{len(positive_cases)}"
    )

    print(
        f"Predicted negative: "
        f"{len(negative_cases)}"
    )

    print(
        f"Positive rate: "
        f"{100.0 * len(positive_cases) / len(results):.2f}%"
    )

    # --------------------------------------------------------
    # Volume statistics
    # --------------------------------------------------------

    volumes = np.array(
        [
            r["predicted_volume_cm3"]
            for r in results
        ],
        dtype=np.float32
    )

    positive_volumes = volumes[
        volumes > 0
    ]

    if positive_volumes.size > 0:

        print(
            f"\nPositive predicted volume:"
        )

        print(
            f"Mean: "
            f"{positive_volumes.mean():.3f} cm³"
        )

        print(
            f"Median: "
            f"{np.median(positive_volumes):.3f} cm³"
        )

        print(
            f"Minimum: "
            f"{positive_volumes.min():.3f} cm³"
        )

        print(
            f"Maximum: "
            f"{positive_volumes.max():.3f} cm³"
        )

    # --------------------------------------------------------
    # High probability but negative
    # --------------------------------------------------------

    suspicious_negative = [
        r
        for r in negative_cases
        if r["max_probability"] >= 0.50
    ]

    print(
        "\nNegative predictions with "
        "max probability >= 0.50:"
    )

    print(
        f"{len(suspicious_negative)}"
    )

    for record in sorted(
        suspicious_negative,
        key=lambda x: x["max_probability"],
        reverse=True
    ):

        print(
            f"\n{record['case']}"
        )

        print(
            f"  Max probability: "
            f"{record['max_probability']:.4f}"
        )

        print(
            f"  Threshold voxels: "
            f"{record['threshold_voxels']}"
        )

        print(
            f"  Components before filter: "
            f"{record['components_before_filter']}"
        )

        print(
            f"  Largest component: "
            f"{record['largest_component_voxels']}"
        )

        print(
            f"  Components kept: "
            f"{record['components_kept']}"
        )

    # ========================================================
    # SAVE REPORT
    # ========================================================

    report_file = (
        OUTPUT_ROOT
        / "meningioma_validation_diagnostics.json"
    )

    summary_file = (
        OUTPUT_ROOT
        / "meningioma_validation_diagnostics_summary.json"
    )

    with open(
        report_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    summary = {
        "total_cases": len(results),
        "predicted_positive": len(positive_cases),
        "predicted_negative": len(negative_cases),
        "positive_rate": (
            len(positive_cases)
            / len(results)
        ),
        "threshold": THRESHOLD,
        "minimum_component_size": MIN_COMPONENT_SIZE,
        "mean_predicted_volume_cm3": float(
            volumes.mean()
        ),
        "median_predicted_volume_cm3": float(
            np.median(volumes)
        ),
        "mean_components_kept": float(
            np.mean(
                [
                    r["components_kept"]
                    for r in results
                ]
            )
        ),
        "suspicious_negative_count": len(
            suspicious_negative
        )
    }

    with open(
        summary_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    print(
        "\nDetailed report:"
    )

    print(
        report_file
    )

    print(
        "\nSummary:"
    )

    print(
        summary_file
    )

    print(
        "\n" + "=" * 100
    )


if __name__ == "__main__":
    main()