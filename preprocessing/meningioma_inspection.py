from pathlib import Path
import numpy as np
import nibabel as nib


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(
    r"C:\3D_DATASET"
)

# We will automatically search for likely Meningioma folders.
KEYWORDS = ["MEN", "Meningioma", "RT"]


# ============================================================
# FIND DATASET DIRECTORIES
# ============================================================

def find_candidate_directories(root: Path):
    """
    Find directories whose names suggest that they contain
    the BraTS Meningioma dataset.
    """

    candidates = []

    for path in root.rglob("*"):
        if not path.is_dir():
            continue

        name = path.name.lower()

        if any(keyword.lower() in name for keyword in KEYWORDS):
            candidates.append(path)

    return candidates


# ============================================================
# FIND NIFTI FILES
# ============================================================

def find_nifti_files(directory: Path):
    """
    Return all NIfTI files inside a directory.
    """

    return sorted(
        list(directory.glob("*.nii")) +
        list(directory.glob("*.nii.gz"))
    )


# ============================================================
# LOAD NIFTI
# ============================================================

def inspect_nifti(path: Path):
    """
    Load a NIfTI file and print important information.
    """

    image = nib.load(str(path))

    data = image.get_fdata(dtype=np.float32)

    print("=" * 80)
    print(f"FILE: {path.name}")
    print(f"PATH: {path}")
    print("-" * 80)

    print(f"Shape:          {image.shape}")
    print(f"Voxel spacing:  {image.header.get_zooms()[:3]}")
    print(f"Data type:      {image.get_data_dtype()}")

    print(f"Minimum:        {np.min(data):.6f}")
    print(f"Maximum:        {np.max(data):.6f}")
    print(f"Mean:           {np.mean(data):.6f}")
    print(f"Std:            {np.std(data):.6f}")

    print(f"Affine:")
    print(image.affine)

    print("-" * 80)

    unique_values = np.unique(data)

    if len(unique_values) <= 20:
        print("Unique values:")
        print(unique_values)

    print()


# ============================================================
# IDENTIFY T1C / GTV
# ============================================================

def classify_file(path: Path):
    """
    Try to identify whether a file is T1c or GTV/mask.
    """

    name = path.name.lower()

    if "t1c" in name:
        return "T1C"

    if "gtv" in name:
        return "GTV"

    if "seg" in name:
        return "SEG"

    return "OTHER"


# ============================================================
# INSPECT ONE CASE
# ============================================================

def inspect_case(case_dir: Path):
    """
    Inspect all NIfTI files in one Meningioma case.
    """

    print("\n")
    print("#" * 100)
    print(f"CASE: {case_dir.name}")
    print(f"DIRECTORY: {case_dir}")
    print("#" * 100)

    files = find_nifti_files(case_dir)

    if not files:
        print("No NIfTI files found.")
        return

    for file in files:

        file_type = classify_file(file)

        print(f"\nDetected type: {file_type}")

        inspect_nifti(file)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("MENINGIOMA DATASET INSPECTION")
    print("=" * 100)

    print(f"\nSearching inside:")
    print(DATASET_ROOT)

    candidates = find_candidate_directories(DATASET_ROOT)

    print("\nCandidate directories:")

    for candidate in candidates:
        print(candidate)

    if not candidates:
        print("\nNo candidate Meningioma directories found.")
        return

    # --------------------------------------------------------
    # Search candidate directories for actual case folders
    # --------------------------------------------------------

    case_candidates = []

    for candidate in candidates:

        for child in candidate.iterdir():

            if not child.is_dir():
                continue

            nifti_files = find_nifti_files(child)

            if nifti_files:
                case_candidates.append(child)

    # Remove duplicates
    case_candidates = sorted(
        set(case_candidates)
    )

    print("\n")
    print("=" * 100)
    print(f"Potential cases found: {len(case_candidates)}")
    print("=" * 100)

    for case in case_candidates[:10]:
        print(case)

    # --------------------------------------------------------
    # Inspect first case
    # --------------------------------------------------------

    if case_candidates:

        print("\nInspecting first case...")
        inspect_case(case_candidates[0])

    print("\nInspection finished.")


if __name__ == "__main__":
    main()