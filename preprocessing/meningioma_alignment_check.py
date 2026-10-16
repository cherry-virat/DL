from pathlib import Path
import numpy as np
import nibabel as nib


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_ROOT = Path(
    r"C:\3D_DATASET\BraTS2024-MEN-RT-TrainingData\BraTS-MEN-RT-Train-v2"
)


# ============================================================
# FIND NIFTI FILES
# ============================================================

def find_nifti_files(case_dir: Path):
    """
    Return all NIfTI files in a case directory.
    """

    return sorted(
        list(case_dir.glob("*.nii")) +
        list(case_dir.glob("*.nii.gz"))
    )


# ============================================================
# FIND T1C AND GTV
# ============================================================

def find_case_files(case_dir: Path):
    """
    Find the T1c MRI and GTV mask for a case.
    """

    t1c_file = None
    gtv_file = None

    for file in find_nifti_files(case_dir):

        name = file.name.lower()

        if name.endswith("_t1c.nii.gz") or name.endswith("_t1c.nii"):
            t1c_file = file

        elif name.endswith("_gtv.nii.gz") or name.endswith("_gtv.nii"):
            gtv_file = file

    return t1c_file, gtv_file


# ============================================================
# AFFINE COMPARISON
# ============================================================

def affine_matches(
    image_affine,
    mask_affine,
    tolerance=1e-4
):
    """
    Check whether image and mask affine matrices
    are approximately equal.
    """

    return np.allclose(
        image_affine,
        mask_affine,
        atol=tolerance,
        rtol=0
    )


# ============================================================
# SPACING COMPARISON
# ============================================================

def spacing_matches(
    image_spacing,
    mask_spacing,
    tolerance=1e-4
):
    """
    Check whether image and mask voxel spacing matches.
    """

    return np.allclose(
        image_spacing,
        mask_spacing,
        atol=tolerance,
        rtol=0
    )


# ============================================================
# MASK VALIDATION
# ============================================================

def validate_mask(mask_data):
    """
    Validate GTV mask.

    Expected:
        background = 0
        tumor      = 1
    """

    unique_values = np.unique(mask_data)

    is_binary = np.all(
        np.isin(
            unique_values,
            [0, 1]
        )
    )

    tumor_voxels = int(
        np.sum(mask_data > 0)
    )

    return (
        is_binary,
        unique_values,
        tumor_voxels
    )


# ============================================================
# CHECK ONE CASE
# ============================================================

def check_case(case_dir: Path):
    """
    Check one training case.
    """

    t1c_file, gtv_file = find_case_files(case_dir)

    result = {
        "case": case_dir.name,
        "t1c_exists": t1c_file is not None,
        "gtv_exists": gtv_file is not None,
        "shape_match": False,
        "spacing_match": False,
        "affine_match": False,
        "mask_binary": False,
        "tumor_voxels": 0,
        "error": None,
    }

    try:

        # ----------------------------------------------------
        # Check files
        # ----------------------------------------------------

        if t1c_file is None:
            result["error"] = "T1c file missing"
            return result

        if gtv_file is None:
            result["error"] = "GTV file missing"
            return result

        # ----------------------------------------------------
        # Load files
        # ----------------------------------------------------

        t1c = nib.load(
            str(t1c_file)
        )

        gtv = nib.load(
            str(gtv_file)
        )

        # ----------------------------------------------------
        # Shape
        # ----------------------------------------------------

        t1c_shape = tuple(
            t1c.shape[:3]
        )

        gtv_shape = tuple(
            gtv.shape[:3]
        )

        result["shape_match"] = (
            t1c_shape == gtv_shape
        )

        # ----------------------------------------------------
        # Spacing
        # ----------------------------------------------------

        t1c_spacing = np.array(
            t1c.header.get_zooms()[:3],
            dtype=np.float64
        )

        gtv_spacing = np.array(
            gtv.header.get_zooms()[:3],
            dtype=np.float64
        )

        result["spacing_match"] = spacing_matches(
            t1c_spacing,
            gtv_spacing
        )

        # ----------------------------------------------------
        # Affine
        # ----------------------------------------------------

        result["affine_match"] = affine_matches(
            t1c.affine,
            gtv.affine
        )

        # ----------------------------------------------------
        # Mask
        # ----------------------------------------------------

        mask_data = gtv.get_fdata()

        (
            mask_binary,
            unique_values,
            tumor_voxels
        ) = validate_mask(mask_data)

        result["mask_binary"] = mask_binary

        result["tumor_voxels"] = tumor_voxels

        # ----------------------------------------------------
        # Save diagnostic information
        # ----------------------------------------------------

        result["t1c_shape"] = t1c_shape
        result["gtv_shape"] = gtv_shape

        result["t1c_spacing"] = tuple(
            np.round(
                t1c_spacing,
                4
            )
        )

        result["gtv_spacing"] = tuple(
            np.round(
                gtv_spacing,
                4
            )
        )

        result["mask_unique_values"] = unique_values.tolist()

    except Exception as e:

        result["error"] = str(e)

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("MENINGIOMA TRAINING DATA ALIGNMENT CHECK")
    print("=" * 100)

    print()
    print("Training root:")
    print(TRAIN_ROOT)

    # --------------------------------------------------------
    # Check root
    # --------------------------------------------------------

    if not TRAIN_ROOT.exists():

        print()
        print("ERROR:")
        print("Training directory does not exist.")

        return

    # --------------------------------------------------------
    # Find case directories
    # --------------------------------------------------------

    case_dirs = sorted(
        [
            path
            for path in TRAIN_ROOT.iterdir()
            if path.is_dir()
        ]
    )

    print()
    print(
        f"Training case directories found: "
        f"{len(case_dirs)}"
    )

    if not case_dirs:

        print()
        print("No training cases found.")

        return

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    total = len(case_dirs)

    t1c_missing = 0
    gtv_missing = 0

    shape_fail = 0
    spacing_fail = 0
    affine_fail = 0
    mask_fail = 0

    tumor_empty = 0
    valid_cases = 0

    # --------------------------------------------------------
    # Collect detailed failures
    # --------------------------------------------------------

    failures = []

    tumor_voxel_counts = []

    # --------------------------------------------------------
    # Check all cases
    # --------------------------------------------------------

    for index, case_dir in enumerate(
        case_dirs,
        start=1
    ):

        result = check_case(
            case_dir
        )

        print(
            f"\rChecking case "
            f"{index}/{total}: "
            f"{case_dir.name}",
            end=""
        )

        # ----------------------------------------------------
        # Missing files
        # ----------------------------------------------------

        if not result["t1c_exists"]:

            t1c_missing += 1

        if not result["gtv_exists"]:

            gtv_missing += 1

        # ----------------------------------------------------
        # Error
        # ----------------------------------------------------

        if result["error"] is not None:

            failures.append(result)

            continue

        # ----------------------------------------------------
        # Shape
        # ----------------------------------------------------

        if not result["shape_match"]:

            shape_fail += 1

        # ----------------------------------------------------
        # Spacing
        # ----------------------------------------------------

        if not result["spacing_match"]:

            spacing_fail += 1

        # ----------------------------------------------------
        # Affine
        # ----------------------------------------------------

        if not result["affine_match"]:

            affine_fail += 1

        # ----------------------------------------------------
        # Mask
        # ----------------------------------------------------

        if not result["mask_binary"]:

            mask_fail += 1

        # ----------------------------------------------------
        # Tumor voxels
        # ----------------------------------------------------

        tumor_voxels = result[
            "tumor_voxels"
        ]

        tumor_voxel_counts.append(
            tumor_voxels
        )

        if tumor_voxels == 0:

            tumor_empty += 1

        # ----------------------------------------------------
        # Fully valid case
        # ----------------------------------------------------

        if (
            result["shape_match"]
            and result["spacing_match"]
            and result["affine_match"]
            and result["mask_binary"]
            and tumor_voxels > 0
        ):

            valid_cases += 1

        else:

            failures.append(result)

    print()
    print()

    # ========================================================
    # SUMMARY
    # ========================================================

    print("=" * 100)
    print("ALIGNMENT SUMMARY")
    print("=" * 100)

    print()
    print(
        f"Total training cases:       {total}"
    )

    print(
        f"T1c missing:                {t1c_missing}"
    )

    print(
        f"GTV missing:                {gtv_missing}"
    )

    print(
        f"Shape mismatches:           {shape_fail}"
    )

    print(
        f"Spacing mismatches:         {spacing_fail}"
    )

    print(
        f"Affine mismatches:          {affine_fail}"
    )

    print(
        f"Non-binary masks:           {mask_fail}"
    )

    print(
        f"Empty tumor masks:          {tumor_empty}"
    )

    print()
    print(
        f"Fully valid cases:          {valid_cases}"
    )

    # ========================================================
    # TUMOR SIZE STATISTICS
    # ========================================================

    if tumor_voxel_counts:

        tumor_array = np.array(
            tumor_voxel_counts,
            dtype=np.float64
        )

        print()
        print("=" * 100)
        print("TUMOR VOXEL STATISTICS")
        print("=" * 100)

        print()
        print(
            f"Minimum tumor voxels:       "
            f"{int(np.min(tumor_array))}"
        )

        print(
            f"Maximum tumor voxels:       "
            f"{int(np.max(tumor_array))}"
        )

        print(
            f"Mean tumor voxels:          "
            f"{np.mean(tumor_array):.2f}"
        )

        print(
            f"Median tumor voxels:        "
            f"{np.median(tumor_array):.2f}"
        )

        print(
            f"Std tumor voxels:           "
            f"{np.std(tumor_array):.2f}"
        )

    # ========================================================
    # FAILURES
    # ========================================================

    if failures:

        print()
        print("=" * 100)
        print("PROBLEMATIC CASES")
        print("=" * 100)

        for result in failures[:50]:

            print()
            print(
                f"Case: {result['case']}"
            )

            if result["error"]:

                print(
                    f"Error: {result['error']}"
                )

                continue

            print(
                f"Shape match:   "
                f"{result['shape_match']}"
            )

            print(
                f"Spacing match: "
                f"{result['spacing_match']}"
            )

            print(
                f"Affine match:  "
                f"{result['affine_match']}"
            )

            print(
                f"Mask binary:   "
                f"{result['mask_binary']}"
            )

            print(
                f"Tumor voxels:  "
                f"{result['tumor_voxels']}"
            )

    else:

        print()
        print(
            "No problematic cases found."
        )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print()
    print("=" * 100)

    if valid_cases == total:

        print(
            "STATUS: ALL TRAINING CASES PASSED"
        )

    else:

        print(
            "STATUS: SOME TRAINING CASES NEED ATTENTION"
        )

    print("=" * 100)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()