from pathlib import Path
import numpy as np
import SimpleITK as sitk


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_SPACING = (1.0, 1.0, 1.0)


# ============================================================
# LOAD NIFTI
# ============================================================

def load_nifti(path):
    """
    Load a NIfTI image using SimpleITK.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    image = sitk.ReadImage(str(path))

    return image


# ============================================================
# CALCULATE NEW SIZE
# ============================================================

def calculate_new_size(image, target_spacing):
    """
    Calculate the output image size required to preserve
    the physical field of view after changing voxel spacing.
    """

    original_spacing = np.array(image.GetSpacing(), dtype=float)
    original_size = np.array(image.GetSize(), dtype=int)
    target_spacing = np.array(target_spacing, dtype=float)

    new_size = np.round(
        original_size * original_spacing / target_spacing
    ).astype(int)

    # SimpleITK requires minimum dimension of 1
    new_size = np.maximum(new_size, 1)

    return tuple(int(x) for x in new_size)


# ============================================================
# RESAMPLE IMAGE
# ============================================================

def resample_image(image, target_spacing=TARGET_SPACING):
    """
    Resample an MRI image using linear interpolation.
    """

    target_spacing = tuple(float(x) for x in target_spacing)

    new_size = calculate_new_size(
        image,
        target_spacing
    )

    resampler = sitk.ResampleImageFilter()

    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)

    # Preserve physical location and orientation
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())

    # MRI intensity interpolation
    resampler.SetInterpolator(
        sitk.sitkLinear
    )

    # Background value
    resampler.SetDefaultPixelValue(0)

    # Preserve image metadata
    resampler.SetTransform(
        sitk.Transform()
    )

    output = resampler.Execute(image)

    return output


# ============================================================
# RESAMPLE SEGMENTATION
# ============================================================

def resample_mask(mask, target_spacing=TARGET_SPACING):
    """
    Resample a segmentation mask using nearest-neighbor
    interpolation.

    IMPORTANT:
    Never use linear interpolation for segmentation labels.
    """

    target_spacing = tuple(float(x) for x in target_spacing)

    new_size = calculate_new_size(
        mask,
        target_spacing
    )

    resampler = sitk.ResampleImageFilter()

    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)

    resampler.SetOutputOrigin(mask.GetOrigin())
    resampler.SetOutputDirection(mask.GetDirection())

    # IMPORTANT:
    # Segmentation labels must remain discrete.
    resampler.SetInterpolator(
        sitk.sitkNearestNeighbor
    )

    resampler.SetDefaultPixelValue(0)

    resampler.SetTransform(
        sitk.Transform()
    )

    output = resampler.Execute(mask)

    return output


# ============================================================
# SAVE NIFTI
# ============================================================

def save_nifti(image, output_path):
    """
    Save a SimpleITK image as NIfTI.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    sitk.WriteImage(
        image,
        str(output_path)
    )


# ============================================================
# INFORMATION
# ============================================================

def print_image_info(name, image):
    """
    Print important image information.
    """

    print(f"\n{name}")
    print("-" * 60)

    print("Size:")
    print(image.GetSize())

    print("Spacing:")
    print(image.GetSpacing())

    print("Origin:")
    print(image.GetOrigin())

    print("Direction:")
    print(image.GetDirection())


# ============================================================
# TEST RESAMPLING
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("MRI RESAMPLING TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # CHANGE THESE PATHS TO YOUR REAL TEST CASE
    # --------------------------------------------------------

    case_dir = Path(
        r"C:\3D_DATASET\BraTS2024-MEN-RT-TrainingData"
        r"\BraTS-MEN-RT-Train-v2"
        r"\BraTS-MEN-RT-0002-1"
    )

    t1c_path = case_dir / "BraTS-MEN-RT-0002-1_t1c.nii.gz"
    gtv_path = case_dir / "BraTS-MEN-RT-0002-1_gtv.nii.gz"

    output_dir = Path(
        "outputs/preprocessing_test/meningioma"
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    print("\nLoading files...")

    t1c = load_nifti(t1c_path)
    gtv = load_nifti(gtv_path)

    print_image_info("Original T1c", t1c)
    print_image_info("Original GTV", gtv)

    # --------------------------------------------------------
    # RESAMPLE
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("RESAMPLING")
    print("=" * 70)

    print(
        f"\nTarget spacing: {TARGET_SPACING}"
    )

    t1c_resampled = resample_image(
        t1c,
        TARGET_SPACING
    )

    gtv_resampled = resample_mask(
        gtv,
        TARGET_SPACING
    )

    # --------------------------------------------------------
    # INFORMATION
    # --------------------------------------------------------

    print_image_info(
        "Resampled T1c",
        t1c_resampled
    )

    print_image_info(
        "Resampled GTV",
        gtv_resampled
    )

    # --------------------------------------------------------
    # CHECK ALIGNMENT
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("ALIGNMENT CHECK")
    print("=" * 70)

    print(
        "\nT1c size:",
        t1c_resampled.GetSize()
    )

    print(
        "GTV size:",
        gtv_resampled.GetSize()
    )

    print(
        "\nT1c spacing:",
        t1c_resampled.GetSpacing()
    )

    print(
        "GTV spacing:",
        gtv_resampled.GetSpacing()
    )

    size_match = (
        t1c_resampled.GetSize()
        ==
        gtv_resampled.GetSize()
    )

    spacing_match = np.allclose(
        t1c_resampled.GetSpacing(),
        gtv_resampled.GetSpacing()
    )

    origin_match = np.allclose(
        t1c_resampled.GetOrigin(),
        gtv_resampled.GetOrigin()
    )

    direction_match = np.allclose(
        t1c_resampled.GetDirection(),
        gtv_resampled.GetDirection()
    )

    print(
        "\nSize match:",
        size_match
    )

    print(
        "Spacing match:",
        spacing_match
    )

    print(
        "Origin match:",
        origin_match
    )

    print(
        "Direction match:",
        direction_match
    )

    # --------------------------------------------------------
    # CHECK GTV LABELS
    # --------------------------------------------------------

    gtv_array = sitk.GetArrayFromImage(
        gtv_resampled
    )

    unique_values = np.unique(
        gtv_array
    )

    print("\n")
    print("=" * 70)
    print("GTV LABEL CHECK")
    print("=" * 70)

    print(
        "\nUnique GTV values:"
    )

    print(unique_values)

    print(
        "\nNon-zero GTV voxels:",
        np.count_nonzero(gtv_array)
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("SAVING")
    print("=" * 70)

    save_nifti(
        t1c_resampled,
        output_dir / "BraTS-MEN-RT-0002-1_t1c_resampled.nii.gz"
    )

    save_nifti(
        gtv_resampled,
        output_dir / "BraTS-MEN-RT-0002-1_gtv_resampled.nii.gz"
    )

    print(
        "\n✓ Resampled T1c saved"
    )

    print(
        "✓ Resampled GTV saved"
    )

    print(
        "\nResampling test completed."
    )