from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


TRAIN_ROOT = Path(
    r"C:\3D_DATASET\BraTS2024-MEN-RT-TrainingData\BraTS-MEN-RT-Train-v2"
)

OUTPUT_DIR = Path(r"outputs\meningioma_visualization")


def find_case_files(case_dir):
    t1c_file = None
    gtv_file = None

    for file in case_dir.glob("*.nii.gz"):
        name = file.name.lower()

        if name.endswith("_t1c.nii.gz"):
            t1c_file = file

        elif name.endswith("_gtv.nii.gz"):
            gtv_file = file

    return t1c_file, gtv_file


def normalize_image(image):
    """
    Robust intensity normalization for visualization.
    """

    image = image.astype(np.float32)

    nonzero = image[image > 0]

    if len(nonzero) == 0:
        return np.zeros_like(image)

    low = np.percentile(nonzero, 1)
    high = np.percentile(nonzero, 99)

    image = np.clip(image, low, high)

    image = (image - low) / (high - low + 1e-8)

    return image


def get_tumor_slices(mask):
    """
    Return axial slices containing tumor.
    """

    tumor_voxels = np.where(mask > 0)

    if len(tumor_voxels[0]) == 0:
        return []

    z_indices = tumor_voxels[2]

    return np.unique(z_indices)


def visualize_case(case_dir):
    t1c_file, gtv_file = find_case_files(case_dir)

    if t1c_file is None or gtv_file is None:
        print(f"Skipping {case_dir.name}: missing file")
        return

    print(f"\nProcessing: {case_dir.name}")

    t1c_img = nib.load(str(t1c_file))
    gtv_img = nib.load(str(gtv_file))

    t1c = t1c_img.get_fdata().astype(np.float32)
    gtv = gtv_img.get_fdata()

    print("T1c shape:", t1c.shape)
    print("GTV shape:", gtv.shape)

    tumor_slices = get_tumor_slices(gtv)

    if len(tumor_slices) == 0:
        print("No tumor found.")
        return

    print("Tumor slice range:")
    print(
        f"Z = {tumor_slices.min()} to {tumor_slices.max()}"
    )

    # Choose the slice containing the largest tumor area.
    tumor_area_per_slice = np.sum(gtv > 0, axis=(0, 1))

    best_slice = int(np.argmax(tumor_area_per_slice))

    print("Selected slice:", best_slice)
    print(
        "Tumor voxels on selected slice:",
        int(tumor_area_per_slice[best_slice])
    )

    image = normalize_image(t1c[:, :, best_slice])
    mask = gtv[:, :, best_slice] > 0

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # -------------------------------------------------
    # Figure 1: T1c only
    # -------------------------------------------------

    plt.figure(figsize=(7, 7))

    plt.imshow(
        np.rot90(image),
        cmap="gray"
    )

    plt.title(
        f"{case_dir.name} - T1c"
    )

    plt.axis("off")

    t1c_output = OUTPUT_DIR / f"{case_dir.name}_t1c.png"

    plt.savefig(
        t1c_output,
        bbox_inches="tight",
        dpi=150
    )

    plt.close()

    # -------------------------------------------------
    # Figure 2: T1c + GTV overlay
    # -------------------------------------------------

    plt.figure(figsize=(7, 7))

    plt.imshow(
        np.rot90(image),
        cmap="gray"
    )

    overlay = np.ma.masked_where(
        ~np.rot90(mask),
        np.rot90(mask)
    )

    plt.imshow(
        overlay,
        cmap="autumn",
        alpha=0.45
    )

    plt.title(
        f"{case_dir.name} - T1c + GTV"
    )

    plt.axis("off")

    overlay_output = (
        OUTPUT_DIR
        / f"{case_dir.name}_overlay.png"
    )

    plt.savefig(
        overlay_output,
        bbox_inches="tight",
        dpi=150
    )

    plt.close()

    # -------------------------------------------------
    # Figure 3: T1c + GTV contour
    # -------------------------------------------------

    plt.figure(figsize=(7, 7))

    plt.imshow(
        np.rot90(image),
        cmap="gray"
    )

    plt.contour(
        np.rot90(mask),
        levels=[0.5],
        linewidths=1.5
    )

    plt.title(
        f"{case_dir.name} - GTV Boundary"
    )

    plt.axis("off")

    contour_output = (
        OUTPUT_DIR
        / f"{case_dir.name}_contour.png"
    )

    plt.savefig(
        contour_output,
        bbox_inches="tight",
        dpi=150
    )

    plt.close()

    print("Saved:")
    print(" ", t1c_output)
    print(" ", overlay_output)
    print(" ", contour_output)


def main():

    print("=" * 90)
    print("MENINGIOMA T1c + GTV VISUALIZATION")
    print("=" * 90)

    if not TRAIN_ROOT.exists():

        print("\nERROR:")
        print("Training directory does not exist:")
        print(TRAIN_ROOT)

        return

    case_dirs = sorted(
        [
            path
            for path in TRAIN_ROOT.iterdir()
            if path.is_dir()
        ]
    )

    if len(case_dirs) == 0:

        print("\nNo cases found.")

        return

    print(
        f"\nTotal cases available: {len(case_dirs)}"
    )

    # Visualize representative cases.
    #
    # First case
    # Middle case
    # Last case

    selected_indices = [
        0,
        len(case_dirs) // 2,
        len(case_dirs) - 1
    ]

    selected_indices = sorted(
        set(selected_indices)
    )

    print("\nSelected cases:")

    for index in selected_indices:

        print(
            f"{index + 1}: "
            f"{case_dirs[index].name}"
        )

    for index in selected_indices:

        visualize_case(
            case_dirs[index]
        )

    print("\n" + "=" * 90)

    print("VISUALIZATION COMPLETE")

    print("=" * 90)

    print("\nOutput directory:")

    print(OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()