from pathlib import Path
from collections import Counter

import nibabel as nib
import numpy as np


TRAIN_ROOT = Path(
    r"C:\3D_DATASET\BraTS2024-MEN-RT-TrainingData\BraTS-MEN-RT-Train-v2"
)


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


def get_nonzero_bbox(image):
    """
    Find bounding box of non-zero voxels.

    Returns:
        min_x, max_x,
        min_y, max_y,
        min_z, max_z
    """

    nonzero = np.argwhere(image > 0)

    if len(nonzero) == 0:
        return None

    min_coords = nonzero.min(axis=0)
    max_coords = nonzero.max(axis=0)

    return (
        int(min_coords[0]),
        int(max_coords[0]),
        int(min_coords[1]),
        int(max_coords[1]),
        int(min_coords[2]),
        int(max_coords[2]),
    )


def main():

    print("=" * 100)
    print("MENINGIOMA DATASET-WIDE SPATIAL STATISTICS")
    print("=" * 100)

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

    print(f"\nTotal training cases: {len(case_dirs)}")

    shapes = Counter()
    spacings = Counter()

    tumor_voxels_list = []
    tumor_volume_mm3_list = []

    brain_bbox_sizes = []

    failed_cases = []

    for index, case_dir in enumerate(case_dirs, 1):

        try:

            t1c_file, gtv_file = find_case_files(case_dir)

            if t1c_file is None:
                raise RuntimeError("T1c file missing")

            if gtv_file is None:
                raise RuntimeError("GTV file missing")

            t1c_img = nib.load(str(t1c_file))
            gtv_img = nib.load(str(gtv_file))

            t1c = t1c_img.get_fdata()
            gtv = gtv_img.get_fdata()

            # -------------------------------------------------
            # Shape
            # -------------------------------------------------

            shape = tuple(t1c.shape[:3])

            shapes[shape] += 1

            # -------------------------------------------------
            # Spacing
            # -------------------------------------------------

            spacing = tuple(
                np.round(
                    t1c_img.header.get_zooms()[:3],
                    4
                )
            )

            spacings[spacing] += 1

            # -------------------------------------------------
            # Tumor statistics
            # -------------------------------------------------

            tumor_voxels = int(
                np.sum(gtv > 0)
            )

            tumor_voxels_list.append(
                tumor_voxels
            )

            voxel_volume_mm3 = (
                float(spacing[0])
                * float(spacing[1])
                * float(spacing[2])
            )

            tumor_volume_mm3 = (
                tumor_voxels
                * voxel_volume_mm3
            )

            tumor_volume_mm3_list.append(
                tumor_volume_mm3
            )

            # -------------------------------------------------
            # Brain/non-zero bounding box
            # -------------------------------------------------

            bbox = get_nonzero_bbox(t1c)

            if bbox is not None:

                min_x, max_x, min_y, max_y, min_z, max_z = bbox

                bbox_size = (
                    max_x - min_x + 1,
                    max_y - min_y + 1,
                    max_z - min_z + 1
                )

                brain_bbox_sizes.append(
                    bbox_size
                )

            # -------------------------------------------------
            # Progress
            # -------------------------------------------------

            print(
                f"\rProcessing {index}/{len(case_dirs)}: "
                f"{case_dir.name}",
                end=""
            )

        except Exception as e:

            failed_cases.append(
                (case_dir.name, str(e))
            )

    print("\n")

    # =========================================================
    # SHAPE DISTRIBUTION
    # =========================================================

    print("=" * 100)
    print("IMAGE SHAPE DISTRIBUTION")
    print("=" * 100)

    for shape, count in shapes.most_common():

        print(
            f"{shape}: {count} cases"
        )

    # =========================================================
    # SPACING DISTRIBUTION
    # =========================================================

    print("\n" + "=" * 100)
    print("VOXEL SPACING DISTRIBUTION")
    print("=" * 100)

    for spacing, count in spacings.most_common():

        print(
            f"{spacing} mm: {count} cases"
        )

    # =========================================================
    # TUMOR VOXELS
    # =========================================================

    if tumor_voxels_list:

        tumor_voxels_array = np.array(
            tumor_voxels_list,
            dtype=np.float64
        )

        print("\n" + "=" * 100)
        print("TUMOR VOXEL STATISTICS")
        print("=" * 100)

        print(
            f"Minimum:  {tumor_voxels_array.min():.2f}"
        )

        print(
            f"Maximum:  {tumor_voxels_array.max():.2f}"
        )

        print(
            f"Mean:     {tumor_voxels_array.mean():.2f}"
        )

        print(
            f"Median:   {np.median(tumor_voxels_array):.2f}"
        )

        print(
            f"Std:      {tumor_voxels_array.std():.2f}"
        )

    # =========================================================
    # PHYSICAL TUMOR VOLUME
    # =========================================================

    if tumor_volume_mm3_list:

        volume_array = np.array(
            tumor_volume_mm3_list,
            dtype=np.float64
        )

        print("\n" + "=" * 100)
        print("TUMOR PHYSICAL VOLUME")
        print("=" * 100)

        print(
            f"Minimum:  {volume_array.min():.2f} mm³"
        )

        print(
            f"Maximum:  {volume_array.max():.2f} mm³"
        )

        print(
            f"Mean:     {volume_array.mean():.2f} mm³"
        )

        print(
            f"Median:   {np.median(volume_array):.2f} mm³"
        )

        print(
            f"Std:      {volume_array.std():.2f} mm³"
        )

        print(
            f"\nMinimum:  "
            f"{volume_array.min() / 1000:.4f} cm³"
        )

        print(
            f"Maximum:  "
            f"{volume_array.max() / 1000:.4f} cm³"
        )

        print(
            f"Mean:     "
            f"{volume_array.mean() / 1000:.4f} cm³"
        )

        print(
            f"Median:   "
            f"{np.median(volume_array) / 1000:.4f} cm³"
        )

    # =========================================================
    # BRAIN BOUNDING BOX
    # =========================================================

    if brain_bbox_sizes:

        bbox_array = np.array(
            brain_bbox_sizes,
            dtype=np.float64
        )

        print("\n" + "=" * 100)
        print("NON-ZERO T1c BOUNDING BOX")
        print("=" * 100)

        for axis, name in enumerate(
            ["X", "Y", "Z"]
        ):

            values = bbox_array[:, axis]

            print(
                f"{name} minimum:  {values.min():.2f}"
            )

            print(
                f"{name} maximum:  {values.max():.2f}"
            )

            print(
                f"{name} mean:     {values.mean():.2f}"
            )

            print(
                f"{name} median:   {np.median(values):.2f}"
            )

            print()

    # =========================================================
    # FAILURES
    # =========================================================

    print("=" * 100)
    print("PROCESSING SUMMARY")
    print("=" * 100)

    print(
        f"Successfully processed: "
        f"{len(case_dirs) - len(failed_cases)}"
    )

    print(
        f"Failed cases: "
        f"{len(failed_cases)}"
    )

    if failed_cases:

        print("\nProblematic cases:")

        for case_name, error in failed_cases[:50]:

            print(
                f"- {case_name}: {error}"
            )

    else:

        print(
            "\nNo processing failures."
        )

    print("\n" + "=" * 100)
    print("DATASET STATISTICS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()