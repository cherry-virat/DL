from preprocessing.glioma_dataset import (
    GliomaDataset,
    DATASET_ROOT,
    get_cases,
)


TARGET_CASE = "BraTS-GLI-00464-001"


cases = get_cases(
    DATASET_ROOT
)


target = [
    case
    for case in cases
    if case.name == TARGET_CASE
]


if len(target) != 1:

    raise RuntimeError(
        f"Could not find {TARGET_CASE}"
    )


dataset = GliomaDataset(
    target
)


patches = dataset[0]


print("=" * 70)
print("SMALL-VOLUME CASE TEST")
print("=" * 70)

print(
    "Case:",
    TARGET_CASE
)

print(
    "Number of patches:",
    len(patches)
)


for i, patch in enumerate(
    patches,
    start=1,
):

    image = patch["image"]

    label = patch["label"]

    print(
        f"\nPatch {i}"
    )

    print(
        "Image shape:",
        tuple(image.shape)
    )

    print(
        "Label shape:",
        tuple(label.shape)
    )

    print(
        "Tumor voxels:",
        int(
            (label > 0).sum().item()
        )
    )


    assert tuple(
        image.shape
    ) == (
        4,
        64,
        64,
        64,
    )


    assert tuple(
        label.shape
    ) == (
        1,
        64,
        64,
        64,
    )


print(
    "\n" + "=" * 70
)

print(
    "SMALL-VOLUME CASE TEST PASSED"
)

print(
    "=" * 70
)