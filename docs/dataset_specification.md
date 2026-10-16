# Dataset Specification

## 1. Glioma Dataset

Dataset:
BraTS 2023 Adult Glioma Challenge

Number of training cases:
1251

### Modalities

Each Glioma case contains:

- T1
- T1c
- T2
- FLAIR
- SEG

### Observed native image properties

Typical image shape:

240 × 240 × 155

Observed voxel spacing:

1.0 × 1.0 × 1.0 mm

Image and segmentation:

- Same shape
- Same spacing
- Same orientation

### Glioma Input

The model will use four MRI modalities:

T1 + T1c + T2 + FLAIR

Therefore:

4-channel input tensor

Shape:

(C, D, H, W)

where:

C = 4


## 2. Meningioma Dataset

Dataset:
BraTS-MEN-RT

Number of training cases:

500

### Modalities

Each case contains:

- T1c
- GTV

### Observed properties

Image dimensions vary between cases.

Voxel spacing also varies between cases.

However, within each case:

- T1c and GTV have matching shapes
- T1c and GTV have matching spacing
- T1c and GTV have matching orientation

### Meningioma Input

The model will use:

T1c

Therefore:

1-channel input tensor

Shape:

(C, D, H, W)

where:

C = 1


## 3. Preprocessing Strategy

### Glioma

T1 + T1c + T2 + FLAIR
        ↓
Orientation verification
        ↓
Intensity normalization
        ↓
Brain/background cropping
        ↓
Patch extraction
        ↓
4-channel tensor


### Meningioma

T1c
 ↓
Orientation standardization
 ↓
Resampling
 ↓
Intensity normalization
 ↓
Brain/background cropping
 ↓
Patch extraction
 ↓
1-channel tensor


## 4. Segmentation

Glioma segmentation mask:

SEG

Meningioma segmentation mask:

GTV

Image interpolation:

Linear

Segmentation interpolation:

Nearest neighbor


## 5. Training Considerations

Because the available GPU has 6 GB VRAM, full-resolution 3D training will not be assumed.

Patch-based training will be investigated.

Initial patch size:

64 × 64 × 64

A larger patch such as:

96 × 96 × 96

will be tested only if GPU memory permits.