# Multimodal Meningioma Inference Integration Guide

## 1. Overview & System Architecture

This document describes the isolated integration service for the **4-Channel Multi-Modal Meningioma 3D U-Net** segmentation pipeline in the brain tumor segmentation platform.

The system connects four MRI modalities to an evaluated 3D convolutional neural network, executing sliding-window inference with Gaussian blending, empty-air masking on prediction, and automated volumetric quantification for all BraTS tumor subregions.

> [!NOTE]
> As of October 2026, the 4-channel multimodal model is the **sole active meningioma model** across the platform. Both `tumor_type="meningioma"` and `tumor_type="meningioma_multimodal"` route directly to `MeningiomaMultimodalInferenceService`. The legacy single-channel (T1c-only) model has been decommissioned from active API routing. Any request missing one or more of the 4 required modalities will be rejected with an HTTP 400 validation error (never silently falling back to a 1-channel model).

```
                      Inbound Client Request
                                │
        ┌───────────────────────┴───────────────────────┐
        ▼                                               ▼
POST /api/segment/meningioma-multimodal       POST /api/segment
 (Compatible direct endpoint)             (tumor_type="meningioma" or "meningioma_multimodal")
        │                                               │
        └───────────────────────┬───────────────────────┘
                                ▼
                    InferenceRouter.route()
                                │
                                ▼
            MeningiomaMultimodalInferenceService
            ├── Preprocessing: normalize_nonzero_zscore (4 channels)
            ├── Input Geometry & Affine Congruence Verification
            ├── Model: 4-Channel 3D U-Net (InstanceNorm3d, base_channels=16)
            ├── MONAI Sliding-Window Inference (ROI: 64x64x64, Gaussian)
            ├── Empty-Air Masking (prediction zeroed where input channels == 0)
            └── Subregion Quantification: WT (1,2,3), TC (1,3), ET (3), NCR (1), ED (2)
                                │
                                ▼
            Standard SegmentationResult Data Contract (JSON)
            + 3D NIfTI Multiclass Segmentation Mask (.nii.gz)
```

### Verified Checkpoint
- **File**: `outputs/meningioma_multimodal_experiment/full800_brainaware_20261014/checkpoints/meningioma_multimodal_best.pth`
- **Architecture**: 3D U-Net (`in_channels=4`, `num_classes=4`, `base_channels=16`, `InstanceNorm3d`)
- **Parameters**: 1,461,636 parameters (~17.6 MB)
- **SHA-256**: `D65C60DFFB5FBA1DFFD176C364DC5BA17D55E7D7AB2A2F8D378FD9F06B36F569`
- **Validation Benchmark (200 cases)**:
  - Macro WT Dice: **0.7991** | Median WT Dice: **0.9006** (IQR: 0.2092) | Pooled WT Dice: **0.9273**
  - Macro TC Dice: **0.8141** | Macro ET Dice: **0.8208**

> [!NOTE]
> **Cohort Distinction**: The 200-case benchmark (Macro WT Dice: **0.7991**, Median: **0.9006**, Pooled: **0.9273**) represents the full, independent validation cohort from `data/meningioma_multimodal/split.json`. During training, an in-training subset of 25 validation cases achieved **0.8312 Macro WT Dice** at epoch 25. The production service cites the full 200-case benchmark.

---

## 2. Input Requirements & Preprocessing

### Modality Channel Ordering
Inputs must provide four co-registered 3D MRI modalities:
1. `Channel 0 (t1n)`: Native T1-weighted MRI
2. `Channel 1 (t1c)`: Post-contrast T1-weighted MRI
3. `Channel 2 (t2w)`: T2-weighted MRI
4. `Channel 3 (t2f)`: T2-FLAIR MRI

### Spatial Geometry Validation
Before running inference, the service verifies:
- All four modalities exist and are readable NIfTI volumes.
- All four volumes share identical 3D matrix dimensions `(D, H, W)`.
- All four affine matrices match within a tolerance of `1e-3`.
- All voxel intensities are finite (`np.isfinite`).

### Intensity Normalization
Each modality channel is normalized independently using `normalize_nonzero_zscore`:
$$\text{normalized}[\text{vol} \ne 0] = \frac{\text{vol}[\text{vol} \ne 0] - \mu}{\sigma}$$
- Background/air voxels (`vol == 0`) remain strictly `0.0`.
- Applied strictly per-channel and per-scan (zero cross-case leakage).

---

## 3. Inference & Post-Processing Specifications

### Sliding-Window Inference
- **ROI Size**: `(64, 64, 64)`
- **Sliding-Window Batch Size**: `2` (safe for 6 GB VRAM GPUs)
- **Overlap**: `0.50`
- **Blending Mode**: Gaussian importance weighting
- **Precision**: Automatic Mixed Precision (`torch.amp.autocast`)

### Empty-Air Masking
Predictions are masked against background air voxels where all four input channels are zero:
$$\text{is\_air} = \bigwedge_{c=0}^{3} (\text{image}_c == 0)$$
$$\text{prediction}[\text{is\_air}] = 0 \quad (\text{Background})$$
Empty-air masking is applied to the output predictions only; the neural network receives unaltered normalized inputs.

### Multiclass Argmax & Decision Rule
The four-channel multimodal network outputs logits across 4 discrete classes. Inference selects discrete class assignments via `torch.argmax(logits, dim=1)`. Because argmax directly assigns discrete classes without evaluating a sigmoid or softmax probability cutoff, no probability threshold is applied (`threshold_applied = 0.0`).

### Connected-Component Filtering Policy
The formal 200-case evaluation was performed with no connected-component noise removal (`min_component_size = 0`). The deployed service maintains `min_component_size_voxels = 0` to ensure exact parity with the audited validation benchmark.

### Tumor Subregions & Label Definitions
- Class `0`: Background
- Class `1`: Necrotic Tumor Core (NCR)
- Class `2`: Peritumoral Edema (ED)
- Class `3`: Enhancing Tumor (ET)
- **Whole Tumor (WT)**: Classes `1`, `2`, `3`
- **Tumor Core (TC)**: Classes `1`, `3`
- **Enhancing Tumor (ET)**: Class `3`

### Separate Multiclass and Binary GTV Masks
To eliminate semantic ambiguity across client integrations:
1. **Multiclass Mask** (`*_seg.nii.gz`): Retained under `meningioma_multimodal.masks.multiclass_mask` containing discrete subregion labels `[0, 1, 2, 3]`.
2. **Binary GTV Mask** (`*_gtv.nii.gz`): Generated as `(multiclass_mask > 0).astype(uint8)` and saved under `meningioma.gtv_mask` with strictly binary labels `0` (Background) and `1` (Whole Tumor / GTV composite). Both masks share identical dimensions, voxel spacing, and affine matrices.

---

## 4. API Endpoints

### 1. File Upload: `POST /api/upload`
Uploads four NIfTI files (`.nii` or `.nii.gz`).

**Request (multipart/form-data)**:
```bash
curl -X POST http://127.0.0.1:5000/api/upload \
  -F "files=@BraTS-MEN-00001-000-t1n.nii.gz" \
  -F "files=@BraTS-MEN-00001-000-t1c.nii.gz" \
  -F "files=@BraTS-MEN-00001-000-t2w.nii.gz" \
  -F "files=@BraTS-MEN-00001-000-t2f.nii.gz"
```

**Response**:
```json
{
  "request_id": "REQ-7B1A2C3D",
  "status": "uploaded",
  "files_count": 4,
  "files": [
    {"filename": "BraTS-MEN-00001-000-t1n.nii.gz", "inferred_modality": "T1", "size_bytes": 4821900},
    {"filename": "BraTS-MEN-00001-000-t1c.nii.gz", "inferred_modality": "T1c", "size_bytes": 4912000},
    {"filename": "BraTS-MEN-00001-000-t2w.nii.gz", "inferred_modality": "T2", "size_bytes": 4791000},
    {"filename": "BraTS-MEN-00001-000-t2f.nii.gz", "inferred_modality": "FLAIR", "size_bytes": 4850000}
  ],
  "detected_modalities": {
    "T1": "BraTS-MEN-00001-000-t1n.nii.gz",
    "T1c": "BraTS-MEN-00001-000-t1c.nii.gz",
    "T2": "BraTS-MEN-00001-000-t2w.nii.gz",
    "FLAIR": "BraTS-MEN-00001-000-t2f.nii.gz"
  }
}
```

---

### 2. Dedicated Multimodal Segmentation: `POST /api/segment/meningioma-multimodal`
Dedicated endpoint retained as a fully compatible alias to the 4-channel multimodal pipeline.

**Request**:
```json
{
  "request_id": "REQ-7B1A2C3D",
  "patient_id": "BraTS-MEN-00001",
  "modality_map": {
    "t1n": "BraTS-MEN-00001-000-t1n.nii.gz",
    "t1c": "BraTS-MEN-00001-000-t1c.nii.gz",
    "t2w": "BraTS-MEN-00001-000-t2w.nii.gz",
    "t2f": "BraTS-MEN-00001-000-t2f.nii.gz"
  }
}
```

*Note: `modality_map` is optional if standard filenames containing `t1n`, `t1c`, `t2w`, `t2f` (or `t1`, `t1ce`, `t2`, `flair`) were uploaded.*

---

### 3. Unified Segmentation Route: `POST /api/segment`
The primary unified segmentation route routes both `tumor_type: "meningioma"` and `tumor_type: "meningioma_multimodal"` to the 4-channel multimodal service:
```json
{
  "request_id": "REQ-7B1A2C3D",
  "tumor_type": "meningioma"
}
```

> [!IMPORTANT]
> **Strict Modality Requirement**: Meningioma requests require all four modalities (`t1n`, `t1c`, `t2w`, `t2f`). If any modality is missing, the API responds with HTTP 400 (`"Meningioma segmentation requires 4 modalities (t1n, t1c, t2w, t2f). Missing: ..."`). Silent fallback to 1-channel inference is permanently disabled.

---

### 4. Response Schema (`SegmentationResult`)
```json
{
  "request": {
    "request_id": "REQ-7B1A2C3D",
    "tumor_type": "meningioma_multimodal",
    "patient_id": "BraTS-MEN-00001",
    "session_id": null,
    "timestamp": "2026-10-15T05:30:00Z"
  },
  "input_metadata": {
    "modality_names": ["t1n", "t1c", "t2w", "t2f"],
    "original_shape": [240, 240, 155],
    "original_spacing": [1.0, 1.0, 1.0],
    "voxel_volume_mm3": 1.0,
    "resampled_shape": [240, 240, 155],
    "resampled_spacing": [1.0, 1.0, 1.0]
  },
  "volumetrics": {
    "total_tumor_voxels": 45210,
    "total_tumor_volume_mm3": 45210.0,
    "total_tumor_volume_cm3": 45.21,
    "voxel_dimensions_mm": [1.0, 1.0, 1.0]
  },
  "quality_status": {
    "status": "success",
    "inference_success": true,
    "pipeline_frozen": true,
    "checkpoint_used": "outputs/meningioma_multimodal_experiment/full800_brainaware_20261014/checkpoints/meningioma_multimodal_best.pth",
    "threshold_applied": 0.0,
    "min_component_size_voxels": 0,
    "warnings": [],
    "qc_verdict": "PASS WITH VALIDATION BENCHMARK (0.7991 Macro WT Dice / 0.9006 Median WT Dice)",
    "qc_notes": "4-channel multimodal 3D U-Net evaluated on 200 validation cases (Macro WT Dice: 0.7991, Median WT Dice: 0.9006, Pooled WT Dice: 0.9273; note: in-training 25-case validation subset achieved 0.8312 Macro WT Dice). Multiclass argmax inference with empty-air masking; no connected-component filtering applied. Research prototype only. Not approved for primary clinical diagnosis or medical decision making."
  },
  "meningioma_multimodal": {
    "subregions": {
      "wt_voxels": 45210,
      "wt_volume_mm3": 45210.0,
      "wt_volume_cm3": 45.21,
      "tc_voxels": 38100,
      "tc_volume_mm3": 38100.0,
      "tc_volume_cm3": 38.10,
      "et_voxels": 35400,
      "et_volume_mm3": 35400.0,
      "et_volume_cm3": 35.40,
      "ncr_net_voxels": 2700,
      "ncr_net_volume_mm3": 2700.0,
      "ncr_net_volume_cm3": 2.70,
      "edema_voxels": 7110,
      "edema_volume_mm3": 7110.0,
      "edema_volume_cm3": 7.11
    },
    "masks": {
      "multiclass_mask": {
        "mask_path": "outputs/api_runs/REQ-7B1A2C3D/masks/REQ-7B1A2C3D_meningioma_seg.nii.gz",
        "mask_format": "nifti",
        "is_original_space": true,
        "shape": [240, 240, 155],
        "non_zero_voxels": 45210
      }
    },
    "label_definitions": {
      "0": "Background",
      "1": "Necrotic Tumor Core (NCR)",
      "2": "Peritumoral Edema (ED)",
      "3": "Enhancing Tumor (ET)"
    },
    "is_original_space_restored": true,
    "disclaimer": "Research prototype only. Not approved for primary clinical diagnosis or medical decision making."
  },
  "meningioma": {
    "gtv_voxels": 45210,
    "gtv_volume_mm3": 45210.0,
    "gtv_volume_cm3": 45.21,
    "gtv_mask": {
      "mask_path": "outputs/api_runs/REQ-7B1A2C3D/masks/REQ-7B1A2C3D_meningioma_gtv.nii.gz",
      "mask_format": "nifti",
      "is_original_space": true,
      "shape": [240, 240, 155],
      "non_zero_voxels": 45210
    },
    "is_original_space_restored": true,
    "label_definitions": {
      "0": "Background",
      "1": "Whole Tumor (GTV composite)"
    }
  }
}
```

---

### 5. Download Segmentation Masks: `GET /api/mask/<request_id>`
Downloads either the 3D multiclass mask or the binary GTV mask via the optional `type` query parameter:

- **Multiclass Mask** (Default / `type=multiclass`): Labels 0, 1, 2, 3
  ```bash
  curl -O http://127.0.0.1:5000/api/mask/REQ-7B1A2C3D
  ```

- **Binary GTV Mask** (`type=gtv`): Strictly labels 0 and 1
  ```bash
  curl -O "http://127.0.0.1:5000/api/mask/REQ-7B1A2C3D?type=gtv"
  ```

- **Invalid Type Handling**: Requesting an unsupported mask type (e.g., `?type=invalid`) returns **HTTP 400 Bad Request** (`{"detail": "Invalid mask type '...'. Supported types: 'multiclass', 'gtv'."}`).

#### GTV Mask Semantics & Equivalence Verification
- The binary GTV mask contains strictly labels `0` (Background) and `1` (Whole Tumor / Gross Tumor Volume).
- Foreground equivalence is preserved: `(multiclass_mask > 0) == (gtv_mask == 1)` across all voxels.
- Both mask files share identical voxel dimensions `(D, H, W)`, voxel spacing, and affine transformation matrices.
- **Empirical Verification (`REQ-MEN-VERIFY-20261016-155143`)**: Multiclass labels `[0, 2, 3]`; GTV labels `[0, 1]`; volume shape `(240, 240, 155)`; matching diagonal affine; foreground equivalence verified across exactly 2,465 voxels. *(Note: These are verification results for the tested cases, not guarantees for every possible input.)*

---

## 5. Service Startup & Verification

### Start the REST API Server
```powershell
.\.venv\Scripts\python.exe api\app.py
```
Default URL: `http://127.0.0.1:5000`

### Verify Health Endpoint
```powershell
curl http://127.0.0.1:5000/api/health
```

### Run Isolated Test Suite
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_meningioma_multimodal_inference.py -v
```

### Run Entire Regression Test Suite
```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
**Current Test Status**: **86 passed, 2 warnings** (verified via `pytest -q`).

---

## 6. Regulatory & Research Scope Disclaimer

> [!CAUTION]
> **RESEARCH PROTOTYPE ONLY**:
> This software and its generated segmentation masks and volumetric estimates are intended strictly for scientific research and benchmarking purposes. They are not approved by the FDA, CE, or any regulatory body for primary diagnostic use, patient management, or clinical decision making.

