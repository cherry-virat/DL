# Deep Learning-Based 3D Brain Tumor Detection, Segmentation, and Quantitative Analysis

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c.svg)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-1.4+-green.svg)](https://monai.io/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-black.svg)](https://flask.palletsprojects.com/)
[![Tests](https://img.shields.io/badge/Tests-86%20passed%2C%202%20warnings-brightgreen.svg)]()

---

## 1. Project Objective & Overview

This project provides an end-to-end deep learning platform for automated 3D magnetic resonance imaging (MRI) brain tumor segmentation, subregion differentiation, and quantitative volumetric analysis. The system supports two distinct brain tumor pathologies using dedicated four-channel 3D convolutional neural networks:

1. **Adult Diffuse Glioma**: Segmented into Necrotic/Non-Enhancing Core (NCR/NET, label 1), Peritumoral Edema (ED, label 2), and Contrast-Enhancing Tumor (ET, label 3), yielding composite Whole Tumor (WT = 1 + 2 + 3) and Tumor Core (TC = 1 + 3).
2. **Meningioma**: Segmented into Necrotic Core (NCR, label 1), Peritumoral Edema (ED, label 2), and Contrast-Enhancing Tumor (ET, label 3), yielding composite Whole Tumor / Gross Tumor Volume (GTV = 1 + 2 + 3).

> [!CAUTION]
> **RESEARCH & BENCHMARKING PROTOTYPE ONLY**:
> This software and its generated segmentation masks and volumetric estimates are intended strictly for scientific research and algorithm benchmarking. They have not been cleared or approved by the FDA, CE, or any regulatory body for clinical diagnostic use, treatment planning, or patient management.

---

## 2. Models, Architectures & Active Checkpoints

Both models accept **four co-registered 3D MRI modalities** in standardized channel order:
- **Channel 0 (`t1n` / `T1`)**: Native T1-weighted MRI
- **Channel 1 (`t1c` / `T1c`)**: Post-contrast T1-weighted MRI
- **Channel 2 (`t2w` / `T2`)**: T2-weighted MRI
- **Channel 3 (`t2f` / `FLAIR`)**: T2-Fluid-Attenuated Inversion Recovery (FLAIR) MRI

Input tensor dimensions: `(Batch, 4, Depth, Height, Width)`.

### High-Level System Architecture & Flow

```
1. Input Validation & Preprocessing
   ├── Verify presence and integrity of 4 NIfTI modalities
   ├── Validate affine matrix and spatial dimensions match
   └── Apply per-channel non-zero Z-score intensity normalization
                    │
                    ▼
2. 3D Model Inference
   ├── MONAI sliding-window inference (ROI: 64x64x64, overlap: 0.50)
   ├── Gaussian importance blending across overlapping sub-volumes
   └── PyTorch Automatic Mixed Precision (AMP) on NVIDIA CUDA GPU
                    │
                    ▼
3. Multiclass Segmentation Mask
   ├── Glioma: Probability threshold (P(WT) >= 0.60) + 3D CC noise filter (<1000 vx)
   ├── Meningioma: Discrete argmax across 4 classes + Empty-air background masking
   └── Export 4-class NIfTI mask: *_seg.nii.gz (labels 0, 1, 2, 3)
                    │
                    ▼
4. Binary GTV Mask Output
   ├── Derived strictly as binary_gtv = (multiclass_mask > 0).astype(uint8)
   ├── Exact foreground equivalence with labels 0 (Background) and 1 (Tumor)
   └── Export binary NIfTI mask: *_gtv.nii.gz (identical shape, affine, spacing)
                    │
                    ▼
5. Unified API Response & Client Downloads
   ├── Standardized SegmentationResult JSON (volumes in cm³, QC metadata)
   ├── GET /api/mask/<id> (downloads multiclass mask)
   └── GET /api/mask/<id>?type=gtv (downloads binary GTV mask)
```

| Dimension | Glioma Segmentation Pipeline | Meningioma Segmentation Pipeline |
| :--- | :--- | :--- |
| **Pathology Target** | Adult Diffuse Glioma (ASNR-MICCAI BraTS GLI) | Meningioma (BraTS MEN-RT) |
| **Model Architecture** | MONAI 3D Residual U-Net | Custom 4-Channel 3D U-Net |
| **Channels / Capacity** | `(16, 32, 64, 128, 256)`, 2 residual units/level | `base_channels=16`, `InstanceNorm3d` |
| **Parameters** | 1,461,636 parameters | 1,461,636 parameters (~17.6 MB) |
| **Active Checkpoint Path** | [`outputs/glioma_training/glioma_best_medium_experiment.pth`](outputs/glioma_training/glioma_best_medium_experiment.pth) | [`outputs/meningioma_multimodal_experiment/full800_brainaware_20261014/checkpoints/meningioma_multimodal_best.pth`](outputs/meningioma_multimodal_experiment/full800_brainaware_20261014/checkpoints/meningioma_multimodal_best.pth) |
| **Intensity Normalization** | Per-channel non-zero $Z$-score normalization | Per-channel non-zero $Z$-score normalization |
| **Spatial Spacing** | $1.0\,\text{mm}$ isotropic resampling | Retains BraTS acquisition spacing ($1.0\,\text{mm}$) |
| **Sliding-Window Inference** | ROI $(64, 64, 64)$, overlap $0.50$, Gaussian weighting | ROI $(64, 64, 64)$, overlap $0.50$, Gaussian, AMP enabled |
| **Decision Rule** | $P(\text{WT}) \ge 0.60$ thresholding | Direct discrete multiclass `argmax` |
| **Connected Components** | 3D 6-connectivity, removes clusters $<1000$ voxels | None ($\text{min\_size}=0$, raw boundary preservation) |
| **Empty-Air Masking** | Handled by probability threshold + CC filter | Applied: $\bigwedge_{c=0}^3 I_c == 0 \implies \text{Pred}=0$ |
| **Active Routing Role** | Sole active glioma model | Sole active meningioma model (1-channel decommissioned) |

---

## 3. Verified Metric Reporting & Evaluation Provenance

All metrics are reported from the project's frozen evaluation artifacts. No metrics are fabricated or derived from unverified runs.

### 3.1 Glioma Evaluation Benchmark (20 Validation Cases)

Evaluated via [`segmentation/glioma_formal_evaluation.py`](segmentation/glioma_formal_evaluation.py) and permanently recorded in [`outputs/glioma_formal_evaluation/evaluation_summary.md`](outputs/glioma_formal_evaluation/evaluation_summary.md):

| Target Subregion | Macro Mean Dice ($\pm$ Std) | Macro Median Dice (IQR) | Micro / Global Pooled Dice |
| :--- | :---: | :---: | :---: |
| **Whole Tumor (WT)** | **`0.8478`** ($\pm 0.1466$) | **`0.9013`** ($0.0619$) | **`0.9042`** |
| **Tumor Core (TC)** | **`0.7240`** ($\pm 0.1997$) | **`0.7871`** ($0.2172$) | **`0.7666`** |
| **Enhancing Tumor (ET)** | **`0.7072`** ($\pm 0.2295$) | **`0.7874`** ($0.2338$) | **`0.8401`** |

- **Volumetric Concordance**: Pearson $r = \mathbf{0.9842}$ ($p = 5.45 \times 10^{-15}$), Linear $R^2 = \mathbf{0.9666}$, $\text{MAE} = 6,148.5\,\text{mm}^3$.
- **False Positive Elimination**: Connected-component filtering discarded $1,135$ spurious noise clusters ($98.02\%$ reduction), removing $9,478$ FP voxels.
- **Provenance & Methodology Disclosure**:
  - The evaluation was conducted on the **20 validation cases** from [`preprocessing/glioma_split.py`](preprocessing/glioma_split.py) (`seed=42`).
  - These exact 20 cases were evaluated during training in [`segmentation/glioma_train_medium.py`](segmentation/glioma_train_medium.py) to select `glioma_best_medium_experiment.pth`. In addition, cases from this cohort were used to tune the probability threshold ($\tau=0.60$) and cluster size ($1{,}000$ voxels).
  - Consequently, these metrics must strictly be reported as **Internal Validation / Development Results**, not independent held-out test results.
- **Clarification on `0.8529`**: The number `0.8529` is an **Intersection-over-Union (IoU)** metric from a connected-component sensitivity sweep at `min_size=500` ([`outputs/glioma_connected_component_analysis/connected_component_results.txt`](outputs/glioma_connected_component_analysis/connected_component_results.txt#L14)). It is **not** a macro WT Dice score.

### 3.2 Meningioma Evaluation Benchmark (200 Holdout Validation Cases)

Evaluated via [`scripts/evaluate_meningioma_multimodal_final200.py`](scripts/evaluate_meningioma_multimodal_final200.py) and permanently recorded in [`outputs/meningioma_multimodal_experiment/final_validation_20261015/`](outputs/meningioma_multimodal_experiment/final_validation_20261015/):

| Target Subregion | Macro Mean Dice ($\pm$ Std) | Macro Median Dice (IQR) | Micro / Global Pooled Dice |
| :--- | :---: | :---: | :---: |
| **Whole Tumor (WT)** | **`0.7991`** ($\pm 0.2361$) | **`0.9006`** ($0.2092$) | **`0.9273`** |
| **Tumor Core (TC)** | **`0.8141`** ($\pm 0.2343$) | **`0.9167`** ($0.1983$) | **`0.9329`** |
| **Enhancing Tumor (ET)** | **`0.8208`** ($\pm 0.2324$) | **`0.9234`** ($0.1947$) | **`0.9360`** |

- **Provenance & Checkpoint Attribution**:
  - These metrics represent the post-hoc evaluation across all **200 holdout validation cases** in [`data/meningioma_multimodal/split.json`](data/meningioma_multimodal/split.json) using the primary checkpoint (`outputs/.../checkpoints/meningioma_multimodal_best.pth`, SHA-256 `D65C60DF...`, Epoch 25).
  - This 200-case holdout result must not be confused with:
    1. The in-training 25-case validation subset monitoring score (**`0.8312` Macro WT Dice**).
    2. The separate October 15 rerun checkpoint (**`0.8138` Macro WT Dice** from online validation).

---

## 4. REST API Server & Inference Workflow

The Flask API provides an asynchronous-compatible file upload, automated validation, routed segmentation execution, and dual-mask download service.

### 4.1 Starting the API Server
```powershell
.\.venv\Scripts\python.exe api\app.py
```
The server binds to `http://127.0.0.1:5000`.

### 4.2 Health Check Endpoint
```bash
curl http://127.0.0.1:5000/api/health
```
**Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "device": "cuda:0",
  "gpu_name": "NVIDIA GeForce RTX 3050 Laptop GPU",
  "supported_tumor_types": ["glioma", "meningioma"],
  "supported_formats": [".nii", ".nii.gz"]
}
```

### 4.3 End-to-End Inference Workflow

```
1. Client Upload           2. Segmentation Execution           3. Retrieval & Download
   POST /api/upload           POST /api/segment                  GET /api/results/<id>
  (4 MRI NIfTI files)   ───> (tumor_type, request_id)     ───>   GET /api/mask/<id>
                                                                 GET /api/mask/<id>?type=gtv
```

#### Step 1: Upload Raw Scans (`POST /api/upload`)
Uploads 4 NIfTI files via `multipart/form-data`:
```powershell
curl -X POST http://127.0.0.1:5000/api/upload `
  -F "files=@BraTS-GLI-00022-000-t1n.nii.gz" `
  -F "files=@BraTS-GLI-00022-000-t1c.nii.gz" `
  -F "files=@BraTS-GLI-00022-000-t2w.nii.gz" `
  -F "files=@BraTS-GLI-00022-000-t2f.nii.gz"
```
Returns a unique `request_id` (e.g. `REQ-A1B2C3D4`) and automatically detected modalities.

#### Step 2: Run Segmentation (`POST /api/segment`)
Executes routed inference:
```powershell
curl -X POST http://127.0.0.1:5000/api/segment `
  -H "Content-Type: application/json" `
  -d '{"request_id": "REQ-A1B2C3D4", "tumor_type": "glioma"}'
```
*(For meningioma, specify `"tumor_type": "meningioma"`. Dedicated endpoints `POST /api/segment/glioma` and `POST /api/segment/meningioma-multimodal` are also available.)*

#### Step 3: Retrieve Results (`GET /api/results/<request_id>`)
Retrieves complete volumetric quantification and metadata adhering to `integration/result_schema.py`.

#### Step 4: Storage Structure
All run artifacts are saved under an isolated request folder:
```
outputs/api_runs/<request_id>/
├── uploads/       # Raw uploaded MRI files
├── masks/
│   ├── <request_id>_<tumor>_seg.nii.gz   # Multiclass segmentation mask
│   └── <request_id>_<tumor>_gtv.nii.gz   # Binary GTV mask
└── result.json    # Serialized SegmentationResult payload
```

---

## 5. Verified GTV Semantics & Mask Download Endpoints

The API distinguishes multiclass segmentation from binary Gross Tumor Volume (GTV) masks:

### Download Routes
- **Multiclass Segmentation Mask** (Default):
  ```powershell
  curl -O http://127.0.0.1:5000/api/mask/<request_id>
  # Equivalent: curl -O http://127.0.0.1:5000/api/mask/<request_id>?type=multiclass
  ```
  Returns `*_seg.nii.gz` containing discrete labels `[0, 1, 2, 3]` (0: Background, 1: NCR/NET, 2: ED, 3: ET).
- **Binary Gross Tumor Volume (GTV) Mask**:
  ```powershell
  curl -O "http://127.0.0.1:5000/api/mask/<request_id>?type=gtv"
  ```
  Returns `*_gtv.nii.gz` containing strictly binary labels `0` (Background) and `1` (Whole Tumor / GTV composite).
- **Invalid Type Rejection**:
  Requesting any unsupported mask type (e.g. `?type=invalid`) returns an explicit **HTTP 400 Bad Request** (`{"detail": "Invalid mask type '...'. Supported types: 'multiclass', 'gtv'."}`).

### Mask Properties & Foreground Equivalence
1. **Strict Label Constraints**: The binary GTV mask contains only labels `0` and `1`.
2. **Foreground Equivalence**: Across all voxels, the binary GTV foreground strictly equals the non-zero foreground of the multiclass mask:
   $$(\text{multiclass\_mask} > 0) \equiv (\text{gtv\_mask} == 1)$$
3. **Geometry Invariance**: Multiclass and binary GTV masks share identical matrix dimensions `(D, H, W)`, voxel spacing, origin, and affine direction matrices.

### Empirical Verification on Real Inference Runs
Real live smoke tests verified these properties on physical GPU inference:
- **Glioma Case (`REQ-GLIOMA-VERIFY-20261016-160027`)**:
  - Multiclass labels: `[0, 1, 2, 3]` (NCR: 20,245, ED: 75,839, ET: 37,577 voxels).
  - Binary GTV labels: `[0, 1]` (Total GTV: 133,661 voxels $\equiv 133.661\,\text{cm}^3$).
  - Spatial shape: `(240, 240, 155)`, matching affine matrix.
  - Foreground equivalence: `np.array_equal((seg > 0), (gtv == 1)) == True`.
- **Meningioma Case (`REQ-MEN-VERIFY-20261016-155143`)**:
  - Multiclass labels: `[0, 2, 3]` (ED: 2,367, ET: 98 voxels).
  - Binary GTV labels: `[0, 1]` (Total GTV: 2,465 voxels $\equiv 2.465\,\text{cm}^3$).
  - Spatial shape: `(240, 240, 155)`, matching affine matrix.
  - Foreground equivalence: `np.array_equal((seg > 0), (gtv == 1)) == True`.

*(Note: These empirical checks confirm correctness for the verified cases; they reflect technical pipeline verification rather than guarantees across every unconstrained input).*

---

## 6. Automated Regression Test Suite

The test suite validates schema immutability, modality enforcement, router dispatch, multiclass and GTV mask generation, binary equivalence, air masking, and REST endpoints.

### Running the Tests
```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

### Verified Test Result
```
........................................................................ [ 83%]
..............                                                           [100%]
86 passed, 2 warnings in 21.13s
```
*(The 2 warnings are PyTorch `torch.jit.interface` deprecation notices).*

### Key Test Suites
- [`tests/test_api.py`](tests/test_api.py): REST endpoints, uploads, error handling, default multiclass download, `?type=gtv` binary download, and HTTP 400 rejection on invalid type.
- [`tests/test_inference_services.py`](tests/test_inference_services.py): `GliomaInferenceService` and `MeningiomaInferenceService` synthetic inference, binary GTV mask equivalence, and QC notes provenance.
- [`tests/test_inference_router.py`](tests/test_inference_router.py): Strict 4-modality requirement, error handling, and routing.
- [`tests/test_meningioma_multimodal_inference.py`](tests/test_meningioma_multimodal_inference.py): 4-channel multimodal pipeline, separate mask files, and 200-case QC benchmarks.
- [`tests/test_result_schema.py`](tests/test_result_schema.py): Data contract serialization, volume hierarchy ($WT = NCR + ED + ET$), and immutability.

---

## 7. Current Operational Limitations

1. **Glioma Small-Lesion Filtering**: The 3D 6-connectivity connected component filter removes clusters smaller than $1{,}000$ voxels ($1.0\,\text{cm}^3$). While this successfully removes $98.02\%$ of spurious false-positive noise, it intentionally suppresses true micro-foci or satellite lesions $< 1.0\,\text{cm}^3$.
2. **Meningioma Raw Boundary Decisions**: The multimodal meningioma pipeline applies discrete argmax with empty-air masking and no connected-component filtering ($\text{min\_size} = 0$). While small dural lesions are preserved, peripheral over-prediction along skull base dura or venous sinuses can occur.
3. **Cohort Independence & Generalization**:
   - Glioma evaluation metrics reflect the **20-case internal validation split** used during development for checkpoint selection and post-processing tuning. They do not represent multi-center external test validation.
   - Meningioma metrics reflect an internal single-source 200-case holdout partition from `split.json`.
4. **Clinical Scope**: Both models are strictly research prototypes for algorithm benchmarking; they have not undergone clinical trials and are not approved for primary diagnostic or therapeutic decisions.

---

## 8. Shortest Safe Demonstration Sequence

Demonstrate the active platform without altering checkpoints or retraining:

```powershell
# 1. Start the Flask REST API Server
.\.venv\Scripts\python.exe api\app.py

# 2. Verify API Health (reports status, CUDA device, RTX 3050, and 4 modalities)
curl http://127.0.0.1:5000/api/health

# 3. Retrieve Verified Smoke-Test Result (Glioma real inference run)
curl http://127.0.0.1:5000/api/results/REQ-GLIOMA-VERIFY-20261016-160027

# 4. Download Multiclass Segmentation Mask (Labels 0, 1, 2, 3)
curl -O http://127.0.0.1:5000/api/mask/REQ-GLIOMA-VERIFY-20261016-160027

# 5. Download Binary GTV Mask (Labels 0 and 1 only; foreground equivalent)
curl -O "http://127.0.0.1:5000/api/mask/REQ-GLIOMA-VERIFY-20261016-160027?type=gtv"
```

---

## 9. Technology Stack

- **Deep Learning**: PyTorch 2.5+, MONAI 1.4+
- **Medical Imaging I/O & Preprocessing**: SimpleITK, NiBabel, SciPy, NumPy
- **REST API Backend**: Flask 3.0+, Flask-CORS, Werkzeug
- **Testing & Verification**: Pytest 8.3+ (**86 passed, 2 warnings in 21.13s**)
- **Target Hardware Acceleration**: NVIDIA CUDA (NVIDIA GeForce RTX 3050 Laptop GPU verified)

---

## 10. Repository Directory Structure

```
C:\Users\nchan\Desktop\DL\
├── api/
│   └── app.py                             # Flask REST API server
├── docs/
│   ├── dataset_specification.md           # Dataset modality and shape specs
│   ├── evaluation_audit_20261016.md       # Audit report on splits and hashes
│   ├── final_model_comparison_20261016.md # Formal comparative model report
│   ├── GLIOMA_FINAL_EVALUATION.md         # 20-case formal glioma evaluation guide
│   ├── meningioma_multimodal_inference.md # Multimodal meningioma integration guide
│   └── segmentation_error_analysis_...md  # Granular error and failure mode analysis
├── integration/
│   ├── glioma_inference_service.py        # Frozen 4-channel Glioma service
│   ├── meningioma_multimodal_service.py   # Frozen 4-channel Meningioma service
│   ├── inference_router.py                # Unified request router
│   └── result_schema.py                   # Master common result schema
├── models/
│   └── meningioma_multimodal_unet.py      # 4-channel 3D U-Net architecture
├── outputs/
│   ├── api_runs/                          # Isolated REST request execution runs
│   ├── glioma_formal_evaluation/          # Metrics, summary, and per-case CSV
│   ├── glioma_training/                   # glioma_best_medium_experiment.pth
│   └── meningioma_multimodal_experiment/  # meningioma_multimodal_best.pth
├── preprocessing/                         # NIfTI loading, resampling, splits
├── segmentation/                          # Formal evaluation & training scripts
└── tests/                                 # 86 unit and integration regression tests
```

---

## 11. License & Attribution

- **Challenge Annotations**: Ground-truth masks and multi-parametric MRI sequences are provided by the ASNR-MICCAI BraTS 2023 Glioma Challenge and BraTS Meningioma Challenge cohorts.
- **Frameworks**: Built with [MONAI](https://monai.io/), [PyTorch](https://pytorch.org/), [SimpleITK](https://simpleitk.org/), and [NiBabel](https://nipy.org/nibabel/).