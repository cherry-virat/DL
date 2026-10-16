# Technical Audit: 4-Channel 3D U-Net Evaluation Pipelines (Glioma & Multimodal Meningioma)

**Audit Date:** 2026-10-16  
**Auditor:** Antigravity AI Engine  
**Project Root:** `C:\Users\nchan\Desktop\DL`  
**Status:** Complete & Verified  

---

## 1. Executive Summary & Master Audit Verdict

This audit performs an exhaustive, line-by-line verification of the two primary 4-channel 3D U-Net brain tumor segmentation evaluation pipelines:
1. **Glioma Evaluation Pipeline:** [`segmentation/glioma_formal_evaluation.py`](file:///C:/Users/nchan/Desktop/DL/segmentation/glioma_formal_evaluation.py)
2. **Multimodal Meningioma Evaluation Pipeline:** [`scripts/evaluate_meningioma_multimodal_final200.py`](file:///C:/Users/nchan/Desktop/DL/scripts/evaluate_meningioma_multimodal_final200.py)

### Master Audit Findings
- **Architectural & Channel Congruence:** Both pipelines correctly instantiate 4-input-channel, 4-output-class 3D U-Nets. The input modality ordering (`[t1n, t1c, t2w, t2f]`) is identical between training and evaluation for both models.
- **Glioma Split Status (CRITICAL CLARIFICATION):** The 20 cases evaluated by `glioma_formal_evaluation.py` are **NOT an independent test cohort**. They constitute the **internal validation cohort** used during training for model selection (`BEST_MODEL_PATH`), and the first 5 of these cases were used to tune the probability threshold ($\tau=0.60$) and connected-component size threshold ($1,000$ voxels). They must strictly be designated as an **Internal Validation Set**.
- **Meningioma Split Integrity:** The partition in [`data/meningioma_multimodal/split.json`](file:///C:/Users/nchan/Desktop/DL/data/meningioma_multimodal/split.json) has exactly 800 training cases and 200 validation cases with **zero overlap**, zero duplicates, and 1,000 fully verified `case_details` keys.
- **Meningioma Discrepancy Reconciled (79.91% vs. 81.38%):** The two scores originate from **two distinct training runs and checkpoints**. 
  - **$79.91\%$ Macro WT Dice** is the post-hoc 200-case evaluation of the Oct 14 checkpoint (`full800_brainaware_20261014`, Epoch 25).
  - **$81.38\%$ Macro WT Dice** is the online 200-case validation score achieved at Epoch 22 of the Oct 15 rerun checkpoint (`full800_brainaware_rerun_20261015`).
  These checkpoints have distinct SHA-256 hashes and must not be conflated.
- **Metric Mathematics:** Metric calculations for WT, TC, and ET Dice, Precision, and Recall are mathematically sound and correctly handle empty ground-truth and empty prediction edge cases.
- **Code & Weight Preservation:** Zero model weights, dataset files, or split files were modified or staged.

---

## 2. Checkpoint Manifest, Architectures & Cryptographic Hashes

Every evaluated model checkpoint has been verified on disk and hashed via SHA-256:

| Pipeline | Target Checkpoint Path | Epoch | Parameters | SHA-256 Checksum |
| :--- | :--- | :---: | :---: | :--- |
| **Glioma 4-Channel U-Net** | `outputs/glioma_training/glioma_best_medium_experiment.pth` | 20 | 1,461,636 | `7140007B90E730FCDD0F1C9EF2EFB5B7DA3C6613461DEB0F2299AB558786C9B5` |
| **Meningioma 4-Channel (Run 1)** | `outputs/meningioma_multimodal_experiment/full800_brainaware_20261014/checkpoints/meningioma_multimodal_best.pth` | 25 | 1,461,636 | `D65C60DFFB5FBA1DFFD176C364DC5BA17D55E7D7AB2A2F8D378FD9F06B36F569` |
| **Meningioma 4-Channel (Run 2 Rerun)** | `outputs/meningioma_multimodal_experiment/full800_brainaware_rerun_20261015/checkpoints/meningioma_multimodal_best.pth` | 22 | 1,461,636 | `B32D848129B247FEEBD1620902C4F57B5B562C17978263A45C94B8DD94138A12` |
| **Meningioma Single-Channel Baseline** | `outputs/meningioma_training/meningioma_best.pth` | Best | 1,458,898 | `78F419A24D538E7057A22DE3BD0352193E41184A1FD2AB6F118D213A5B0FD301` |

### Channel Ordering Verification
Both pipelines strictly enforce the standardized 4-modality channel mapping:
- **Channel 0:** Native T1-weighted (`t1n` / `T1`)
- **Channel 1:** Contrast-enhanced T1-weighted (`t1c` / `T1c`)
- **Channel 2:** T2-weighted (`t2w` / `T2`)
- **Channel 3:** T2-FLAIR (`t2f` / `FLAIR`)

Input tensor shape to both neural networks: `(Batch, 4, Depth, Height, Width)`.

---

## 3. Dataset Splits, Provenance & Overlap Verification

### 3.1 Glioma Dataset Split Trace
- **Source Repository:** `C:\3D_DATASET\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData` ($N=1,251$ multi-parametric cases).
- **Split Implementation:** [`preprocessing/glioma_split.py`](file:///C:/Users/nchan/Desktop/DL/preprocessing/glioma_split.py) via `create_train_val_split(cases, train_ratio=0.90, seed=42)`.
- **Training Subset:** First 100 cases from the training split (`train_cases[:100]`).
- **Validation Subset:** First 20 cases from the validation split (`validation_cases[:20]`).
- **Overlap Check:**
  $$\text{Train Cases} \cap \text{Validation Cases} = \emptyset \quad (\text{Zero Overlap Verified})$$

#### Independence & Data Leakage Audit for Glioma
An inspection of `segmentation/glioma_train_medium.py` reveals:
1. Lines 1805–1842: `save_checkpoint(..., path=BEST_MODEL_PATH)` updates `glioma_best_medium_experiment.pth` whenever validation Dice on these **exact 20 cases** improves during training.
2. An inspection of `segmentation/glioma_threshold_analysis.py` (lines 88, 814–829) and `segmentation/glioma_connected_component_analysis.py` (lines 90, 978–993) reveals that the first **5 cases** of these 20 validation cases were evaluated across candidate thresholds $\tau \in [0.10, 0.90]$ and component filters to select $\tau = 0.60$ and cluster size $1,000$ voxels.
3. **Verdict:** The 20 glioma cases were directly involved in **checkpoint selection** and **hyperparameter tuning**. They must be described strictly as an **Internal Validation / Development Set**, and never as an independent external test set.

### 3.2 Meningioma Dataset Split Trace
- **Source File:** [`data/meningioma_multimodal/split.json`](file:///C:/Users/nchan/Desktop/DL/data/meningioma_multimodal/split.json).
- **Cohort Counts:**
  - `train_cases`: **800 cases**
  - `val_cases`: **200 cases**
  - `total_cases`: **1,000 cases**
- **Set Theory & Integrity Validation:**
  - `train_cases ∩ val_cases` = **0 cases** (Zero Overlap)
  - Duplicate entries in `train_cases`: **0**
  - Duplicate entries in `val_cases`: **0**
  - Metadata `case_details` entries: **1,000** ($100\%$ key matching)
- **Verdict:** Clean, deterministic patient-level partition with zero leakage.

---

## 4. Preprocessing & Post-Processing Pipeline Verification

| Pipeline Dimension | Glioma (`glioma_formal_evaluation.py`) | Meningioma (`evaluate_meningioma_multimodal_final200.py`) |
| :--- | :--- | :--- |
| **Image Resampling** | $1.0\,\text{mm}$ isotropic spacing via SimpleITK linear interpolation | Retains native BraTS acquisition spacing $(1.0\,\text{mm})$ |
| **Intensity Normalization** | Robust min-max scaling ($1^{\text{st}} - 99^{\text{th}}$ percentiles to $[0, 1]$ within non-zero brain) | Non-zero Z-score normalization per channel: $\frac{x - \mu}{\sigma}$ |
| **Sliding Window ROI** | $(64, 64, 64)$ | $(64, 64, 64)$ |
| **Sliding Window Overlap**| $0.50$ (Gaussian importance weighting) | $0.50$ (Gaussian importance weighting) |
| **Inference Batch Size** | 1 | 2 (with Automatic Mixed Precision) |
| **Probability Threshold** | $\tau = 0.60$ on aggregate tumor probability $P(1)+P(2)+P(3)$ | None (Direct `torch.argmax(logits, dim=1)`) |
| **Connected Components** | 3D 6-connectivity; discards clusters $< 1,000\,\text{voxels}$ | None |
| **Empty-Air Masking** | Not applied (handled by $\tau=0.60$ + CC filter) | **Applied** ($\bigwedge_{c=0}^3 I_c == 0 \implies \text{Pred} = 0$) |
| **Air False Positives** | Suppressed by CC filter | **0 voxels** across all 200 cases ($100\%$ eliminated) |

---

## 5. Metric Definitions, Calculations & Edge-Case Handling

Both evaluation scripts calculate subregion metrics according to the official BraTS standard:

### 5.1 Subregion Formulations
- **Whole Tumor (WT):** Labels $1 + 2 + 3$ (Necrotic Core + Edema + Enhancing Tumor)
- **Tumor Core (TC):** Labels $1 + 3$ (Necrotic Core + Enhancing Tumor)
- **Enhancing Tumor (ET):** Label $3$ (Active Contrast-Enhancing Core)

### 5.2 Micro vs. Macro vs. Median Formulations
- **Macro Average:**
  $$\text{Macro Metric} = \frac{1}{N} \sum_{i=1}^N \text{Metric}_i$$
  Calculates metrics on each patient independently and takes the unweighted arithmetic mean. Highly sensitive to small lesions.
- **Median Metric:**
  $$\text{Median Metric} = \text{Median}(\{\text{Metric}_1, \text{Metric}_2, \dots, \text{Metric}_N\})$$
  Represents central tendency across the cohort, completely robust to extreme outlier cases.
- **Micro / Pooled Metric (Volume-Weighted):**
  $$\text{Micro Dice} = \frac{2 \sum_{i=1}^N \text{TP}_i}{2 \sum_{i=1}^N \text{TP}_i + \sum_{i=1}^N \text{FP}_i + \sum_{i=1}^N \text{FN}_i}$$
  Pools all voxels globally across the entire cohort. Naturally weighted by lesion volume.

### 5.3 Zero-Division & Empty-Mask Edge-Case Audit
Inspecting lines 248–282 in `glioma_formal_evaluation.py` and lines 385–396 in `meningioma_multimodal_dataset.py`:
- **Case 1: Both Ground Truth and Prediction are Empty ($\text{TP}=\text{FP}=\text{FN}=0$):**
  $$\text{Dice} = 1.0, \quad \text{IoU} = 1.0, \quad \text{Precision} = 1.0, \quad \text{Recall} = 1.0$$
  *Correctly rewards the model for true negative classification.*
- **Case 2: Ground Truth Empty, Prediction Non-Empty ($\text{TP}=0, \text{FP}>0, \text{FN}=0$):**
  $$\text{Dice} = 0.0, \quad \text{IoU} = 0.0, \quad \text{Precision} = 0.0, \quad \text{Recall} = 1.0$$
  *Correctly penalizes false-positive hallucination.*
- **Case 3: Ground Truth Non-Empty, Prediction Empty ($\text{TP}=0, \text{FP}=0, \text{FN}>0$):**
  $$\text{Dice} = 0.0, \quad \text{IoU} = 0.0, \quad \text{Precision} = 1.0, \quad \text{Recall} = 0.0$$
  *Correctly identifies zero sensitivity on complete miss.*

---

## 6. Glioma Volumetric Concordance ($R^2$ and Pearson $r$) Audit

Inspecting lines 480–512 and 711–715 of `segmentation/glioma_formal_evaluation.py`:
- **Evaluated Variables:** `gt_vols = [c["wt"]["gt_volume"] for c in case_results]` and `pred_vols = [c["wt"]["pred_volume"] for c in case_results]`.
- **Cases Included:** Exactly all **20 validation cases** (no exclusions, no outliers dropped).
- **Exact Mathematical Formulas:**
  - Pearson Correlation:
    $$r = \frac{\sum (y_i - \bar{y})(\hat{y}_i - \bar{\hat{y}})}{\sqrt{\sum (y_i - \bar{y})^2 \sum (\hat{y}_i - \bar{\hat{y}})^2}} = \mathbf{0.984216} \quad (p = 5.45 \times 10^{-15})$$
  - Spearman Rank Correlation:
    $$\rho = \mathbf{0.977444} \quad (p = 1.32 \times 10^{-13})$$
  - Coefficient of Determination:
    $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2} = \mathbf{0.966613} \quad (96.66\% \text{ explained variance})$$
  - Mean Absolute Error ($\text{MAE}$): **$6,148.5\,\text{voxels}$** ($6.15\,\text{cm}^3$)
  - Mean Relative Error ($\text{MRE}$): **$15.43\%$**
- **Audit Verdict:** The volume analysis is verified, fully grounded, and mathematically reproducible.

---

## 7. Reconciliation of Meningioma Score Discrepancy (79.91% vs. 81.38%)

There is an apparent discrepancy in project records between two reported Whole Tumor Dice scores for multimodal meningioma. This audit completely reconciles their origin:

```
                                  TRAINING FORK
                                        │
           ┌────────────────────────────┴────────────────────────────┐
           ▼                                                         ▼
     Run 1 (Oct 14, 2026)                                      Run 2 (Oct 15, 2026)
`full800_brainaware_20261014`                             `full800_brainaware_rerun_20261015`
  - Training Epochs: 25                                     - Training Epochs: 25
  - Val Monitoring: 25 cases                                - Val Monitoring: 200 cases (every epoch)
  - Selected Checkpoint: Epoch 25                           - Selected Checkpoint: Epoch 22
  - SHA-256: `D65C60DFFB5FBA1D...`                          - SHA-256: `B32D848129B247FE...`
  - Post-hoc 200-Case Evaluation:                           - Online 200-Case Evaluation:
    * Macro WT Dice: 79.91%                                   * Macro WT Dice: 81.38%
    * Median WT Dice: 90.06%                                  * Macro TC Dice: 83.27%
    * Pooled WT Dice: 92.73%                                  * Macro ET Dice: 83.95%
    * Report: `final_validation_20261015/`                    * Log: `training_log.json`
```

### Key Audit Facts
1. **$79.91\%$ Macro WT Dice:**
   - **Run:** `outputs/meningioma_multimodal_experiment/full800_brainaware_20261014/`
   - **Checkpoint:** `checkpoints/meningioma_multimodal_best.pth` (Hash: `D65C60DF...`, Epoch 25).
   - **Evaluator:** Evaluated post-training across all 200 holdout cases by `scripts/evaluate_meningioma_multimodal_final200.py`.
   - **Output Evidence:** [`outputs/meningioma_multimodal_experiment/final_validation_20261015/full200_validation_metrics.json`](file:///C:/Users/nchan/Desktop/DL/outputs/meningioma_multimodal_experiment/final_validation_20261015/full200_validation_metrics.json).
   - **Production Role:** This is the audited model integrated into [`api/app.py`](file:///C:/Users/nchan/Desktop/DL/api/app.py) and [`integration/meningioma_multimodal_service.py`](file:///C:/Users/nchan/Desktop/DL/integration/meningioma_multimodal_service.py).
2. **$81.38\%$ Macro WT Dice:**
   - **Run:** `outputs/meningioma_multimodal_experiment/full800_brainaware_rerun_20261015/`
   - **Checkpoint:** `checkpoints/meningioma_multimodal_best.pth` (Hash: `B32D8481...`, Epoch 22).
   - **Evaluator:** Online validation loop during training (`val_interval=1`, `val_cases_count=200`).
   - **Output Evidence:** `outputs/meningioma_multimodal_experiment/full800_brainaware_rerun_20261015/logs/training_log.json`.
3. **Mandatory Audit Rule:**
   Do **NOT** combine, average, or conflate metrics from these two checkpoints. When presenting the primary system deliverables, report **$79.91\%$ Macro WT Dice** ($92.73\%$ pooled, $90.06\%$ median) for the primary integrated checkpoint (`D65C60DF...`), and cite the rerun checkpoint (`B32D8481...`, $81.38\%$) as a separate confirmation run.

---

## 8. Summary of Corrections & Recommended Adjustments

Before publishing, demonstrating, or submitting evaluation results, apply these terminology and reporting corrections:

1. **Terminology Correction (Glioma):**  
   - Replace any mention of "20 independent test cases" with **"20 internal validation / development cases"** in research notes, abstracts, and presentation slides.
2. **Checkpoint Attribution (Meningioma):**  
   - Explicitly cite the checkpoint hash when quoting either $79.91\%$ or $81.38\%$ WT Dice to maintain strict scientific reproducibility.
3. **Metric Reporting Rule:**  
   - Always report Macro Mean alongside Median and Pooled Dice to clearly show that the unweighted score is driven by small lesion sensitivity.

---

## 9. Next Steps: Exact Terminal Commands to Run

Cherry should run the following commands to verify both evaluation pipelines and display their audited metrics:

### Step 1: Run Formal Glioma Evaluation (20 Validation Cases)
```powershell
Set-Location -Path "C:\Users\nchan\Desktop\DL"
.\.venv\Scripts\python.exe segmentation/glioma_formal_evaluation.py
```
*Expected: 20 cases evaluated, WT Micro Dice = 90.42%, WT Median Dice = 90.13%, Volume $R^2 = 0.967$.*

### Step 2: Run Full Meningioma Evaluation (200 Holdout Cases)
```powershell
Set-Location -Path "C:\Users\nchan\Desktop\DL"
.\.venv\Scripts\python.exe scripts/evaluate_meningioma_multimodal_final200.py `
    --dataset-root "C:\3D_DATASET\BraTS2023_MEN" `
    --split-json "data/meningioma_multimodal/split.json" `
    --unet-checkpoint "outputs/meningioma_multimodal_experiment/full800_brainaware_20261014/checkpoints/meningioma_multimodal_best.pth" `
    --output-dir "outputs/meningioma_multimodal_experiment/final_validation_20261015" `
    --device cuda
```
*Expected: 200 cases evaluated, WT Pooled Dice = 92.73%, WT Median Dice = 90.06%, WT Macro Dice = 79.91%.*

### Step 3: Run the Complete 78-Test Integration Suite
```powershell
Set-Location -Path "C:\Users\nchan\Desktop\DL"
.\.venv\Scripts\pytest.exe -q
```
*Expected: 78 passed in ~18 seconds.*

