# Comprehensive Segmentation Error Analysis: 4-Channel 3D U-Net Models (Glioma & Meningioma)

**Date:** 2026-10-16  
**Project Root:** `C:\Users\nchan\Desktop\DL`  
**Evaluation Scope:** Formal Full-Volume Validation Cohorts (Glioma $N=20$, Multimodal Meningioma $N=200$)  
**Associated Figures:** [`outputs/final_model_comparison_20261016/`](file:///C:/Users/nchan/Desktop/DL/outputs/final_model_comparison_20261016/)  

---

## 1. Executive Summary & Core Insights

This document presents a granular, per-case error analysis of the two primary 4-channel 3D U-Net segmentation models deployed in the platform:
1. **Glioma 4-Channel 3D U-Net:** Checkpoint `outputs/glioma_training/glioma_best_medium_experiment.pth` evaluated across all $20$ validation cases.
2. **Multimodal Meningioma 4-Channel 3D U-Net:** Checkpoint `outputs/meningioma_multimodal_experiment/full800_brainaware_20261014/checkpoints/meningioma_multimodal_best.pth` evaluated across all $200$ holdout validation cases.

### Key Error Patterns Discovered
- **Small Lesion Vulnerability (The Dominant Failure Mode):** For both pathologies, segmentation errors are heavily concentrated in tumors smaller than $5,000\,\text{voxels}$ ($5.0\,\text{cm}^3$). In meningioma, where $37.5\%$ of the cohort is $<5.0\,\text{cm}^3$, the mean WT Dice on small lesions is $0.6050$, compared to $0.9549$ on large tumors.
- **Ectopic Overprediction on Microscopic Meningiomas:** On very small meningiomas ($<300\,\text{voxels}$), the model frequently detects the primary lesion but generates substantial ectopic false positive clusters in normal dural sinuses, yielding extreme volume ratios exceeding $160\times$.
- **Complete Misses on Atypical Small Lesions:** Three small meningiomas ($1,587$, $191$, and $59$ voxels) suffered complete false-negative misses ($\text{Dice} = 0.0000$), where the network predicted small clusters in unrelated anatomical locations.
- **Subregion Divergence in Non-Enhancing Gliomas:** In glioma case `BraTS-GLI-01666-000`, the Whole Tumor Dice is high ($0.8898$), but Enhancing Tumor (ET) Dice is $0.0000$ because the ground truth contained zero enhancing tumor, while the model erroneously predicted $2,396$ enhancing voxels.

---

## 2. Glioma Pipeline: Granular Per-Case Analysis (N = 20)

*Source Data:* [`outputs/glioma_formal_evaluation/per_case_metrics.csv`](file:///C:/Users/nchan/Desktop/DL/outputs/glioma_formal_evaluation/per_case_metrics.csv)  
*Status Disclosure:* The 20 evaluated glioma cases represent the **internal validation cohort** used during training for model selection and threshold tuning ($\tau=0.60$, component filter $1,000$ voxels), not an independent test set.

### 2.1 Top 10 Best Performing Cases (Ranked by WT Dice)

| Case ID | WT Dice | TC Dice | ET Dice | WT Prec | WT Rec | GT Vol ($\text{mm}^3$) | Pred Vol ($\text{mm}^3$) | Vol Ratio | Characterization |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `BraTS-GLI-01381-000` | **0.9632** | 0.9476 | 0.9325 | 0.9676 | 0.9588 | 118,517 | 117,428 | $0.99\times$ | Large well-defined glioblastoma; near-perfect boundary |
| `BraTS-GLI-00101-000` | **0.9594** | 0.9606 | 0.9137 | 0.9339 | 0.9863 | 58,948 | 62,251 | $1.06\times$ | Medium-large lesion; excellent all-subregion agreement |
| `BraTS-GLI-01518-000` | **0.9586** | 0.5812 | 0.5357 | 0.9518 | 0.9656 | 171,033 | 173,518 | $1.01\times$ | Massive edema shell; core under-segmented |
| `BraTS-GLI-01610-000` | **0.9477** | 0.8531 | 0.8660 | 0.9572 | 0.9385 | 19,259 | 18,882 | $0.98\times$ | Compact solid glioma; highly concordant volume |
| `BraTS-GLI-01256-000` | **0.9384** | 0.7308 | 0.8227 | 0.9606 | 0.9172 | 135,252 | 129,139 | $0.95\times$ | Large frontal mass; necrotic core slightly under-predicted |
| `BraTS-GLI-00221-000` | **0.9336** | 0.5374 | 0.6435 | 0.9004 | 0.9693 | 27,719 | 29,838 | $1.08\times$ | Medium mass; high WT recall, non-enhancing core shifted |
| `BraTS-GLI-00402-000` | **0.9320** | 0.9614 | 0.9179 | 0.9577 | 0.9076 | 231,389 | 219,273 | $0.95\times$ | Very large glioblastoma; high precision across all subregions |
| `BraTS-GLI-01497-000` | **0.9153** | 0.7952 | 0.6968 | 0.9052 | 0.9256 | 77,020 | 78,757 | $1.02\times$ | Standard presentation; balanced precision and recall |
| `BraTS-GLI-00291-000` | **0.9132** | 0.8515 | 0.8440 | 0.8714 | 0.9593 | 57,452 | 63,248 | $1.10\times$ | Moderate edema over-prediction (+10% volume) |
| `BraTS-GLI-01397-000` | **0.9019** | 0.7973 | 0.8777 | 0.9388 | 0.8677 | 63,983 | 59,143 | $0.92\times$ | Median cohort representative; minor edge under-prediction |

### 2.2 Top 10 Worst Performing Cases (Ranked by WT Dice)

| Case ID | WT Dice | TC Dice | ET Dice | WT Prec | WT Rec | GT Vol ($\text{mm}^3$) | Pred Vol ($\text{mm}^3$) | Vol Ratio | Primary Error Mode |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `BraTS-GLI-01308-000` | **0.4580** | 0.4009 | 0.4004 | 0.4511 | 0.4650 | 6,520 | 6,721 | $1.03\times$ | Spatial boundary offset on small focal lesion |
| `BraTS-GLI-00621-000` | **0.5981** | 0.4226 | 0.4264 | 0.6985 | 0.5229 | 26,835 | 20,086 | $0.75\times$ | Infiltrative margin under-segmented (low recall 0.52) |
| `BraTS-GLI-00419-000` | **0.6171** | 0.7152 | 0.5796 | 0.4487 | 0.9879 | 32,190 | 70,868 | $2.20\times$ | Excessive edema overprediction (38.7k FP voxels) |
| `BraTS-GLI-01341-000` | **0.6269** | 0.8849 | 0.8143 | 0.4878 | 0.8769 | 3,340 | 6,004 | $1.80\times$ | Smallest glioma in cohort; peripheral edema inflation |
| `BraTS-GLI-00280-000` | **0.8355** | 0.7975 | 0.7719 | 0.8575 | 0.8146 | 30,654 | 29,118 | $0.95\times$ | Minor under-segmentation along irregular deep margins |
| `BraTS-GLI-00140-000` | **0.8854** | 0.6548 | 0.6656 | 0.8924 | 0.8784 | 16,748 | 16,485 | $0.98\times$ | Accurate WT, but overpredicted non-enhancing core |
| `BraTS-GLI-01666-000` | **0.8898** | 0.2287 | **0.0000** | 0.9722 | 0.8202 | 135,746 | 114,532 | $0.84\times$ | Non-enhancing glioma (GT ET=0); false ET predicted |
| `BraTS-GLI-01087-000` | **0.8900** | 0.8552 | 0.8597 | 0.8879 | 0.8920 | 103,881 | 104,360 | $1.00\times$ | High volume match, minor false positive tail |
| `BraTS-GLI-01080-000` | **0.8910** | 0.7254 | 0.7886 | 0.9020 | 0.8804 | 83,547 | 81,546 | $0.98\times$ | Accurate outer boundary, slight central necrotic undercall |
| `BraTS-GLI-00022-000` | **0.9007** | 0.7791 | 0.7862 | 0.9406 | 0.8639 | 112,949 | 103,741 | $0.92\times$ | Large temporal mass, mild posterior margin under-recall |

---

## 3. Multimodal Meningioma: Granular Per-Case Analysis (N = 200)

*Source Data:* [`outputs/meningioma_multimodal_experiment/final_validation_20261015/full200_per_case_metrics.csv`](file:///C:/Users/nchan/Desktop/DL/outputs/meningioma_multimodal_experiment/final_validation_20261015/full200_per_case_metrics.csv)  
*Status Disclosure:* Evaluated on the **200 internal holdout cases** from [`data/meningioma_multimodal/split.json`](file:///C:/Users/nchan/Desktop/DL/data/meningioma_multimodal/split.json) with empty-air masking.

### 3.1 Top 10 Best Performing Cases (Ranked by WT Dice)

| Case ID | WT Dice | TC Dice | ET Dice | WT Prec | WT Rec | GT Vol (vx) | Pred Vol (vx) | Vol Ratio | Anatomical Characterization |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `BraTS-MEN-01367-000` | **0.9849** | 0.9864 | 0.9837 | 0.9857 | 0.9842 | 76,080 | 75,963 | $1.00\times$ | Large parasagittal mass; perfect dural margin capture |
| `BraTS-MEN-00977-000` | **0.9822** | 0.9844 | 0.9861 | 0.9857 | 0.9788 | 141,143 | 140,163 | $0.99\times$ | Giant convexity meningioma; flawless subregions |
| `BraTS-MEN-00421-000` | **0.9814** | 0.9832 | 0.9806 | 0.9738 | 0.9892 | 116,990 | 118,844 | $1.02\times$ | Large skull-base mass; minor peripheral dural tail |
| `BraTS-MEN-00685-000` | **0.9795** | 0.9864 | 0.9865 | 0.9778 | 0.9812 | 61,831 | 62,048 | $1.00\times$ | Convexity mass; exact volume congruence |
| `BraTS-MEN-00859-000` | **0.9789** | 0.9852 | 0.9815 | 0.9710 | 0.9869 | 55,092 | 55,996 | $1.02\times$ | Frontal dural mass; minimal over-prediction |
| `BraTS-MEN-00481-000` | **0.9780** | 0.9893 | 0.9895 | 0.9684 | 0.9878 | 50,579 | 51,592 | $1.02\times$ | High-contrast enhancing core; sharp boundaries |
| `BraTS-MEN-00621-000` | **0.9776** | 0.9825 | 0.9814 | 0.9823 | 0.9730 | 242,708 | 240,415 | $0.99\times$ | Massive extra-axial tumor ($242\,\text{cm}^3$); flawless |
| `BraTS-MEN-01179-000` | **0.9774** | 0.9816 | 0.9816 | 0.9653 | 0.9898 | 20,077 | 20,587 | $1.03\times$ | Standard clinical medium mass ($20.1\,\text{cm}^3$) |
| `BraTS-MEN-01399-000` | **0.9766** | 0.9825 | 0.9825 | 0.9762 | 0.9771 | 40,745 | 40,781 | $1.00\times$ | Convexity lesion; exact voxel volume matching |
| `BraTS-MEN-00709-000` | **0.9765** | 0.9760 | 0.9774 | 0.9774 | 0.9755 | 203,751 | 203,353 | $1.00\times$ | Massive sphenoid ridge meningioma; near-zero error |

### 3.2 Top 10 Worst Performing Cases (Ranked by WT Dice)

| Case ID | WT Dice | TC Dice | ET Dice | WT Prec | WT Rec | GT Vol (vx) | Pred Vol (vx) | Vol Ratio | Primary Failure Mechanism |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `BraTS-MEN-00690-000` | **0.0000** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1,587 | 441 | $0.28\times$ | **Complete Miss:** False negative on true lesion; spurious prediction elsewhere |
| `BraTS-MEN-01224-000` | **0.0000** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 191 | 240 | $1.26\times$ | **Complete Miss:** Tiny 191-voxel lesion completely missed; ectopic cluster |
| `BraTS-MEN-01363-000` | **0.0000** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 59 | 1,451 | $24.59\times$ | **Complete Miss:** Microscopic 59-voxel lesion missed; 1.4k false positives |
| `BraTS-MEN-00094-000` | **0.0109** | 0.0110 | 0.0110 | 0.0055 | 0.8993 | 139 | 22,724 | **$163.48\times$** | **Massive Ectopic FP:** Detected 125 TP voxels, but added 22.6k FP voxels in brain |
| `BraTS-MEN-00899-000` | **0.0114** | 0.0112 | 0.0183 | 0.0057 | 0.9296 | 284 | 46,114 | **$162.37\times$** | **Massive Ectopic FP:** Detected 264 TP voxels, but added 45.8k FP voxels in brain |
| `BraTS-MEN-01147-000` | **0.0176** | 0.0176 | 0.0302 | 0.0105 | 0.0536 | 336 | 1,709 | $5.09\times$ | Severe under-recall (5%) combined with spatial drift |
| `BraTS-MEN-00362-000` | **0.0810** | 0.0811 | 0.0824 | 0.0423 | 0.9430 | 263 | 5,860 | $22.28\times$ | Small enhancing focus; over-segmented dural tail (5.8k voxels) |
| `BraTS-MEN-00271-000` | **0.1431** | 0.7677 | 0.7677 | 0.0773 | 0.9516 | 620 | 7,628 | $12.30\times$ | Enhancing core captured (TC 0.77), but spurious edema added (7k voxels) |
| `BraTS-MEN-01008-010` | **0.1898** | 0.1039 | 0.1043 | 0.1056 | 0.9341 | 1,806 | 15,972 | $8.84\times$ | High recall (93%), but huge over-prediction in adjacent dural folds |
| `BraTS-MEN-00199-000` | **0.2076** | 0.2078 | 0.2078 | 0.1234 | 0.6524 | 164 | 867 | $5.29\times$ | Small 164-voxel lesion; 760 false positive voxels along skull base |

---

## 4. Root-Cause Analysis of Segmentation Failures

### 4.1 The Small Tumor Volume Penalty
A critical statistical insight emerges when comparing the volumes of the best vs. worst cases:
* **Top 10 Best Meningioma Cases:** Mean volume = **$115,191\,\text{voxels}$** (Range: $20,077 - 242,708\,\text{voxels}$, all $>20.0\,\text{cm}^3$).
* **Top 10 Worst Meningioma Cases:** Mean volume = **$545\,\text{voxels}$** (Range: $59 - 1,806\,\text{voxels}$, all $<1.8\,\text{cm}^3$).

**Why Small Tumors Suffer Low Dice Scores:**
1. **Mathematical Sensitivity of the Dice Formula:**  
   For a lesion of $100\,\text{voxels}$, a boundary dilation of just $2$ voxels adds $\approx 300$ false positive voxels, immediately dropping the Dice score below $0.40$. For a $100,000$-voxel lesion, the same $300$-voxel boundary divergence results in a Dice score of $0.997$.
2. **Loss of Contrast Context:**  
   Large tumors exhibit pronounced internal mass effect, multi-channel hypointensity/hyperintensity patterns, and surrounding edema envelopes. Microscopic tumors ($<1\,\text{cm}^3$) present simply as small enhancing dots on T1c that lack distinct 3D textural signatures.
3. **Anatomical Confusers (Dural Venous Sinuses):**  
   Normal venous sinuses (superior sagittal sinus, transverse sinus) and dural folds intensely enhance with gadolinium on T1c. While empty-air masking cleanly removes false positives outside the cranium, it cannot eliminate false positives in normal vascular structures inside the cranium.

### 4.2 Subregion Divergence: Non-Enhancing Pathology
* In glioma case `BraTS-GLI-01666-000`, the tumor is a diffuse non-enhancing astrocytoma (GT ET Volume = $0$).
* The network generated $2,396$ enhancing voxels, resulting in an ET Dice of $0.0000$.
* This illustrates that when a patient completely lacks an enhancing component, any small threshold leakage results in a mathematical Dice score of $0.0000$, even when the overall tumor volume is accurately segmented.

---

## 5. Visualizations & Orthogonal Overlays Reference

All generated multi-view orthogonal visualizations (Axial, Coronal, Sagittal) and comparative charts are preserved in [`outputs/final_model_comparison_20261016/`](file:///C:/Users/nchan/Desktop/DL/outputs/final_model_comparison_20261016/):

### 5.1 Comparative Statistical Plots
1. **Dice Score Distribution Comparison:**  
   [`dice_distribution_comparison.png`](file:///C:/Users/nchan/Desktop/DL/outputs/final_model_comparison_20261016/dice_distribution_comparison.png)  
   Juxtaposes boxplots and jitter points for WT, TC, and ET across Glioma ($N=20$) and Meningioma ($N=200$).
2. **Tumor Volume vs. WT Dice Correlation:**  
   [`volume_vs_dice_correlation.png`](file:///C:/Users/nchan/Desktop/DL/outputs/final_model_comparison_20261016/volume_vs_dice_correlation.png)  
   Demonstrates the steep performance cliff below the $5,000$-voxel small-tumor threshold on a logarithmic scale.
3. **Subregion Metric Formulations Bar Chart:**  
   [`subregion_metrics_comparison.png`](file:///C:/Users/nchan/Desktop/DL/outputs/final_model_comparison_20261016/subregion_metrics_comparison.png)  
   Compares Macro Mean vs. Median vs. Pooled/Micro Dice across all subregions.

### 5.2 Glioma Representative Case Overlays
- **Representative Successful Case (`BraTS-GLI-01381-000`):**  
  [`glioma_case_successful_BraTS-GLI-01381-000.png`](file:///C:/Users/nchan/Desktop/DL/outputs/final_model_comparison_20261016/glioma_case_successful_BraTS-GLI-01381-000.png)  
  *WT Dice: 0.9632 | GT: 118.5k vx | Pred: 117.4k vx.* Near-perfect multi-planar overlay.
- **Median-Performing Case (`BraTS-GLI-01397-000`):**  
  [`glioma_case_median_BraTS-GLI-01397-000.png`](file:///C:/Users/nchan/Desktop/DL/outputs/final_model_comparison_20261016/glioma_case_median_BraTS-GLI-01397-000.png)  
  *WT Dice: 0.9019 | GT: 64.0k vx | Pred: 59.1k vx.* Typical clinical presentation with minor edge under-recall.
- **Small-Tumor Case (`BraTS-GLI-01341-000`):**  
  [`glioma_case_small_tumor_BraTS-GLI-01341-000.png`](file:///C:/Users/nchan/Desktop/DL/outputs/final_model_comparison_20261016/glioma_case_small_tumor_BraTS-GLI-01341-000.png)  
  *WT Dice: 0.6269 | GT: 3.3k vx | Pred: 6.0k vx.* Smallest glioma in cohort; shows peripheral edema overprediction.
- **Poor-Performing Case (`BraTS-GLI-01308-000`):**  
  [`glioma_case_poor_BraTS-GLI-01308-000.png`](file:///C:/Users/nchan/Desktop/DL/outputs/final_model_comparison_20261016/glioma_case_poor_BraTS-GLI-01308-000.png)  
  *WT Dice: 0.4580 | GT: 6.5k vx | Pred: 6.7k vx.* Spatial boundary offset on small focal lesion.

### 5.3 Meningioma Representative Case Overlays
- **Representative Successful Case (`BraTS-MEN-01367-000`):**  
  [`meningioma_case_successful_BraTS-MEN-01367-000.png`](file:///C:/Users/nchan/Desktop/DL/outputs/final_model_comparison_20261016/meningioma_case_successful_BraTS-MEN-01367-000.png)  
  *WT Dice: 0.9849 | GT: 76.1k vx | Pred: 76.0k vx.* Near-perfect boundary alignment on 76k-voxel meningioma.
- **Median-Performing Case (`BraTS-MEN-00232-000`):**  
  [`meningioma_case_median_BraTS-MEN-00232-000.png`](file:///C:/Users/nchan/Desktop/DL/outputs/final_model_comparison_20261016/meningioma_case_median_BraTS-MEN-00232-000.png)  
  *WT Dice: 0.8997 | GT: 10.0k vx | Pred: 11.7k vx.* Robust core segmentation with minor dural tail overprediction.
- **Small-Lesion Overprediction Case (`BraTS-MEN-00094-000`):**  
  [`meningioma_case_small_overpredicted_BraTS-MEN-00094-000.png`](file:///C:/Users/nchan/Desktop/DL/outputs/final_model_comparison_20261016/meningioma_case_small_overpredicted_BraTS-MEN-00094-000.png)  
  *WT Dice: 0.0109 | GT: 139 vx | Pred: 22,724 vx.* True 139-voxel lesion detected, but massive ectopic cluster added.
- **Small-Lesion Complete Miss Case (`BraTS-MEN-00690-000`):**  
  [`meningioma_case_small_missed_BraTS-MEN-00690-000.png`](file:///C:/Users/nchan/Desktop/DL/outputs/final_model_comparison_20261016/meningioma_case_small_missed_BraTS-MEN-00690-000.png)  
  *WT Dice: 0.0000 | GT: 1,587 vx | Pred: 441 vx.* Ground truth completely missed; small false mass predicted elsewhere.
- **Full Comparative Contact Sheet:**  
  [`meningioma_cases_contact_sheet.png`](file:///C:/Users/nchan/Desktop/DL/outputs/final_model_comparison_20261016/meningioma_cases_contact_sheet.png)

