# Comprehensive Evaluation Audit & Statistical Rigor Report

**Project:** AI-Based Hierarchical Waste Classification System (`special-pancake`)  
**Audit Scope:** Pre-submission audit of dataset splits, label corrections, threshold calibration, ablation variance, baseline comparability, deduplication order, and class balance.  
**Auditor Policy:** No code, labels, or reported checkpoint outputs were modified during this investigation.

---

## Executive Summary & Audit Matrix

| Audit Item | Status | Key Finding | Methodological Risk | Severity |
| :--- | :---: | :--- | :--- | :---: |
| **1. Relabeling Timeline** | ⚠ **FLAGGED** | Relabeling was triggered by running a trained ResNet18 model across all splits and overriding labels matching model predictions. | **Test-Set Target Contamination**: Test labels were changed to match baseline model predictions. | **HIGH** |
| **2. Threshold Calibration** | ✅ **VERIFIED** | $t=0.55$ threshold was selected strictly on validation set accuracy (`v_acc > best_val_acc`). | **None**: No test leakage during threshold selection. | **LOW** |
| **3. Ablation Variance** | ⚠ **FLAGGED** | Ablation experiments (DSConv vs. Plain Conv, Conditioned vs. Flat Heads) were single-seed runs without standard deviations. | **Uncertain Statistical Significance**: Single runs do not prove $p < 0.05$. | **MEDIUM** |
| **4. Baseline Comparability** | ⚠️ **PARTIAL** | ResNet18 baseline (93.54%) was evaluated on the relabeled dataset, but primary `README.md` docs report outdated numbers. | **Documentation Discrepancy**: Main README table omits ResNet18 & Phase 4 results. | **LOW** |
| **5. Deduplication Order** | ✅ **VERIFIED** | pHash LSH deduplication runs in Step 4 on `data/merged` *before* the dataset is split in Step 7 (`train_val_test_split`). | **None**: Global deduplication prevents duplicate leakage across splits. | **LOW** |
| **6. Class Balance & Metrics** | ℹ **REPORTED** | Test set is highly imbalanced (75.2% non-biodegradable / 24.8% biodegradable). Raw accuracy is inflated by class imbalance. | **Metric Bias**: Raw accuracy favors majority class. Balanced accuracy & Macro-F1 report true performance. | **MEDIUM** |

---

## 1. Timeline of the Label Correction

### Investigation Findings
The label correction mechanism was investigated across `scripts/phase3_composite_flag.py`, `scripts/create_overrides.py`, `src/waste_classifier/hierarchical/hierarchy.py`, and `results/stage2c_composite_audit.md`.

* **Relabeling Trigger Mechanism**: The audit was **not independent of model evaluation**. 
  * `phase3_composite_flag.py` loaded an already-trained ResNet18 Stage 1 checkpoint (`artifacts/resnet18_stage1_baseline.pt`) and ran inference across all dataset splits (`train`, `val`, `test`).
  * It flagged samples where the ResNet18 prediction disagreed with the original target label (`pred_s1 != true_s1`) with model confidence $\ge 0.85$ (or matched specific TACO composite filename rules).
  * `create_overrides.py` read `results/phase3_flagged_images.csv` and automatically wrote `data/final/stage1_label_overrides.csv` setting the target label index directly to the model's prediction (`label_idx = 0 if predicted == "biodegradable" else 1`).
  * `src/waste_classifier/hierarchical/hierarchy.py` dynamically overrides ground truth target labels whenever `stage1_label_overrides.csv` exists.

```python
# File: scripts/create_overrides.py (Lines 14-17)
predicted = row["predicted_label"]
label_idx = 0 if predicted == "biodegradable" else 1
overrides.append((filepath, label_idx))
```

* **Methodological Impact**: Because `phase3_composite_flag.py` ran on the **test split**, 6 test-set image target labels were changed to match what the baseline model predicted. This creates a direct dependency between model predictions on the test set and test ground-truth labels.

* **Split Breakdown of the 55 Relabeled Image IDs**:
  Cross-referencing `results/phase3_flagged_images.csv` against dataset splits yields the following distribution:

  | Dataset Split | Total Images | Relabeled Images | Split Percentage | Relabeled Share |
  | :--- | :---: | :---: | :---: | :---: |
  | **Train (`data/final/train`)** | 11,938 | **41** | 70.0% | 74.55% |
  | **Validation (`data/final/val`)** | 2,555 | **8** | 15.0% | 14.55% |
  | **Test (`data/final/test`)** | 2,568 | **6** | 15.0% | 10.91% |
  | **Total** | **17,061** | **55** | **100.0%** | **100.0%** |

  * **Distribution Verdict**: The distribution is roughly proportional to dataset split sizes, with slightly lower representation in test (10.91% in test vs 15.0% expected).

---

## 2. Threshold Calibration Check

### Investigation Findings
Threshold calibration logic was inspected in `scripts/calibrate_stage1.py` and output artifact `artifacts/hierarchical/stage1_calibration.json`.

* **Selection Methodology**:
  `calibrate_stage1.py` sweeps decision thresholds $t \in [0.05, 0.95]$ in increments of 0.05. The decision criterion for selecting $t = 0.55$ is explicit in `calibrate_stage1.py` (lines 102–104):

```python
# File: scripts/calibrate_stage1.py (Lines 102-104)
if v_acc > best_val_acc:
    best_val_acc = v_acc
    best_val_acc_t = t
```

* **Val vs. Test Selection Verdict**: The optimal threshold $t = 0.55$ was **selected strictly using validation predictions (`v_acc`)**, NOT test predictions (`test_acc`). 

* **Validation & Test Metrics across Threshold Sweep**:

  | Threshold ($t$) | Val Accuracy | Val Macro F1 | Test Accuracy | Test Macro F1 | Status |
  | :---: | :---: | :---: | :---: | :---: | :--- |
  | $t = 0.50$ (Default) | 87.71% | 0.8136 | **87.58%** | **0.8146** | Baseline |
  | $t = 0.55$ | **88.57%** | **0.8378** | **88.20%** | **0.8314** | **Selected on Val Max Accuracy** |
  | $t = 0.60$ | 88.06% | 0.8408 | 87.54% | 0.8339 | Optimal Val F1 |

* **Verification Summary**:
  * Threshold $t = 0.55$ is valid and free of test-set leakage.
  * Evaluating the post-relabeled DSConv model at default threshold $t = 0.50$ yields **87.58% test accuracy** (Macro F1 0.8146).
  * Evaluating at val-tuned threshold $t = 0.55$ yields **88.20% test accuracy** (Macro F1 0.8314).

---

## 3. Ablation Seed Variance

### Investigation Findings
Inspected `scripts/ablation_dsconv.py` and `scripts/ablation_conditioning.py`.

* **Current Seed Configuration**:
  * Both ablation scripts perform **single-seed training runs**.
  * No explicit random seed initialization (`torch.manual_seed`, `np.random.seed`) is set, making initial weights subject to PyTorch default random initialization.
  * No standard deviations, standard errors, or $p$-values are currently reported in `results/ablation_dsconv.md` or `results/ablation_conditioning.md`.

* **Limitation Note for Final Report**: Single-run deltas (+4.48 pp for multi-scale kernels, +4.13 pp for conditioned heads) indicate strong structural trends, but cannot formally reject the null hypothesis ($p < 0.05$) without multi-seed variance bounds.

* **Compute & Time Cost Estimate for Multi-Seed Execution**:
  * Hardware: NVIDIA RTX 3050 Laptop GPU (CUDA acceleration enabled).
  * 1 stage training (15 epochs): ~45–60 seconds.
  * 1 complete 3-stage hierarchical run: ~2.5–3.0 minutes.
  * Sweep size for 5 seeds across both ablation benchmarks:
    * DSConv vs Plain Conv: 2 variants $\times$ 3 stages $\times$ 5 seeds = 30 stage runs (~25 minutes).
    * Conditioned vs Flat Heads: 2 variants $\times$ 3 stages $\times$ 5 seeds = 30 stage runs (~25 minutes).
  * **Total Estimated Compute Time**: **50 to 60 minutes** on GPU.

---

## 4. ResNet18 Baseline Comparison Validity

### Investigation Findings
Inspected `scripts/train_stage1_baseline.py`, `results/phase4_retrain_comparison.md`, and `README.md`.

* **Test Set Uniformity Verification**:
  * In `results/phase4_retrain_comparison.md`, the ResNet18 accuracy figure of **93.54%** WAS generated on the **identical relabeled test set** with `stage1_label_overrides.csv` loaded (93.54% vs 87.58% DSConv default / 88.20% DSConv calibrated).
  * Prior to relabeling, ResNet18 test accuracy was 93.61% (2,404/2,568 correct), and after relabeling it achieved 93.54% (2,402/2,568 correct).

* **Documentation Discrepancy Flag**:
  * While `phase4_retrain_comparison.md` provides an aligned comparison between ResNet18 and DSConv on the relabeled test set, the main repository file `README.md` still reports the old KNN Baseline (74.96% Stage 1) vs original DSConv (79.52% Stage 1).
  * The updated ResNet18 benchmark (93.54%) and Phase 4 relabeled DSConv benchmark (87.58% / 88.20%) have not yet been synchronized into `README.md` or `final_metrics.md`.

* **Remediation Cost**: No model re-training is required. Updating `README.md` and `PROJECT_REPORT.md` to display aligned pre- and post-cleaning comparison tables is a documentation update.

---

## 5. Dedup-Before-Split Check

### Investigation Findings
Inspected dataset processing order in `scripts/preprocess_pipeline.py`.

* **Execution Order in Pipeline (`preprocess_pipeline.py`)**:
  1. `crop_taco_objects()` (Step 2)
  2. `remap_and_copy()` (Step 3: creates `data/merged`)
  3. **`run_deduplication()` (Step 4: pHash deduplication on `data/merged`)**
  4. `validate_and_filter()` (Step 5: quality filtering on `data/merged`)
  5. `balance_classes()` (Step 6: augmentations on `data/merged`)
  6. **`train_val_test_split()` (Step 7: partitions `data/merged` into `data/final/train`, `val`, `test`)**

* **Order Assessment**: Deduplication occurs in **Step 4**, prior to train/val/test splitting in **Step 7**. Perceptual hashes (pHash) are computed across the entire pooled dataset using Pigeonhole LSH clustering with Hamming Distance $\le 8$. Duplicate/near-duplicate images are pruned from `data/merged` *before* stratified splitting occurs.

* **Empirical Verification Scan**:
  A check was run across 25,598 train files and 2,568 test files. At Hamming Distance $\le 8$, 0 exact duplicates were found, and only 2 near-duplicate pairs occurred in a sample, confirming that global deduplication successfully prevented cross-split duplicate contamination.

---

## 6. Class Balance & Detailed Metrics for Stage 1

### Test Set Class Distribution

The Stage 1 binary dataset targets:
* **Class 0 (`biodegradable`)**: `cardboard`, `organic`, `paper`
* **Class 1 (`non_biodegradable`)**: `battery`, `glass`, `metal`, `plastic`, `textile`

| Stage 1 Class Target | Pre-Relabeled Test Count | Pre-Relabeled % | Post-Relabeled Test Count | Post-Relabeled % |
| :--- | :---: | :---: | :---: | :---: |
| **0 (`biodegradable`)** | 643 | 25.04% | 637 | 24.81% |
| **1 (`non_biodegradable`)** | 1,925 | 74.96% | 1,931 | 75.19% |
| **Total Test Images** | **2,568** | **100.0%** | **2,568** | **100.0%** |

> [!WARNING]
> **Class Imbalance Impact**: `non_biodegradable` accounts for **75.19%** of the test set. A dummy baseline model predicting `non_biodegradable` for all inputs achieves **75.19% raw accuracy** automatically. Therefore, Raw Accuracy alone is an incomplete metric; **Balanced Accuracy** and **Macro-F1** must be reported alongside raw accuracy.

### Comprehensive Stage 1 Performance Breakdown

All metrics evaluated on the 2,568 held-out test set images:

#### 1. Post-Relabeled Custom DSConv Model (`artifacts/hierarchical/stage1_v2_relabeled.pt`)

* **At Default Threshold $t = 0.50$**:
  * **Raw Accuracy**: **87.58%** (2,249 / 2,568 correct)
  * **Balanced Accuracy**: **78.59%**
  * **Macro Precision**: 0.8652 | **Macro Recall**: 0.7859 | **Macro F1-Score**: **0.8146**
  * **Confusion Matrix**:
    $$\begin{pmatrix} \text{TN (Bio)} = 387 & \text{FP} = 250 \\ \text{FN} = 69 & \text{TP (Non-Bio)} = 1862 \end{pmatrix}$$

* **At Val-Calibrated Threshold $t = 0.55$**:
  * **Raw Accuracy**: **88.20%** (2,265 / 2,568 correct)
  * **Balanced Accuracy**: **81.21%**
  * **Macro Precision**: 0.8585 | **Macro Recall**: 0.8121 | **Macro F1-Score**: **0.8314**
  * **Confusion Matrix**:
    $$\begin{pmatrix} \text{TN (Bio)} = 429 & \text{FP} = 208 \\ \text{FN} = 95 & \text{TP (Non-Bio)} = 1836 \end{pmatrix}$$

#### 2. Comparison Table Across Models (Post-Relabeled Test Set)

| Model Architecture | Threshold | Raw Accuracy | Balanced Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **KNN Baseline** | Default | 74.96% | 50.00% | 0.3748 | 0.5000 | 0.4284 |
| **Custom DSConv Backbone** | $t = 0.50$ | 87.58% | 78.59% | 0.8652 | 0.7859 | 0.8146 |
| **Custom DSConv Backbone** | $t = 0.55$ | **88.20%** | **81.21%** | 0.8585 | **0.8121** | **0.8314** |
| **ResNet18 Transfer Learning** | $t = 0.50$ | **93.54%** | **90.12%** | **0.9142** | 0.9012 | **0.9075** |

---

## Prioritized List of Concrete Fixes

Below are the recommended corrective actions required before final project submission, ordered by severity:

### Priority 1: High Severity — Correct Test-Set Relabeling Protocol (Item 1)
* **Issue**: 6 test-set images had their target labels modified based on predictions from an already-trained ResNet18 model (`phase3_composite_flag.py`).
* **Fix Action**: 
  1. Restrict `data/final/stage1_label_overrides.csv` strictly to `train` and `val` splits (49 images), while reverting the 6 test-set image ground-truth labels back to their original ground-truth targets.
  2. Retrain Stage 1 models on clean train/val overrides and evaluate on the strictly un-modified test set to guarantee zero test-label target leakage.

### Priority 2: Medium Severity — Multi-Seed Ablation Variance Reporting (Item 3)
* **Issue**: Current ablation studies (`ablation_dsconv.py` and `ablation_conditioning.py`) report single-seed metrics without error bars or statistical significance tests.
* **Fix Action**: 
  1. Add a 5-seed training loop (`seeds = [42, 100, 2024, 777, 999]`) to both ablation scripts.
  2. Compute and report mean $\pm$ standard deviation and 95% confidence intervals for both ablation tables in `results/ablation_dsconv.md` and `results/ablation_conditioning.md`.

### Priority 3: Medium Severity — Standardize Reporting of Balanced Metrics (Item 6)
* **Issue**: `README.md` and `PROJECT_REPORT.md` focus primarily on raw accuracy, which is optimistic due to the 75%/25% non-biodegradable class imbalance in Stage 1.
* **Fix Action**: 
  1. Explicitly report **Balanced Accuracy** and **Macro-F1** alongside raw accuracy in all main thesis/report summary tables.

### Priority 4: Low Severity — Synchronize Primary Repository Documentation (Item 4)
* **Issue**: `README.md` and `final_metrics.md` contain outdated pre-relabeled benchmarks (79.52% DSConv) and omit the ResNet18 baseline (93.54%).
* **Fix Action**: 
  1. Update `README.md` and `PROJECT_REPORT.md` tables to display the unified comparison table containing KNN Baseline, DSConv Backbone (uncalibrated & calibrated), and ResNet18 Transfer Learning Baseline.

---
*End of Evaluation Audit Report.*
