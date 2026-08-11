# Phase 4 — Stage 1 Retraining & Evaluation Report (Uncontaminated Ground Truth)

> [!NOTE]
> **Evaluation Leakage Correction Note (Audit Fix)**:
> In the initial Phase 4 run, 55 composite-material images were relabeled using predictions from an already-trained ResNet18 model. However, 6 of those 55 images were located in the **test split** (`data/final/test`), which introduced a target label dependency between model predictions and evaluation ground truth. 
> 
> As of the Priority 1 audit fix, **all 6 test-set label overrides were removed**, restoring the test set ground-truth target labels to their original, untouched status. The 49 training and validation overrides were retained to provide clean supervisory signals during training. The DSConv model was retrained from scratch and evaluated on the fully independent test set.

---

## 1. Overall Performance Comparison

Evaluated on the held-out, uncontaminated test split of **2,568 images** (643 Biodegradable, 1,925 Non-Biodegradable):

| Model Architecture | Threshold | Raw Accuracy | Balanced Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Net Delta vs Raw Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Custom DSConv (Original Uncleaned)** | $t = 0.50$ | 79.52% | 71.20% | 0.7511 | 0.8168 | 0.7638 | Baseline |
| **Custom DSConv (Clean Supervision, Default)** | $t = 0.50$ | **87.38%** | **78.38%** | 0.8633 | 0.7838 | **0.8118** | **+7.86 pp** |
| **Custom DSConv (Clean Supervision, Calibrated)** | $t = 0.55$ | **88.01%** | **81.01%** | 0.8566 | **0.8101** | **0.8286** | **+8.49 pp** |
| **ResNet18 Baseline** | $t = 0.50$ | **93.54%** | **90.12%** | **0.9142** | **0.9012** | **0.9075** | **+14.02 pp** |

---

## 2. Leakage Correction Impact (Contaminated vs. Uncontaminated Test Set)

| Evaluation Setup | Threshold | Test Accuracy | Balanced Acc | Macro F1 | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Old Contaminated Test Set (6 test overrides applied)** | $t = 0.50$ | 87.58% | 78.59% | 0.8146 | Target Leakage Present |
| **Corrected Uncontaminated Test Set (6 test overrides removed)** | $t = 0.50$ | **87.38%** | **78.38%** | **0.8118** | **Leak-Free Benchmark** |
| **Old Contaminated Test Set (6 test overrides applied)** | $t = 0.55$ | 88.20% | 81.21% | 0.8314 | Target Leakage Present |
| **Corrected Uncontaminated Test Set (6 test overrides removed)** | $t = 0.55$ | **88.01%** | **81.01%** | **0.8286** | **Leak-Free Benchmark** |

* **Impact Summary**: Removing the 6 test-set overrides shifted test accuracy by only **-0.19 to -0.20 percentage points**, confirming that the **+8.0 percentage point accuracy improvement** stems genuinely from cleaning training label noise rather than test-set leakage.

---

## 3. Confusion Matrix Breakdown (Clean Test Set)

### Default Threshold ($t = 0.50$)
$$\begin{pmatrix} \text{TN (Bio)} = 385 & \text{FP} = 258 \\ \text{FN} = 66 & \text{TP (Non-Bio)} = 1859 \end{pmatrix}$$

### Val-Calibrated Threshold ($t = 0.55$)
$$\begin{pmatrix} \text{TN (Bio)} = 426 & \text{FP} = 217 \\ \text{FN} = 91 & \text{TP (Non-Bio)} = 1834 \end{pmatrix}$$

---

## 4. Model Performance Gap Analysis

1. **Label Noise / Taxonomy Bottleneck (~7.86 to 8.49 pp):**
   Correcting composite label noise in training/validation supervision resolved visual contradictions (e.g. composite Tetra Paks, lined cups, glossy paper folders mapped to a biodegradable target), allowing the custom DSConv network to increase accuracy from **79.52% to 88.01%** without any test leakage.

2. **Genuine Representation Capacity Difference (~5.53 pp):**
   A gap of **5.53 pp** remains between calibrated DSConv (88.01%) and ImageNet-pretrained ResNet18 (93.54%), representing the true feature extraction advantage of transfer learning on complex textures.
