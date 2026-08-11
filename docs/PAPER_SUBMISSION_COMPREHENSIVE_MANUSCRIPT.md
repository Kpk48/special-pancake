# Comprehensive Research Manuscript: Hierarchical AI-Based Waste Classification via Multi-Scale Depthwise Separable Convolutions and Class-Conditioned Classifier Heads

**Document Type:** Final Year Capstone Project Research Manuscript  
**Target Repository:** `special-pancake`  
**System Name:** Hierarchical AI Waste Classification System  
**Date:** August 2026  

---

## Abstract

Automated waste sorting systems must operate effectively across diverse scales, lighting conditions, and severe class imbalances. Standard single-scale convolutional neural networks (CNNs) often struggle to simultaneously capture broad structural morphology (e.g., container geometry) and fine-grained spatial textures (e.g., paper vs. cardboard fibers). Furthermore, flat multi-class classifiers suffer from high error propagation across granular target categories. 

To resolve these challenges, we present a **3-stage coarse-to-fine hierarchical classification pipeline** driven by a custom **Multi-Scale Depthwise Separable Convolutional Backbone (`DSConv2D`)** and **Hierarchical Class Embedding Conditioning**. The `DSConv2D` backbone incorporates five parallel inception-style kernel branches ($11 \times 11, 9 \times 9, 7 \times 7, 5 \times 5, 3 \times 3$) to extract diverse multi-receptive field features without excessive parameter overhead (95,472 total parameters). Downstream classifier heads (Stage 2 and Stage 3) dynamically integrate upstream class predictions via learned categorical embedding vectors ($d_{\text{emb}} = 16$). 

Evaluated on a strictly clean held-out test set of 2,568 images (derived from a 17,061-image deduplicated pool), our validation-tuned Stage 1 binary classifier ($t = 0.55$) achieves **88.20% Top-1 accuracy**, **81.21% Balanced Accuracy**, and **0.8314 Macro F1-Score**, significantly outperforming a KNN baseline (74.96% accuracy, 50.00% BAcc). Parameter-matched ablation studies demonstrate that multi-scale receptive field branches provide a **+4.48 percentage point (pp)** gain over single-scale convolutions, while hierarchical head conditioning yields a **+4.13 pp to +4.69 pp** accuracy boost across downstream stages. Transfer learning with ResNet18 provides a high-capacity baseline benchmark of **93.54% Stage 1 accuracy**.

---

## 1. Introduction & Problem Statement

Municipal solid waste management and Autonomous Underwater Vehicle (AUV) marine debris collection require real-time, lightweight, and robust visual sorting models. Visual waste recognition presents three fundamental computer vision challenges:

1. **Scale Variance & Multi-Receptive Field Requirements**: Waste items range from small battery caps ($< 2 \text{ cm}$) to large corrugated boxes ($> 50 \text{ cm}$). Single fixed-kernel convolutions fail to extract optimal features across both extremes.
2. **Class Imbalance Bias**: Real-world waste collections exhibit severe imbalance. In our standardized test split, non-biodegradable materials represent **75.19%** of images, while biodegradable materials account for only **24.81%**. Naive unweighted accuracy metrics inflate majority-class performance.
3. **Inter-Class Visual Similarity**: Distinguishing between chemically distinct materials (e.g., paper vs. cardboard, or bio-degradable organic waste vs. textiles) requires conditioning fine-grained classifiers on coarse material context.

### 1.1 Project Contributions
This work delivers a fully audited, reproducible waste classification framework:
- **Novel Backbone Architecture**: Design of a lightweight `DSConv2D` backbone utilizing parallel multi-scale depthwise separable kernel stacks.
- **Hierarchical Conditioning Mechanism**: Implementation of stage-to-stage categorical embedding propagation for coarse-to-fine classification.
- **Audit-Verified Methodological Rigor**: Elimination of test-set target contamination (reverting 6 test-split label overrides) and execution of global deduplication prior to train/val/test partitioning.
- **Comprehensive Benchmarking & Ablation Analysis**: Evaluation across Raw Accuracy, Balanced Accuracy, and Macro F1, accompanied by parameter-matched ablation experiments.

---

## 2. Dataset Pipeline & Methodological Rigor

### 2.1 Raw Data Ingestion & Deduplication Pipeline
The full dataset comprises **17,061 images** aggregated from public datasets (TACO, Garbage-in-Containers) and internal domain collections. To prevent duplicate leakage across dataset partitions, processing follows a strict 7-step pipeline in `scripts/preprocess_pipeline.py`:

```
Step 1: Crop TACO Objects -> Step 2: Remap Categories -> Step 3: Copy to data/merged
                                                                    │
                                                                    ▼
Step 7: Stratified Split ◄── Step 6: Augmentations ◄── Step 5: Filter ◄── Step 4: pHash Dedup
 (Train/Val/Test)                                                          (Global Hamming <= 8)
```

> [!IMPORTANT]
> **Dedup-Before-Split Integrity (Step 4)**: Perceptual hashing (pHash) using Pigeonhole Locality-Sensitive Hashing (LSH) is executed globally on `data/merged` at Step 4 *before* stratified partitioning in Step 7. Near-duplicate images (Hamming distance $\le 8$) are pruned globally, ensuring zero cross-split duplicate contamination between training and evaluation sets.

### 2.2 Dataset Partitioning & Class Distribution
The dataset is stratified into three held-out splits:

| Partition | Image Count | Percentage | Primary Purpose |
| :--- | :---: | :---: | :--- |
| **Train (`data/final/train`)** | 11,938 | 70.0% | Model parameter optimization via Focal Loss |
| **Validation (`data/final/val`)** | 2,555 | 15.0% | Decision threshold calibration ($t = 0.55$) & hyperparameter selection |
| **Test (`data/final/test`)** | 2,568 | 15.0% | Unbiased final performance benchmarking |
| **Total Pool** | **17,061** | **100.0%** | **Deduplicated Dataset** |

#### Stage 1 Test Set Class Balance
- **Class 0 (`biodegradable`)**: `cardboard`, `organic`, `paper` $\rightarrow$ **637 images (24.81%)**
- **Class 1 (`non_biodegradable`)**: `battery`, `glass`, `metal`, `plastic`, `textile` $\rightarrow$ **1,931 images (75.19%)**
- **Majority Class Baseline**: A dummy classifier predicting `non_biodegradable` for all samples achieves **75.19% raw accuracy** automatically.

### 2.3 Evaluation Audit & Test Set Decontamination
A comprehensive audit (`evaluation_audit.md`) inspected all label modifications:
- **Audit Flag Identified**: An automated script (`phase3_composite_flag.py`) previously ran inference using a ResNet18 checkpoint across all splits and generated 55 label overrides in `data/final/stage1_label_overrides.csv`.
- **Test-Set Target Contamination Risk**: 6 of those overrides modified target labels on the held-out **test split**, creating dependency between model predictions and test ground truth.
- **Priority 1 Fix Action**: The 6 test-split overrides were deleted, reverting all 2,568 test image labels back to original ground truth. Overrides were strictly restricted to `train` (41) and `val` (8) splits.

---

## 3. System Architecture & Mathematical Formulations

### 3.1 Coarse-to-Fine Hierarchy Definition
The target categories are organized into three hierarchical levels:

$$\begin{aligned}
\mathcal{S}_1 &= \{\text{biodegradable (0)}, \text{non\_biodegradable (1)}\} \\
\mathcal{S}_2 &= \{\text{paper\_cardboard (0)}, \text{organic (1)}, \text{glass (2)}, \text{metal (3)}, \text{plastic (4)}, \text{textile\_battery (5)}\} \\
\mathcal{S}_3 &= \{\text{battery (0)}, \text{cardboard (1)}, \text{glass (2)}, \text{metal (3)}, \text{organic (4)}, \text{paper (5)}, \text{plastic (6)}, \text{textile (7)}\}
\end{aligned}$$

### 3.2 Multi-Scale DSConv2D Backbone Architecture
The custom `DSConv2DBackbone` processes input images $\mathbf{X} \in \mathbb{R}^{3 \times 128 \times 128}$. To extract receptive field features across multiple spatial frequencies, input features are routed through 5 parallel depthwise separable convolution branches:

$$\mathbf{F}_{\text{ms}} = \text{Concat}\Big( \text{DSConv}_{11\times11}(\mathbf{X}), \text{DSConv}_{9\times9}(\mathbf{X}), \text{DSConv}_{7\times7}(\mathbf{X}), \text{DSConv}_{5\times5}(\mathbf{X}), \text{DSConv}_{3\times3}(\mathbf{X}) \Big)$$

Each depthwise separable block decomposes standard 2D convolution into depthwise spatial filtering and point-wise $1 \times 1$ channel mixing, reducing parameter count from $\mathcal{O}(K^2 \cdot C_{\text{in}} \cdot C_{\text{out}})$ to $\mathcal{O}(K^2 \cdot C_{\text{in}} + C_{\text{in}} \cdot C_{\text{out}})$.

- **Total Backbone Parameters**: **95,472 parameters**
- **Feature Vector Output**: $\mathbf{f} \in \mathbb{R}^{128}$

### 3.3 Class-Conditioned Classifier Heads
Rather than evaluating each stage in isolation, downstream stages concatenate a learned categorical embedding of the previous stage prediction:

- **Stage 1**: $\hat{y}_1 = \text{argmax} \big( \mathbf{W}_1 \mathbf{f} + \mathbf{b}_1 \big)$
- **Stage 2**: $\mathbf{e}_1 = \text{Embedding}_1(\hat{y}_1) \in \mathbb{R}^{16} \implies \hat{y}_2 = \text{MLP}_2\big([\mathbf{f} \,\|\, \mathbf{e}_1]\big) \in \mathbb{R}^6$
- **Stage 3**: $\mathbf{e}_2 = \text{Embedding}_2(\hat{y}_2) \in \mathbb{R}^{16} \implies \hat{y}_3 = \text{MLP}_3\big([\mathbf{f} \,\|\, \mathbf{e}_2]\big) \in \mathbb{R}^8$

### 3.4 Loss Function & Optimization
All stages are trained using **Class-Weighted Focal Loss** to combat class imbalance:

$$\mathcal{L}_{\text{Focal}}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

where $\gamma = 2.0$, and class weights $\alpha_c$ are calculated inversely proportional to class frequencies:

$$\alpha_c = \frac{N}{C \cdot N_c}$$

- **Optimization Setup**: Adam Optimizer ($\text{lr} = 0.001$), Batch Size = 64, Epochs = 15 per stage.

---

## 4. Decision Threshold Calibration

### 4.1 Calibration Methodology
The default decision threshold $t = 0.50$ classifies an image as `non_biodegradable` if predicted probability $P(y=1|\mathbf{X}) \ge 0.50$. Using `scripts/calibrate_stage1.py`, we perform a grid sweep $t \in [0.05, 0.95]$ in increments of 0.05 **strictly on the validation set (`data/final/val`)**.

```python
# Criterion in scripts/calibrate_stage1.py
if v_acc > best_val_acc:
    best_val_acc = v_acc
    best_val_acc_t = t
```

### 4.2 Threshold Sweep Results (Validation vs. Test)

| Decision Threshold ($t$) | Val Accuracy | Val Macro F1 | Test Accuracy | Test Macro F1 | Calibration Status |
| :---: | :---: | :---: | :---: | :---: | :--- |
| $t = 0.50$ (Default) | 87.71% | 0.8136 | 87.58% | 0.8146 | Standard Baseline |
| **$t = 0.55$** | **88.57%** | **0.8378** | **88.20%** | **0.8314** | **Optimal Validation Accuracy** |
| $t = 0.60$ | 88.06% | 0.8408 | 87.54% | 0.8339 | Optimal Validation F1 |

### 4.3 Validation Override Sensitivity Check
To verify that $t = 0.55$ was not biased by the 8 validation-split label overrides, a sensitivity check was performed by temporarily excluding all validation overrides (`scripts/precheck_val_sensitivity.py`):
- **Validation Accuracy at $t = 0.55$ (Val Overrides Excluded)**: **88.49%** (vs **87.63%** at default $t = 0.50$).
- **Conclusion**: Threshold $t = 0.55$ is robust and optimal on validation data independent of label corrections.

---

## 5. Experimental Benchmarks & Results

All models were evaluated on the **2,568 clean held-out test set images**:

### 5.1 Stage 1 Model Performance Comparison

| Model Architecture | Threshold ($t$) | Raw Accuracy | Balanced Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **KNN Baseline** | $0.50$ | 74.96% | 50.00% | 0.3748 | 0.5000 | 0.4284 |
| **Custom DSConv Backbone** | $0.50$ | 87.58% | 78.59% | 0.8652 | 0.7859 | 0.8146 |
| **Custom DSConv Backbone (Calibrated)** | **$0.55$** | **88.20%** | **81.21%** | **0.8585** | **0.8121** | **0.8314** |
| **ResNet18 Transfer Learning** | $0.50$ | **93.54%** | **90.12%** | **0.9142** | **0.9012** | **0.9075** |

### 5.2 Stage 1 Confusion Matrix Analysis ($t = 0.55$)

$$\begin{pmatrix} 
\text{True Biodegradable (TN)} = 429 & \text{False Non-Bio (FP)} = 208 \\ 
\text{False Bio (FN)} = 95 & \text{True Non-Bio (TP)} = 1836 
\end{pmatrix}$$

- **Biodegradable Class Sensitivity (Recall)**: $\frac{429}{429 + 208} = \mathbf{67.35\%}$
- **Non-Biodegradable Class Sensitivity (Recall)**: $\frac{1836}{1836 + 95} = \mathbf{95.08\%}$
- **Balanced Accuracy**: $\frac{67.35\% + 95.08\%}{2} = \mathbf{81.21\%}$

### 5.3 Complete Hierarchical Pipeline Performance

| Stage Level | Target Categories | Top-1 Accuracy | Macro F1-Score | Key Performance Attribute |
| :--- | :---: | :---: | :---: | :--- |
| **Stage 1 (Binary)** | 2 | **88.20%** | **0.8314** | High Non-Bio recall (95.08%) |
| **Stage 2 (Coarse)** | 6 | **66.98%** | **0.5882** | Effective organic/glass isolation |
| **Stage 3 (Fine)** | 8 | **63.40%** | **0.5551** | Fine material texture discrimination |

---

## 6. Parameter-Matched Ablation Studies

### 6.1 Ablation Study 1: Multi-Scale DSConv vs. Single-Scale Conv Stacks
To evaluate whether performance gains stem from multi-scale kernel diversity or raw parameter count, we designed a parameter-matched single-scale ($7 \times 7$) plain convolutional backbone (`PlainConv2DBackbone`).

- **Full DSConv2D Backbone Parameters**: **95,472 parameters**
- **Plain Conv Stack Parameters**: **92,400 parameters** (Matched within 3.22%)

#### Results Summary (Stage 1 Threshold $t = 0.55$)

| Model Variant | Stage 1 Acc | Stage 1 F1 | Stage 2 Acc | Stage 2 F1 | Stage 3 Acc | Stage 3 F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full DSConv Backbone (Multi-Scale)** | **88.20%** | **0.8314** | **66.98%** | **0.5882** | **63.40%** | **0.5551** |
| **Plain Conv Stack (Single-Scale)** | 83.72% | 0.7681 | 61.35% | 0.5228 | 57.94% | 0.4916 |
| **Multi-Scale Advantage ($\Delta$)** | **+4.48 pp** | **+0.0633** | **+5.63 pp** | **+0.0654** | **+5.46 pp** | **+0.0635** |

### 6.2 Ablation Study 2: Conditioned vs. Flat Classifier Heads
To measure the benefit of stage-to-stage embedding propagation, we evaluated conditioned heads against flat independent classifiers without embedding feedback.

- **Stage 2 Model Parameters**: Conditioned `105,446` vs Flat `104,390` (Matched within 1.00%)
- **Stage 3 Model Parameters**: Conditioned `105,640` vs Flat `104,520` (Matched within 1.06%)

#### Results Summary (Stage 1 Threshold $t = 0.55$)

| Model Variant | Stage 2 Acc | Stage 2 F1 | Stage 3 Acc | Stage 3 F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Conditioned Heads (Hierarchical Embedding)** | **66.98%** | **0.5882** | **63.40%** | **0.5551** |
| **Flat Independent Heads (No Conditioning)** | 62.85% | 0.5442 | 58.71% | 0.5083 |
| **Conditioning Advantage ($\Delta$)** | **+4.13 pp** | **+0.0440** | **+4.69 pp** | **+0.0468** |

---

## 7. Qualitative Analysis & Failure Modes

Inspection of misclassified test samples (`results/stage1_hard_errors.csv` and `results/stage2b_paper_cardboard_hard_errors.csv`) highlights three primary failure modes:

1. **Textural Ambiguity (Paper vs. Cardboard)**: Thin, unprinted cardboard boxes with smooth surfaces are frequently misclassified as heavy kraft paper due to near-identical pixel intensity distributions.
2. **Soiled / Composite Materials**: Bio-contaminated food packaging (e.g., greasy pizza boxes) creates conflicting signals between organic waste textures and cardboard geometry.
3. **Extreme Scale Cap Distortions**: Small plastic caps ($< 15 \times 15$ pixels after cropping) lose fine edge features under spatial resolution downsampling ($128 \times 128$).

---

## 8. Summary of Software Repository Artifacts

All scripts and reports are organized within the workspace repository:

- `src/waste_classifier/hierarchical/backbone.py`: Contains `DSConv2DBackbone` and parameter-matched `PlainConv2DBackbone`.
- `src/waste_classifier/hierarchical/hierarchy.py`: Defines mapping dictionaries and `get_stage1_label()` override lookup logic.
- `scripts/preprocess_pipeline.py`: Implements the 7-step dataset pipeline including Step 4 global pHash deduplication.
- `scripts/calibrate_stage1.py`: Sweeps decision thresholds $t \in [0.05, 0.95]$ on validation predictions.
- `scripts/precheck_val_sensitivity.py`: Pre-check script verifying validation threshold robustness.
- `scripts/run_5seed_ablations.py`: Optimized 5-seed ablation training suite.
- `scripts/format_ablation_reports.py`: Formats multi-seed ablation tables and performs noise overlap verification.
- `evaluation_audit.md`: Formal pre-submission evaluation audit document.
- `results/ablation_dsconv.md`: Detailed multi-scale kernel ablation report.
- `results/ablation_conditioning.md`: Detailed head conditioning ablation report.

---
*End of Comprehensive Research Manuscript.*
