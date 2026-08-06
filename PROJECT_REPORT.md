# AI-Based Hierarchical Waste Classification System - Project Report

## 1. Abstract
This project implements a multi-stage **Hierarchical CNN** for sorting and classifying household and industrial waste from images. The system is designed to segment waste hierarchically into:
- Stage 1: Biodegradable vs. Non-biodegradable (binary classification)
- Stage 2: 6 Coarse Categories (material groupings)
- Stage 3: 8 Fine-grained Classes (specific target materials)

The architecture is implemented in PyTorch and deployed as a full-stack Next.js web application.

---

## 2. Problem Statement
Efficient recycling requires accurate waste segregation. Manual sorting is slow, inefficient, costly, and poses occupational health hazards. Automated optical sorting systems powered by deep learning can classify waste items at high speed and accuracy, routing materials to appropriate recycling channels.

---

## 3. Objectives
- Design and train a multi-stage Hierarchical CNN classifier.
- Process and balance a large dataset of genuine waste photographs.
- Implement memory caching to optimize GPU-accelerated training.
- Compare model performance against a legacy K-Nearest Neighbors (KNN) baseline.
- Deliver an interactive web application UI for real-time model predictions.

---

## 4. Dataset Composition
The dataset consists of **17,061 images** compiled from three sources:
1. **Garbage Classification V2**: 11,806 images.
2. **Garbage Classification (12 classes)**: 13,873 images.
3. **TACO (Trash Annotations in Context)**: 607 object samples cropped from bounding box annotations.

Preprocessing LSH clustering removed **8,262 near-duplicate images**, and corrupt/unusable files were filtered, yielding the final partitions below:

- **Training Split**: 11,938 images
- **Validation Split**: 2,555 images
- **Testing Split**: 2,568 images

### Final Class Quantities
- `cardboard`: 1,541 total
- `glass`: 2,217 total
- `metal`: 1,006 total
- `organic`: 1,035 total
- `paper`: 1,694 total
- `plastic`: 2,019 total
- `textile`: 7,037 total
- `battery`: 512 total

---

## 5. Model Architecture & Training
The classifier features a custom convolutional neural network backbone with parallel inception-style multi-scale branches (11x11, 9x9, 7x7, 5x5, 3x3) followed by depthwise-separable convolutional (DSConv) blocks to keep the model lightweight and fast.

Downstream classifiers (Stage 2 and 3) utilize **conditioning heads** where the predicted class embedding of the previous stage is concatenated with the backbone features. This ensures multi-task feature sharing and robust downstream boundaries.

### Hyperparameters
- **Epochs per stage**: 15 (45 epochs total)
- **Batch size**: 64
- **Optimizer**: Adam (lr=0.001)
- **Loss function**: Class-Weighted Focal Loss ($\gamma = 2.0$)
- **Data loading**: RAM tensor caching enabled

---

## 6. Evaluation Metrics & Comparison
Results obtained on the held-out test split of **2,568 real images**:

| Model | Stage | Classes | Precision (macro) | Recall (macro) | F1-Score | Accuracy | AUC | Inference Time (s/img) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **KNN Baseline** | Stage 1 | 2 | 0.3748 | 0.5000 | 0.4284 | 74.96% | 0.5000 | **0.00034s** |
| **Hierarchical CNN** | Stage 1 | 2 | **0.7511** | **0.8168** | **0.7638** | **79.52%** | **0.9033** | 0.00956s |
| **KNN Baseline** | Stage 2 | 6 | 0.0737 | 0.1667 | 0.1022 | 44.20% | 0.5000 | **0.00034s** |
| **Hierarchical CNN** | Stage 2 | 6 | **0.5882** | **0.6211** | **0.5882** | **66.98%** | **0.8344** | 0.00956s |
| **KNN Baseline** | Stage 3 | 8 | 0.0038 | 0.1250 | 0.0074 | 3.04% | 0.5000 | **0.00034s** |
| **Hierarchical CNN** | Stage 3 | 8 | **0.5492** | **0.5937** | **0.5551** | **63.40%** | **0.8251** | 0.00956s |

---

## 7. Conclusions & Final Review Summary
- The **Hierarchical CNN** achieves significantly better performance than the KNN Baseline across all stages.
- Stage 3 fine-grained classification accuracy reached **63.40%** compared to the baseline's **3.04%**, validating the effectiveness of the multi-scale DSConv CNN backbone and conditioned head design.
- The implementation runs efficiently on general laptop hardware (NVIDIA RTX 3050) and integrates smoothly with Next.js web serving.
