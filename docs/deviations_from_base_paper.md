# Deviations from Base Paper

This document describes the architectural and design deviations in our hierarchical CNN system compared to the reference paper: Nahiduzzaman et al., "An automated waste classification system using deep learning techniques," *Knowledge-Based Systems* 310 (2025) 113028.

---

## 1. Class Taxonomy Adaptation

- **Reference Paper**: 
  - Stage 1: 2 classes (biodegradable vs. non-biodegradable)
  - Stage 2: 9 classes
  - Stage 3: 36 classes
- **Our Implementation**: Adjusted to align with the 11 classes present in this final-year dataset:
  - **Stage 1 (2 classes)**:
    - `0`: Biodegradable (`organic`, `paper`, `cardboard`, `wood`)
    - `1`: Non-biodegradable (`glass`, `metal`, `plastic`, `textile`, `battery`, `ceramic`, `nylon`)
  - **Stage 2 (6 classes)**: Coarse groups:
    - `0`: Paper/Cardboard
    - `1`: Organic/Wood
    - `2`: Glass
    - `3`: Metal
    - `4`: Plastic/Nylon
    - `5`: Textile/Battery/Ceramic
  - **Stage 3 (11 classes)**: Fine-grained categories:
    - `battery`, `cardboard`, `ceramic`, `glass`, `metal`, `nylon`, `organic`, `paper`, `plastic`, `textile`, `wood`.

---

## 2. Conditioning vs. Dataset Routing

- **Reference Paper**: Routes input data downstream (i.e. separate classifiers are trained on isolated subsets of the dataset based on Stage 1 / Stage 2 outputs).
- **Our Implementation**: Implements **conditioned heads** using learnable class embeddings. The models take the previous stage's predicted class as an embedding vector and concatenate it with the CNN features at the linear classification head. This architecture allows:
  - Global weight sharing of the backbone feature representation.
  - End-to-end multi-task/multi-stage gradient flow.
  - Robustness against classification errors in early stages, as downstream heads learn to adjust classification boundaries based on conditioning state.

---

## 3. Classifier Heads and Data Sparsity

- **Reference Paper**: Employs an Ensemble Extreme Learning Machine (En-ELM) classifier.
- **Our Implementation**:
  - Employs Standard Cross-Entropy with a configurable Class-Weighted Focal Loss to handle class imbalance.
  - Integrates an optional **Prototypical metric learning head** for Stage 3 to alleviate severe class sparsity.

---

## 4. Project Scope

- **Reference Paper**: Discusses physical sorting setups.
- **Our Implementation**: Scoped exclusively to the software, dataset engineering, and deep learning modeling side. Physical sorting hardware, embedded firmware, and microcontroller integration are explicitly out of scope.
