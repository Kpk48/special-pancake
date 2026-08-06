# AI-Based Hierarchical Waste Classification System

This repository implements a multi-stage **Hierarchical CNN** for classifying waste images into structured material categories. It is designed as a final-year engineering project extending the architecture in Nahiduzzaman et al., "An automated waste classification system using deep learning techniques" (*Knowledge-Based Systems*, 2025).

The system classifies waste across three hierarchical levels:
1. **Stage 1 (Binary)**: Biodegradable vs. Non-biodegradable.
2. **Stage 2 (6 Coarse Categories)**: `paper_cardboard`, `organic`, `glass`, `metal`, `plastic`, `textile_battery`.
3. **Stage 3 (8 Fine-grained Classes)**: `battery`, `cardboard`, `glass`, `metal`, `organic`, `paper`, `plastic`, `textile`.

---

## 🚀 Key Achievements

- **Production-Ready Deep Learning**: Shifted from a simple KNN baseline to a multi-stage **Hierarchical CNN** built in PyTorch.
- **Genuine Datasets**: Merged, cleaned, and split **17,061 real photographs** from *Garbage Classification V2*, *Garbage Classification (12 classes)*, and *TACO bounding box crops*.
- **RAM Caching & GPU Acceleration**: Implemented in-memory tensor caching to bypass Windows disk reading overhead, allowing full CUDA GPU training on an NVIDIA RTX 3050 Laptop GPU.
- **Modern Next.js Frontend**: A full-stack web UI built with Next.js, React, and TypeScript that interfaces with the trained PyTorch models for inference.

---

## 📊 Dataset Statistics

The preprocessing pipeline cleaned duplicates using Perceptual Hashing (pHash), removed corrupt/low-dimension samples, and partitioned the data:

- **Training set**: 11,938 images
- **Validation set**: 2,555 images
- **Testing set**: 2,568 images
- **Total Dataset Size**: 17,061 images

### Class Distribution

| Class Name | Train Split | Val Split | Test Split | Total Images |
| --- | --- | --- | --- | --- |
| `cardboard` | 1,078 | 231 | 232 | 1,541 |
| `glass` | 1,551 | 332 | 334 | 2,217 |
| `metal` | 704 | 150 | 152 | 1,006 |
| `organic` | 724 | 155 | 156 | 1,035 |
| `paper` | 1,185 | 254 | 255 | 1,694 |
| `plastic` | 1,413 | 302 | 304 | 2,019 |
| `textile` | 4,925 | 1,055 | 1,057 | 7,037 |
| `battery` | 358 | 76 | 78 | 512 |

---

## 📈 Performance Summary

Tested on the held-out test split of **2,568 real images**, comparing the **Hierarchical CNN** against the **KNN Baseline**:

| Model | Stage | Classes | Precision (macro) | Recall (macro) | F1-Score | Accuracy | AUC | Inference Time (s/img) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **KNN Baseline** | Stage 1 | 2 | 0.3748 | 0.5000 | 0.4284 | 74.96% | 0.5000 | **0.00034s** |
| **Hierarchical CNN** | Stage 1 | 2 | **0.7511** | **0.8168** | **0.7638** | **79.52%** | **0.9033** | 0.00956s |
| **KNN Baseline** | Stage 2 | 6 | 0.0737 | 0.1667 | 0.1022 | 44.20% | 0.5000 | **0.00034s** |
| **Hierarchical CNN** | Stage 2 | 6 | **0.5882** | **0.6211** | **0.5882** | **66.98%** | **0.8344** | 0.00956s |
| **KNN Baseline** | Stage 3 | 8 | 0.0038 | 0.1250 | 0.0074 | 3.04% | 0.5000 | **0.00034s** |
| **Hierarchical CNN** | Stage 3 | 8 | **0.5492** | **0.5937** | **0.5551** | **63.40%** | **0.8251** | 0.00956s |

---

## ⚙️ Running the Project

### 1. Run Preprocessing Pipeline
To download the datasets, crop TACO bounding boxes, deduplicate using Pigeonhole LSH clustering, and write final splits:
```bash
python scripts/preprocess_pipeline.py
```

### 2. Train the Hierarchical CNN
To train all three stages sequentially for 15 epochs on CUDA:
```bash
set PYTHONPATH=src&& python -m waste_classifier.hierarchical.train_hierarchical --data data/final --epochs 15 --batch-size 64 --lr 0.001 --loss-type focal_loss --max-samples-per-class 0 --model-dir artifacts/hierarchical
```

### 3. Run Evaluation
To evaluate both models on the test split, producing report card metrics and confusion statistics:
```bash
set PYTHONPATH=src&& python -m waste_classifier.evaluate --data data/final --knn-model artifacts/waste_model.json --cnn-model-dir artifacts/hierarchical --out-md results/comparison.md --out-png results/comparison.png
```

### 4. Start Next.js App
Start the Next.js development server to browse the interactive UI:
```bash
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) to upload images and run real-time hierarchical inference!

### 5. Run Unit Tests
To confirm network shapes and logic:
```bash
set PYTHONPATH=src&& python -m unittest discover -s tests
```
