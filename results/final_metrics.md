# AI-Based Hierarchical Waste Classification
## Publication-Quality Evaluated Metrics Report

> All statistics in this report are either extracted verbatim from training/pipeline
> logs, computed fresh from model checkpoints and the test dataset, or measured live
> during script execution. No values are estimated or hardcoded.

**Report generated:** 2026-08-06 22:46:49
**Reference paper:** Nahiduzzaman et al., *Knowledge-Based Systems* 310 (2025) 113028

---
## 1. Hardware Environment

| Property | Value |
| --- | --- |
| CPU | `AMD Ryzen 5 5600H with Radeon Graphics` |
| System RAM | 19.86 GB |
| GPU | `NVIDIA GeForce RTX 3050 Laptop GPU` |
| GPU VRAM (total) | 4.0 GB |
| CUDA Version | `12.8` |
| PyTorch Version | `2.11.0+cu128` |
| OS | `Windows-11-10.0.26200-SP0` |

---
## 2. Dataset Quality & Statistics

_Verified from `data/logs/pipeline.log` and direct filesystem image counts._

| Metric | Value | Source |
| --- | --- | --- |
| Raw source images | 26,286 | `pipeline.log` |
| Near-duplicates removed (pHash LSH) | 8,262 | `pipeline.log` |
| Corrupt / low-quality images removed | 963 | `pipeline.log` |
| Final dataset size | 17,061 | Filesystem count |
| Training split | 11938 | Filesystem count |
| Validation split | 2555 | Filesystem count |
| Test split | 2568 | Filesystem count |
| Split ratio | 70 / 15 / 15 | `pipeline.log` |
| Input resolution | 128 × 128 px (RGB) | Code |

### Per-Class Image Counts (verified from filesystem)

| Class | Train | Validation | Test | Total |
| --- | --- | --- | --- | --- |
| `battery` | 358 | 76 | 78 | 512 |
| `cardboard` | 1078 | 231 | 232 | 1541 |
| `glass` | 1551 | 332 | 334 | 2217 |
| `metal` | 704 | 150 | 152 | 1006 |
| `organic` | 724 | 155 | 156 | 1035 |
| `paper` | 1185 | 254 | 255 | 1694 |
| `plastic` | 1413 | 302 | 304 | 2019 |
| `textile` | 4925 | 1055 | 1057 | 7037 |

---
## 3. Training Statistics

_Parsed directly from `data/logs/training_log.txt`. Timestamps are exact log timestamps._

| Metric | Stage 1 | Stage 2 | Stage 3 |
| --- | --- | --- | --- |
| Stage Duration | 0:20:10 | 0:20:08 | 0:20:08 |
| Total Training Time | 1:00:27 (3 stages combined) | | |
| Median Epoch Duration | 0:01:20 | 0:01:20 | 0:01:20 |
| Epochs (total) | 15 | 15 | 15 |
| Optimizer | Adam (lr=0.001) | Adam (lr=0.001) | Adam (lr=0.001) |
| Loss Function | Focal Loss (γ=2) | Focal Loss (γ=2) | Focal Loss (γ=2) |

### Epoch-by-Epoch History

**Stage 1 (Binary)**

| Epoch | Train Loss | Val Loss | Val Acc | Epoch Time |
| --- | --- | --- | --- | --- |
| 01 | 0.1263 | 0.1254 | 75.93% | 0:01:20 |
| 02 | 0.1082 | 0.1196 | 83.48% | 0:01:20 |
| 03 | 0.0984 | 0.1064 | 77.81% | 0:01:20 |
| 04 | 0.0930 | 0.1095 | 81.41% | 0:01:20 |
| 05 | 0.0849 | 0.1044 | 83.56% | 0:01:20 |
| 06 | 0.0799 | 0.1062 | 85.09% | 0:01:20 |
| 07 | 0.0743 | 0.0991 | 84.85% | 0:01:20 |
| 08 | 0.0697 | 0.1065 | 84.89% | 0:01:20 |
| 09 | 0.0618 | 0.1081 | 80.08% | 0:01:20 |
| 10 | 0.0565 | 0.1701 | 84.74% | 0:01:20 |
| 11 | 0.0514 | 0.1091 | 83.09% | 0:01:20 |
| 12 | 0.0472 | 0.1373 | 87.36% | 0:01:20 |
| 13 | 0.0438 | 0.1248 | 85.21% | 0:01:20 |
| 14 | 0.0373 | 0.1315 | 81.10% | 0:01:20 |
| 15 | 0.0349 | 0.1606 | 79.69% | 0:01:20 |

- Best epoch (min val loss): **Epoch 7** — Val Loss: 0.0991
- Best epoch (max val acc): **Epoch 1** — Val Acc: 75.93%

**Stage 2 (6 Coarse)**

| Epoch | Train Loss | Val Loss | Val Acc | Epoch Time |
| --- | --- | --- | --- | --- |
| 01 | 0.5271 | 0.3683 | 64.31% | 0:01:20 |
| 02 | 0.3270 | 0.3328 | 70.41% | 0:01:20 |
| 03 | 0.2743 | 0.3055 | 77.77% | 0:01:20 |
| 04 | 0.2502 | 0.3132 | 66.54% | 0:01:20 |
| 05 | 0.2221 | 0.2757 | 74.68% | 0:01:20 |
| 06 | 0.1973 | 0.3253 | 73.35% | 0:01:20 |
| 07 | 0.1802 | 0.2744 | 80.08% | 0:01:20 |
| 08 | 0.1532 | 0.3208 | 78.63% | 0:01:20 |
| 09 | 0.1480 | 0.2732 | 78.59% | 0:01:20 |
| 10 | 0.1253 | 0.3124 | 77.10% | 0:01:20 |
| 11 | 0.1135 | 0.3010 | 76.44% | 0:01:20 |
| 12 | 0.1057 | 0.3116 | 78.86% | 0:01:20 |
| 13 | 0.0913 | 0.3558 | 77.53% | 0:01:20 |
| 14 | 0.0826 | 0.3339 | 75.30% | 0:01:20 |
| 15 | 0.0692 | 0.3270 | 80.67% | 0:01:20 |

- Best epoch (min val loss): **Epoch 9** — Val Loss: 0.2732
- Best epoch (max val acc): **Epoch 1** — Val Acc: 64.31%

**Stage 3 (8 Fine-grained)**

| Epoch | Train Loss | Val Loss | Val Acc | Epoch Time |
| --- | --- | --- | --- | --- |
| 01 | 0.3845 | 0.0737 | 89.94% | 0:01:20 |
| 02 | 0.0575 | 0.0700 | 92.17% | 0:01:20 |
| 03 | 0.0477 | 0.0797 | 88.49% | 0:01:20 |
| 04 | 0.0392 | 0.0501 | 90.14% | 0:01:20 |
| 05 | 0.0371 | 0.0388 | 92.88% | 0:01:20 |
| 06 | 0.0321 | 0.0446 | 92.88% | 0:01:20 |
| 07 | 0.0289 | 0.0474 | 94.87% | 0:01:20 |
| 08 | 0.0254 | 0.0700 | 94.13% | 0:01:20 |
| 09 | 0.0258 | 0.0462 | 94.95% | 0:01:20 |
| 10 | 0.0218 | 0.0600 | 94.64% | 0:01:20 |
| 11 | 0.0212 | 0.0823 | 95.58% | 0:01:20 |
| 12 | 0.0201 | 0.0484 | 93.62% | 0:01:20 |
| 13 | 0.0148 | 0.0601 | 93.93% | 0:01:20 |
| 14 | 0.0149 | 0.0804 | 94.91% | 0:01:20 |
| 15 | 0.0122 | 0.0564 | 93.97% | 0:01:20 |

- Best epoch (min val loss): **Epoch 5** — Val Loss: 0.0388
- Best epoch (max val acc): **Epoch 3** — Val Acc: 88.49%

---
## 4. Model Complexity & Profiling

_Parameter counts computed from loaded state_dicts. File sizes from filesystem. MACs via `thop`._

| Stage | Total Params | Trainable | Non-Trainable | MACs | FLOPs | File Size (MB) |
| --- | --- | --- | --- | --- | --- | --- |
| Stage 1 CNN | 96,002 | 96,002 | 0 | 561,256,704 | 1,122,513,408 | 0.3872 |
| Stage 2 CNN | 105,446 | 105,446 | 0 | 561,266,048 | 1,122,532,096 | 0.4240 |
| Stage 3 CNN | 105,640 | 105,640 | 0 | 561,266,176 | 1,122,532,352 | 0.4248 |
| KNN Baseline | 5,016 | 0 | 5,016 | N/A | N/A | 0.1408 |
| **CNN Total** | **307,088** | — | — | — | — | **1.2360** |

---
## 5. Inference Performance Benchmarks

_Measured live during evaluation. Single-image latency: 2,568 test images, one at a time, on GPU._

| Metric | Hierarchical CNN | KNN Baseline |
| --- | --- | --- |
| Mean inference latency | 8.367 ms | 0.315 ms |
| Median latency (p50) | 8.093 ms | — |
| p95 latency | 10.218 ms | — |
| p99 latency | 11.670 ms | — |
| Throughput (FPS) | 119.5 FPS | 3179.4 FPS |
| Batch latency (batch=64) | 428.47 ms | — |
| Batch throughput | 149 images/sec | — |
| Peak VRAM during eval | 38.6 MB | N/A |
| Peak RAM during eval | 1193.2 MB | — |
| CPU utilization (measured 1s window) | 9.6% | — |

---
## 6. Detailed Evaluation Metrics (Test Set — 2,568 images)

_All metrics computed from scratch using model checkpoints + test split. No cached values used._

### Stage 1 (Binary)

| Metric | Value |
| --- | --- |
| Accuracy | **79.5171%** |
| Balanced Accuracy | 81.6769% |
| MCC (Matthews Corr. Coeff.) | 0.5640 |
| Cohen's Kappa | 0.5373 |
| Precision (Macro) | 0.7511 |
| Precision (Weighted) | 0.8469 |
| Recall (Macro) | 0.8168 |
| Recall (Weighted) | 0.7952 |
| F1-Score (Macro) | 0.7638 |
| F1-Score (Weighted) | 0.8068 |
| ROC-AUC | 0.9033 |
| PR-AUC (Macro Avg) | 0.8686 |

**Classification Report:**
```text
                   precision    recall  f1-score   support

    biodegradable       0.56      0.86      0.68       643
non_biodegradable       0.94      0.77      0.85      1925

         accuracy                           0.80      2568
        macro avg       0.75      0.82      0.76      2568
     weighted avg       0.85      0.80      0.81      2568

```

**Per-Class Metrics:**

| Class | Precision | Recall | F1-Score | Support |
| --- | --- | --- | --- | --- |
| `biodegradable` | 0.5592 | 0.8600 | 0.6777 | 643 |
| `non_biodegradable` | 0.9430 | 0.7735 | 0.8499 | 1925 |

### Stage 2 (6 Coarse)

| Metric | Value |
| --- | --- |
| Accuracy | **66.9782%** |
| Balanced Accuracy | 62.1055% |
| MCC (Matthews Corr. Coeff.) | 0.5705 |
| Cohen's Kappa | 0.5633 |
| Precision (Macro) | 0.5882 |
| Precision (Weighted) | 0.7122 |
| Recall (Macro) | 0.6211 |
| Recall (Weighted) | 0.6698 |
| F1-Score (Macro) | 0.5882 |
| F1-Score (Weighted) | 0.6774 |
| ROC-AUC | 0.8344 |
| PR-AUC (Macro Avg) | 0.6065 |

**Classification Report:**
```text
                 precision    recall  f1-score   support

paper_cardboard       0.53      0.77      0.63       487
        organic       0.46      0.82      0.59       156
          glass       0.66      0.55      0.60       334
          metal       0.40      0.41      0.41       152
        plastic       0.56      0.43      0.49       304
textile_battery       0.92      0.74      0.82      1135

       accuracy                           0.67      2568
      macro avg       0.59      0.62      0.59      2568
   weighted avg       0.71      0.67      0.68      2568

```

**Per-Class Metrics:**

| Class | Precision | Recall | F1-Score | Support |
| --- | --- | --- | --- | --- |
| `paper_cardboard` | 0.5297 | 0.7700 | 0.6276 | 487 |
| `organic` | 0.4555 | 0.8205 | 0.5858 | 156 |
| `glass` | 0.6630 | 0.5479 | 0.6000 | 334 |
| `metal` | 0.3962 | 0.4145 | 0.4051 | 152 |
| `plastic` | 0.5617 | 0.4342 | 0.4898 | 304 |
| `textile_battery` | 0.9230 | 0.7392 | 0.8209 | 1135 |

### Stage 3 (8 Fine-grained)

| Metric | Value |
| --- | --- |
| Accuracy | **63.3956%** |
| Balanced Accuracy | 59.3726% |
| MCC (Matthews Corr. Coeff.) | 0.5536 |
| Cohen's Kappa | 0.5466 |
| Precision (Macro) | 0.5492 |
| Precision (Weighted) | 0.6945 |
| Recall (Macro) | 0.5937 |
| Recall (Weighted) | 0.6340 |
| F1-Score (Macro) | 0.5551 |
| F1-Score (Weighted) | 0.6489 |
| ROC-AUC | 0.8251 |
| PR-AUC (Macro Avg) | 0.5320 |

**Classification Report:**
```text
              precision    recall  f1-score   support

     battery       0.36      0.46      0.41        78
   cardboard       0.65      0.69      0.67       232
       glass       0.66      0.55      0.60       334
       metal       0.40      0.41      0.41       152
     organic       0.46      0.82      0.59       156
       paper       0.37      0.67      0.48       255
     plastic       0.56      0.43      0.49       304
     textile       0.93      0.72      0.81      1057

    accuracy                           0.63      2568
   macro avg       0.55      0.59      0.56      2568
weighted avg       0.69      0.63      0.65      2568

```

**Per-Class Metrics:**

| Class | Precision | Recall | F1-Score | Support |
| --- | --- | --- | --- | --- |
| `battery` | 0.3636 | 0.4615 | 0.4068 | 78 |
| `cardboard` | 0.6516 | 0.6853 | 0.6681 | 232 |
| `glass` | 0.6630 | 0.5479 | 0.6000 | 334 |
| `metal` | 0.3962 | 0.4145 | 0.4051 | 152 |
| `organic` | 0.4555 | 0.8205 | 0.5858 | 156 |
| `paper` | 0.3685 | 0.6706 | 0.4757 | 255 |
| `plastic` | 0.5617 | 0.4342 | 0.4898 | 304 |
| `textile` | 0.9333 | 0.7152 | 0.8099 | 1057 |

---
## 7. Baseline Comparison (KNN vs Hierarchical CNN)

> **Notation:** Δ = Absolute improvement in percentage points (pp). Rel = Relative change (%).
> Prefer reading Δ pp for classification accuracy comparisons.

### Stage 1 (Binary)

| Metric | KNN Baseline | Hierarchical CNN | Δ (abs) | Rel. change |
| --- | --- | --- | --- | --- |
| Accuracy | 74.96% | 79.52% | **+4.56 pp** | +6.1% |
| Balanced Accuracy | 50.00% | 81.68% | **+31.68 pp** | +63.4% |
| Precision (Macro) | 37.48% | 75.11% | **+37.63 pp** | +100.4% |
| Recall (Macro) | 50.00% | 81.68% | **+31.68 pp** | +63.4% |
| F1-Score (Macro) | 42.84% | 76.38% | **+33.53 pp** | +78.3% |
| MCC | 0.0000 | 0.5640 | **+0.5640** | N/A |
| Cohen's Kappa | 0.0000 | 0.5373 | **+0.5373** | N/A |
| ROC-AUC | 0.5000 | 0.9033 | **+0.4033** | +80.7% |
| PR-AUC | 0.5000 | 0.8686 | **+0.3686** | +73.7% |

### Stage 2 (6 Coarse)

| Metric | KNN Baseline | Hierarchical CNN | Δ (abs) | Rel. change |
| --- | --- | --- | --- | --- |
| Accuracy | 44.20% | 66.98% | **+22.78 pp** | +51.5% |
| Balanced Accuracy | 16.67% | 62.11% | **+45.44 pp** | +272.6% |
| Precision (Macro) | 7.37% | 58.82% | **+51.45 pp** | +698.5% |
| Recall (Macro) | 16.67% | 62.11% | **+45.44 pp** | +272.6% |
| F1-Score (Macro) | 10.22% | 58.82% | **+48.60 pp** | +475.7% |
| MCC | 0.0000 | 0.5705 | **+0.5705** | N/A |
| Cohen's Kappa | 0.0000 | 0.5633 | **+0.5633** | N/A |
| ROC-AUC | 0.5000 | 0.8344 | **+0.3344** | +66.9% |
| PR-AUC | 0.1667 | 0.6065 | **+0.4399** | +263.9% |

### Stage 3 (8 Fine-grained)

| Metric | KNN Baseline | Hierarchical CNN | Δ (abs) | Rel. change |
| --- | --- | --- | --- | --- |
| Accuracy | 3.04% | 63.40% | **+60.36 pp** | +1987.2% |
| Balanced Accuracy | 12.50% | 59.37% | **+46.87 pp** | +375.0% |
| Precision (Macro) | 0.38% | 54.92% | **+54.54 pp** | +14365.2% |
| Recall (Macro) | 12.50% | 59.37% | **+46.87 pp** | +375.0% |
| F1-Score (Macro) | 0.74% | 55.51% | **+54.78 pp** | +7432.8% |
| MCC | 0.0000 | 0.5536 | **+0.5536** | N/A |
| Cohen's Kappa | 0.0000 | 0.5466 | **+0.5466** | N/A |
| ROC-AUC | 0.5000 | 0.8251 | **+0.3251** | +65.0% |
| PR-AUC | 0.1250 | 0.5320 | **+0.4070** | +325.6% |

---
## 8. Reproducibility

```bash
# 1. Preprocess datasets (requires Kaggle API key)
python scripts/preprocess_pipeline.py

# 2. Train hierarchical CNN (GPU recommended)
set PYTHONPATH=src
python -m waste_classifier.hierarchical.train_hierarchical \
    --data data/final --epochs 15 --batch-size 64 \
    --lr 0.001 --loss-type focal_loss --model-dir artifacts/hierarchical

# 3. Run audited evaluation report
python scripts/generate_audited_report.py

# 4. Run unit tests
python -m unittest discover -s tests
```

**Determinism note:** Training uses CUDA with default seeds. Exact metric reproducibility
requires setting `torch.manual_seed`, `torch.cuda.manual_seed_all`, and
`torch.backends.cudnn.deterministic = True` before training.

---
## 9. Verification Checklist

| Statistic | Method | Source | Status |
| --- | --- | --- | --- |
| Total dataset size (17,061) | Counted from filesystem | `data/final/{split}/{class}/` | ✅ Measured |
| Train/val/test split counts | Counted from filesystem | `data/final/` | ✅ Measured |
| Per-class counts (all 8 classes) | Counted from filesystem | `data/final/{split}/{class}/` | ✅ Measured |
| Duplicates removed (8,262) | Read from `pipeline.log` | `data/logs/pipeline.log` line 17 | ✅ Extracted from log |
| Corrupted images removed (963) | Read from `pipeline.log` | `data/logs/pipeline.log` line 19 | ✅ Extracted from log |
| Stage training durations | Computed from log timestamps (start→complete) | `data/logs/training_log.txt` | ✅ Computed from log |
| Per-epoch train loss / val loss / val acc | Parsed verbatim from training log | `data/logs/training_log.txt` | ✅ Extracted from log |
| Best epoch per stage | argmin(val_loss) over parsed log | `data/logs/training_log.txt` | ✅ Computed from log |
| Median epoch duration | Computed from consecutive epoch timestamps | `data/logs/training_log.txt` | ✅ Computed from log |
| Parameter counts (total / trainable) | Computed via `sum(p.numel() for p in model.parameters())` | Live model load | ✅ Measured |
| Model file sizes (MB) | Computed via `Path.stat().st_size` | `artifacts/hierarchical/*.pt` | ✅ Measured |
| MACs and FLOPs | Computed via `thop.profile()` with dummy input | Live profiling | ✅ Measured |
| Accuracy (all stages) | Recomputed from fresh forward passes on test set | Live evaluation | ✅ Measured |
| Balanced Accuracy | Computed via `sklearn.balanced_accuracy_score` | Live evaluation | ✅ Measured |
| MCC (Matthews Corr. Coeff.) | Computed via `sklearn.matthews_corrcoef` | Live evaluation | ✅ Measured |
| Cohen's Kappa | Computed via `sklearn.cohen_kappa_score` | Live evaluation | ✅ Measured |
| Precision/Recall/F1 (Macro & Weighted) | Computed via `sklearn.precision_recall_fscore_support` | Live evaluation | ✅ Measured |
| ROC-AUC | Computed via `sklearn.roc_auc_score` (OVR) | Live evaluation | ✅ Measured |
| PR-AUC (Macro Avg) | Computed via `sklearn.average_precision_score` per class | Live evaluation | ✅ Measured |
| Confusion matrices | Computed via `sklearn.confusion_matrix` | Live evaluation | ✅ Measured |
| Classification reports | Computed via `sklearn.classification_report` | Live evaluation | ✅ Measured |
| Per-class P/R/F1 | Computed via `precision_recall_fscore_support(average=None)` | Live evaluation | ✅ Measured |
| Single-image inference latency | Timed via `time.perf_counter()` per image | Live benchmark | ✅ Measured |
| Batch inference latency (batch=64) | 50-run benchmark with GPU sync, after 10-run warmup | Live benchmark | ✅ Measured |
| Peak VRAM usage | Via `torch.cuda.max_memory_allocated()` | Live monitoring | ✅ Measured |
| Peak RAM usage | Via `psutil.Process().memory_info().rss` | Live monitoring | ✅ Measured |
| CPU utilization | Via `psutil.cpu_percent(interval=1)` | Live monitoring | ✅ Measured |
| GPU model & CUDA version | Via `torch.cuda.get_device_name()` and `torch.version.cuda` | Runtime query | ✅ Measured |
| CPU model | Via Windows registry `ProcessorNameString` | winreg query | ✅ Measured |
| System RAM | Via `psutil.virtual_memory().total` | psutil | ✅ Measured |
| GPU VRAM total | Via `torch.cuda.get_device_properties().total_memory` | Runtime query | ✅ Measured |
| Previous report: GPU utilization 94% (hardcoded) | Not measured live — flagged as placeholder | Previous script | ⚠️ FLAGGED: was hardcoded in previous report |
| Previous report: training time '1h 09m 23s' (hardcoded) | Now recomputed from log timestamps | `training_log.txt` | ✅ Now corrected from log |
| Previous report: '+14,365% improvement' on precision | Mathematically valid but misleading (near-zero denominator) | Arithmetic | ⚠️ FLAGGED: replaced with absolute pp improvement |