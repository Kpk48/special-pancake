# Stage 1 Sanity Baseline Diagnosis Report (Phase 2)

This report details the comparison between the Stage 1 hierarchical custom backbone and the pretrained ResNet18 transfer-learning baseline.

## 1. Overall Performance Comparison

| Model | Backbone Type | Test Accuracy | Biodegradable Predictions | Non-Biodegradable Predictions |
| --- | --- | --- | --- | --- |
| **Custom Hierarchical CNN** | Custom From-Scratch | 79.52% | 989 | 1579 |
| **ResNet18 Baseline** | Pretrained (ImageNet) | 93.61% | 567 | 2001 |

> [!NOTE]
> The true test set distribution has **643** biodegradable images and **1925** non-biodegradable images. The custom backbone was heavily biased toward predicting biodegradable classes (over-predicting by 346 samples), whereas the ResNet18 baseline prediction distribution is much closer to the true distribution.

---

## 2. Baseline Error Breakdowns

### Error Rate by Material Class
The breakdown of ResNet18 baseline errors across the 8 fine-grained classes shows class-specific performance variation:

| Class Name | Total Test Images | Error Count | Error Rate | Category Type |
| --- | --- | --- | --- | --- |
| `paper` | 255 | 65 | 25.49% | Biodegradable |
| `cardboard` | 232 | 48 | 20.69% | Biodegradable |
| `battery` | 78 | 6 | 7.69% | Non-biodegradable |
| `plastic` | 304 | 16 | 5.26% | Non-biodegradable |
| `organic` | 156 | 7 | 4.49% | Biodegradable |
| `glass` | 334 | 10 | 3.00% | Non-biodegradable |
| `metal` | 152 | 4 | 2.63% | Non-biodegradable |
| `textile` | 1057 | 8 | 0.76% | Non-biodegradable |

### Error Rate by Source-Dataset Tag
The breakdown of ResNet18 baseline errors by source dataset reveals structural domain differences:

| Source Tag | Total Test Images | Error Count | Error Rate | Description |
| --- | --- | --- | --- | --- |
| `taco_` | 86 | 23 | 26.74% | Real-world in-context cropped bounding boxes |
| `gcv2_` | 556 | 69 | 12.41% | Garbage Classification V2 |
| `gc12_` | 1926 | 72 | 3.74% | Garbage Classification (12 classes) |

---

## 3. Diagnostic Summary

### Does the transfer baseline outperform the custom backbone significantly (>5pts)?
**Yes.** The ResNet18 baseline achieves a test accuracy of **93.61%**, which is **14.09 percentage points** higher than the custom backbone's **79.52%**. This is a massive, highly significant margin of improvement.

### Does the transfer baseline still show elevated error rates concentrated in specific source-dataset tags (particularly taco_*), even with a stronger pretrained backbone?
**Yes.** Despite the high overall accuracy, the baseline exhibits an error rate of **26.74%** on `taco_` samples and **12.41%** on `gcv2_` samples. In contrast, it achieves a very low error rate of **3.74%** on the cleaner `gc12_` samples.

### Is this primarily (a) a backbone-capacity problem, (b) a domain-shift/data-quality problem, or (c) both?
This is **both** a backbone-capacity and a domain-shift/data-quality problem:

1. **Backbone-Capacity Component:** The dramatic 14.09% accuracy increase demonstrates that the custom from-scratch backbone was severely bottlenecked by its representation learning capacity. A pretrained network with rich feature maps is critical for learning generalizable boundaries on this dataset.
2. **Domain-Shift/Data-Quality Component:** Even with the stronger backbone, the error rate on `taco_` is extremely high (26.74%). This is because `taco_` images are real-world photographs with complex lighting, dirt, and noise, whereas the `gc12_` dataset consists of clean, center-focused images with white backgrounds. The model struggles to transfer features across these domains. Furthermore, the high error rates in the `paper` (25.49%) and `cardboard` (20.69%) classes point to visual ambiguity between paper products and non-biodegradable packaging materials (such as juice boxes or wrappers), suggesting fine-grained class confusion that a binary split struggles to resolve.
