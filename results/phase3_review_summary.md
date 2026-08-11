# Phase 3 — Semi-Automated Composite Relabeling Review Summary

This report summarizes the results of running the ResNet18 Stage 1 baseline over the full dataset (10,308 total images of paper, cardboard, and textile) to surface high-confidence model disagreements for manual review.

## 1. Flagging Statistics

*   **Total Model-Flagged Images (Confidence > 0.85, prediction disagrees with label):** 55 images
*   **Total Known-Composite TACO Images (by filename prefix rule):** 0 images

> [!NOTE]
> No "known-composite" TACO images were flagged by the filename rule. A search of the final dataset confirmed that none of the TACO subcategories representing composite items (e.g. `Paper cup`, `Meal carton`, `Pizza box`, `Wrapping paper`, `Magazine paper`, `Aluminium blister pack`) are present in the dataset splits. They were either filtered out during deduplication or were not included in the source download. All TACO paper/cardboard images present in the final dataset belong to the `Paper bag` or `Corrugated carton` subcategories.

### Flagged Images by Class

| Fine-grained Class | Total in Dataset | Model Flagged | % of Class | Known Composite |
| :--- | :--- | :--- | :--- | :--- |
| `cardboard` | 1,548 | 11 | 0.7% | 0 |
| `paper` | 1,723 | 43 | 2.5% | 0 |
| `textile` | 7,037 | 1 | 0.0% | 0 |

---

## 2. Per-Class and Source-Tag Breakdown

### Cardboard
Total model-flagged: **11**
*   `gc12_`: 4
*   `gcv2_`: 5
*   `unknown` (extra dataset): 2

### Paper
Total model-flagged: **43**
*   `gc12_`: 26
*   `gcv2_`: 14
*   `taco_`: 3

### Textile
Total model-flagged: **1**
*   `gc12_` (`gc12_shoes_6b1414.jpg`): 1

---

## 3. Visual Review Batches (Contact Sheets)

The 55 model-flagged images have been exported into contact sheets containing up to 30 images each in the directory `results/phase3_review_batches/`:
1.  **Batch 1 (`model_flagged_batch01_of02.png`):** 30 images (mostly high-confidence paper cups, drink cartons, wrappers, and cards).
2.  **Batch 2 (`model_flagged_batch02_of02.png`):** 25 images (including cardboard juice boxes, envelopes, and 1 shoe image).

The complete list of flagged file paths, true labels, model predictions, and confidence scores is saved in [phase3_flagged_images.csv](file:///c:/Users/mrbub/special-pancake/results/phase3_flagged_images.csv).

---

## 4. Next Steps

Please review the contact sheets in `results/phase3_review_batches/` or the list in [phase3_flagged_images.csv](file:///c:/Users/mrbub/special-pancake/results/phase3_flagged_images.csv). Once you provide the list of confirmed relabels (specifying which images should be remapped to non-biodegradable or corrected), we will implement the dataset relabeling and proceed to the retraining phase.
