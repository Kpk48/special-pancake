# AI-Based Waste Classification System - Project Notes

## Project Abstract

The AI-Based Waste Classification System classifies waste images into material categories such as cardboard, glass, metal, organic, paper, and plastic. The goal is to support faster and more consistent waste segregation in recycling centers, smart bins, and educational sustainability applications.

## Problem Statement

Manual waste segregation is time-consuming, inconsistent, and can expose workers to unsafe material. An image-based classifier can assist by identifying the likely material category of a waste item and routing it to the correct disposal or recycling process.

## Objectives

- Build an image classification workflow for waste segregation.
- Prepare a dataset with labeled waste categories.
- Train a model and evaluate its performance.
- Provide a simple interface for predicting the class of a new waste image.
- Document real datasets suitable for the final experiment.

## Dataset Used Now

The repo uses a generated demo dataset:

```text
data/demo_waste/
```

It contains 192 synthetic PPM images across six classes:

- cardboard
- glass
- metal
- organic
- paper
- plastic

This dataset has already been used to train `artifacts/waste_model.json`.

## Datasets To Use For Final Work

| Dataset | Use Case | Notes |
| --- | --- | --- |
| TrashNet | Baseline image classification | Compact and common in academic waste classification projects. |
| Kaggle Garbage Classification (12 classes) | Multi-class household waste classification | 15,150 images across 12 household waste classes according to the Kaggle dataset card. |
| Garbage Dataset / V2 | Larger final experiment | Kaggle dataset card lists 10 classes and 13,348 images. |
| TACO | Detection or segmentation | Use if the project should locate waste objects in real-world scenes. |
| YOLO Garbage Detection | Real-time object detection | YOLO train/valid/test format with six waste classes. |

## Methodology

1. Collect labeled images with one folder per waste category.
2. Preprocess images into a consistent format and size.
3. Extract image features or train a CNN.
4. Split data into training and testing sets.
5. Train the classifier.
6. Evaluate using accuracy, precision, recall, F1-score, and confusion matrix.
7. Deploy prediction through a CLI or browser-based UI.

## Current Model

The current implementation uses a k-nearest-neighbors classifier over handcrafted image features:

- RGB channel means
- RGB channel standard deviations
- brightness statistics
- simple texture score
- dominant color ratios

This is intentionally dependency-free so it can run immediately on the available system. For the final version, replace the baseline with MobileNetV2, EfficientNet, or a custom CNN trained on a real dataset.

## Current Training Result

The demo training command:

```bash
PYTHONPATH=src python3 -m waste_classifier.train --data data/demo_waste --model artifacts/waste_model.json
```

Result:

- Training images: 144
- Test images: 48
- Accuracy: 1.00 on the synthetic demo test split

The score proves the pipeline works, not that the system is production-ready. Real photographs will be harder and should be used for final evaluation.

## Future Enhancements

- Add JPEG/PNG image support using Pillow or OpenCV.
- Train a transfer-learning CNN on TrashNet or Garbage Classification V2.
- Add confusion matrix visualization.
- Add camera input for real-time classification.
- Add hazardous/e-waste classes such as battery and medical waste.
