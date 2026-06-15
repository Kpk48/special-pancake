# AI-Based Waste Classification System

This project classifies waste images into material categories for smart-bin and recycling workflows. It is designed for a final-year review: the repo includes a working training pipeline, a demo image dataset, a saved model, a CLI predictor, a lightweight local web app, tests, and notes on the larger datasets to use for the full report.

The current implementation uses a dependency-free baseline classifier so it runs on a plain Python installation. It extracts simple RGB histogram and texture features from images, then trains a k-nearest-neighbors classifier. For final deployment, you can replace the baseline with a CNN or transfer-learning model while keeping the same dataset folder structure.

## What Is Built

- Multi-class waste classifier
- Demo dataset generator
- Training and evaluation pipeline
- Saved JSON model artifact
- CLI prediction command
- Production-style React + TypeScript frontend using Next.js
- Next.js API route that calls the trained Python model
- Unit tests for feature extraction and model prediction

## Dataset Used In This Repo

I generated and used a local demo dataset at:

```text
data/demo_waste/
```

Classes:

- `cardboard`
- `glass`
- `metal`
- `organic`
- `paper`
- `plastic`

The demo images are synthetic `.ppm` images with class-specific colors and textures. They are intentionally small so the project can be trained and verified without downloads. This is good for review demonstrations and code validation, but the final report should clearly state that production accuracy must be measured on real waste photographs.

## Recommended Real Datasets

Use these datasets for the actual experiment section:

1. **TrashNet**
   - Best for: compact academic baseline.
   - Classes commonly include cardboard, glass, metal, paper, plastic, and trash.
   - Source: [TrashNet GitHub](https://github.com/garythung/trashnet)

2. **Garbage Classification (12 classes)**
   - Best for: stronger multi-class household waste classification.
   - Includes 15,150 images across 12 classes such as paper, cardboard, biological, metal, plastic, glass variants, clothes, shoes, batteries, and trash.
   - Source: [Kaggle Garbage Classification](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)

3. **Garbage Dataset / Garbage Classification V2**
   - Best for: larger final-year evaluation with more class coverage.
   - Kaggle lists 10 classes and 13,348 images in the current dataset card.
   - Source: [Kaggle Garbage Dataset](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2)

4. **TACO**
   - Best for: object detection or segmentation instead of only image classification.
   - Use this if your project evolves into locating waste objects inside a scene.
   - Source: [TACO paper](https://arxiv.org/abs/2003.06975)

5. **YOLO Garbage Detection - 6 Waste Categories**
   - Best for: real-time detection with bounding boxes.
   - Kaggle lists train/valid/test folders with YOLO labels for biodegradable, cardboard, glass, metal, paper, and plastic.
   - Source: [Kaggle Garbage Detection](https://www.kaggle.com/datasets/viswaprakash1990/garbage-detection/data)

## Expected Dataset Format

Put real images in one folder per class:

```text
data/real_waste/
  cardboard/
    img001.ppm
  glass/
    img002.ppm
  metal/
  organic/
  paper/
  plastic/
```

The dependency-free baseline currently reads binary or ASCII PPM images (`.ppm`). For real JPEG/PNG datasets, either:

- convert images to PPM before training, or
- upgrade `src/waste_classifier/image_io.py` to use Pillow/OpenCV.

## Run The Project

Generate the demo dataset:

```bash
python3 scripts/generate_demo_dataset.py
```

Train the model:

```bash
PYTHONPATH=src python3 -m waste_classifier.train --data data/demo_waste --model artifacts/waste_model.json
```

Predict one image:

```bash
PYTHONPATH=src python3 -m waste_classifier.predict data/demo_waste/plastic/plastic_001.ppm --model artifacts/waste_model.json
```

Start the production frontend:

```bash
npm install
npm run dev
```

Then open:

```text
http://localhost:3000
```

The frontend includes upload-based prediction, model readiness cards, and direct links to all recommended datasets.

Optional legacy Python-only web app:

```bash
PYTHONPATH=src python3 -m waste_classifier.app --model artifacts/waste_model.json --port 8000
```

Then open:

```text
http://localhost:8000
```

Run tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

## Deploy To Vercel

This project is ready for Vercel. The production frontend is a Next.js app and the `/api/predict` route runs entirely in Node.js, so it does not depend on spawning Python in serverless production.

```bash
npm run build
vercel deploy --prod --yes
```

If the Vercel CLI asks you to log in:

```bash
vercel login
```

## Suggested Review-1 PPT Content

- Problem statement: manual waste segregation is slow, inconsistent, and unsafe.
- Objective: classify waste into recyclable/material categories using image-based AI.
- Dataset: start with the generated demo dataset for prototype validation; use TrashNet or Garbage Classification V2 for final model training.
- Methodology: image acquisition, preprocessing, feature extraction/CNN, model training, evaluation, and prediction UI.
- Metrics: accuracy, precision, recall, F1-score, confusion matrix.
- Future enhancement: replace baseline with MobileNetV2/EfficientNet transfer learning and support real-time camera input.
