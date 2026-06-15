# Dataset Notes

The project currently uses `data/demo_waste/`, a generated six-class PPM image dataset. It was created with:

```bash
python3 scripts/generate_demo_dataset.py
```

This demo dataset is for project verification, review presentation, and pipeline testing. For final accuracy claims, use real waste photographs from TrashNet, Kaggle Garbage Classification, Garbage Dataset V2, or TACO depending on the scope.

Expected structure:

```text
data/<dataset_name>/
  cardboard/
  glass/
  metal/
  organic/
  paper/
  plastic/
```

The current baseline reads `.ppm` files. Convert real dataset images to PPM or add Pillow/OpenCV support for JPEG/PNG images.
