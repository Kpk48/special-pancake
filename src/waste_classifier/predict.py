"""Predict the waste class of a PPM image."""

from __future__ import annotations

import argparse
import json

from .features import extract_features
from .model import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify one waste image.")
    parser.add_argument("image", help="Path to a .ppm image.")
    parser.add_argument("--model", default="artifacts/waste_model.json", help="Model JSON path.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    model = load_model(args.model)
    features = extract_features(args.image)
    label = model.predict(features)
    probabilities = model.predict_proba(features)

    if args.json:
        print(json.dumps({"label": label, "probabilities": probabilities}))
        return

    print(f"Predicted class: {label}")
    print("Neighbor confidence:")
    for class_name, confidence in probabilities.items():
        print(f"  {class_name}: {confidence:.2f}")


if __name__ == "__main__":
    main()
