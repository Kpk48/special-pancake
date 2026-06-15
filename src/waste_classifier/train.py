"""Train the waste classifier."""

from __future__ import annotations

import argparse
import json

from .dataset import load_dataset, load_multiple_datasets, train_test_split
from .metrics import evaluate
from .model import train_classifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an AI-based waste classifier.")
    parser.add_argument("--data", default="data/expanded_waste", help="Dataset directory with one folder per class.")
    parser.add_argument("--model", default="artifacts/waste_model.json", help="Output model JSON path.")
    parser.add_argument("--k", type=int, default=3, help="Number of nearest neighbors.")
    args = parser.parse_args()

    samples = load_dataset(args.data)
    train_samples, test_samples = train_test_split(samples)
    model = train_classifier(
        [sample.features for sample in train_samples],
        [sample.label for sample in train_samples],
        k=args.k,
    )
    predictions = [model.predict(sample.features) for sample in test_samples]
    report = evaluate([sample.label for sample in test_samples], predictions)
    model.save(args.model)

    print(f"Trained on {len(train_samples)} images and tested on {len(test_samples)} images.")
    print(f"Saved model to {args.model}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
