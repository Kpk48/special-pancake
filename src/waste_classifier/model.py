"""A small k-nearest-neighbors model for image classification."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from math import sqrt
from pathlib import Path


@dataclass
class WasteKNNClassifier:
    labels: list[str]
    vectors: list[list[float]]
    k: int = 3

    def predict(self, features: list[float]) -> str:
        if not self.vectors:
            raise ValueError("Model has not been trained")

        distances = [
            (self._distance(features, vector), label)
            for vector, label in zip(self.vectors, self.labels)
        ]
        neighbors = sorted(distances, key=lambda item: item[0])[: self.k]
        counts = Counter(label for _, label in neighbors)
        return counts.most_common(1)[0][0]

    def predict_proba(self, features: list[float]) -> dict[str, float]:
        distances = [
            (self._distance(features, vector), label)
            for vector, label in zip(self.vectors, self.labels)
        ]
        neighbors = sorted(distances, key=lambda item: item[0])[: self.k]
        counts = Counter(label for _, label in neighbors)
        total = sum(counts.values())
        return {label: count / total for label, count in sorted(counts.items())}

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps({"k": self.k, "labels": self.labels, "vectors": self.vectors}, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _distance(left: list[float], right: list[float]) -> float:
        return sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def train_classifier(features: list[list[float]], labels: list[str], k: int = 3) -> WasteKNNClassifier:
    if len(features) != len(labels):
        raise ValueError("Feature and label counts do not match")
    if not features:
        raise ValueError("Cannot train on an empty dataset")
    return WasteKNNClassifier(labels=labels, vectors=features, k=k)


def load_model(path: str | Path) -> WasteKNNClassifier:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return WasteKNNClassifier(labels=data["labels"], vectors=data["vectors"], k=data["k"])
