"""Evaluation metrics for classification."""

from __future__ import annotations

from collections import Counter


def evaluate(expected: list[str], predicted: list[str]) -> dict[str, object]:
    if len(expected) != len(predicted):
        raise ValueError("Expected and predicted label counts do not match")
    labels = sorted(set(expected) | set(predicted))
    correct = sum(1 for actual, guess in zip(expected, predicted) if actual == guess)
    total = len(expected)

    per_class: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = sum(1 for actual, guess in zip(expected, predicted) if actual == label and guess == label)
        fp = sum(1 for actual, guess in zip(expected, predicted) if actual != label and guess == label)
        fn = sum(1 for actual, guess in zip(expected, predicted) if actual == label and guess != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}

    confusion = Counter((actual, guess) for actual, guess in zip(expected, predicted))
    return {
        "accuracy": correct / total if total else 0.0,
        "total": total,
        "per_class": per_class,
        "confusion": {f"{actual}->{guess}": count for (actual, guess), count in sorted(confusion.items())},
    }
