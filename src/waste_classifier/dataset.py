"""Dataset loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .features import extract_features


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label: str
    features: list[float]


def iter_image_paths(
    dataset_dir: str | Path,
    class_mapping: Optional[dict[str, str]] = None
) -> list[tuple[Path, str]]:
    """Iterate over image paths and labels, optionally mapping class names."""
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    items: list[tuple[Path, str]] = []
    for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        class_name = class_dir.name
        mapped_label = class_mapping.get(class_name, class_name) if class_mapping else class_name
        for image_path in sorted(class_dir.glob("*.ppm")):
            items.append((image_path, mapped_label))

    if not items:
        raise ValueError(f"No .ppm images found under {root}")
    return items


def load_dataset(
    dataset_dir: str | Path,
    class_mapping: Optional[dict[str, str]] = None
) -> list[Sample]:
    """Load dataset from a directory, optionally mapping class names."""
    return [
        Sample(image_path=image_path, label=label, features=extract_features(image_path))
        for image_path, label in iter_image_paths(dataset_dir, class_mapping)
    ]


def load_multiple_datasets(
    dataset_dirs: list[tuple[str | Path, Optional[dict[str, str]]]],
) -> list[Sample]:
    """Load and merge samples from multiple dataset directories.
    
    Args:
        dataset_dirs: List of (dataset_path, class_mapping) tuples.
                     class_mapping is optional and maps source class names to unified labels.
    
    Returns:
        Combined list of samples from all datasets.
    """
    all_samples: list[Sample] = []
    for dataset_dir, class_mapping in dataset_dirs:
        samples = load_dataset(dataset_dir, class_mapping)
        all_samples.extend(samples)
    return all_samples


def train_test_split(samples: list[Sample], test_ratio: float = 0.25) -> tuple[list[Sample], list[Sample]]:
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")
    by_label: dict[str, list[Sample]] = {}
    for sample in samples:
        by_label.setdefault(sample.label, []).append(sample)

    train: list[Sample] = []
    test: list[Sample] = []
    for label_samples in by_label.values():
        cutoff = max(1, int(len(label_samples) * test_ratio))
        test.extend(label_samples[:cutoff])
        train.extend(label_samples[cutoff:])

    if not train or not test:
        raise ValueError("Dataset is too small for a train/test split")
    return train, test
