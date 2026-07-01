"""Targeted minority-class augmentations for handling class imbalance."""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class Cutout:
    """Apply a random cutout/erasing mask to a tensor image."""

    def __init__(self, size: int = 16, p: float = 0.5) -> None:
        self.size = size
        self.p = p

    def __call__(self, img_tensor: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img_tensor
        
        _, h, w = img_tensor.shape
        y = random.randint(0, h - self.size)
        x = random.randint(0, w - self.size)
        
        img_tensor[:, y : y + self.size, x : x + self.size] = 0.0
        return img_tensor


class TargetedAugmentedDataset(Dataset):
    """Dataset wrapper that duplicates and augments minority class samples on-the-fly."""

    def __init__(
        self,
        base_dataset: Dataset,
        base_factor: float = 2.0,
        max_copies: int = 8,
    ) -> None:
        """
        Args:
            base_dataset: The underlying PyTorch Dataset (must return (image, target)).
            base_factor: Base multiplier to compute inverse frequency duplication.
            max_copies: Maximum augmented copies allowed per minority sample.
        """
        self.base_dataset = base_dataset
        self.base_factor = base_factor
        self.max_copies = max_copies

        # Retrieve targets of all samples
        if hasattr(base_dataset, "targets") and base_dataset.targets is not None:
            self.targets = [int(t) for t in base_dataset.targets]
        else:
            self.targets = []
            for i in range(len(base_dataset)):
                _, target = base_dataset[i]
                self.targets.append(int(target))

        # Count frequencies
        self.class_counts = Counter(self.targets)
        self.max_count = max(self.class_counts.values()) if self.class_counts else 1

        # Build index mapping
        self.indices_map: list[tuple[int, bool]] = []  # (original_index, should_augment)

        for idx, target in enumerate(self.targets):
            # Original sample is always added
            self.indices_map.append((idx, False))

            # Duplication copies inversely proportional to class frequency
            class_count = self.class_counts[target]
            # If class_count is 0 or very small, duplication factor is higher
            ratio = self.max_count / max(class_count, 1)
            num_copies = int(self.base_factor * ratio) - 1
            num_copies = max(0, min(num_copies, self.max_copies))

            # Add duplicated indices flagged for augmentation
            for _ in range(num_copies):
                self.indices_map.append((idx, True))

        # Transformations for minority augmentations
        self.augment_transform = transforms.Compose([
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ])
        self.cutout = Cutout(size=16, p=0.5)

    def __len__(self) -> int:
        return len(self.indices_map)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        orig_idx, should_augment = self.indices_map[index]
        img, target = self.base_dataset[orig_idx]

        # Apply augmentation if flagged
        if should_augment:
            # Assumes img is already converted to PIL Image or Tensor
            if not isinstance(img, torch.Tensor):
                # If PIL image, convert to tensor after augmentations
                img = self.augment_transform(img)
                img_tensor = transforms.ToTensor()(img)
            else:
                # If already tensor
                img_tensor = self.augment_transform(img)
            
            img_tensor = self.cutout(img_tensor)
        else:
            if not isinstance(img, torch.Tensor):
                img_tensor = transforms.ToTensor()(img)
            else:
                img_tensor = img

        return img_tensor, target
