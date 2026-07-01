"""Dataset stage hierarchy definition and label mapping."""

from __future__ import annotations

# Stage 1 classes: 2 classes
STAGE1_CLASSES = ["biodegradable", "non_biodegradable"]

# Stage 2 classes: 6 classes
STAGE2_CLASSES = [
    "paper_cardboard",
    "organic_wood",
    "glass",
    "metal",
    "plastic_nylon",
    "textile_battery_ceramic",
]

# Stage 3 classes: 11 classes (our standard dataset classes)
STAGE3_CLASSES = [
    "battery",
    "cardboard",
    "ceramic",
    "glass",
    "metal",
    "nylon",
    "organic",
    "paper",
    "plastic",
    "textile",
    "wood",
]

# Stage 3 label name to Stage 1 index mapping
STAGE3_TO_STAGE1 = {
    "cardboard": 0,
    "organic": 0,
    "paper": 0,
    "wood": 0,
    "battery": 1,
    "ceramic": 1,
    "glass": 1,
    "metal": 1,
    "nylon": 1,
    "plastic": 1,
    "textile": 1,
}

# Stage 3 label name to Stage 2 index mapping
STAGE3_TO_STAGE2 = {
    "paper": 0,
    "cardboard": 0,
    "organic": 1,
    "wood": 1,
    "glass": 2,
    "metal": 3,
    "plastic": 4,
    "nylon": 4,
    "textile": 5,
    "battery": 5,
    "ceramic": 5,
}


def get_stage1_label(stage3_name: str) -> int:
    """Returns Stage 1 class index for a given Stage 3 class name."""
    if stage3_name not in STAGE3_TO_STAGE1:
        raise ValueError(f"Unknown class name: {stage3_name}")
    return STAGE3_TO_STAGE1[stage3_name]


def get_stage2_label(stage3_name: str) -> int:
    """Returns Stage 2 class index for a given Stage 3 class name."""
    if stage3_name not in STAGE3_TO_STAGE2:
        raise ValueError(f"Unknown class name: {stage3_name}")
    return STAGE3_TO_STAGE2[stage3_name]
