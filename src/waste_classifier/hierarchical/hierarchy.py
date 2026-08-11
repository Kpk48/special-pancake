"""Dataset stage hierarchy definition and label mapping."""

from __future__ import annotations

import os
import csv
from pathlib import Path

STAGE1_CLASSES = ["biodegradable", "non_biodegradable"]

STAGE2_CLASSES = [
    "paper_cardboard",
    "organic",
    "glass",
    "metal",
    "plastic",
    "textile_battery",
]

STAGE3_CLASSES = [
    "battery",
    "cardboard",
    "glass",
    "metal",
    "organic",
    "paper",
    "plastic",
    "textile",
]

STAGE3_TO_STAGE1 = {
    "cardboard": 0,
    "organic": 0,
    "paper": 0,
    "battery": 1,
    "glass": 1,
    "metal": 1,
    "plastic": 1,
    "textile": 1,
}

STAGE3_TO_STAGE2 = {
    "paper": 0,
    "cardboard": 0,
    "organic": 1,
    "glass": 2,
    "metal": 3,
    "plastic": 4,
    "textile": 5,
    "battery": 5,
}

def _normalise_path(p: str) -> str:
    p = p.replace("\\", "/")
    idx = p.find("data/final")
    if idx != -1:
        return p[idx:]
    return p

_OVERRIDES = {}
_overrides_file = Path("data/final/stage1_label_overrides.csv")
if _overrides_file.exists():
    with open(_overrides_file, mode="r", encoding="utf-8") as _f:
        _reader = csv.DictReader(_f)
        for _row in _reader:
            _OVERRIDES[_normalise_path(_row["file_path"])] = int(_row["label_idx"])

def get_stage1_label(stage3_name: str, filepath: str | None = None) -> int:
    """Returns Stage 1 class index for a given Stage 3 class name."""
    if filepath is not None:
        norm_p = _normalise_path(filepath)
        if norm_p in _OVERRIDES:
            return _OVERRIDES[norm_p]
            
    if stage3_name not in STAGE3_TO_STAGE1:
        raise ValueError(f"Unknown class name: {stage3_name}")
    return STAGE3_TO_STAGE1[stage3_name]

def get_stage2_label(stage3_name: str) -> int:
    """Returns Stage 2 class index for a given Stage 3 class name."""
    if stage3_name not in STAGE3_TO_STAGE2:
        raise ValueError(f"Unknown class name: {stage3_name}")
    return STAGE3_TO_STAGE2[stage3_name]
