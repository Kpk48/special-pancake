"""Stage 1 Model: 2-class classification (biodegradable vs non-biodegradable)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import DSConv2DBackbone


class Stage1Model(nn.Module):
    """Stage 1 Classifier Model."""

    def __init__(self, feature_dim: int = 128) -> None:
        super().__init__()
        self.backbone = DSConv2DBackbone(feature_dim=feature_dim)
        self.classifier = nn.Linear(feature_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)
