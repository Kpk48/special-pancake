"""Stage 2 Model: 6-class classification, conditioned on Stage 1 predicted class."""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import DSConv2DBackbone


class Stage2Model(nn.Module):
    """Stage 2 Classifier Model."""

    def __init__(self, feature_dim: int = 128, embedding_dim: int = 16) -> None:
        super().__init__()
        self.backbone = DSConv2DBackbone(feature_dim=feature_dim)
        
        # Stage 1 conditioning embedding: 2 classes
        self.stage1_embedding = nn.Embedding(
            num_embeddings=2,
            embedding_dim=embedding_dim,
        )
        
        # Combined classifier head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim + embedding_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 6),
        )

    def forward(self, x: torch.Tensor, stage1_class: torch.Tensor) -> torch.Tensor:
        # Extract visual features: (B, feature_dim)
        features = self.backbone(x)
        
        # Get conditioning embedding: (B, embedding_dim)
        cond_emb = self.stage1_embedding(stage1_class)
        
        # Concatenate features and conditioning
        combined = torch.cat([features, cond_emb], dim=-1)
        
        return self.classifier(combined)
