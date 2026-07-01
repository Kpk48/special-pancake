"""Stage 3 Model: 11-class classification, conditioned on Stage 2 predicted class, with optional metric-learning prototypical head."""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import DSConv2DBackbone


class PrototypicalHead(nn.Module):
    """Prototypical Network Style Classifier Head.
    
    Computes negative squared Euclidean distance to learnable class prototype vectors.
    """

    def __init__(self, in_features: int, num_classes: int, embedding_dim: int = 64) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.prototypes = nn.Parameter(torch.randn(num_classes, embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project combined features to prototype embedding space
        embeddings = self.projection(x)  # Shape: (B, embedding_dim)
        
        # Compute squared Euclidean distance to class prototypes
        # (B, 1, embedding_dim) - (1, num_classes, embedding_dim) -> (B, num_classes, embedding_dim)
        diff = embeddings.unsqueeze(1) - self.prototypes.unsqueeze(0)
        distances = torch.sum(diff ** 2, dim=-1)  # Shape: (B, num_classes)
        
        # Return negative distance as logits (closer prototype = higher logit)
        return -distances


class Stage3Model(nn.Module):
    """Stage 3 Classifier Model."""

    def __init__(
        self,
        feature_dim: int = 128,
        embedding_dim: int = 16,
        use_prototypical: bool = False,
        proto_dim: int = 64,
    ) -> None:
        super().__init__()
        self.backbone = DSConv2DBackbone(feature_dim=feature_dim)
        self.use_prototypical = use_prototypical

        # Stage 2 conditioning embedding: 6 classes
        self.stage2_embedding = nn.Embedding(
            num_embeddings=6,
            embedding_dim=embedding_dim,
        )

        in_classifier_features = feature_dim + embedding_dim

        if use_prototypical:
            self.classifier = PrototypicalHead(
                in_features=in_classifier_features,
                num_classes=11,
                embedding_dim=proto_dim,
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(in_classifier_features, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 11),
            )

    def forward(self, x: torch.Tensor, stage2_class: torch.Tensor) -> torch.Tensor:
        # Extract visual features: (B, feature_dim)
        features = self.backbone(x)

        # Get conditioning embedding: (B, embedding_dim)
        cond_emb = self.stage2_embedding(stage2_class)

        # Concatenate features and conditioning
        combined = torch.cat([features, cond_emb], dim=-1)

        return self.classifier(combined)
