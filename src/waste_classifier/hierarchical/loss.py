"""Class-weighted Multi-class Focal Loss in PyTorch."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class Focal Loss."""

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        """
        Args:
            alpha: Class weights tensor of shape [num_classes] or None.
            gamma: Focusing parameter.
            reduction: Reduction type ('mean', 'sum', 'none').
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(inputs, dim=-1)
        pt = torch.exp(log_p).gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        log_pt = log_p.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        loss = -((1.0 - pt) ** self.gamma) * log_pt

        if self.alpha is not None:
            # Gather class weights for each target sample
            alpha_t = self.alpha.to(targets.device).gather(dim=-1, index=targets)
            loss = loss * alpha_t

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
