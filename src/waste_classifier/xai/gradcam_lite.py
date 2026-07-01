"""Lightweight Grad-CAM implementation for visual model explainability."""

from __future__ import annotations

import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAMLite:
    """Lightweight Grad-CAM execution targeting the last Conv2d layer of the backbone."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.device = next(model.parameters()).device
        
        # Locate the target convolutional layer
        # In our backbone: backbone.dsconv4.pointwise
        self.target_layer = self.model.backbone.dsconv4.pointwise
        
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.handlers = []

    def _save_activations(self, module: nn.Module, input: Any, output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _save_gradients(self, module: nn.Module, grad_input: Any, grad_output: torch.Tensor) -> None:
        self.gradients = grad_output[0].detach()

    def register_hooks(self) -> None:
        """Registers forward and backward hooks to capture intermediate variables."""
        self.activations = None
        self.gradients = None
        self.handlers = [
            self.target_layer.register_forward_hook(self._save_activations),
            self.target_layer.register_full_backward_hook(self._save_gradients),
        ]

    def remove_hooks(self) -> None:
        """Cleans up registered hook handlers."""
        for handler in self.handlers:
            handler.remove()
        self.handlers = []

    def generate_heatmap(
        self,
        image_tensor: torch.Tensor,
        stage2_class: torch.Tensor,
        target_class: int | None = None,
    ) -> tuple[torch.Tensor, int, float]:
        """Generates a Grad-CAM heatmap for a single image tensor.
        
        Returns:
            heatmap: 2D Tensor normalized to [0, 1].
            pred_class: Predicted class index.
            latency_ms: Execution time in milliseconds.
        """
        start_time = time.perf_counter()
        
        self.model.eval()
        self.register_hooks()

        # Prepare inputs (add batch dimension if needed)
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        if stage2_class.dim() == 0:
            stage2_class = stage2_class.unsqueeze(0)

        image_tensor = image_tensor.to(self.device)
        stage2_class = stage2_class.to(self.device)

        # Require gradients on the inputs to enable backward pass
        self.model.zero_grad()
        
        # Forward pass
        logits = self.model(image_tensor, stage2_class)
        pred_class = logits.argmax(dim=-1).item()

        if target_class is None:
            target_class = pred_class

        # Backward pass on target class score
        score = logits[0, target_class]
        score.backward()

        self.remove_hooks()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Gradients or activations were not captured. Make sure backward pass is computed.")

        # Grad-CAM computation
        # gradients shape: [1, C, H, W], activations shape: [1, C, H, W]
        # Pool gradients over spatial dimensions (H, W) to compute channel weights
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # Shape: [1, C, 1, 1]
        
        # Weighted combination of activations
        cam = torch.sum(weights * self.activations, dim=1).squeeze(0)  # Shape: [H, W]
        
        # Apply ReLU to keep only positive contributions
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        min_val, max_val = cam.min(), cam.max()
        if max_val > min_val:
            cam = (cam - min_val) / (max_val - min_val)
        else:
            cam = torch.zeros_like(cam)

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return cam.cpu(), pred_class, latency_ms
