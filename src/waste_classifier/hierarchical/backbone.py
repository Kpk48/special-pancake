"""DSConv2D Backbone for Hierarchical Waste Classification."""

from __future__ import annotations

import torch
import torch.nn as nn


class DSConvBlock(nn.Module):
    """Depthwise-Separable Convolutional Block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        # Depthwise conv
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        # Pointwise conv
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)
        return self.relu(out)


class DSConv2DBackbone(nn.Module):
    """CNN backbone featuring parallel branches followed by 4 sequential DSConv blocks."""

    def __init__(self, feature_dim: int = 128) -> None:
        super().__init__()

        # 5 Parallel Branches with different kernel sizes
        self.branch11x11 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=11, padding=5, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.branch9x9 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.branch7x7 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Concatenated features have 32 * 5 = 160 channels
        # Downsample with MaxPool2D
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 4 Sequential DSConv Blocks with 128, 64, 32, 16 filters
        self.dsconv1 = DSConvBlock(160, 128)
        self.dsconv2 = DSConvBlock(128, 64)
        self.dsconv3 = DSConvBlock(64, 32)
        self.dsconv4 = DSConvBlock(32, 16)

        # Final projection head
        # Input images are 128x128. After 5 max pools, resolution is 128 / 32 = 4x4.
        self.flatten_dim = 16 * 4 * 4  # 256
        self.fc = nn.Linear(self.flatten_dim, feature_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass through parallel branches
        b11 = self.branch11x11(x)
        b9 = self.branch9x9(x)
        b7 = self.branch7x7(x)
        b5 = self.branch5x5(x)
        b3 = self.branch3x3(x)

        # Concatenate branches along channel dimension
        out = torch.cat([b11, b9, b7, b5, b3], dim=1)  # Shape: (B, 160, H, W)
        out = self.pool(out)  # Downsample

        # Sequential DSConv Blocks with pooling
        out = self.pool(self.dsconv1(out))
        out = self.pool(self.dsconv2(out))
        out = self.pool(self.dsconv3(out))
        out = self.pool(self.dsconv4(out))

        # Flatten and project to embedding space
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)
        return self.relu(out)
