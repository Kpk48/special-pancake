"""Multi-scale DSConv2D and MobileNetV3 backbones for Hierarchical Waste Classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models


class DSConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
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
    def __init__(self, feature_dim: int = 128) -> None:
        super().__init__()
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
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dsconv1 = DSConvBlock(160, 128)
        self.dsconv2 = DSConvBlock(128, 64)
        self.dsconv3 = DSConvBlock(64, 32)
        self.dsconv4 = DSConvBlock(32, 16)
        self.flatten_dim = 16 * 4 * 4
        self.fc = nn.Linear(self.flatten_dim, feature_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b11 = self.branch11x11(x)
        b9 = self.branch9x9(x)
        b7 = self.branch7x7(x)
        b5 = self.branch5x5(x)
        b3 = self.branch3x3(x)
        out = torch.cat([b11, b9, b7, b5, b3], dim=1)
        out = self.pool(out)
        out = self.pool(self.dsconv1(out))
        out = self.pool(self.dsconv2(out))
        out = self.pool(self.dsconv3(out))
        out = self.pool(self.dsconv4(out))
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)
        return self.relu(out)


class PlainConv2DBackbone(nn.Module):
    def __init__(self, feature_dim: int = 128) -> None:
        super().__init__()
        self.plain_conv = nn.Sequential(
            nn.Conv2d(3, 160, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dsconv1 = DSConvBlock(160, 128)
        self.dsconv2 = DSConvBlock(128, 64)
        self.dsconv3 = DSConvBlock(64, 32)
        self.dsconv4 = DSConvBlock(32, 16)
        self.flatten_dim = 16 * 4 * 4
        self.fc = nn.Linear(self.flatten_dim, feature_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.plain_conv(x)
        out = self.pool(out)
        out = self.pool(self.dsconv1(out))
        out = self.pool(self.dsconv2(out))
        out = self.pool(self.dsconv3(out))
        out = self.pool(self.dsconv4(out))
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)
        return self.relu(out)


class MobileNetV3Backbone(nn.Module):
    FREEZE_UP_TO_IDX = 5

    def __init__(self, feature_dim: int = 128) -> None:
        super().__init__()
        base = tv_models.mobilenet_v3_small(weights=tv_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.avgpool = base.avgpool

        for idx, block in enumerate(self.features):
            if idx <= self.FREEZE_UP_TO_IDX:
                for param in block.parameters():
                    param.requires_grad = False

        self.proj = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(576, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.proj(out)
        out = nn.functional.normalize(out, p=2, dim=1)
        return out

