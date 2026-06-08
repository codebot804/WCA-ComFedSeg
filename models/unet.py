"""Lightweight 2D U-Net for binary medical image segmentation.

Normalization is configurable: most methods use GroupNorm, while FedBN-style
methods use BatchNorm so that normalization statistics can remain local.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: str = "group") -> None:
        super().__init__()
        norm1 = _make_norm(norm, out_channels)
        norm2 = _make_norm(norm, out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm1,
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm2,
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def _make_norm(norm: str, channels: int) -> nn.Module:
    if norm == "group":
        return nn.GroupNorm(num_groups=1, num_channels=channels)
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    raise ValueError("norm must be 'group' or 'batch'.")


class UNet2D(nn.Module):
    """Small CPU-friendly U-Net for binary segmentation logits."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 8,
        norm: str = "group",
    ) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels, norm=norm)
        self.enc2 = ConvBlock(base_channels, base_channels * 2, norm=norm)
        self.bottleneck = ConvBlock(base_channels * 2, base_channels * 4, norm=norm)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2, norm=norm)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels, norm=norm)
        self.head = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        nn.init.constant_(self.head.bias, 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, kernel_size=2))
        b = self.bottleneck(F.max_pool2d(e2, kernel_size=2))

        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.head(d1)


def build_unet2d(base_channels: int = 8, norm: str = "group") -> UNet2D:
    return UNet2D(in_channels=1, out_channels=1, base_channels=base_channels, norm=norm)
