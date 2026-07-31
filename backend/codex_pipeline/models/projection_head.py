"""Shared projection head and device selection for training and evaluation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional


class ProjectionHead(nn.Module):
    """Map frozen DINO features into the runtime's normalized embedding space."""

    def __init__(self, input_dim: int = 384, embedding_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim, embedding_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return functional.normalize(self.net(features), p=2, dim=-1)


def get_device(device_cfg: str) -> torch.device:
    """Resolve an explicit device or the first supported accelerator."""

    if device_cfg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_cfg)
