"""
DINOv2-Small embedding model with trainable projection head.

Architecture:
    DINOv2-ViT-S/14 (frozen early blocks) → projection MLP → L2-normalized embedding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv2Embedding(nn.Module):
    """
    DINOv2 backbone + projection head for metric learning.

    Args:
        backbone_name: DINOv2 model name (default: dinov2_vits14)
        embedding_dim: output embedding dimension
        freeze_blocks: number of transformer blocks to freeze (from the start)
        pretrained: whether to load pretrained weights
    """

    def __init__(
        self,
        backbone_name: str = "dinov2_vits14",
        embedding_dim: int = 128,
        freeze_blocks: int = 9,
        pretrained: bool = True,
    ):
        super().__init__()

        # Load DINOv2 backbone from torch hub
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            backbone_name,
            pretrained=pretrained,
        )

        # DINOv2-S/14 has hidden_dim=384, 12 blocks
        self.hidden_dim = self.backbone.embed_dim
        self.embedding_dim = embedding_dim

        # Freeze early transformer blocks
        self._freeze_blocks(freeze_blocks)

        # Projection head: maps DINOv2 features to embedding space
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, embedding_dim),
        )

    def _freeze_blocks(self, num_blocks: int):
        """Freeze patch embedding and first num_blocks transformer blocks."""
        # Freeze patch embedding
        for param in self.backbone.patch_embed.parameters():
            param.requires_grad = False

        # Freeze CLS token and position embedding
        if hasattr(self.backbone, "cls_token"):
            self.backbone.cls_token.requires_grad = False
        if hasattr(self.backbone, "pos_embed"):
            self.backbone.pos_embed.requires_grad = False

        # Freeze transformer blocks
        for i, block in enumerate(self.backbone.blocks):
            if i < num_blocks:
                for param in block.parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input images (B, 3, H, W)

        Returns:
            L2-normalized embeddings (B, embedding_dim)
        """
        # DINOv2 forward — get CLS token
        features = self.backbone(x)  # (B, hidden_dim)

        # Project to embedding space
        embeddings = self.projection(features)  # (B, embedding_dim)

        # L2 normalize
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings

    def get_trainable_params(self) -> list:
        """Return only trainable parameters (for optimizer)."""
        return [p for p in self.parameters() if p.requires_grad]

    def count_params(self) -> dict:
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "trainable_pct": 100.0 * trainable / total,
        }
