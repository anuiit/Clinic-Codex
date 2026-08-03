"""
Inference Engine for the Codex pipeline.

Loads a trained prototypical network checkpoint (prototypes + projection head
weights) and a frozen DINOv2-ViT-S/14 backbone, then classifies single crops
or batches of crops by cosine similarity to the stored class prototypes.

Typical usage::

    engine = InferenceEngine(prototype_path="./prototypes/prototypes.pt")
    result = engine.classify_crop(image_np)   # image_np: H×W×3 uint8
    print(result.class_name, result.similarity)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Projection head — architecture must exactly match training
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """
    Two-layer MLP projection head with GELU activation and dropout.

    Maps DINOv2 CLS-token features (384-d for ViT-S/14) to a lower-dimensional
    L2-normalised embedding space (default 128-d).

    Args:
        input_dim:     dimensionality of the backbone CLS token (384 for ViT-S/14).
        embedding_dim: output embedding dimensionality.
    """

    def __init__(self, input_dim: int = 384, embedding_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: backbone features (B, input_dim).

        Returns:
            L2-normalised embeddings (B, embedding_dim).
        """
        return F.normalize(self.net(x), p=2, dim=-1)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    """
    Output of a single-image classification call.

    Attributes:
        class_name:  human-readable class label of the best match.
        class_label: integer label index of the best match.
        similarity:  cosine similarity score of the best match in [−1, 1].
        rejected:    True when ``similarity < rejection_threshold`` — the crop
                     is considered an unknown / out-of-distribution sample.
        top_k:       list of up to *k* dicts, each with keys
                     ``class_label``, ``class_name``, ``similarity``.
                     Sorted in descending order by similarity.
    """

    class_name: str
    class_label: int
    similarity: float
    rejected: bool
    top_k: List[Dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class InferenceEngine:
    """
    End-to-end inference engine for codex element classification.

    The engine:
    1. Loads prototypes and projection head weights from a ``.pt`` checkpoint.
    2. Loads a frozen DINOv2-ViT-S/14 backbone from ``torch.hub``.
    3. Preprocesses images with the same transform used during validation.
    4. Embeds images through backbone + projection head.
    5. Classifies by cosine similarity to the stored prototypes, applying an
       optional rejection threshold for out-of-distribution samples.

    Args:
        prototype_path:    path to the exported ``prototypes.pt`` artifact.
        backbone_name:     torch.hub model name; must match training backbone.
        device:            ``"auto"`` selects CUDA → MPS → CPU automatically;
                           otherwise pass ``"cuda"``, ``"mps"``, or ``"cpu"``.
        image_size:        spatial resolution expected by the backbone (224).
        rejection_threshold: cosine similarity below which a prediction is
                             marked as *rejected* (out of distribution).
        top_k:             number of nearest-prototype candidates to include
                           in :attr:`ClassificationResult.top_k`.
    """

    def __init__(
        self,
        prototype_path: str = "./prototypes/prototypes.pt",
        backbone_name: str = "dinov2_vits14",
        device: str = "auto",
        image_size: int = 224,
        rejection_threshold: float = 0.35,
        top_k: int = 3,
    ) -> None:
        self.image_size = image_size
        self.rejection_threshold = rejection_threshold
        self.top_k = top_k

        # ---- device selection ----
        self.device = self._resolve_device(device)
        logger.info("InferenceEngine using device: %s", self.device)

        # ---- load prototype artifact ----
        checkpoint = self._load_checkpoint(prototype_path)
        self.prototypes: torch.Tensor = checkpoint["prototypes"].to(self.device)
        self.class_names: Dict[int, str] = checkpoint["class_names"]
        self.class_labels: torch.Tensor = checkpoint["class_labels"].to(self.device)
        embedding_dim: int = int(checkpoint["embedding_dim"])
        hidden_dim: int = int(checkpoint["hidden_dim"])

        logger.info(
            "Loaded %d prototypes  (embedding_dim=%d, hidden_dim=%d)",
            self.prototypes.size(0),
            embedding_dim,
            hidden_dim,
        )

        # ---- projection head ----
        self.projection_head = ProjectionHead(
            input_dim=hidden_dim,
            embedding_dim=embedding_dim,
        ).to(self.device)
        self.projection_head.load_state_dict(checkpoint["model_state_dict"])
        self.projection_head.eval()

        # ---- DINOv2 backbone (frozen, eval) ----
        logger.info("Loading DINOv2 backbone: %s", backbone_name)
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            backbone_name,
            pretrained=True,
        ).to(self.device)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        # ---- preprocessing transform (matches validation transform) ----
        self._transform = A.Compose(
            [
                A.LongestMaxSize(max_size=image_size),
                A.PadIfNeeded(
                    min_height=image_size,
                    min_width=image_size,
                    border_mode=0,          # cv2.BORDER_CONSTANT
                    value=(255, 255, 255),  # white padding
                ),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                ToTensorV2(),
            ]
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Apply the validation preprocessing pipeline to a single crop.

        The transform:
        * resizes the longest edge to :attr:`image_size` (preserving aspect ratio),
        * pads the shorter edge with white pixels to reach ``image_size × image_size``,
        * normalises with ImageNet mean/std,
        * converts to a float32 tensor.

        Args:
            image: uint8 NumPy array with shape (H, W, 3) in RGB order.

        Returns:
            Float tensor of shape (1, 3, ``image_size``, ``image_size``).
        """
        augmented = self._transform(image=image)
        tensor: torch.Tensor = augmented["image"]          # (3, H, W) float32
        return tensor.unsqueeze(0).to(self.device)         # (1, 3, H, W)

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """
        Pass a batch of preprocessed images through backbone + projection head.

        Args:
            images: float tensor of shape (B, 3, H, W), already on the
                    correct device.

        Returns:
            L2-normalised embeddings of shape (B, embedding_dim).
        """
        with torch.no_grad():
            features = self.backbone(images)           # (B, hidden_dim)
            embeddings = self.projection_head(features)  # (B, embedding_dim)
        return embeddings

    def classify_crop(self, image: np.ndarray) -> ClassificationResult:
        """
        Classify a single crop image.

        Args:
            image: uint8 NumPy array of shape (H, W, 3) in RGB order.

        Returns:
            :class:`ClassificationResult` for the crop.
        """
        tensor = self.preprocess(image)          # (1, 3, H, W)
        embedding = self.embed(tensor)           # (1, 128)
        return self._embedding_to_result(embedding[0])

    def classify_batch(self, images: List[np.ndarray]) -> List[ClassificationResult]:
        """
        Classify a list of crop images in a single forward pass.

        All images are preprocessed independently (each is padded/resized to
        the same square resolution) and stacked into a single batch.

        Args:
            images: list of uint8 NumPy arrays, each of shape (H, W, 3).

        Returns:
            List of :class:`ClassificationResult`, one per input image,
            in the same order as ``images``.
        """
        if not images:
            return []

        # Stack all preprocessed tensors into one batch
        tensors = [self.preprocess(img) for img in images]   # list of (1,3,H,W)
        batch = torch.cat(tensors, dim=0)                     # (B, 3, H, W)
        embeddings = self.embed(batch)                        # (B, 128)

        return [self._embedding_to_result(embeddings[i]) for i in range(embeddings.size(0))]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _embedding_to_result(self, embedding: torch.Tensor) -> ClassificationResult:
        """
        Convert a single (embedding_dim,) tensor into a ClassificationResult.

        Computes cosine similarity against all stored prototypes (both the
        embedding and prototypes are already L2-normalised, so this reduces
        to a dot product), takes the top-k, and applies the rejection
        threshold to the best match.

        Args:
            embedding: (embedding_dim,) float tensor on :attr:`device`.

        Returns:
            :class:`ClassificationResult`.
        """
        # similarities: (num_classes,)
        similarities = torch.mv(self.prototypes, embedding)

        k = min(self.top_k, similarities.size(0))
        top_values, top_indices = similarities.topk(k)

        top_k_list: List[Dict] = []
        for sim_val, class_idx in zip(top_values.tolist(), top_indices.tolist()):
            top_k_list.append(
                {
                    "class_label": class_idx,
                    "class_name": self.class_names.get(class_idx, str(class_idx)),
                    "similarity": float(sim_val),
                }
            )

        best = top_k_list[0]
        return ClassificationResult(
            class_name=best["class_name"],
            class_label=best["class_label"],
            similarity=best["similarity"],
            rejected=best["similarity"] < self.rejection_threshold,
            top_k=top_k_list,
        )

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """
        Resolve the device string to a :class:`torch.device`.

        ``"auto"`` selects CUDA if available, then MPS (Apple Silicon),
        then CPU.

        Args:
            device: ``"auto"``, ``"cuda"``, ``"mps"``, or ``"cpu"``.

        Returns:
            Resolved :class:`torch.device`.
        """
        if device != "auto":
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _load_checkpoint(path: str) -> dict:
        """
        Load the ``prototypes.pt`` artifact safely.

        Args:
            path: filesystem path to the ``.pt`` file.

        Returns:
            Dictionary with keys: ``prototypes``, ``class_names``,
            ``class_labels``, ``embedding_dim``, ``hidden_dim``,
            ``model_state_dict``.

        Raises:
            FileNotFoundError: if *path* does not exist.
            KeyError: if expected keys are missing from the checkpoint.
        """
        import os

        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Prototype checkpoint not found at '{path}'. "
                "Run the export script first to generate prototypes.pt."
            )

        checkpoint: dict = torch.load(path, map_location="cpu", weights_only=False)

        required_keys = {
            "prototypes",
            "class_names",
            "class_labels",
            "embedding_dim",
            "hidden_dim",
            "model_state_dict",
        }
        missing = required_keys - checkpoint.keys()
        if missing:
            raise KeyError(
                f"Prototype checkpoint is missing required keys: {missing}. "
                "Re-export using the current export_prototypes script."
            )

        return checkpoint
