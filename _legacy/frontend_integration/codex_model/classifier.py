"""
CodexClassifier — standalone inference module.

Self-contained: does NOT depend on codex_pipeline. Copy this directory
(codex_model/) into any project and install the requirements.txt.

Usage:
    from codex_model import CodexClassifier

    clf = CodexClassifier()                  # auto-detects weights/ next to this file
    result = clf.classify(pil_image_or_array)
    # {'class_name': 'cacahuatl', 'confidence': 0.87, 'top_k': [...]}
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------------
# Projection head (identical to training architecture)
# ---------------------------------------------------------------------------

class _ProjectionHead(nn.Module):
    """Two-layer MLP that projects DINOv2 CLS token to embedding space."""

    def __init__(self, hidden_dim: int = 384, embedding_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=-1)


# ---------------------------------------------------------------------------
# Image preprocessing (must match training exactly)
# ---------------------------------------------------------------------------

def _preprocess_image(
    image: Union[Image.Image, np.ndarray],
    image_size: int = 224,
) -> torch.Tensor:
    """
    Resize-with-padding + ImageNet normalisation.

    Returns a (1, 3, image_size, image_size) float32 tensor ready for the
    DINOv2 backbone.
    """
    # Convert to PIL if needed
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = Image.fromarray(image).convert("RGB")
        elif image.shape[2] == 4:
            image = Image.fromarray(image[:, :, :3])
        else:
            image = Image.fromarray(image)
    image = image.convert("RGB")

    # Resize preserving aspect ratio, then pad to square
    w, h = image.size
    scale = image_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    image = image.resize((new_w, new_h), Image.LANCZOS)

    padded = Image.new("RGB", (image_size, image_size), (128, 128, 128))
    pad_x = (image_size - new_w) // 2
    pad_y = (image_size - new_h) // 2
    padded.paste(image, (pad_x, pad_y))

    # To tensor and normalise
    arr = np.array(padded, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    return tensor


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

class CodexClassifier:
    """
    Codex manuscript element classifier.

    Loads DINOv2 + projection head + prototype store and classifies images
    using nearest-prototype cosine similarity.

    Args:
        model_dir: directory that contains prototypes.pt and projection.pt.
                   Defaults to the ``weights/`` folder next to this file.
        device: torch device string (e.g. 'cpu', 'cuda', 'mps').
                Auto-detected when None.
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        device: Optional[str] = None,
    ):
        # Resolve weights directory
        if model_dir is None:
            model_dir = str(Path(__file__).parent / "weights")
        self._weights_dir = Path(model_dir)

        # Load config
        config_path = Path(__file__).parent / "config.json"
        with open(config_path, "r") as f:
            self.config = json.load(f)

        # Device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        self._load_weights()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_weights(self):
        """Load prototypes, projection head, and DINOv2 backbone."""
        proto_path = self._weights_dir / "prototypes.pt"
        proj_path = self._weights_dir / "projection.pt"

        if not proto_path.exists():
            raise FileNotFoundError(
                f"Prototypes file not found: {proto_path}\n"
                "Run `python -m codex_pipeline.scripts.export_model` first."
            )
        if not proj_path.exists():
            raise FileNotFoundError(
                f"Projection weights not found: {proj_path}\n"
                "Run `python -m codex_pipeline.scripts.export_model` first."
            )

        # --- Prototypes ---
        proto_data = torch.load(proto_path, map_location="cpu", weights_only=False)
        self._prototypes: torch.Tensor = proto_data["prototypes"].to(self.device)
        raw_class_names: dict = proto_data["class_names"]  # {int_label: name}
        # Build list ordered by sorted label integers.
        # Labels can be sparse (e.g. 0,1,3,5,...) so we must not assume 0..N-1.
        sorted_labels = sorted(raw_class_names.keys())
        self._label_index: List[int] = sorted_labels        # label → list position
        self._class_names: List[str] = [raw_class_names[lbl] for lbl in sorted_labels]

        # --- Projection head ---
        embedding_dim: int = self.config.get("embedding_dim", 128)
        hidden_dim: int = self.config.get("hidden_dim", 384)
        self._projection = _ProjectionHead(hidden_dim, embedding_dim).to(self.device)
        proj_state = torch.load(proj_path, map_location="cpu", weights_only=False)
        self._projection.load_state_dict(proj_state)
        self._projection.eval()

        # --- DINOv2 backbone (frozen, from torch.hub) ---
        backbone_name: str = self.config.get("backbone", "dinov2_vits14")
        self._backbone = torch.hub.load(
            "facebookresearch/dinov2",
            backbone_name,
            pretrained=True,
        ).to(self.device)
        self._backbone.eval()

        self._image_size: int = self.config.get("image_size", 224)
        self._rejection_threshold: float = self.config.get("rejection_threshold", 0.35)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def classify(
        self,
        image: Union[Image.Image, np.ndarray],
        top_k: int = 3,
    ) -> dict:
        """
        Classify a single image.

        Args:
            image: PIL Image or numpy array (H, W, 3) uint8.
            top_k: number of top predictions to return.

        Returns:
            {
                "class_name": str,        # top-1 prediction name
                "class_label": int,       # top-1 integer label
                "confidence": float,      # cosine similarity in [-1, 1]
                "rejected": bool,         # True if below rejection threshold
                "top_k": [                # ranked list of length top_k
                    {"class_name": str, "class_label": int,
                     "confidence": float, "rejected": bool},
                    ...
                ],
            }
        """
        tensor = _preprocess_image(image, self._image_size).to(self.device)

        # DINOv2 CLS token
        features = self._backbone(tensor)  # (1, hidden_dim)

        # Project + L2-normalise
        embedding = self._projection(features)  # (1, embedding_dim)

        # Cosine similarity against all prototypes
        similarities = torch.mm(embedding, self._prototypes.t()).squeeze(0)  # (N_classes,)

        k = min(top_k, len(self._class_names))
        top_values, top_indices = similarities.topk(k)

        top_list = []
        for sim_val, class_idx in zip(top_values.tolist(), top_indices.tolist()):
            top_list.append({
                "class_name": self._class_names[class_idx],
                "class_label": class_idx,
                "confidence": sim_val,
                "rejected": sim_val < self._rejection_threshold,
            })

        best = top_list[0]
        return {
            "class_name": best["class_name"],
            "class_label": best["class_label"],
            "confidence": best["confidence"],
            "rejected": best["rejected"],
            "top_k": top_list,
        }

    @torch.no_grad()
    def classify_batch(
        self,
        images: List[Union[Image.Image, np.ndarray]],
        top_k: int = 3,
    ) -> List[dict]:
        """
        Classify a batch of images.

        Args:
            images: list of PIL Images or numpy arrays.
            top_k: number of top predictions per image.

        Returns:
            List of result dicts (same format as classify()).
        """
        tensors = torch.cat(
            [_preprocess_image(img, self._image_size) for img in images],
            dim=0,
        ).to(self.device)  # (B, 3, H, W)

        features = self._backbone(tensors)       # (B, hidden_dim)
        embeddings = self._projection(features)  # (B, embedding_dim)

        similarities = torch.mm(embeddings, self._prototypes.t())  # (B, N_classes)

        k = min(top_k, len(self._class_names))
        results = []
        for i in range(embeddings.size(0)):
            sims = similarities[i]
            top_values, top_indices = sims.topk(k)

            top_list = []
            for sim_val, class_idx in zip(top_values.tolist(), top_indices.tolist()):
                top_list.append({
                    "class_name": self._class_names[class_idx],
                    "class_label": class_idx,
                    "confidence": sim_val,
                    "rejected": sim_val < self._rejection_threshold,
                })

            best = top_list[0]
            results.append({
                "class_name": best["class_name"],
                "class_label": best["class_label"],
                "confidence": best["confidence"],
                "rejected": best["rejected"],
                "top_k": top_list,
            })

        return results

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def num_classes(self) -> int:
        return len(self._class_names)

    @property
    def class_names(self) -> List[str]:
        return list(self._class_names)

    def __repr__(self) -> str:
        return (
            f"CodexClassifier("
            f"backbone={self.config.get('backbone')}, "
            f"num_classes={self.num_classes}, "
            f"device={self.device})"
        )
