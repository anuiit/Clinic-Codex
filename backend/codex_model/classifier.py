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

import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


_BACKBONE_HIDDEN_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
}
_RUNTIME_ARTIFACTS = {
    "runtime/config.json",
    "runtime/weights/projection.pt",
    "runtime/weights/prototypes.pt",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
PREPROCESSING_VERSION = "runtime-lanczos-whitepad-imagenet.v1"


class ModelPackageValidationError(ValueError):
    """Raised before inference when a runtime package is unsafe or inconsistent."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(value: object) -> Path:
    raw = str(value).replace("\\", "/")
    relative = Path(raw)
    if not raw or relative.is_absolute() or ".." in relative.parts:
        raise ModelPackageValidationError(f"unsafe package artifact path: {value}")
    return relative


def _resolve_package_file(version_dir: Path, value: object) -> Path:
    relative = _safe_relative_path(value)
    try:
        version_root = version_dir.resolve(strict=True)
        resolved = (version_dir / relative).resolve(strict=True)
        resolved.relative_to(version_root)
    except (FileNotFoundError, ValueError) as exc:
        raise ModelPackageValidationError(
            f"unsafe or missing package artifact: {relative.as_posix()}"
        ) from exc
    if not resolved.is_file():
        raise ModelPackageValidationError(
            f"package artifact is not a file: {relative.as_posix()}"
        )
    return resolved


def _read_checksum_inventory(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ModelPackageValidationError(f"checksums file not found: {path}")
    records: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or _SHA256_RE.fullmatch(parts[0]) is None:
            raise ModelPackageValidationError(
                f"invalid checksums line {line_number}: {raw_line!r}"
            )
        relative = _safe_relative_path(parts[1].strip()).as_posix()
        if relative in records:
            raise ModelPackageValidationError(f"duplicate checksums entry: {relative}")
        records[relative] = parts[0]
    return records


def _validate_runtime_integrity(runtime_dir: Path) -> None:
    """Validate registry integrity metadata before tensor deserialization."""

    version_dir = runtime_dir.parent
    manifest_path = version_dir / "manifest.json"
    checksums_path = version_dir / "checksums.sha256"
    if not manifest_path.is_file() or not checksums_path.is_file():
        raise ModelPackageValidationError(
            "runtime package requires sibling manifest.json and checksums.sha256"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelPackageValidationError(f"invalid package manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ModelPackageValidationError("package manifest must be a JSON object")
    if manifest.get("schema_version") != 1:
        raise ModelPackageValidationError(
            f"unsupported package schema_version: {manifest.get('schema_version')}"
        )
    if manifest.get("model_id") != "codex_classifier":
        raise ModelPackageValidationError("package manifest model_id mismatch")
    if manifest.get("version_id") != version_dir.name:
        raise ModelPackageValidationError("package manifest version_id mismatch")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ModelPackageValidationError("package manifest artifacts must be a list")
    checksums = _read_checksum_inventory(checksums_path)
    declared: set[str] = set()
    for record in artifacts:
        if not isinstance(record, dict):
            raise ModelPackageValidationError("package artifact records must be objects")
        relative = _safe_relative_path(record.get("path") or "").as_posix()
        if relative in declared:
            raise ModelPackageValidationError(f"duplicate package artifact: {relative}")
        expected = record.get("sha256")
        if not isinstance(expected, str) or _SHA256_RE.fullmatch(expected) is None:
            raise ModelPackageValidationError(f"invalid artifact checksum: {relative}")
        artifact_path = _resolve_package_file(version_dir, relative)
        actual = _sha256_file(artifact_path)
        if actual != expected:
            raise ModelPackageValidationError(
                f"artifact checksum mismatch for {relative}: expected {expected}, got {actual}"
            )
        if checksums.get(relative) != expected:
            raise ModelPackageValidationError(
                f"checksum inventory mismatch for {relative}"
            )
        declared_size = record.get("size")
        if not isinstance(declared_size, int) or declared_size != artifact_path.stat().st_size:
            raise ModelPackageValidationError(f"artifact size mismatch for {relative}")
        declared.add(relative)

    checksum_only = sorted(set(checksums) - declared)
    if checksum_only:
        raise ModelPackageValidationError(
            "checksum inventory contains undeclared artifacts: " + ", ".join(checksum_only)
        )
    missing = sorted(_RUNTIME_ARTIFACTS - declared)
    if missing:
        raise ModelPackageValidationError(
            "runtime artifacts missing from package manifest: " + ", ".join(missing)
        )


def _load_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ModelPackageValidationError(f"classifier config not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ModelPackageValidationError(f"invalid classifier config: {path}") from exc
    if not isinstance(config, dict):
        raise ModelPackageValidationError("classifier config must be a JSON object")
    if "schema_version" in config and config["schema_version"] != 1:
        raise ModelPackageValidationError(
            f"unsupported classifier config schema_version: {config['schema_version']}"
        )
    backbone = config.get("backbone")
    if backbone not in _BACKBONE_HIDDEN_DIMS:
        raise ModelPackageValidationError(f"unsupported DINOv2 backbone: {backbone}")
    hidden_dim = config.get("hidden_dim")
    if hidden_dim != _BACKBONE_HIDDEN_DIMS[backbone]:
        raise ModelPackageValidationError(
            f"hidden_dim {hidden_dim} does not match {backbone}"
        )
    embedding_dim = config.get("embedding_dim")
    if isinstance(embedding_dim, bool) or not isinstance(embedding_dim, int) or embedding_dim <= 0:
        raise ModelPackageValidationError("embedding_dim must be a positive integer")
    image_size = config.get("image_size")
    if (
        isinstance(image_size, bool)
        or not isinstance(image_size, int)
        or image_size < 14
        or image_size > 1024
        or image_size % 14 != 0
    ):
        raise ModelPackageValidationError(
            "image_size must be an integer between 14 and 1024 divisible by 14"
        )
    threshold = config.get("rejection_threshold")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or not -1.0 <= float(threshold) <= 1.0
    ):
        raise ModelPackageValidationError("rejection_threshold must be finite and in [-1, 1]")
    num_classes = config.get("num_classes")
    class_names = config.get("class_names")
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes <= 0:
        raise ModelPackageValidationError("num_classes must be a positive integer")
    if (
        not isinstance(class_names, list)
        or len(class_names) != num_classes
        or any(not isinstance(name, str) or not name.strip() for name in class_names)
        or len(set(class_names)) != len(class_names)
    ):
        raise ModelPackageValidationError("config class_names taxonomy is invalid")
    return config


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

    padded = Image.new("RGB", (image_size, image_size), (255, 255, 255))  # white pad — matches training data
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
        model_dir: legacy weights directory, or an immutable registry runtime
                   directory containing config.json and weights/. Defaults to
                   the legacy weights/ folder next to this file.
        device: torch device string (e.g. 'cpu', 'cuda', 'mps').
                Auto-detected when None.
    """

    def __init__(
        self,
        model_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        backbone_manifest: Optional[Union[str, Path]] = None,
    ):
        module_dir = Path(__file__).parent
        requested_dir = Path(model_dir) if model_dir is not None else module_dir / "weights"
        is_runtime_package = (
            (requested_dir / "config.json").is_file()
            and (requested_dir / "weights").is_dir()
        )
        if is_runtime_package:
            _validate_runtime_integrity(requested_dir)
            self._weights_dir = requested_dir / "weights"
            config_path = requested_dir / "config.json"
        else:
            self._weights_dir = requested_dir
            config_path = module_dir / "config.json"
        self.config = _load_config(config_path)

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self._backbone_manifest = backbone_manifest
        self._load_weights()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validated_labels(raw_labels: object, expected_count: int) -> List[int]:
        if not isinstance(raw_labels, torch.Tensor):
            raise ModelPackageValidationError("prototype class_labels must be a tensor")
        if raw_labels.ndim != 1 or raw_labels.numel() != expected_count:
            raise ModelPackageValidationError(
                "prototype class_labels must be one-dimensional and match num_classes"
            )
        if raw_labels.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise ModelPackageValidationError("prototype class_labels must be integers")
        labels = [int(value) for value in raw_labels.tolist()]
        if labels != sorted(labels) or len(set(labels)) != len(labels):
            raise ModelPackageValidationError(
                "prototype class_labels must be unique and sorted"
            )
        return labels

    def _load_weights(self):
        """Validate tensors, then load projection and DINOv2 backbone."""
        proto_path = self._weights_dir / "prototypes.pt"
        proj_path = self._weights_dir / "projection.pt"

        if not proto_path.is_file():
            raise FileNotFoundError(
                f"Prototypes file not found: {proto_path}\n"
                "Run `python -m codex_pipeline.scripts.export_model` first."
            )
        if not proj_path.is_file():
            raise FileNotFoundError(
                f"Projection weights not found: {proj_path}\n"
                "Run `python -m codex_pipeline.scripts.export_model` first."
            )

        try:
            proto_data = torch.load(proto_path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ModelPackageValidationError(
                f"unable to safely load prototype artifact: {proto_path}"
            ) from exc
        if not isinstance(proto_data, Mapping):
            raise ModelPackageValidationError("prototype artifact must be a mapping")
        required_keys = {"prototypes", "class_names", "class_labels", "embedding_dim"}
        missing_keys = required_keys - set(proto_data)
        if missing_keys:
            raise ModelPackageValidationError(
                "prototype artifact is missing keys: " + ", ".join(sorted(missing_keys))
            )

        prototypes = proto_data["prototypes"]
        if not isinstance(prototypes, torch.Tensor) or prototypes.ndim != 2:
            raise ModelPackageValidationError("prototypes must be a two-dimensional tensor")
        if not prototypes.is_floating_point() or not torch.isfinite(prototypes).all():
            raise ModelPackageValidationError("prototypes must contain finite floating-point values")

        num_classes = self.config["num_classes"]
        embedding_dim = self.config["embedding_dim"]
        if tuple(prototypes.shape) != (num_classes, embedding_dim):
            raise ModelPackageValidationError(
                "prototype shape does not match config num_classes and embedding_dim"
            )
        if proto_data["embedding_dim"] != embedding_dim:
            raise ModelPackageValidationError("prototype embedding_dim does not match config")
        norms = prototypes.norm(dim=1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=1e-3, rtol=1e-3):
            raise ModelPackageValidationError("prototype rows must be L2-normalized")

        raw_class_names = proto_data["class_names"]
        if not isinstance(raw_class_names, Mapping):
            raise ModelPackageValidationError("prototype class_names must be a mapping")
        if any(
            isinstance(label, bool)
            or not isinstance(label, int)
            or not isinstance(name, str)
            or not name.strip()
            for label, name in raw_class_names.items()
        ):
            raise ModelPackageValidationError("prototype class_names mapping is invalid")
        labels = self._validated_labels(proto_data["class_labels"], num_classes)
        if set(labels) != set(raw_class_names):
            raise ModelPackageValidationError(
                "prototype class_labels and class_names labels disagree"
            )
        names = [raw_class_names[label] for label in labels]
        if len(set(names)) != len(names) or names != self.config["class_names"]:
            raise ModelPackageValidationError(
                "prototype taxonomy does not match config class_names"
            )

        try:
            proj_state = torch.load(proj_path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ModelPackageValidationError(
                f"unable to safely load projection artifact: {proj_path}"
            ) from exc
        if not isinstance(proj_state, Mapping) or any(
            not isinstance(key, str)
            or not isinstance(value, torch.Tensor)
            or not torch.isfinite(value).all()
            for key, value in proj_state.items()
        ):
            raise ModelPackageValidationError("projection state_dict is invalid")
        hidden_dim = self.config["hidden_dim"]
        projection = _ProjectionHead(hidden_dim, embedding_dim)
        try:
            projection.load_state_dict(proj_state, strict=True)
        except RuntimeError as exc:
            raise ModelPackageValidationError(
                "projection dimensions do not match classifier config"
            ) from exc

        self._label_index = labels
        self._class_names = names
        self._prototypes = prototypes.to(self.device)
        self._projection = projection.to(self.device)
        self._projection.eval()

        backbone_name = self.config["backbone"]
        if self._backbone_manifest is not None:
            from scripts.pin_dinov2 import load_backbone

            self._backbone, _ = load_backbone(backbone_name, self.device, self._backbone_manifest)
        else:
            self._backbone = torch.hub.load(
                "facebookresearch/dinov2", backbone_name, pretrained=True,
            ).to(self.device).eval()

        self._image_size = self.config["image_size"]
        self._rejection_threshold = float(self.config["rejection_threshold"])

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
                "class_label": self._label_index[class_idx],
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
                    "class_label": self._label_index[class_idx],
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
