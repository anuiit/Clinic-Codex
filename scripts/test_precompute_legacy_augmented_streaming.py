from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest
import torch

from scripts import precompute_legacy_augmented_streaming as module


class _DummyTransform:
    def set_random_seed(self, seed: int) -> None:
        self.seed = seed


@pytest.mark.parametrize(
    ("failure_mode", "expected_message"),
    [("clean", "load_clean failed"), ("augment", "augmentation failed")],
)
def test_build_cache_fails_fast_before_dino_load(tmp_path, monkeypatch, failure_mode, expected_message):
    config = tmp_path / "config.yaml"
    config.write_text(
        """
paths:
  metadata_csv: /tmp/metadata.csv
data:
  image_size: 224
  min_images_per_class: 1
model:
  backbone: vit_small
training:
  seed: 42
""",
        encoding="utf-8",
    )

    metadata = pd.DataFrame(
        [
            {
                "image_path": "/tmp/missing.png",
                "class_label": 7,
                "element_name": "atl",
            }
        ]
    )

    monkeypatch.setattr(module, "load_metadata", lambda path: metadata)
    monkeypatch.setattr(module, "filter_classes", lambda df, min_images: df)
    monkeypatch.setattr(module, "get_train_transform_numpy", lambda cfg, image_size: _DummyTransform())

    hub_called = {"value": False}

    def fail_hub(*args, **kwargs):
        hub_called["value"] = True
        raise AssertionError("DINO should not be loaded before preprocessing errors are raised")

    monkeypatch.setattr(module.torch.hub, "load", fail_hub)

    if failure_mode == "clean":
        def load_clean(*args, **kwargs):
            raise FileNotFoundError("missing file")

        def load_and_augment(*args, **kwargs):
            return torch.zeros(3, 224, 224)
    else:
        def load_clean(*args, **kwargs):
            return torch.zeros(3, 224, 224)

        def load_and_augment(*args, **kwargs):
            raise ValueError("bad augmentation")

    monkeypatch.setattr(module, "load_clean", load_clean)
    monkeypatch.setattr(module, "load_and_augment", load_and_augment)

    args = Namespace(
        config=config,
        output_dir=tmp_path / "out",
        multiplier=5,
        adaptive=False,
        batch_size=2,
        device="cpu",
        seed=42,
        allowed_paths_cache=None,
    )

    with pytest.raises(RuntimeError, match=expected_message):
        module.build_cache(args)

    assert hub_called["value"] is False


class _NoSeedTransform:
    pass


def test_build_cache_requires_seed_api_before_loading_images(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text(
        """
paths:
  metadata_csv: /tmp/metadata.csv
data:
  image_size: 224
  min_images_per_class: 1
model:
  backbone: vit_small
training:
  seed: 42
""",
        encoding="utf-8",
    )

    metadata = pd.DataFrame(
        [
            {
                "image_path": "/tmp/a.png",
                "class_label": 7,
                "element_name": "atl",
            }
        ]
    )

    monkeypatch.setattr(module, "load_metadata", lambda path: metadata)
    monkeypatch.setattr(module, "filter_classes", lambda df, min_images: df)
    monkeypatch.setattr(module, "get_train_transform_numpy", lambda cfg, image_size: _NoSeedTransform())

    called = {"clean": False, "augment": False, "hub": False}

    def fail_clean(*args, **kwargs):
        called["clean"] = True
        raise AssertionError("load_clean should not be called when seed API is missing")

    def fail_augment(*args, **kwargs):
        called["augment"] = True
        raise AssertionError("load_and_augment should not be called when seed API is missing")

    def fail_hub(*args, **kwargs):
        called["hub"] = True
        raise AssertionError("DINO should not be loaded when seed API is missing")

    monkeypatch.setattr(module, "load_clean", fail_clean)
    monkeypatch.setattr(module, "load_and_augment", fail_augment)
    monkeypatch.setattr(module.torch.hub, "load", fail_hub)

    args = Namespace(
        config=config,
        output_dir=tmp_path / "out",
        multiplier=5,
        adaptive=False,
        batch_size=2,
        device="cpu",
        seed=42,
        allowed_paths_cache=None,
    )

    with pytest.raises(RuntimeError, match="set_random_seed"):
        module.build_cache(args)

    assert called == {"clean": False, "augment": False, "hub": False}


class _DummyBackbone(torch.nn.Module):
    def __init__(self, embed_dim: int = 1) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, batch):
        return torch.zeros((batch.shape[0], self.embed_dim), dtype=batch.dtype)


class _DummyTransformWithSeed:
    def set_random_seed(self, seed: int) -> None:
        self.seed = seed


def test_build_cache_filters_allowed_paths_before_image_loading(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text(
        """
paths:
  metadata_csv: /tmp/metadata.csv
data:
  image_size: 224
  min_images_per_class: 1
model:
  backbone: vit_small
training:
  seed: 42
""",
        encoding="utf-8",
    )

    allowed = tmp_path / "allowed.png"
    allowed.write_bytes(b"allowed")
    disallowed = tmp_path / "disallowed.png"
    disallowed.write_bytes(b"disallowed")

    metadata = pd.DataFrame(
        [
            {
                "image_path": str(allowed),
                "class_label": 7,
                "element_name": "atl",
            },
            {
                "image_path": str(disallowed),
                "class_label": 8,
                "element_name": "beta",
            },
        ]
    )

    allowed_cache = tmp_path / "allowed.pt"
    torch.save({"image_paths": [str(allowed)]}, allowed_cache)

    monkeypatch.setattr(module, "load_metadata", lambda path: metadata)
    monkeypatch.setattr(module, "filter_classes", lambda df, min_images: df)
    monkeypatch.setattr(module, "get_train_transform_numpy", lambda cfg, image_size: _DummyTransformWithSeed())
    monkeypatch.setattr(module.torch.hub, "load", lambda *args, **kwargs: _DummyBackbone())

    loaded_paths: list[str] = []

    def load_clean(path, image_size):
        loaded_paths.append(path)
        return torch.zeros(3, 224, 224)

    def load_and_augment(path, transform, image_size):
        loaded_paths.append(f"aug:{path}")
        return torch.zeros(3, 224, 224)

    monkeypatch.setattr(module, "load_clean", load_clean)
    monkeypatch.setattr(module, "load_and_augment", load_and_augment)

    args = Namespace(
        config=config,
        output_dir=tmp_path / "out",
        multiplier=1,
        adaptive=False,
        batch_size=2,
        device="cpu",
        seed=42,
        allowed_paths_cache=allowed_cache,
    )

    payload = module.build_cache(args)

    assert payload["allowed_source_count"] == 1
    assert payload["labels"].tolist() == [7, 7]
    assert loaded_paths == [str(allowed), f"aug:{str(allowed)}"]
    assert str(disallowed) not in loaded_paths


def test_configure_seed_enables_deterministic_algorithms(monkeypatch):
    calls = []

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    monkeypatch.setattr(module.torch, "use_deterministic_algorithms", lambda flag: calls.append(flag))

    module.configure_seed(123)

    assert calls == [True]
    assert module.os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"


def test_configure_seed_rejects_non_deterministic_workspace_config(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "invalid")

    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG must be deterministic"):
        module.configure_seed(123)


def test_build_cache_accepts_relative_metadata_with_absolute_allowed_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    config = tmp_path / "config.yaml"
    config.write_text(
        """
paths:
  metadata_csv: /tmp/metadata.csv
data:
  image_size: 224
  min_images_per_class: 1
model:
  backbone: vit_small
training:
  seed: 42
""",
        encoding="utf-8",
    )

    rel_image = Path("relative.png")
    rel_image.write_bytes(b"image")
    abs_image = rel_image.resolve()

    metadata = pd.DataFrame(
        [
            {
                "image_path": str(rel_image),
                "class_label": 7,
                "element_name": "atl",
            }
        ]
    )

    allowed_cache = tmp_path / "allowed.pt"
    torch.save({"image_paths": [str(abs_image)]}, allowed_cache)

    monkeypatch.setattr(module, "load_metadata", lambda path: metadata)
    monkeypatch.setattr(module, "filter_classes", lambda df, min_images: df)
    monkeypatch.setattr(module, "get_train_transform_numpy", lambda cfg, image_size: _DummyTransformWithSeed())
    monkeypatch.setattr(module.torch.hub, "load", lambda *args, **kwargs: _DummyBackbone())

    seen = []

    def load_clean(path, image_size):
        seen.append(path)
        return torch.zeros(3, 224, 224)

    def load_and_augment(path, transform, image_size):
        seen.append(f"aug:{path}")
        return torch.zeros(3, 224, 224)

    monkeypatch.setattr(module, "load_clean", load_clean)
    monkeypatch.setattr(module, "load_and_augment", load_and_augment)

    args = Namespace(
        config=config,
        output_dir=tmp_path / "out",
        multiplier=1,
        adaptive=False,
        batch_size=2,
        device="cpu",
        seed=42,
        allowed_paths_cache=allowed_cache,
    )

    payload = module.build_cache(args)

    assert payload["allowed_source_count"] == 1
    assert payload["image_paths"] == [str(rel_image), str(rel_image)]
    assert seen == [str(rel_image), f"aug:{str(rel_image)}"]


def test_build_cache_is_repeatable_for_same_seed(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text(
        """
paths:
  metadata_csv: /tmp/metadata.csv
data:
  image_size: 224
  min_images_per_class: 1
model:
  backbone: vit_small
training:
  seed: 42
""",
        encoding="utf-8",
    )

    image = tmp_path / "repeatable.png"
    image.write_bytes(b"image")
    metadata = pd.DataFrame(
        [
            {
                "image_path": str(image),
                "class_label": 7,
                "element_name": "atl",
            }
        ]
    )

    monkeypatch.setattr(module, "load_metadata", lambda path: metadata)
    monkeypatch.setattr(module, "filter_classes", lambda df, min_images: df)
    monkeypatch.setattr(module, "get_train_transform_numpy", lambda cfg, image_size: _DummyTransformWithSeed())
    monkeypatch.setattr(module.torch.hub, "load", lambda *args, **kwargs: _DummyBackbone(embed_dim=4))

    def load_clean(path, image_size):
        return torch.rand(3, 224, 224)

    def load_and_augment(path, transform, image_size):
        return torch.rand(3, 224, 224)

    monkeypatch.setattr(module, "load_clean", load_clean)
    monkeypatch.setattr(module, "load_and_augment", load_and_augment)

    args = Namespace(
        config=config,
        output_dir=tmp_path / "out",
        multiplier=2,
        adaptive=False,
        batch_size=2,
        device="cpu",
        seed=123,
        allowed_paths_cache=None,
    )

    first = module.build_cache(args)
    second = module.build_cache(args)

    assert torch.equal(first["features"], second["features"])
    assert torch.equal(first["labels"], second["labels"])
    assert first["image_paths"] == second["image_paths"]
    assert first["is_augmented"] == second["is_augmented"]
