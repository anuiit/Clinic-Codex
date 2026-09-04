from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch
import pytest

from backend.codex_pipeline.models.projection_head import ProjectionHead
from backend.codex_pipeline.scripts.evaluate import update_annotated_prototypes


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_selective_update_preserves_untouched_prototypes_and_rejects_drift(tmp_path):
    model = ProjectionHead(input_dim=2, embedding_dim=2)
    checkpoint = {"model_state_dict": model.state_dict()}
    (tmp_path / "weights").mkdir()
    base_path = tmp_path / "weights" / "prototypes.pt"
    (tmp_path / "config.json").write_text(json.dumps({"backbone": "test", "image_size": 224, "hidden_dim": 2}))
    base = {"prototypes": torch.eye(2), "class_labels": torch.tensor([4, 9]),
            "class_names": {4: "alpha", 9: "beta"}}
    torch.save(base, base_path)
    torch.save(model.state_dict(), base_path.with_name("projection.pt"))
    manifest = {"rows": [
        {"row_id": "a", "source_kind": "live_annotation", "class_name": "alpha", "dataset_split": "train"},
        {"row_id": "b", "source_kind": "legacy", "class_name": "beta", "dataset_split": "train"},
    ], "duplicates": [], "conflicts": []}
    cache = {"row_ids": ["a", "b"], "labels": torch.tensor([0, 1]),
             "preprocessing": "runtime-lanczos-whitepad-imagenet.v1",
             "backbone": "test", "image_size": 224, "hidden_dim": 2,
             "class_names": {0: "alpha", 1: "beta"}, "dataset_splits": ["train", "train"]}
    fitted = torch.tensor([[0.6, 0.8], [-1.0, 0.0]])
    analysis = {"prototypes": fitted.clone(), "class_labels": torch.tensor([0, 1])}
    result = update_annotated_prototypes(analysis, checkpoint, cache, manifest, base_path)
    assert torch.equal(analysis["prototypes"][0], fitted[0])
    assert torch.equal(analysis["prototypes"][1], base["prototypes"][1])
    assert result["updated_class_names"] == ["alpha"]
    assert result["preserved_class_count"] == 1
    cache["preprocessing"] = "old-bilinear"
    with pytest.raises(ValueError, match="preprocessing.*rebuild"):
        update_annotated_prototypes(analysis, checkpoint, cache, manifest, base_path)
    cache["preprocessing"] = "runtime-lanczos-whitepad-imagenet.v1"
    cache["backbone"] = "different"
    with pytest.raises(ValueError, match="backbone"):
        update_annotated_prototypes(analysis, checkpoint, cache, manifest, base_path)
    cache["backbone"] = "test"
    cache["dataset_splits"][0] = "locked_test"
    with pytest.raises(ValueError, match="labels/splits"):
        update_annotated_prototypes(analysis, checkpoint, cache, manifest, base_path)
    cache["dataset_splits"][0] = "train"
    checkpoint["model_state_dict"]["net.0.weight"] += 1
    with pytest.raises(ValueError, match="unchanged base projection"):
        update_annotated_prototypes(analysis, checkpoint, cache, manifest, base_path)


def test_prototype_export_uses_only_persisted_train_rows(tmp_path: Path) -> None:
    features_path = tmp_path / "features.pt"
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.9, 0.1],
            [0.1, 0.9],
            [-1.0, 0.0],
            [0.0, -1.0],
        ],
        dtype=torch.float32,
    )
    torch.save(
        {
            "features": features,
            "labels": torch.tensor([0, 1, 0, 1, 0, 1]),
            "dataset_splits": ["train", "train", "dev", "dev", "locked_test", "locked_test"],
            "class_names": {0: "alpha", 1: "beta"},
            "hidden_dim": 2,
        },
        features_path,
    )
    model = ProjectionHead(input_dim=2, embedding_dim=2)
    checkpoint_path = tmp_path / "best.pt"
    torch.save(
        {
            "epoch": 0,
            "model_state_dict": model.state_dict(),
            "train_acc": 1.0,
            "val_acc": 1.0,
            "hidden_dim": 2,
            "config": {
                "paths": {"prototype_dir": str(tmp_path / "unused")},
                "data": {"val_fraction": 0.2},
                "model": {"embedding_dim": 2},
                "training": {
                    "seed": 42,
                    "device": "cpu",
                    "temperature": 0.1,
                    "n_way": 2,
                    "q_queries": 1,
                },
                "evaluation": {"k_shot_values": [1]},
            },
        },
        checkpoint_path,
    )
    prototype_dir = tmp_path / "prototypes"

    result = subprocess.run(
        [
            sys.executable,
            "backend/codex_pipeline/scripts/evaluate.py",
            "--checkpoint",
            str(checkpoint_path),
            "--features",
            str(features_path),
            "--split-strategy",
            "persisted",
            "--prototype-split",
            "train",
            "--skip-few-shot",
            "--export-prototypes",
            "--prototype-dir",
            str(prototype_dir),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr

    exported = torch.load(prototype_dir / "prototypes.pt", map_location="cpu", weights_only=True)
    provenance = json.loads((prototype_dir / "provenance.json").read_text(encoding="utf-8"))
    assert exported["class_meta"][0]["count"] == 1
    assert exported["class_meta"][1]["count"] == 1
    assert provenance["prototype_split"] == "train"
    assert provenance["split_strategy"] == "persisted"
