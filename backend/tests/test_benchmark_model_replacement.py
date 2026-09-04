from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "benchmark_model_replacement.py"
spec = importlib.util.spec_from_file_location("benchmark_model_replacement", SCRIPT)
assert spec is not None
assert spec.loader is not None
benchmark = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = benchmark
spec.loader.exec_module(benchmark)

from backend.codex_pipeline.models.projection_head import ProjectionHead


CLASS_COUNT = 286


def test_snapshot_row_identity_takes_priority_over_annotation_index_and_relative_path():
    rows = [{"row_id": "b", "index": 0, "output_path": "Elements/b.bmp"}]
    assert benchmark._resolve_indices(rows, features_len=2, image_paths=["/snapshot/Elements/a.bmp", "/snapshot/Elements/b.bmp"], row_ids=["a", "b"]) == [1]


def _write_package(root: Path, *, nested_runtime: bool = False) -> Path:
    package_root = root / "runtime" if nested_runtime else root
    weights_dir = package_root / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    class_names = [f"class_{idx:03d}" for idx in range(CLASS_COUNT)]
    config = {
        "model_version": "synthetic",
        "backbone": "dinov2_vits14",
        "embedding_dim": CLASS_COUNT,
        "hidden_dim": CLASS_COUNT,
        "image_size": 224,
        "rejection_threshold": 0.35,
        "num_classes": CLASS_COUNT,
        "class_names": class_names,
    }
    (package_root / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    head = ProjectionHead(input_dim=CLASS_COUNT, embedding_dim=CLASS_COUNT)
    with torch.no_grad():
        first = head.net[0]
        second = head.net[3]
        first.weight.zero_()
        first.bias.zero_()
        first.weight.copy_(torch.eye(CLASS_COUNT))
        second.weight.zero_()
        second.bias.zero_()
        second.weight.copy_(torch.eye(CLASS_COUNT))
    torch.save(head.state_dict(), weights_dir / "projection.pt")
    torch.save(
        {
            "prototypes": torch.eye(CLASS_COUNT),
            "class_labels": torch.arange(CLASS_COUNT, dtype=torch.long),
            "class_names": {idx: name for idx, name in enumerate(class_names)},
            "embedding_dim": CLASS_COUNT,
        },
        weights_dir / "prototypes.pt",
    )
    return package_root


def _write_features(path: Path) -> Path:
    rows = []
    features = []
    labels = []
    image_paths = []

    for class_label in range(CLASS_COUNT):
        train_index = len(rows)
        train_feature = torch.zeros(CLASS_COUNT, dtype=torch.float32)
        train_feature[class_label] = 1.0
        features.append(train_feature)
        labels.append(class_label)
        image_paths.append(f"train/{class_label:03d}.pt")
        rows.append(
            {
                "index": train_index,
                "dataset_split": "train",
                "class_label": class_label,
                "class_name": f"class_{class_label:03d}",
                "output_path": f"train/{class_label:03d}.pt",
            }
        )

        eval_index = len(rows)
        eval_feature = torch.zeros(CLASS_COUNT, dtype=torch.float32)
        eval_feature[class_label] = 1.0
        features.append(eval_feature)
        labels.append(class_label)
        image_paths.append(f"test/{class_label:03d}.pt")
        rows.append(
            {
                "index": eval_index,
                "dataset_split": "test",
                "class_label": class_label,
                "class_name": f"class_{class_label:03d}",
                "output_path": f"test/{class_label:03d}.pt",
            }
        )

    payload = {
        "preprocessing": benchmark.PREPROCESSING_VERSION,
        "features": torch.stack(features),
        "labels": torch.tensor(labels, dtype=torch.long),
        "image_paths": image_paths,
        "class_names": {idx: f"class_{idx:03d}" for idx in range(CLASS_COUNT)},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    manifest = {
        "schema_version": "synthetic.split.v1",
        "rows": rows,
    }
    manifest_path = path.with_name("split_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


@pytest.mark.parametrize("preprocessing", [None, "old-bilinear"])
def test_stored_benchmark_rejects_old_preprocessing(tmp_path, preprocessing):
    features = tmp_path / "features.pt"
    manifest = _write_features(features)
    payload = torch.load(features, weights_only=True)
    payload["preprocessing"] = preprocessing
    torch.save(payload, features)
    with pytest.raises(ValueError, match="preprocessing.*rebuild"):
        benchmark.benchmark_model_replacement(features_path=features, split_manifest_path=manifest,
                                             runtime_dir=tmp_path, candidate_dir=tmp_path)


def test_benchmark_model_replacement_uses_exported_prototypes_and_writes_json(tmp_path: Path) -> None:
    features_path = tmp_path / "features.pt"
    split_manifest_path = _write_features(features_path)
    runtime_dir = _write_package(tmp_path / "runtime")
    candidate_root = _write_package(tmp_path / "candidate", nested_runtime=True)
    output_path = tmp_path / "benchmark.json"

    report = benchmark.benchmark_model_replacement(
        features_path=features_path,
        split_manifest_path=split_manifest_path,
        runtime_dir=runtime_dir,
        candidate_dir=candidate_root,
        output_path=output_path,
        batch_size=64,
        device="cpu",
    )

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert report == written
    assert report["schema_version"] == "benchmark-model-replacement.v2"
    assert report["prototype_mode"] == "stored"
    assert report["models"]["runtime"]["prototype_source"]["sha256"]
    assert report["numeric_label_contract"]["equal"] is True
    assert report["class_count"] == CLASS_COUNT
    assert report["eval_split"] == "test"
    assert report["train_examples"] == CLASS_COUNT
    assert report["eval_examples"] == CLASS_COUNT

    runtime = report["models"]["runtime"]
    candidate = report["models"]["candidate"]
    assert runtime["top1_micro"] == 1.0
    assert runtime["top3_micro"] == 1.0
    assert runtime["top1_macro"] == 1.0
    assert runtime["top3_macro"] == 1.0
    assert runtime["coverage"] == 1.0
    assert runtime["coverage_macro"] == 1.0
    assert candidate["top1_micro"] == 1.0
    assert candidate["top3_micro"] == 1.0
    assert candidate["top1_macro"] == 1.0
    assert candidate["top3_macro"] == 1.0
    assert candidate["coverage"] == 1.0
    assert candidate["coverage_macro"] == 1.0
    assert report["comparison"]["top1_micro_delta"] == 0.0
    assert report["comparison"]["top3_micro_delta"] == 0.0
    assert report["comparison"]["coverage_delta"] == 0.0

    assert runtime["train_examples"] == CLASS_COUNT
    assert runtime["eval_examples"] == CLASS_COUNT
    assert candidate["train_examples"] == CLASS_COUNT
    assert candidate["eval_examples"] == CLASS_COUNT
    assert output_path.is_file()


def test_benchmark_model_replacement_detects_degraded_exported_prototypes(tmp_path: Path) -> None:
    features_path = tmp_path / "features.pt"
    split_manifest_path = _write_features(features_path)
    runtime_dir = _write_package(tmp_path / "runtime")
    candidate_root = _write_package(tmp_path / "candidate", nested_runtime=True)
    prototypes_path = candidate_root / "weights" / "prototypes.pt"
    package = torch.load(prototypes_path, map_location="cpu", weights_only=True)
    package["prototypes"] = torch.roll(package["prototypes"], shifts=1, dims=0)
    torch.save(package, prototypes_path)

    stored = benchmark.benchmark_model_replacement(
        features_path=features_path,
        split_manifest_path=split_manifest_path,
        runtime_dir=runtime_dir,
        candidate_dir=candidate_root,
        batch_size=64,
        device="cpu",
    )
    refit = benchmark.benchmark_model_replacement(
        features_path=features_path,
        split_manifest_path=split_manifest_path,
        runtime_dir=runtime_dir,
        candidate_dir=candidate_root,
        batch_size=64,
        device="cpu",
        prototype_mode="refit",
    )

    assert stored["models"]["runtime"]["top1_micro"] == 1.0
    assert stored["models"]["candidate"]["top1_micro"] == 0.0
    assert stored["comparison"]["top1_micro_delta"] == -1.0
    assert refit["models"]["candidate"]["top1_micro"] == 1.0
    assert stored["paired_predictions"][0]["truth_class_name"] == "class_000"
