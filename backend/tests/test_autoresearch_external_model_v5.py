from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "autoresearch_external_model_v5.py"
SPEC = importlib.util.spec_from_file_location(
    "autoresearch_external_model_v5", SCRIPT
)
assert SPEC is not None
assert SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_normalized_source_group_removes_terminal_numeric_suffix() -> None:
    assert (
        module.normalized_source_group(
            r"C:\Corpus\Elements\0001-name\03_04_22-48.jpg"
        )
        == "c:/corpus/elements/0001-name/03_04_22"
    )
    assert (
        module.normalized_source_group(
            "/data/Elements/0001-name/03_04_22-detail.jpg"
        )
        == "/data/elements/0001-name/03_04_22-detail"
    )


def _synthetic_external() -> dict[str, object]:
    paths = [f"/snapshot/out-{index}.bmp" for index in range(8)]
    return {
        "features": torch.zeros(8, module.FEATURE_DIM),
        "labels": torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
        "image_paths": paths,
        "class_names": {0: "a", 1: "b"},
    }


def _synthetic_provenance() -> list[dict[str, object]]:
    sources = [
        "/source/a/page-01.jpg",
        "/source/a/page-02.jpg",
        "/source/a/other.jpg",
        "/source/a/train.jpg",
        "/source/b/page-01.jpg",
        "/source/b/page-02.jpg",
        "/source/b/other.jpg",
        "/source/b/train.jpg",
    ]
    return [
        {
            "cache_index": index,
            "output_path": f"/snapshot/out-{index}.bmp",
            "source_path": source,
            "source_group": module.normalized_source_group(source),
            "class_label": 0 if index < 4 else 1,
            "class_name": "a" if index < 4 else "b",
        }
        for index, source in enumerate(sources)
    ]


def test_leakage_safe_split_preserves_test_and_purges_all_strict_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    external = _synthetic_external()
    provenance = _synthetic_provenance()
    monkeypatch.setattr(
        module,
        "v4_indices",
        lambda *_args: ([0, 2, 4, 6], [1, 5], [0, 1, 2, 4, 5, 6]),
    )

    split = module.leakage_safe_split(
        external, provenance, tmp_path / "unused.json"
    )

    assert split["test"] == [1, 5]
    assert split["dev"] == [2, 6]
    assert split["external_train"] == [3, 7]
    assert split["audit"]["removed_v4_dev_count"] == 2
    assert split["audit"]["v4_test_preserved_exactly"] is True
    assert split["audit"]["overlap_total"] == 0


def test_provenance_validation_rejects_class_label_mismatch() -> None:
    external = {
        "features": torch.zeros(1, module.FEATURE_DIM),
        "labels": torch.tensor([0]),
        "image_paths": ["/snapshot/out.bmp"],
        "class_names": {0: "a"},
    }
    snapshot = {
        "rows": [
            {
                "output_path": "/snapshot/out.bmp",
                "source_path": "/source/a/page-01.jpg",
                "class_label": 1,
                "class_name": "a",
            }
        ]
    }

    with pytest.raises(ValueError, match="class_label mismatch"):
        module.provenance_for_cache(external, snapshot)


def test_cache_taxonomy_validation_requires_exact_dense_runtime_order() -> None:
    cache = {
        "labels": torch.arange(module.CLASS_COUNT),
        "class_names": {
            index: f"class-{index}" for index in range(module.CLASS_COUNT)
        },
    }
    names = [f"class-{index}" for index in range(module.CLASS_COUNT)]
    module.validate_cache_taxonomy(cache, names, cache_name="test")

    cache["class_names"][1] = "wrong"
    with pytest.raises(ValueError, match="taxonomy differs"):
        module.validate_cache_taxonomy(cache, names, cache_name="test")


def test_experiment_grid_has_ten_prototype_only_runtime_variants() -> None:
    specs = module.experiment_specs()

    assert len(specs) == 10
    assert {spec["iteration"] for spec in specs} == set(range(1, 11))
    assert all(spec["projection"] == "historical_frozen" for spec in specs)
    assert all(
        spec["prototype_initialization"] == "historical_runtime"
        for spec in specs
    )
    assert {spec["lr"] for spec in specs} == {0.75e-3, 1.0e-3, 1.25e-3}
    assert {spec["epochs"] for spec in specs} == {75, 100, 125}
    assert 1.0 in {spec["anchor_weight"] for spec in specs}
    assert len({spec["seed"] for spec in specs}) == 10


def test_train_prototypes_keeps_projection_outside_optimizer() -> None:
    embeddings = F.normalize(torch.randn(12, module.EMBEDDING_DIM), dim=1)
    labels = torch.tensor(list(range(3)) * 4)
    historical = F.normalize(
        torch.randn(module.CLASS_COUNT, module.EMBEDDING_DIM), dim=1
    )
    labels = torch.cat(
        [labels, torch.arange(3, module.CLASS_COUNT, dtype=torch.long)]
    )
    embeddings = torch.cat(
        [
            embeddings,
            F.normalize(
                torch.randn(
                    module.CLASS_COUNT - 3, module.EMBEDDING_DIM
                ),
                dim=1,
            ),
        ]
    )
    spec = {
        **module.experiment_specs()[0],
        "epochs": 1,
        "batch_size": 512,
    }

    optimized = module.train_prototypes(
        embeddings, labels, historical, spec, torch.device("cpu")
    )

    assert optimized.shape == (module.CLASS_COUNT, module.EMBEDDING_DIM)
    assert torch.allclose(
        optimized.norm(dim=1), torch.ones(module.CLASS_COUNT), atol=1e-5
    )


def test_sealed_test_refuses_second_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    external = {
        "features": torch.randn(2, module.FEATURE_DIM),
        "labels": torch.tensor([0, 1]),
    }
    model = torch.nn.Identity()
    monkeypatch.setattr(
        module,
        "project",
        lambda *_args, **_kwargs: F.normalize(
            torch.randn(2, module.EMBEDDING_DIM), dim=1
        ),
    )
    prototypes = F.normalize(
        torch.randn(module.CLASS_COUNT, module.EMBEDDING_DIM), dim=1
    )
    sealed = module.SealedTest(
        external=external,
        indices=[0, 1],
        truth=external["labels"],
        projection=model,
        historical_prototypes=prototypes,
        device=torch.device("cpu"),
    )

    sealed.evaluate(prototypes)
    with pytest.raises(RuntimeError, match="exactly once"):
        sealed.evaluate(prototypes)
