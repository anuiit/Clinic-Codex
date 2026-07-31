from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "autoresearch_grouped_cv_v7.py"
SPEC = importlib.util.spec_from_file_location("autoresearch_grouped_cv_v7", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def _cache(path: Path) -> dict:
    return {
        "features": torch.zeros(2, 384),
        "labels": torch.tensor([0, 1]),
        "image_paths": [str(path), str(path)],
        "class_names": {0: "a", 1: "b"},
    }


def test_load_cache_accepts_real_cache_contract_without_embedded_digests(tmp_path: Path, monkeypatch) -> None:
    image = tmp_path / "image.bmp"
    image.write_bytes(b"pixels")
    cache_path = tmp_path / "features.pt"
    torch.save(_cache(image), cache_path)
    loaded = module.load_cache(cache_path, cache_name="legacy cache")
    real_sha256_file = module.v5.sha256_file
    calls = []

    def counted_sha256_file(path: Path) -> str:
        calls.append(path)
        return real_sha256_file(path)

    monkeypatch.setattr(module.v5, "sha256_file", counted_sha256_file)
    rows = module.cache_rows(loaded, cache_name="legacy cache")
    assert len(rows) == 2
    assert rows[0]["source_pixel_sha256"] == rows[1]["source_pixel_sha256"]
    assert len(calls) == 1


def test_external_snapshot_supplies_source_pixel_digest(tmp_path: Path) -> None:
    image = tmp_path / "external.bmp"
    cache = _cache(image)
    snapshot_rows = {
        module.resolved(image): {
            "output_path": str(image),
            "source_path": str(tmp_path / "source.jpg"),
            "source_pixel_sha256": "pixel-digest",
        }
    }
    rows = module.cache_rows(cache, cache_name="external cache", snapshot_rows_by_output=snapshot_rows)
    assert {row["source_pixel_sha256"] for row in rows} == {"pixel-digest"}


def test_fold_support_excludes_unsupported_truth_without_leaking_holdout_group() -> None:
    rows = [
        {"admissible_index": 0, "class_label": 0},
        {"admissible_index": 1, "class_label": 1},
        {"admissible_index": 2, "class_label": 1},
    ]
    folds = [
        {"fold": 1, "row_indices": [0, 1], "source_groups": ["g1"], "class_counts": {0: 1, 1: 1}},
        {"fold": 2, "row_indices": [2], "source_groups": ["g2"], "class_counts": {1: 1}},
    ]
    supported, audit = module.enforce_fold_train_support(rows, folds)
    assert supported[0]["holdout_row_indices"] == [0, 1]
    assert supported[0]["row_indices"] == [1]
    assert audit["excluded_oof_class_labels"] == [0]



def test_aggregate_predictions_reports_per_class_counts_and_scores() -> None:
    metrics = module.aggregate_predictions(
        predictions=[0, 1, 1], truth=[0, 0, 1], top3_hits=[True, True, False]
    )
    assert abs(metrics["top1"] - 2 / 3) < 1e-6
    assert metrics["per_class"]["0"] == {"count": 2, "top1": 0.5, "top3": 1.0}
    assert metrics["per_class"]["1"] == {"count": 1, "top1": 1.0, "top3": 0.0}

def test_experiment_contract_is_ten_non_independent_specs() -> None:
    specs = module.build_specs()
    assert len(specs) == 10
    assert [spec["iteration"] for spec in specs] == list(range(1, 11))
    assert all(spec["track"] != "independent" for spec in specs)
    assert all(spec["initialization"] in {"train_means_with_runtime_fallback", "historical_runtime"} for spec in specs)




def test_resolved_is_lexical_and_does_not_require_existing_paths() -> None:
    assert module.resolved("/missing/root/../root/file.bmp") == "/missing/root/../root/file.bmp"
    assert module.resolved("relative/file.bmp").endswith("/clinic-codex/relative/file.bmp")

def test_source_groups_are_root_independent() -> None:
    left = "/tmp/legacy/Elements/0001-a/03_04_22-27.bmp"
    right = "/mnt/f/source/Elements/0015-a/03_04_22-31.jpg"
    assert module.canonical_source_group(left) == "03_04_22"
    assert module.canonical_source_group(right) == "03_04_22"


def test_manifest_digest_index_avoids_source_file_reads(tmp_path: Path, monkeypatch) -> None:
    cache = {
        "features": torch.zeros(1, 384),
        "labels": torch.tensor([0]),
        "image_paths": ["/missing/Elements/0001-a/03_04_22-27.bmp"],
        "class_names": {0: "a"},
    }
    manifest = tmp_path / "legacy.json"
    manifest.write_text(
        '{"images":[{"output_path":"/frozen/Elements/0001-a/03_04_22-27.bmp","output_sha256":"digest-27"}]}',
        encoding="utf-8",
    )
    digest_index = module.load_legacy_digest_index(manifest)
    monkeypatch.setattr(module.v5, "sha256_file", lambda path: (_ for _ in ()).throw(AssertionError("unexpected read")))
    rows = module.cache_rows(
        cache, cache_name="legacy cache", source_digest_by_relative_path=digest_index
    )
    assert rows[0]["source_pixel_sha256"] == "digest-27"


def test_cross_root_overlap_is_removed_before_cv() -> None:
    legacy = {
        "features": torch.zeros(1, 384),
        "labels": torch.tensor([0]),
        "image_paths": ["/tmp/legacy/Elements/0001-a/03_04_22-27.bmp"],
        "class_names": {0: "a"},
    }
    external_output = "/repo/external/Elements/0001-a/ext.bmp"
    external = {
        "features": torch.ones(1, 384),
        "labels": torch.tensor([0]),
        "image_paths": [external_output],
        "class_names": {0: "a"},
    }
    snapshot = {
        "rows": [{
            "output_path": external_output,
            "source_path": "/mnt/f/source/Elements/0015-a/03_04_22-99.jpg",
            "source_pixel_sha256": "external-digest",
        }]
    }
    rows, audit = module.build_admissible_rows(
        legacy, external, snapshot, {"Elements/0001-a/03_04_22-27.bmp": "legacy-digest"}
    )
    assert len(rows) == 1
    assert audit["external_overlap_removed"] == 1

def test_runner_source_has_no_final_test_input() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "v4-strict-results" not in source
    assert "sealed_test" not in source
    assert '"promotion_eligible": False' in source
    assert '"runtime_exported": False' in source
