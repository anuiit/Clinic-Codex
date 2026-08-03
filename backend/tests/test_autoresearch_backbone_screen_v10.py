from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
for name in ("autoresearch_self_supervised_v9", "autoresearch_raw_feature_diagnostic_v10"):
    script = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)

SCRIPT = ROOT / "scripts" / "autoresearch_backbone_screen_v10.py"
SPEC = importlib.util.spec_from_file_location("autoresearch_backbone_screen_v10", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_backbone_classification_uses_strict_interval_rules() -> None:
    assert module.classify_backbone_interval([0.001, 0.2]) == "B14_raw_superior"
    assert module.classify_backbone_interval([-0.2, -0.001]) == "S14_raw_superior"
    assert module.classify_backbone_interval([-0.2, 0.001]) == "neutral_or_inconclusive"


def test_arm_seed_variant_report_requires_deterministic_duplicates() -> None:
    records = [
        {
            "outer_fold": 1,
            "row_id": "row-1",
            "seed": seed,
            "baseline_topk": [1, 2, 3],
            "candidate_topk": [4, 5, 6],
        }
        for seed in (17, 42, 73)
    ]
    baseline = module.arm_seed_variant_report(records, field="baseline_topk", expected_seeds=[17, 42, 73])
    candidate = module.arm_seed_variant_report(records, field="candidate_topk", expected_seeds=[17, 42, 73])

    assert baseline["maximum_unique_raw_prediction_variants_per_oof_row"] == 1
    assert candidate["maximum_unique_raw_prediction_variants_per_oof_row"] == 1
    assert baseline["missing_or_extra_seed_groups"] == []


def test_validate_backbone_manifest_checks_critical_files_and_weight_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "dinov2"
    (repository / "dinov2/models").mkdir(parents=True)
    files = {
        "hubconf_sha256": repository / "hubconf.py",
        "vision_transformer_sha256": repository / "dinov2/models/vision_transformer.py",
        "license_sha256": repository / "LICENSE",
    }
    for index, path in enumerate(files.values()):
        path.write_text(f"content-{index}\n", encoding="utf-8")
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"weights")
    weight_sha = module.v9.sha256_file(weights)
    monkeypatch.setattr(module, "EXPECTED_WEIGHTS_SHA256", weight_sha)
    manifest = {
        "schema_version": "dinov2-local-pin.v1",
        "backbone": "dinov2_vitb14",
        "repository_path": str(repository),
        "weights_path": str(weights),
        "weights_bytes": weights.stat().st_size,
        "weights_sha256": weight_sha,
        **{key: module.v9.sha256_file(path) for key, path in files.items()},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = module.validate_backbone_manifest(manifest_path)

    assert result["weights_sha256_verified"] == weight_sha
    assert result["critical_source_hashes_verified"] is True


def _write_feature_cache(path: Path, *, manifest_sha256: str) -> None:
    payload = {
        "schema_version": "autoresearch-self-supervised-v10.global-base-feature-cache",
        "base_features": torch.ones(2, 3, dtype=torch.float16),
        "row_id": ["row-1", "row-2"],
        "class_label": [1, 2],
        "class_name": ["one", "two"],
        "component_id": ["component-1", "component-2"],
        "decoded_pixel_sha256": ["pixel-1", "pixel-2"],
        "fold": [1, 2],
        "provenance": {
            "backbone_manifest_sha256": manifest_sha256,
            "weights_sha256": module.EXPECTED_WEIGHTS_SHA256,
        },
    }
    torch.save(payload, path)
    sidecar = {
        "cache_sha256": module.v9.sha256_file(path),
        "row_ids_sha256": module.v9.sha256_json(payload["row_id"]),
    }
    module.feature_cache_sidecar(path).write_text(json.dumps(sidecar), encoding="utf-8")


def test_feature_cache_validation_checks_shape_rows_and_sidecar(tmp_path: Path) -> None:
    path = tmp_path / "features.pt"
    _write_feature_cache(path, manifest_sha256="manifest-hash")

    payload = module.load_feature_cache(
        path,
        expected_manifest_sha256="manifest-hash",
        expected_rows=2,
        expected_dim=3,
    )

    assert payload["_cache_validation"]["rows"] == 2
    assert payload["_cache_validation"]["embedding_dim"] == 3


def test_feature_cache_validation_rejects_sidecar_hash_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "features.pt"
    _write_feature_cache(path, manifest_sha256="manifest-hash")
    sidecar_path = module.feature_cache_sidecar(path)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["cache_sha256"] = "wrong"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    with pytest.raises(ValueError, match="sidecar"):
        module.load_feature_cache(
            path,
            expected_manifest_sha256="manifest-hash",
            expected_rows=2,
            expected_dim=3,
        )
