from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from backend.codex_pipeline.models.projection_head import ProjectionHead

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "autoresearch_elements_model.py"
spec = importlib.util.spec_from_file_location("autoresearch_elements_model", SCRIPT)
assert spec is not None
assert spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def _write_cache(
    path: Path,
    labels: list[int] | None = None,
    image_paths: list[str] | None = None,
) -> Path:
    if labels is None:
        labels = list(range(286))
    features = torch.zeros(len(labels), 384, dtype=torch.float32)
    payload: dict[str, object] = {
        "features": features,
        "labels": torch.tensor(labels, dtype=torch.long),
    }
    if image_paths is not None:
        payload["image_paths"] = image_paths
    torch.save(payload, path)
    return path


def _write_spec(path: Path, **overrides: object) -> Path:
    payload: dict[str, object] = {
        "objective": "proxy_distillation",
        "initialization": "random",
        "seed": 7,
        "epochs": 1,
        "temperature": 0.1,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "proxy_weight": 1.0,
        "teacher_weight": 0.0,
        "hidden_teacher_weight": 0.0,
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _write_runtime_projection(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ProjectionHead(384, 128).state_dict(), path)
    return path


def _paths(prefix: str, count: int = 286) -> list[str]:
    return [f"{prefix}-{index:03d}.png" for index in range(count)]


def test_refit_experiment_writes_runtime_compatible_checkpoint_and_provenance(tmp_path: Path, monkeypatch) -> None:
    spec_path = _write_spec(tmp_path / "recipe.json")
    train_cache = _write_cache(tmp_path / "elements_full.pt")
    output_dir = tmp_path / "refit-output"
    runtime_projection = _write_runtime_projection(tmp_path / "runtime" / "projection.pt")

    monkeypatch.setattr(module, "RUNTIME_PROJECTION", runtime_projection)
    monkeypatch.setattr(module, "train_proxy_or_distillation", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module,
        "classification_metrics",
        lambda *args, **kwargs: {"top1": 0.5, "macro_top1": 0.6, "top3": 0.7},
    )

    result = module.refit_experiment(spec_path, train_cache, output_dir, "cpu")

    assert result["schema_version"] == "autoresearch-refit.v1"
    assert result["runtime_compatible"] is True
    assert result["glyphs_allowed"] is False
    assert result["checkpoint_best_path"].endswith("checkpoints/best.pt")
    assert result["checkpoint_latest_path"].endswith("checkpoints/latest.pt")
    assert result["provenance_path"].endswith("refit_provenance.json")
    assert (output_dir / "checkpoints" / "latest.pt").is_file()
    assert (output_dir / "checkpoints" / "best.pt").is_file()

    provenance = json.loads((output_dir / "refit_provenance.json").read_text(encoding="utf-8"))
    assert provenance["train_cache_path"] == str(train_cache.resolve())
    assert provenance["class_count"] == 286
    assert provenance["train_top1"] == 0.5
    assert provenance["train_macro_top1"] == 0.6
    assert provenance["train_top3"] == 0.7
    assert provenance["teacher_assisted"] is False
    assert provenance["teacher_weight"] == 0.0
    assert provenance["hidden_teacher_weight"] == 0.0
    assert provenance["teacher_temperature"] is None
    assert provenance["runtime_projection_path"] == str(module.RUNTIME_PROJECTION.resolve())
    assert provenance["runtime_projection_sha256"] == module.sha256_file(module.RUNTIME_PROJECTION)
    assert "validation" not in provenance

    checkpoint = torch.load(output_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    assert checkpoint["validation_mode"] == "refit_full_cache_no_holdout"
    assert checkpoint["train_cache_path"] == str(train_cache.resolve())
    assert checkpoint["class_count"] == 286


def test_refit_experiment_records_distillation_provenance(tmp_path: Path, monkeypatch) -> None:
    spec_path = _write_spec(
        tmp_path / "distill.json",
        teacher_weight=0.7,
        hidden_teacher_weight=0.2,
        teacher_temperature=0.3,
    )
    train_cache = _write_cache(tmp_path / "elements_full.pt")
    output_dir = tmp_path / "refit-output"
    runtime_projection = _write_runtime_projection(tmp_path / "runtime" / "projection.pt")

    monkeypatch.setattr(module, "RUNTIME_PROJECTION", runtime_projection)
    monkeypatch.setattr(module, "train_proxy_or_distillation", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module,
        "classification_metrics",
        lambda *args, **kwargs: {"top1": 0.51, "macro_top1": 0.61, "top3": 0.71},
    )

    result = module.refit_experiment(spec_path, train_cache, output_dir, "cpu")

    assert result["teacher_assisted"] is True
    assert result["teacher_weight"] == 0.7
    assert result["hidden_teacher_weight"] == 0.2
    assert result["teacher_temperature"] == 0.3
    assert result["runtime_projection_path"] == str(module.RUNTIME_PROJECTION.resolve())
    assert result["runtime_projection_sha256"] == module.sha256_file(module.RUNTIME_PROJECTION)


def test_main_refit_uses_explicit_train_cache(tmp_path: Path, monkeypatch) -> None:
    spec_path = _write_spec(tmp_path / "recipe.json")
    train_cache = _write_cache(tmp_path / "elements_full.pt")
    output_dir = tmp_path / "refit-output"

    captured: dict[str, object] = {}

    def fake_refit(spec_arg: Path, train_cache_arg: Path, output_dir_arg: Path, device_arg: str):
        captured["spec"] = spec_arg
        captured["train_cache"] = train_cache_arg
        captured["output_dir"] = output_dir_arg
        captured["device"] = device_arg
        return {"ok": True}

    monkeypatch.setattr(module, "refit_experiment", fake_refit)

    exit_code = module.main(
        [
            "refit",
            "--spec",
            str(spec_path),
            "--train-cache",
            str(train_cache),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 0
    assert captured["spec"] == spec_path
    assert captured["train_cache"] == train_cache
    assert captured["output_dir"] == output_dir
    assert captured["device"] == "cpu"


def test_evaluate_cache_accepts_legacy_raw_state_dict_and_reports_metrics(tmp_path: Path, monkeypatch) -> None:
    checkpoint_path = tmp_path / "legacy_checkpoint.pt"
    prototype_cache = _write_cache(tmp_path / "prototype.pt", image_paths=_paths("prototype"))
    evaluation_cache = _write_cache(tmp_path / "evaluation.pt", image_paths=_paths("evaluation"))
    torch.save(ProjectionHead(384, 128).state_dict(), checkpoint_path)

    monkeypatch.setattr(
        module,
        "classification_metrics",
        lambda *args, **kwargs: {"top1": 0.42, "macro_top1": 0.5, "top3": 0.6},
    )

    result = module.evaluate_cache(checkpoint_path, prototype_cache, evaluation_cache, "cpu", 0.4)

    assert result["schema_version"] == "autoresearch-cache-eval.v1"
    assert result["pass"] is True
    assert result["score"] == 0.42
    assert result["train_prototype_top1"] == 0.42
    assert result["train_prototype_macro_top1"] == 0.5
    assert result["train_prototype_top3"] == 0.6
    assert result["prototype_count"] == 286
    assert result["evaluation_count"] == 286
    assert result["prototype_class_count"] == 286
    assert result["evaluation_class_count"] == 286
    assert result["prototype_evaluation_source_overlap_count"] == 0
    assert result["source_disjoint_enforced"] is True
    assert result["source_disjoint_check"] == "enforced"
    assert result["checkpoint_sha256"]
    assert result["prototype_cache_sha256"]
    assert result["evaluation_cache_sha256"]


def test_evaluate_cache_refuses_missing_image_paths_by_default(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "legacy_checkpoint.pt"
    prototype_cache = _write_cache(tmp_path / "prototype.pt")
    evaluation_cache = _write_cache(tmp_path / "evaluation.pt")
    torch.save(ProjectionHead(384, 128).state_dict(), checkpoint_path)

    with pytest.raises(ValueError, match="must expose image_paths"):
        module.evaluate_cache(checkpoint_path, prototype_cache, evaluation_cache, "cpu", 0.5)


def test_evaluate_cache_allows_missing_image_paths_when_explicitly_posthoc(tmp_path: Path, monkeypatch) -> None:
    checkpoint_path = tmp_path / "legacy_checkpoint.pt"
    prototype_cache = _write_cache(tmp_path / "prototype.pt")
    evaluation_cache = _write_cache(tmp_path / "evaluation.pt")
    torch.save(ProjectionHead(384, 128).state_dict(), checkpoint_path)

    monkeypatch.setattr(
        module,
        "classification_metrics",
        lambda *args, **kwargs: {"top1": 0.75, "macro_top1": 0.8, "top3": 0.9},
    )

    result = module.evaluate_cache(
        checkpoint_path,
        prototype_cache,
        evaluation_cache,
        "cpu",
        0.5,
        allow_source_overlap=True,
    )

    assert result["schema_version"] == "autoresearch-cache-eval.v1"
    assert result["pass"] is True
    assert result["prototype_evaluation_source_overlap_count"] is None
    assert result["source_disjoint_enforced"] is False
    assert result["source_disjoint_check"] == "unavailable"
    assert result["score"] == 0.75
    assert result["prototype_class_count"] == 286
    assert result["evaluation_class_count"] == 286
    assert result["train_prototype_top1"] == 0.75
    assert result["train_prototype_macro_top1"] == 0.8
    assert result["train_prototype_top3"] == 0.9


def test_evaluate_cache_refuses_asymmetric_missing_image_paths_by_default(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "legacy_checkpoint.pt"
    prototype_cache = _write_cache(tmp_path / "prototype.pt", image_paths=_paths("prototype"))
    evaluation_cache = _write_cache(tmp_path / "evaluation.pt")
    torch.save(ProjectionHead(384, 128).state_dict(), checkpoint_path)

    with pytest.raises(ValueError, match="must expose image_paths"):
        module.evaluate_cache(checkpoint_path, prototype_cache, evaluation_cache, "cpu", 0.5)


def test_evaluate_cache_refuses_relative_absolute_alias_overlap_by_default(tmp_path: Path, monkeypatch) -> None:
    checkpoint_path = tmp_path / "legacy_checkpoint.pt"
    source_file = tmp_path / "source-a.png"
    source_file.write_text("x", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    prototype_cache = _write_cache(
        tmp_path / "prototype.pt",
        [10, 20, 30],
        [str(source_file.resolve())],
    )
    evaluation_cache = _write_cache(
        tmp_path / "evaluation.pt",
        [10, 30],
        ["source-a.png"],
    )
    torch.save(ProjectionHead(384, 128).state_dict(), checkpoint_path)

    with pytest.raises(ValueError, match="share 1 source images"):
        module.evaluate_cache(checkpoint_path, prototype_cache, evaluation_cache, "cpu", 0.5)


def test_evaluate_cache_allows_relative_absolute_alias_overlap_when_explicitly_posthoc(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint_path = tmp_path / "legacy_checkpoint.pt"
    source_file = tmp_path / "source-a.png"
    source_file.write_text("x", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    prototype_cache = _write_cache(
        tmp_path / "prototype.pt",
        [10, 20, 30],
        [str(source_file.resolve())],
    )
    evaluation_cache = _write_cache(
        tmp_path / "evaluation.pt",
        [10, 30],
        ["source-a.png"],
    )
    torch.save(ProjectionHead(384, 128).state_dict(), checkpoint_path)

    monkeypatch.setattr(
        module,
        "classification_metrics",
        lambda *args, **kwargs: {"top1": 0.75, "macro_top1": 0.8, "top3": 0.9},
    )

    result = module.evaluate_cache(
        checkpoint_path,
        prototype_cache,
        evaluation_cache,
        "cpu",
        0.5,
        allow_source_overlap=True,
    )

    assert result["prototype_evaluation_source_overlap_count"] == 1
    assert result["source_disjoint_enforced"] is False
    assert result["source_disjoint_check"] == "allowed"
    assert result["pass"] is True
    assert result["score"] == 0.75


def test_main_evaluate_cache_uses_explicit_caches(tmp_path: Path, monkeypatch) -> None:
    checkpoint_path = tmp_path / "legacy_checkpoint.pt"
    prototype_cache = _write_cache(tmp_path / "prototype.pt", image_paths=_paths("prototype"))
    evaluation_cache = _write_cache(tmp_path / "evaluation.pt", image_paths=_paths("evaluation"))

    captured: dict[str, object] = {}

    def fake_evaluate_cache(
        checkpoint_arg: Path,
        prototype_cache_arg: Path,
        evaluation_cache_arg: Path,
        device_arg: str,
        min_score_arg: float,
        allow_source_overlap_arg: bool,
    ):
        captured["checkpoint"] = checkpoint_arg
        captured["prototype_cache"] = prototype_cache_arg
        captured["evaluation_cache"] = evaluation_cache_arg
        captured["device"] = device_arg
        captured["min_score"] = min_score_arg
        captured["allow_source_overlap"] = allow_source_overlap_arg
        return {"ok": True}

    monkeypatch.setattr(module, "evaluate_cache", fake_evaluate_cache)

    exit_code = module.main(
        [
            "evaluate-cache",
            "--checkpoint",
            str(checkpoint_path),
            "--prototype-cache",
            str(prototype_cache),
            "--evaluation-cache",
            str(evaluation_cache),
            "--device",
            "cpu",
            "--min-score",
            "0.4",
        ]
    )

    assert exit_code == 0
    assert captured["checkpoint"] == checkpoint_path
    assert captured["prototype_cache"] == prototype_cache
    assert captured["evaluation_cache"] == evaluation_cache
    assert captured["device"] == "cpu"
    assert captured["min_score"] == 0.4
    assert captured["allow_source_overlap"] is False


def test_main_evaluate_cache_allows_source_overlap_flag(tmp_path: Path, monkeypatch) -> None:
    checkpoint_path = tmp_path / "legacy_checkpoint.pt"
    source_file = tmp_path / "source-a.png"
    source_file.write_text("x", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    prototype_cache = _write_cache(
        tmp_path / "prototype.pt",
        [10, 20, 30],
        [str(source_file.resolve())],
    )
    evaluation_cache = _write_cache(
        tmp_path / "evaluation.pt",
        [10, 30],
        ["source-a.png"],
    )

    captured: dict[str, object] = {}

    def fake_evaluate_cache(
        checkpoint_arg: Path,
        prototype_cache_arg: Path,
        evaluation_cache_arg: Path,
        device_arg: str,
        min_score_arg: float,
        allow_source_overlap_arg: bool,
    ):
        captured["allow_source_overlap"] = allow_source_overlap_arg
        return {"ok": True}

    monkeypatch.setattr(module, "evaluate_cache", fake_evaluate_cache)

    exit_code = module.main(
        [
            "evaluate-cache",
            "--checkpoint",
            str(checkpoint_path),
            "--prototype-cache",
            str(prototype_cache),
            "--evaluation-cache",
            str(evaluation_cache),
            "--device",
            "cpu",
            "--min-score",
            "0.4",
            "--allow-source-overlap",
        ]
    )

    assert exit_code == 0
    assert captured["allow_source_overlap"] is True
