from __future__ import annotations

import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path

import pytest
import torch
import yaml

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from codex_pipeline.models.projection_head import ProjectionHead  # noqa: E402
from codex_pipeline.scripts import train  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state_dict() -> OrderedDict[str, torch.Tensor]:
    torch.manual_seed(17)
    return ProjectionHead(input_dim=4, embedding_dim=2).state_dict()


@pytest.mark.parametrize(
    ("nested", "expected_format"),
    [(False, "raw_state_dict"), (True, "checkpoint_model_state_dict")],
)
def test_load_initial_projection_accepts_raw_and_nested_state_dicts(
    tmp_path, nested, expected_format
):
    source = tmp_path / "projection.pt"
    expected = _state_dict()
    torch.save({"model_state_dict": expected, "epoch": 9} if nested else expected, source)
    model = ProjectionHead(input_dim=4, embedding_dim=2)

    provenance = train.load_initial_projection(model, source)

    assert provenance == {
        "path": str(source.resolve()),
        "sha256": _sha256(source),
        "format": expected_format,
    }
    for key, value in expected.items():
        assert torch.equal(model.state_dict()[key], value)


@pytest.mark.parametrize("failure", ["keys", "shape", "nonfinite"])
def test_load_initial_projection_rejects_incompatible_or_nonfinite_state(tmp_path, failure):
    source = tmp_path / f"bad-{failure}.pt"
    state = OrderedDict((key, value.clone()) for key, value in _state_dict().items())
    first_key = next(iter(state))
    if failure == "keys":
        state["unexpected"] = state.pop(first_key)
        match = "keys do not exactly match"
    elif failure == "shape":
        state[first_key] = state[first_key][:-1]
        match = "shape mismatch"
    else:
        state[first_key].view(-1)[0] = float("nan")
        match = "non-finite"
    torch.save(state, source)

    with pytest.raises(SystemExit, match=match):
        train.load_initial_projection(ProjectionHead(input_dim=4, embedding_dim=2), source)


def _write_eval_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "paths": {"checkpoint_dir": str(tmp_path / "unused")},
                "data": {"val_fraction": 0.25, "split_strategy": "per_class"},
                "model": {"embedding_dim": 2},
                "training": {
                    "episodes_per_epoch": 1,
                    "num_epochs": 0,
                    "n_way": 2,
                    "k_shot": 1,
                    "q_queries": 1,
                    "learning_rate": 1e-5,
                    "weight_decay": 1e-4,
                    "lr_scheduler": "cosine",
                    "warmup_epochs": 0,
                    "temperature": 0.1,
                    "device": "cpu",
                    "seed": 42,
                },
                "evaluation": {"num_eval_episodes": 1},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    common = {"hidden_dim": 4, "class_names": {0: "zero", 1: "one"}}
    train_features = tmp_path / "train.pt"
    validation_features = tmp_path / "dev.pt"
    torch.save(
        {
            **common,
            "features": torch.tensor(
                [[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0],
                 [0.0, 1.0, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0]]
            ),
            "labels": torch.tensor([0, 0, 1, 1]),
        },
        train_features,
    )
    torch.save(
        {
            **common,
            "features": torch.tensor([[0.8, 0.2, 0.0, 0.0], [0.2, 0.8, 0.0, 0.0]]),
            "labels": torch.tensor([0, 1]),
        },
        validation_features,
    )
    init_projection = tmp_path / "projection.pt"
    torch.save(_state_dict(), init_projection)
    return config, train_features, validation_features, init_projection


def test_eval_only_preserves_projection_and_writes_deterministic_aliases(tmp_path, monkeypatch):
    config, features, validation, init_projection = _write_eval_fixture(tmp_path)
    checkpoints = tmp_path / "checkpoints"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config", str(config),
            "--features", str(features),
            "--validation-features", str(validation),
            "--checkpoint-dir", str(checkpoints),
            "--init-projection", str(init_projection),
            "--eval-only",
        ],
    )

    train.main()

    best = torch.load(checkpoints / "best.pt", map_location="cpu", weights_only=True)
    latest = torch.load(checkpoints / "latest.pt", map_location="cpu", weights_only=True)
    expected = torch.load(init_projection, map_location="cpu", weights_only=True)
    assert (checkpoints / "best.pt").read_bytes() == (checkpoints / "latest.pt").read_bytes()
    assert best["epoch"] == latest["epoch"] == -1
    assert best["eval_only"] is True
    assert "optimizer_state_dict" not in best
    for key, value in expected.items():
        assert torch.equal(best["model_state_dict"][key], value)

    manifest_text = (checkpoints / "training_manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert manifest["epoch"] == -1
    assert manifest["eval_only"] is True
    assert manifest["init_projection"] == best["init_projection"]
    assert manifest["init_projection"]["sha256"] == _sha256(init_projection)
    assert manifest["checkpoints"]["best"] == manifest["checkpoints"]["latest"]

    # Re-running the exact descriptive evaluation has stable checkpoint and manifest bytes.
    checkpoint_sha = _sha256(checkpoints / "latest.pt")
    train.main()
    assert _sha256(checkpoints / "latest.pt") == checkpoint_sha
    assert (checkpoints / "training_manifest.json").read_text(encoding="utf-8") == manifest_text


def test_resume_and_init_projection_are_mutually_exclusive(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--resume", "resume.pt", "--init-projection", "projection.pt"],
    )
    with pytest.raises(SystemExit, match="2"):
        train.main()
    assert "not allowed with argument" in capsys.readouterr().err


def test_eval_only_requires_init_projection(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["train.py", "--eval-only"])
    with pytest.raises(SystemExit, match="2"):
        train.main()
    assert "--eval-only requires --init-projection" in capsys.readouterr().err


def test_training_checkpoint_saves_and_resume_restores_scheduler_states(tmp_path, monkeypatch):
    config, features, validation, init_projection = _write_eval_fixture(tmp_path)
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    payload["training"]["num_epochs"] = 2
    payload["training"]["warmup_epochs"] = 1
    config.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    checkpoints = tmp_path / "scheduler-checkpoints"
    common_args = [
        "train.py",
        "--config", str(config),
        "--features", str(features),
        "--validation-features", str(validation),
        "--checkpoint-dir", str(checkpoints),
    ]
    monkeypatch.setattr(
        sys,
        "argv",
        [*common_args, "--init-projection", str(init_projection)],
    )

    train.main()

    completed = torch.load(checkpoints / "latest.pt", map_location="cpu", weights_only=False)
    assert completed["scheduler_state_dict"]["last_epoch"] == 1
    assert completed["warmup_scheduler_state_dict"]["last_epoch"] == 1

    # Re-enter at epoch 1 with advanced scheduler states. A fresh scheduler
    # would finish at last_epoch=1, while the restored scheduler advances to 2.
    completed["epoch"] = 0
    resume = tmp_path / "resume.pt"
    torch.save(completed, resume)
    monkeypatch.setattr(sys, "argv", [*common_args, "--resume", str(resume)])
    train.main()

    resumed = torch.load(checkpoints / "latest.pt", map_location="cpu", weights_only=False)
    assert resumed["scheduler_state_dict"]["last_epoch"] == 2
    assert resumed["warmup_scheduler_state_dict"]["last_epoch"] == 1
