from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/autoresearch_vicreg_backbone_v19r2.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "autoresearch_vicreg_backbone_v19r2_tested", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_image_rows(tmp_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sentinel = tmp_path / "outer-fold-sentinel.png"
    Image.new("RGB", (32, 32), (255, 0, 0)).save(sentinel)
    rows.append(
        {
            "row_id": "outer-fold-sentinel",
            "image_path": str(sentinel),
            "fold": 1,
        }
    )
    for index in range(12):
        path = tmp_path / f"train-{index:02d}.png"
        Image.new("RGB", (32, 32), (index, index * 2, index * 3)).save(path)
        rows.append(
            {
                "row_id": f"train-{index:02d}",
                "image_path": str(path),
                "fold": 2,
            }
        )
    return rows


class RecordingBackbone(torch.nn.Module):
    def __init__(self, *, fail_on_call: int | None = None) -> None:
        super().__init__()
        self.fail_on_call = fail_on_call
        self.forward_shapes: list[tuple[int, ...]] = []

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        self.forward_shapes.append(tuple(batch.shape))
        if len(self.forward_shapes) == self.fail_on_call:
            raise RuntimeError("synthetic forward failure")
        return torch.zeros((batch.shape[0], 768), dtype=torch.float32, device=batch.device)


def test_smoke_and_precompute_share_fixed_extraction_geometry(monkeypatch) -> None:
    module = load_module()
    parser = module.build_parser()
    smoke = parser.parse_args(["smoke-extraction"])
    precompute = parser.parse_args(["precompute"])
    control = parser.parse_args(["replay-c1"])

    assert smoke.image_batch_size == precompute.image_batch_size == 4
    assert smoke.num_workers == precompute.num_workers == 0
    assert smoke.smoke_batches == 3
    assert precompute.freeze_audit == module.DEFAULT_PHASE2A_FREEZE_AUDIT
    assert precompute.authorization == module.DEFAULT_PHASE2A_AUTHORIZATION
    assert control.freeze_audit == module.DEFAULT_PHASE2B_FREEZE_AUDIT
    assert control.authorization == module.DEFAULT_PHASE2B_AUTHORIZATION
    assert control.freeze_audit != precompute.freeze_audit
    assert control.authorization != precompute.authorization
    assert module.validate_extraction_geometry(
        image_batch_size=4,
        num_workers=0,
        smoke_batches=3,
        smoke_fold=1,
    ) == {
        "row_batch_size": 4,
        "num_workers": 0,
        "views_including_base": 9,
        "effective_images_per_full_batch": 36,
        "smoke_batches": 3,
        "smoke_fold": 1,
    }

    calls: list[dict[str, int | None]] = []

    def stop_after_geometry(
        *,
        image_batch_size: int,
        num_workers: int,
        smoke_batches: int | None = None,
        smoke_fold: int | None = None,
    ) -> dict[str, int]:
        calls.append(
            {
                "image_batch_size": image_batch_size,
                "num_workers": num_workers,
                "smoke_batches": smoke_batches,
                "smoke_fold": smoke_fold,
            }
        )
        raise RuntimeError("geometry observed")

    authorization_phases: list[str] = []

    def observe_phase(unused, *, expected_phase: str):
        authorization_phases.append(expected_phase)
        return {"authorized": True}

    monkeypatch.setattr(module, "validate_phase_authorization", observe_phase)
    monkeypatch.setattr(module, "validate_extraction_geometry", stop_after_geometry)
    with pytest.raises(RuntimeError, match="geometry observed"):
        module.command_precompute(precompute)
    assert authorization_phases == ["phase_2a_extraction"]
    assert calls == [
        {
            "image_batch_size": 4,
            "num_workers": 0,
            "smoke_batches": None,
            "smoke_fold": None,
        }
    ]


def test_replay_c1_targets_only_phase_2b(monkeypatch) -> None:
    module = load_module()
    args = module.build_parser().parse_args(["replay-c1"])
    observed: list[str] = []

    def stop_at_authorization(unused, *, expected_phase: str):
        observed.append(expected_phase)
        raise RuntimeError("phase observed")

    monkeypatch.setattr(module, "validate_phase_authorization", stop_at_authorization)
    with pytest.raises(RuntimeError, match="phase observed"):
        module.command_replay_c1(args)

    assert observed == ["phase_2b_c1_control"]


@pytest.mark.parametrize(
    "override",
    [
        ["--authorization", "elsewhere.json"],
        ["--freeze-audit", "elsewhere.json"],
    ],
)
def test_phase_2b_authorization_rejects_path_drift(override: list[str]) -> None:
    module = load_module()
    args = module.build_parser().parse_args(["replay-c1", *override])

    with pytest.raises(ValueError, match="phase 2b"):
        module.validate_phase_authorization(
            args,
            expected_phase="phase_2b_c1_control",
        )


def test_precompute_propagates_fixed_geometry_to_real_v9_extractor(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = load_module()
    args = module.build_parser().parse_args(
        [
            "precompute",
            "--device",
            "cpu",
            "--b14-cache-dir",
            str(tmp_path / "cache"),
            "--cache-manifest",
            str(tmp_path / "cache-manifest.json"),
        ]
    )
    calls: list[tuple[int, int, int]] = []

    monkeypatch.setattr(
        module,
        "validate_phase_authorization",
        lambda *args, **kwargs: {"authorized": True},
    )
    monkeypatch.setattr(
        module,
        "consume_authorization",
        lambda *args, **kwargs: {"path": "claim.json", "sha256": "claim"},
    )
    monkeypatch.setattr(module, "validate_static_contract", lambda unused: {})
    monkeypatch.setattr(module, "assert_ext4_workspace_path", lambda unused: {})
    monkeypatch.setattr(
        module.v9,
        "load_corpus",
        lambda *args, **kwargs: ([{}] * 9990, {}),
    )
    monkeypatch.setattr(module.v9, "resolve_device", lambda unused: torch.device("cpu"))
    monkeypatch.setattr(
        module.backbone_screen,
        "validate_backbone_manifest",
        lambda unused: {},
    )
    monkeypatch.setattr(
        module.v9,
        "load_backbone",
        lambda *args, **kwargs: (torch.nn.Identity(), {}),
    )
    monkeypatch.setattr(module, "load_canonical_cache", lambda *args, **kwargs: {})

    def precompute_spy(*args, fold: int, batch_size: int, num_workers: int, **kwargs):
        calls.append((fold, batch_size, num_workers))
        return tmp_path / f"fold-{fold}.pt"

    monkeypatch.setattr(module.v9, "precompute_fold", precompute_spy)
    monkeypatch.setattr(
        module,
        "validate_b14_cache",
        lambda path, *, fold, **kwargs: (
            {},
            {"fold": fold, "train_rows": 4, "oof_rows": 1, "cache_bytes": 1},
        ),
    )
    monkeypatch.setattr(module, "EXPECTED_TRAIN_ROWS_ACROSS_FOLDS", 20)
    monkeypatch.setattr(module, "EXPECTED_UNIQUE_OOF_ROWS", 5)
    monkeypatch.setattr(
        module,
        "runtime_hashes",
        lambda unused: dict(module.EXPECTED_RUNTIME_SHA256),
    )

    result = module.command_precompute(args)

    assert result["pass"] is True
    assert calls == [(fold, 4, 0) for fold in module.EXPECTED_FOLDS]


@pytest.mark.parametrize(
    "override",
    [
        ["--output", "elsewhere.json"],
        ["--manifest", "elsewhere.jsonl"],
        ["--audit", "elsewhere.json"],
        ["--smoke-fold", "2"],
        ["--device", "cpu"],
    ],
)
def test_smoke_authorization_rejects_command_drift(override: list[str]) -> None:
    module = load_module()
    args = module.build_parser().parse_args(
        [
            "smoke-extraction",
            "--authorization",
            str(module.DEFAULT_SMOKE_AUTHORIZATION),
            *override,
        ]
    )

    with pytest.raises(ValueError):
        module.validate_smoke_authorization(args)


def test_authorization_claim_is_durable_and_non_replayable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = load_module()
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_text("{}\n", encoding="utf-8")
    claim_path = tmp_path / "authorizations" / "claim.json"
    args = module.build_parser().parse_args(
        ["smoke-extraction", "--authorization", str(authorization_path)]
    )
    authorization = {
        "execution_claim_path": str(claim_path),
        "command_contract_sha256": "frozen-command",
    }
    monkeypatch.setattr(module, "RUN", tmp_path)

    first = module.consume_authorization(args, authorization)

    assert first["sha256"] == module.v9.sha256_file(claim_path)
    with pytest.raises(FileExistsError, match="already consumed"):
        module.consume_authorization(args, authorization)


def test_deadline_must_be_exactly_six_hours_and_unexpired() -> None:
    module = load_module()
    started = datetime.now(timezone.utc) - timedelta(minutes=1)

    valid = module.validate_deadline(
        {
            "started_at": started.isoformat(),
            "deadline_at": (started + timedelta(hours=6)).isoformat(),
            "max_runtime_seconds": 21_600,
        }
    )
    assert valid["max_runtime_seconds"] == 21_600

    with pytest.raises(ValueError, match="exactly six hours"):
        module.validate_deadline(
            {
                "started_at": started.isoformat(),
                "deadline_at": (started + timedelta(hours=5)).isoformat(),
                "max_runtime_seconds": 21_600,
            }
        )
    with pytest.raises(TimeoutError, match="expired"):
        module.validate_deadline(
            {
                "started_at": (started - timedelta(hours=7)).isoformat(),
                "deadline_at": (started - timedelta(hours=1)).isoformat(),
                "max_runtime_seconds": 21_600,
            }
        )


def test_real_cpu_dataset_completes_three_full_36_image_forwards(
    tmp_path: Path,
) -> None:
    module = load_module()
    rows = make_image_rows(tmp_path)
    backbone = RecordingBackbone()

    result = module.run_smoke_pipeline(
        backbone,
        rows,
        fold=1,
        image_batch_size=4,
        num_workers=0,
        smoke_batches=3,
        device=torch.device("cpu"),
    )

    assert backbone.forward_shapes == [(36, 3, 224, 224)] * 3
    assert result["base_feature_shape"] == [12, 768]
    assert result["view_feature_shape"] == [12, 8, 768]
    assert result["source_image_reads"] == 12
    assert result["oof_source_image_reads"] == 0
    assert result["candidate_predictions"] == 0
    assert result["candidate_scores"] == 0
    assert result["persistent_feature_cache"] is False
    assert not any(path.suffix in {".pt", ".pth"} for path in tmp_path.iterdir())


def test_real_cpu_dataset_stops_on_first_failed_forward_without_fallback(
    tmp_path: Path,
) -> None:
    module = load_module()
    rows = make_image_rows(tmp_path)
    backbone = RecordingBackbone(fail_on_call=2)

    with pytest.raises(RuntimeError, match="synthetic forward failure"):
        module.run_smoke_pipeline(
            backbone,
            rows,
            fold=1,
            image_batch_size=4,
            num_workers=0,
            smoke_batches=3,
            device=torch.device("cpu"),
        )

    assert backbone.forward_shapes == [(36, 3, 224, 224)] * 2
    assert not any(path.suffix in {".pt", ".pth"} for path in tmp_path.iterdir())


def test_command_records_terminal_failure_and_never_retries(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = load_module()
    output = tmp_path / "smoke-audit.json"
    cache_dir = tmp_path / "cache"
    args = module.build_parser().parse_args(
        [
            "smoke-extraction",
            "--authorization",
            str(tmp_path / "authorization.json"),
            "--b14-cache-dir",
            str(cache_dir),
            "--output",
            str(output),
        ]
    )
    backbone = RecordingBackbone()

    monkeypatch.setattr(
        module,
        "validate_smoke_authorization",
        lambda unused: {
            "authorized": True,
            "runtime_sha256": dict(module.EXPECTED_RUNTIME_SHA256),
        },
    )
    monkeypatch.setattr(
        module,
        "consume_authorization",
        lambda *args, **kwargs: {"path": "claim.json", "sha256": "claim"},
    )
    monkeypatch.setattr(
        module,
        "load_smoke_corpus",
        lambda *args, **kwargs: (
            [{}] * 9990,
            {"model_labels_read": 0, "model_labels_computed": 0},
        ),
    )
    monkeypatch.setattr(
        module,
        "runtime_hashes",
        lambda unused: dict(module.EXPECTED_RUNTIME_SHA256),
    )
    monkeypatch.setattr(module.v9, "resolve_device", lambda unused: torch.device("cuda"))
    monkeypatch.setattr(
        module.v9,
        "load_backbone",
        lambda *args, **kwargs: (backbone, {"backbone": "synthetic"}),
    )
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        kwargs["opened_paths"].extend(
            Path(f"opened-{index}.png") for index in range(4)
        )
        raise RuntimeError("synthetic pipeline failure")

    monkeypatch.setattr(module, "run_smoke_pipeline", fail_once)
    monkeypatch.setattr(module.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(module.torch.cuda, "reset_peak_memory_stats", lambda unused: None)
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)

    result = module.command_smoke_extraction(args)

    assert calls == 1
    assert result["pass"] is False
    assert result["status"] == "terminal_failed_pipeline_smoke"
    assert result["retry_allowed"] is False
    assert result["phase_2_allowed"] is False
    assert result["failure"]["message"] == "synthetic pipeline failure"
    assert result["cache_files"] == []
    assert result["operation_counts"]["candidate_predictions"] == 0
    assert result["operation_counts"]["source_image_reads"] == 4
    assert json.loads(output.read_text(encoding="utf-8")) == result


def test_prior_v19_failure_and_runner_remain_immutable() -> None:
    module = load_module()

    assert module.v9.sha256_file(
        module.SCRIPTS_DIR / "autoresearch_vicreg_backbone_v19.py"
    ) == "14066b79538a5d3176bdf76f3a6475235da91480faf86a758c328234c6b60219"
    assert (
        module.v9.sha256_file(module.PRIOR_V19_FAILURE_AUDIT)
        == "6cfff7acdce3af4b8ef080bdf21dcf54788379c479a9f0f4322283d77f27a47b"
    )


def test_phase3_cli_defaults_and_contracts_are_distinct() -> None:
    module = load_module()
    train = module.build_parser().parse_args(["train-evaluate"])
    replay = module.build_parser().parse_args(["replay"])

    assert train.freeze_audit == module.DEFAULT_PHASE3A_FREEZE_AUDIT
    assert train.authorization == module.DEFAULT_PHASE3A_AUTHORIZATION
    assert replay.freeze_audit == module.DEFAULT_PHASE3B_FREEZE_AUDIT
    assert replay.authorization == module.DEFAULT_PHASE3B_AUTHORIZATION
    assert train.authorization != replay.authorization
    assert module.phase_command_contract(train)["command"] == "train-evaluate"
    assert module.phase_command_contract(replay)["command"] == "replay"
    assert (
        module.v9.sha256_json(module.phase_command_contract(train))
        != module.v9.sha256_json(module.phase_command_contract(replay))
    )


@pytest.mark.parametrize(
    ("command", "override", "phase"),
    [
        ("train-evaluate", ["--authorization", "elsewhere.json"], "phase 3a"),
        ("train-evaluate", ["--freeze-audit", "elsewhere.json"], "phase 3a"),
        ("replay", ["--authorization", "elsewhere.json"], "phase 3b"),
        ("replay", ["--freeze-audit", "elsewhere.json"], "phase 3b"),
    ],
)
def test_phase3_authorizations_reject_path_drift(
    command: str,
    override: list[str],
    phase: str,
) -> None:
    module = load_module()
    args = module.build_parser().parse_args([command, *override])
    expected_phase = (
        "phase_3a_train_evaluate" if command == "train-evaluate" else "phase_3b_replay"
    )

    with pytest.raises(ValueError, match=phase):
        module.validate_phase_authorization(args, expected_phase=expected_phase)


def test_phase3_claim_deadline_is_derived_at_consumption(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = load_module()
    run = tmp_path / "run"
    authorization_dir = run / "authorizations"
    authorization_dir.mkdir(parents=True)
    authorization_path = authorization_dir / "phase-3a.json"
    claim_path = authorization_dir / "phase-3a.execution-claim.json"
    authorization_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(module, "RUN", run)
    args = module.build_parser().parse_args(
        ["train-evaluate", "--authorization", str(authorization_path)]
    )
    authorization = {
        "execution_claim_path": str(claim_path),
        "command_contract_sha256": "frozen-contract",
        "deadline_policy": module.PHASE3_DEADLINE_POLICY,
    }

    record = module.consume_authorization(args, authorization)
    payload = json.loads(claim_path.read_text(encoding="utf-8"))
    started_at = datetime.fromisoformat(payload["started_at"])
    deadline_at = datetime.fromisoformat(payload["deadline_at"])

    assert record["deadline_policy"] == module.PHASE3_DEADLINE_POLICY
    assert int((deadline_at - started_at).total_seconds()) == module.MAX_RUNTIME_SECONDS
    assert payload["max_runtime_seconds"] == module.MAX_RUNTIME_SECONDS
    module.enforce_execution_claim_deadline(record)


def test_train_evaluate_consumes_only_phase3a_claim(monkeypatch) -> None:
    module = load_module()
    args = module.build_parser().parse_args(["train-evaluate"])
    observed: dict[str, Any] = {}

    def validate(unused, *, expected_phase: str):
        observed["phase"] = expected_phase
        return {"authorization": "3a"}

    def consume(unused, authorization):
        observed["consumed"] = authorization
        return {"claim": "3a"}

    def run_core(unused, **kwargs):
        observed["core"] = kwargs
        return {"pass": False}

    monkeypatch.setattr(module, "validate_phase_authorization", validate)
    monkeypatch.setattr(module, "consume_authorization", consume)
    monkeypatch.setattr(module, "run_candidate_evaluation", run_core)

    result = module.command_train_evaluate(args)

    assert result == {"pass": False}
    assert observed["phase"] == "phase_3a_train_evaluate"
    assert observed["consumed"] == {"authorization": "3a"}
    assert observed["core"] == {
        "recorded_authorization": {"authorization": "3a"},
        "recorded_execution_claim": {"claim": "3a"},
        "deadline_claim": {"claim": "3a"},
    }


def test_replay_rejects_invalid_handoff_before_consuming_phase3b(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = load_module()
    args = module.build_parser().parse_args(
        [
            "replay",
            "--output-dir",
            str(tmp_path / "original"),
            "--replay-dir",
            str(tmp_path / "replay"),
        ]
    )
    consumed = False

    monkeypatch.setattr(
        module,
        "validate_phase_authorization",
        lambda *args, **kwargs: {"authorization": "3b"},
    )

    def reject_handoff(unused):
        raise ValueError("handoff drift")

    def consume(*args, **kwargs):
        nonlocal consumed
        consumed = True
        return {"claim": "3b"}

    monkeypatch.setattr(module, "validate_phase3a_handoff", reject_handoff)
    monkeypatch.setattr(module, "consume_authorization", consume)

    with pytest.raises(ValueError, match="handoff drift"):
        module.command_replay(args)

    assert consumed is False


def test_replay_consumes_phase3b_then_calls_nonrecursive_core(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = load_module()
    replay_dir = tmp_path / "replay"
    args = module.build_parser().parse_args(
        [
            "replay",
            "--output-dir",
            str(tmp_path / "original"),
            "--replay-dir",
            str(replay_dir),
        ]
    )
    observed: dict[str, Any] = {}
    original = {
        "phase_authorization": {"authorization": "3a"},
        "execution_claim": {"claim": "3a"},
    }

    def validate(unused, *, expected_phase: str):
        observed["phase"] = expected_phase
        return {"authorization": "3b"}

    def consume(unused, authorization):
        observed["consumed"] = authorization
        return {"claim": "3b"}

    def core(core_args, **kwargs):
        observed["output_dir"] = core_args.output_dir
        observed["core"] = kwargs
        raise RuntimeError("nonrecursive core reached")

    monkeypatch.setattr(module, "validate_phase_authorization", validate)
    monkeypatch.setattr(
        module,
        "validate_phase3a_handoff",
        lambda unused: {"original": original},
    )
    monkeypatch.setattr(module, "consume_authorization", consume)
    monkeypatch.setattr(module, "run_candidate_evaluation", core)
    monkeypatch.setattr(
        module,
        "command_train_evaluate",
        lambda unused: pytest.fail("replay must not call the train wrapper"),
    )

    with pytest.raises(RuntimeError, match="nonrecursive core reached"):
        module.command_replay(args)

    assert observed["phase"] == "phase_3b_replay"
    assert observed["consumed"] == {"authorization": "3b"}
    assert observed["output_dir"] == replay_dir
    assert observed["core"] == {
        "recorded_authorization": {"authorization": "3a"},
        "recorded_execution_claim": {"claim": "3a"},
        "deadline_claim": {"claim": "3b"},
    }


def test_scientific_verdict_is_classified_only_from_replayed_evaluation(
    monkeypatch,
) -> None:
    module = load_module()
    observed: dict[str, Any] = {}
    evaluation = {
        "integrity_pass": True,
        "engagement_pass": True,
        "b0_mission_anchor_pass": True,
        "c1_noninferiority_pass": True,
        "candidate_minus_c1": {
            "overall": {"delta_top1": 0.02},
            "bootstrap": {"delta_top1_95": [0.01, 0.03]},
            "positive_seed_count": 3,
        },
    }

    def classify(**kwargs):
        observed.update(kwargs)
        return "supported_strong", "candidate is better"

    monkeypatch.setattr(module.v17, "classify_verdict", classify)

    verdict = module.classify_candidate_verdict(evaluation)

    assert verdict == ("supported_strong", "candidate is better")
    assert observed["integrity_pass"] is True
    assert observed["engagement_pass"] is True
    assert observed["c1_delta_top1"] == 0.02
    assert observed["c1_bootstrap_lower"] == 0.01
    assert observed["c1_positive_seed_count"] == 3


def test_prior_v19r_terminal_artifacts_remain_immutable() -> None:
    module = load_module()

    assert (
        module.v9.sha256_file(module.PRIOR_V19R1_RUNNER)
        == module.EXPECTED_PRIOR_V19R1_RUNNER_SHA256
    )
    assert (
        module.v9.sha256_file(module.PRIOR_V19R1_TEST)
        == module.EXPECTED_PRIOR_V19R1_TEST_SHA256
    )
    assert (
        module.v9.sha256_file(module.PRIOR_V19R1_FAILURE_AUDIT)
        == module.EXPECTED_PRIOR_V19R1_FAILURE_AUDIT_SHA256
    )
    assert (
        module.v9.sha256_file(module.PRIOR_V19R1_ROOT_CAUSE_AUDIT)
        == module.EXPECTED_PRIOR_V19R1_ROOT_CAUSE_AUDIT_SHA256
    )
    assert (
        module.v9.sha256_file(module.PRIOR_V19R1_CLAIM)
        == module.EXPECTED_PRIOR_V19R1_CLAIM_SHA256
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA regression requires CUDA")
def test_cuda_embedding_snapshot_preserves_values_and_autograd() -> None:
    module = load_module()
    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(20260805)
    features = torch.randn((17, module.EMBED_DIM), generator=generator)

    reference, _ = module.aligned_candidate_initialization(17)
    candidate, _ = module.aligned_candidate_initialization(17)
    reference = reference.to(device)
    with torch.inference_mode():
        expected = module.v9.embed_in_batches(reference, features, device)

    actual = module.embedding_snapshot(
        candidate,
        features,
        device=device,
        limit=len(features),
    )

    assert torch.equal(actual, expected)
    assert all(not parameter.is_inference() for parameter in candidate.parameters())

    candidate.train()
    projected = candidate.net(features.to(device))
    loss = projected.square().mean()
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in candidate.parameters()
        if parameter.requires_grad
    ]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_embedding_snapshot_moves_model_before_inference_context() -> None:
    module = load_module()
    source = Path(module.__file__).read_text(encoding="utf-8")
    function = source[
        source.index("def embedding_snapshot(") : source.index("def support_bin_three(")
    ]

    assert function.index("model = model.to(device)") < function.index(
        "with torch.inference_mode():"
    )
    assert function.count("model = model.to(device)") == 1
