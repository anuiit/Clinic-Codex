from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from backend.services.model_registry import ModelRegistry, sha256_file
from scripts import evaluate_r2_e2e as e2e


REPO_ROOT = Path(__file__).resolve().parents[2]
R2_VERSION = "20260805T112535Z-vicreg-full-data-r2"
E2E_SPEC = REPO_ROOT / "backend" / "model_registry" / "specs" / "r2-e2e-promotion-v1.json"
BUILD_SPEC = REPO_ROOT / "backend" / "model_registry" / "specs" / "r2-full-data-production-v1.json"


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _promotion_contract() -> dict[str, object]:
    return {
        "e2e_report_required": True,
        "e2e_spec_path": "backend/model_registry/specs/r2-e2e-promotion-v1.json",
        "e2e_spec_sha256": sha256_file(E2E_SPEC),
        "build_spec_path": "backend/model_registry/specs/r2-full-data-production-v1.json",
        "build_spec_sha256": sha256_file(BUILD_SPEC),
    }


def _write_manifest(path: Path) -> Path:
    return _write_json(
        path,
        {
            "schema_version": 1,
            "model_id": "codex_classifier",
            "version_id": R2_VERSION,
            "status": "candidate",
            "promotion": _promotion_contract(),
        },
    )


def _paired_evidence(tmp_path: Path) -> tuple[Path, Path, Path]:
    rows = []
    for index in range(600):
        truth = index % 100
        runtime_wrong = index % 5 == 0
        runtime_label = (truth + 1) % 100 if runtime_wrong else truth
        rows.append(
            {
                "sample_id": f"eval-{index}",
                "pixel_sha256": f"eval-pixel-{index}",
                "case_id": f"eval-case-{index // 6}",
                "class_label": truth,
                "runtime": {
                    "class_label": runtime_label,
                    "confidence": 0.9,
                    "latency_ms": 10.0,
                    "top_k": [runtime_label],
                },
                "candidate": {
                    "class_label": truth,
                    "confidence": 0.99,
                    "latency_ms": 15.0,
                    "top_k": [truth, (truth + 1) % 100, (truth + 2) % 100],
                },
            }
        )
    predictions = _write_json(tmp_path / "paired.json", {"rows": rows})
    provenance = _write_json(
        tmp_path / "training.json",
        {
            "pixel_hashes": [f"train-pixel-{index}" for index in range(600)],
            "case_ids": [f"train-case-{index}" for index in range(100)],
        },
    )
    operational = _write_json(
        tmp_path / "operational.json",
        {"shadow": {"requests": 500, "candidate_errors": 0}},
    )
    return predictions, provenance, operational


def test_evaluate_r2_e2e_is_deterministic_and_passes_all_gates(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path / "manifest.json")
    predictions, provenance, operational = _paired_evidence(tmp_path)

    first = e2e.evaluate_r2_e2e(
        predictions_path=predictions,
        training_provenance_path=provenance,
        candidate_manifest_path=manifest_path,
        operational_evidence_path=operational,
        bootstrap_replicates=200,
    )
    second = e2e.evaluate_r2_e2e(
        predictions_path=predictions,
        training_provenance_path=provenance,
        candidate_manifest_path=manifest_path,
        operational_evidence_path=operational,
        bootstrap_replicates=200,
    )

    assert first == second
    assert first["overall_pass"] is True
    assert all(first["gates"].values())
    assert first["dataset"]["pixel_hash_overlap_count"] == 0
    assert first["dataset"]["case_or_specimen_overlap_count"] == 0
    assert abs(first["comparison"]["top1_delta"] - 0.2) < 1e-12
    assert first["performance"]["p95_latency_ratio"] == 1.5
    assert first["comparison"]["bootstrap_unit"] == "case"
    assert first["integrity"]["report_payload_sha256"] == e2e.report_payload_sha256(first)


def _write_runtime(runtime_dir: Path, label: str) -> None:
    (runtime_dir / "weights").mkdir(parents=True, exist_ok=True)
    (runtime_dir / "weights" / "prototypes.pt").write_bytes(f"{label}-prototypes".encode())
    (runtime_dir / "weights" / "projection.pt").write_bytes(f"{label}-projection".encode())
    _write_json(runtime_dir / "config.json", {"model_version": label, "class_names": [label]})


def _guarded_registry(tmp_path: Path) -> tuple[ModelRegistry, Path]:
    runtime = tmp_path / "backend" / "codex_model"
    _write_runtime(runtime, "original")
    source = tmp_path / "candidate-source"
    _write_runtime(source, "candidate")
    registry = ModelRegistry(
        tmp_path / "backend" / "model_registry",
        repo_root=tmp_path,
        runtime_model_dir=runtime,
    )
    registry.create_version_from_artifacts(
        R2_VERSION,
        artifact_sources={
            "runtime/weights/prototypes.pt": source / "weights" / "prototypes.pt",
            "runtime/weights/projection.pt": source / "weights" / "projection.pt",
            "runtime/config.json": source / "config.json",
        },
        metadata={"promotion": _promotion_contract()},
    )
    return registry, runtime


def _passing_report(registry: ModelRegistry, tmp_path: Path) -> Path:
    manifest_path = registry.version_dir(R2_VERSION) / "manifest.json"
    spec = json.loads(E2E_SPEC.read_text(encoding="utf-8"))
    report = {
        "schema_version": e2e.REPORT_SCHEMA_VERSION,
        "candidate_version_id": R2_VERSION,
        "bindings": {
            "evaluation_spec": {"sha256": sha256_file(E2E_SPEC)},
            "build_spec": {"sha256": sha256_file(BUILD_SPEC)},
            "candidate_manifest": {"sha256": sha256_file(manifest_path)},
            "paired_predictions": {"sha256": "1" * 64},
            "training_provenance": {"sha256": "2" * 64},
            "operational_evidence": {"sha256": "3" * 64},
        },
        "dataset": {
            "rows": 600,
            "classes": 100,
            "pixel_hashes_complete_and_unique": True,
            "pixel_disjoint_verified": True,
            "pixel_hash_overlap_count": 0,
            "case_or_specimen_available": True,
            "case_or_specimen_kind": "case",
            "case_or_specimen_groups": 100,
            "case_or_specimen_disjoint_verified": True,
            "case_or_specimen_overlap_count": 0,
        },
        "models": {},
        "comparison": {
            "top1_delta": 0.02,
            "macro_top1_delta": 0.01,
            "top3_delta": 0.01,
            "top1_paired_bootstrap_95": [0.01, 0.03],
            "mcnemar_exact_p": 0.01,
        },
        "rejection": {
            "candidate_expected_calibration_error": 0.05,
            "accepted_accuracy_delta": 0.01,
            "rejection_rate_absolute_delta": 0.01,
        },
        "performance": {"p95_latency_ratio": 1.5, "candidate_error_rate": 0.0},
        "operational": {"pass": True},
        "gates": {},
        "overall_pass": False,
        "integrity": {"algorithm": "sha256-canonical-json-v1"},
    }
    report["gates"] = e2e._compute_gates(report, spec)
    report["overall_pass"] = all(report["gates"].values())
    e2e.seal_report(report)
    return _write_json(tmp_path / "e2e-report.json", report)


def _run_promote(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "scripts/promote_model.py",
            R2_VERSION,
            *args,
            "--registry-dir",
            str(tmp_path / "backend" / "model_registry"),
            "--runtime-dir",
            str(tmp_path / "backend" / "codex_model"),
            "--json",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )


def test_r2_promotion_rejects_missing_report(tmp_path: Path) -> None:
    _guarded_registry(tmp_path)
    result = _run_promote(tmp_path, "--dry-run")
    assert result.returncode == 2
    assert "requires a passing E2E report" in json.loads(result.stdout)["error"]


def test_r2_promotion_rejects_tampered_report(tmp_path: Path) -> None:
    registry, _ = _guarded_registry(tmp_path)
    report_path = _passing_report(registry, tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["comparison"]["top1_delta"] = 0.9
    _write_json(report_path, report)
    result = _run_promote(tmp_path, "--dry-run", "--e2e-report", str(report_path))
    assert result.returncode == 2
    assert "payload hash mismatch" in json.loads(result.stdout)["error"]


def test_r2_promotion_rejects_spec_hash_mismatch(tmp_path: Path) -> None:
    registry, _ = _guarded_registry(tmp_path)
    report_path = _passing_report(registry, tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["bindings"]["evaluation_spec"]["sha256"] = "0" * 64
    e2e.seal_report(report)
    _write_json(report_path, report)
    result = _run_promote(tmp_path, "--dry-run", "--e2e-report", str(report_path))
    assert result.returncode == 2
    assert "evaluation_spec hash mismatch" in json.loads(result.stdout)["error"]


def test_r2_promotion_rejects_manifest_hash_mismatch(tmp_path: Path) -> None:
    registry, _ = _guarded_registry(tmp_path)
    report_path = _passing_report(registry, tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["bindings"]["candidate_manifest"]["sha256"] = "0" * 64
    e2e.seal_report(report)
    _write_json(report_path, report)
    result = _run_promote(tmp_path, "--dry-run", "--e2e-report", str(report_path))
    assert result.returncode == 2
    assert "candidate_manifest hash mismatch" in json.loads(result.stdout)["error"]


def test_r2_promotion_accepts_bound_passing_report_in_dry_run(tmp_path: Path) -> None:
    registry, runtime = _guarded_registry(tmp_path)
    report_path = _passing_report(registry, tmp_path)
    before = (runtime / "config.json").read_bytes()
    result = _run_promote(tmp_path, "--dry-run", "--e2e-report", str(report_path))
    assert result.returncode == 0, result.stdout + result.stderr
    body = json.loads(result.stdout)
    assert body["dry_run"] is True
    assert body["e2e_report"]["overall_pass"] is True
    assert (runtime / "config.json").read_bytes() == before
    assert registry.read_index()["aliases"]["promoted"] is None
