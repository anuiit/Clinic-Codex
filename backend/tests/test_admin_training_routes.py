# type: ignore
# pyright: reportMissingImports=false
"""Tests for guarded local admin training routes."""
from __future__ import annotations

import base64
import ctypes
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from backend.app.config import Settings
from backend.app.factory import create_app
from backend.services import training_jobs
from backend.services.model_registry import ModelRegistry

DISABLED_BY_DEFAULT_REASON = "disabled_by_default: set ENABLE_ADMIN_TRAINING_JOBS=1 to allow local launches"


def _png_data_url(size=(12, 12)):
    buf = io.BytesIO()
    Image.new("RGB", size, color=(230, 230, 230)).save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def _payload(analysis_id: str):
    return {
        "analysis_id": analysis_id,
        "image_data_url": _png_data_url(),
        "annotations": [{"index": 0, "class_name": "atl", "bbox": [0, 0, 4, 4]}],
    }


def _refresh_snapshot_checksums(settings: Settings) -> None:
    snapshot_dir = settings.admin_training_snapshot_path
    assert snapshot_dir is not None
    manifest_path = snapshot_dir / "snapshot_manifest.json"
    metadata_path = snapshot_dir / "metadata.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checksums = {
        "schema_version": "training-snapshot-checksums.v1",
        "snapshot_id": manifest["snapshot_id"],
        "snapshot_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "metadata_csv_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
    }
    (snapshot_dir / "checksums.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_training_snapshot(settings: Settings) -> None:
    snapshot_dir = settings.admin_training_snapshot_path
    assert snapshot_dir is not None
    (snapshot_dir / "Elements").mkdir(parents=True, exist_ok=True)
    metadata_path = snapshot_dir / "metadata.csv"
    metadata_path.write_text("image_path,element_name,class_label\n", encoding="utf-8")
    review_hash = training_jobs._sha256_file(settings.annotations_dir / "review-index.json")
    rows = []
    for item in training_jobs.AnnotationReviewStore(
        settings.annotations_dir
    ).iter_approved_annotations():
        rows.append(
            {
                "source_kind": "live_annotation",
                "source_id": f"{item['analysis_id']}:{item['index']}",
                "class_name": item["class_name"],
                "bbox": item["bbox"],
                "source_fingerprint_v1": item["source_fingerprint"],
                "source_sha256": training_jobs._sha256_file(
                    Path(item["crop_path"])
                ),
                "source_image_sha256": training_jobs._sha256_file(
                    Path(item["image_path"])
                ),
            }
        )
    live_records = [
        {
            key: row[key]
            for key in (
                "source_id",
                "class_name",
                "bbox",
                "source_fingerprint_v1",
                "source_sha256",
                "source_image_sha256",
            )
        }
        for row in rows
    ]
    manifest = {
        "schema_version": "training-snapshot.v2",
        "snapshot_id": "snapshot-test",
        "class_order": ["atl"],
        "class_count": 1,
        "row_count": len(rows),
        "live_annotation_count": len(rows),
        "live_annotations_sha256": training_jobs._live_annotations_sha256(
            live_records
        ),
        "ready_for_training": True,
        "promotion_evaluation_ready": False,
        "split_counts": {"train": 1, "dev": 0, "locked_test": 0},
        "source_manifests": (
            [
                {
                    "kind": "annotation-review-index.v1",
                    "path": str(settings.annotations_dir / "review-index.json"),
                    "sha256": review_hash,
                }
            ]
            if review_hash
            else []
        ),
        "rows": rows,
        "duplicates": [],
        "conflicts": [],
    }
    manifest_path = snapshot_dir / "snapshot_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _refresh_snapshot_checksums(settings)


def _settings(tmp_path: Path, *, enabled: bool = False, model_dir: str = ""):
    backend_root = tmp_path / "backend"
    (backend_root / "codex_pipeline" / "config").mkdir(parents=True, exist_ok=True)
    (backend_root / "codex_pipeline" / "config" / "default.yaml").write_text(
        "data:\n  image_size: 224\ntraining:\n  num_epochs: 100\nmodel:\n  backbone: dinov2_vits14\nevaluation:\n  num_eval_episodes: 10\ninference:\n  top_k: 3\n",
        encoding="utf-8",
    )
    (backend_root / "codex_pipeline" / "config" / "snapshot-warmstart.yaml").write_text(
        "data:\n  split_strategy: persisted\ntraining:\n  checkpoint_selection: latest\n",
        encoding="utf-8",
    )
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script = scripts_dir / "retrain.sh"
    script.write_text("#!/usr/bin/env bash\necho retrain $@\n", encoding="utf-8")
    script.chmod(0o755)
    (backend_root / "codex_model" / "weights").mkdir(parents=True, exist_ok=True)
    (backend_root / "codex_model" / "config.json").write_text(
        '{"class_names":["atl"]}\n',
        encoding="utf-8",
    )
    (backend_root / "codex_model" / "weights" / "projection.pt").write_bytes(
        b"projection"
    )
    backbone_manifest = tmp_path / "dinov2-pin.json"
    backbone_manifest.write_text("{}\n", encoding="utf-8")
    settings = Settings(
        backend_root=backend_root,
        testing=True,
        model_dir=model_dir,
        enable_admin_training_jobs=enabled,
        admin_training_snapshot_dir=str(tmp_path / "snapshot"),
        admin_training_backbone_manifest=str(backbone_manifest),
    )
    _write_training_snapshot(settings)
    return settings


def _client(settings):
    app = create_app(settings=settings)
    return app, app.test_client()


def _approve_one(client, analysis_id: str = "training-ready-1"):
    assert client.post("/save-annotation", json=_payload(analysis_id)).status_code == 200
    assert client.post(f"/admin/annotations/{analysis_id}/0/review", json={"status": "approved"}).status_code == 200
    settings = client.application.extensions["clinic_services"].settings
    _write_training_snapshot(settings)


def test_training_summary_is_visible_but_launch_disabled_by_default(tmp_path):
    settings = _settings(tmp_path, enabled=False)
    _app, client = _client(settings)
    assert client.post("/save-annotation", json=_payload("training-summary-1")).status_code == 200
    assert client.post("/admin/annotations/training-summary-1/0/review", json={"status": "approved"}).status_code == 200

    resp = client.get("/admin/training/summary")

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["training_jobs_enabled"] is False
    assert body["launch_allowed_for_request"] is False
    assert body["launch_disabled_reasons"] == [DISABLED_BY_DEFAULT_REASON]
    assert body["data"]["trainable"] == 1
    assert body["data"]["per_class"] == {"atl": 1}
    assert set(body["data"]["split_counts"]) == {"train", "val", "test", "excluded"}
    assert (
        body["data"]["split_counts"]["train"]
        + body["data"]["split_counts"]["val"]
        + body["data"]["split_counts"]["test"]
    ) == 1
    assert body["data"]["split_counts"]["excluded"] == 0
    assert body["parameters"]["editable"]["device"] == ["auto", "cpu", "mps", "cuda"]
    assert body["paths"]["model_registry_dir"] == str(settings.model_registry_dir)
    assert body["paths"]["promote_script"].endswith("scripts/promote_model.py")
    assert body["artifacts"]["model_registry"]["status"] == "not_initialized"
    assert body["artifacts"]["model_registry"]["aliases"]["promoted"] is None


def test_training_summary_enabled_loopback_allows_launch(tmp_path):
    settings = _settings(tmp_path, enabled=True)
    _app, client = _client(settings)
    _approve_one(client, "training-enabled-1")

    resp = client.get(
        "/admin/training/summary",
        headers={"Host": "localhost", "Origin": "http://localhost:7118"},
        environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
    )

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["training_jobs_enabled"] is True
    assert body["launch_allowed_for_request"] is True
    assert body["launch_disabled_reasons"] == []
    assert body["training_snapshot"]["snapshot_id"] == "snapshot-test"
    assert body["training_snapshot"]["live_annotation_count"] == 1
    assert body["training_snapshot"]["valid"] is True


def test_training_summary_blocks_a_snapshot_older_than_current_reviews(tmp_path):
    settings = _settings(tmp_path, enabled=True)
    _app, client = _client(settings)
    assert client.post("/save-annotation", json=_payload("training-stale-snapshot")).status_code == 200
    assert (
        client.post(
            "/admin/annotations/training-stale-snapshot/0/review",
            json={"status": "approved"},
        ).status_code
        == 200
    )

    body = client.get(
        "/admin/training/summary",
        headers={"Host": "localhost", "Origin": "http://localhost:7118"},
        environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
    ).get_json()

    assert body["launch_allowed_for_request"] is False
    assert body["training_snapshot"]["valid"] is False
    assert any(
        reason.startswith("training_snapshot_stale:")
        for reason in body["launch_disabled_reasons"]
    )


def test_training_start_rejects_snapshot_missing_a_current_live_row(
    tmp_path, monkeypatch
):
    settings = _settings(tmp_path, enabled=True)
    _app, client = _client(settings)
    _approve_one(client, "training-incomplete-snapshot")
    manifest_path = settings.admin_training_snapshot_path / "snapshot_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["rows"] = []
    manifest["live_annotation_count"] = 0
    manifest["live_annotations_sha256"] = training_jobs._live_annotations_sha256([])
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _refresh_snapshot_checksums(settings)
    monkeypatch.setattr(
        "backend.services.training_jobs.subprocess.Popen",
        lambda *_args, **_kwargs: pytest.fail("Popen must not be called"),
    )

    response = client.post(
        "/admin/training/jobs", json={"dry_run": True}, headers={"Host": "localhost"}
    )

    assert response.status_code == 403
    assert "training_snapshot_stale" in response.get_json()["error"]


def test_training_start_rejects_snapshot_after_approved_crop_drifts(
    tmp_path, monkeypatch
):
    settings = _settings(tmp_path, enabled=True)
    _app, client = _client(settings)
    _approve_one(client, "training-drifted-crop")
    crop_path = next(
        training_jobs.AnnotationReviewStore(
            settings.annotations_dir
        ).iter_approved_annotations()
    )["crop_path"]
    Path(crop_path).write_bytes(b"drifted")
    monkeypatch.setattr(
        "backend.services.training_jobs.subprocess.Popen",
        lambda *_args, **_kwargs: pytest.fail("Popen must not be called"),
    )

    response = client.post(
        "/admin/training/jobs", json={"dry_run": True}, headers={"Host": "localhost"}
    )

    assert response.status_code == 403
    assert "training_snapshot_stale" in response.get_json()["error"]


def test_training_start_revalidates_snapshot_under_launch_guard(tmp_path, monkeypatch):
    settings = _settings(tmp_path, enabled=True)
    app, client = _client(settings)
    _approve_one(client, "training-toctou")
    service = app.extensions["clinic_services"].admin_training_service()
    valid = service._training_snapshot_info()
    stale = {
        **valid,
        "valid": False,
        "errors": [
            "training_snapshot_stale: rebuild it from the current approved annotations"
        ],
    }
    snapshots = iter([valid, stale])
    monkeypatch.setattr(service, "_training_snapshot_info", lambda: next(snapshots))
    monkeypatch.setattr(
        "backend.services.training_jobs.subprocess.Popen",
        lambda *_args, **_kwargs: pytest.fail("Popen must not be called"),
    )

    response = client.post(
        "/admin/training/jobs", json={"dry_run": True}, headers={"Host": "localhost"}
    )

    assert response.status_code == 403
    assert "training_snapshot_stale" in response.get_json()["error"]


def test_training_start_rejects_model_dir_override(tmp_path, monkeypatch):
    settings = _settings(
        tmp_path, enabled=True, model_dir=str(tmp_path / "custom-runtime")
    )
    _app, client = _client(settings)
    _approve_one(client, "training-model-dir")
    monkeypatch.setattr(
        "backend.services.training_jobs.subprocess.Popen",
        lambda *_args, **_kwargs: pytest.fail("Popen must not be called"),
    )

    response = client.post(
        "/admin/training/jobs", json={"dry_run": True}, headers={"Host": "localhost"}
    )

    assert response.status_code == 403
    assert "training_model_dir_override_unsupported" in response.get_json()["error"]


def test_training_summary_enabled_blocks_launch_until_annotation_is_trainable(tmp_path):
    settings = _settings(tmp_path, enabled=True)
    _app, client = _client(settings)

    resp = client.get(
        "/admin/training/summary",
        headers={"Host": "localhost", "Origin": "http://localhost:7118"},
        environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
    )

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["training_jobs_enabled"] is True
    assert body["launch_allowed_for_request"] is False
    assert body["launch_disabled_reasons"] == [
        "no_trainable_annotations: approve at least one current annotation before launching retraining"
    ]

    start = client.post(
        "/admin/training/jobs",
        json={"dry_run": True, "device": "cpu", "batch_size": 8},
        headers={"Host": "localhost", "Origin": "http://localhost:7118"},
        environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
    )

    assert start.status_code == 403
    assert "no_trainable_annotations" in start.get_json()["error"]


def test_local_helpers_include_ipv6_loopback_and_reject_remote_addresses():
    assert training_jobs.is_loopback_address("::1") is True
    assert training_jobs.is_loopback_address("::ffff:127.0.0.1") is True
    assert training_jobs.is_loopback_address("[::1]:7117") is True
    assert training_jobs.is_loopback_address("127.0.0.1") is True
    assert training_jobs.is_loopback_address("192.0.2.10") is False
    assert training_jobs.is_local_origin(None) is True
    assert training_jobs.is_local_origin("http://[::1]:7118") is True
    assert training_jobs.is_local_origin("http://192.0.2.10:7118") is False


@pytest.mark.parametrize(
    "path",
    [
        "/admin/training/summary",
        "/admin/training/jobs/latest",
        "/admin/training/jobs/some-run",
    ],
)
def test_admin_training_status_routes_reject_non_loopback_requests(tmp_path, path):
    settings = _settings(tmp_path, enabled=True)
    _app, client = _client(settings)

    resp = client.get(
        path,
        headers={"Host": "localhost", "Origin": "http://localhost:7118"},
        environ_overrides={"REMOTE_ADDR": "192.0.2.10"},
    )

    assert resp.status_code == 403
    body = resp.get_json()
    assert body["error_code"] == "LOCAL_ONLY_FORBIDDEN"
    assert "non_loopback_remote_addr" in body["reasons"]


def test_admin_route_inventory_has_explicit_local_only_policy(tmp_path):
    settings = _settings(tmp_path, enabled=False)
    app, _test_client = _client(settings)
    admin_rules = {
        (next(iter(rule.methods - {"HEAD", "OPTIONS"})), rule.rule, rule.endpoint)
        for rule in app.url_map.iter_rules()
        if rule.rule.startswith("/admin/")
    }

    assert admin_rules == {
        ("GET", "/admin/annotations", "admin_annotations.list_admin_annotations"),
        ("GET", "/admin/annotations/<analysis_id>/image", "admin_annotations.get_admin_annotation_image"),
        ("GET", "/admin/annotations/<analysis_id>/<int:index>/crop", "admin_annotations.get_admin_annotation_crop"),
        ("POST", "/admin/annotations/<analysis_id>/<int:index>/review", "admin_annotations.set_admin_annotation_review"),
        ("POST", "/admin/annotations/<analysis_id>/<int:index>/modify", "admin_annotations.modify_admin_annotation_element"),
        ("GET", "/admin/training/summary", "admin_training.get_admin_training_summary"),
        ("GET", "/admin/training/jobs/latest", "admin_training.get_latest_admin_training_job"),
        ("GET", "/admin/training/jobs/<run_id>", "admin_training.get_admin_training_job"),
        ("POST", "/admin/training/jobs", "admin_training.start_admin_training_job"),
    }

    service_guarded = {"admin_training.start_admin_training_job"}
    for _method, _rule, endpoint in admin_rules:
        if endpoint in service_guarded:
            continue
        assert getattr(app.view_functions[endpoint], "__clinic_local_required__", False), endpoint


def test_training_summary_surfaces_model_registry_candidate_health(tmp_path):
    settings = _settings(tmp_path, enabled=False)
    source = tmp_path / "candidate-source"
    (source / "weights").mkdir(parents=True)
    (source / "weights" / "prototypes.pt").write_bytes(b"candidate-prototypes")
    (source / "weights" / "projection.pt").write_bytes(b"candidate-projection")
    (source / "config.json").write_text('{"model_version":"candidate"}', encoding="utf-8")
    registry = ModelRegistry(
        settings.model_registry_dir,
        repo_root=tmp_path,
        runtime_model_dir=settings.backend_root / "codex_model",
    )
    registry.create_version_from_artifacts(
        "20260527T010203Z-test-candidate",
        artifact_sources={
            "runtime/weights/prototypes.pt": source / "weights" / "prototypes.pt",
            "runtime/weights/projection.pt": source / "weights" / "projection.pt",
            "runtime/config.json": source / "config.json",
        },
    )
    _app, client = _client(settings)

    body = client.get("/admin/training/summary").get_json()

    registry_body = body["artifacts"]["model_registry"]
    assert registry_body["status"] == "ok"
    assert registry_body["aliases"]["candidate"] == "20260527T010203Z-test-candidate"
    assert registry_body["latest_candidate"]["manifest_health"]["status"] == "healthy"
    assert body["paths"]["candidate_version_dir"].endswith(
        "backend/model_registry/versions/20260527T010203Z-test-candidate"
    )


def test_training_summary_marks_registry_candidate_unhealthy_on_checksum_drift(tmp_path):
    settings = _settings(tmp_path, enabled=False)
    source = tmp_path / "candidate-source"
    (source / "weights").mkdir(parents=True)
    (source / "weights" / "prototypes.pt").write_bytes(b"candidate-prototypes")
    (source / "weights" / "projection.pt").write_bytes(b"candidate-projection")
    (source / "config.json").write_text('{"model_version":"candidate"}', encoding="utf-8")
    registry = ModelRegistry(
        settings.model_registry_dir,
        repo_root=tmp_path,
        runtime_model_dir=settings.backend_root / "codex_model",
    )
    registry.create_version_from_artifacts(
        "20260527T010203Z-test-candidate",
        artifact_sources={
            "runtime/weights/prototypes.pt": source / "weights" / "prototypes.pt",
            "runtime/weights/projection.pt": source / "weights" / "projection.pt",
            "runtime/config.json": source / "config.json",
        },
    )
    (registry.version_dir("20260527T010203Z-test-candidate") / "checksums.sha256").write_text(
        "0" * 64 + "  runtime/config.json\n",
        encoding="utf-8",
    )
    _app, client = _client(settings)

    body = client.get("/admin/training/summary").get_json()

    health = body["artifacts"]["model_registry"]["latest_candidate"]["manifest_health"]
    assert health["status"] == "unhealthy"
    assert any("checksums_sha256 mismatch" in error for error in health["errors"])


def test_training_summary_surfaces_incomplete_promotion_marker(tmp_path):
    settings = _settings(tmp_path, enabled=False)
    settings.model_registry_dir.mkdir(parents=True)
    (settings.model_registry_dir / "promotion_in_progress.json").write_text(
        '{"version_id":"v1","action":"promote"}',
        encoding="utf-8",
    )
    _app, client = _client(settings)

    body = client.get("/admin/training/summary").get_json()

    registry_body = body["artifacts"]["model_registry"]
    assert registry_body["status"] == "promotion_in_progress"
    assert registry_body["promotion_in_progress"]["version_id"] == "v1"


def test_training_start_disabled_by_default_returns_403(tmp_path):
    settings = _settings(tmp_path, enabled=False)
    _app, client = _client(settings)

    resp = client.post("/admin/training/jobs", json={"dry_run": True, "device": "cpu", "batch_size": 8})

    assert resp.status_code == 403
    error = resp.get_json()["error"]
    assert DISABLED_BY_DEFAULT_REASON in error
    assert "no_trainable_annotations" in error




def test_training_start_reports_review_store_unavailable_without_downgrading_to_no_data(tmp_path, monkeypatch):
    settings = _settings(tmp_path, enabled=True)
    _app, client = _client(settings)

    def fail_queue(self):
        raise RuntimeError("review manifest is corrupt")

    monkeypatch.setattr("backend.services.annotation_review.AnnotationReviewStore.list_queue", fail_queue)

    resp = client.post(
        "/admin/training/jobs",
        json={"dry_run": True, "device": "cpu", "batch_size": 8},
        headers={"Host": "localhost", "Origin": "http://localhost:7118"},
        environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
    )

    assert resp.status_code == 403
    error = resp.get_json()["error"]
    assert "review_store_unavailable" in error
    assert "review manifest is corrupt" in error
    assert "no_trainable_annotations" not in error

def test_training_start_rejects_nonlocal_remote_host_and_origin(tmp_path):
    settings = _settings(tmp_path, enabled=True)
    _app, client = _client(settings)
    _approve_one(client, "training-nonlocal-1")

    nonlocal_remote = client.post(
        "/admin/training/jobs",
        json={"dry_run": True},
        headers={"Host": "localhost"},
        environ_overrides={"REMOTE_ADDR": "10.0.0.5"},
    )
    nonlocal_host = client.post(
        "/admin/training/jobs",
        json={"dry_run": True},
        headers={"Host": "example.com", "Origin": "http://localhost:7118"},
    )
    nonlocal_origin = client.post(
        "/admin/training/jobs",
        json={"dry_run": True},
        headers={"Host": "localhost", "Origin": "http://evil.example"},
    )

    assert nonlocal_remote.status_code == 403
    assert "non_loopback_remote_addr" in nonlocal_remote.get_json()["error"]
    assert nonlocal_host.status_code == 403
    assert "nonlocal_host" in nonlocal_host.get_json()["error"]
    assert nonlocal_origin.status_code == 403
    assert "nonlocal_origin" in nonlocal_origin.get_json()["error"]


@pytest.mark.parametrize(
    "payload, error",
    [
        ({"dry_run": True, "command": "rm -rf /"}, "unknown field"),
        ({"dry_run": True, "device": "shell"}, "device must be one of"),
        ({"dry_run": True, "batch_size": 999}, "batch_size must be between"),
    ],
)
def test_training_start_rejects_malicious_or_invalid_payloads(tmp_path, payload, error):
    settings = _settings(tmp_path, enabled=True)
    _app, client = _client(settings)
    _approve_one(client, "training-payload-1")

    resp = client.post("/admin/training/jobs", json=payload, headers={"Host": "localhost"})

    assert resp.status_code == 400
    assert error in resp.get_json()["error"]


def test_training_start_rejects_when_launch_guard_is_already_held(tmp_path):
    settings = _settings(tmp_path, enabled=True)
    app, client = _client(settings)
    _approve_one(client, "training-lock-1")
    settings.admin_training_runs_dir.mkdir(parents=True)
    (settings.admin_training_runs_dir / ".launch.lock").write_text('{"pid":999999}\n', encoding="utf-8")

    resp = client.post("/admin/training/jobs", json={"dry_run": True}, headers={"Host": "localhost"})

    assert resp.status_code == 409
    assert "launch already in progress" in resp.get_json()["error"]


def test_training_start_records_allowlisted_dry_run_and_blocks_concurrent_runs(tmp_path, monkeypatch):
    settings = _settings(tmp_path, enabled=True)
    _app, client = _client(settings)
    _approve_one(client, "training-start-1")
    calls = []

    class FakePopen:
        pid = 4242

        def __init__(self, command, cwd, env, stdout, stderr, text):
            calls.append({"command": command, "cwd": cwd, "env": env, "stderr": stderr, "text": text})
            stdout.write("mock dry run started\n")
            stdout.flush()

    monkeypatch.setattr("backend.services.training_jobs.subprocess.Popen", FakePopen)

    resp = client.post(
        "/admin/training/jobs",
        json={"dry_run": True, "device": "cpu", "batch_size": 8, "notes": "smoke"},
        headers={"Host": "localhost", "Origin": "http://localhost:7118"},
    )

    assert resp.status_code == 202
    body = resp.get_json()
    job = body["job"]
    assert job["status"] == "running"
    assert job["dry_run"] is True
    assert job["device"] == "cpu"
    assert job["batch_size"] == 8
    assert job["pid"] == 4242
    assert job["command"] == [
        "bash",
        str(settings.admin_training_script_path),
        "--elements-dir",
        str(settings.admin_training_snapshot_path / "Elements"),
        "--approved-manifest",
        str(settings.admin_training_snapshot_path / "snapshot_manifest.json"),
        "--metadata-csv",
        str(settings.admin_training_snapshot_path / "metadata.csv"),
        "--backbone-manifest",
        str(settings.admin_training_backbone_manifest_path),
        "--config",
        str(settings.admin_training_config_path),
        "--init-projection",
        str(settings.classifier_weights_dir / "projection.pt"),
        "--dry-run",
    ]
    assert "--allow-runtime-write" not in job["command"]
    assert job["training_snapshot"]["snapshot_id"] == "snapshot-test"
    assert job["training_snapshot"]["live_annotation_count"] == 1
    assert job["training_snapshot_manifest_hash"]
    assert job["model_version_id"].startswith("20")
    assert job["candidate_version_dir"].endswith(job["model_version_id"])
    assert job["candidate_manifest_path"].endswith(f"{job['model_version_id']}/manifest.json")
    assert job["env"] == {
        "BATCH_SIZE": "8",
        "DEVICE": "cpu",
        "MODEL_REGISTRY_DIR": str(settings.model_registry_dir),
        "MODEL_VERSION_ID": job["model_version_id"],
        "PYTHONUNBUFFERED": "1",
    }
    assert calls[0]["env"]["BATCH_SIZE"] == "8"
    assert calls[0]["env"]["DEVICE"] == "cpu"
    assert calls[0]["env"]["MODEL_VERSION_ID"] == job["model_version_id"]
    assert calls[0]["env"]["MODEL_REGISTRY_DIR"] == str(settings.model_registry_dir)
    assert "--allow-runtime-write" not in calls[0]["command"]
    assert "command" not in calls[0]["env"]

    latest = client.get("/admin/training/jobs/latest").get_json()["job"]
    assert latest["run_id"] == job["run_id"]
    assert latest["log_tail"] == ["mock dry run started"]

    conflict = client.post("/admin/training/jobs", json={"dry_run": True}, headers={"Host": "localhost"})
    assert conflict.status_code == 409
    assert "already running" in conflict.get_json()["error"]


def test_training_latest_recovers_stale_running_job_after_backend_restart(tmp_path, monkeypatch):
    settings = _settings(tmp_path, enabled=True)
    run_dir = settings.admin_training_runs_dir / "stale-run"
    run_dir.mkdir(parents=True)
    (run_dir / "train.log").write_text("previous process disappeared\n", encoding="utf-8")
    (run_dir / "status.json").write_text(
        json.dumps(
            {
                "run_id": "stale-run",
                "status": "running",
                "dry_run": True,
                "device": "cpu",
                "batch_size": 8,
                "started_at": "2026-05-27T08:00:00+00:00",
                "exit_code": None,
                "pid": 987654,
                "log_path": str(run_dir / "train.log"),
            }
        ),
        encoding="utf-8",
    )
    _app, client = _client(settings)
    _approve_one(client, "training-stale-1")
    monkeypatch.setattr("backend.services.training_jobs._process_is_alive", lambda _pid: False)
    calls = []

    class FakePopen:
        pid = 4243

        def __init__(self, command, cwd, env, stdout, stderr, text):
            calls.append({"command": command, "cwd": cwd, "env": env})
            stdout.write("new dry run started\n")
            stdout.flush()

    monkeypatch.setattr("backend.services.training_jobs.subprocess.Popen", FakePopen)

    latest = client.get("/admin/training/jobs/latest").get_json()["job"]

    assert latest["run_id"] == "stale-run"
    assert latest["status"] == "failed"
    assert "no longer running" in latest["error"]
    assert latest["log_tail"] == ["previous process disappeared"]

    resp = client.post(
        "/admin/training/jobs",
        json={"dry_run": True, "device": "cpu", "batch_size": 8},
        headers={"Host": "localhost"},
    )

    assert resp.status_code == 202
    assert calls


def test_training_latest_recovers_dry_run_without_handle_even_if_pid_was_reused(tmp_path, monkeypatch):
    settings = _settings(tmp_path, enabled=True)
    run_dir = settings.admin_training_runs_dir / "reused-pid-dry-run"
    run_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text(
        json.dumps(
            {
                "run_id": "reused-pid-dry-run",
                "status": "running",
                "dry_run": True,
                "device": "cpu",
                "batch_size": 8,
                "started_at": "2026-05-27T08:00:00+00:00",
                "exit_code": None,
                "pid": 222222,
                "log_path": str(run_dir / "train.log"),
            }
        ),
        encoding="utf-8",
    )
    _app, client = _client(settings)
    monkeypatch.setattr("backend.services.training_jobs._process_is_alive", lambda _pid: True)

    latest = client.get("/admin/training/jobs/latest").get_json()["job"]

    assert latest["run_id"] == "reused-pid-dry-run"
    assert latest["status"] == "failed"
    assert "no longer running" in latest["error"]


def test_training_latest_recovers_full_run_when_lock_pid_does_not_match(tmp_path, monkeypatch):
    settings = _settings(tmp_path, enabled=True)
    run_dir = settings.admin_training_runs_dir / "stale-full-run"
    run_dir.mkdir(parents=True)
    (settings.backend_root / ".retrain.lock").write_text("111111\n", encoding="utf-8")
    (run_dir / "status.json").write_text(
        json.dumps(
            {
                "run_id": "stale-full-run",
                "status": "running",
                "dry_run": False,
                "device": "cpu",
                "batch_size": 8,
                "started_at": "2026-05-27T08:00:00+00:00",
                "exit_code": None,
                "pid": 222222,
                "lock_path": str(settings.backend_root / ".retrain.lock"),
                "log_path": str(run_dir / "train.log"),
            }
        ),
        encoding="utf-8",
    )
    _app, client = _client(settings)
    monkeypatch.setattr("backend.services.training_jobs._process_is_alive", lambda _pid: True)

    latest = client.get("/admin/training/jobs/latest").get_json()["job"]

    assert latest["run_id"] == "stale-full-run"
    assert latest["status"] == "failed"
    assert "no longer running" in latest["error"]


def test_training_latest_recovers_full_run_when_process_identity_does_not_match(tmp_path, monkeypatch):
    settings = _settings(tmp_path, enabled=True)
    run_dir = settings.admin_training_runs_dir / "reused-pid-full-run"
    run_dir.mkdir(parents=True)
    (settings.backend_root / ".retrain.lock").write_text("222222\n", encoding="utf-8")
    (run_dir / "status.json").write_text(
        json.dumps(
            {
                "run_id": "reused-pid-full-run",
                "status": "running",
                "dry_run": False,
                "device": "cpu",
                "batch_size": 8,
                "started_at": "2026-05-27T08:00:00+00:00",
                "exit_code": None,
                "pid": 222222,
                "process_identity": "linux-proc-start:old-process",
                "lock_path": str(settings.backend_root / ".retrain.lock"),
                "log_path": str(run_dir / "train.log"),
            }
        ),
        encoding="utf-8",
    )
    _app, client = _client(settings)
    monkeypatch.setattr("backend.services.training_jobs._process_is_alive", lambda _pid: True)
    monkeypatch.setattr("backend.services.training_jobs._process_identity", lambda _pid: "linux-proc-start:new-process")

    latest = client.get("/admin/training/jobs/latest").get_json()["job"]

    assert latest["run_id"] == "reused-pid-full-run"
    assert latest["status"] == "failed"
    assert "no longer running" in latest["error"]


def test_training_latest_recovers_full_run_when_process_identity_is_unavailable(tmp_path, monkeypatch):
    settings = _settings(tmp_path, enabled=True)
    run_dir = settings.admin_training_runs_dir / "unknown-identity-full-run"
    run_dir.mkdir(parents=True)
    (settings.backend_root / ".retrain.lock").write_text("222222\n", encoding="utf-8")
    (run_dir / "status.json").write_text(
        json.dumps(
            {
                "run_id": "unknown-identity-full-run",
                "status": "running",
                "dry_run": False,
                "device": "cpu",
                "batch_size": 8,
                "started_at": "2026-05-27T08:00:00+00:00",
                "exit_code": None,
                "pid": 222222,
                "process_identity": None,
                "lock_path": str(settings.backend_root / ".retrain.lock"),
                "log_path": str(run_dir / "train.log"),
            }
        ),
        encoding="utf-8",
    )
    _app, client = _client(settings)
    monkeypatch.setattr("backend.services.training_jobs._process_is_alive", lambda _pid: True)
    monkeypatch.setattr("backend.services.training_jobs._process_identity", lambda _pid: None)

    latest = client.get("/admin/training/jobs/latest").get_json()["job"]

    assert latest["run_id"] == "unknown-identity-full-run"
    assert latest["status"] == "failed"
    assert "no longer running" in latest["error"]


def test_training_latest_keeps_full_run_running_when_lock_and_process_identity_match(tmp_path, monkeypatch):
    settings = _settings(tmp_path, enabled=True)
    run_dir = settings.admin_training_runs_dir / "active-full-run"
    run_dir.mkdir(parents=True)
    (settings.backend_root / ".retrain.lock").write_text("222222\n", encoding="utf-8")
    (run_dir / "status.json").write_text(
        json.dumps(
            {
                "run_id": "active-full-run",
                "status": "running",
                "dry_run": False,
                "device": "cpu",
                "batch_size": 8,
                "started_at": "2026-05-27T08:00:00+00:00",
                "exit_code": None,
                "pid": 222222,
                "process_identity": "linux-proc-start:same-process",
                "lock_path": str(settings.backend_root / ".retrain.lock"),
                "log_path": str(run_dir / "train.log"),
            }
        ),
        encoding="utf-8",
    )
    _app, client = _client(settings)
    monkeypatch.setattr("backend.services.training_jobs._process_is_alive", lambda _pid: True)
    monkeypatch.setattr("backend.services.training_jobs._process_identity", lambda _pid: "linux-proc-start:same-process")

    latest = client.get("/admin/training/jobs/latest").get_json()["job"]

    assert latest["run_id"] == "active-full-run"
    assert latest["status"] == "running"


def test_process_liveness_uses_safe_windows_query(monkeypatch):
    calls = []
    monkeypatch.setattr(training_jobs.os, "name", "nt", raising=False)
    monkeypatch.setattr(
        training_jobs,
        "_windows_process_is_alive",
        lambda pid: calls.append(pid) or False,
    )

    assert training_jobs._process_is_alive("12345") is False
    assert calls == [12345]


def test_windows_process_helpers_configure_explicit_ctypes_signatures():
    from ctypes import wintypes

    class FakeFunction:
        argtypes = None
        restype = None

        def __call__(self, *_args):
            return 0

    kernel32 = SimpleNamespace(
        OpenProcess=FakeFunction(),
        GetLastError=FakeFunction(),
        GetExitCodeProcess=FakeFunction(),
        GetProcessTimes=FakeFunction(),
        CloseHandle=FakeFunction(),
    )

    training_jobs._configure_kernel32_process_signatures(kernel32, ctypes, wintypes)

    assert kernel32.OpenProcess.argtypes == (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    assert kernel32.OpenProcess.restype is wintypes.HANDLE
    assert kernel32.GetExitCodeProcess.argtypes == (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    assert kernel32.GetExitCodeProcess.restype is wintypes.BOOL
    assert kernel32.GetProcessTimes.argtypes == (
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
    )
    assert kernel32.GetProcessTimes.restype is wintypes.BOOL
    assert kernel32.CloseHandle.argtypes == (wintypes.HANDLE,)
    assert kernel32.CloseHandle.restype is wintypes.BOOL
