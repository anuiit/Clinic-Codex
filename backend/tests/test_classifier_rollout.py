from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from backend.app.config import Settings
from backend.app.services.classifier_rollout import (
    CandidateTimeout,
    ClassifierRollout,
    content_digest,
    selected_by_percent,
)


def _result(name: str) -> dict:
    return {
        "class_name": name,
        "class_label": 1,
        "confidence": 0.8,
        "rejected": False,
        "top_k": [],
    }


class PrimaryClassifier:
    def __init__(self) -> None:
        self.single_calls = 0
        self.batch_calls = 0

    def classify(self, _image, **_kwargs):
        self.single_calls += 1
        return _result("primary")

    def classify_batch(self, images):
        self.batch_calls += 1
        return [_result("primary") for _ in images]


class FakeRegistry:
    def __init__(self, runtime_dir: Path) -> None:
        self.runtime_dir = runtime_dir
        self.calls: list[str] = []

    def resolve_runtime_package(self, reference: str) -> Path:
        self.calls.append(reference)
        return self.runtime_dir


class FakeClient:
    state = "not_started"

    def __init__(self, *, error: BaseException | None = None, blocker: threading.Event | None = None) -> None:
        self.error = error
        self.blocker = blocker
        self.calls: list[tuple[list[np.ndarray], int]] = []
        self.closed = False

    def infer(self, images, *, top_k=3):
        arrays = [np.asarray(image) for image in images]
        self.calls.append((arrays, top_k))
        if self.blocker is not None:
            self.blocker.wait(timeout=2.0)
        if self.error is not None:
            raise self.error
        return [_result("candidate") for _ in arrays], 0.01

    def close(self):
        self.closed = True


def _settings(tmp_path: Path, mode: str, **overrides) -> Settings:
    values = {
        "backend_root": tmp_path / "backend",
        "testing": True,
        "classifier_rollout_mode": mode,
        "classifier_candidate_reference": "r2",
        "classifier_canary_percent": 100.0,
        "classifier_shadow_sample_percent": 100.0,
        "classifier_candidate_timeout_seconds": 0.05,
        "classifier_shadow_queue_size": 1,
        "classifier_shadow_max_inflight": 1,
    }
    values.update(overrides)
    return Settings(**values)


def _rollout(tmp_path, mode, *, client=None, settings_overrides=None):
    primary = PrimaryClassifier()
    registry = FakeRegistry(tmp_path / "backend" / "model_registry" / "versions" / "r2" / "runtime")
    client = client or FakeClient()
    settings = _settings(tmp_path, mode, **(settings_overrides or {}))
    rollout = ClassifierRollout(
        settings,
        lambda: primary,
        registry=registry,
        client_factory=lambda _path, _settings: client,
    )
    return rollout, primary, registry, client


def test_off_never_resolves_or_loads_candidate(tmp_path):
    rollout, primary, registry, client = _rollout(tmp_path, "off")

    response = rollout.classify(np.zeros((3, 4, 3), dtype=np.uint8))

    assert response["class_name"] == "primary"
    assert primary.single_calls == 1
    assert registry.calls == []
    assert client.calls == []
    assert rollout.readiness()["active_mode"] == "off"
    assert rollout.readiness()["degraded"] is False


def test_shadow_returns_primary_and_degrades_after_candidate_failure(tmp_path):
    rollout, primary, _registry, client = _rollout(
        tmp_path, "shadow", client=FakeClient(error=RuntimeError("boom"))
    )

    response = rollout.classify(np.ones((3, 4, 3), dtype=np.uint8))

    assert response["class_name"] == "primary"
    assert primary.single_calls == 1
    assert rollout.wait_for_shadow()
    report = rollout.readiness()
    assert report["requested_mode"] == "shadow"
    assert report["active_mode"] == "off"
    assert report["degraded"] is True
    assert report["metrics"]["candidate_errors"] == 1
    assert len(client.calls) == 1


def test_shadow_timeout_is_fail_open_and_observable(tmp_path):
    rollout, primary, _registry, _client = _rollout(
        tmp_path, "shadow", client=FakeClient(error=CandidateTimeout("slow"))
    )

    response = rollout.classify(np.ones((3, 4, 3), dtype=np.uint8))

    assert response["class_name"] == "primary"
    assert primary.single_calls == 1
    assert rollout.wait_for_shadow()
    report = rollout.readiness()
    assert report["blocker"] == "candidate_timeout"
    assert report["metrics"]["candidate_timeouts"] == 1


def test_shadow_queue_and_inflight_are_bounded(tmp_path):
    release = threading.Event()
    rollout, primary, _registry, _client = _rollout(
        tmp_path, "shadow", client=FakeClient(blocker=release)
    )
    image = np.ones((3, 4, 3), dtype=np.uint8)

    for _ in range(3):
        assert rollout.classify(image)["class_name"] == "primary"

    assert primary.single_calls == 3
    metrics = rollout.metrics.snapshot()
    assert metrics["shadow_submitted"] == 2
    assert metrics["shadow_queue_full"] == 1
    release.set()
    assert rollout.wait_for_shadow()


def test_shadow_slot_released_on_submit_error(tmp_path, monkeypatch):
    rollout, primary, _registry, _client = _rollout(tmp_path, "shadow")
    image = np.ones((3, 4, 3), dtype=np.uint8)

    # Force submit() to raise, as it does after executor shutdown. The acquired
    # slot must be released and _shadow_active rolled back so capacity is not
    # leaked and wait_for_shadow() does not hang.
    def _raising_submit(*_args, **_kwargs):
        raise RuntimeError("executor shutdown")

    from concurrent.futures import ThreadPoolExecutor

    rollout._shadow_executor = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(rollout._shadow_executor, "submit", _raising_submit)

    rollout._submit_shadow([image], top_k=3, primary=[{"class_name": "primary"}])

    metrics = rollout.metrics.snapshot()
    assert metrics["shadow_submitted"] == 1
    assert metrics["shadow_submit_error"] == 1
    # The slot was released: wait_for_shadow must not hang on a leaked counter.
    assert rollout._shadow_active == 0
    assert rollout.wait_for_shadow(timeout=1.0)


def test_content_hash_is_stable_and_covers_every_batch_image():
    first = np.zeros((2, 2, 3), dtype=np.uint8)
    second = np.ones((2, 2, 3), dtype=np.uint8)
    original = content_digest([first, second])

    assert original == content_digest([first.copy(), second.copy()])
    changed = second.copy()
    changed[-1, -1, -1] = 2
    assert original != content_digest([first, changed])
    assert original != content_digest([second, first])
    assert selected_by_percent(original, 0.0) is False
    assert selected_by_percent(original, 100.0) is True


def test_canary_is_blocked_without_valid_offline_report(tmp_path):
    rollout, primary, registry, client = _rollout(tmp_path, "canary")
    images = [np.zeros((2, 2, 3), dtype=np.uint8), np.ones((2, 2, 3), dtype=np.uint8)]

    response = rollout.classify_batch(images)

    assert [row["class_name"] for row in response] == ["primary", "primary"]
    assert primary.batch_calls == 1
    assert registry.calls == ["r2"]
    assert client.calls == []
    report = rollout.readiness()
    assert report["requested_mode"] == "canary"
    assert report["active_mode"] == "off"
    assert report["blocker"] == "offline_report_missing"


def test_canary_routes_one_complete_batch_to_candidate_stably(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "backend.app.services.classifier_rollout.validate_offline_report",
        lambda *_args, **_kwargs: (True, None),
    )
    rollout, primary, registry, client = _rollout(tmp_path, "canary")
    images = [np.zeros((2, 2, 3), dtype=np.uint8), np.ones((2, 2, 3), dtype=np.uint8)]

    first = rollout.classify_batch(images)
    second = rollout.classify_batch([image.copy() for image in images])

    assert [row["class_name"] for row in first] == ["candidate", "candidate"]
    assert second == first
    assert primary.batch_calls == 0
    assert registry.calls == ["r2"]
    assert len(client.calls) == 2
    assert all(len(call_images) == 2 for call_images, _top_k in client.calls)
    assert rollout.metrics.snapshot()["canary_selected"] == 2


def test_canary_candidate_error_falls_back_to_primary(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "backend.app.services.classifier_rollout.validate_offline_report",
        lambda *_args, **_kwargs: (True, None),
    )
    rollout, primary, _registry, _client = _rollout(
        tmp_path, "canary", client=FakeClient(error=RuntimeError("candidate failed"))
    )

    response = rollout.classify_batch([np.zeros((2, 2, 3), dtype=np.uint8)])

    assert response[0]["class_name"] == "primary"
    assert primary.batch_calls == 1
    metrics = rollout.metrics.snapshot()
    assert metrics["candidate_errors"] == 1
    assert metrics["candidate_fallbacks"] == 1


def test_settings_environment_defaults_to_safe_off_and_cpu(monkeypatch):
    monkeypatch.delenv("CLASSIFIER_ROLLOUT_MODE", raising=False)
    monkeypatch.delenv("CLASSIFIER_CANDIDATE_DEVICE", raising=False)
    monkeypatch.delenv("CLASSIFIER_CANARY_PERCENT", raising=False)

    settings = Settings.from_env()

    assert settings.classifier_rollout_mode == "off"
    assert settings.classifier_candidate_device == "cpu"
    assert settings.classifier_canary_percent == 0.0


def test_settings_environment_reads_bounded_rollout_controls(monkeypatch):
    monkeypatch.setenv("CLASSIFIER_ROLLOUT_MODE", "shadow")
    monkeypatch.setenv("CLASSIFIER_CANDIDATE_REFERENCE", "version-r2")
    monkeypatch.setenv("CLASSIFIER_SHADOW_QUEUE_SIZE", "3")
    monkeypatch.setenv("CLASSIFIER_SHADOW_MAX_INFLIGHT", "2")
    monkeypatch.setenv("CLASSIFIER_CANDIDATE_BATCH_SIZE", "7")
    monkeypatch.setenv("CLASSIFIER_CANARY_PERCENT", "12.5")

    settings = Settings.from_env()

    assert settings.classifier_rollout_mode == "shadow"
    assert settings.classifier_candidate_reference == "version-r2"
    assert settings.classifier_shadow_queue_size == 3
    assert settings.classifier_shadow_max_inflight == 2
    assert settings.classifier_candidate_batch_size == 7
    assert settings.classifier_canary_percent == 12.5
