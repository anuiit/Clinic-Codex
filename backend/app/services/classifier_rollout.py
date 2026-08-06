"""Fail-open classifier candidate rollout with process isolation.

The ``off`` path does not resolve, import, or load the candidate. Shadow
inference runs outside the Flask process and retains aggregate counters only.
Canary routing hashes the complete ordered image payload and stays blocked
until an immutable offline evaluation report is validated.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import multiprocessing
import queue
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from backend.app.config import Settings


class CandidateUnavailable(RuntimeError):
    """The candidate cannot safely serve the current request."""


class CandidateTimeout(CandidateUnavailable):
    """The isolated candidate did not answer within its deadline."""


def _as_array(image: Image.Image | np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(image))


def content_digest(images: Sequence[Image.Image | np.ndarray]) -> bytes:
    """Hash all ordered pixels plus their shape and dtype."""

    digest = hashlib.sha256()
    digest.update(len(images).to_bytes(8, "big"))
    for index, image in enumerate(images):
        array = _as_array(image)
        digest.update(index.to_bytes(8, "big"))
        dtype = array.dtype.str.encode("ascii")
        digest.update(len(dtype).to_bytes(4, "big"))
        digest.update(dtype)
        digest.update(len(array.shape).to_bytes(4, "big"))
        for dimension in array.shape:
            digest.update(int(dimension).to_bytes(8, "big"))
        digest.update(array.tobytes(order="C"))
    return digest.digest()


def selected_by_percent(digest: bytes, percent: float) -> bool:
    if percent <= 0:
        return False
    if percent >= 100:
        return True
    bucket = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return bucket < percent / 100.0


def _candidate_process_main(
    runtime_dir: str,
    device: str,
    max_batch_size: int,
    request_queue,
    response_queue,
) -> None:
    """Child-only entrypoint; heavy model imports stay outside Flask."""

    try:
        try:
            from backend.codex_model import CodexClassifier
        except ImportError:  # pragma: no cover - backend-root compatibility
            from codex_model import CodexClassifier  # type: ignore
        classifier = CodexClassifier(model_dir=runtime_dir, device=device)
        response_queue.put({"kind": "ready"})
    except BaseException as exc:
        response_queue.put({"kind": "startup_error", "error_type": type(exc).__name__})
        return

    while True:
        job = request_queue.get()
        if job is None:
            return
        try:
            results: list[dict[str, Any]] = []
            images = job["images"]
            top_k = int(job.get("top_k", 3))
            for start in range(0, len(images), max_batch_size):
                results.extend(
                    classifier.classify_batch(images[start : start + max_batch_size], top_k=top_k)
                )
            response_queue.put({"kind": "result", "job_id": job["job_id"], "results": results})
        except BaseException as exc:
            response_queue.put(
                {"kind": "inference_error", "job_id": job["job_id"], "error_type": type(exc).__name__}
            )


class CandidateProcessClient:
    """One bounded, restartable process shared by shadow and canary."""

    def __init__(self, runtime_dir: Path, settings: Settings) -> None:
        self.runtime_dir = Path(runtime_dir)
        self.device = settings.classifier_candidate_device
        self.timeout_seconds = settings.classifier_candidate_timeout_seconds
        self.max_batch_size = settings.classifier_candidate_batch_size
        self.max_images = settings.classifier_candidate_max_images
        self._context = multiprocessing.get_context("spawn")
        self._lock = threading.Lock()
        self._process = None
        self._request_queue = None
        self._response_queue = None
        self._ready = False
        atexit.register(self.close)

    @property
    def state(self) -> str:
        if self._process is None:
            return "not_started"
        return "running" if self._process.is_alive() and self._ready else "unavailable"

    def infer(
        self, images: Sequence[Image.Image | np.ndarray], *, top_k: int = 3
    ) -> tuple[list[dict[str, Any]], float]:
        if not images:
            return [], 0.0
        if len(images) > self.max_images:
            raise CandidateUnavailable(
                f"candidate request has {len(images)} images; maximum is {self.max_images}"
            )
        started = time.monotonic()
        if not self._lock.acquire(timeout=self.timeout_seconds):
            raise CandidateTimeout("candidate in-flight limit reached")
        try:
            deadline = started + self.timeout_seconds
            self._ensure_started_locked(deadline)
            job_id = uuid.uuid4().hex
            payload = {
                "job_id": job_id,
                "images": [_as_array(image) for image in images],
                "top_k": top_k,
            }
            try:
                self._request_queue.put(payload, timeout=max(0.001, deadline - time.monotonic()))
            except queue.Full as exc:
                raise CandidateTimeout("candidate request queue is full") from exc
            response = self._next_response_locked(deadline)
            if response.get("kind") != "result" or response.get("job_id") != job_id:
                detail = response.get("error_type", response.get("kind", "unknown"))
                raise CandidateUnavailable(f"candidate worker failed: {detail}")
            return list(response["results"]), time.monotonic() - started
        except CandidateTimeout:
            self._terminate_locked()
            raise
        finally:
            self._lock.release()

    def _ensure_started_locked(self, deadline: float) -> None:
        if self._process is not None and self._process.is_alive() and self._ready:
            return
        self._terminate_locked()
        self._request_queue = self._context.Queue(maxsize=1)
        self._response_queue = self._context.Queue(maxsize=1)
        self._process = self._context.Process(
            target=_candidate_process_main,
            args=(
                str(self.runtime_dir), self.device, self.max_batch_size,
                self._request_queue, self._response_queue,
            ),
            daemon=True,
            name="clinic-candidate-classifier",
        )
        self._process.start()
        response = self._next_response_locked(deadline)
        if response.get("kind") != "ready":
            self._terminate_locked()
            detail = response.get("error_type", response.get("kind", "unknown"))
            raise CandidateUnavailable(f"candidate startup failed: {detail}")
        self._ready = True

    def _next_response_locked(self, deadline: float) -> dict[str, Any]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise CandidateTimeout("candidate inference timed out")
        try:
            return self._response_queue.get(timeout=remaining)
        except queue.Empty as exc:
            raise CandidateTimeout("candidate inference timed out") from exc

    def _terminate_locked(self) -> None:
        process = self._process
        self._process = None
        self._ready = False
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
            if process.is_alive():  # pragma: no cover - platform dependent
                process.kill()
                process.join(timeout=1.0)
        for mp_queue in (self._request_queue, self._response_queue):
            if mp_queue is not None:
                mp_queue.cancel_join_thread()
                mp_queue.close()
        self._request_queue = None
        self._response_queue = None

    def close(self) -> None:
        if not self._lock.acquire(timeout=0.1):
            return
        try:
            self._terminate_locked()
        finally:
            self._lock.release()


class AggregateRolloutMetrics:
    """Thread-safe, in-memory counters; never stores inputs or predictions."""

    _NAMES = (
        "requests", "shadow_submitted", "shadow_skipped", "shadow_queue_full",
        "shadow_submit_error",
        "candidate_completed", "candidate_errors", "candidate_timeouts",
        "candidate_fallbacks", "canary_selected", "top1_disagreements",
        "rejection_disagreements",
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters = {name: 0 for name in self._NAMES}
        self._latency_sum = 0.0
        self._latency_max = 0.0

    def increment(self, name: str) -> None:
        with self._lock:
            self._counters[name] += 1

    def observe_candidate(
        self,
        latency: float,
        primary: Sequence[Mapping[str, Any]] | None,
        candidate: Sequence[Mapping[str, Any]],
    ) -> None:
        with self._lock:
            self._counters["candidate_completed"] += 1
            self._latency_sum += latency
            self._latency_max = max(self._latency_max, latency)
            if primary is not None and len(primary) == len(candidate):
                self._counters["top1_disagreements"] += sum(
                    left.get("class_name") != right.get("class_name")
                    for left, right in zip(primary, candidate)
                )
                self._counters["rejection_disagreements"] += sum(
                    bool(left.get("rejected")) != bool(right.get("rejected"))
                    for left, right in zip(primary, candidate)
                )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            completed = self._counters["candidate_completed"]
            result = dict(self._counters)
            result["candidate_latency_seconds_mean"] = self._latency_sum / completed if completed else 0.0
            result["candidate_latency_seconds_max"] = self._latency_max
            return result


def validate_offline_report(
    report_path: Path | None, runtime_dir: Path, repo_root: Path
) -> tuple[bool, str | None]:
    """Delegate all immutable bindings and metric gates to the E2E harness."""

    if report_path is None or not report_path.is_file():
        return False, "offline_report_missing"
    manifest_path = runtime_dir.parent / "manifest.json"
    if not manifest_path.is_file():
        return False, "candidate_manifest_missing"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, "candidate_manifest_invalid_json"
    if not isinstance(manifest, dict):
        return False, "candidate_manifest_invalid_shape"
    try:
        from scripts.evaluate_r2_e2e import E2EEvaluationError, validate_promotion_report
        validate_promotion_report(
            report_path,
            manifest=manifest,
            manifest_path=manifest_path,
            repo_root=repo_root,
        )
    except (E2EEvaluationError, OSError, ValueError, TypeError):
        return False, "offline_report_rejected"
    return True, None


class ClassifierRollout:
    """Route requests without changing the public classification contract."""

    def __init__(
        self,
        settings: Settings,
        primary_provider: Callable[[], Any],
        *,
        registry: Any | None = None,
        client_factory: Callable[[Path, Settings], Any] = CandidateProcessClient,
    ) -> None:
        self.settings = settings
        self._primary_provider = primary_provider
        self._registry = registry
        self._client_factory = client_factory
        self._candidate_lock = threading.Lock()
        self._candidate_runtime_dir: Path | None = None
        self._candidate_client: Any | None = None
        self._candidate_blocker: str | None = None
        self._degraded_reason: str | None = None
        self.metrics = AggregateRolloutMetrics()
        self._shadow_executor = None
        self._shadow_slots = threading.BoundedSemaphore(
            settings.classifier_shadow_queue_size + settings.classifier_shadow_max_inflight
        )
        self._shadow_condition = threading.Condition()
        self._shadow_active = 0

    @property
    def requested_mode(self) -> str:
        return self.settings.classifier_rollout_mode

    def classify(self, image: Image.Image | np.ndarray, **kwargs):
        results = self._route([image], top_k=int(kwargs.get("top_k", 3)), single=(image, kwargs))
        return results[0]

    def classify_batch(self, images: list[Image.Image | np.ndarray]):
        return self._route(images, top_k=3, single=None)

    def _route(self, images, *, top_k: int, single):
        self.metrics.increment("requests")
        if self.requested_mode == "off":
            return self._primary(images, single)
        if self.requested_mode == "shadow":
            primary = self._primary(images, single)
            try:
                digest = content_digest(images)
                if selected_by_percent(digest, self.settings.classifier_shadow_sample_percent):
                    self._submit_shadow(images, top_k=top_k, primary=primary)
                else:
                    self.metrics.increment("shadow_skipped")
            except Exception as exc:
                self._mark_failure(f"shadow_dispatch_{type(exc).__name__}")
            return primary
        if not self._canary_is_unlocked():
            self.metrics.increment("candidate_fallbacks")
            return self._primary(images, single)
        try:
            digest = content_digest(images)
        except Exception as exc:
            self._mark_failure(f"canary_hash_{type(exc).__name__}")
            self.metrics.increment("candidate_fallbacks")
            return self._primary(images, single)
        if not selected_by_percent(digest, self.settings.classifier_canary_percent):
            return self._primary(images, single)
        self.metrics.increment("canary_selected")
        try:
            candidate, latency = self._infer_candidate(images, top_k=top_k)
            self.metrics.observe_candidate(latency, None, candidate)
            return candidate
        except CandidateTimeout:
            self._mark_failure("candidate_timeout", timeout=True)
        except Exception as exc:
            self._mark_failure(f"candidate_{type(exc).__name__}")
        self.metrics.increment("candidate_fallbacks")
        return self._primary(images, single)

    def _primary(self, images, single):
        classifier = self._primary_provider()
        if single is not None:
            image, kwargs = single
            return [classifier.classify(image, **kwargs)]
        return classifier.classify_batch(images)

    def _registry_instance(self):
        if self._registry is None:
            from backend.services.model_registry import ModelRegistry
            self._registry = ModelRegistry(
                self.settings.model_registry_dir,
                repo_root=self.settings.backend_root.parent,
                runtime_model_dir=self.settings.backend_root / "codex_model",
            )
        return self._registry

    def _ensure_candidate(self):
        if self.requested_mode == "off":
            raise CandidateUnavailable("candidate rollout is off")
        with self._candidate_lock:
            if self._candidate_client is not None:
                return self._candidate_client
            try:
                runtime_dir = Path(
                    self._registry_instance().resolve_runtime_package(
                        self.settings.classifier_candidate_reference
                    )
                )
                client = self._client_factory(runtime_dir, self.settings)
            except Exception as exc:
                self._candidate_blocker = f"candidate_package_{type(exc).__name__}"
                raise CandidateUnavailable(self._candidate_blocker) from exc
            self._candidate_runtime_dir = runtime_dir
            self._candidate_client = client
            self._candidate_blocker = None
            return client

    def _infer_candidate(self, images, *, top_k: int):
        results, latency = self._ensure_candidate().infer(images, top_k=top_k)
        if len(results) != len(images):
            raise CandidateUnavailable("candidate result count does not match request")
        return results, latency

    def _submit_shadow(self, images, *, top_k: int, primary) -> None:
        if not self._shadow_slots.acquire(blocking=False):
            self.metrics.increment("shadow_queue_full")
            self._degraded_reason = "shadow_queue_full"
            return
        if self._shadow_executor is None:
            from concurrent.futures import ThreadPoolExecutor
            self._shadow_executor = ThreadPoolExecutor(
                max_workers=self.settings.classifier_shadow_max_inflight,
                thread_name_prefix="clinic-shadow",
            )
        with self._shadow_condition:
            self._shadow_active += 1
        self.metrics.increment("shadow_submitted")
        try:
            future = self._shadow_executor.submit(
                self._shadow_task,
                [_as_array(image) for image in images],
                top_k,
                [dict(result) for result in primary],
            )
        except Exception:
            # Submission failed (executor shutdown, RuntimeError): release the
            # acquired slot and roll back the active counter so capacity is not
            # permanently leaked and wait_for_shadow() does not hang.
            self.metrics.increment("shadow_submit_error")
            self._shadow_slots.release()
            with self._shadow_condition:
                self._shadow_active -= 1
                self._shadow_condition.notify_all()
            return
        def done(_future):
            self._shadow_slots.release()
            with self._shadow_condition:
                self._shadow_active -= 1
                self._shadow_condition.notify_all()
        future.add_done_callback(done)

    def _shadow_task(self, images, top_k: int, primary) -> None:
        try:
            candidate, latency = self._infer_candidate(images, top_k=top_k)
            self.metrics.observe_candidate(latency, primary, candidate)
        except CandidateTimeout:
            self._mark_failure("candidate_timeout", timeout=True)
        except Exception as exc:
            self._mark_failure(f"candidate_{type(exc).__name__}")

    def _mark_failure(self, reason: str, *, timeout: bool = False) -> None:
        self._degraded_reason = reason
        self.metrics.increment("candidate_timeouts" if timeout else "candidate_errors")

    def _canary_is_unlocked(self) -> bool:
        try:
            self._ensure_candidate()
        except CandidateUnavailable:
            return False
        valid, blocker = validate_offline_report(
            self.settings.classifier_offline_report_path,
            self._candidate_runtime_dir,
            self.settings.backend_root.parent,
        )
        self._candidate_blocker = blocker
        return valid

    def readiness(self) -> dict[str, Any]:
        requested = self.requested_mode
        if requested == "off":
            return {
                "requested_mode": "off", "active_mode": "off", "degraded": False,
                "candidate_reference": self.settings.classifier_candidate_reference,
                "worker_state": "not_started", "blocker": None,
                "metrics": self.metrics.snapshot(),
            }
        try:
            client = self._ensure_candidate()
        except CandidateUnavailable:
            client = None
        if requested == "canary" and client is not None:
            self._canary_is_unlocked()
        blocker = self._candidate_blocker or self._degraded_reason
        return {
            "requested_mode": requested,
            "active_mode": requested if blocker is None else "off",
            "degraded": blocker is not None,
            "candidate_reference": self.settings.classifier_candidate_reference,
            "candidate_runtime_dir": str(self._candidate_runtime_dir) if self._candidate_runtime_dir else None,
            "candidate_device": self.settings.classifier_candidate_device,
            "canary_percent": self.settings.classifier_canary_percent,
            "shadow_sample_percent": self.settings.classifier_shadow_sample_percent,
            "worker_state": client.state if client is not None else "not_started",
            "blocker": blocker,
            "metrics": self.metrics.snapshot(),
        }

    def wait_for_shadow(self, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        with self._shadow_condition:
            while self._shadow_active:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._shadow_condition.wait(remaining)
        return True

    def close(self) -> None:
        if self._shadow_executor is not None:
            self._shadow_executor.shutdown(wait=False, cancel_futures=True)
        if self._candidate_client is not None:
            self._candidate_client.close()
