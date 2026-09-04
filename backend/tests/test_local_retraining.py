"""Local-prior regression checks; set CLINIC_LOCAL_RETRAIN_E2E=1 for real CPU inference."""
from __future__ import annotations

import base64
import copy
import io
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from scripts import retrain_local as local
from backend.app.config import Settings
from backend.app.factory import create_app
from backend.services.annotation_review import AnnotationReviewStore
from backend.services.annotation_storage import save_annotation

ROOT = Path(__file__).resolve().parents[2]


def test_service_import_order_keeps_route_exception_identity():
    subprocess.run([sys.executable, "-c", (
        "from scripts import retrain_local; "
        "from backend.app.services import container; "
        "from backend.services import training_jobs, annotation_review; "
        "assert container.AdminTrainingService is training_jobs.AdminTrainingService; "
        "assert container.AnnotationReviewStore is annotation_review.AnnotationReviewStore"
    )], cwd=ROOT, check=True)


def test_training_lock_excludes_concurrent_jobs_and_recovers_after_kill(tmp_path):
    code = (
        "from pathlib import Path; import sys, time; "
        "from scripts.retrain_local import training_lock; "
        "\nwith training_lock(Path(sys.argv[1])):\n print('locked', flush=True); time.sleep(60)"
    )
    process = subprocess.Popen([sys.executable, "-c", code, str(tmp_path)], cwd=ROOT,
                               stdout=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == "locked"
        with pytest.raises((OSError, RuntimeError)):
            with local.training_lock(tmp_path):
                pytest.fail("concurrent job acquired the lock")
        if os.name == "nt":
            # A Windows venv adds a redirector parent; kill the actual worker too.
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           capture_output=True, check=True, timeout=10)
        else:
            process.kill()
        process.wait(timeout=10)
        assert (tmp_path / ".retrain.lock").exists()
        with local.training_lock(tmp_path):
            assert (tmp_path / ".retrain.lock").read_text() == str(os.getpid())
        assert not (tmp_path / ".retrain.lock").exists()
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)


def prior_fixture():
    return {
        "prototypes": torch.tensor([[1., 0.], [0., 1.]]),
        "class_labels": torch.tensor([2, 8]), "class_names": {2: "atl", 8: "calli"},
        "class_meta": {2: {"count": 2, "variance": 0.2}, 8: {"count": 1}},
    }


def test_prior_sum_matches_direct_reconstruction_and_replays_idempotently():
    prior = prior_fixture()
    new = torch.tensor([[0., 1.]])
    first = local.merge_prior(prior, new, [2])
    expected = torch.nn.functional.normalize(torch.tensor([1.6, 1.]), dim=0)
    assert torch.allclose(first["prototypes"][0], expected)
    assert torch.equal(first["prototypes"][1], prior["prototypes"][1])
    assert torch.equal(first["class_labels"], prior["class_labels"])
    assert torch.equal(local.merge_prior(prior, new, [2])["prototypes"], first["prototypes"])
    assert torch.equal(local.merge_prior(prior, torch.empty(0, 2), [])["prototypes"], prior["prototypes"])
    assert prior["class_meta"][2]["count"] == 2


@pytest.mark.parametrize("mutation", ["count", "variance", "nan", "unknown"])
def test_prior_rejects_invalid_inputs(mutation):
    prior = prior_fixture()
    embeddings, labels = torch.tensor([[1., 0.]]), [2]
    if mutation == "count":
        prior["class_meta"][8]["count"] = 0
    elif mutation == "variance":
        prior["class_meta"][2]["variance"] = float("nan")
    elif mutation == "nan":
        embeddings[0, 0] = float("nan")
    else:
        labels = [100]
    with pytest.raises(ValueError):
        local.merge_prior(prior, embeddings, labels)


def approved_store(tmp_path, second_class="atl"):
    annotations = tmp_path / "annotations"
    image = Image.new("RGB", (24, 24), "red")
    store = AnnotationReviewStore(annotations)
    for index, name in enumerate(["atl", second_class]):
        analysis = f"approved-{index}"
        save_annotation(analysis, image, [{"index": 0, "class_name": name, "bbox": [0, 0, 24, 24]}],
                        base_dir=annotations, elements_dir=tmp_path / "elements")
        store.set_status(analysis, 0, "approved")
    return store


def test_capture_deduplicates_pixels_and_keeps_review_copy(tmp_path):
    store = approved_store(tmp_path)
    snapshot = local.capture_approvals(store, tmp_path / "snapshot", {2: "atl", 8: "calli"})
    assert (snapshot["approved_count"], snapshot["unique_count"], snapshot["duplicate_count"]) == (2, 1, 1)
    assert (tmp_path / "snapshot/review-index.json").read_bytes() == store.manifest_path.read_bytes()
    store.set_status("approved-0", 0, "rejected")
    next_snapshot = local.capture_approvals(store, tmp_path / "next", {2: "atl", 8: "calli"})
    assert next_snapshot["approved_count"] == 1


def test_capture_refuses_conflicting_pixels(tmp_path):
    store = approved_store(tmp_path, "calli")
    with pytest.raises(ValueError, match="conflicting"):
        local.capture_approvals(store, tmp_path / "snapshot", {2: "atl", 8: "calli"})


def test_capture_refuses_review_mutation(tmp_path, monkeypatch):
    store = approved_store(tmp_path)
    copyfile = local.shutil.copyfile
    def mutate(source, destination):
        result = copyfile(source, destination)
        store.set_status("approved-0", 0, "rejected")
        return result
    monkeypatch.setattr(local.shutil, "copyfile", mutate)
    with pytest.raises(ValueError, match="changed"):
        local.capture_approvals(store, tmp_path / "snapshot", {2: "atl", 8: "calli"})


def test_real_cpu_retraining_in_clean_checkout(tmp_path):
    if os.environ.get("CLINIC_LOCAL_RETRAIN_E2E") != "1":
        pytest.skip("opt-in: downloads pinned public backbone if absent, runs real CPU inference")
    clone = tmp_path / "fresh-clone"
    clone.mkdir()
    files = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
    ).decode().split("\0")
    for relative in files:
        if not relative:
            continue
        source = ROOT / relative
        if source.is_file():
            target = clone / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    backend = clone / "backend"
    assert not (backend / "training_corpus").exists()
    assert not (backend / "annotations/review-index.json").exists()
    subprocess.run([
        sys.executable, "-m", "codex_pipeline.scripts.export_model", "--allow-runtime-write",
        "--prototypes", "prototypes/prototypes.pt", "--weights-dir", "codex_model/weights",
        "--config-template", "codex_model/config.json", "--config-out", "codex_model/config.json",
    ], cwd=backend, check=True)
    pin = backend / "training_corpus/backbone-pins/dinov2-vits14-local.json"
    subprocess.run([sys.executable, "scripts/pin_dinov2.py", "--download", "--output", str(pin)],
                   cwd=clone, check=True)
    settings = Settings(
        backend_root=backend, testing=True, enable_admin_training_jobs=True,
        auth_required=True, auth_secret_key="isolated-e2e-secret", auth_cookie_secure=False,
        allow_local_admin_self_review=True,
    )
    app = create_app(settings=settings)
    client = app.test_client()
    created = client.post("/auth/bootstrap", json={
        "email": "fresh-user@example.test", "password": "fresh-user-test-password",
    })
    assert created.status_code == 201
    login = client.post("/auth/login", json={
        "email": "fresh-user@example.test", "password": "fresh-user-test-password",
    })
    assert login.status_code == 200
    headers = {"X-CSRF-Token": login.get_json()["csrf_token"]}
    summary = client.get("/admin/training/summary").get_json()
    assert summary["training_snapshot"]["mode"] == "local_prior"
    assert not summary["launch_allowed_for_request"]
    sample = next(path for path in (backend / "data/elements_sample").rglob("*")
                  if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
    with Image.open(sample) as image:
        image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        size = image.size
        # Confirm base inference uses installed local assets, with no private corpus.
        prediction = app.extensions["clinic_services"].get_classifier().classify(image)
    name = sample.parent.name.split("-", 1)[1]
    payload = {
        "analysis_id": "fresh-user-annotation",
        "image_data_url": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(),
        "annotations": [{"index": 0, "class_name": name, "bbox": [0, 0, *size]}],
    }
    assert client.post("/save-annotation", json=payload, headers=headers).status_code == 200
    assert client.post("/admin/annotations/fresh-user-annotation/0/review", json={"status": "approved"}, headers=headers).status_code == 200
    summary = client.get("/admin/training/summary").get_json()
    assert summary["launch_allowed_for_request"], summary["launch_disabled_reasons"]
    protected = [settings.class_config_path, settings.classifier_weights_dir / "projection.pt",
                 settings.classifier_weights_dir / "prototypes.pt", settings.annotations_dir / "review-index.json"]
    hashes = {path: local.sha256_file(path) for path in protected}
    candidates = []
    for dry_run in (True, False, False):
        response = client.post("/admin/training/jobs", json={"dry_run": dry_run, "device": "cpu", "batch_size": 1}, headers=headers)
        assert response.status_code == 202, response.get_json()
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            job = client.get("/admin/training/jobs/latest").get_json()["job"]
            if job["status"] != "running":
                break
            time.sleep(0.25)
        assert job["status"] == "succeeded", job
        if not dry_run:
            version = Path(job["candidate_version_dir"])
            manifest = json.loads((version / "manifest.json").read_text())
            assert manifest["promotion"]["blocked"] is True
            assert job["result"]["unique_count"] == 1
            assert 0 <= job["result"]["candidate_correct"] <= 1
            assert job["result"]["generalization_validated"] is False
            candidates.append(torch.load(version / "runtime/weights/prototypes.pt", weights_only=True))
    assert torch.equal(candidates[0]["prototypes"], candidates[1]["prototypes"])
    assert hashes == {path: local.sha256_file(path) for path in protected}
    print(f"FRESH_CLONE={clone}")
    print(json.dumps(job["result"]))
