"""Opt-in HTTP -> GPU -> candidate check using existing local approved images only.

Set CLINIC_RETRAIN_E2E_SOURCE_BACKEND and ADMIN_TRAINING_SNAPSHOT_DIR to run.
The source annotations and active weights are read-only; outputs stay in this repo.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

import pytest
import torch
from PIL import Image
from werkzeug.serving import make_server

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.config import Settings
from backend.app.factory import create_app
from codex_model.classifier import CodexClassifier
from backend.codex_pipeline.scripts.precompute_embeddings import load_backbone


def test_existing_annotations_retrain_through_real_http(monkeypatch):
    source = os.environ.get("CLINIC_RETRAIN_E2E_SOURCE_BACKEND")
    snapshot = os.environ.get("ADMIN_TRAINING_SNAPSHOT_DIR")
    if not source or not snapshot:
        pytest.skip("requires explicit local corpus and snapshot; runs real training")
    source = Path(source).resolve()

    class LocalSettings(Settings):
        @property
        def annotations_dir(self):
            return source / "annotations"

        @property
        def classifier_weights_dir(self):
            return source / "codex_model" / "weights"

        @property
        def class_config_path(self):
            return source / "codex_model" / "config.json"

    settings = LocalSettings(
        testing=True, enable_admin_training_jobs=True,
        admin_training_snapshot_dir=snapshot,
        admin_training_backbone_manifest=str(source / "training_corpus/backbone-pins/dinov2-vits14-local.json"),
    )
    protected = [settings.class_config_path, settings.classifier_weights_dir / "projection.pt",
                 settings.classifier_weights_dir / "prototypes.pt", settings.annotations_dir / "review-index.json"]
    hashes = lambda: {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    before = hashes()
    server = make_server("127.0.0.1", 0, create_app(settings=settings), threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"

    def request(path, payload=None):
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(base_url + path, data=data,
                                     headers={"Content-Type": "application/json", "Origin": base_url})
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.status, json.load(response)

    def wait_job():
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            _, response = request("/admin/training/jobs/latest")
            job = response["job"]
            if job["status"] != "running":
                assert job["status"] == "succeeded", job.get("log_tail")
                return job
            time.sleep(0.5)
        pytest.fail("real retraining did not finish within 180 seconds")

    try:
        _, summary = request("/admin/training/summary")
        assert summary["launch_allowed_for_request"], summary["launch_disabled_reasons"]
        assert summary["training_snapshot"]["live_train_count"] > 0
        for dry_run in (True, False):
            status, _ = request("/admin/training/jobs", {"dry_run": dry_run, "device": "cuda", "batch_size": 16})
            assert status == 202
            job = wait_job()
        version = Path(job["candidate_version_dir"])
        base = torch.load(settings.classifier_weights_dir / "prototypes.pt", weights_only=True)
        candidate = torch.load(version / "runtime/weights/prototypes.pt", weights_only=True)
        assert torch.equal(base["class_labels"], candidate["class_labels"])
        assert base["class_names"] == candidate["class_names"]
        base_projection = torch.load(settings.classifier_weights_dir / "projection.pt", weights_only=True)
        projection = torch.load(version / "runtime/weights/projection.pt", weights_only=True)
        assert all(torch.equal(base_projection[key], projection[key]) for key in base_projection)
        provenance = json.loads((version / "prototypes/provenance.json").read_text())["prototype_update"]
        touched = set(provenance["updated_class_names"])
        for index, label in enumerate(base["class_labels"].tolist()):
            if base["class_names"][label] not in touched:
                assert torch.equal(base["prototypes"][index], candidate["prototypes"][index])
        metrics = {}
        inventory = {item["path"]: item["sha256"] for item in json.loads((version / "manifest.json").read_text())["artifacts"]}
        for split in ("dev", "locked_test"):
            report_path = version / f"evaluation/{split}.json"
            assert inventory[f"evaluation/{split}.json"] == hashlib.sha256(report_path.read_bytes()).hexdigest()
            benchmark = json.loads((version / f"evaluation/{split}.json").read_text())
            assert benchmark["prototype_mode"] == "stored"
            assert benchmark["eval_split"] == split
            assert benchmark["numeric_label_contract"]["equal"]
            metrics[split] = {name: {key: values[key] for key in ("top1_micro", "top3_micro", "eval_examples")}
                              for name, values in benchmark["models"].items()}
        # Use the same real, hash-verified local DINO model without a network lookup.
        backbone, _ = load_backbone("dinov2_vits14", torch.device("cuda"), settings.admin_training_backbone_manifest_path)
        monkeypatch.setattr(torch.hub, "load", lambda *_args, **_kwargs: backbone)
        candidate_model = CodexClassifier(version / "runtime", device="cuda")
        active_model = CodexClassifier(settings.classifier_weights_dir, device="cuda")
        manifest = json.loads((Path(snapshot) / "snapshot_manifest.json").read_text())
        live_rows = [row for row in manifest["rows"] if row["row_id"] in provenance["live_annotation_usage"]["training_row_ids"]]
        images = []
        for row in live_rows:
            with Image.open(Path(snapshot) / row["output_path"]) as image:
                images.append(image.convert("RGB"))
        predictions = candidate_model.classify_batch(images)
        active_predictions = active_model.classify_batch(images)
        cache = torch.load(version / "training_data/precomputed/features.pt", weights_only=True)
        live_indices = [cache["row_ids"].index(row["row_id"]) for row in live_rows]
        for model, results in ((active_model, active_predictions), (candidate_model, predictions)):
            with torch.no_grad():
                cached_embeddings = model._projection(cache["features"][live_indices].to(model.device))
                cached_scores = cached_embeddings @ model._prototypes.t()
            for scores, result in zip(cached_scores, results):
                assert model._label_index[scores.argmax().item()] == result["class_label"]
                assert scores.max().item() == pytest.approx(result["confidence"], abs=2e-5)
        live_result = {"count": len(live_rows), "evaluation_role": "training-fit diagnostic, not holdout"}
        for name, results in (("runtime", active_predictions), ("candidate", predictions)):
            live_result[name] = sum(row["class_name"] == result["class_name"] for row, result in zip(live_rows, results))
            assert all(base["class_names"][result["class_label"]] == result["class_name"] for result in results)
        assert hashes() == before
        report = {"http_dry_run_and_full_run": "passed", "candidate_version_dir": str(version),
                  "snapshot": summary["training_snapshot"], "prototype_update": provenance,
                  "active_runtime_and_reviews_unchanged": True, "holdout_metrics": metrics,
                  "real_image_inference": live_result, "cached_real_image_parity": True}
        output = version / "evaluation/http-e2e.json"
        output.parent.mkdir(exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"HTTP_E2E_REPORT={output}")
    finally:
        server.shutdown()
        thread.join(timeout=5)
