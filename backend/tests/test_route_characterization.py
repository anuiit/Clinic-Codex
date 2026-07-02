# type: ignore
# pyright: reportMissingImports=false
"""Characterization coverage for the Phase 0 Flask API routes.

These tests intentionally exercise the legacy import path (`examples.flask_api`)
so they can be written before Phase 1 route extraction and remain as parity
coverage once the module becomes a compatibility shim.
"""
from __future__ import annotations

import base64
import importlib
import io
import os
import sys
import types
from dataclasses import dataclass

import pytest
from PIL import Image


@dataclass
class StubProposal:
    bbox: list[float]
    crop: object | None


class StubCodexClassifier:
    classify_calls = 0
    classify_batch_calls = 0

    def __init__(self, *args, **kwargs):
        pass

    def classify(self, *args, **kwargs):
        type(self).classify_calls += 1
        return {
            "class_name": "atl",
            "class_label": "Water",
            "confidence": 0.72,
            "rejected": False,
            "top_k": [
                {"class_name": "atl", "class_label": "Water", "confidence": 0.72},
                {"class_name": "tochtli", "class_label": "Rabbit", "confidence": 0.31},
            ],
        }

    def classify_batch(self, images):
        type(self).classify_batch_calls += 1
        return [
            {
                "class_name": "atl",
                "confidence": 0.72,
                "rejected": False,
                "top_k": [],
            }
            for _ in images
        ]


class StubMobileSAMSegmenter:
    init_calls = 0

    def __init__(self, *args, **kwargs):
        type(self).init_calls += 1

    def segment_page(self, img):
        return ["proposal-1", "proposal-2"]

    def extract_crops(self, img, proposals):
        crop = Image.new("RGB", (2, 2), color=(255, 255, 255))
        return [
            StubProposal([1.2, 2.8, 3.0, 4.0], crop),
            StubProposal([9, 9, 1, 1], None),
        ]


def _install_ml_stubs():
    codex_model = types.ModuleType("codex_model")
    codex_model.CodexClassifier = StubCodexClassifier
    sys.modules["codex_model"] = codex_model

    codex_pipeline = types.ModuleType("codex_pipeline")
    segmentation = types.ModuleType("codex_pipeline.segmentation")
    segmentation.MobileSAMSegmenter = StubMobileSAMSegmenter
    sys.modules["codex_pipeline"] = codex_pipeline
    sys.modules["codex_pipeline.segmentation"] = segmentation


def _load_legacy_module():
    _install_ml_stubs()
    backend_root = os.path.dirname(os.path.dirname(__file__))
    if backend_root not in sys.path:
        sys.path.insert(0, backend_root)
    sys.modules.pop("examples.flask_api", None)
    return importlib.import_module("examples.flask_api")


@pytest.fixture()
def flask_mod(monkeypatch):
    StubCodexClassifier.classify_calls = 0
    StubCodexClassifier.classify_batch_calls = 0
    StubMobileSAMSegmenter.init_calls = 0
    mod = _load_legacy_module()
    mod.app.config["TESTING"] = True
    services = mod.app.extensions["clinic_services"]
    monkeypatch.setattr(services, "_raise_for_missing_classifier_assets", lambda: None)
    monkeypatch.setattr(services, "_raise_for_missing_mobile_sam_checkpoint", lambda: None)
    return mod


@pytest.fixture()
def client(flask_mod):
    with flask_mod.app.test_client() as c:
        yield c


def _png_bytes(size=(8, 6)):
    buf = io.BytesIO()
    Image.new("RGB", size, color=(120, 130, 140)).save(buf, format="PNG")
    return buf.getvalue()


def _png_base64(size=(8, 6)):
    return base64.b64encode(_png_bytes(size)).decode("ascii")


def _post_image(client, path, field="image"):
    return client.post(path, data={field: (io.BytesIO(_png_bytes()), "glyph.png")})


def assert_top_keys(body, keys):
    assert set(body.keys()) == set(keys)


def test_classes_response_shape(client):
    resp = client.get("/classes")
    assert resp.status_code == 200
    body = resp.get_json()
    assert_top_keys(body, ["num_classes", "class_names"])
    assert isinstance(body["class_names"], list)


@pytest.mark.parametrize("route", ["/similar", "/trust", "/similar-samples"])
def test_bbox_json_routes_missing_payload_shape(client, route):
    resp = client.post(route, json={})
    assert resp.status_code == 400
    assert resp.get_json() == {
        "error": {"code": "INVALID_REQUEST", "message": "image_base64 and bbox required"}
    }


@pytest.mark.parametrize("route", ["/similar", "/trust", "/similar-samples"])
def test_bbox_json_routes_invalid_image_shape(client, route):
    resp = client.post(route, json={"image_base64": "not-an-image", "bbox": [0, 0, 1, 1]})
    assert resp.status_code == 400
    body = resp.get_json()
    assert set(body["error"].keys()) == {"code", "message"}
    assert body["error"]["code"] == "INVALID_IMAGE"


@pytest.mark.parametrize("route", ["/similar", "/trust", "/similar-samples"])
def test_bbox_json_routes_invalid_bbox_shape(client, route):
    resp = client.post(route, json={"image_base64": _png_base64(), "bbox": [0, 0, 99, 99]})
    assert resp.status_code == 400
    assert resp.get_json() == {
        "error": {"code": "INVALID_BBOX", "message": "bbox out of image bounds"}
    }


def test_similar_happy_path_shape(client):
    resp = client.post("/similar", json={"image_base64": _png_base64(), "bbox": [0, 0, 4, 4]})
    assert resp.status_code == 200
    body = resp.get_json()
    assert_top_keys(body, ["query", "best_match", "results"])
    assert body["query"] == {"bbox": [0, 0, 4, 4], "mode": "prototype"}
    assert set(body["best_match"].keys()) == {"class_name", "similarity", "rejected"}
    assert set(body["results"][0].keys()) == {
        "rank",
        "match_type",
        "class_name",
        "class_label",
        "similarity",
        "band",
        "asset",
    }


def test_trust_happy_path_shape(client):
    resp = client.post(
        "/trust",
        json={"image_base64": _png_base64(), "bbox": [0, 0, 4, 4], "predicted_class": "atl"},
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert_top_keys(body, ["query", "trust"])
    assert set(body["trust"].keys()) == {
        "predicted_class_rank",
        "predicted_class_similarity",
        "top1_class",
        "top1_similarity",
        "margin_to_second",
        "above_rejection_threshold",
        "rejection_threshold",
        "ambiguous",
        "entropy",
        "top_k",
    }


def test_classify_missing_and_happy_shapes(client):
    missing = client.post("/classify", data={})
    assert missing.status_code == 400
    assert missing.get_json() == {"error": "No 'image' file in request"}

    happy = _post_image(client, "/classify")
    assert happy.status_code == 200
    assert_top_keys(happy.get_json(), ["class_name", "class_label", "confidence", "rejected", "top_k"])


def test_classify_batch_missing_and_happy_shapes(client):
    missing = client.post("/classify-batch", data={})
    assert missing.status_code == 400
    assert missing.get_json() == {"error": "No 'images' files in request"}

    happy = _post_image(client, "/classify-batch", field="images")
    assert happy.status_code == 200
    body = happy.get_json()
    assert isinstance(body, list)
    assert body and set(body[0].keys()) == {"class_name", "confidence", "rejected", "top_k"}


def test_segment_missing_and_happy_shapes_use_batch_classify(client):
    missing = client.post("/segment", data={})
    assert missing.status_code == 400
    assert missing.get_json() == {"error": "No 'image' file in request"}

    happy = _post_image(client, "/segment")
    assert happy.status_code == 200
    body = happy.get_json()
    assert_top_keys(body, ["num_elements", "image_size", "elements"])
    assert body["image_size"] == [8, 6]
    assert body["num_elements"] == 1
    assert body["elements"][0]["bbox"] == [1, 2, 3, 4]
    assert {"class_name", "confidence", "rejected", "top_k"}.issubset(body["elements"][0].keys())
    assert StubMobileSAMSegmenter.init_calls == 1
    assert StubCodexClassifier.classify_calls == 0
    assert StubCodexClassifier.classify_batch_calls == 1
