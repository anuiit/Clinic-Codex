from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass

import pytest
from PIL import Image

from backend.app.config import Settings
from backend.app.factory import create_app
from backend.services.annotation_storage import (
    AnnotationDiskFullError,
    AnnotationPermissionError,
    AnnotationStorageError,
    decode_image_data_url,
)


@dataclass
class Proposal:
    bbox: list[float]
    crop: object | None


class ContractServices:
    def __init__(self):
        self.classify_calls = 0
        self.classify_batch_calls = 0
        self.segment_calls = 0
        self.save_exc: Exception | None = None
        self.decode_exc: Exception | None = None

    def load_classes(self):
        return {"num_classes": 2, "class_names": ["atl", "tochtli"]}

    def classify(self, image, **kwargs):
        self.classify_calls += 1
        top_k = kwargs.get("top_k", 2)
        return {
            "class_name": "atl",
            "class_label": "Water",
            "confidence": 0.72,
            "rejected": False,
            "top_k": [
                {"class_name": "atl", "class_label": "Water", "confidence": 0.72},
                {"class_name": "tochtli", "class_label": "Rabbit", "confidence": 0.31},
            ][:top_k],
        }

    def classify_batch(self, images):
        self.classify_batch_calls += 1
        return [
            {
                "class_name": "atl",
                "class_label": "Water",
                "confidence": 0.72,
                "rejected": False,
                "top_k": [],
            }
            for _ in images
        ]

    def segment_page(self, image):
        self.segment_calls += 1
        crop = Image.new("RGB", (2, 2), color=(255, 255, 255))
        return [Proposal([1.2, 2.8, 3.0, 4.0], crop), Proposal([9, 9, 1, 1], None)]

    def sample_index(self):
        return {"atl": [{"path": "/tmp/atl.png", "class_name": "atl"}]}

    def decode_annotation_image(self, data_url):
        if self.decode_exc:
            raise self.decode_exc
        return decode_image_data_url(data_url)

    def save_annotation(self, analysis_id, image, annotations):
        if self.save_exc:
            raise self.save_exc
        return {
            "status": "ok",
            "analysis_id": analysis_id,
            "saved_count": len(annotations),
            "classes": [item["class_name"] for item in annotations],
            "saved_at": "2026-05-22T00:00:00+00:00",
        }


def _png_bytes(size=(8, 6)):
    buf = io.BytesIO()
    Image.new("RGB", size, color=(120, 130, 140)).save(buf, format="PNG")
    return buf.getvalue()


def _png_base64(size=(8, 6)):
    return base64.b64encode(_png_bytes(size)).decode("ascii")


def _png_data_url(size=(10, 10)):
    return f"data:image/png;base64,{_png_base64(size)}"


def _post_image(client, path, field="image"):
    return client.post(path, data={field: (io.BytesIO(_png_bytes()), "glyph.png")})


def _valid_save_payload():
    return {
        "analysis_id": "contract-save-001",
        "image_name": "glyph.png",
        "image_data_url": _png_data_url(),
        "timestamp": 1770000000000,
        "annotations": [{"index": 0, "class_name": "atl", "bbox": [0, 0, 5, 5]}],
    }


@pytest.fixture()
def app_and_services(tmp_path):
    services = ContractServices()
    app = create_app(settings=Settings(backend_root=tmp_path, testing=True), services=services)
    return app, services


@pytest.fixture()
def client(app_and_services):
    app, _services = app_and_services
    with app.test_client() as c:
        yield c


def test_api_contract_health_and_classes(client):
    health = client.get("/health")
    assert health.status_code == 200
    assert health.get_json() == {"status": "ok"}

    classes = client.get("/classes")
    assert classes.status_code == 200
    assert classes.get_json() == {"num_classes": 2, "class_names": ["atl", "tochtli"]}


def test_api_contract_classify_and_batch_shapes(client, app_and_services):
    _app, services = app_and_services

    missing = client.post("/classify", data={})
    assert missing.status_code == 400
    assert missing.get_json() == {"error": "No 'image' file in request"}

    happy = _post_image(client, "/classify")
    assert happy.status_code == 200
    body = happy.get_json()
    assert set(body) == {"class_name", "class_label", "confidence", "rejected", "top_k"}
    assert body["class_label"] == "Water"
    assert body["top_k"][0]["class_label"] == "Water"

    missing_batch = client.post("/classify-batch", data={})
    assert missing_batch.status_code == 400
    assert missing_batch.get_json() == {"error": "No 'images' files in request"}

    batch = _post_image(client, "/classify-batch", field="images")
    assert batch.status_code == 200
    assert isinstance(batch.get_json(), list)
    assert set(batch.get_json()[0]) == {"class_name", "class_label", "confidence", "rejected", "top_k"}
    assert services.classify_calls == 1
    assert services.classify_batch_calls == 1


def test_api_contract_segment_shape_and_missing_image(client, app_and_services):
    _app, services = app_and_services

    missing = client.post("/segment", data={})
    assert missing.status_code == 400
    assert missing.get_json() == {"error": "No 'image' file in request"}

    happy = _post_image(client, "/segment")
    assert happy.status_code == 200
    body = happy.get_json()
    assert set(body) == {"num_elements", "image_size", "elements"}
    assert body["image_size"] == [8, 6]
    assert body["num_elements"] == 1
    assert body["elements"][0]["bbox"] == [1, 2, 3, 4]
    assert {"class_name", "class_label", "confidence", "rejected", "top_k"}.issubset(body["elements"][0])
    assert services.segment_calls == 1
    assert services.classify_calls == 0
    assert services.classify_batch_calls == 1


@pytest.mark.parametrize("route", ["/similar", "/trust"])
def test_api_contract_bbox_route_error_shapes(client, route):
    missing = client.post(route, json={})
    assert missing.status_code == 400
    assert missing.get_json() == {
        "error": {"code": "INVALID_REQUEST", "message": "image_base64 and bbox required"}
    }

    invalid_image = client.post(route, json={"image_base64": "bad", "bbox": [0, 0, 1, 1]})
    assert invalid_image.status_code == 400
    assert invalid_image.get_json()["error"]["code"] == "INVALID_IMAGE"

    invalid_bbox = client.post(route, json={"image_base64": _png_base64(), "bbox": [0, 0, 99, 99]})
    assert invalid_bbox.status_code == 400
    assert invalid_bbox.get_json() == {
        "error": {"code": "INVALID_BBOX", "message": "bbox out of image bounds"}
    }


def test_api_contract_similar_success_shape(client):
    resp = client.post("/similar", json={"image_base64": _png_base64(), "bbox": [0, 0, 4, 4], "limit": 2})
    assert resp.status_code == 200
    body = resp.get_json()
    assert set(body) == {"query", "best_match", "results"}
    assert body["query"] == {"bbox": [0, 0, 4, 4], "mode": "prototype"}
    assert set(body["best_match"]) == {"class_name", "similarity", "rejected"}
    assert set(body["results"][0]) == {
        "rank",
        "match_type",
        "class_name",
        "class_label",
        "similarity",
        "band",
        "asset",
    }
    assert body["results"][0]["class_label"] == "Water"


def test_api_contract_trust_success_shape(client):
    resp = client.post(
        "/trust",
        json={"image_base64": _png_base64(), "bbox": [0, 0, 4, 4], "predicted_class": "atl", "top_k": 2},
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert set(body) == {"query", "trust"}
    assert body["query"] == {"bbox": [0, 0, 4, 4], "predicted_class": "atl"}
    assert set(body["trust"]) == {
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
    assert body["trust"]["top_k"][0]["class_label"] == "Water"


def test_api_contract_save_annotation_success_and_validation_shapes(client):
    happy = client.post("/save-annotation", data=json.dumps(_valid_save_payload()), content_type="application/json")
    assert happy.status_code == 200
    assert happy.get_json() == {
        "status": "ok",
        "analysis_id": "contract-save-001",
        "saved_count": 1,
        "classes": ["atl"],
        "saved_at": "2026-05-22T00:00:00+00:00",
    }

    invalid_json = client.post("/save-annotation", data="not-json", content_type="application/json")
    assert invalid_json.status_code == 400
    assert invalid_json.get_json() == {"status": "error", "error": "invalid JSON"}

    for missing_field in ("analysis_id", "image_data_url", "annotations"):
        payload = _valid_save_payload()
        payload.pop(missing_field)
        resp = client.post("/save-annotation", json=payload)
        assert resp.status_code == 400
        assert resp.get_json() == {"status": "error", "error": f"missing field: {missing_field}"}

    empty_annotations = _valid_save_payload()
    empty_annotations["annotations"] = []
    resp = client.post("/save-annotation", json=empty_annotations)
    assert resp.status_code == 400
    assert resp.get_json() == {"status": "error", "error": "annotations must be a non-empty list"}

    blank_analysis = _valid_save_payload()
    blank_analysis["analysis_id"] = " "
    resp = client.post("/save-annotation", json=blank_analysis)
    assert resp.status_code == 400
    assert resp.get_json() == {"status": "error", "error": "missing field: analysis_id"}


def test_api_contract_save_annotation_decode_and_storage_error_shapes(client, app_and_services):
    _app, services = app_and_services

    services.decode_exc = ValueError("bad data url")
    resp = client.post("/save-annotation", json=_valid_save_payload())
    assert resp.status_code == 400
    assert resp.get_json() == {"status": "error", "error": "bad data url"}
    services.decode_exc = None

    services.save_exc = AnnotationPermissionError("denied")
    resp = client.post("/save-annotation", json=_valid_save_payload())
    assert resp.status_code == 409
    assert resp.get_json()["error_code"] == "PERMISSION_DENIED"
    assert resp.get_json()["hint"]

    services.save_exc = AnnotationDiskFullError("no space")
    resp = client.post("/save-annotation", json=_valid_save_payload())
    assert resp.status_code == 507
    assert resp.get_json()["error_code"] == "DISK_FULL"

    services.save_exc = AnnotationStorageError("storage broke")
    resp = client.post("/save-annotation", json=_valid_save_payload())
    assert resp.status_code == 500
    assert resp.get_json() == {
        "status": "error",
        "error_code": "STORAGE_ERROR",
        "message": "storage broke",
        "hint": None,
    }

    services.save_exc = RuntimeError("boom")
    resp = client.post("/save-annotation", json=_valid_save_payload())
    assert resp.status_code == 500
    body = resp.get_json()
    assert body["status"] == "error"
    assert body["error_code"] == "INTERNAL_ERROR"
    assert body["trace_id"]
    assert "id=" in body["message"]
