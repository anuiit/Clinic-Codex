from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

from backend.app.config import Settings
from backend.app.factory import create_app


class StubServices:
    def __init__(self):
        self.calls = []

    def classify(self, image, **kwargs):
        self.calls.append(kwargs)
        return {
            "class_name": "atl",
            "confidence": 0.72,
            "rejected": False,
            "top_k": [
                {"class_name": "atl", "class_label": "Water", "confidence": 0.72},
                {"class_name": "tochtli", "class_label": "Rabbit", "confidence": 0.31},
            ],
        }

    def sample_index(self):
        return {"atl": [{"path": "/tmp/atl.png", "class_name": "atl"}]}


def _png_base64(size=(8, 6)):
    buf = io.BytesIO()
    Image.new("RGB", size, color=(120, 130, 140)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.fixture()
def client():
    app = create_app(settings=Settings(testing=True), services=StubServices())
    with app.test_client() as c:
        yield c


@pytest.mark.parametrize("route", ["/similar", "/trust"])
def test_invalid_request_image_and_bbox_shapes(client, route):
    assert client.post(route, json={}).get_json() == {
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


def test_similar_happy_path_shape(client):
    resp = client.post("/similar", json={"image_base64": _png_base64(), "bbox": [0, 0, 4, 4]})
    assert resp.status_code == 200
    body = resp.get_json()
    assert set(body) == {"query", "best_match", "results"}
    assert body["query"] == {"bbox": [0, 0, 4, 4], "mode": "prototype"}
    assert set(body["best_match"]) == {"class_name", "similarity", "rejected"}
    assert body["results"][0]["band"] == "high"


@pytest.mark.parametrize("limit", [0, 51, "many", True])
def test_similar_rejects_invalid_limit(client, limit):
    resp = client.post("/similar", json={"image_base64": _png_base64(), "bbox": [0, 0, 4, 4], "limit": limit})
    assert resp.status_code == 400
    assert resp.get_json()["error"]["code"] == "INVALID_REQUEST"
    assert "limit must" in resp.get_json()["error"]["message"]


def test_trust_happy_path_shape(client):
    resp = client.post(
        "/trust",
        json={"image_base64": _png_base64(), "bbox": [0, 0, 4, 4], "predicted_class": "atl"},
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert set(body) == {"query", "trust"}
    assert body["trust"]["predicted_class_rank"] == 1
    assert body["trust"]["top1_class"] == "atl"
    assert "top_k" in body["trust"]


def test_trust_missing_predicted_class_defaults_to_classifier_top_class(client):
    resp = client.post("/trust", json={"image_base64": _png_base64(), "bbox": [0, 0, 4, 4]})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["query"]["predicted_class"] == "atl"
    assert body["trust"]["predicted_class_rank"] == 1


@pytest.mark.parametrize("top_k", [0, 51, "many", False])
def test_trust_rejects_invalid_top_k(client, top_k):
    resp = client.post("/trust", json={"image_base64": _png_base64(), "bbox": [0, 0, 4, 4], "top_k": top_k})
    assert resp.status_code == 400
    assert resp.get_json()["error"]["code"] == "INVALID_REQUEST"
    assert "top_k must" in resp.get_json()["error"]["message"]


@pytest.mark.parametrize("route", ["/similar", "/trust"])
def test_routes_reject_non_numeric_bbox_values(client, route):
    resp = client.post(route, json={"image_base64": _png_base64(), "bbox": [0, "top", 4, 4]})
    assert resp.status_code == 400
    assert resp.get_json() == {
        "error": {"code": "INVALID_BBOX", "message": "bbox values must be numeric"}
    }
