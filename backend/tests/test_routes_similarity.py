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

    def load_classes(self):
        return {"num_classes": 2, "class_names": ["atl", "tochtli"]}


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


@pytest.mark.parametrize("route", ["/similar", "/trust"])
def test_crop_routes_reject_images_above_configured_pixel_limit(route):
    services = StubServices()
    app = create_app(settings=Settings(testing=True, max_image_pixels=4, max_image_dimension=100), services=services)
    with app.test_client() as client:
        resp = client.post(route, json={"image_base64": _png_base64(size=(8, 6)), "bbox": [0, 0, 4, 4]})

    assert resp.status_code == 400
    body = resp.get_json()
    assert body["error"]["code"] == "INVALID_IMAGE"
    assert "maximum size" in body["error"]["message"]
    assert services.calls == []


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


@pytest.mark.parametrize("route", ["/similar", "/trust"])
@pytest.mark.parametrize(
    "bbox",
    [
        [0, 0, 0, 10],
        [0, 0, 10, 0],
        [0, 0, -5, 10],
        [0, 0, 10, -5],
    ],
)
def test_crop_routes_reject_non_positive_bbox_dimensions(client, route, bbox):
    resp = client.post(route, json={"image_base64": _png_base64(size=(12, 12)), "bbox": bbox})

    assert resp.status_code == 400
    assert resp.get_json()["error"]["code"] == "INVALID_BBOX"


def test_similar_fills_asset_from_sample_index(client):
    resp = client.post("/similar", json={"image_base64": _png_base64(), "bbox": [0, 0, 4, 4]})
    assert resp.status_code == 200
    results = resp.get_json()["results"]
    assert results[0]["class_name"] == "atl"
    assert results[0]["asset"] == "/samples/atl/atl.png"
    # tochtli has no entry in the stub sample index: asset stays null.
    assert results[1]["class_name"] == "tochtli"
    assert results[1]["asset"] is None


def test_samples_coverage_reports_class_exemplar_presence(client):
    resp = client.get("/samples/coverage")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["total_classes"] == 2
    assert body["covered_classes"] == 1
    assert body["covered"] == ["atl"]


class FileBackedSampleServices(StubServices):
    def __init__(self, sample_file):
        super().__init__()
        self._sample_file = sample_file

    def sample_index(self):
        return {"atl": [{"path": str(self._sample_file), "class_name": "atl"}]}


def test_sample_image_by_id_serves_indexed_file(tmp_path):
    backend_root = tmp_path / "backend"
    sample = backend_root / "data" / "elements_sample" / "0001-atl" / "atl.png"
    sample.parent.mkdir(parents=True)
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(sample, format="PNG")

    app = create_app(
        settings=Settings(testing=True, backend_root=backend_root),
        services=FileBackedSampleServices(sample),
    )
    with app.test_client() as client:
        resp = client.get("/samples/atl/atl.png")
        assert resp.status_code == 200
        assert resp.data == sample.read_bytes()


@pytest.mark.parametrize(
    "url",
    [
        "/samples/tochtli/atl.png",  # wrong class for this file
        "/samples/atl/missing.png",  # unknown filename
        "/samples/atl/..%2F..%2Fetc%2Fpasswd",  # traversal attempt
        "/samples/atl/.hidden.png",  # dotfile rejected
    ],
)
def test_sample_image_by_id_rejects_non_indexed_paths(tmp_path, url):
    backend_root = tmp_path / "backend"
    sample = backend_root / "data" / "elements_sample" / "0001-atl" / "atl.png"
    sample.parent.mkdir(parents=True)
    Image.new("RGB", (4, 4)).save(sample, format="PNG")

    app = create_app(
        settings=Settings(testing=True, backend_root=backend_root),
        services=FileBackedSampleServices(sample),
    )
    with app.test_client() as client:
        assert client.get(url).status_code == 404
