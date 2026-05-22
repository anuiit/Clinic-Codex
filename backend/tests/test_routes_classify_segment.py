from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image

from backend.app.config import Settings
from backend.app.factory import create_app


@dataclass
class Proposal:
    bbox: list[float]
    crop: object | None


class StubServices:
    def __init__(self):
        self.classify_calls = 0
        self.classify_batch_calls = 0
        self.segment_calls = 0

    def classify(self, image, **kwargs):
        self.classify_calls += 1
        return {"class_name": "atl", "confidence": 0.72, "rejected": False, "top_k": []}

    def classify_batch(self, images):
        self.classify_batch_calls += 1
        return [{"class_name": "atl", "confidence": 0.72, "rejected": False, "top_k": []} for _ in images]

    def segment_page(self, image):
        self.segment_calls += 1
        crop = Image.new("RGB", (2, 2), color=(255, 255, 255))
        return [Proposal([1.2, 2.8, 3.0, 4.0], crop), Proposal([9, 9, 1, 1], None)]


def _png_bytes(size=(8, 6)):
    buf = io.BytesIO()
    Image.new("RGB", size, color=(120, 130, 140)).save(buf, format="PNG")
    return buf.getvalue()


def _post_image(client, path, field="image"):
    return client.post(path, data={field: (io.BytesIO(_png_bytes()), "glyph.png")})


def test_classify_and_classify_batch_legacy_routes_are_retained():
    services = StubServices()
    app = create_app(settings=Settings(testing=True), services=services)
    with app.test_client() as client:
        missing = client.post("/classify", data={})
        assert missing.status_code == 400
        assert missing.get_json() == {"error": "No 'image' file in request"}

        happy = _post_image(client, "/classify")
        assert happy.status_code == 200
        assert happy.get_json()["class_name"] == "atl"

        missing_batch = client.post("/classify-batch", data={})
        assert missing_batch.status_code == 400
        assert missing_batch.get_json() == {"error": "No 'images' files in request"}

        happy_batch = _post_image(client, "/classify-batch", field="images")
        assert happy_batch.status_code == 200
        assert happy_batch.get_json()[0]["class_name"] == "atl"

    assert services.classify_calls == 1
    assert services.classify_batch_calls == 1


def test_segment_shape_and_per_crop_classify_without_batching():
    services = StubServices()
    app = create_app(settings=Settings(testing=True), services=services)
    with app.test_client() as client:
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
    assert body["elements"][0]["class_name"] == "atl"
    assert services.segment_calls == 1
    assert services.classify_calls == 1
    assert services.classify_batch_calls == 0
