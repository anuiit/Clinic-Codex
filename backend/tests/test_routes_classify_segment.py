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
        self.batch_images = []
        self.proposals = None
        self.last_proposals = []
        self.batch_results = None

    def classify(self, image, **kwargs):
        self.classify_calls += 1
        return {"class_name": "atl", "confidence": 0.72, "rejected": False, "top_k": []}

    def classify_batch(self, images):
        self.classify_batch_calls += 1
        self.batch_images = list(images)
        if self.batch_results is not None:
            return self.batch_results
        return [
            {
                "class_name": "atl" if idx == 0 else f"class-{idx}",
                "confidence": 0.72 - idx / 100,
                "rejected": False,
                "top_k": [],
            }
            for idx, _image in enumerate(images)
        ]

    def segment_page(self, image):
        self.segment_calls += 1
        if self.proposals is not None:
            self.last_proposals = self.proposals
            return self.proposals
        crop_a = Image.new("RGB", (2, 2), color=(255, 255, 255))
        crop_b = Image.new("RGB", (3, 3), color=(20, 20, 20))
        self.last_proposals = [
            Proposal([1.2, 2.8, 3.0, 4.0], crop_a),
            Proposal([9, 9, 1, 1], None),
            Proposal([5.9, 6.1, 7.4, 8.8], crop_b),
        ]
        return self.last_proposals


def _png_bytes(size=(8, 6)):
    buf = io.BytesIO()
    Image.new("RGB", size, color=(120, 130, 140)).save(buf, format="PNG")
    return buf.getvalue()


def _post_image(client, path, field="image"):
    return client.post(path, data={field: (io.BytesIO(_png_bytes()), "glyph.png")})


def _invalid_image_file(name="not-image.txt"):
    return (io.BytesIO(b"not an image"), name)


def _valid_image_file(name="glyph.png"):
    return (io.BytesIO(_png_bytes()), name)


INVALID_IMAGE_RESPONSE = {"error": {"code": "INVALID_IMAGE", "message": "uploaded file is not a valid image"}}


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


def test_classify_invalid_upload_returns_client_error_before_service_call():
    services = StubServices()
    app = create_app(settings=Settings(testing=True), services=services)
    with app.test_client() as client:
        resp = client.post("/classify", data={"image": _invalid_image_file()})

    assert resp.status_code == 400
    assert resp.get_json() == INVALID_IMAGE_RESPONSE
    assert services.classify_calls == 0


def test_classify_batch_invalid_upload_returns_client_error_before_service_call():
    services = StubServices()
    app = create_app(settings=Settings(testing=True), services=services)
    with app.test_client() as client:
        resp = client.post(
            "/classify-batch",
            data={"images": [_valid_image_file("valid.png"), _invalid_image_file("broken.txt")]},
        )

    assert resp.status_code == 400
    assert resp.get_json() == INVALID_IMAGE_RESPONSE
    assert services.classify_batch_calls == 0


def test_segment_shape_and_batch_classifies_valid_crops_once():
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
    assert body["num_elements"] == 2
    assert body["elements"][0]["bbox"] == [1, 2, 3, 4]
    assert body["elements"][0]["class_name"] == "atl"
    assert body["elements"][1]["bbox"] == [5, 6, 7, 8]
    assert body["elements"][1]["class_name"] == "class-1"
    assert services.segment_calls == 1
    assert services.classify_calls == 0
    assert services.classify_batch_calls == 1
    assert services.batch_images == [proposal.crop for proposal in services.last_proposals if proposal.crop is not None]


def test_segment_invalid_upload_returns_client_error_before_service_calls():
    services = StubServices()
    app = create_app(settings=Settings(testing=True), services=services)
    with app.test_client() as client:
        resp = client.post("/segment", data={"image": _invalid_image_file()})

    assert resp.status_code == 400
    assert resp.get_json() == INVALID_IMAGE_RESPONSE
    assert services.segment_calls == 0
    assert services.classify_calls == 0
    assert services.classify_batch_calls == 0


def test_segment_skips_batch_when_no_valid_crops():
    services = StubServices()
    services.proposals = [Proposal([1, 2, 3, 4], None)]
    app = create_app(settings=Settings(testing=True), services=services)
    with app.test_client() as client:
        resp = _post_image(client, "/segment")

    assert resp.status_code == 200
    assert resp.get_json() == {"num_elements": 0, "image_size": [8, 6], "elements": []}
    assert services.classify_calls == 0
    assert services.classify_batch_calls == 0


def test_segment_rejects_batch_result_count_mismatch():
    services = StubServices()
    services.batch_results = [{"class_name": "only-one", "confidence": 0.1, "rejected": True, "top_k": []}]
    app = create_app(settings=Settings(testing=True), services=services)
    with app.test_client() as client:
        resp = _post_image(client, "/segment")

    assert resp.status_code == 500
    assert resp.get_json() == {
        "error": {
            "code": "SEGMENT_CLASSIFICATION_MISMATCH",
            "message": "segment classifier result count did not match proposal count",
        }
    }
