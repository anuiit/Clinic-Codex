from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from backend.app.config import Settings
from backend.app.errors import ModelAssetUnavailable
from backend.app.factory import create_app


@dataclass
class Proposal:
    bbox: list[float]
    crop: object | None


class PoisonServices:
    def get_classifier(self):  # pragma: no cover - should not be called by /ready
        raise AssertionError("/ready must not instantiate classifier")

    def get_segmenter(self):  # pragma: no cover - should not be called by /ready
        raise AssertionError("/ready must not instantiate segmenter")

    def classify(self, image, **kwargs):  # pragma: no cover - should not be called by /ready
        raise AssertionError("/ready must not classify")

    def classify_batch(self, images):  # pragma: no cover - should not be called by /ready
        raise AssertionError("/ready must not batch classify")

    def segment_page(self, image):  # pragma: no cover - should not be called by /ready
        raise AssertionError("/ready must not segment")


class MissingAssetServices:
    def __init__(self, asset="classifier_prototypes"):
        self.asset = asset

    def classify(self, image, **kwargs):
        raise ModelAssetUnavailable(self.asset, "/offline/missing.pt", "Prepare local model assets before retrying.")

    def classify_batch(self, images):
        raise ModelAssetUnavailable(self.asset, "/offline/missing.pt", "Prepare local model assets before retrying.")

    def segment_page(self, image):
        if self.asset == "mobile_sam_checkpoint":
            raise ModelAssetUnavailable(self.asset, "/offline/mobile_sam.pt", "Prepare local MobileSAM checkpoint before retrying.")
        return [Proposal([1, 2, 3, 4], Image.new("RGB", (2, 2)))]


class RuntimeBoomServices:
    def segment_page(self, image):
        return [Proposal([1, 2, 3, 4], Image.new("RGB", (2, 2)))]

    def classify_batch(self, images):
        raise RuntimeError("not a model asset problem")


def _png_bytes(size=(8, 6)):
    buf = io.BytesIO()
    Image.new("RGB", size, color=(120, 130, 140)).save(buf, format="PNG")
    return buf.getvalue()


def _post_image(client, path, field="image"):
    return client.post(path, data={field: (io.BytesIO(_png_bytes()), "glyph.png")})


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        backend_root=tmp_path / "backend-root",
        model_dir=str(tmp_path / "weights"),
        mobile_sam_checkpoint=str(tmp_path / "mobile_sam.pt"),
        testing=True,
    )


def _write_classifier_assets(settings: Settings, *, prototypes=True, projection=True) -> None:
    settings.classifier_weights_dir.mkdir(parents=True, exist_ok=True)
    if prototypes:
        (settings.classifier_weights_dir / "prototypes.pt").write_bytes(b"proto")
    if projection:
        (settings.classifier_weights_dir / "projection.pt").write_bytes(b"proj")


def _checks_by_name(body):
    return {check["name"]: check for check in body["checks"]}


def test_ready_reports_all_assets_present_without_instantiating_models(tmp_path):
    settings = _settings(tmp_path)
    _write_classifier_assets(settings)
    settings.mobile_sam_checkpoint_path.write_bytes(b"sam")
    app = create_app(settings=settings, services=PoisonServices())

    with app.test_client() as client:
        resp = client.get("/ready")

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ready"] is True
    assert body["status"] == "ready"
    checks = _checks_by_name(body)
    assert checks["classifier_prototypes"]["available"] is True
    assert checks["classifier_projection"]["available"] is True
    assert checks["mobile_sam_checkpoint"]["available"] is True


def test_ready_reports_missing_classifier_prototype(tmp_path):
    settings = _settings(tmp_path)
    _write_classifier_assets(settings, prototypes=False, projection=True)
    settings.mobile_sam_checkpoint_path.write_bytes(b"sam")
    app = create_app(settings=settings, services=PoisonServices())

    with app.test_client() as client:
        resp = client.get("/ready")

    assert resp.status_code == 503
    body = resp.get_json()
    assert body["ready"] is False
    check = _checks_by_name(body)["classifier_prototypes"]
    assert check["available"] is False
    assert check["path"].endswith("prototypes.pt")
    assert check["hint"]


def test_ready_reports_missing_classifier_projection(tmp_path):
    settings = _settings(tmp_path)
    _write_classifier_assets(settings, prototypes=True, projection=False)
    settings.mobile_sam_checkpoint_path.write_bytes(b"sam")
    app = create_app(settings=settings, services=PoisonServices())

    with app.test_client() as client:
        resp = client.get("/ready")

    assert resp.status_code == 503
    check = _checks_by_name(resp.get_json())["classifier_projection"]
    assert check["available"] is False
    assert check["path"].endswith("projection.pt")


def test_ready_reports_missing_mobile_sam_without_creating_checkpoint(tmp_path):
    settings = _settings(tmp_path)
    _write_classifier_assets(settings)
    app = create_app(settings=settings, services=PoisonServices())

    with app.test_client() as client:
        resp = client.get("/ready")

    assert resp.status_code == 503
    check = _checks_by_name(resp.get_json())["mobile_sam_checkpoint"]
    assert check["available"] is False
    assert check["path"] == str(settings.mobile_sam_checkpoint_path)
    assert not settings.mobile_sam_checkpoint_path.exists()


def test_classify_returns_offline_friendly_missing_asset_error():
    app = create_app(settings=Settings(testing=True), services=MissingAssetServices())
    with app.test_client() as client:
        resp = _post_image(client, "/classify")

    assert resp.status_code == 503
    body = resp.get_json()["error"]
    assert body["code"] == "MODEL_ASSET_UNAVAILABLE"
    assert body["asset"] == "classifier_prototypes"
    assert body["path"] == "/offline/missing.pt"
    assert body["hint"]


def test_segment_returns_offline_friendly_missing_asset_error_from_batch_classification():
    app = create_app(settings=Settings(testing=True), services=MissingAssetServices())
    with app.test_client() as client:
        resp = _post_image(client, "/segment")

    assert resp.status_code == 503
    body = resp.get_json()["error"]
    assert body["code"] == "MODEL_ASSET_UNAVAILABLE"
    assert body["asset"] == "classifier_prototypes"


def test_segment_returns_offline_friendly_missing_asset_error_from_segmentation():
    app = create_app(settings=Settings(testing=True), services=MissingAssetServices("mobile_sam_checkpoint"))
    with app.test_client() as client:
        resp = _post_image(client, "/segment")

    assert resp.status_code == 503
    body = resp.get_json()["error"]
    assert body["code"] == "MODEL_ASSET_UNAVAILABLE"
    assert body["asset"] == "mobile_sam_checkpoint"


def test_unrelated_segment_runtime_errors_are_not_labeled_missing_assets():
    app = create_app(settings=Settings(testing=False), services=RuntimeBoomServices())
    with app.test_client() as client:
        resp = _post_image(client, "/segment")

    assert resp.status_code == 500
    assert resp.get_json() is None
