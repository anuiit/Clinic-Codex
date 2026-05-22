# type: ignore
# pyright: reportMissingImports=false
"""Tests for the /save-annotation Flask endpoint."""
from __future__ import annotations

import base64
import io
import json

import pytest
from PIL import Image as PILImage

from backend.app.config import Settings
from backend.app.factory import create_app
from backend.services.annotation_storage import (
    AnnotationDiskFullError,
    AnnotationPermissionError,
    decode_image_data_url,
)


class SaveServices:
    def __init__(self, tmp_path):
        self.tmp_path = tmp_path
        self.raise_exc: Exception | None = None

    def decode_annotation_image(self, data_url):
        return decode_image_data_url(data_url)

    def save_annotation(self, analysis_id, image, annotations):
        if self.raise_exc:
            raise self.raise_exc
        return {
            "status": "ok",
            "analysis_id": analysis_id,
            "saved_count": len(annotations),
            "classes": [annotations[0]["class_name"]],
            "saved_at": "2026-05-22T00:00:00+00:00",
        }


def _png_data_url():
    buf = io.BytesIO()
    PILImage.new("RGB", (10, 10), color=(200, 200, 200)).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def _valid_payload():
    return {
        "analysis_id": "test-endpoint-001",
        "image_data_url": _png_data_url(),
        "annotations": [
            {"index": 0, "class_name": "atl", "bbox": [0, 0, 5, 5]},
        ],
    }


@pytest.fixture()
def app_and_services(tmp_path):
    services = SaveServices(tmp_path)
    app = create_app(settings=Settings(backend_root=tmp_path, testing=True), services=services)
    return app, services


@pytest.fixture()
def client(app_and_services):
    app, _services = app_and_services
    with app.test_client() as c:
        yield c


def test_save_annotation_happy_path(client):
    resp = client.post(
        "/save-annotation",
        data=json.dumps(_valid_payload()),
        content_type="application/json",
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok"
    assert body["analysis_id"] == "test-endpoint-001"


def test_save_annotation_permission_denied(client, app_and_services):
    _app, services = app_and_services
    services.raise_exc = AnnotationPermissionError("denied")

    resp = client.post(
        "/save-annotation",
        data=json.dumps(_valid_payload()),
        content_type="application/json",
    )
    assert resp.status_code == 409
    body = resp.get_json()
    assert body["error_code"] == "PERMISSION_DENIED"
    assert "Droits" in body["message"]


def test_save_annotation_disk_full(client, app_and_services):
    _app, services = app_and_services
    services.raise_exc = AnnotationDiskFullError("no space")

    resp = client.post(
        "/save-annotation",
        data=json.dumps(_valid_payload()),
        content_type="application/json",
    )
    assert resp.status_code == 507
    body = resp.get_json()
    assert body["error_code"] == "DISK_FULL"


def test_save_annotation_unknown_error(client, app_and_services):
    _app, services = app_and_services
    services.raise_exc = RuntimeError("boom")

    resp = client.post(
        "/save-annotation",
        data=json.dumps(_valid_payload()),
        content_type="application/json",
    )
    assert resp.status_code == 500
    body = resp.get_json()
    assert body["error_code"] == "INTERNAL_ERROR"
    assert body["trace_id"]
    assert "id=" in body["message"]
