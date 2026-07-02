# type: ignore
# pyright: reportMissingImports=false
"""Tests for local-only admin annotation review routes."""
from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

from backend.app.config import Settings
from backend.app.factory import create_app


def _png_data_url(size=(12, 12)):
    buf = io.BytesIO()
    Image.new("RGB", size, color=(230, 230, 230)).save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def _payload(analysis_id: str):
    return {
        "analysis_id": analysis_id,
        "image_data_url": _png_data_url(),
        "annotations": [
            {"index": 0, "class_name": "atl", "bbox": [0, 0, 4, 4]},
            {"index": 1, "class_name": "calli", "bbox": [4, 4, 4, 4]},
        ],
    }


@pytest.fixture()
def settings(tmp_path):
    return Settings(backend_root=tmp_path, testing=True)


def _client(settings):
    app = create_app(settings=settings)
    return app, app.test_client()


def test_save_annotation_submissions_appear_pending_in_admin_queue(settings):
    _app, client = _client(settings)
    save = client.post("/save-annotation", json=_payload("route-pending-1"))
    assert save.status_code == 200

    resp = client.get("/admin/annotations")

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["local_only"] is True
    assert "not production-secured" in body["warning"]
    assert body["counts"]["pending"] == 2
    assert body["counts"]["trainable"] == 0
    assert body["analyses"][0]["analysis_id"] == "route-pending-1"
    assert [item["review_status"] for item in body["analyses"][0]["elements"]] == [
        "pending",
        "pending",
    ]


@pytest.mark.parametrize(
    ("method", "path", "json_body"),
    [
        ("get", "/admin/annotations", None),
        ("get", "/admin/annotations/guarded-route-1/image", None),
        ("get", "/admin/annotations/guarded-route-1/0/crop", None),
        ("post", "/admin/annotations/guarded-route-1/0/review", {"status": "approved"}),
        ("post", "/admin/annotations/guarded-route-1/0/modify", {"class_name": "atl", "bbox": [0, 0, 4, 4]}),
    ],
)
def test_admin_annotation_routes_reject_non_loopback_requests(settings, method, path, json_body):
    _app, client = _client(settings)
    assert client.post("/save-annotation", json=_payload("guarded-route-1")).status_code == 200

    request = getattr(client, method)
    kwargs = {"environ_overrides": {"REMOTE_ADDR": "192.0.2.10"}, "headers": {"Host": "localhost"}}
    if json_body is not None:
        kwargs["json"] = json_body
    resp = request(path, **kwargs)

    assert resp.status_code == 403
    body = resp.get_json()
    assert body["error_code"] == "LOCAL_ONLY_FORBIDDEN"
    assert "non_loopback_remote_addr" in body["reasons"]


def test_admin_review_mutation_persists_across_app_reload(settings):
    _app, client = _client(settings)
    assert client.post("/save-annotation", json=_payload("route-mixed-1")).status_code == 200

    approved = client.post("/admin/annotations/route-mixed-1/0/review", json={"status": "approved"})
    rejected = client.post("/admin/annotations/route-mixed-1/1/review", json={"status": "rejected"})

    assert approved.status_code == 200
    assert approved.get_json()["element"]["review_status"] == "approved"
    assert rejected.status_code == 200
    assert rejected.get_json()["element"]["review_status"] == "rejected"

    _reloaded_app, reloaded = _client(settings)
    queue = reloaded.get("/admin/annotations").get_json()
    elements = queue["analyses"][0]["elements"]

    assert {item["index"]: item["review_status"] for item in elements} == {
        0: "approved",
        1: "rejected",
    }
    assert queue["counts"]["trainable"] == 1


def test_admin_review_media_routes_serve_original_and_crop(settings):
    _app, client = _client(settings)
    assert client.post("/save-annotation", json=_payload("route-media-1")).status_code == 200

    image = client.get("/admin/annotations/route-media-1/image")
    crop = client.get("/admin/annotations/route-media-1/0/crop")
    missing_crop = client.get("/admin/annotations/route-media-1/99/crop")

    assert image.status_code == 200
    assert image.mimetype == "image/png"
    assert crop.status_code == 200
    assert crop.mimetype == "image/png"
    assert missing_crop.status_code == 404


def test_admin_review_route_returns_clear_errors(settings):
    _app, client = _client(settings)
    assert client.post("/save-annotation", json=_payload("route-errors-1")).status_code == 200

    invalid_json = client.post(
        "/admin/annotations/route-errors-1/0/review",
        data="not json",
        content_type="application/json",
    )
    assert invalid_json.status_code == 400
    assert invalid_json.get_json() == {"status": "error", "error": "invalid JSON"}

    missing_status = client.post("/admin/annotations/route-errors-1/0/review", json={})
    assert missing_status.status_code == 400
    assert missing_status.get_json() == {"status": "error", "error": "missing field: status"}

    bad_status = client.post("/admin/annotations/route-errors-1/0/review", json={"status": "validated"})
    assert bad_status.status_code == 400
    assert "status must be one of" in bad_status.get_json()["error"]

    missing_element = client.post("/admin/annotations/route-errors-1/99/review", json={"status": "approved"})
    assert missing_element.status_code == 404
    assert "route-errors-1:99" in missing_element.get_json()["error"]


def test_no_admin_retrain_endpoint_is_registered(settings):
    app, client = _client(settings)

    assert client.post("/admin/annotations/retrain").status_code == 404
    assert all("retrain" not in rule.rule for rule in app.url_map.iter_rules())


def test_admin_modify_route_updates_element_and_defaults_pending(settings):
    _app, client = _client(settings)
    assert client.post("/save-annotation", json=_payload("route-modify-1")).status_code == 200
    assert client.post("/admin/annotations/route-modify-1/0/review", json={"status": "approved"}).status_code == 200

    resp = client.post(
        "/admin/annotations/route-modify-1/0/modify",
        json={"class_name": "new-atl", "bbox": [1, 2, 5, 6]},
    )

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["element"]["class_name"] == "new-atl"
    assert body["element"]["bbox"] == [1, 2, 5, 6]
    assert body["element"]["review_status"] == "pending"
    assert body["element"]["trainable"] is False
    assert body["counts"]["pending"] == 2

    crop = client.get("/admin/annotations/route-modify-1/0/crop")
    assert crop.status_code == 200
    with Image.open(io.BytesIO(crop.data)) as image:
        assert image.size == (5, 6)


def test_admin_modify_route_can_save_and_approve(settings):
    _app, client = _client(settings)
    assert client.post("/save-annotation", json=_payload("route-modify-approve-1")).status_code == 200

    resp = client.post(
        "/admin/annotations/route-modify-approve-1/0/modify",
        json={"class_name": "approved-atl", "bbox": [0, 0, 4, 4], "approve_after_save": True},
    )

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["element"]["review_status"] == "approved"
    assert body["element"]["trainable"] is True
    assert body["counts"]["approved"] == 1
    assert body["counts"]["trainable"] == 1


def test_admin_modify_route_returns_clear_errors(settings):
    _app, client = _client(settings)
    assert client.post("/save-annotation", json=_payload("route-modify-errors-1")).status_code == 200

    invalid_json = client.post(
        "/admin/annotations/route-modify-errors-1/0/modify",
        data="not json",
        content_type="application/json",
    )
    assert invalid_json.status_code == 400
    assert invalid_json.get_json() == {"status": "error", "error": "invalid JSON"}

    missing_class = client.post("/admin/annotations/route-modify-errors-1/0/modify", json={"bbox": [0, 0, 4, 4]})
    assert missing_class.status_code == 400
    assert missing_class.get_json()["error"] == "missing field: class_name"

    invalid_bbox = client.post(
        "/admin/annotations/route-modify-errors-1/0/modify",
        json={"class_name": "atl", "bbox": [0, 0, -4, 4]},
    )
    assert invalid_bbox.status_code == 400
    assert "width" in invalid_bbox.get_json()["error"]

    missing_element = client.post(
        "/admin/annotations/route-modify-errors-1/99/modify",
        json={"class_name": "atl", "bbox": [0, 0, 4, 4]},
    )
    assert missing_element.status_code == 404
    assert "route-modify-errors-1:99" in missing_element.get_json()["error"]
