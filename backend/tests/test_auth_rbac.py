from __future__ import annotations

import base64
import io
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor

import pytest
from PIL import Image

from backend.app.config import Settings
from backend.app.factory import create_app
from backend.services.annotation_review import AnnotationReviewStore, AnnotationReviewValidationError
from backend.services.annotation_storage import save_annotation


class Services:
    def __init__(self):
        self.saved = []

    def decode_annotation_image(self, value):
        from backend.services.annotation_storage import decode_image_data_url
        return decode_image_data_url(value)

    def save_annotation(self, analysis_id, image, annotations, *, author_id=None):
        self.saved.append((analysis_id, author_id))
        return {"status": "ok", "analysis_id": analysis_id}

    def list_annotation_reviews(self):
        return {"status": "ok", "analyses": []}


def _settings(tmp_path):
    return Settings(
        backend_root=tmp_path,
        testing=True,
        auth_required=True,
        auth_secret_key="test-secret",
        auth_bootstrap_email="admin@example.test",
        auth_bootstrap_password="admin-password",
        cors_origins=("http://localhost:7118",),
    )
def test_auth_required_is_secure_by_default_outside_testing(monkeypatch):
    monkeypatch.delenv("AUTH_REQUIRED", raising=False)
    monkeypatch.delenv("FLASK_TESTING", raising=False)
    assert Settings.from_env().authentication_enabled is True
    monkeypatch.setenv("AUTH_REQUIRED", "false")
    assert Settings.from_env().authentication_enabled is False



def _login(client, email, password):
    response = client.post("/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200
    return response.get_json()["csrf_token"]


def test_first_admin_bootstrap_is_local_validated_and_closes_after_creation(tmp_path):
    app = create_app(
        replace(
            _settings(tmp_path),
            auth_bootstrap_email="",
            auth_bootstrap_password="",
            auth_cookie_secure=False,
        ),
        services=Services(),
    )

    with app.test_client() as client:
        status = client.get("/auth/bootstrap/status")
        assert status.status_code == 200
        assert status.get_json() == {
            "status": "ok",
            "auth_enabled": True,
            "bootstrap_available": True,
        }

        remote_status = client.get(
            "/auth/bootstrap/status",
            environ_overrides={"REMOTE_ADDR": "203.0.113.9"},
        )
        assert remote_status.status_code == 403
        assert remote_status.get_json()["error_code"] == "LOCAL_ONLY_FORBIDDEN"

        invalid_email = client.post(
            "/auth/bootstrap",
            json={"email": "not-an-email", "password": "long-enough-password"},
        )
        assert invalid_email.status_code == 400
        assert invalid_email.get_json()["error_code"] == "INVALID_EMAIL"

        weak_password = client.post(
            "/auth/bootstrap",
            json={"email": "admin@example.test", "password": "too-short"},
        )
        assert weak_password.status_code == 400
        assert weak_password.get_json()["error_code"] == "INVALID_PASSWORD"

        created = client.post(
            "/auth/bootstrap",
            json={"email": "Admin@Example.test", "password": "a-secure-local-password"},
        )
        assert created.status_code == 201
        payload = created.get_json()
        assert payload["user"]["email"] == "admin@example.test"
        assert payload["user"]["role"] == "org_admin"
        assert payload["user"]["roles"] == ["org_admin"]
        assert payload["csrf_token"]
        assert "HttpOnly" in created.headers["Set-Cookie"]

        assert client.get("/auth/bootstrap/status").get_json()["bootstrap_available"] is False
        second = client.post(
            "/auth/bootstrap",
            json={"email": "second@example.test", "password": "another-secure-password"},
        )
        assert second.status_code == 409
        assert second.get_json()["error_code"] == "BOOTSTRAP_UNAVAILABLE"

        assert client.post(
            "/auth/logout",
            headers={"X-CSRF-Token": payload["csrf_token"]},
        ).status_code == 200
        assert client.post(
            "/auth/login",
            json={"email": "admin@example.test", "password": "a-secure-local-password"},
        ).status_code == 200


def test_first_admin_creation_is_atomic_under_concurrency(tmp_path):
    app = create_app(
        replace(_settings(tmp_path), auth_bootstrap_email="", auth_bootstrap_password=""),
        services=Services(),
    )
    store = app.extensions["clinic_auth_store"]

    def create(index: int):
        try:
            return store.create_first_admin(
                f"admin-{index}@example.test",
                "a-secure-local-password",
            )
        except ValueError as error:
            return str(error)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(create, range(8)))

    created = [result for result in results if isinstance(result, dict)]
    unavailable = [result for result in results if result == "bootstrap unavailable"]
    assert len(created) == 1
    assert len(unavailable) == 7
    assert store.bootstrap_available() is False


def test_bootstrap_status_reports_disabled_auth_without_a_store(tmp_path):
    app = create_app(replace(_settings(tmp_path), auth_required=False), services=Services())

    with app.test_client() as client:
        response = client.get("/auth/bootstrap/status")

    assert response.status_code == 200
    assert response.get_json() == {
        "status": "ok",
        "auth_enabled": False,
        "bootstrap_available": False,
    }


def test_auth_session_is_server_side_revocable_and_cors_supports_credentials(tmp_path):
    app = create_app(_settings(tmp_path), services=Services())
    with app.test_client() as client:
        login_csrf = _login(client, "admin@example.test", "admin-password")
        me = client.get("/auth/me").get_json()
        assert me["user"]["role"] == "org_admin"
        assert me["user"]["roles"] == ["org_admin"]
        assert "annotation.queue.read" in me["user"]["permissions"]
        assert "training.read" in me["user"]["permissions"]
        assert me["csrf_token"] == login_csrf
        csrf = me["csrf_token"]
        denied = client.post("/auth/logout")
        assert denied.status_code == 403
        assert client.post("/auth/logout", headers={"X-CSRF-Token": csrf}).status_code == 200
        assert client.get("/auth/me").status_code == 401
        preflight = client.options("/save-annotation", headers={"Origin": "http://localhost:7118"})
        assert preflight.headers["Access-Control-Allow-Credentials"] == "true"
        assert "X-CSRF-Token" in preflight.headers["Access-Control-Allow-Headers"]
        assert preflight.headers["Vary"] == "Origin"


def test_session_cookie_attributes_follow_settings(tmp_path):
    app = create_app(_settings(tmp_path), services=Services())
    with app.test_client() as client:
        login = client.post("/auth/login", json={"email": "admin@example.test", "password": "admin-password"})
        cookie = login.headers["Set-Cookie"]
        assert "HttpOnly" in cookie
        assert "SameSite=Lax" in cookie
        assert "Secure" in cookie
        csrf = login.get_json()["csrf_token"]
        cleared = client.post("/auth/logout", headers={"X-CSRF-Token": csrf}).headers["Set-Cookie"]
        assert "SameSite=Lax" in cleared
        assert "Secure" in cleared

    insecure_app = create_app(replace(_settings(tmp_path), auth_cookie_secure=False), services=Services())
    with insecure_app.test_client() as client:
        login = client.post("/auth/login", json={"email": "admin@example.test", "password": "admin-password"})

    assert "Secure" not in login.headers["Set-Cookie"]


def test_me_reports_auth_disabled_without_requiring_a_session(tmp_path):
    app = create_app(replace(_settings(tmp_path), auth_required=False), services=Services())

    with app.test_client() as client:
        response = client.get("/auth/me")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok", "auth_enabled": False, "user": None}



def test_annotation_submission_requires_permission_csrf_and_records_author(tmp_path):
    services = Services()
    app = create_app(_settings(tmp_path), services=services)
    store = app.extensions["clinic_auth_store"]
    store.create_user("writer@example.test", "writer-password", "contributor")
    data = io.BytesIO()
    Image.new("RGB", (4, 4)).save(data, format="PNG")
    payload = {"analysis_id": "entry-1", "image_data_url": "data:image/png;base64," + base64.b64encode(data.getvalue()).decode(), "annotations": [{"index": 0, "class_name": "atl", "bbox": [0, 0, 2, 2]}]}
    with app.test_client() as client:
        assert client.post("/save-annotation", json=payload).status_code == 401
        csrf = _login(client, "writer@example.test", "writer-password")
        assert client.post("/save-annotation", json=payload).status_code == 403
        assert client.post("/save-annotation", json=payload, headers={"X-CSRF-Token": csrf}).status_code == 200
    assert services.saved == [("entry-1", store.authenticate("writer@example.test", "writer-password")["id"])]


def test_review_blocks_known_self_review_but_allows_legacy_unknown_author(tmp_path):
    image = Image.new("RGB", (8, 8))
    save_annotation("known", image, [{"index": 0, "class_name": "atl", "bbox": [0, 0, 4, 4]}], tmp_path, tmp_path / "unused", author_id="user-1")
    save_annotation("legacy", image, [{"index": 0, "class_name": "atl", "bbox": [0, 0, 4, 4]}], tmp_path, tmp_path / "unused")
    reviews = AnnotationReviewStore(tmp_path)
    with pytest.raises(AnnotationReviewValidationError, match="self-review"):
        reviews.set_status("known", 0, "approved", reviewer_id="user-1")
    reviews.set_status("legacy", 0, "approved", reviewer_id="user-1")
    decision = reviews._load_manifest()["decisions"]["legacy:0"]
    assert decision["reviewed_by"] == "user-1"

def test_login_rate_limit_and_role_denial(tmp_path):
    app = create_app(_settings(tmp_path), services=Services())
    store = app.extensions["clinic_auth_store"]
    store.create_user("writer@example.test", "writer-password", "contributor")
    with app.test_client() as client:
        responses = [
            client.post("/auth/login", json={"email": "unknown@example.test", "password": "wrong"})
            for _ in range(5)
        ]
        assert [response.status_code for response in responses[:4]] == [401, 401, 401, 401]
        assert responses[4].status_code == 429
        csrf = _login(client, "writer@example.test", "writer-password")
        denied = client.get("/admin/annotations", headers={"Host": "localhost"})
        assert denied.status_code == 403
        assert denied.get_json()["error_code"] == "PERMISSION_DENIED"
        assert client.post("/auth/logout", headers={"X-CSRF-Token": csrf}).status_code == 200
        assert client.get("/auth/me").status_code == 401


def test_review_route_rejects_self_review_over_http(tmp_path):
    app = create_app(_settings(tmp_path))
    image = io.BytesIO()
    Image.new("RGB", (8, 8)).save(image, format="PNG")
    payload = {
        "analysis_id": "self-review-http",
        "image_data_url": "data:image/png;base64," + base64.b64encode(image.getvalue()).decode(),
        "annotations": [{"index": 0, "class_name": "atl", "bbox": [0, 0, 4, 4]}],
    }
    with app.test_client() as client:
        csrf = _login(client, "admin@example.test", "admin-password")
        assert client.post("/save-annotation", json=payload, headers={"X-CSRF-Token": csrf}).status_code == 200
        response = client.post(
            "/admin/annotations/self-review-http/0/review",
            json={"status": "approved"},
            headers={"Host": "localhost", "X-CSRF-Token": csrf},
        )
    assert response.status_code == 403
    assert "self-review" in response.get_json()["error"]


def test_review_mutations_keep_independent_concurrent_decisions(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    image = Image.new("RGB", (8, 8))
    save_annotation(
        "concurrent", image,
        [
            {"index": 0, "class_name": "atl", "bbox": [0, 0, 4, 4]},
            {"index": 1, "class_name": "atl", "bbox": [4, 4, 4, 4]},
        ],
        tmp_path, tmp_path / "unused",
    )
    reviews = AnnotationReviewStore(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda index: reviews.set_status("concurrent", index, "approved", reviewer_id="reviewer"), [0, 1]))
    decisions = reviews._load_manifest()["decisions"]
    assert decisions["concurrent:0"]["status"] == "approved"
    assert decisions["concurrent:1"]["status"] == "approved"


@pytest.mark.parametrize("action", ["review", "modify"])
@pytest.mark.parametrize("case", ["local_admin", "disabled", "later_admin", "reviewer", "remote", "network_host", "no_csrf"])
def test_local_initial_admin_self_review_is_scoped_and_audited(tmp_path, case, action):
    settings = replace(
        _settings(tmp_path), allow_local_admin_self_review=case != "disabled",
        host="0.0.0.0" if case == "network_host" else "127.0.0.1",
    )
    app = create_app(settings)
    store = app.extensions["clinic_auth_store"]
    email, password = "admin@example.test", "admin-password"
    if case in {"later_admin", "reviewer"}:
        email = "second@example.test"
        store.create_user(email, password, "reviewer" if case == "reviewer" else "org_admin")
    actor = store.authenticate(email, password)
    save_annotation(
        "own", Image.new("RGB", (8, 8)),
        [{"index": 0, "class_name": "atl", "bbox": [0, 0, 4, 4]}],
        settings.annotations_dir, settings.elements_dir, author_id=actor["id"],
    )
    body = {"status": "approved"}
    if action == "modify":
        body.update(class_name="calli", bbox=[1, 1, 4, 4])
    # A client-supplied permission must never enable the exception.
    body["allow_self_review"] = True
    with app.test_client() as client:
        csrf = _login(client, email, password)
        response = client.post(
            f"/admin/annotations/own/0/{action}", json=body,
            headers={} if case == "no_csrf" else {"X-CSRF-Token": csrf},
            environ_overrides={"REMOTE_ADDR": "203.0.113.9"} if case == "remote" else {},
        )
    assert response.status_code == (200 if case == "local_admin" else 403)
    decisions = app.extensions["clinic_services"].annotation_review_store()._load_manifest()["decisions"]
    if case == "local_admin":
        assert decisions["own:0"]["reviewed_by"] == actor["id"]
        assert decisions["own:0"]["status"] == "approved"
    else:
        assert "own:0" not in decisions
