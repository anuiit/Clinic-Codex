from __future__ import annotations

from backend.app.config import Settings
from backend.app.factory import create_app


class StubServices:
    def load_classes(self):
        return {"num_classes": 2, "class_names": ["atl", "tochtli"]}


def test_create_app_uses_injected_settings_and_services(tmp_path):
    settings = Settings(
        backend_root=tmp_path,
        testing=True,
        host="127.0.0.1",
        port=5001,
        cors_origins=("http://example.test",),
        enable_legacy_endpoints=True,
    )
    services = StubServices()
    app = create_app(settings=settings, services=services)

    assert app.config["TESTING"] is True
    assert app.config["CLINIC_SETTINGS"] == settings
    assert app.extensions["clinic_services"] is services

    routes = {rule.rule for rule in app.url_map.iter_rules()}
    assert {"/health", "/ready", "/classes", "/save-annotation", "/similar", "/trust", "/segment"}.issubset(routes)
    assert {"/classify", "/classify-batch", "/sample-image", "/similar-samples"}.issubset(routes)


def test_wsgi_exports_default_app(monkeypatch):
    monkeypatch.setenv("AUTH_REQUIRED", "false")
    monkeypatch.delitem(__import__("sys").modules, "backend.wsgi", raising=False)
    from backend.wsgi import app

    assert app is not None
    assert "/health" in {rule.rule for rule in app.url_map.iter_rules()}


def test_health_and_cors_are_cheap():
    app = create_app(settings=Settings(testing=True, cors_origins=("http://localhost:7118",)), services=StubServices())
    with app.test_client() as client:
        resp = client.get("/health", headers={"Origin": "http://localhost:7118"})

    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}
    assert resp.headers["Access-Control-Allow-Origin"] == "http://localhost:7118"


def test_cors_does_not_allow_unknown_or_missing_origins():
    app = create_app(settings=Settings(testing=True, cors_origins=("http://localhost:7118",)), services=StubServices())
    with app.test_client() as client:
        unknown = client.get("/health", headers={"Origin": "http://evil.example"})
        no_origin = client.get("/health")

    assert unknown.status_code == 200
    assert "Access-Control-Allow-Origin" not in unknown.headers
    assert no_origin.status_code == 200
    assert "Access-Control-Allow-Origin" not in no_origin.headers


def test_legacy_sample_routes_can_be_disabled_without_hiding_active_routes(tmp_path):
    settings = Settings(backend_root=tmp_path, testing=True, enable_legacy_endpoints=False)
    app = create_app(settings=settings, services=StubServices())
    routes = {rule.rule for rule in app.url_map.iter_rules()}
    assert {"/health", "/ready", "/classes", "/save-annotation", "/similar", "/trust", "/segment"}.issubset(routes)
    assert {"/classify", "/classify-batch"}.issubset(routes)
    assert "/sample-image" not in routes
    assert "/similar-samples" not in routes
