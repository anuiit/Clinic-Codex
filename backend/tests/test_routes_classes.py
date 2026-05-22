from __future__ import annotations

from backend.app.config import Settings
from backend.app.factory import create_app


class StubServices:
    def load_classes(self):
        return {"num_classes": 2, "class_names": ["atl", "tochtli"]}


def test_classes_returns_configured_shape():
    app = create_app(settings=Settings(testing=True), services=StubServices())
    with app.test_client() as client:
        resp = client.get("/classes")
    assert resp.status_code == 200
    assert resp.get_json() == {"num_classes": 2, "class_names": ["atl", "tochtli"]}
