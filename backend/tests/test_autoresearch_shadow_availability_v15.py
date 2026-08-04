from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/autoresearch_shadow_availability_v15.py"
spec = importlib.util.spec_from_file_location(
    "autoresearch_shadow_availability_v15", SCRIPT
)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def _source_scan(
    *, persisted_fields: list[str], local_only: bool = False, predictions: bool = False
) -> dict[str, object]:
    evidence = {category: [] for category in module.EVIDENCE_PATTERNS}
    evidence["persistence"] = [{"token": "save_annotation"}]
    evidence["consent"] = [{"token": "consent"}]
    evidence["retention"] = [{"token": "retention"}]
    if local_only:
        evidence["local_only"] = [{"token": "local/dev-only"}]
    if predictions:
        evidence["prediction_exposure"] = [{"token": "predicted_class"}]
    return {
        "endpoints": [{"route": "/save-annotation"}],
        "persisted_annotation_metadata_fields": persisted_fields,
        "evidence": evidence,
    }


def _dynamic(post_freeze: int) -> dict[str, object]:
    return {"annotations": {"post_freeze_top_level_directories": post_freeze}}


def test_v15_contract_is_hash_pinned_private_and_model_free() -> None:
    contract = module.validate_contract()
    assert contract["spec_sha256"] == module.EXPECTED_SPEC_SHA256
    assert contract["evaluator_sha256"] == module.EXPECTED_EVALUATOR_SHA256
    assert contract["surface_manifest_sha256"] == module.EXPECTED_MANIFEST_SHA256
    assert len(contract["manifest"]["tracked_sources"]) == 26
    assert contract["spec"]["model_inference_allowed"] is False
    assert contract["spec"]["external_fetch_allowed"] is False


def test_route_extraction_reports_literal_path_without_line_contents(tmp_path: Path) -> None:
    evidence = module.token_evidence(
        "backend/app/routes/sample.py",
        '@bp.post("/save")\nsecret = "never emitted"\nrequest.files["image"]\n',
        "capture_input",
        ("request.files",),
    )
    assert evidence == [
        {
            "source": "backend/app/routes/sample.py",
            "line": 3,
            "token": "request.files",
            "category": "capture_input",
        }
    ]
    assert "never emitted" not in json.dumps(evidence)
    match = module.ROUTE_PATTERN.search('@bp.post("/save")')
    assert match and match.group("route") == "/save"


def test_literal_persisted_fields_are_extracted_from_named_assignment() -> None:
    source = """
def save():
    metadata = {"analysis_id": "secret", "uploaded_at": "now"}
    response = {"ignored": True}
"""
    assert module.literal_dict_keys_from_assignment(source, "metadata") == {
        "analysis_id",
        "uploaded_at",
    }


def test_tree_snapshot_aggregates_without_emitting_names_or_bytes(tmp_path: Path) -> None:
    analysis = tmp_path / "private-analysis-id"
    analysis.mkdir()
    (analysis / "metadata.json").write_text("private label", encoding="utf-8")
    (analysis / "image.png").write_bytes(b"private pixels")
    result = module.aggregate_tree_metadata(tmp_path, freeze_timestamp=0.0)
    rendered = json.dumps(result)
    assert result["top_level_directories"] == 1
    assert result["post_freeze_top_level_directories"] == 1
    assert result["file_count"] == 2
    assert result["suffix_counts"] == {".json": 1, ".png": 1}
    assert "private-analysis-id" not in rendered
    assert "private label" not in rendered
    assert "private pixels" not in rendered


def test_sqlite_snapshot_emits_only_schema_counts_and_timestamp_range(
    tmp_path: Path,
) -> None:
    path = tmp_path / "auth.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE users (id TEXT, email TEXT, created_at TEXT)"
    )
    connection.execute(
        "INSERT INTO users VALUES (?, ?, ?)",
        ("private-user-id", "secret@example.test", "2026-08-03T00:00:00Z"),
    )
    connection.commit()
    connection.close()
    result = module.aggregate_sqlite_metadata(path)
    rendered = json.dumps(result)
    assert result["tables"][0]["row_count"] == 1
    assert result["tables"][0]["columns"] == ["id", "email", "created_at"]
    assert "private-user-id" not in rendered
    assert "secret@example.test" not in rendered


def test_flow_with_document_identity_and_controls_supports_hypothesis() -> None:
    result = module.classify_flow(
        _source_scan(persisted_fields=["document_id", "analysis_id"]),
        _dynamic(3),
    )
    assert result["classification"] == "flow_exists_with_document_identity"
    assert result["instrument_usable"] is True
    assert result["hypothesis_supported"] is True


def test_local_only_or_fallback_identity_cannot_support_instrument() -> None:
    local = module.classify_flow(
        _source_scan(persisted_fields=["document_id"], local_only=True),
        _dynamic(3),
    )
    fallback = module.classify_flow(
        _source_scan(persisted_fields=["analysis_id"], predictions=True),
        _dynamic(3),
    )
    assert local["classification"] == "no_flow"
    assert local["instrument_usable"] is False
    assert fallback["classification"] == "flow_exists_without_document_identity"
    assert fallback["instrument_usable"] is False
    assert fallback["double_blind_annotation_feasible"] is False


def test_rate_is_not_extrapolated_before_seven_days() -> None:
    now = datetime.now(timezone.utc).timestamp()
    result = module.rate_evidence(4, now - 24 * 60 * 60)
    assert result["rate_evidence_sufficient"] is False
    assert result["observed_records_per_week"] is None
    assert result["descriptive_scenario_documents"] == 48


def test_audit_serialization_is_deterministic(tmp_path: Path) -> None:
    audit = {
        "schema_version": "test",
        "privacy_counters": {"dynamic_file_bytes_read": 0},
    }
    first = module.write_audit(tmp_path / "first", audit)
    replay = module.write_audit(tmp_path / "replay", audit)
    assert first["audit_sha256"] == replay["audit_sha256"]
