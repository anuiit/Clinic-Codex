#!/usr/bin/env python3
"""Audit the Visual Lexicon manuscript ceiling without touching model efficacy.

This v12 gate-zero runner fetches one official metadata page, verifies the
predeclared Tlachia exclusions, and computes the largest possible number of
manuscript-level components.  A failed ceiling short-circuits every record
detail request, image download, label read, and model prediction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = (
    ROOT
    / ".omc/autoresearch/elements-baseline-replacement/runs/"
    "20260803-independent-instrument-v12"
)
SPEC_PATH = RUN_DIR / "specs/iteration-0001.json"
EVALUATOR_PATH = RUN_DIR / "evaluator.json"
EXCLUSION_RULES_PATH = RUN_DIR / "manuscript-exclusion-rules.json"
DEFAULT_CACHE_DIR = RUN_DIR / "cache"
DEFAULT_OUTPUT_DIR = RUN_DIR / "iteration-0001"
DEFAULT_REPLAY_DIR = RUN_DIR / "iteration-0001-replay"
DEFAULT_EVALUATION_PATH = RUN_DIR / "evaluations/iteration-0001.json"

EXPECTED_SPEC_SHA256 = "6a630f12d80ea9b3542ee8abdb216ef30f539df0e6c55cd4fddde265871a2835"
EXPECTED_EVALUATOR_SHA256 = (
    "55bcb0833210ab9871985b072ad3da19f9d8996b6957d1e1dc49bb7f62c85c7e"
)
EXPECTED_EXCLUSION_RULES_SHA256 = (
    "add2a06eee290032c5479cbe9cf4522d6a6ac995a5a90d590047776cc965f756"
)
EXPECTED_RUNTIME_PROJECTION_SHA256 = (
    "0ad6ce348eaf44c8844415a769b048b8f155017befa93f16254b1263b9241210"
)
EXPECTED_RUNTIME_PROTOTYPES_SHA256 = (
    "ffc52f03696f03780a01302708b53e319dc24069187a6fd1a2927ea537b1cc54"
)
EXPECTED_RUNTIME_CONFIG_SHA256 = (
    "f2150a97ec7eee50a42ac1a108ae4bc092d46a6b8e4a2d91de59349560fbf18b"
)
USER_AGENT = (
    "clinic-codex-autoresearch-v12/1.0 "
    "(metadata-only reproducibility audit; no image acquisition)"
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class ManuscriptSelectParser(HTMLParser):
    """Extract one Drupal manuscript taxonomy select using only stdlib HTML."""

    def __init__(self, select_id: str) -> None:
        super().__init__(convert_charrefs=True)
        self.select_id = select_id
        self.select_seen = 0
        self.inside_select = False
        self.current_option: dict[str, Any] | None = None
        self.options: list[dict[str, str]] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attributes = dict(attrs)
        if tag == "select" and attributes.get("id") == self.select_id:
            self.select_seen += 1
            if self.select_seen > 1:
                raise ValueError(f"duplicate manuscript select: {self.select_id}")
            self.inside_select = True
            return
        if tag == "option" and self.inside_select:
            if self.current_option is not None:
                raise ValueError("nested option in manuscript select")
            self.current_option = {
                "value": attributes.get("value", ""),
                "text": [],
            }

    def handle_data(self, data: str) -> None:
        if self.current_option is not None:
            self.current_option["text"].append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "option" and self.current_option is not None:
            text = " ".join("".join(self.current_option["text"]).split())
            self.options.append(
                {"value": str(self.current_option["value"]), "display_name": text}
            )
            self.current_option = None
            return
        if tag == "select" and self.inside_select:
            if self.current_option is not None:
                raise ValueError("unterminated option in manuscript select")
            self.inside_select = False


def parse_manuscript_inventory(html: bytes, *, select_id: str) -> list[dict[str, Any]]:
    parser = ManuscriptSelectParser(select_id)
    parser.feed(html.decode("utf-8"))
    parser.close()
    if parser.select_seen != 1:
        raise ValueError(
            f"expected one manuscript select {select_id!r}, got {parser.select_seen}"
        )
    any_options = [option for option in parser.options if option["value"] == "All"]
    if len(any_options) != 1:
        raise ValueError("manuscript select must contain exactly one All option")

    inventory: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for option in parser.options:
        if option["value"] == "All":
            continue
        try:
            term_id = int(option["value"])
        except ValueError as error:
            raise ValueError(
                f"non-integer manuscript term id: {option['value']!r}"
            ) from error
        if term_id in seen_ids:
            raise ValueError(f"duplicate manuscript term id: {term_id}")
        if not option["display_name"]:
            raise ValueError(f"empty manuscript display name for term {term_id}")
        seen_ids.add(term_id)
        inventory.append(
            {"term_id": term_id, "display_name": option["display_name"]}
        )
    return sorted(inventory, key=lambda row: row["term_id"])


def validate_contract(
    spec_path: Path = SPEC_PATH,
    evaluator_path: Path = EVALUATOR_PATH,
    rules_path: Path = EXCLUSION_RULES_PATH,
) -> dict[str, Any]:
    hashes = {
        "spec_sha256": sha256_file(spec_path),
        "evaluator_sha256": sha256_file(evaluator_path),
        "exclusion_rules_sha256": sha256_file(rules_path),
    }
    expected = {
        "spec_sha256": EXPECTED_SPEC_SHA256,
        "evaluator_sha256": EXPECTED_EVALUATOR_SHA256,
        "exclusion_rules_sha256": EXPECTED_EXCLUSION_RULES_SHA256,
    }
    for name, expected_hash in expected.items():
        if hashes[name] != expected_hash:
            raise ValueError(
                f"v12 contract hash mismatch for {name}: {hashes[name]}"
            )

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    evaluator = json.loads(evaluator_path.read_text(encoding="utf-8"))
    rules = json.loads(rules_path.read_text(encoding="utf-8"))
    gates = spec.get("pre_registered_gates", {})
    if gates.get("minimum_admissible_manuscript_components") != 42:
        raise ValueError("v12 manuscript threshold contract mismatch")
    if gates.get("expected_predeclared_exclusions") != 9:
        raise ValueError("v12 exclusion-count contract mismatch")
    if spec.get("model_inference_allowed") is not False:
        raise ValueError("v12 gate zero must forbid model inference")
    if spec.get("final_test_read_allowed") is not False:
        raise ValueError("v12 gate zero must forbid final-test access")
    if evaluator.get("promotion_eligible") is not False:
        raise ValueError("v12 metadata audit must forbid promotion")
    if len(rules.get("excluded_visual_lexicon_terms", [])) != 9:
        raise ValueError("v12 rules do not contain exactly nine exclusions")
    return {
        "spec": spec,
        "evaluator": evaluator,
        "rules": rules,
        **hashes,
    }


def _hash_matches(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: {actual}")
    return actual


def collect_local_evidence(contract: dict[str, Any]) -> dict[str, Any]:
    inputs = contract["spec"]["inputs"]
    codex_path = Path(inputs["local_codex_catalog"])
    elements_path = Path(inputs["local_element_catalog"])
    glyph_elements_path = Path(inputs["local_glyph_element_catalog"])

    source_hashes = {
        "codex_csv_sha256": _hash_matches(
            codex_path,
            inputs["local_codex_catalog_sha256"],
            "local codex catalogue",
        ),
        "elements_csv_sha256": _hash_matches(
            elements_path,
            inputs["local_element_catalog_sha256"],
            "local element catalogue",
        ),
        "glyph_elements_csv_sha256": _hash_matches(
            glyph_elements_path,
            inputs["local_glyph_element_catalog_sha256"],
            "local glyph-element catalogue",
        ),
    }
    with codex_path.open("r", encoding="utf-8-sig", newline="") as handle:
        codex_rows = list(csv.DictReader(handle))
    codex_ids = sorted(int(row["id"]) for row in codex_rows)
    expected_codex_rows = contract["spec"]["pre_registered_gates"][
        "local_codex_rows_equals"
    ]
    if len(codex_rows) != expected_codex_rows:
        raise ValueError("unexpected local codex row count")
    if codex_ids != list(range(1, 51)):
        raise ValueError("local codex ids must be exactly 1 through 50")

    archive_pattern = Path(inputs["local_archive_glob"])
    archive_names = sorted(
        path.name for path in archive_pattern.parent.glob(archive_pattern.name)
    )
    archive_inventory = "".join(f"{name}\n" for name in archive_names).encode()
    archive_inventory_sha256 = sha256_bytes(archive_inventory)
    if len(archive_names) != inputs["local_archive_count"]:
        raise ValueError(f"unexpected local archive count: {len(archive_names)}")
    if archive_inventory_sha256 != inputs["local_archive_name_inventory_sha256"]:
        raise ValueError(
            "local archive inventory hash mismatch: " + archive_inventory_sha256
        )

    linked_paths = {
        "v8_provenance_audit_sha256": (
            ROOT
            / ".omc/autoresearch/elements-baseline-replacement/runs/"
            "20260731-council-guided-v8/iteration-0003/"
            "collection-provenance-audit.json"
        ),
        "v8_prequarantine_manifest_sha256": (
            ROOT
            / ".omc/autoresearch/elements-baseline-replacement/runs/"
            "20260731-council-guided-v8/iteration-0002/provenance-manifest.jsonl"
        ),
        "v11_candidate_manifest_sha256": (
            ROOT
            / "backend/model_registry/versions/"
            "20260803T202104Z-vicreg-full-data-v11/manifest.json"
        ),
    }
    linked_hashes: dict[str, str] = {}
    for key, path in linked_paths.items():
        linked_hashes[key] = _hash_matches(path, inputs[key], key)

    runtime_paths = {
        "config_sha256": ROOT / "backend/codex_model/config.json",
        "projection_sha256": ROOT / "backend/codex_model/weights/projection.pt",
        "prototypes_sha256": ROOT / "backend/codex_model/weights/prototypes.pt",
    }
    runtime_expected = {
        "config_sha256": EXPECTED_RUNTIME_CONFIG_SHA256,
        "projection_sha256": EXPECTED_RUNTIME_PROJECTION_SHA256,
        "prototypes_sha256": EXPECTED_RUNTIME_PROTOTYPES_SHA256,
    }
    runtime_hashes = {
        key: _hash_matches(path, runtime_expected[key], f"runtime {key}")
        for key, path in runtime_paths.items()
    }
    runtime_config = json.loads(runtime_paths["config_sha256"].read_text())
    if runtime_config.get("num_classes") != 286:
        raise ValueError("runtime class-count contract mismatch")

    return {
        "local_codex_rows": len(codex_rows),
        "local_codex_ids_sha256": sha256_bytes(
            json.dumps(codex_ids, separators=(",", ":")).encode()
        ),
        "local_archive_count": len(archive_names),
        "local_archive_names": archive_names,
        "local_archive_name_inventory_sha256": archive_inventory_sha256,
        "source_hashes": source_hashes,
        "linked_hashes": linked_hashes,
        "runtime_hashes": runtime_hashes,
        "runtime_num_classes": runtime_config["num_classes"],
    }


def build_audit(
    html: bytes,
    *,
    contract: dict[str, Any],
    local_evidence: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = contract["spec"]
    inputs = spec["inputs"]
    gates = spec["pre_registered_gates"]
    inventory_rows = parse_manuscript_inventory(
        html, select_id=inputs["manuscript_select_id"]
    )
    if len(inventory_rows) != inputs["expected_manuscript_option_count"]:
        raise ValueError(
            "official manuscript count changed: "
            f"expected {inputs['expected_manuscript_option_count']}, "
            f"got {len(inventory_rows)}"
        )

    by_id = {row["term_id"]: row for row in inventory_rows}
    exclusions = contract["rules"]["excluded_visual_lexicon_terms"]
    excluded_ids: set[int] = set()
    verified_exclusions: list[dict[str, Any]] = []
    for exclusion in exclusions:
        term_id = int(exclusion["term_id"])
        if term_id in excluded_ids:
            raise ValueError(f"duplicate excluded manuscript term: {term_id}")
        excluded_ids.add(term_id)
        official = by_id.get(term_id)
        if official is None:
            raise ValueError(f"excluded manuscript term absent from catalogue: {term_id}")
        if official["display_name"] != exclusion["visual_manuscript"]:
            raise ValueError(
                f"manuscript display mismatch for {term_id}: "
                f"{official['display_name']!r}"
            )
        verified_exclusions.append(
            {
                "term_id": term_id,
                "display_name": official["display_name"],
                "overlap_kind": exclusion["overlap_kind"],
                "evidence": exclusion["evidence"],
            }
        )

    strict_ceiling = len(inventory_rows) - len(excluded_ids)
    minimum_components = gates["minimum_admissible_manuscript_components"]
    ceiling_gate_pass = strict_ceiling >= minimum_components
    short_circuit = not ceiling_gate_pass
    inventory = {
        "schema_version": (
            "autoresearch-independent-instrument-v12.manuscript-inventory"
        ),
        "source_url": inputs["visual_lexicon_advanced_search_url"],
        "source_html_sha256": sha256_bytes(html),
        "select_id": inputs["manuscript_select_id"],
        "manuscript_count": len(inventory_rows),
        "manuscripts": inventory_rows,
    }
    audit = {
        "schema_version": (
            "autoresearch-independent-instrument-v12.catalogue-ceiling-audit"
        ),
        "mission": "elements-baseline-replacement",
        "run_id": "20260803-independent-instrument-v12",
        "iteration": 1,
        "factor": (
            "Visual Lexicon manuscript-component feasibility under "
            "conservative exclusions"
        ),
        "contract_hashes": {
            "spec_sha256": contract["spec_sha256"],
            "evaluator_sha256": contract["evaluator_sha256"],
            "exclusion_rules_sha256": contract["exclusion_rules_sha256"],
        },
        "source": {
            "url": inputs["visual_lexicon_advanced_search_url"],
            "html_sha256": sha256_bytes(html),
            "manuscript_select_id": inputs["manuscript_select_id"],
        },
        "gate_zero": {
            "catalogue_manuscript_count": len(inventory_rows),
            "predeclared_excluded_manuscript_count": len(excluded_ids),
            "strict_admissible_manuscript_component_ceiling": strict_ceiling,
            "minimum_required_manuscript_components": minimum_components,
            "minimum_required_active_classes_if_ceiling_passes": gates[
                "minimum_active_runtime_classes"
            ],
            "component_ceiling_gate_pass": ceiling_gate_pass,
            "verified_exclusions": sorted(
                verified_exclusions, key=lambda row: row["term_id"]
            ),
            "explicitly_reviewed_non_matches": contract["rules"].get(
                "non_matches_explicitly_reviewed", []
            ),
        },
        "local_evidence": local_evidence,
        "short_circuit": {
            "triggered": short_circuit,
            "reason": (
                f"strict ceiling {strict_ceiling} is below preregistered "
                f"minimum {minimum_components}"
                if short_circuit
                else None
            ),
            "record_detail_pages_fetched": 0,
            "images_downloaded": 0,
            "labels_unlocked": 0,
            "model_predictions": 0,
            "one_shot_evaluation_consumed": False,
        },
        "scientific_boundary": {
            "metadata_only": True,
            "model_efficacy_measured": False,
            "holdout_or_final_test_read": False,
            "runtime_unchanged": True,
            "promotion_eligible": False,
        },
        "decision": (
            "metadata_gate_zero_pass"
            if ceiling_gate_pass
            else "instrument_not_acquired"
        ),
    }
    return inventory, audit


def write_audit_outputs(
    output_dir: Path,
    inventory: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, str]:
    inventory_path = output_dir / "manuscript-inventory.json"
    audit_path = output_dir / "catalogue-ceiling-audit.json"
    write_json(inventory_path, inventory)
    write_json(audit_path, audit)
    return {
        "inventory_path": str(inventory_path.resolve()),
        "inventory_sha256": sha256_file(inventory_path),
        "audit_path": str(audit_path.resolve()),
        "audit_sha256": sha256_file(audit_path),
    }


def evaluate(
    *,
    audit: dict[str, Any],
    first_outputs: dict[str, str],
    replay_outputs: dict[str, str],
    output_path: Path,
) -> dict[str, Any]:
    replay_exact = (
        first_outputs["inventory_sha256"] == replay_outputs["inventory_sha256"]
        and first_outputs["audit_sha256"] == replay_outputs["audit_sha256"]
    )
    gate_zero = audit["gate_zero"]
    short_circuit = audit["short_circuit"]
    integrity_gates = {
        "catalogue_manuscript_count_equals_50": (
            gate_zero["catalogue_manuscript_count"] == 50
        ),
        "excluded_term_count_equals_9": (
            gate_zero["predeclared_excluded_manuscript_count"] == 9
        ),
        "excluded_terms_present_and_names_exact": True,
        "local_codex_rows_equals_50": (
            audit["local_evidence"]["local_codex_rows"] == 50
        ),
        "local_archive_count_equals_49": (
            audit["local_evidence"]["local_archive_count"] == 49
        ),
        "local_and_linked_hashes_match": True,
        "runtime_hashes_unchanged": True,
        "deterministic_replay_same_audit_hash": replay_exact,
        "record_detail_pages_fetched_equals_0": (
            short_circuit["record_detail_pages_fetched"] == 0
        ),
        "images_downloaded_equals_0": short_circuit["images_downloaded"] == 0,
        "model_predictions_equals_0": short_circuit["model_predictions"] == 0,
        "holdout_or_final_test_unread": (
            audit["scientific_boundary"]["holdout_or_final_test_read"] is False
        ),
        "runtime_unchanged": audit["scientific_boundary"]["runtime_unchanged"],
    }
    execution_integrity_pass = all(integrity_gates.values())
    feasibility_pass = gate_zero["component_ceiling_gate_pass"]
    evaluation = {
        "schema_version": (
            "autoresearch-independent-instrument-v12.evaluation"
        ),
        "mission": "elements-baseline-replacement",
        "run_id": "20260803-independent-instrument-v12",
        "iteration": 1,
        "name": "visual-lexicon-manuscript-ceiling-gate-zero",
        "pass": execution_integrity_pass and feasibility_pass,
        "execution_integrity_pass": execution_integrity_pass,
        "instrument_acquired": False,
        "promotion_eligible": False,
        "strict_component_ceiling": gate_zero[
            "strict_admissible_manuscript_component_ceiling"
        ],
        "minimum_required_components": gate_zero[
            "minimum_required_manuscript_components"
        ],
        "short_circuit_triggered": short_circuit["triggered"],
        "final_test_read": False,
        "runtime_unchanged": True,
        "model_efficacy_measured": False,
        "integrity_gates": integrity_gates,
        "artifacts": {
            "first": first_outputs,
            "replay": replay_outputs,
        },
        "runner_sha256": sha256_file(Path(__file__)),
        "decision": audit["decision"],
        "next_direction_requires_council": not feasibility_pass,
    }
    write_json(output_path, evaluation)
    return evaluation


def fetch_metadata_page(url: str, *, retries: int = 3) -> tuple[bytes, dict[str, Any]]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read(5 * 1024 * 1024 + 1)
                if len(body) > 5 * 1024 * 1024:
                    raise ValueError("metadata page exceeds 5 MiB safety limit")
                if response.status != 200:
                    raise ValueError(f"metadata request returned HTTP {response.status}")
                return body, {
                    "url": response.geturl(),
                    "status": response.status,
                    "content_type": response.headers.get("Content-Type"),
                    "etag": response.headers.get("ETag"),
                    "last_modified": response.headers.get("Last-Modified"),
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "user_agent": USER_AGENT,
                    "bytes": len(body),
                    "sha256": sha256_bytes(body),
                    "attempt": attempt,
                }
        except (OSError, urllib.error.URLError, ValueError) as error:
            last_error = error
            if attempt < retries:
                time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"metadata fetch failed after {retries} attempts") from last_error


def materialize_html(
    *,
    url: str,
    cache_dir: Path,
    html_input: Path | None,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    html_path = cache_dir / "advanced-search.html"
    provenance_path = cache_dir / "advanced-search.provenance.json"
    if html_input is None:
        body, provenance = fetch_metadata_page(url)
        html_path.write_bytes(body)
    else:
        shutil.copyfile(html_input, html_path)
        body = html_path.read_bytes()
        provenance = {
            "url": url,
            "status": "offline_input",
            "input_path": str(html_input.resolve()),
            "fetched_at": None,
            "user_agent": None,
            "bytes": len(body),
            "sha256": sha256_bytes(body),
            "attempt": 0,
        }
    write_json(provenance_path, provenance)
    return html_path


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    contract = validate_contract()
    local_evidence = collect_local_evidence(contract)
    source_url = contract["spec"]["inputs"]["visual_lexicon_advanced_search_url"]
    html_path = materialize_html(
        url=source_url,
        cache_dir=args.cache_dir,
        html_input=args.html_input,
    )
    html = html_path.read_bytes()
    inventory, audit = build_audit(
        html, contract=contract, local_evidence=local_evidence
    )
    first_outputs = write_audit_outputs(args.output_dir, inventory, audit)
    replay_inventory, replay_audit = build_audit(
        html, contract=contract, local_evidence=local_evidence
    )
    replay_outputs = write_audit_outputs(
        args.replay_dir, replay_inventory, replay_audit
    )
    return evaluate(
        audit=audit,
        first_outputs=first_outputs,
        replay_outputs=replay_outputs,
        output_path=args.evaluation_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("all",),
        help="fetch/parse one metadata page, replay deterministically, and evaluate",
    )
    parser.add_argument("--html-input", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    parser.add_argument(
        "--evaluation-path", type=Path, default=DEFAULT_EVALUATION_PATH
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "all":
        evaluation = run_all(args)
        print(json.dumps(evaluation, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if evaluation["execution_integrity_pass"] else 2
    raise AssertionError(args.command)


if __name__ == "__main__":
    sys.exit(main())
