#!/usr/bin/env python3
"""Recertify the Visual Lexicon ceiling under the expanded v16 envelope.

Iteration 1 is deliberately local and metadata-only.  It groups the 50 local
catalogue rows into unique physical documents, preserves every v12 exclusion,
and emits an exhaustive disposition for the 50 frozen Visual Lexicon terms.
It performs no network request, image access, model inference, or label read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = (
    ROOT
    / ".omc/autoresearch/elements-baseline-replacement/runs/"
    "20260803-institutional-inventory-v16"
)
SPEC_PATH = RUN_DIR / "specs/iteration-0001.json"
EVALUATOR_PATH = RUN_DIR / "evaluator.json"
RULES_PATH = RUN_DIR / "local-identity-crosswalk-rules.json"
DEFAULT_OUTPUT_DIR = RUN_DIR / "iteration-0001"
DEFAULT_REPLAY_DIR = RUN_DIR / "iteration-0001-replay"
DEFAULT_EVALUATION_PATH = RUN_DIR / "evaluations/iteration-0001.json"

EXPECTED_SPEC_SHA256 = (
    "518c7d124906f42183aae944850f748d52306a3bab0911290a46fe2df7a6f7f7"
)
EXPECTED_EVALUATOR_SHA256 = (
    "13bbbc4a71066ae9c6549d19cbefada94ce5098bbc1fbd593b1d5cbe178e5722"
)
EXPECTED_RULES_SHA256 = (
    "4b9b88d9c14cf9b9f20c5145a77b0318fb18facc8f48137468bbfd5fe0e101f8"
)
EXPECTED_RUNTIME_HASHES = {
    "projection_sha256": "0ad6ce348eaf44c8844415a769b048b8f155017befa93f16254b1263b9241210",
    "prototypes_sha256": "ffc52f03696f03780a01302708b53e319dc24069187a6fd1a2927ea537b1cc54",
    "config_sha256": "f2150a97ec7eee50a42ac1a108ae4bc092d46a6b8e4a2d91de59349560fbf18b",
}
GENERIC_IDENTITY_TOKENS = frozenset({"codex", "codice"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_input(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def normalize_identity(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value).casefold()
    without_marks = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    tokens = re.sub(r"[^a-z0-9]+", " ", without_marks).split()
    return " ".join(token for token in tokens if token not in GENERIC_IDENTITY_TOKENS)


def validate_contract(
    spec_path: Path = SPEC_PATH,
    evaluator_path: Path = EVALUATOR_PATH,
    rules_path: Path = RULES_PATH,
) -> dict[str, Any]:
    hashes = {
        "spec_sha256": sha256_file(spec_path),
        "evaluator_sha256": sha256_file(evaluator_path),
        "rules_sha256": sha256_file(rules_path),
    }
    expected = {
        "spec_sha256": EXPECTED_SPEC_SHA256,
        "evaluator_sha256": EXPECTED_EVALUATOR_SHA256,
        "rules_sha256": EXPECTED_RULES_SHA256,
    }
    if hashes != expected:
        raise ValueError(f"v16 contract hash mismatch: {hashes}")

    spec = read_json(spec_path)
    evaluator = read_json(evaluator_path)
    rules = read_json(rules_path)
    gates = spec["pre_registered_gates"]
    integrity = spec["integrity_constraints"]
    if gates["local_codex_rows_equals"] != 50:
        raise ValueError("v16 local catalogue row gate changed")
    if gates["local_physical_document_identities_equals"] != 28:
        raise ValueError("v16 physical-document gate changed")
    if gates["minimum_admissible_manuscript_components"] != 42:
        raise ValueError("v16 minimum component threshold changed")
    if gates["portfolio_direct_authorization_threshold"] != 63:
        raise ValueError("v16 direct authorization threshold changed")
    forbidden = (
        "network_fetch_allowed",
        "record_detail_fetch_allowed",
        "image_read_or_download_allowed",
        "model_inference_allowed",
        "labels_or_final_test_read_allowed",
        "runtime_write_allowed",
        "manual_post_result_crosswalk_edit_allowed",
    )
    if any(integrity[name] is not False for name in forbidden):
        raise ValueError("v16 gate zero contains an allowed forbidden operation")
    if evaluator["promotion_eligible"] is not False:
        raise ValueError("v16 gate zero cannot be promotion eligible")
    if rules["policy"]["monotonicity"] != "v12 exclusions may not be removed":
        raise ValueError("v16 exclusion monotonicity changed")
    return {
        "spec": spec,
        "evaluator": evaluator,
        "rules": rules,
        **hashes,
    }


def _assert_hash(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: {actual}")
    return actual


def runtime_hashes() -> dict[str, str]:
    paths = {
        "projection_sha256": ROOT / "backend/codex_model/weights/projection.pt",
        "prototypes_sha256": ROOT / "backend/codex_model/weights/prototypes.pt",
        "config_sha256": ROOT / "backend/codex_model/config.json",
    }
    actual = {name: sha256_file(path) for name, path in paths.items()}
    if actual != EXPECTED_RUNTIME_HASHES:
        raise ValueError(f"runtime hash mismatch: {actual}")
    return actual


def load_frozen_inputs(contract: dict[str, Any]) -> dict[str, Any]:
    inputs = contract["spec"]["inputs"]
    resolved = {
        name: resolve_input(inputs[name])
        for name in (
            "local_codex_catalog",
            "visual_lexicon_inventory",
            "v12_exclusion_rules",
            "v13_archive_crosswalk",
        )
    }
    hashes = {
        "local_codex_catalog_sha256": _assert_hash(
            resolved["local_codex_catalog"],
            inputs["local_codex_catalog_sha256"],
            "local codex catalogue",
        ),
        "visual_lexicon_inventory_sha256": _assert_hash(
            resolved["visual_lexicon_inventory"],
            inputs["visual_lexicon_inventory_sha256"],
            "Visual Lexicon inventory",
        ),
        "v12_exclusion_rules_sha256": _assert_hash(
            resolved["v12_exclusion_rules"],
            inputs["v12_exclusion_rules_sha256"],
            "v12 exclusion rules",
        ),
        "v13_archive_crosswalk_sha256": _assert_hash(
            resolved["v13_archive_crosswalk"],
            inputs["v13_archive_crosswalk_sha256"],
            "v13 archive crosswalk",
        ),
    }
    with resolved["local_codex_catalog"].open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        codex_rows = list(csv.DictReader(handle))
    visual_inventory = read_json(resolved["visual_lexicon_inventory"])
    v12_rules = read_json(resolved["v12_exclusion_rules"])
    v13_crosswalk = read_json(resolved["v13_archive_crosswalk"])
    return {
        "codex_rows": codex_rows,
        "visual_inventory": visual_inventory,
        "v12_rules": v12_rules,
        "v13_crosswalk": v13_crosswalk,
        "input_hashes": hashes,
    }


def _require_unique(values: Iterable[int], label: str) -> list[int]:
    materialized = list(values)
    if len(materialized) != len(set(materialized)):
        raise ValueError(f"duplicate {label}")
    return materialized


def build_physical_documents(
    codex_rows: list[dict[str, str]],
    rules: dict[str, Any],
    gates: dict[str, Any],
    v13_crosswalk: dict[str, Any],
) -> list[dict[str, Any]]:
    rows_by_id = {int(row["id"]): row for row in codex_rows}
    ids = _require_unique(rows_by_id, "local codex id")
    expected_ids = list(range(1, gates["local_codex_rows_equals"] + 1))
    if sorted(ids) != expected_ids:
        raise ValueError("local codex ids must be exactly 1 through 50")

    assigned: dict[int, str] = {}
    documents: list[dict[str, Any]] = []
    for override in rules["physical_group_overrides"]:
        group_ids = _require_unique(override["local_codex_ids"], "override codex id")
        for codex_id in group_ids:
            if codex_id not in rows_by_id:
                raise ValueError(f"unknown codex id in physical override: {codex_id}")
            if codex_id in assigned:
                raise ValueError(f"codex id assigned twice: {codex_id}")
            assigned[codex_id] = override["physical_identity"]
        documents.append(
            {
                "physical_identity": override["physical_identity"],
                "local_codex_ids": sorted(group_ids),
                "local_titles": [rows_by_id[value]["titre"] for value in sorted(group_ids)],
                "grouping_basis": "frozen_override",
                "evidence": override["evidence"],
            }
        )

    for codex_id in expected_ids:
        if codex_id in assigned:
            continue
        documents.append(
            {
                "physical_identity": f"codex_{codex_id}",
                "local_codex_ids": [codex_id],
                "local_titles": [rows_by_id[codex_id]["titre"]],
                "grouping_basis": "default_singleton",
                "evidence": "No frozen physical grouping override.",
            }
        )
        assigned[codex_id] = f"codex_{codex_id}"

    accounted = sorted(
        codex_id for document in documents for codex_id in document["local_codex_ids"]
    )
    if accounted != expected_ids:
        raise ValueError("local codex ids are not accounted exactly once")
    if len(documents) != gates["local_physical_document_identities_equals"]:
        raise ValueError(f"unexpected physical-document count: {len(documents)}")

    mh_ids = gates["matricula_huexotzinco_section_ids_equal"]
    mh = next(
        (
            document
            for document in documents
            if document["physical_identity"] == "matricula_de_huexotzinco"
        ),
        None,
    )
    if mh is None or mh["local_codex_ids"] != mh_ids:
        raise ValueError("Matrícula de Huexotzinco physical grouping mismatch")

    archives_by_codex_id: dict[int, list[str]] = {value: [] for value in expected_ids}
    for archive in v13_crosswalk["archives"]:
        if archive["mapping_status"] != "unique_exact_cote_mapping":
            continue
        archives_by_codex_id[int(archive["unique_codex_id"])].append(
            archive["archive_name"]
        )
    for document in documents:
        document["supporting_unique_archives"] = sorted(
            archive
            for codex_id in document["local_codex_ids"]
            for archive in archives_by_codex_id[codex_id]
        )
        document["normalized_titles"] = [
            normalize_identity(title) for title in document["local_titles"]
        ]
    expected_mh_archives = [f"t_387_{index:02d}.zip" for index in range(1, 24)]
    if mh["supporting_unique_archives"] != expected_mh_archives:
        raise ValueError("v13 does not support all 23 frozen MH sections")
    return sorted(documents, key=lambda row: min(row["local_codex_ids"]))


def build_visual_crosswalk(
    visual_inventory: dict[str, Any],
    v12_rules: dict[str, Any],
    rules: dict[str, Any],
    physical_documents: list[dict[str, Any]],
    gates: dict[str, Any],
) -> list[dict[str, Any]]:
    manuscripts = visual_inventory["manuscripts"]
    if visual_inventory["manuscript_count"] != gates["visual_lexicon_terms_equals"]:
        raise ValueError("frozen Visual Lexicon manuscript count mismatch")
    term_ids = _require_unique(
        (int(row["term_id"]) for row in manuscripts), "Visual Lexicon term id"
    )
    if len(term_ids) != gates["visual_lexicon_terms_equals"]:
        raise ValueError("unexpected Visual Lexicon term count")

    physical_ids = {row["physical_identity"] for row in physical_documents}
    frozen_by_term = {
        int(row["term_id"]): row for row in rules["visual_exclusion_mappings"]
    }
    v12_by_term = {
        int(row["term_id"]): row
        for row in v12_rules["excluded_visual_lexicon_terms"]
    }
    if set(frozen_by_term) != set(v12_by_term):
        raise ValueError("v16 rules must preserve exactly every v12 exclusion")
    distinct_by_term: dict[int, list[dict[str, Any]]] = {}
    for review in rules["explicit_distinct_identity_reviews"]:
        distinct_by_term.setdefault(int(review["term_id"]), []).append(review)
        if review["against_physical_identity"] not in physical_ids:
            raise ValueError("distinct review references unknown physical identity")

    crosswalk: list[dict[str, Any]] = []
    for manuscript in manuscripts:
        term_id = int(manuscript["term_id"])
        display_name = manuscript["display_name"]
        base = {
            "term_id": term_id,
            "display_name": display_name,
            "normalized_display_name": normalize_identity(display_name),
        }
        if term_id in frozen_by_term:
            mapping = frozen_by_term[term_id]
            prior = v12_by_term[term_id]
            if display_name != prior["visual_manuscript"]:
                raise ValueError(f"v12 display mismatch for term {term_id}")
            if mapping["match_kind"] != prior["overlap_kind"]:
                raise ValueError(f"v12 overlap kind mismatch for term {term_id}")
            if mapping["physical_identity"] not in physical_ids:
                raise ValueError(f"unknown mapped physical identity for term {term_id}")
            crosswalk.append(
                {
                    **base,
                    "disposition": "excluded_preserved_from_v12",
                    "physical_identity": mapping["physical_identity"],
                    "match_kind": mapping["match_kind"],
                    "frozen_aliases": mapping["frozen_aliases"],
                    "normalized_aliases": [
                        normalize_identity(value) for value in mapping["frozen_aliases"]
                    ],
                    "evidence": prior["evidence"],
                }
            )
        elif term_id in distinct_by_term:
            crosswalk.append(
                {
                    **base,
                    "disposition": "admissible_explicit_distinct_identity_review",
                    "reviews": distinct_by_term[term_id],
                }
            )
        else:
            crosswalk.append(
                {
                    **base,
                    "disposition": "admissible_no_frozen_identity_match",
                    "evidence": "No explicit frozen identity or source-family match to the 28 local physical documents.",
                }
            )

    if len(crosswalk) != len(term_ids):
        raise ValueError("Visual Lexicon terms are not accounted exactly once")
    return sorted(crosswalk, key=lambda row: row["term_id"])


def build_audit(contract: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    frozen = load_frozen_inputs(contract)
    gates = contract["spec"]["pre_registered_gates"]
    before_runtime = runtime_hashes()
    physical_documents = build_physical_documents(
        frozen["codex_rows"], contract["rules"], gates, frozen["v13_crosswalk"]
    )
    visual_crosswalk = build_visual_crosswalk(
        frozen["visual_inventory"],
        frozen["v12_rules"],
        contract["rules"],
        physical_documents,
        gates,
    )
    excluded = [
        row for row in visual_crosswalk if row["disposition"].startswith("excluded")
    ]
    ambiguous = [
        row for row in visual_crosswalk if "ambiguous" in row["disposition"]
    ]
    admissible = [
        row for row in visual_crosswalk if row["disposition"].startswith("admissible")
    ]
    ceiling = len(admissible)
    if len(excluded) < gates["expanded_exclusion_count_minimum"]:
        raise ValueError("expanded exclusion envelope removed a v12 exclusion")
    if len(excluded) + len(admissible) != gates["visual_lexicon_terms_equals"]:
        raise ValueError("crosswalk dispositions are not exhaustive")
    if ceiling >= gates["portfolio_direct_authorization_threshold"]:
        decision = "portfolio_ceiling_acquired"
    elif ceiling >= gates["minimum_admissible_manuscript_components"]:
        decision = "intermediate_ceiling_requires_council"
    else:
        decision = "visual_lexicon_ceiling_recertified_but_insufficient"
    after_runtime = runtime_hashes()
    runtime_unchanged = before_runtime == after_runtime == EXPECTED_RUNTIME_HASHES

    audit = {
        "schema_version": "autoresearch-institutional-inventory-v16.gate-zero-audit",
        "iteration": 1,
        "phase": "visual_lexicon_gate_zero",
        "contract_hashes": {
            key: contract[key]
            for key in ("spec_sha256", "evaluator_sha256", "rules_sha256")
        },
        "input_hashes": frozen["input_hashes"],
        "metrics": {
            "local_codex_row_count": len(frozen["codex_rows"]),
            "local_physical_document_identity_count": len(physical_documents),
            "matricula_huexotzinco_section_count": 23,
            "visual_lexicon_term_count": len(visual_crosswalk),
            "preserved_v12_exclusion_count": len(excluded),
            "new_ambiguous_exclusion_count": len(ambiguous),
            "strict_admissible_visual_lexicon_physical_document_ceiling": ceiling,
            "crosswalk_coverage_fraction": 1.0,
        },
        "gates": {
            "all_local_ids_accounted_exactly_once": True,
            "all_visual_terms_accounted_exactly_once": True,
            "v12_exclusions_preserved": len(excluded) == 9,
            "ambiguity_excludes": True,
            "minimum_42_gate_pass": ceiling >= 42,
            "direct_authorization_63_gate_pass": ceiling >= 63,
        },
        "forbidden_operation_counters": {
            "network_fetches": 0,
            "record_detail_fetches": 0,
            "images_read_or_downloaded": 0,
            "model_predictions": 0,
            "labels_read": 0,
            "final_test_reads": 0,
            "runtime_writes": 0,
        },
        "runtime_before": before_runtime,
        "runtime_after": after_runtime,
        "runtime_unchanged": runtime_unchanged,
        "execution_integrity_pass": runtime_unchanged,
        "hypothesis_supported": True,
        "instrument_acquired": decision == "portfolio_ceiling_acquired",
        "promotion_eligible": False,
        "decision": decision,
        "next_action": "Submit gate zero to Council before freezing any official institutional discovery manifest.",
    }
    return physical_documents, visual_crosswalk, audit


def write_outputs(
    output_dir: Path,
    physical_documents: list[dict[str, Any]],
    visual_crosswalk: list[dict[str, Any]],
    audit: dict[str, Any],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "physical_document_crosswalk": output_dir / "physical-document-crosswalk.json",
        "visual_lexicon_crosswalk": output_dir / "visual-lexicon-crosswalk.json",
        "gate_zero_audit": output_dir / "gate-zero-audit.json",
    }
    write_json(
        paths["physical_document_crosswalk"],
        {
            "schema_version": "autoresearch-institutional-inventory-v16.physical-crosswalk",
            "physical_document_count": len(physical_documents),
            "documents": physical_documents,
        },
    )
    write_json(
        paths["visual_lexicon_crosswalk"],
        {
            "schema_version": "autoresearch-institutional-inventory-v16.visual-crosswalk",
            "term_count": len(visual_crosswalk),
            "terms": visual_crosswalk,
        },
    )
    write_json(paths["gate_zero_audit"], audit)
    return {name: sha256_file(path) for name, path in paths.items()}


def evaluate(
    audit: dict[str, Any],
    first_hashes: dict[str, str],
    replay_hashes: dict[str, str],
    output_path: Path,
) -> dict[str, Any]:
    deterministic_replay = first_hashes == replay_hashes
    integrity = audit["execution_integrity_pass"] and deterministic_replay
    evaluation = {
        "schema_version": "autoresearch-institutional-inventory-v16.evaluation",
        "iteration": 1,
        "execution_integrity_pass": integrity,
        "pass": audit["instrument_acquired"] and integrity,
        "hypothesis_supported": audit["hypothesis_supported"] and integrity,
        "instrument_acquired": audit["instrument_acquired"],
        "strict_component_ceiling": audit["metrics"][
            "strict_admissible_visual_lexicon_physical_document_ceiling"
        ],
        "local_physical_document_identity_count": audit["metrics"][
            "local_physical_document_identity_count"
        ],
        "preserved_v12_exclusion_count": audit["metrics"][
            "preserved_v12_exclusion_count"
        ],
        "deterministic_replay": deterministic_replay,
        "first_output_hashes": first_hashes,
        "replay_output_hashes": replay_hashes,
        "runtime_unchanged": audit["runtime_unchanged"],
        "final_test_read": False,
        "promotion_eligible": False,
        "decision": audit["decision"],
    }
    write_json(output_path, evaluation)
    return evaluation


def run(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    replay_dir: Path = DEFAULT_REPLAY_DIR,
    evaluation_path: Path = DEFAULT_EVALUATION_PATH,
) -> dict[str, Any]:
    contract = validate_contract()
    for path in (output_dir, replay_dir):
        if path.exists():
            shutil.rmtree(path)
    physical_documents, visual_crosswalk, audit = build_audit(contract)
    first_hashes = write_outputs(
        output_dir, physical_documents, visual_crosswalk, audit
    )
    replay_documents, replay_crosswalk, replay_audit = build_audit(contract)
    replay_hashes = write_outputs(
        replay_dir, replay_documents, replay_crosswalk, replay_audit
    )
    return evaluate(audit, first_hashes, replay_hashes, evaluation_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    parser.add_argument("--evaluation-path", type=Path, default=DEFAULT_EVALUATION_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evaluation = run(args.output_dir, args.replay_dir, args.evaluation_path)
    print(json.dumps(evaluation, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
