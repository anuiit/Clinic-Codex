#!/usr/bin/env python3
"""Run v16 gate 0-bis over the frozen Visual Lexicon term crosswalk.

The runner deduplicates physical/provenance components, applies two frozen
fail-closed ambiguity exclusions, and produces a corrected strict ceiling.
It is a closed local audit: no network, image, label, or model access occurs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = (
    ROOT
    / ".omc/autoresearch/elements-baseline-replacement/runs/"
    "20260803-institutional-inventory-v16"
)
SPEC_PATH = RUN_DIR / "specs/iteration-0002.json"
EVALUATOR_PATH = RUN_DIR / "evaluator-iteration-0002.json"
RULES_PATH = RUN_DIR / "visual-physical-dedup-rules.json"
DEFAULT_OUTPUT_DIR = RUN_DIR / "iteration-0002"
DEFAULT_REPLAY_DIR = RUN_DIR / "iteration-0002-replay"
DEFAULT_EVALUATION_PATH = RUN_DIR / "evaluations/iteration-0002.json"

EXPECTED_SPEC_SHA256 = (
    "1ec45f6fc763191c2102ee922ff47c30d9536899679ff068a7289b3d56ebdae0"
)
EXPECTED_EVALUATOR_SHA256 = (
    "1d7c5ff14a10726e925afb6e2b051d55e86f6d35b3bfa9a1263562dfad2d66d5"
)
EXPECTED_RULES_SHA256 = (
    "a19db82032dc84c29ddf146fd29cc6efec269b5933d43a7489d2864453a5789a"
)
EXPECTED_RUNTIME_HASHES = {
    "projection_sha256": "0ad6ce348eaf44c8844415a769b048b8f155017befa93f16254b1263b9241210",
    "prototypes_sha256": "ffc52f03696f03780a01302708b53e319dc24069187a6fd1a2927ea537b1cc54",
    "config_sha256": "f2150a97ec7eee50a42ac1a108ae4bc092d46a6b8e4a2d91de59349560fbf18b",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def resolve_input(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _unique(values: Iterable[int], label: str) -> list[int]:
    materialized = list(values)
    if len(materialized) != len(set(materialized)):
        raise ValueError(f"duplicate {label}")
    return materialized


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
        raise ValueError(f"v16 gate 0-bis contract hash mismatch: {hashes}")
    spec = read_json(spec_path)
    evaluator = read_json(evaluator_path)
    rules = read_json(rules_path)
    gates = spec["pre_registered_gates"]
    if gates["visual_lexicon_terms_equals"] != 50:
        raise ValueError("term count gate changed")
    if gates["strict_independent_component_ceiling_equals"] != 37:
        raise ValueError("strict component gate changed")
    if gates["minimum_admissible_components"] != 42:
        raise ValueError("minimum component threshold changed")
    if gates["portfolio_direct_authorization_threshold"] != 63:
        raise ValueError("direct authorization threshold changed")
    forbidden = spec["integrity_constraints"]
    allowed_false = [name for name, value in forbidden.items() if name.endswith("allowed")]
    if not allowed_false or any(forbidden[name] is not False for name in allowed_false):
        raise ValueError("gate 0-bis permits a forbidden operation")
    if evaluator["promotion_eligible"] is not False:
        raise ValueError("gate 0-bis cannot be promotion eligible")
    return {"spec": spec, "evaluator": evaluator, "rules": rules, **hashes}


def _assert_hash(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: {actual}")
    return actual


def load_inputs(contract: dict[str, Any]) -> dict[str, Any]:
    inputs = contract["spec"]["inputs"]
    names = (
        "iteration_0001_visual_crosswalk",
        "iteration_0001_physical_crosswalk",
        "iteration_0001_audit",
        "iteration_0001_evaluation",
    )
    paths = {name: resolve_input(inputs[name]) for name in names}
    hashes = {
        f"{name}_sha256": _assert_hash(
            paths[name], inputs[f"{name}_sha256"], name
        )
        for name in names
    }
    return {
        "visual_crosswalk": read_json(paths["iteration_0001_visual_crosswalk"]),
        "physical_crosswalk": read_json(paths["iteration_0001_physical_crosswalk"]),
        "prior_audit": read_json(paths["iteration_0001_audit"]),
        "prior_evaluation": read_json(paths["iteration_0001_evaluation"]),
        "input_hashes": hashes,
    }


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


def build_gate_zero_bis(
    contract: dict[str, Any], inputs: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    gates = contract["spec"]["pre_registered_gates"]
    rules = contract["rules"]
    prior_terms = inputs["visual_crosswalk"]["terms"]
    if len(prior_terms) != gates["visual_lexicon_terms_equals"]:
        raise ValueError("prior Visual Lexicon term count mismatch")
    prior_by_id = {int(row["term_id"]): row for row in prior_terms}
    if len(prior_by_id) != len(prior_terms):
        raise ValueError("duplicate prior Visual Lexicon term id")

    prior_excluded = {
        term_id
        for term_id, row in prior_by_id.items()
        if row["disposition"].startswith("excluded")
    }
    prior_admitted = set(prior_by_id) - prior_excluded
    if len(prior_excluded) != gates["preserved_v12_exclusion_count_equals"]:
        raise ValueError("prior v12 exclusion count mismatch")
    if len(prior_admitted) != gates[
        "historical_no_known_local_overlap_term_count_equals"
    ]:
        raise ValueError("historical upper-bound count mismatch")

    group_for_term: dict[int, dict[str, Any]] = {}
    for group in rules["same_component_groups"]:
        term_ids = _unique(group["term_ids"], "same-component term id")
        if len(term_ids) < 2:
            raise ValueError("same-component group must contain at least two terms")
        for term_id in term_ids:
            if term_id not in prior_admitted:
                raise ValueError(f"group term is not previously admitted: {term_id}")
            if term_id in group_for_term:
                raise ValueError(f"term belongs to two component groups: {term_id}")
            group_for_term[term_id] = group
    if len(rules["same_component_groups"]) != gates[
        "same_component_group_count_equals"
    ]:
        raise ValueError("same-component group count mismatch")

    extra_exclusions = {
        int(row["term_id"]): row
        for row in rules["additional_ambiguity_exclusions"]
    }
    if len(extra_exclusions) != gates["new_ambiguity_exclusion_count_equals"]:
        raise ValueError("additional ambiguity exclusion count mismatch")
    if not set(extra_exclusions).issubset(prior_admitted):
        raise ValueError("additional exclusion was not previously admitted")
    if set(extra_exclusions) & set(group_for_term):
        raise ValueError("term cannot be grouped and additionally excluded")

    dispositions: list[dict[str, Any]] = []
    components_by_id: dict[str, dict[str, Any]] = {}
    admitted_term_ids: list[int] = []
    for term_id, prior in sorted(prior_by_id.items()):
        base = {
            "term_id": term_id,
            "display_name": prior["display_name"],
            "prior_disposition": prior["disposition"],
        }
        if term_id in prior_excluded:
            dispositions.append(
                {**base, "disposition": "excluded_preserved_from_v12", "component_id": None}
            )
            continue
        if term_id in extra_exclusions:
            dispositions.append(
                {
                    **base,
                    "disposition": "excluded_unresolved_identity_or_source_family",
                    "component_id": None,
                    "reason": extra_exclusions[term_id]["reason"],
                    "against": extra_exclusions[term_id]["against"],
                }
            )
            continue

        admitted_term_ids.append(term_id)
        group = group_for_term.get(term_id)
        component_id = group["component_id"] if group else f"visual_term_{term_id}"
        component = components_by_id.setdefault(
            component_id,
            {
                "component_id": component_id,
                "component_kind": group["component_kind"] if group else "physical_document",
                "term_ids": [],
                "display_names": [],
                "evidence": group["evidence"] if group else "Singleton frozen Visual Lexicon identity.",
                "collision_audit_pending": True,
            },
        )
        component["term_ids"].append(term_id)
        component["display_names"].append(prior["display_name"])
        dispositions.append(
            {
                **base,
                "disposition": rules["policy"]["prior_admissible_relabel"],
                "component_id": component_id,
            }
        )

    if len(admitted_term_ids) != gates["admitted_term_count_equals"]:
        raise ValueError(f"unexpected admitted term count: {len(admitted_term_ids)}")
    components = sorted(components_by_id.values(), key=lambda row: min(row["term_ids"]))
    if len(components) != gates["strict_independent_component_ceiling_equals"]:
        raise ValueError(f"unexpected strict component ceiling: {len(components)}")

    component_300 = next(row for row in dispositions if row["term_id"] == 300)
    component_367 = next(row for row in dispositions if row["term_id"] == 367)
    if component_300["component_id"] == component_367["component_id"]:
        raise ValueError("terms 300 and 367 must retain distinct components")
    if set(admitted_term_ids) != {
        term_id for component in components for term_id in component["term_ids"]
    }:
        raise ValueError("admitted terms are not accounted exactly once")

    local_identities = {
        row["physical_identity"]
        for row in inputs["physical_crosswalk"]["documents"]
    }
    envelope_additions = {
        "schema_version": "autoresearch-institutional-inventory-v16.envelope-additions",
        "local_alias_additions": rules["local_alias_additions"],
        "future_external_exclusion_candidates": rules[
            "future_external_exclusion_candidates"
        ],
    }
    for addition in envelope_additions["local_alias_additions"]:
        if addition["physical_identity"] not in local_identities:
            raise ValueError("local alias addition references unknown identity")
    codex_49 = next(
        row
        for row in envelope_additions["local_alias_additions"]
        if row["physical_identity"] == "codex_49"
    )
    if set(codex_49["aliases"]) != {
        "TEPEUCILA",
        "tepeucila",
        "t_tepeuc",
        "t_tepeuc.zip",
    }:
        raise ValueError("codex_49 aliases are incomplete")

    ceiling = len(components)
    if ceiling >= gates["portfolio_direct_authorization_threshold"]:
        decision = "portfolio_ceiling_acquired"
    elif ceiling >= gates["minimum_admissible_components"]:
        decision = "intermediate_ceiling_requires_council"
    else:
        decision = "visual_lexicon_physical_ceiling_corrected_but_insufficient"
    audit = {
        "schema_version": "autoresearch-institutional-inventory-v16.gate-zero-bis-audit",
        "iteration": 2,
        "phase": "visual_lexicon_physical_provenance_deduplication",
        "contract_hashes": {
            key: contract[key]
            for key in ("spec_sha256", "evaluator_sha256", "rules_sha256")
        },
        "input_hashes": inputs["input_hashes"],
        "metrics": {
            "visual_lexicon_term_count": len(dispositions),
            "historical_no_known_local_overlap_term_count": len(prior_admitted),
            "preserved_v12_exclusion_count": len(prior_excluded),
            "new_ambiguity_exclusion_count": len(extra_exclusions),
            "admitted_term_count": len(admitted_term_ids),
            "same_component_group_count": len(rules["same_component_groups"]),
            "strict_independent_visual_lexicon_component_ceiling": ceiling,
            "all_term_coverage_fraction": 1.0,
        },
        "gates": {
            "all_visual_terms_accounted_exactly_once": len(dispositions) == 50,
            "all_admitted_terms_accounted_exactly_once": True,
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
        "historical_41_superseded_for_composite_arithmetic": True,
        "collision_audit_pending": True,
        "promotion_eligible": False,
        "instrument_acquired": decision == "portfolio_ceiling_acquired",
        "hypothesis_supported": ceiling == 37,
        "decision": decision,
        "next_action": "Submit corrected physical/provenance ceiling to Council before endpoint discovery.",
    }
    return dispositions, components, envelope_additions, audit


def build_audit(contract: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    before = runtime_hashes()
    outputs = build_gate_zero_bis(contract, load_inputs(contract))
    after = runtime_hashes()
    if before != after or after != EXPECTED_RUNTIME_HASHES:
        raise ValueError("runtime changed during gate 0-bis")
    outputs[3]["runtime_before"] = before
    outputs[3]["runtime_after"] = after
    outputs[3]["runtime_unchanged"] = True
    outputs[3]["execution_integrity_pass"] = True
    return outputs


def write_outputs(
    output_dir: Path,
    dispositions: list[dict[str, Any]],
    components: list[dict[str, Any]],
    envelope_additions: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "term_dispositions": output_dir / "visual-term-dispositions.json",
        "independent_components": output_dir / "independent-components.json",
        "envelope_additions": output_dir / "envelope-additions.json",
        "gate_zero_bis_audit": output_dir / "gate-zero-bis-audit.json",
    }
    write_json(
        paths["term_dispositions"],
        {
            "schema_version": "autoresearch-institutional-inventory-v16.term-dispositions",
            "term_count": len(dispositions),
            "terms": dispositions,
        },
    )
    write_json(
        paths["independent_components"],
        {
            "schema_version": "autoresearch-institutional-inventory-v16.independent-components",
            "component_count": len(components),
            "components": components,
        },
    )
    write_json(paths["envelope_additions"], envelope_additions)
    write_json(paths["gate_zero_bis_audit"], audit)
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
        "schema_version": "autoresearch-institutional-inventory-v16.gate-zero-bis-evaluation",
        "iteration": 2,
        "execution_integrity_pass": integrity,
        "pass": audit["instrument_acquired"] and integrity,
        "hypothesis_supported": audit["hypothesis_supported"] and integrity,
        "instrument_acquired": audit["instrument_acquired"],
        "historical_term_upper_bound": audit["metrics"][
            "historical_no_known_local_overlap_term_count"
        ],
        "corrected_strict_component_ceiling": audit["metrics"][
            "strict_independent_visual_lexicon_component_ceiling"
        ],
        "deterministic_replay": deterministic_replay,
        "first_output_hashes": first_hashes,
        "replay_output_hashes": replay_hashes,
        "runtime_unchanged": audit["runtime_unchanged"],
        "collision_audit_pending": True,
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
    first_outputs = build_audit(contract)
    first_hashes = write_outputs(output_dir, *first_outputs)
    replay_outputs = build_audit(contract)
    replay_hashes = write_outputs(replay_dir, *replay_outputs)
    return evaluate(first_outputs[3], first_hashes, replay_hashes, evaluation_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    parser.add_argument("--evaluation-path", type=Path, default=DEFAULT_EVALUATION_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(args.output_dir, args.replay_dir, args.evaluation_path)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
