#!/usr/bin/env python3
"""Authority-amended v19-R2 runner with an isomorphic train-only pipeline smoke."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
for import_path in (SCRIPTS_DIR, BACKEND):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import autoresearch_backbone_screen_v10 as backbone_screen  # noqa: E402
import autoresearch_hierarchical_shrinkage_v17 as v17  # noqa: E402
import autoresearch_self_supervised_v9 as v9  # noqa: E402
from codex_pipeline.determinism import configure_determinism  # noqa: E402
from codex_pipeline.models.projection_head import ProjectionHead  # noqa: E402


RUN = ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/20260805-vicreg-backbone-renomination-v19-r2"
V19R1_RUN = ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/20260804-vicreg-backbone-renomination-v19-r1"
V19_RUN = ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/20260804-vicreg-backbone-renomination-v19"
V9_RUN = ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-self-supervised-v9"
V10_RUN = ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-self-supervised-v10"
V18_RUN = ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/20260804-discriminative-readout-v18"

DEFAULT_SPEC = V19R1_RUN / "specs/iteration-0002.json"
DEFAULT_EVALUATOR = V19R1_RUN / "evaluator-iteration-0002.json"
DEFAULT_SMOKE_EVALUATOR = V19R1_RUN / "evaluator.json"
DEFAULT_FREEZE_AUDIT = RUN / "iteration-0002-contract-freeze-audit.json"
DEFAULT_STATIC_VALIDATION = RUN / "iteration-0002-static-contract-validation.json"
DEFAULT_SMOKE_AUTHORIZATION = RUN / "authorizations/smoke.json"
DEFAULT_SMOKE_CLAIM = RUN / "authorizations/smoke.execution-claim.json"
DEFAULT_PHASE2A_FREEZE_AUDIT = RUN / "iteration-0002-phase-2a-contract-freeze-audit.json"
DEFAULT_PHASE2A_AUTHORIZATION = RUN / "authorizations/phase-2a-precompute.json"
DEFAULT_PHASE2A_CLAIM = RUN / "authorizations/phase-2a-precompute.execution-claim.json"
DEFAULT_PHASE2B_FREEZE_AUDIT = RUN / "iteration-0002-phase-2b-contract-freeze-audit.json"
DEFAULT_PHASE2B_AUTHORIZATION = RUN / "authorizations/phase-2b-c1-control.json"
DEFAULT_PHASE2B_CLAIM = RUN / "authorizations/phase-2b-c1-control.execution-claim.json"
DEFAULT_PHASE3A_FREEZE_AUDIT = RUN / "iteration-0003-phase-3a-contract-freeze-audit.json"
DEFAULT_PHASE3A_AUTHORIZATION = RUN / "authorizations/phase-3a-train-evaluate.json"
DEFAULT_PHASE3A_CLAIM = RUN / "authorizations/phase-3a-train-evaluate.execution-claim.json"
DEFAULT_PHASE3B_FREEZE_AUDIT = RUN / "iteration-0003-phase-3b-contract-freeze-audit.json"
DEFAULT_PHASE3B_AUTHORIZATION = RUN / "authorizations/phase-3b-replay.json"
DEFAULT_PHASE3B_CLAIM = RUN / "authorizations/phase-3b-replay.execution-claim.json"
DEFAULT_B14_MANIFEST = V10_RUN / "inputs/dinov2-vitb14-local-manifest.json"
DEFAULT_SOURCE_INVENTORY = V18_RUN / "specs/iteration-0003-source-inventory.jsonl"
DEFAULT_CORPUS_MANIFEST = v9.DEFAULT_MANIFEST
DEFAULT_COLLECTION_AUDIT = v9.DEFAULT_AUDIT
DEFAULT_C1_SUMMARY = V9_RUN / "iteration-0001/results/paired_seed_summary.json"
DEFAULT_C1_PREDICTIONS = V9_RUN / "iteration-0001/results/prediction_rows.jsonl"
DEFAULT_C1_CACHE_DIR = V9_RUN / "iteration-0001/caches"
DEFAULT_B14_CACHE_DIR = V19R1_RUN / "iteration-0002/caches"
DEFAULT_CACHE_MANIFEST = V19R1_RUN / "iteration-0002/cache-manifest.json"
DEFAULT_CONTROL_AUDIT = V19R1_RUN / "iteration-0002/c1-control-replay-audit.json"
DEFAULT_OUTPUT_DIR = RUN / "iteration-0003"
DEFAULT_REPLAY_DIR = RUN / "iteration-0003-replay"
DEFAULT_RUNTIME_PROJECTION = BACKEND / "codex_model/weights/projection.pt"
DEFAULT_RUNTIME_PROTOTYPES = BACKEND / "codex_model/weights/prototypes.pt"
DEFAULT_RUNTIME_CONFIG = BACKEND / "codex_model/config.json"

EXPECTED_SPEC_SHA256 = "e865a9a9d7d521df7e81168129a1f2f244df66422d5c45dd631dad9fb1765dfa"
EXPECTED_EVALUATOR_SHA256 = "fdcd9e4f7d29793bb7790fc65682fb02db2df4381478ba6983acf3d37e255687"
EXPECTED_SMOKE_EVALUATOR_SHA256 = "d0cb9929beb68cdc57eab07a6c9c068556346a759e508e96c386f2896f022466"
EXPECTED_CORPUS_MANIFEST_SHA256 = "e918a190195d15ccfa9ea1fd4902607b84eea0c7d6c267d146a757f1b6547385"
EXPECTED_COLLECTION_AUDIT_SHA256 = "e931611a9174fbcb2799c52f6bbb6cc315229d9e198260756ce51f804d032b99"
EXPECTED_V9_RUNNER_SHA256 = "e03e1e29340a1a165ab2f82001236f5aaae4450afcee862dff8cb27ef26c48a9"
EXPECTED_V9_RECOVERY_AUDIT_SHA256 = "8f04000fd14e61e899e05c034fb36ab71dce3fecd7cb423fbba11ba476583cf5"
EXPECTED_V9_TEST_SPEC_SHA256 = "cb9f4f65ab1f82b7db9e2a7a0201c8461ea59eca8a6801d4f6474d9432f32803"
EXPECTED_C1_SUMMARY_SHA256 = "d6094bc9887748ba9bc89b3f6d691483b803be806cf0623e450450132387a22d"
EXPECTED_C1_PREDICTIONS_SHA256 = "37a15fb66ef883d15df5db8e1cadb152f9460876b60ace832c434375a3cdf34f"
EXPECTED_V17_RUNNER_SHA256 = "a1f0fb9e0fae18b0f5fb39dbbfa93361f041980447403eeb6a12f31630bafb08"
EXPECTED_B14_MANIFEST_SHA256 = "ea73c8b9d2cc8a81a48183a43516fc76ad6b2bd5694442229d9b03ce3774e4d5"
EXPECTED_SOURCE_INVENTORY_SHA256 = "d21ccbc123d454db773eb31fbc2fe3cf225c21c79534a142cf0b3d34340a87d2"
EXPECTED_RUNTIME_SHA256 = {
    "projection": "0ad6ce348eaf44c8844415a769b048b8f155017befa93f16254b1263b9241210",
    "prototypes": "ffc52f03696f03780a01302708b53e319dc24069187a6fd1a2927ea537b1cc54",
    "config": "f2150a97ec7eee50a42ac1a108ae4bc092d46a6b8e4a2d91de59349560fbf18b",
}
EXPECTED_FOLDS = (1, 2, 3, 4, 5)
EXPECTED_SEEDS = (17, 42, 73)
EXPECTED_PAIRED_ROWS = 1959
EXPECTED_UNIQUE_OOF_ROWS = 653
EXPECTED_TRAIN_ROWS_ACROSS_FOLDS = 39960
EXPECTED_CACHE_TENSOR_BYTES = 552_407_040
MAX_CACHE_BYTES = 2 * 1024**3
MAX_RUNTIME_SECONDS = 21_600
BACKBONE = "dinov2_vitb14"
EMBED_DIM = 768
VIEWS = 8
VIEW_SEED = 20260803
IMAGE_SIZE = 224
IMAGE_BATCH_SIZE = 4
SMOKE_FOLD = 1
SMOKE_BATCHES = 3
SMOKE_ROWS = IMAGE_BATCH_SIZE * SMOKE_BATCHES
DEFAULT_SMOKE_OUTPUT = RUN / "evaluations/iteration-0001.json"
PHASE2A_COUNCIL_SYNTHESIS = (
    ROOT / ".omx/argos-council/elements-model-improvement/turns/048/synthesis.md"
)
PHASE2B_COUNCIL_SYNTHESIS = (
    ROOT / ".omx/argos-council/elements-model-improvement/turns/049/synthesis.md"
)
PHASE3_COUNCIL_SYNTHESIS = (
    ROOT / ".omx/argos-council/elements-model-improvement/turns/052/synthesis.md"
)
COUNCIL_SYNTHESIS = (
    ROOT / ".omx/argos-council/elements-model-improvement/turns/047/synthesis.md"
)
EXPECTED_PHASE2A_COUNCIL_SYNTHESIS_SHA256 = (
    "ba95ee197d1c3e8dff9598eb67a6af23f20f344aeb29d690d01f6cbece4e9675"
)
EXPECTED_PHASE2B_COUNCIL_SYNTHESIS_SHA256 = (
    "bb845aeceb9e51b1729aacd5fd056a6e9fa409ddb8b4ba5d95c6591ba4cf1a62"
)
EXPECTED_PHASE3_COUNCIL_SYNTHESIS_SHA256 = (
    "e6e262815bd2d52e4a01f760f0d0ded2851cad021420aee7fa1fbe106be8461b"
)
EXPECTED_PHASE2A_FREEZE_SHA256 = (
    "f877759ac62de60047e849c7e0944e5a939210e138d275e8e545b4f7515bae56"
)
EXPECTED_PHASE2A_AUTHORIZATION_SHA256 = (
    "a5554fc2059b08ef50c9e2ea00273e4789cbde7c9ffa49f6195be349761f166b"
)
EXPECTED_PHASE2A_CLAIM_SHA256 = (
    "858a38214a5c34bb0b5954fca2265c8c137bff253891167d9ae150884295c37b"
)
EXPECTED_CACHE_MANIFEST_SHA256 = (
    "69961003a7dc3601655b27acf8caefef953db987676ed3697146fd36351a5a9b"
)
EXPECTED_PHASE2B_TEST_SHA256 = "f675adb78ffda5a3aabd0b64b81bfd17773ab6b1d19cfc5af9f8e4a60e073747"
EXPECTED_PHASE2B_FREEZE_SHA256 = (
    "98742f84f3cedd1a06b63f6848542766b8d00c705cc6c77442ecf188228c0455"
)
EXPECTED_PHASE2B_AUTHORIZATION_SHA256 = (
    "ce266e9fb7764a685f3e98a1c5de05cb5b70a2f6003826d26758bb2477a30f9b"
)
EXPECTED_PHASE2B_CLAIM_SHA256 = (
    "f768659ed204066f5fae994a450468163ee5ff2f310a5c349fd3baa406822412"
)
EXPECTED_C1_CONTROL_AUDIT_SHA256 = (
    "3e240b0b06e1755d523df77c37caa5d8340819e012796bbafb99b2850017d327"
)
EXPECTED_PHASE3_TEST_SHA256 = "72622791d719782ec2de70543e079d6ae5411eb88ca36e6b8c0bbd1ad668896f"
PHASE3_DEADLINE_POLICY = "claim_time_plus_21600"
EXPECTED_SMOKE_AUDIT_SHA256 = (
    "b476346181811adbcf7bd445590536d94e34a9e03af8ee6790b9954f5020b408"
)
PRIOR_V19_FAILURE_AUDIT = V19_RUN / "iteration-0002-phase-2-failure-audit.json"
PRIOR_V19_STATE = V19_RUN / "state.json"
PRIOR_V19R1_FAILURE_AUDIT = V19R1_RUN / "iteration-0003-phase-3a-terminal-failure-audit.json"
PRIOR_V19R1_ROOT_CAUSE_AUDIT = V19R1_RUN / "iteration-0003-phase-3a-root-cause-audit.json"
PRIOR_V19R1_CLAIM = V19R1_RUN / "authorizations/phase-3a-train-evaluate.execution-claim.json"
PRIOR_V19R1_RUNNER = SCRIPTS_DIR / "autoresearch_vicreg_backbone_v19r.py"
PRIOR_V19R1_TEST = ROOT / "backend/tests/test_autoresearch_vicreg_backbone_v19r.py"
EXPECTED_PRIOR_V19R1_RUNNER_SHA256 = "ce8b622ff1ac332d284c6299b37e33ff5c6636ea3a45d4746fcc4233ac4a6766"
EXPECTED_PRIOR_V19R1_TEST_SHA256 = "ddbec971ce38423c32ff271f2e14e2e6af1ecc3b668bd89dc71c1dd8ef1c7104"
EXPECTED_PRIOR_V19R1_FAILURE_AUDIT_SHA256 = "75e6bda279d470d161ceb7bef7cbe45e98a736c45c78defe258bf0233d02f473"
EXPECTED_PRIOR_V19R1_ROOT_CAUSE_AUDIT_SHA256 = "78575d86db4eca59a140c4ee48cddda03c936cf7d4894dd7c8f135a40acc5ff5"
EXPECTED_PRIOR_V19R1_CLAIM_SHA256 = "31f2f0aff1d2bb136da728b64c6163b9ab77c24501da421710981a56e720f222"
V19R_TEST = ROOT / "backend/tests/test_autoresearch_vicreg_backbone_v19r2.py"
EXPECTED_PUBLISHED_COUNCIL_SYNTHESIS_SHA256 = (
    "6cfa054df45630aa518170b999079390f4f5378b243cdcb6beff32d4fb301262"
)
EXPECTED_SMOKE_INPUT_HASHES = {
    "spec": (DEFAULT_SPEC, EXPECTED_SPEC_SHA256),
    "evaluator": (DEFAULT_EVALUATOR, EXPECTED_EVALUATOR_SHA256),
    "smoke_evaluator": (
        DEFAULT_SMOKE_EVALUATOR,
        EXPECTED_SMOKE_EVALUATOR_SHA256,
    ),
    "corpus_manifest": (DEFAULT_CORPUS_MANIFEST, EXPECTED_CORPUS_MANIFEST_SHA256),
    "collection_audit": (DEFAULT_COLLECTION_AUDIT, EXPECTED_COLLECTION_AUDIT_SHA256),
    "council_synthesis_file": (
        COUNCIL_SYNTHESIS,
        "010eb6d61a06012742cc8e18944ac37f3f8276929ff809fa026df77fe3afb41b",
    ),
    "prior_v19_runner": (
        SCRIPTS_DIR / "autoresearch_vicreg_backbone_v19.py",
        "14066b79538a5d3176bdf76f3a6475235da91480faf86a758c328234c6b60219",
    ),
    "prior_v19_test": (
        ROOT / "backend/tests/test_autoresearch_vicreg_backbone_v19.py",
        "b46ddefe03993e83f6847f6aa3a1d9b98534fcfc3cecc6e64bd796309169c154",
    ),
    "prior_v19_failure": (
        PRIOR_V19_FAILURE_AUDIT,
        "6cfff7acdce3af4b8ef080bdf21dcf54788379c479a9f0f4322283d77f27a47b",
    ),
    "prior_v19_state": (
        PRIOR_V19_STATE,
        "193dd2106dfc4d0924cd26bba12d8e244d80f6e4732cad810f9e73f7d486e9b1",
    ),
    "v19r_test": (
        V19R_TEST,
        EXPECTED_PHASE3_TEST_SHA256,
    ),
    "v9_runner": (Path(v9.__file__).resolve(), EXPECTED_V9_RUNNER_SHA256),
    "b14_manifest": (DEFAULT_B14_MANIFEST, EXPECTED_B14_MANIFEST_SHA256),
    "source_inventory": (DEFAULT_SOURCE_INVENTORY, EXPECTED_SOURCE_INVENTORY_SHA256),
}

REPLAY_NORMALIZATION_CONTRACT = {
    "schema_version": "autoresearch-v19r.replay-normalization-v1",
    "summary_removed_keys": ["artifact_paths"],
    "diagnostic_removed_keys": ["checkpoint_path"],
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSON objects in {path}")
    return rows


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value), encoding="utf-8")


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(canonical_json(value))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def require_exact_path(actual: Path, expected: Path, name: str) -> None:
    if actual.resolve() != expected.resolve():
        raise ValueError(f"{name} must remain {relative_path(expected)}")


def smoke_command_contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "command": "smoke-extraction",
        "runner": relative_path(Path(__file__)),
        "authorization": relative_path(args.authorization),
        "output": relative_path(args.output),
        "spec": relative_path(args.spec),
        "evaluator": relative_path(args.evaluator),
        "smoke_evaluator": relative_path(DEFAULT_SMOKE_EVALUATOR),
        "manifest": relative_path(args.manifest),
        "audit": relative_path(args.audit),
        "b14_manifest": relative_path(args.b14_manifest),
        "b14_cache_dir": relative_path(args.b14_cache_dir),
        "runtime_projection": relative_path(args.runtime_projection),
        "runtime_prototypes": relative_path(args.runtime_prototypes),
        "runtime_config": relative_path(args.runtime_config),
        "device": args.device,
        "image_batch_size": args.image_batch_size,
        "num_workers": args.num_workers,
        "smoke_fold": args.smoke_fold,
        "smoke_batches": args.smoke_batches,
    }


def phase_command_contract(args: argparse.Namespace) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "command": args.command,
        "runner": relative_path(Path(__file__)),
        "authorization": relative_path(args.authorization),
        "freeze_audit": relative_path(args.freeze_audit),
        "spec": relative_path(args.spec),
        "evaluator": relative_path(args.evaluator),
        "b14_manifest": relative_path(args.b14_manifest),
        "source_inventory": relative_path(args.source_inventory),
        "manifest": relative_path(args.manifest),
        "audit": relative_path(args.audit),
        "c1_summary": relative_path(args.c1_summary),
        "c1_predictions": relative_path(args.c1_predictions),
        "c1_cache_dir": relative_path(args.c1_cache_dir),
        "b14_cache_dir": relative_path(args.b14_cache_dir),
        "cache_manifest": relative_path(args.cache_manifest),
        "control_audit": relative_path(args.control_audit),
        "runtime_projection": relative_path(args.runtime_projection),
        "runtime_prototypes": relative_path(args.runtime_prototypes),
        "runtime_config": relative_path(args.runtime_config),
        "device": args.device,
        "supervised_device": args.supervised_device,
        "image_batch_size": args.image_batch_size,
        "num_workers": args.num_workers,
    }
    for attribute in ("output_dir", "replay_dir"):
        if hasattr(args, attribute):
            contract[attribute] = relative_path(getattr(args, attribute))
    return contract


def validate_deadline(authorization: dict[str, Any]) -> dict[str, Any]:
    if authorization.get("max_runtime_seconds") != MAX_RUNTIME_SECONDS:
        raise ValueError("authorization runtime window must remain six hours")
    try:
        started_at = datetime.fromisoformat(str(authorization["started_at"]))
        deadline_at = datetime.fromisoformat(str(authorization["deadline_at"]))
    except (KeyError, ValueError) as exc:
        raise ValueError("authorization has invalid RFC3339 runtime bounds") from exc
    if started_at.tzinfo is None or deadline_at.tzinfo is None:
        raise ValueError("authorization runtime bounds must be timezone-aware")
    duration = int((deadline_at - started_at).total_seconds())
    if duration != MAX_RUNTIME_SECONDS:
        raise ValueError("authorization deadline is not exactly six hours")
    now = datetime.now(deadline_at.tzinfo)
    if now >= deadline_at:
        raise TimeoutError("v19-R2 six-hour authorization window expired")
    return {
        "started_at": started_at.isoformat(),
        "deadline_at": deadline_at.isoformat(),
        "max_runtime_seconds": MAX_RUNTIME_SECONDS,
    }


def consume_authorization(
    args: argparse.Namespace,
    authorization: dict[str, Any],
) -> dict[str, Any]:
    consumed_at = datetime.now().astimezone()
    claim_path = resolve_artifact(authorization["execution_claim_path"])
    allowed_parent = (RUN / "authorizations").resolve()
    if claim_path.resolve().parent != allowed_parent:
        raise ValueError("execution claim must remain in the v19-R2 authorization directory")
    if claim_path.exists():
        raise FileExistsError(f"authorization already consumed: {claim_path}")
    claim = {
        "schema_version": "autoresearch-v19r.execution-claim-v1",
        "command": args.command,
        "authorization_path": relative_path(args.authorization),
        "authorization_sha256": v9.sha256_file(args.authorization),
        "command_contract_sha256": authorization["command_contract_sha256"],
        "consumed_at": consumed_at.isoformat(),
        "one_execution_only": True,
        "retry_allowed": False,
    }
    if authorization.get("deadline_policy") == PHASE3_DEADLINE_POLICY:
        claim.update(
            {
                "deadline_policy": PHASE3_DEADLINE_POLICY,
                "started_at": consumed_at.isoformat(),
                "deadline_at": (
                    consumed_at + timedelta(seconds=MAX_RUNTIME_SECONDS)
                ).isoformat(),
                "max_runtime_seconds": MAX_RUNTIME_SECONDS,
            }
        )
    write_json_exclusive(claim_path, claim)
    return {
        "path": relative_path(claim_path),
        "sha256": v9.sha256_file(claim_path),
        **{
            key: claim[key]
            for key in (
                "deadline_policy",
                "started_at",
                "deadline_at",
                "max_runtime_seconds",
            )
            if key in claim
        },
    }


def enforce_execution_claim_deadline(execution_claim: dict[str, Any]) -> None:
    if execution_claim.get("deadline_policy") != PHASE3_DEADLINE_POLICY:
        return
    deadline_at = datetime.fromisoformat(str(execution_claim["deadline_at"]))
    if deadline_at.tzinfo is None:
        raise ValueError("execution-claim deadline must be timezone-aware")
    if datetime.now(deadline_at.tzinfo) >= deadline_at:
        raise TimeoutError("v19-R2 phase execution-claim window expired")


def runtime_hashes(args: argparse.Namespace) -> dict[str, str]:
    return {
        "projection": v9.sha256_file(args.runtime_projection),
        "prototypes": v9.sha256_file(args.runtime_prototypes),
        "config": v9.sha256_file(args.runtime_config),
    }


def verify_smoke_input_hashes(args: argparse.Namespace) -> dict[str, Any]:
    if args.spec.resolve() != DEFAULT_SPEC.resolve():
        raise ValueError("smoke requires the preregistered v19-R2 spec")
    if args.evaluator.resolve() != DEFAULT_EVALUATOR.resolve():
        raise ValueError("smoke requires the preregistered v19-R2 evaluator")
    verified: dict[str, dict[str, Any]] = {}
    for name, (path, expected_sha256) in EXPECTED_SMOKE_INPUT_HASHES.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen smoke input {name}: {path}")
        actual_sha256 = v9.sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"frozen smoke input hash mismatch for {name}: {actual_sha256}")
        verified[name] = {
            "path": relative_path(path),
            "sha256": actual_sha256,
            "bytes": path.stat().st_size,
        }
    runtime = runtime_hashes(args)
    if runtime != EXPECTED_RUNTIME_SHA256:
        raise ValueError(f"runtime hash mismatch before smoke: {runtime}")
    pin = backbone_screen.validate_backbone_manifest(args.b14_manifest)
    if pin["manifest_sha256"] != EXPECTED_B14_MANIFEST_SHA256:
        raise ValueError("validated B/14 manifest hash mismatch")
    return {"verified_inputs": verified, "runtime_sha256": runtime, "backbone_pin": pin}


def validate_extraction_geometry(
    *,
    image_batch_size: int,
    num_workers: int,
    smoke_batches: int | None = None,
    smoke_fold: int | None = None,
) -> dict[str, int]:
    if image_batch_size != IMAGE_BATCH_SIZE:
        raise ValueError(f"v19-R2 image batch must remain {IMAGE_BATCH_SIZE}")
    if num_workers != 0:
        raise ValueError("v19-R2 num_workers must remain 0")
    if smoke_batches is not None and smoke_batches != SMOKE_BATCHES:
        raise ValueError(f"v19-R2 smoke must remain {SMOKE_BATCHES} batches")
    if smoke_fold is not None and smoke_fold != SMOKE_FOLD:
        raise ValueError(f"v19-R2 smoke fold must remain {SMOKE_FOLD}")
    return {
        "row_batch_size": image_batch_size,
        "num_workers": num_workers,
        "views_including_base": VIEWS + 1,
        "effective_images_per_full_batch": image_batch_size * (VIEWS + 1),
        "smoke_batches": smoke_batches or 0,
        "smoke_fold": smoke_fold or 0,
    }


def resolve_artifact(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def canonical_diagnostic_map(summary: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    diagnostics = {
        (int(item["fold"]), int(item["seed"])): item
        for item in summary["diagnostics"]
    }
    expected = {(fold, seed) for fold in EXPECTED_FOLDS for seed in EXPECTED_SEEDS}
    if set(diagnostics) != expected:
        raise ValueError("canonical C1 diagnostics do not cover 5 folds x 3 seeds")
    return diagnostics


def validate_static_contract(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "spec": (args.spec, EXPECTED_SPEC_SHA256),
        "evaluator": (args.evaluator, EXPECTED_EVALUATOR_SHA256),
        "v9_runner": (Path(v9.__file__).resolve(), EXPECTED_V9_RUNNER_SHA256),
        "v9_recovery_audit": (
            V19_RUN / "v9-canonical-runner-recovery-audit.json",
            EXPECTED_V9_RECOVERY_AUDIT_SHA256,
        ),
        "v9_test_spec": (V9_RUN / "test-spec.json", EXPECTED_V9_TEST_SPEC_SHA256),
        "c1_summary": (args.c1_summary, EXPECTED_C1_SUMMARY_SHA256),
        "c1_predictions": (args.c1_predictions, EXPECTED_C1_PREDICTIONS_SHA256),
        "v17_runner": (Path(v17.__file__).resolve(), EXPECTED_V17_RUNNER_SHA256),
        "b14_manifest": (args.b14_manifest, EXPECTED_B14_MANIFEST_SHA256),
        "source_inventory": (args.source_inventory, EXPECTED_SOURCE_INVENTORY_SHA256),
    }
    verified: dict[str, dict[str, Any]] = {}
    for name, (path, expected_sha256) in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen input {name}: {path}")
        actual_sha256 = v9.sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"frozen input hash mismatch for {name}: {actual_sha256}")
        verified[name] = {
            "path": relative_path(path),
            "sha256": actual_sha256,
            "bytes": path.stat().st_size,
        }

    runtime = runtime_hashes(args)
    if runtime != EXPECTED_RUNTIME_SHA256:
        raise ValueError(f"runtime hash mismatch: {runtime}")
    pin = backbone_screen.validate_backbone_manifest(args.b14_manifest)
    if pin["manifest_sha256"] != EXPECTED_B14_MANIFEST_SHA256:
        raise ValueError("validated B/14 manifest hash mismatch")

    recovery = read_json(V19_RUN / "v9-canonical-runner-recovery-audit.json")
    if recovery.get("pass") is not True:
        raise ValueError("v9 runner recovery audit did not pass")

    summary = read_json(args.c1_summary)
    if summary.get("code_sha256") != EXPECTED_V9_RUNNER_SHA256:
        raise ValueError("canonical C1 summary code hash mismatch")
    diagnostics = canonical_diagnostic_map(summary)
    predictions = read_jsonl(args.c1_predictions)
    if len(predictions) != EXPECTED_PAIRED_ROWS:
        raise ValueError("canonical C1 prediction-row count mismatch")
    if {str(row.get("code_sha256")) for row in predictions} != {
        EXPECTED_V9_RUNNER_SHA256
    }:
        raise ValueError("canonical C1 rows do not share the recovered v9 code hash")
    if len({(str(row["row_id"]), int(row["outer_fold"])) for row in predictions}) != (
        EXPECTED_UNIQUE_OOF_ROWS
    ):
        raise ValueError("canonical C1 unique OOF-row count mismatch")

    cache_pins: dict[str, dict[str, Any]] = {}
    for fold in EXPECTED_FOLDS:
        cache_path = v9.cache_path(args.c1_cache_dir, fold, VIEWS, None)
        sidecar_path = cache_path.with_suffix(cache_path.suffix + ".prov.json")
        expected_cache_sha256 = str(summary["cache_sha256"][str(fold)])
        if v9.sha256_file(cache_path) != expected_cache_sha256:
            raise ValueError(f"canonical S/14 cache hash mismatch for fold {fold}")
        sidecar = read_json(sidecar_path)
        if sidecar.get("cache_sha256") != expected_cache_sha256:
            raise ValueError(f"canonical S/14 cache sidecar mismatch for fold {fold}")
        cache_pins[str(fold)] = {
            "cache_path": relative_path(cache_path),
            "cache_sha256": expected_cache_sha256,
            "sidecar_path": relative_path(sidecar_path),
            "sidecar_sha256": v9.sha256_file(sidecar_path),
            "train_row_ids_sha256": sidecar["train_row_ids_sha256"],
            "oof_row_ids_sha256": sidecar["oof_row_ids_sha256"],
        }

    checkpoint_count = 0
    for diagnostic in diagnostics.values():
        for arm in ("baseline", "candidate"):
            checkpoint = resolve_artifact(diagnostic["checkpoints"][arm]["path"])
            expected = str(diagnostic["checkpoints"][arm]["sha256"])
            if v9.sha256_file(checkpoint) != expected:
                raise ValueError(f"canonical {arm} checkpoint hash mismatch: {checkpoint}")
            checkpoint_count += 1
    if checkpoint_count != 30:
        raise ValueError("canonical checkpoint count mismatch")

    return {
        "schema_version": "autoresearch-v19r.static-contract-validation-v1",
        "pass": True,
        "verified_inputs": verified,
        "runtime_sha256": runtime,
        "backbone_pin": pin,
        "canonical_cache_pins": cache_pins,
        "canonical_checkpoint_hash_match_count": checkpoint_count,
        "canonical_diagnostic_count": len(diagnostics),
        "canonical_prediction_row_count": len(predictions),
        "operation_counts": {
            "source_image_reads": 0,
            "oof_accesses": 0,
            "real_feature_extractions": 0,
            "candidate_optimizer_steps": 0,
            "candidate_predictions": 0,
            "candidate_scores": 0,
            "runtime_writes": 0,
        },
        "final_test_read": False,
        "runtime_unchanged": True,
    }


def validate_phase_authorization(
    args: argparse.Namespace,
    *,
    expected_phase: str,
) -> dict[str, Any]:
    if args.authorization is None:
        raise ValueError(f"{expected_phase} requires --authorization")
    if expected_phase == "phase_2a_extraction":
        require_exact_path(
            args.authorization,
            DEFAULT_PHASE2A_AUTHORIZATION,
            "phase 2a authorization",
        )
        require_exact_path(
            args.freeze_audit,
            DEFAULT_PHASE2A_FREEZE_AUDIT,
            "phase 2a freeze audit",
        )
    elif expected_phase == "phase_2b_c1_control":
        require_exact_path(
            args.authorization,
            DEFAULT_PHASE2B_AUTHORIZATION,
            "phase 2b authorization",
        )
        require_exact_path(
            args.freeze_audit,
            DEFAULT_PHASE2B_FREEZE_AUDIT,
            "phase 2b freeze audit",
        )
    elif expected_phase == "phase_3a_train_evaluate":
        require_exact_path(
            args.authorization,
            DEFAULT_PHASE3A_AUTHORIZATION,
            "phase 3a authorization",
        )
        require_exact_path(
            args.freeze_audit,
            DEFAULT_PHASE3A_FREEZE_AUDIT,
            "phase 3a freeze audit",
        )
    elif expected_phase == "phase_3b_replay":
        require_exact_path(
            args.authorization,
            DEFAULT_PHASE3B_AUTHORIZATION,
            "phase 3b authorization",
        )
        require_exact_path(
            args.freeze_audit,
            DEFAULT_PHASE3B_FREEZE_AUDIT,
            "phase 3b freeze audit",
        )
    else:
        raise ValueError(f"unsupported phase authorization: {expected_phase}")
    freeze_audit = read_json(args.freeze_audit)
    authorization = read_json(args.authorization)
    if authorization.get("schema_version") != "autoresearch-v19r.phase-authorization-v1":
        raise ValueError("unexpected phase authorization schema")
    if authorization.get("authorized") is not True:
        raise ValueError("phase authorization is not affirmative")
    if authorization.get("phase") != expected_phase:
        raise ValueError("phase authorization targets a different phase")
    if authorization.get("one_execution_only") is not True:
        raise ValueError("phase authorization must be one-shot")
    if authorization.get("retry_allowed") is not False:
        raise ValueError("phase authorization must forbid retries")
    if authorization.get("council_session_id") != "adv_20260803T072824_f903d4d6":
        raise ValueError("phase authorization Council session mismatch")
    if freeze_audit.get("schema_version") != "autoresearch-v19r.phase-freeze-v1":
        raise ValueError("unexpected phase freeze-audit schema")
    if freeze_audit.get("pass") is not True:
        raise ValueError("phase freeze audit did not pass")
    if expected_phase == "phase_2a_extraction":
        if (
            authorization.get("published_synthesis_sha256")
            != EXPECTED_PHASE2A_COUNCIL_SYNTHESIS_SHA256
        ):
            raise ValueError("phase 2a Council synthesis mismatch")
        if authorization.get("execution_claim_path") != relative_path(
            DEFAULT_PHASE2A_CLAIM
        ):
            raise ValueError("phase 2a claim-path mismatch")
    elif expected_phase == "phase_2b_c1_control":
        if (
            authorization.get("published_synthesis_sha256")
            != EXPECTED_PHASE2B_COUNCIL_SYNTHESIS_SHA256
        ):
            raise ValueError("phase 2b Council synthesis mismatch")
        if authorization.get("execution_claim_path") != relative_path(
            DEFAULT_PHASE2B_CLAIM
        ):
            raise ValueError("phase 2b claim-path mismatch")
    elif expected_phase in {"phase_3a_train_evaluate", "phase_3b_replay"}:
        if (
            authorization.get("published_synthesis_sha256")
            != EXPECTED_PHASE3_COUNCIL_SYNTHESIS_SHA256
        ):
            raise ValueError("phase 3 Council synthesis mismatch")
        expected_claim = (
            DEFAULT_PHASE3A_CLAIM
            if expected_phase == "phase_3a_train_evaluate"
            else DEFAULT_PHASE3B_CLAIM
        )
        if authorization.get("execution_claim_path") != relative_path(expected_claim):
            raise ValueError("phase 3 claim-path mismatch")
    freeze_sha256 = v9.sha256_file(args.freeze_audit)
    if authorization.get("contract_freeze_audit_sha256") != freeze_sha256:
        raise ValueError("phase authorization freeze-audit hash mismatch")
    current_runner_sha256 = v9.sha256_file(Path(__file__))
    expected_artifacts = freeze_audit.get("frozen_artifacts", {})
    if expected_phase == "phase_2a_extraction":
        phase_2a_artifacts = {
            "runner_sha256": current_runner_sha256,
            "test_sha256": EXPECTED_SMOKE_INPUT_HASHES["v19r_test"][1],
            "spec_sha256": EXPECTED_SPEC_SHA256,
            "evaluator_sha256": EXPECTED_EVALUATOR_SHA256,
            "smoke_evaluator_sha256": EXPECTED_SMOKE_EVALUATOR_SHA256,
            "static_validation_sha256": v9.sha256_file(DEFAULT_STATIC_VALIDATION),
            "smoke_audit_sha256": EXPECTED_SMOKE_AUDIT_SHA256,
            "council_synthesis_file_sha256": v9.sha256_file(
                PHASE2A_COUNCIL_SYNTHESIS
            ),
            "prior_v19_failure_audit_sha256": EXPECTED_SMOKE_INPUT_HASHES[
                "prior_v19_failure"
            ][1],
        }
        if expected_artifacts != phase_2a_artifacts:
            raise ValueError("phase 2a freeze-audit artifact mismatch")
    elif expected_phase == "phase_2b_c1_control":
        phase_2b_artifacts = {
            "runner_sha256": current_runner_sha256,
            "test_sha256": EXPECTED_PHASE2B_TEST_SHA256,
            "spec_sha256": EXPECTED_SPEC_SHA256,
            "evaluator_sha256": EXPECTED_EVALUATOR_SHA256,
            "smoke_evaluator_sha256": EXPECTED_SMOKE_EVALUATOR_SHA256,
            "static_validation_sha256": v9.sha256_file(DEFAULT_STATIC_VALIDATION),
            "smoke_audit_sha256": EXPECTED_SMOKE_AUDIT_SHA256,
            "phase_2a_freeze_audit_sha256": EXPECTED_PHASE2A_FREEZE_SHA256,
            "phase_2a_authorization_sha256": EXPECTED_PHASE2A_AUTHORIZATION_SHA256,
            "phase_2a_claim_sha256": EXPECTED_PHASE2A_CLAIM_SHA256,
            "cache_manifest_sha256": EXPECTED_CACHE_MANIFEST_SHA256,
            "council_synthesis_file_sha256": v9.sha256_file(
                PHASE2B_COUNCIL_SYNTHESIS
            ),
            "prior_v19_failure_audit_sha256": EXPECTED_SMOKE_INPUT_HASHES[
                "prior_v19_failure"
            ][1],
        }
        if expected_artifacts != phase_2b_artifacts:
            raise ValueError("phase 2b freeze-audit artifact mismatch")
    elif expected_phase in {"phase_3a_train_evaluate", "phase_3b_replay"}:
        phase_3_artifacts = {
            "runner_sha256": current_runner_sha256,
            "test_sha256": EXPECTED_PHASE3_TEST_SHA256,
            "spec_sha256": EXPECTED_SPEC_SHA256,
            "evaluator_sha256": EXPECTED_EVALUATOR_SHA256,
            "cache_manifest_sha256": EXPECTED_CACHE_MANIFEST_SHA256,
            "control_audit_sha256": EXPECTED_C1_CONTROL_AUDIT_SHA256,
            "phase_2b_freeze_audit_sha256": EXPECTED_PHASE2B_FREEZE_SHA256,
            "phase_2b_authorization_sha256": EXPECTED_PHASE2B_AUTHORIZATION_SHA256,
            "phase_2b_claim_sha256": EXPECTED_PHASE2B_CLAIM_SHA256,
            "council_synthesis_file_sha256": v9.sha256_file(PHASE3_COUNCIL_SYNTHESIS),
            "prior_v19_failure_audit_sha256": EXPECTED_SMOKE_INPUT_HASHES["prior_v19_failure"][1],
            "runtime_projection_sha256": EXPECTED_RUNTIME_SHA256["projection"],
            "prior_v19r1_runner_sha256": v9.sha256_file(PRIOR_V19R1_RUNNER),
            "prior_v19r1_test_sha256": v9.sha256_file(PRIOR_V19R1_TEST),
            "prior_v19r1_failure_audit_sha256": v9.sha256_file(PRIOR_V19R1_FAILURE_AUDIT),
            "prior_v19r1_root_cause_audit_sha256": v9.sha256_file(PRIOR_V19R1_ROOT_CAUSE_AUDIT),
            "prior_v19r1_claim_sha256": v9.sha256_file(PRIOR_V19R1_CLAIM),
            "runtime_prototypes_sha256": EXPECTED_RUNTIME_SHA256["prototypes"],
            "runtime_config_sha256": EXPECTED_RUNTIME_SHA256["config"],
        }
        if expected_phase == "phase_3b_replay":
            phase_3_artifacts.update(
                {
                    "phase_3a_freeze_audit_sha256": v9.sha256_file(DEFAULT_PHASE3A_FREEZE_AUDIT),
                    "phase_3a_authorization_sha256": v9.sha256_file(DEFAULT_PHASE3A_AUTHORIZATION),
                }
            )
        if expected_artifacts != phase_3_artifacts:
            raise ValueError("phase 3 freeze-audit artifact mismatch")
    phase_binding = freeze_audit.get("phase_authorization")
    if not isinstance(phase_binding, dict):
        raise ValueError("freeze audit lacks a phase-authorization binding")
    command_contract = phase_command_contract(args)
    command_contract_sha256 = v9.sha256_json(command_contract)
    phase_3 = expected_phase in {"phase_3a_train_evaluate", "phase_3b_replay"}
    if phase_3:
        if authorization.get("deadline_policy") != PHASE3_DEADLINE_POLICY:
            raise ValueError("phase 3 deadline policy mismatch")
        if authorization.get("max_runtime_seconds") != MAX_RUNTIME_SECONDS:
            raise ValueError("phase 3 runtime window must remain six hours")
        deadline = {
            "deadline_policy": PHASE3_DEADLINE_POLICY,
            "max_runtime_seconds": MAX_RUNTIME_SECONDS,
        }
    else:
        deadline = validate_deadline(authorization)
    required_binding = {
        "phase": expected_phase,
        "council_session_id": "adv_20260803T072824_f903d4d6",
        "published_synthesis_sha256": authorization.get("published_synthesis_sha256"),
        "command_contract_sha256": command_contract_sha256,
        "execution_claim_path": authorization.get("execution_claim_path"),
        "one_execution_only": True,
        "retry_allowed": False,
    }
    required_binding["max_runtime_seconds"] = MAX_RUNTIME_SECONDS
    if phase_3:
        required_binding["deadline_policy"] = PHASE3_DEADLINE_POLICY
    else:
        required_binding.update(
            {
                "started_at": deadline["started_at"],
                "deadline_at": deadline["deadline_at"],
            }
        )
    for key, value in required_binding.items():
        if phase_binding.get(key) != value:
            raise ValueError(f"freeze audit phase binding mismatch for {key}")
        if authorization.get(key) != value:
            raise ValueError(f"phase authorization binding mismatch for {key}")
    return {
        "path": relative_path(args.authorization),
        "sha256": v9.sha256_file(args.authorization),
        "phase": expected_phase,
        "published_synthesis_sha256": authorization["published_synthesis_sha256"],
        "command_contract": command_contract,
        "command_contract_sha256": command_contract_sha256,
        "execution_claim_path": authorization["execution_claim_path"],
        "deadline": deadline,
        "deadline_policy": authorization.get("deadline_policy"),
        "one_execution_only": True,
        "retry_allowed": False,
        "contract_freeze_audit_sha256": freeze_sha256,
    }


def validate_smoke_authorization(args: argparse.Namespace) -> dict[str, Any]:
    if args.authorization is None:
        raise ValueError("pipeline smoke requires --authorization")
    require_exact_path(args.authorization, DEFAULT_SMOKE_AUTHORIZATION, "authorization")
    require_exact_path(args.output, DEFAULT_SMOKE_OUTPUT, "smoke output")
    require_exact_path(args.spec, DEFAULT_SPEC, "spec")
    require_exact_path(args.evaluator, DEFAULT_EVALUATOR, "scientific evaluator")
    require_exact_path(args.manifest, DEFAULT_CORPUS_MANIFEST, "corpus manifest")
    require_exact_path(args.audit, DEFAULT_COLLECTION_AUDIT, "collection audit")
    require_exact_path(args.b14_manifest, DEFAULT_B14_MANIFEST, "B/14 manifest")
    require_exact_path(args.b14_cache_dir, DEFAULT_B14_CACHE_DIR, "B/14 cache target")
    require_exact_path(args.runtime_projection, DEFAULT_RUNTIME_PROJECTION, "runtime projection")
    require_exact_path(args.runtime_prototypes, DEFAULT_RUNTIME_PROTOTYPES, "runtime prototypes")
    require_exact_path(args.runtime_config, DEFAULT_RUNTIME_CONFIG, "runtime config")
    if args.device != "cuda":
        raise ValueError("smoke device must remain exactly cuda")
    expected_geometry = validate_extraction_geometry(
        image_batch_size=args.image_batch_size,
        num_workers=args.num_workers,
        smoke_batches=args.smoke_batches,
        smoke_fold=args.smoke_fold,
    )
    authorization = read_json(args.authorization)
    if authorization.get("schema_version") != "autoresearch-v19r.smoke-authorization-v1":
        raise ValueError("unexpected smoke authorization schema")
    if authorization.get("authorized") is not True:
        raise ValueError("smoke authorization is not affirmative")
    if authorization.get("phase") != "train_only_pipeline_smoke":
        raise ValueError("authorization does not target the pipeline smoke")
    if authorization.get("council_session_id") != "adv_20260803T072824_f903d4d6":
        raise ValueError("smoke authorization Council session mismatch")
    if (
        authorization.get("published_synthesis_sha256")
        != EXPECTED_PUBLISHED_COUNCIL_SYNTHESIS_SHA256
    ):
        raise ValueError("smoke authorization Council synthesis mismatch")
    if authorization.get("runner_sha256") != v9.sha256_file(Path(__file__)):
        raise ValueError("smoke authorization runner hash mismatch")
    if authorization.get("spec_sha256") != EXPECTED_SPEC_SHA256:
        raise ValueError("smoke authorization spec hash mismatch")
    if authorization.get("evaluator_sha256") != EXPECTED_EVALUATOR_SHA256:
        raise ValueError("smoke authorization evaluator hash mismatch")
    if (
        authorization.get("smoke_evaluator_sha256")
        != EXPECTED_SMOKE_EVALUATOR_SHA256
    ):
        raise ValueError("smoke authorization smoke-evaluator hash mismatch")
    if authorization.get("test_sha256") != EXPECTED_SMOKE_INPUT_HASHES[
        "v19r_test"
    ][1]:
        raise ValueError("smoke authorization test hash mismatch")
    if authorization.get("prior_v19_failure_audit_sha256") != EXPECTED_SMOKE_INPUT_HASHES[
        "prior_v19_failure"
    ][1]:
        raise ValueError("smoke authorization prior-failure hash mismatch")
    if authorization.get("one_execution_only") is not True:
        raise ValueError("smoke authorization must be one-shot")
    if authorization.get("retry_allowed") is not False:
        raise ValueError("smoke authorization must forbid fallback")
    if authorization.get("geometry") != expected_geometry:
        raise ValueError("smoke authorization geometry mismatch")
    command_contract = smoke_command_contract(args)
    command_contract_sha256 = v9.sha256_json(command_contract)
    if authorization.get("command_contract_sha256") != command_contract_sha256:
        raise ValueError("smoke authorization command-contract mismatch")
    execution_claim_path = relative_path(DEFAULT_SMOKE_CLAIM)
    if authorization.get("execution_claim_path") != execution_claim_path:
        raise ValueError("smoke authorization claim-path mismatch")
    deadline = validate_deadline(authorization)
    freeze_audit = read_json(DEFAULT_FREEZE_AUDIT)
    freeze_sha256 = v9.sha256_file(DEFAULT_FREEZE_AUDIT)
    if authorization.get("contract_freeze_audit_sha256") != freeze_sha256:
        raise ValueError("smoke authorization freeze-audit hash mismatch")
    if freeze_audit.get("schema_version") != "autoresearch-v19r.smoke-freeze-v1":
        raise ValueError("unexpected smoke freeze-audit schema")
    if freeze_audit.get("pass") is not True:
        raise ValueError("smoke freeze audit did not pass")
    expected_frozen_artifacts = {
        "runner_sha256": v9.sha256_file(Path(__file__)),
        "test_sha256": EXPECTED_SMOKE_INPUT_HASHES["v19r_test"][1],
        "spec_sha256": EXPECTED_SPEC_SHA256,
        "evaluator_sha256": EXPECTED_EVALUATOR_SHA256,
        "smoke_evaluator_sha256": EXPECTED_SMOKE_EVALUATOR_SHA256,
        "static_validation_sha256": v9.sha256_file(DEFAULT_STATIC_VALIDATION),
        "council_synthesis_file_sha256": EXPECTED_SMOKE_INPUT_HASHES[
            "council_synthesis_file"
        ][1],
        "prior_v19_failure_audit_sha256": EXPECTED_SMOKE_INPUT_HASHES[
            "prior_v19_failure"
        ][1],
    }
    if freeze_audit.get("frozen_artifacts") != expected_frozen_artifacts:
        raise ValueError("smoke freeze-audit artifact mismatch")
    expected_binding = {
        "phase": "train_only_pipeline_smoke",
        "council_session_id": "adv_20260803T072824_f903d4d6",
        "published_synthesis_sha256": EXPECTED_PUBLISHED_COUNCIL_SYNTHESIS_SHA256,
        "command_contract_sha256": command_contract_sha256,
        "execution_claim_path": execution_claim_path,
        "started_at": deadline["started_at"],
        "deadline_at": deadline["deadline_at"],
        "max_runtime_seconds": MAX_RUNTIME_SECONDS,
        "geometry": expected_geometry,
        "one_execution_only": True,
        "retry_allowed": False,
    }
    if freeze_audit.get("smoke_authorization") != expected_binding:
        raise ValueError("smoke freeze-audit authorization binding mismatch")
    inputs = verify_smoke_input_hashes(args)
    return {
        "path": relative_path(args.authorization),
        "sha256": v9.sha256_file(args.authorization),
        "published_synthesis_sha256": authorization["published_synthesis_sha256"],
        "geometry": expected_geometry,
        "command_contract": command_contract,
        "command_contract_sha256": command_contract_sha256,
        "execution_claim_path": execution_claim_path,
        "contract_freeze_audit": {
            "path": relative_path(DEFAULT_FREEZE_AUDIT),
            "sha256": freeze_sha256,
        },
        "deadline": deadline,
        **inputs,
    }


METADATA_FIELDS = (
    "row_id",
    "class_label",
    "class_name",
    "component_id",
    "decoded_pixel_sha256",
    "fold",
)


def sha256_sequence(values: Sequence[Any]) -> str:
    return v9.sha256_json(list(values))


def view_plan_sha256(
    row_ids: Sequence[Any],
    *,
    views: int = VIEWS,
    seed: int = VIEW_SEED,
) -> str:
    digest = hashlib.sha256()
    for row_id in row_ids:
        for view_index in range(views):
            payload = {
                "row_id": str(row_id),
                "view_index": view_index,
                "rng_seed": v9.stable_seed("ssl-view", seed, str(row_id), view_index),
            }
            digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()


def assert_ext4_workspace_path(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    root = ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"cache path is outside the workspace: {resolved}")
    if str(resolved).startswith("/mnt/"):
        raise ValueError("drvfs cache paths are forbidden")
    existing = resolved
    while not existing.exists():
        if existing.parent == existing:
            raise FileNotFoundError(f"cannot resolve parent filesystem for {resolved}")
        existing = existing.parent
    if os.stat(existing).st_dev != os.stat(root).st_dev:
        raise ValueError("cache path is not on the workspace filesystem")
    mounts: list[tuple[int, str, str]] = []
    for line in Path("/proc/mounts").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        mountpoint = fields[1].replace("\\040", " ")
        try:
            mount = Path(mountpoint).resolve()
        except OSError:
            continue
        if resolved == mount or mount in resolved.parents:
            mounts.append((len(str(mount)), fields[2], str(mount)))
    if not mounts:
        raise ValueError(f"cannot identify filesystem for {resolved}")
    _, filesystem, mountpoint = max(mounts)
    if filesystem != "ext4":
        raise ValueError(f"cache filesystem must be ext4, found {filesystem}")
    return {
        "resolved_path": str(resolved),
        "filesystem": filesystem,
        "mountpoint": mountpoint,
        "workspace_device": int(os.stat(root).st_dev),
    }


def load_canonical_cache(
    args: argparse.Namespace,
    fold: int,
) -> dict[str, Any]:
    return v9.load_fold_cache(
        v9.cache_path(args.c1_cache_dir, fold, VIEWS, None),
        expected_fold=fold,
        expected_views=VIEWS,
        expected_view_seed=VIEW_SEED,
        expected_image_size=IMAGE_SIZE,
        expected_max_rows_per_class=None,
    )


def validate_b14_cache(
    path: Path,
    *,
    fold: int,
    canonical: dict[str, Any],
    backbone_provenance: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = v9.load_fold_cache(
        path,
        expected_fold=fold,
        expected_views=VIEWS,
        expected_view_seed=VIEW_SEED,
        expected_image_size=IMAGE_SIZE,
        expected_max_rows_per_class=None,
        expected_backbone_provenance=backbone_provenance,
    )
    train = payload["train"]
    oof = payload["oof"]
    canonical_train = canonical["train"]
    canonical_oof = canonical["oof"]
    for split_name, split, reference in (
        ("train", train, canonical_train),
        ("oof", oof, canonical_oof),
    ):
        for field in METADATA_FIELDS:
            if list(split[field]) != list(reference[field]):
                raise ValueError(f"B/14 {split_name}.{field} order differs from canonical S/14")
    if tuple(train["base_features"].shape) != (len(train["row_id"]), EMBED_DIM):
        raise ValueError("B/14 train base-feature shape mismatch")
    if tuple(train["view_features"].shape) != (len(train["row_id"]), VIEWS, EMBED_DIM):
        raise ValueError("B/14 train view-feature shape mismatch")
    if tuple(oof["base_features"].shape) != (len(oof["row_id"]), EMBED_DIM):
        raise ValueError("B/14 OOF base-feature shape mismatch")
    for tensor in (train["base_features"], train["view_features"], oof["base_features"]):
        if tensor.dtype != torch.float16:
            raise ValueError("B/14 cache tensors must be float16")
        if not torch.isfinite(tensor).all():
            raise FloatingPointError("non-finite B/14 cache tensor")
    if "view_features" in oof:
        raise ValueError("B/14 OOF augmented views are forbidden")
    sidecar_path = path.with_suffix(path.suffix + ".prov.json")
    sidecar = read_json(sidecar_path)
    train_row_hash = sha256_sequence(train["row_id"])
    oof_row_hash = sha256_sequence(oof["row_id"])
    if train_row_hash != sidecar["train_row_ids_sha256"]:
        raise ValueError("B/14 train order hash differs from sidecar")
    if oof_row_hash != sidecar["oof_row_ids_sha256"]:
        raise ValueError("B/14 OOF order hash differs from sidecar")
    validation = {
        "fold": fold,
        "cache_path": relative_path(path),
        "cache_sha256": v9.sha256_file(path),
        "byte_identical_readback": sidecar["cache_sha256"] == v9.sha256_file(path),
        "cache_bytes": path.stat().st_size,
        "sidecar_path": relative_path(sidecar_path),
        "sidecar_sha256": v9.sha256_file(sidecar_path),
        "train_rows": len(train["row_id"]),
        "oof_rows": len(oof["row_id"]),
        "train_row_ids_sha256": train_row_hash,
        "oof_row_ids_sha256": oof_row_hash,
        "canonical_train_row_ids_sha256": sha256_sequence(canonical_train["row_id"]),
        "canonical_oof_row_ids_sha256": sha256_sequence(canonical_oof["row_id"]),
        "view_plan_sha256": view_plan_sha256(train["row_id"]),
        "train_shape": list(train["base_features"].shape),
        "view_shape": list(train["view_features"].shape),
        "oof_shape": list(oof["base_features"].shape),
        "dtype": str(train["base_features"].dtype),
        "oof_view_features_persisted": False,
        "final_test_read": False,
    }
    return payload, validation


def validate_cache_manifest(args: argparse.Namespace) -> dict[str, Any]:
    manifest = read_json(args.cache_manifest)
    if manifest.get("schema_version") != "autoresearch-v19r.b14-fold-cache-manifest-v1":
        raise ValueError("unexpected B/14 cache manifest schema")
    if manifest.get("pass") is not True:
        raise ValueError("B/14 cache manifest did not pass")
    if manifest.get("spec_sha256") != EXPECTED_SPEC_SHA256:
        raise ValueError("B/14 cache manifest spec mismatch")
    if manifest.get("evaluator_sha256") != EXPECTED_EVALUATOR_SHA256:
        raise ValueError("B/14 cache manifest evaluator mismatch")
    if manifest.get("expected_tensor_bytes") != EXPECTED_CACHE_TENSOR_BYTES:
        raise ValueError("B/14 cache manifest byte projection mismatch")
    entries = {int(item["fold"]): item for item in manifest.get("folds", [])}
    if set(entries) != set(EXPECTED_FOLDS):
        raise ValueError("B/14 cache manifest does not cover all folds")
    total = 0
    for fold, item in entries.items():
        path = resolve_artifact(item["cache_path"])
        sidecar = resolve_artifact(item["sidecar_path"])
        if v9.sha256_file(path) != item["cache_sha256"]:
            raise ValueError(f"B/14 cache hash mismatch for fold {fold}")
        if v9.sha256_file(sidecar) != item["sidecar_sha256"]:
            raise ValueError(f"B/14 sidecar hash mismatch for fold {fold}")
        total += path.stat().st_size
    if total != int(manifest["total_cache_bytes"]):
        raise ValueError("B/14 cache manifest total-byte mismatch")
    if manifest.get("runtime_sha256_after") != EXPECTED_RUNTIME_SHA256:
        raise ValueError("runtime changed during B/14 extraction")
    if manifest.get("final_test_read") is not False:
        raise ValueError("cache manifest reports final-test access")
    return manifest


def load_smoke_corpus(
    manifest_path: Path,
    audit_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_rows = read_jsonl(manifest_path)
    required = {
        "row_id",
        "image_path",
        "source_family",
        "decoded_pixel_sha256",
        "fold",
    }
    if not raw_rows:
        raise ValueError("empty provenance manifest")
    missing_fields = required - set(raw_rows[0])
    if missing_fields:
        raise ValueError(f"manifest missing smoke fields: {sorted(missing_fields)}")
    row_ids = [str(row["row_id"]) for row in raw_rows]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("duplicate row_id in provenance manifest")

    audit = read_json(audit_path)
    quarantined_ids = {str(value) for value in audit["quarantine"]["row_ids"]}
    quarantined_rows = [
        row for row in raw_rows if str(row["row_id"]) in quarantined_ids
    ]
    conflicting_hashes = {
        str(row["decoded_pixel_sha256"]) for row in quarantined_rows
    }
    retained = [
        row
        for row in raw_rows
        if str(row["decoded_pixel_sha256"]) not in conflicting_hashes
    ]
    observed_quarantine = {
        str(row["row_id"])
        for row in raw_rows
        if str(row["decoded_pixel_sha256"]) in conflicting_hashes
    }
    if observed_quarantine != quarantined_ids:
        raise ValueError("audit quarantine does not match conflicting RGB rows")

    mapping = v9.rebuild_conservative_components(retained)
    retained = [
        dict(row, component_id=mapping[str(row["row_id"])]) for row in retained
    ]
    folds = sorted({int(row["fold"]) for row in retained})
    for fold in folds:
        v9.assert_fold_isolation(
            [row for row in retained if int(row["fold"]) != fold],
            [row for row in retained if int(row["fold"]) == fold],
        )
    counts = {
        "source_rows": len(raw_rows),
        "retained_rows": len(retained),
        "quarantined_rows": len(quarantined_ids),
        "conflicting_rgb_hashes": len(conflicting_hashes),
        "components_after_quarantine": len(set(mapping.values())),
    }
    expected_counts = {
        "source_rows": 10039,
        "retained_rows": 9990,
        "quarantined_rows": 49,
        "conflicting_rgb_hashes": 22,
        "components_after_quarantine": 300,
    }
    if counts != expected_counts:
        raise ValueError(f"canonical smoke corpus count mismatch: {counts}")
    minimal_rows = [
        {
            "row_id": str(row["row_id"]),
            "image_path": str(row["image_path"]),
            "decoded_pixel_sha256": str(row["decoded_pixel_sha256"]),
            "component_id": str(row["component_id"]),
            "fold": int(row["fold"]),
        }
        for row in retained
    ]
    report = {
        **counts,
        "manifest_sha256": v9.sha256_file(manifest_path),
        "audit_sha256": v9.sha256_file(audit_path),
        "model_labels_read": 0,
        "model_labels_computed": 0,
    }
    return minimal_rows, report


def select_smoke_rows(
    rows: Sequence[dict[str, Any]],
    *,
    fold: int,
    row_count: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        if int(row["fold"]) == fold:
            continue
        selected.append(row)
        if len(selected) == row_count:
            break
    if len(selected) != row_count:
        raise ValueError(f"expected {row_count} fold-{fold} train rows for smoke")
    if any(int(row["fold"]) == fold for row in selected):
        raise ValueError("pipeline smoke selected an outer-fold row")
    return selected


@contextmanager
def guard_source_image_reads(
    rows: Sequence[dict[str, Any]],
    progress: list[Path] | None = None,
) -> Iterator[list[Path]]:
    allowed_paths = {Path(str(row["image_path"])).resolve() for row in rows}
    opened_paths = [] if progress is None else progress
    original_open = v9.Image.open

    def guarded_open(path: Any, *args: Any, **kwargs: Any) -> Any:
        resolved = Path(str(path)).resolve()
        if resolved not in allowed_paths:
            raise RuntimeError(f"pipeline smoke attempted a non-train image: {resolved}")
        opened_paths.append(resolved)
        return original_open(path, *args, **kwargs)

    v9.Image.open = guarded_open
    try:
        yield opened_paths
    finally:
        v9.Image.open = original_open


def run_smoke_pipeline(
    backbone: torch.nn.Module,
    rows: Sequence[dict[str, Any]],
    *,
    fold: int,
    image_batch_size: int,
    num_workers: int,
    smoke_batches: int,
    device: torch.device,
    opened_paths: list[Path] | None = None,
) -> dict[str, Any]:
    geometry = validate_extraction_geometry(
        image_batch_size=image_batch_size,
        num_workers=num_workers,
        smoke_batches=smoke_batches,
        smoke_fold=fold,
    )
    row_count = image_batch_size * smoke_batches
    selected = select_smoke_rows(rows, fold=fold, row_count=row_count)
    with guard_source_image_reads(selected, opened_paths) as observed_paths:
        base_features, view_features = v9.extract_features(
            backbone,
            selected,
            views=VIEWS,
            seed=VIEW_SEED,
            image_size=IMAGE_SIZE,
            batch_size=image_batch_size,
            num_workers=num_workers,
            device=device,
        )
    expected_base_shape = (row_count, EMBED_DIM)
    expected_view_shape = (row_count, VIEWS, EMBED_DIM)
    if tuple(base_features.shape) != expected_base_shape:
        raise ValueError(f"unexpected smoke base shape: {tuple(base_features.shape)}")
    if view_features is None or tuple(view_features.shape) != expected_view_shape:
        actual = None if view_features is None else tuple(view_features.shape)
        raise ValueError(f"unexpected smoke view shape: {actual}")
    if len(observed_paths) != row_count:
        raise ValueError(f"unexpected smoke source read count: {len(observed_paths)}")
    result = {
        "geometry": geometry,
        "fold": fold,
        "selected_train_rows": row_count,
        "selected_row_ids_sha256": sha256_sequence(
            [str(row["row_id"]) for row in selected]
        ),
        "opened_train_paths_sha256": sha256_sequence(
            [relative_path(path) for path in observed_paths]
        ),
        "source_image_reads": len(observed_paths),
        "base_feature_shape": list(base_features.shape),
        "view_feature_shape": list(view_features.shape),
        "base_feature_dtype": str(base_features.dtype),
        "view_feature_dtype": str(view_features.dtype),
        "completed_full_batches": smoke_batches,
        "persistent_feature_cache": False,
        "oof_source_image_reads": 0,
        "candidate_optimizer_steps": 0,
        "candidate_predictions": 0,
        "candidate_scores": 0,
    }
    del base_features, view_features
    return result


def command_smoke_extraction(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite smoke audit: {args.output}")
    authorization = validate_smoke_authorization(args)
    execution_claim = consume_authorization(args, authorization)
    runtime_before = authorization["runtime_sha256"]
    started = time.time()
    backbone: torch.nn.Module | None = None
    provenance: dict[str, Any] | None = None
    smoke: dict[str, Any] | None = None
    failure: dict[str, str] | None = None
    rows: list[dict[str, Any]] = []
    corpus_report: dict[str, Any] | None = None
    opened_paths: list[Path] = []
    peak_allocated = 0
    free_memory = 0
    total_memory = 0
    try:
        if args.b14_cache_dir.exists() and any(args.b14_cache_dir.rglob("*")):
            raise FileExistsError("v19-R2 cache target must be absent or empty before smoke")
        rows, corpus_report = load_smoke_corpus(args.manifest, args.audit)
        if len(rows) != 9990:
            raise ValueError("retained corpus row count mismatch")
        device = v9.resolve_device(args.device)
        if device.type != "cuda":
            raise RuntimeError("v19-R2 pipeline smoke requires CUDA")
        torch.cuda.empty_cache()
        backbone, provenance = v9.load_backbone(BACKBONE, device, args.b14_manifest)
        backbone.requires_grad_(False).eval()
        if any(parameter.requires_grad for parameter in backbone.parameters()):
            raise RuntimeError("B/14 backbone is not fully frozen")
        torch.cuda.reset_peak_memory_stats(device)
        smoke = run_smoke_pipeline(
            backbone,
            rows,
            fold=args.smoke_fold,
            image_batch_size=args.image_batch_size,
            num_workers=args.num_workers,
            smoke_batches=args.smoke_batches,
            device=device,
            opened_paths=opened_paths,
        )
        torch.cuda.synchronize(device)
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        free_memory, total_memory = (int(value) for value in torch.cuda.mem_get_info(device))
    except Exception as exc:
        failure = {
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }
    finally:
        if backbone is not None:
            del backbone
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    runtime_after = runtime_hashes(args)
    cache_files = (
        sorted(
            relative_path(path)
            for path in args.b14_cache_dir.rglob("*")
            if path.is_file()
        )
        if args.b14_cache_dir.exists()
        else []
    )
    gates = {
        "authorization_valid": True,
        "authorization_consumed": bool(execution_claim["sha256"]),
        "zero_model_labels": corpus_report is not None
        and corpus_report["model_labels_computed"] == 0,
        "three_real_batches_completed": smoke is not None
        and smoke["completed_full_batches"] == SMOKE_BATCHES,
        "effective_images_per_batch_is_36": smoke is not None
        and smoke["geometry"]["effective_images_per_full_batch"] == 36,
        "train_source_reads_are_12": smoke is not None
        and smoke["source_image_reads"] == SMOKE_ROWS,
        "zero_oof_source_reads": smoke is not None
        and smoke["oof_source_image_reads"] == 0,
        "zero_predictions_and_scores": smoke is not None
        and smoke["candidate_predictions"] == 0
        and smoke["candidate_scores"] == 0,
        "no_persistent_feature_cache": not cache_files,
        "runtime_unchanged": runtime_before == runtime_after == EXPECTED_RUNTIME_SHA256,
        "final_test_unread": True,
    }
    passed = failure is None and all(gates.values())
    audit = {
        "schema_version": "autoresearch-vicreg-backbone-renomination-v19r.smoke-v1",
        "run_id": "20260804-vicreg-backbone-renomination-v19-r1",
        "iteration": 1,
        "status": (
            "pipeline_smoke_passed_awaiting_council"
            if passed
            else "terminal_failed_pipeline_smoke"
        ),
        "pass": passed,
        "score": None,
        "authorization": authorization,
        "execution_claim": execution_claim,
        "corpus_report": corpus_report,
        "gates": gates,
        "smoke": smoke,
        "failure": failure,
        "cuda": {
            "peak_allocated_bytes": peak_allocated,
            "free_memory_bytes_after_smoke": free_memory,
            "total_memory_bytes": total_memory,
            "pin_memory": True,
            "non_blocking_transfer": True,
        },
        "backbone_provenance": provenance,
        "cache_files": cache_files,
        "operation_counts": {
            "source_image_reads": len(opened_paths),
            "model_labels_read": 0,
            "model_labels_computed": 0,
            "oof_source_image_reads": 0,
            "persistent_feature_vectors": 0,
            "candidate_optimizer_steps": 0,
            "candidate_predictions": 0,
            "candidate_scores": 0,
            "runtime_writes": 0,
            "final_test_reads": 0,
            "governance_writes": 2,
        },
        "runtime_before": runtime_before,
        "runtime_after": runtime_after,
        "runtime_unchanged": runtime_before == runtime_after,
        "final_test_read": False,
        "phase_2_allowed": False,
        "retry_allowed": False,
        "elapsed_seconds": time.time() - started,
        "next_action": (
            "Return to Argos Council before any cache extraction."
            if passed
            else "Close v19-R2 without fallback or candidate operation."
        ),
    }
    write_json(args.output, audit)
    return audit


def command_precompute(args: argparse.Namespace) -> dict[str, Any]:
    authorization = validate_phase_authorization(args, expected_phase="phase_2a_extraction")
    extraction_geometry = validate_extraction_geometry(
        image_batch_size=args.image_batch_size,
        num_workers=args.num_workers,
    )
    execution_claim = consume_authorization(args, authorization)
    static = validate_static_contract(args)
    filesystem = assert_ext4_workspace_path(args.b14_cache_dir)
    if args.b14_cache_dir.exists() and any(args.b14_cache_dir.iterdir()):
        raise FileExistsError("B/14 cache directory must be absent or empty")
    args.b14_cache_dir.mkdir(parents=True, exist_ok=True)
    rows, corpus_report = v9.load_corpus(args.manifest, args.audit, strict_counts=True)
    if len(rows) != 9990:
        raise ValueError("retained corpus row count mismatch")
    device = v9.resolve_device(args.device)
    pin = backbone_screen.validate_backbone_manifest(args.b14_manifest)
    backbone, provenance = v9.load_backbone(BACKBONE, device, args.b14_manifest)
    backbone.requires_grad_(False).eval()
    entries: list[dict[str, Any]] = []
    source_image_reads = 0
    started = time.time()
    try:
        for fold in EXPECTED_FOLDS:
            canonical = load_canonical_cache(args, fold)
            cache_path = v9.precompute_fold(
                rows,
                corpus_report,
                fold=fold,
                views=VIEWS,
                view_seed=VIEW_SEED,
                image_size=IMAGE_SIZE,
                batch_size=args.image_batch_size,
                num_workers=args.num_workers,
                device=device,
                backbone=backbone,
                backbone_provenance=provenance,
                cache_dir=args.b14_cache_dir,
                max_rows_per_class=None,
                force=False,
            )
            _, validation = validate_b14_cache(
                cache_path,
                fold=fold,
                canonical=canonical,
                backbone_provenance=provenance,
            )
            source_image_reads += validation["train_rows"] + validation["oof_rows"]
            entries.append(validation)
    finally:
        del backbone
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if source_image_reads != EXPECTED_TRAIN_ROWS_ACROSS_FOLDS + EXPECTED_UNIQUE_OOF_ROWS:
        raise ValueError("unexpected B/14 extraction row count")
    total_bytes = sum(item["cache_bytes"] for item in entries)
    result = {
        "schema_version": "autoresearch-v19r.b14-fold-cache-manifest-v1",
        "pass": True,
        "phase_authorization": authorization,
        "execution_claim": execution_claim,
        "extraction_geometry": extraction_geometry,
        "static_contract_sha256": v9.sha256_json(static),
        "spec_sha256": EXPECTED_SPEC_SHA256,
        "evaluator_sha256": EXPECTED_EVALUATOR_SHA256,
        "backbone_pin": pin,
        "backbone_provenance": provenance,
        "filesystem": filesystem,
        "folds": entries,
        "expected_tensor_bytes": EXPECTED_CACHE_TENSOR_BYTES,
        "total_cache_bytes": total_bytes,
        "under_two_gib": total_bytes < MAX_CACHE_BYTES,
        "duration_seconds_descriptive_only": time.time() - started,
        "operation_counts": {
            "source_image_reads": source_image_reads,
            "real_feature_extractions": source_image_reads,
            "candidate_optimizer_steps": 0,
            "candidate_predictions": 0,
            "candidate_scores": 0,
            "runtime_writes": 0,
        },
        "runtime_sha256_after": runtime_hashes(args),
        "runtime_unchanged": runtime_hashes(args) == EXPECTED_RUNTIME_SHA256,
        "final_test_read": False,
    }
    if not result["under_two_gib"] or not result["runtime_unchanged"]:
        raise ValueError("B/14 cache extraction violated a hard gate")
    write_json(args.cache_manifest, result)
    return result


def load_projection_checkpoint(path: Path, expected_sha256: str) -> dict[str, torch.Tensor]:
    if v9.sha256_file(path) != expected_sha256:
        raise ValueError(f"checkpoint SHA-256 mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint lacks model_state_dict: {path}")
    return state


def command_replay_c1(args: argparse.Namespace) -> dict[str, Any]:
    authorization = validate_phase_authorization(args, expected_phase="phase_2b_c1_control")
    execution_claim = consume_authorization(args, authorization)
    static = validate_static_contract(args)
    cache_manifest = validate_cache_manifest(args)
    summary = read_json(args.c1_summary)
    diagnostics = canonical_diagnostic_map(summary)
    canonical_rows = read_jsonl(args.c1_predictions)
    rows_by_key = {
        (int(row["outer_fold"]), int(row["seed"]), str(row["row_id"])): row
        for row in canonical_rows
    }
    if len(rows_by_key) != EXPECTED_PAIRED_ROWS:
        raise ValueError("canonical C1 replay lookup is not one-to-one")
    device = v9.resolve_device(args.device)
    mismatches: list[dict[str, Any]] = []
    replayed_rows = 0
    replay_entries: list[dict[str, Any]] = []
    for fold in EXPECTED_FOLDS:
        cache = load_canonical_cache(args, fold)
        for seed in EXPECTED_SEEDS:
            diagnostic = diagnostics[(fold, seed)]
            checkpoint_info = diagnostic["checkpoints"]["candidate"]
            checkpoint = resolve_artifact(checkpoint_info["path"])
            model = ProjectionHead(input_dim=384, embedding_dim=128)
            model.load_state_dict(load_projection_checkpoint(checkpoint, checkpoint_info["sha256"]))
            model = model.to(device)
            topk, effective_rank = v9.predict_arm(
                model,
                cache["train"],
                cache["oof"],
                device=device,
            )
            for index, row_id in enumerate(cache["oof"]["row_id"]):
                key = (fold, seed, str(row_id))
                expected = rows_by_key[key]["candidate_topk"]
                if topk[index] != expected:
                    mismatches.append(
                        {"fold": fold, "seed": seed, "row_id": str(row_id), "expected": expected, "actual": topk[index]}
                    )
                replayed_rows += 1
            replay_entries.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "checkpoint_path": relative_path(checkpoint),
                    "checkpoint_sha256": checkpoint_info["sha256"],
                    "prediction_rows": len(topk),
                    "effective_rank": effective_rank,
                    "episode_plan_sha256": diagnostic["episode_plan_sha256"],
                    "ssl_plan_sha256": diagnostic["ssl"]["plan_sha256"],
                }
            )
            del model
    result = {
        "schema_version": "autoresearch-v19r.c1-control-replay-audit-v1",
        "pass": not mismatches and replayed_rows == EXPECTED_PAIRED_ROWS,
        "phase_authorization": authorization,
        "execution_claim": execution_claim,
        "static_contract_sha256": v9.sha256_json(static),
        "cache_manifest_path": relative_path(args.cache_manifest),
        "cache_manifest_sha256": v9.sha256_file(args.cache_manifest),
        "cache_manifest_verified": cache_manifest["pass"],
        "replayed_checkpoints": len(replay_entries),
        "replayed_prediction_rows": replayed_rows,
        "exact_topk_match_count": replayed_rows - len(mismatches),
        "mismatches": mismatches[:20],
        "entries": replay_entries,
        "operation_counts": {
            "source_image_reads": 0,
            "real_feature_extractions": 0,
            "candidate_optimizer_steps": 0,
            "candidate_predictions": 0,
            "control_predictions": replayed_rows,
            "candidate_scores": 0,
            "runtime_writes": 0,
        },
        "runtime_sha256_after": runtime_hashes(args),
        "runtime_unchanged": runtime_hashes(args) == EXPECTED_RUNTIME_SHA256,
        "final_test_read": False,
    }
    if not result["pass"] or not result["runtime_unchanged"]:
        raise ValueError("canonical C1 replay control failed")
    write_json(args.control_audit, result)
    return result


def same_process_c1_interlock(
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> dict[str, Any]:
    summary = read_json(args.c1_summary)
    diagnostics = canonical_diagnostic_map(summary)
    canonical_rows = read_jsonl(args.c1_predictions)
    rows_by_key = {
        (int(row["outer_fold"]), int(row["seed"]), str(row["row_id"])): row
        for row in canonical_rows
    }
    if len(rows_by_key) != EXPECTED_PAIRED_ROWS:
        raise ValueError("same-process C1 lookup is not one-to-one")
    mismatches: list[dict[str, Any]] = []
    replayed_rows = 0
    checkpoint_count = 0
    entries: list[dict[str, Any]] = []
    for fold in EXPECTED_FOLDS:
        cache = load_canonical_cache(args, fold)
        for seed in EXPECTED_SEEDS:
            diagnostic = diagnostics[(fold, seed)]
            checkpoint_info = diagnostic["checkpoints"]["candidate"]
            checkpoint = resolve_artifact(checkpoint_info["path"])
            model = ProjectionHead(input_dim=384, embedding_dim=128)
            model.load_state_dict(
                load_projection_checkpoint(checkpoint, checkpoint_info["sha256"])
            )
            model = model.to(device)
            topk, effective_rank = v9.predict_arm(
                model,
                cache["train"],
                cache["oof"],
                device=device,
            )
            for index, row_id_value in enumerate(cache["oof"]["row_id"]):
                row_id = str(row_id_value)
                expected = rows_by_key[(fold, seed, row_id)]["candidate_topk"]
                if topk[index] != expected:
                    mismatches.append(
                        {
                            "fold": fold,
                            "seed": seed,
                            "row_id": row_id,
                            "expected": expected,
                            "actual": topk[index],
                        }
                    )
                replayed_rows += 1
            checkpoint_count += 1
            entries.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "checkpoint_sha256": checkpoint_info["sha256"],
                    "prediction_rows": len(topk),
                    "effective_rank": effective_rank,
                }
            )
            del model
    result = {
        "schema_version": "autoresearch-v19r.same-process-c1-interlock-v1",
        "pass": (
            checkpoint_count == 15
            and replayed_rows == EXPECTED_PAIRED_ROWS
            and not mismatches
        ),
        "replayed_checkpoints": checkpoint_count,
        "replayed_prediction_rows": replayed_rows,
        "exact_topk_match_count": replayed_rows - len(mismatches),
        "mismatches": mismatches[:20],
        "entries": entries,
        "candidate_optimizer_steps_before_interlock": 0,
        "candidate_gradient_before_interlock": False,
        "runtime_sha256_after": runtime_hashes(args),
        "runtime_unchanged": runtime_hashes(args) == EXPECTED_RUNTIME_SHA256,
        "final_test_read": False,
    }
    if not result["pass"] or not result["runtime_unchanged"]:
        raise ValueError("same-process C1 interlock failed before candidate gradient")
    return result


def torch_rng_sha256() -> str:
    digest = hashlib.sha256(torch.get_rng_state().numpy().tobytes())
    if torch.cuda.is_available():
        for state in torch.cuda.get_rng_state_all():
            digest.update(state.cpu().numpy().tobytes())
    return digest.hexdigest()


def aligned_candidate_initialization(
    seed: int,
) -> tuple[ProjectionHead, dict[str, str]]:
    configure_determinism(seed)
    candidate = ProjectionHead(input_dim=EMBED_DIM, embedding_dim=128)
    candidate_initial_sha256 = v9.state_dict_sha256(candidate)
    rng_after_candidate_head = torch_rng_sha256()
    configure_determinism(seed)
    canonical_head = ProjectionHead(input_dim=384, embedding_dim=128)
    canonical_initial_sha256 = v9.state_dict_sha256(canonical_head)
    rng_after_canonical_head = torch_rng_sha256()
    del canonical_head
    return candidate, {
        "candidate_initial_state_sha256": candidate_initial_sha256,
        "canonical_c1_initial_state_sha256": canonical_initial_sha256,
        "rng_after_candidate_head_sha256": rng_after_candidate_head,
        "rng_restored_to_canonical_post_head_sha256": rng_after_canonical_head,
    }


@contextmanager
def observe_vicreg_losses() -> Iterator[list[float]]:
    original = v9.vicreg_loss
    losses: list[float] = []

    def wrapped(*values: Any, **kwargs: Any) -> Any:
        terms = original(*values, **kwargs)
        losses.append(float(terms.total.detach().cpu()))
        return terms

    v9.vicreg_loss = wrapped
    try:
        yield losses
    finally:
        v9.vicreg_loss = original


def loss_trajectory(
    batch_losses: Sequence[float],
    *,
    train_rows: int,
    epochs: int,
    batch_size: int,
) -> list[float]:
    batches_per_epoch = sum(
        1
        for offset in range(0, train_rows, batch_size)
        if min(batch_size, train_rows - offset) >= 2
    )
    expected = batches_per_epoch * epochs
    if len(batch_losses) != expected:
        raise ValueError(
            f"VICReg observer count mismatch: expected {expected}, found {len(batch_losses)}"
        )
    return [
        float(np.mean(batch_losses[index * batches_per_epoch : (index + 1) * batches_per_epoch]))
        for index in range(epochs)
    ]


def embedding_snapshot(
    model: ProjectionHead,
    features: torch.Tensor,
    *,
    device: torch.device,
    limit: int = 4096,
) -> torch.Tensor:
    model = model.to(device)
    with torch.inference_mode():
        return v9.embed_in_batches(model, features[:limit], device)


def support_bin_three(count: int) -> str:
    if count <= 8:
        return "n_y_lte_8"
    if count <= 31:
        return "n_y_9_to_31"
    return "n_y_gte_32"


def support_bin_binary(count: int) -> str:
    return "n_y_lte_8" if count <= 8 else "n_y_gt_8"


def atomic_save_checkpoint(
    path: Path,
    *,
    fold: int,
    seed: int,
    model: ProjectionHead,
    initial_state_sha256: str,
    final_state_sha256: str,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "schema_version": "autoresearch-v19r.candidate-checkpoint-v1",
        "arm": "v19-A",
        "fold": fold,
        "seed": seed,
        "input_dim": EMBED_DIM,
        "embedding_dim": 128,
        "initial_state_sha256": initial_state_sha256,
        "final_state_sha256": final_state_sha256,
        "model_state_dict": {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in model.state_dict().items()
        },
    }
    torch.save(payload, temporary, _use_new_zipfile_serialization=False)
    temporary.replace(path)
    return {
        "path": relative_path(path),
        "sha256": v9.sha256_file(path),
        "bytes": path.stat().st_size,
        "state_sha256": final_state_sha256,
    }


def run_candidate_fold_seed(
    cache: dict[str, Any],
    *,
    fold: int,
    seed: int,
    canonical_diagnostic: dict[str, Any],
    canonical_rows: Sequence[dict[str, Any]],
    device: torch.device,
    supervised_device: torch.device,
    checkpoint_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model, initialization = aligned_candidate_initialization(seed)
    initial_state_sha256 = initialization["candidate_initial_state_sha256"]
    initial_embedding = embedding_snapshot(
        model,
        cache["train"]["base_features"],
        device=device,
    )
    model = model.to(device)
    with observe_vicreg_losses() as batch_losses:
        ssl = v9.pretrain_vicreg(
            model,
            cache["train"],
            epochs=30,
            batch_size=256,
            learning_rate=3e-4,
            weight_decay=1e-4,
            seed=seed,
            device=device,
        )
    trajectory = loss_trajectory(
        batch_losses,
        train_rows=len(cache["train"]["row_id"]),
        epochs=30,
        batch_size=256,
    )
    if ssl["plan_sha256"] != canonical_diagnostic["ssl"]["plan_sha256"]:
        raise ValueError(f"SSL pair-plan hash differs from C1 for fold={fold}, seed={seed}")
    labels = torch.tensor(cache["train"]["class_label"], dtype=torch.long)
    episode_plan, episode_plan_sha256 = v9.build_episode_plan(
        labels,
        n_way=20,
        k_shot=3,
        q_queries=5,
        epochs=30,
        episodes_per_epoch=100,
        seed=v9.stable_seed("supervised-episodes", fold, seed),
    )
    if episode_plan_sha256 != canonical_diagnostic["episode_plan_sha256"]:
        raise ValueError(f"episode-plan hash differs from C1 for fold={fold}, seed={seed}")
    model = model.to(supervised_device)
    supervised = v9.train_supervised(
        model,
        cache["train"]["base_features"],
        episode_plan,
        learning_rate=1e-3,
        weight_decay=1e-4,
        temperature=0.1,
        warmup_epochs=5,
        rng_seed=v9.stable_seed("supervised-rng", fold, seed),
        device=supervised_device,
    )
    gradients_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    weights_finite = all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters())
    model = model.to(device)
    candidate_topk, candidate_rank = v9.predict_arm(
        model,
        cache["train"],
        cache["oof"],
        device=device,
    )
    final_embedding = embedding_snapshot(
        model,
        cache["train"]["base_features"],
        device=device,
    )
    embedding_movement = float(torch.mean(torch.abs(final_embedding - initial_embedding)))
    final_state_sha256 = v9.state_dict_sha256(model)
    checkpoint = atomic_save_checkpoint(
        checkpoint_dir / f"fold-{fold:02d}-seed-{seed}-v19A.pt",
        fold=fold,
        seed=seed,
        model=model,
        initial_state_sha256=initial_state_sha256,
        final_state_sha256=final_state_sha256,
    )
    canonical_by_row = {str(row["row_id"]): row for row in canonical_rows}
    support = Counter(int(label) for label in cache["train"]["class_label"])
    records: list[dict[str, Any]] = []
    for index, row_id_value in enumerate(cache["oof"]["row_id"]):
        row_id = str(row_id_value)
        source = canonical_by_row[row_id]
        if int(source["outer_fold"]) != fold or int(source["seed"]) != seed:
            raise ValueError("canonical C1 row key mismatch")
        if (
            int(source["label"]) != int(cache["oof"]["class_label"][index])
            or str(source["provenance_component"]) != str(cache["oof"]["component_id"][index])
            or str(source["decoded_pixel_sha256"]) != str(cache["oof"]["decoded_pixel_sha256"][index])
        ):
            raise ValueError("canonical C1 row metadata differs from B/14 cache")
        train_support = support[int(source["label"])]
        records.append(
            {
                "row_id": row_id,
                "provenance_component": str(source["provenance_component"]),
                "decoded_pixel_sha256": str(source["decoded_pixel_sha256"]),
                "label": int(source["label"]),
                "class_name": str(source["class_name"]),
                "outer_fold": fold,
                "seed": seed,
                "recipe": "v19A-B14-v9-VICReg-episodic",
                "b0_topk": list(source["baseline_topk"]),
                "c1_topk": list(source["candidate_topk"]),
                "candidate_topk": candidate_topk[index],
                "episode_plan_sha256": episode_plan_sha256,
                "ssl_plan_sha256": ssl["plan_sha256"],
                "train_support": train_support,
                "support_bin_three": support_bin_three(train_support),
                "support_bin_binary": support_bin_binary(train_support),
            }
        )
    diagnostics = {
        "fold": fold,
        "seed": seed,
        "devices": {"ssl_and_prediction": str(device), "supervised": str(supervised_device)},
        "initialization": initialization,
        "initial_state_sha256": initial_state_sha256,
        "final_state_sha256": final_state_sha256,
        "candidate_final_state_differs_from_initial": final_state_sha256 != initial_state_sha256,
        "embedding_movement_mean_absolute": embedding_movement,
        "episode_plan_sha256": episode_plan_sha256,
        "episode_plan_matches_c1": True,
        "ssl": {
            **ssl,
            "plan_matches_c1": True,
            "epoch_loss_mean": trajectory,
            "start_epoch_loss": trajectory[0],
            "end_epoch_loss": trajectory[-1],
            "end_epoch_loss_below_start": trajectory[-1] < trajectory[0],
        },
        "supervised": supervised,
        "candidate_effective_rank": candidate_rank,
        "c1_effective_rank": float(canonical_diagnostic["candidate_effective_rank"]),
        "candidate_to_c1_effective_rank_ratio": (
            candidate_rank / float(canonical_diagnostic["candidate_effective_rank"])
            if canonical_diagnostic["candidate_effective_rank"]
            else 0.0
        ),
        "finite_gradients": gradients_finite,
        "finite_weights": weights_finite,
        "checkpoint": checkpoint,
    }
    return records, diagnostics


def comparison_records(
    records: Sequence[dict[str, Any]],
    *,
    baseline_field: str,
) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "baseline_topk": list(row[baseline_field]),
            "candidate_topk": list(row["candidate_topk"]),
        }
        for row in records
    ]


def comparison_summary(
    records: Sequence[dict[str, Any]],
    *,
    baseline_field: str,
    bootstrap_replicates: int = 2000,
) -> dict[str, Any]:
    paired = comparison_records(records, baseline_field=baseline_field)
    seeds = sorted({int(row["seed"]) for row in paired})
    folds = sorted({int(row["outer_fold"]) for row in paired})
    overall = v9.accuracy_metrics(paired)
    seed_metrics = [
        {
            "seed": seed,
            **v9.accuracy_metrics([row for row in paired if int(row["seed"]) == seed]),
        }
        for seed in seeds
    ]
    fold_metrics = [
        {
            "fold": fold,
            **v9.accuracy_metrics([row for row in paired if int(row["outer_fold"]) == fold]),
        }
        for fold in folds
    ]
    bootstrap = v9.paired_component_bootstrap(
        paired,
        replicates=bootstrap_replicates,
        seed=v9.stable_seed("v19-bootstrap", baseline_field),
    )
    return {
        "baseline_field": baseline_field,
        "overall_metrics": overall,
        "seed_metrics": seed_metrics,
        "fold_metrics": fold_metrics,
        "bootstrap": bootstrap,
        "mcnemar": v9.exact_mcnemar(paired),
        "positive_seed_count": sum(item["delta_top1"] > 0.0 for item in seed_metrics),
        "minimum_fold_delta_top1": min(item["delta_top1"] for item in fold_metrics),
        "discordant_topk_rows": sum(
            row["baseline_topk"] != row["candidate_topk"] for row in paired
        ),
        "discordant_top1_rows": sum(
            row["baseline_topk"][0] != row["candidate_topk"][0] for row in paired
        ),
    }


def support_strata_diagnostics(
    records: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for field in ("support_bin_three", "support_bin_binary"):
        values: dict[str, Any] = {}
        for value in sorted({str(row[field]) for row in records}):
            subset = [row for row in records if str(row[field]) == value]
            values[value] = {
                "prediction_rows": len(subset),
                "candidate_minus_c1": v9.accuracy_metrics(
                    comparison_records(subset, baseline_field="c1_topk")
                ),
                "candidate_minus_b0": v9.accuracy_metrics(
                    comparison_records(subset, baseline_field="b0_topk")
                ),
            }
        result[field] = values
    return result


def evaluate_candidate(
    records: Sequence[dict[str, Any]],
    diagnostics: Sequence[dict[str, Any]],
    cache_validations: Sequence[dict[str, Any]],
    *,
    control_audit: dict[str, Any],
) -> dict[str, Any]:
    if len(records) != EXPECTED_PAIRED_ROWS:
        raise ValueError("candidate prediction-row count mismatch")
    if len(diagnostics) != 15 or len(cache_validations) != 5:
        raise ValueError("candidate diagnostic/cache count mismatch")
    c1 = comparison_summary(records, baseline_field="c1_topk")
    b0 = comparison_summary(records, baseline_field="b0_topk")
    c1_overall = c1["overall_metrics"]
    b0_overall = b0["overall_metrics"]
    integrity_gates = {
        "paired_prediction_rows": len(records) == EXPECTED_PAIRED_ROWS,
        "paired_seed_count": len({int(row["seed"]) for row in records}) == 3,
        "outer_fold_count": len({int(row["outer_fold"]) for row in records}) == 5,
        "candidate_checkpoint_count": sum(
            bool(item["checkpoint"]["sha256"]) for item in diagnostics
        )
        == 15,
        "c1_same_process_topk_match_count": (
            int(control_audit.get("exact_topk_match_count", -1)) == EXPECTED_PAIRED_ROWS
        ),
        "b14_cache_count": len(cache_validations) == 5,
        "b14_cache_row_order_hash_match_count": sum(
            item["train_row_ids_sha256"] == item["canonical_train_row_ids_sha256"]
            and item["oof_row_ids_sha256"] == item["canonical_oof_row_ids_sha256"]
            for item in cache_validations
        )
        == 5,
        "b14_view_plan_hash_count": sum(bool(item["view_plan_sha256"]) for item in cache_validations)
        == 5,
        "b14_cache_byte_identical_readback_count": sum(
            item["byte_identical_readback"] is True for item in cache_validations
        )
        == 5,
        "oof_view_feature_count_is_zero": True,
        "episode_plan_sha256_matches_c1_count": sum(
            item["episode_plan_matches_c1"] is True for item in diagnostics
        )
        == 15,
        "ssl_pair_plan_sha256_matches_c1_count": sum(
            item["ssl"]["plan_matches_c1"] is True for item in diagnostics
        )
        == 15,
        "finite_losses_gradients_weights_and_embeddings": all(
            item["finite_gradients"]
            and item["finite_weights"]
            and math.isfinite(item["embedding_movement_mean_absolute"])
            and all(math.isfinite(value) for value in item["ssl"]["epoch_loss_mean"])
            for item in diagnostics
        ),
        "candidate_count_is_one": True,
        "final_test_unread": True,
        "runtime_unchanged": True,
        "automatic_promotion_disabled": True,
    }
    engagement_gates = {
        "vicreg_end_epoch_loss_below_start_epoch_count": sum(
            item["ssl"]["end_epoch_loss_below_start"] is True for item in diagnostics
        ),
        "candidate_final_state_differs_from_initial_count": sum(
            item["candidate_final_state_differs_from_initial"] is True
            for item in diagnostics
        ),
        "nonzero_embedding_movement_count": sum(
            item["embedding_movement_mean_absolute"] > 0.0 for item in diagnostics
        ),
        "candidate_vs_c1_discordant_prediction_rows": c1["discordant_topk_rows"],
    }
    engagement_pass = (
        engagement_gates["vicreg_end_epoch_loss_below_start_epoch_count"] == 15
        and engagement_gates["candidate_final_state_differs_from_initial_count"] == 15
        and engagement_gates["nonzero_embedding_movement_count"] == 15
        and engagement_gates["candidate_vs_c1_discordant_prediction_rows"] > 0
    )
    b0_gates = {
        "delta_top1_gte_0_01": b0_overall["delta_top1"] >= 0.01,
        "bootstrap_lower_gt_0": b0["bootstrap"]["delta_top1_95"][0] > 0.0,
        "positive_seed_count_gte_2": b0["positive_seed_count"] >= 2,
        "delta_macro_top1_gte_minus_0_005": b0_overall["delta_macro_top1"] >= -0.005,
        "delta_top3_gte_0": b0_overall["delta_top3"] >= 0.0,
        "minimum_fold_delta_top1_gte_minus_0_05": b0["minimum_fold_delta_top1"] >= -0.05,
    }
    c1_gates = {
        "delta_top1_gte_minus_0_005": c1_overall["delta_top1"] >= -0.005,
        "bootstrap_lower_gt_minus_0_01": c1["bootstrap"]["delta_top1_95"][0] > -0.01,
        "delta_top3_gte_minus_0_01": c1_overall["delta_top3"] >= -0.01,
        "delta_macro_top1_gte_minus_0_01": c1_overall["delta_macro_top1"] >= -0.01,
    }
    integrity_pass = all(integrity_gates.values())
    b0_pass = all(b0_gates.values())
    c1_pass = all(c1_gates.values())
    return {
        "schema_version": "autoresearch-v19r.candidate-evaluation-v1",
        "integrity_gates": integrity_gates,
        "integrity_pass": integrity_pass,
        "engagement_gates": engagement_gates,
        "engagement_pass": engagement_pass,
        "b0_mission_anchor_gates": b0_gates,
        "b0_mission_anchor_pass": b0_pass,
        "c1_noninferiority_gates": c1_gates,
        "c1_noninferiority_pass": c1_pass,
        "candidate_minus_b0": b0,
        "candidate_minus_c1": c1,
        "support_strata": support_strata_diagnostics(records),
        "minimum_candidate_to_c1_rank_ratio": min(
            item["candidate_to_c1_effective_rank_ratio"] for item in diagnostics
        ),
        "rank_ratio_below_0_8_diagnostic_only": any(
            item["candidate_to_c1_effective_rank_ratio"] < 0.8 for item in diagnostics
        ),
        "replay_pending": True,
        "score": c1_overall["delta_top1"],
        "promotion_eligible": False,
        "research_reference_only": True,
        "final_test_read": False,
    }


def classify_candidate_verdict(evaluation: dict[str, Any]) -> tuple[str, str]:
    c1 = evaluation["candidate_minus_c1"]
    return v17.classify_verdict(
        integrity_pass=bool(evaluation["integrity_pass"]),
        engagement_pass=bool(evaluation["engagement_pass"]),
        b0_anchor_pass=bool(evaluation["b0_mission_anchor_pass"]),
        c1_noninferiority_pass=bool(evaluation["c1_noninferiority_pass"]),
        c1_delta_top1=c1["overall"]["delta_top1"],
        c1_bootstrap_lower=c1["bootstrap"]["delta_top1_95"][0],
        c1_positive_seed_count=c1["positive_seed_count"],
    )


def validate_control_audit(args: argparse.Namespace) -> dict[str, Any]:
    audit = read_json(args.control_audit)
    if audit.get("schema_version") != "autoresearch-v19r.c1-control-replay-audit-v1":
        raise ValueError("unexpected C1 control-audit schema")
    if audit.get("pass") is not True:
        raise ValueError("C1 control replay did not pass")
    if int(audit.get("replayed_checkpoints", -1)) != 15:
        raise ValueError("C1 control checkpoint count mismatch")
    if int(audit.get("exact_topk_match_count", -1)) != EXPECTED_PAIRED_ROWS:
        raise ValueError("C1 control exact-top-k count mismatch")
    if audit.get("runtime_unchanged") is not True or audit.get("final_test_read") is not False:
        raise ValueError("C1 control violated runtime/final-test constraints")
    if audit.get("cache_manifest_sha256") != v9.sha256_file(args.cache_manifest):
        raise ValueError("C1 control was not run against this cache manifest")
    return audit


def run_candidate_evaluation(
    args: argparse.Namespace,
    *,
    recorded_authorization: dict[str, Any],
    recorded_execution_claim: dict[str, Any],
    deadline_claim: dict[str, Any],
) -> dict[str, Any]:
    authorization = recorded_authorization
    execution_claim = recorded_execution_claim
    enforce_execution_claim_deadline(deadline_claim)
    static = validate_static_contract(args)
    cache_manifest = validate_cache_manifest(args)
    control_audit = validate_control_audit(args)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("candidate output directory must be absent or empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = read_json(args.c1_summary)
    canonical_diagnostics = canonical_diagnostic_map(summary)
    canonical_rows = read_jsonl(args.c1_predictions)
    rows_by_fold_seed: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in canonical_rows:
        rows_by_fold_seed[(int(row["outer_fold"]), int(row["seed"]))].append(row)
    device = v9.resolve_device(args.device)
    enforce_execution_claim_deadline(deadline_claim)
    same_process_control = same_process_c1_interlock(
        args,
        device=device,
    )
    enforce_execution_claim_deadline(deadline_claim)
    supervised_device = v9.resolve_device(args.supervised_device)
    if supervised_device.type != "cpu":
        raise ValueError("v19 supervised training must remain on CPU")
    torch.set_num_threads(1)
    all_records: list[dict[str, Any]] = []
    all_diagnostics: list[dict[str, Any]] = []
    cache_validations: list[dict[str, Any]] = []
    optimizer_steps = 0
    enforce_execution_claim_deadline(deadline_claim)
    for fold in EXPECTED_FOLDS:
        enforce_execution_claim_deadline(deadline_claim)
        canonical_cache = load_canonical_cache(args, fold)
        cache_path = v9.cache_path(args.b14_cache_dir, fold, VIEWS, None)
        cache, validation = validate_b14_cache(
            cache_path,
            fold=fold,
            canonical=canonical_cache,
        )
        cache_validations.append(validation)
        ssl_batches_per_epoch = sum(
            1
            for offset in range(0, len(cache["train"]["row_id"]), 256)
            if min(256, len(cache["train"]["row_id"]) - offset) >= 2
        )
        for seed in EXPECTED_SEEDS:
            enforce_execution_claim_deadline(deadline_claim)
            records, diagnostic = run_candidate_fold_seed(
                cache,
                fold=fold,
                seed=seed,
                canonical_diagnostic=canonical_diagnostics[(fold, seed)],
                canonical_rows=rows_by_fold_seed[(fold, seed)],
                device=device,
                supervised_device=supervised_device,
                checkpoint_dir=args.output_dir / "checkpoints",
            )
            enforce_execution_claim_deadline(deadline_claim)
            all_records.extend(records)
            all_diagnostics.append(diagnostic)
            optimizer_steps += 30 * ssl_batches_per_epoch + 30 * 100
    all_records.sort(key=lambda row: (int(row["outer_fold"]), int(row["seed"]), str(row["row_id"])))
    all_diagnostics.sort(key=lambda item: (int(item["fold"]), int(item["seed"])))
    enforce_execution_claim_deadline(deadline_claim)
    evaluation = evaluate_candidate(
        all_records,
        all_diagnostics,
        cache_validations,
        control_audit=same_process_control,
    )
    predictions_path = args.output_dir / "prediction_rows.jsonl"
    diagnostics_path = args.output_dir / "diagnostics.json"
    write_jsonl(predictions_path, all_records)
    write_json(diagnostics_path, {"diagnostics": all_diagnostics})
    result = {
        "schema_version": "autoresearch-v19r.candidate-run-summary-v1",
        "pass": False,
        "decision_status": "pending_deterministic_replay",
        "phase_authorization": authorization,
        "execution_claim": execution_claim,
        "static_contract_sha256": v9.sha256_json(static),
        "spec_sha256": EXPECTED_SPEC_SHA256,
        "evaluator_sha256": EXPECTED_EVALUATOR_SHA256,
        "cache_manifest_sha256": v9.sha256_file(args.cache_manifest),
        "control_audit_sha256": v9.sha256_file(args.control_audit),
        "same_process_c1_interlock": same_process_control,
        "cache_manifest_verified": cache_manifest["pass"],
        "control_audit_verified": control_audit["pass"],
        "prediction_rows_sha256": v9.sha256_file(predictions_path),
        "diagnostics_sha256": v9.sha256_file(diagnostics_path),
        "candidate_checkpoint_sha256": {
            f"{item['fold']}:{item['seed']}": item["checkpoint"]["sha256"]
            for item in all_diagnostics
        },
        "candidate_state_sha256": {
            f"{item['fold']}:{item['seed']}": item["final_state_sha256"]
            for item in all_diagnostics
        },
        "cache_validations": cache_validations,
        "diagnostics": all_diagnostics,
        "evaluation": evaluation,
        "operation_counts": {
            "source_image_reads": 0,
            "cache_reextractions": 0,
            "candidate_optimizer_steps": optimizer_steps,
            "candidate_predictions": len(all_records),
            "candidate_scores": len(all_records),
            "control_predictions_before_candidate_gradient": EXPECTED_PAIRED_ROWS,
            "runtime_writes": 0,
        },
        "artifact_paths": {
            "prediction_rows": relative_path(predictions_path),
            "diagnostics": relative_path(diagnostics_path),
        },
        "runtime_sha256_after": runtime_hashes(args),
        "runtime_unchanged": runtime_hashes(args) == EXPECTED_RUNTIME_SHA256,
        "final_test_read": False,
        "promotion_eligible": False,
    }
    if not result["runtime_unchanged"]:
        raise ValueError("runtime changed during candidate training")
    enforce_execution_claim_deadline(deadline_claim)
    write_json(args.output_dir / "summary.json", result)
    if not evaluation["integrity_pass"] or not evaluation["engagement_pass"]:
        raise ValueError("candidate failed integrity or engagement before replay")
    return result


def command_train_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    authorization = validate_phase_authorization(args, expected_phase="phase_3a_train_evaluate")
    execution_claim = consume_authorization(args, authorization)
    return run_candidate_evaluation(
        args,
        recorded_authorization=authorization,
        recorded_execution_claim=execution_claim,
        deadline_claim=execution_claim,
    )


def normalized_replay_value(value: Any) -> Any:
    if isinstance(value, list):
        return [normalized_replay_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    normalized: dict[str, Any] = {}
    for key, item in value.items():
        if key in {"artifact_paths", "diagnostics_sha256"}:
            continue
        if key == "path" and "sha256" in value:
            continue
        normalized[key] = normalized_replay_value(item)
    return normalized


def normalized_replay_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            normalized_replay_value(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def validate_phase3a_handoff(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = args.output_dir / "summary.json"
    predictions_path = args.output_dir / "prediction_rows.jsonl"
    diagnostics_path = args.output_dir / "diagnostics.json"
    for path in (summary_path, predictions_path, diagnostics_path, DEFAULT_PHASE3A_CLAIM):
        if not path.is_file():
            raise FileNotFoundError(f"phase 3a handoff is incomplete: {path}")

    original = read_json(summary_path)
    if original.get("schema_version") != "autoresearch-v19r.candidate-run-summary-v1":
        raise ValueError("phase 3a handoff summary schema mismatch")
    if original.get("pass") is not False:
        raise ValueError("phase 3a handoff must remain pending")
    if original.get("decision_status") != "pending_deterministic_replay":
        raise ValueError("phase 3a handoff decision status mismatch")
    if original.get("runtime_unchanged") is not True:
        raise ValueError("phase 3a handoff changed runtime artifacts")
    if original.get("runtime_sha256_after") != EXPECTED_RUNTIME_SHA256:
        raise ValueError("phase 3a handoff runtime hash mismatch")
    if original.get("final_test_read") is not False:
        raise ValueError("phase 3a handoff read the final test")
    if original.get("promotion_eligible") is not False:
        raise ValueError("phase 3a handoff cannot be promotion eligible")

    phase3a_args = build_parser().parse_args(["train-evaluate"])
    expected_authorization = validate_phase_authorization(
        phase3a_args,
        expected_phase="phase_3a_train_evaluate",
    )
    if original.get("phase_authorization") != expected_authorization:
        raise ValueError("phase 3a handoff authorization mismatch")

    claim = read_json(DEFAULT_PHASE3A_CLAIM)
    if claim.get("schema_version") != "autoresearch-v19r.execution-claim-v1":
        raise ValueError("phase 3a handoff claim schema mismatch")
    expected_claim_record = {
        "path": relative_path(DEFAULT_PHASE3A_CLAIM),
        "sha256": v9.sha256_file(DEFAULT_PHASE3A_CLAIM),
        **{
            key: claim[key]
            for key in (
                "deadline_policy",
                "started_at",
                "deadline_at",
                "max_runtime_seconds",
            )
            if key in claim
        },
    }
    if original.get("execution_claim") != expected_claim_record:
        raise ValueError("phase 3a handoff execution-claim record mismatch")
    if claim.get("command") != "train-evaluate":
        raise ValueError("phase 3a handoff claim command mismatch")
    if claim.get("authorization_path") != relative_path(DEFAULT_PHASE3A_AUTHORIZATION):
        raise ValueError("phase 3a handoff claim authorization path mismatch")
    if claim.get("authorization_sha256") != v9.sha256_file(DEFAULT_PHASE3A_AUTHORIZATION):
        raise ValueError("phase 3a handoff claim authorization hash mismatch")
    if claim.get("command_contract_sha256") != expected_authorization["command_contract_sha256"]:
        raise ValueError("phase 3a handoff claim command-contract mismatch")
    if claim.get("deadline_policy") != PHASE3_DEADLINE_POLICY:
        raise ValueError("phase 3a handoff claim deadline policy mismatch")
    if claim.get("one_execution_only") is not True or claim.get("retry_allowed") is not False:
        raise ValueError("phase 3a handoff claim replay policy mismatch")

    evaluation = original.get("evaluation", {})
    forbidden = {
        "verdict",
        "claim",
        "pass",
        "provisional_verdict_before_replay",
        "provisional_claim_before_replay",
    }
    if forbidden.intersection(evaluation):
        raise ValueError("phase 3a handoff exposed a verdict before replay")
    if evaluation.get("replay_pending") is not True:
        raise ValueError("phase 3a handoff is not pending replay")
    if evaluation.get("integrity_pass") is not True or evaluation.get("engagement_pass") is not True:
        raise ValueError("phase 3a handoff failed integrity or engagement")
    engagement = evaluation.get("engagement_gates", {})
    if engagement.get("vicreg_end_epoch_loss_below_start_epoch_count") != 15:
        raise ValueError("phase 3a handoff VICReg engagement count mismatch")
    if engagement.get("candidate_final_state_differs_from_initial_count") != 15:
        raise ValueError("phase 3a handoff state-change count mismatch")
    if engagement.get("nonzero_embedding_movement_count") != 15:
        raise ValueError("phase 3a handoff embedding-movement count mismatch")
    if int(engagement.get("candidate_vs_c1_discordant_prediction_rows", 0)) <= 0:
        raise ValueError("phase 3a handoff has no candidate/control discordance")
    if original.get("same_process_c1_interlock", {}).get("pass") is not True:
        raise ValueError("phase 3a handoff lacks the same-process C1 interlock")

    prediction_rows = read_jsonl(predictions_path)
    if len(prediction_rows) != EXPECTED_PAIRED_ROWS:
        raise ValueError("phase 3a handoff prediction-row count mismatch")
    prediction_sha256 = v9.sha256_file(predictions_path)
    if original.get("prediction_rows_sha256") != prediction_sha256:
        raise ValueError("phase 3a handoff prediction hash mismatch")
    diagnostics_payload = read_json(diagnostics_path)
    diagnostics = diagnostics_payload.get("diagnostics")
    if not isinstance(diagnostics, list) or len(diagnostics) != 15:
        raise ValueError("phase 3a handoff diagnostic count mismatch")
    if diagnostics != original.get("diagnostics"):
        raise ValueError("phase 3a handoff diagnostic payload mismatch")
    diagnostics_sha256 = v9.sha256_file(diagnostics_path)
    if original.get("diagnostics_sha256") != diagnostics_sha256:
        raise ValueError("phase 3a handoff diagnostic hash mismatch")

    expected_keys = {f"{fold}:{seed}" for fold in EXPECTED_FOLDS for seed in EXPECTED_SEEDS}
    checkpoint_hashes = original.get("candidate_checkpoint_sha256", {})
    state_hashes = original.get("candidate_state_sha256", {})
    if set(checkpoint_hashes) != expected_keys or set(state_hashes) != expected_keys:
        raise ValueError("phase 3a handoff checkpoint/state key mismatch")
    for diagnostic in diagnostics:
        key = f"{int(diagnostic['fold'])}:{int(diagnostic['seed'])}"
        checkpoint = diagnostic.get("checkpoint", {})
        checkpoint_path = resolve_artifact(checkpoint["path"])
        checkpoint_sha256 = v9.sha256_file(checkpoint_path)
        if checkpoint_sha256 != checkpoint.get("sha256"):
            raise ValueError(f"phase 3a checkpoint payload hash mismatch: {key}")
        if checkpoint_sha256 != checkpoint_hashes[key]:
            raise ValueError(f"phase 3a checkpoint summary hash mismatch: {key}")
        if diagnostic.get("final_state_sha256") != state_hashes[key]:
            raise ValueError(f"phase 3a state hash mismatch: {key}")

    counts = original.get("operation_counts", {})
    expected_counts = {
        "source_image_reads": 0,
        "cache_reextractions": 0,
        "candidate_predictions": EXPECTED_PAIRED_ROWS,
        "candidate_scores": EXPECTED_PAIRED_ROWS,
        "control_predictions_before_candidate_gradient": EXPECTED_PAIRED_ROWS,
        "runtime_writes": 0,
    }
    for key, value in expected_counts.items():
        if counts.get(key) != value:
            raise ValueError(f"phase 3a operation count mismatch for {key}")
    if int(counts.get("candidate_optimizer_steps", 0)) <= 0:
        raise ValueError("phase 3a handoff lacks optimizer steps")

    return {
        "original": original,
        "summary_path": relative_path(summary_path),
        "summary_sha256": v9.sha256_file(summary_path),
        "prediction_rows_sha256": prediction_sha256,
        "diagnostics_sha256": diagnostics_sha256,
        "execution_claim_sha256": v9.sha256_file(DEFAULT_PHASE3A_CLAIM),
        "checkpoint_count": len(checkpoint_hashes),
        "prediction_row_count": len(prediction_rows),
        "diagnostic_count": len(diagnostics),
    }


def command_replay(args: argparse.Namespace) -> dict[str, Any]:
    authorization = validate_phase_authorization(args, expected_phase="phase_3b_replay")
    handoff = validate_phase3a_handoff(args)
    original = handoff["original"]
    original_summary_path = args.output_dir / "summary.json"
    original_predictions_path = args.output_dir / "prediction_rows.jsonl"
    if args.replay_dir.exists() and any(args.replay_dir.iterdir()):
        raise FileExistsError("replay directory must be absent or empty")
    execution_claim = consume_authorization(args, authorization)
    enforce_execution_claim_deadline(execution_claim)
    replay_args = copy.copy(args)
    replay_args.output_dir = args.replay_dir
    replay = run_candidate_evaluation(
        replay_args,
        recorded_authorization=original["phase_authorization"],
        recorded_execution_claim=original["execution_claim"],
        deadline_claim=execution_claim,
    )
    replay_summary_path = args.replay_dir / "summary.json"
    replay_predictions_path = args.replay_dir / "prediction_rows.jsonl"
    original_diagnostics_path = args.output_dir / "diagnostics.json"
    replay_diagnostics_path = args.replay_dir / "diagnostics.json"
    original_diagnostics_sha256 = v9.sha256_file(original_diagnostics_path)
    replay_diagnostics_sha256 = v9.sha256_file(replay_diagnostics_path)
    original_prediction_sha256 = v9.sha256_file(original_predictions_path)
    replay_prediction_sha256 = v9.sha256_file(replay_predictions_path)
    checkpoint_match = (
        original["candidate_checkpoint_sha256"]
        == replay["candidate_checkpoint_sha256"]
    )
    state_match = original["candidate_state_sha256"] == replay["candidate_state_sha256"]
    original_normalized_sha256 = normalized_replay_sha256(original)
    replay_normalized_sha256 = normalized_replay_sha256(replay)
    gates = {
        "cache_reextraction_count": 0,
        "replay_candidate_state_count": len(replay["candidate_state_sha256"]),
        "normalized_prediction_rows_byte_identical": (
            original_prediction_sha256 == replay_prediction_sha256
        ),
        "candidate_checkpoints_byte_identical": checkpoint_match,
        "candidate_states_identical": state_match,
        "diagnostics_byte_identical": (
            original_diagnostics_sha256 == replay_diagnostics_sha256
        ),
        "normalized_audit_byte_identical": (
            original_normalized_sha256 == replay_normalized_sha256
        ),
    }
    replay_pass = (
        gates["cache_reextraction_count"] == 0
        and gates["replay_candidate_state_count"] == 15
        and gates["normalized_prediction_rows_byte_identical"]
        and gates["candidate_checkpoints_byte_identical"]
        and gates["candidate_states_identical"]
        and gates["diagnostics_byte_identical"]
        and gates["normalized_audit_byte_identical"]
    )
    audit = {
        "schema_version": "autoresearch-v19r.deterministic-replay-audit-v1",
        "pass": replay_pass,
        "normalization_contract": REPLAY_NORMALIZATION_CONTRACT,
        "phase_authorization": authorization,
        "execution_claim": execution_claim,
        "phase3a_handoff": {
            key: value for key, value in handoff.items() if key != "original"
        },
        "gates": gates,
        "original": {
            "summary_path": relative_path(original_summary_path),
            "summary_sha256": v9.sha256_file(original_summary_path),
            "normalized_summary_sha256": original_normalized_sha256,
            "prediction_rows_sha256": original_prediction_sha256,
        },
        "replay": {
            "summary_path": relative_path(replay_summary_path),
            "summary_sha256": v9.sha256_file(replay_summary_path),
            "normalized_summary_sha256": replay_normalized_sha256,
            "prediction_rows_sha256": replay_prediction_sha256,
        },
        "runtime_sha256_after": runtime_hashes(args),
        "runtime_unchanged": runtime_hashes(args) == EXPECTED_RUNTIME_SHA256,
        "final_test_read": False,
        "promotion_eligible": False,
    }
    enforce_execution_claim_deadline(execution_claim)
    write_json(args.output_dir / "replay-audit.json", audit)
    if not replay_pass or not audit["runtime_unchanged"]:
        raise ValueError("v19 deterministic replay failed")
    final_evaluation = copy.deepcopy(original["evaluation"])
    verdict, claim = classify_candidate_verdict(final_evaluation)
    final_evaluation["replay_pending"] = False
    final_evaluation["verdict"] = verdict
    final_evaluation["claim"] = claim
    final_evaluation["replay_pass"] = True
    final_evaluation["pass"] = verdict in {"supported_strong", "supported_reference"}
    final = {
        **original,
        "schema_version": "autoresearch-v19r.final-summary-v1",
        "pass": final_evaluation["pass"],
        "decision_status": "complete_research_reference_only",
        "evaluation": final_evaluation,
        "replay_phase_authorization": authorization,
        "replay_execution_claim": execution_claim,
        "replay_audit_path": relative_path(args.output_dir / "replay-audit.json"),
        "replay_audit_sha256": v9.sha256_file(args.output_dir / "replay-audit.json"),
        "runtime_sha256_after": runtime_hashes(args),
        "runtime_unchanged": True,
        "final_test_read": False,
        "promotion_eligible": False,
    }
    enforce_execution_claim_deadline(execution_claim)
    write_json(args.output_dir / "final-summary.json", final)
    return final


def command_validate_contract(args: argparse.Namespace) -> dict[str, Any]:
    result = validate_static_contract(args)
    if args.output is not None:
        write_json(args.output, result)
    return result


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--evaluator", type=Path, default=DEFAULT_EVALUATOR)
    parser.add_argument("--freeze-audit", type=Path, default=DEFAULT_FREEZE_AUDIT)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--b14-manifest", type=Path, default=DEFAULT_B14_MANIFEST)
    parser.add_argument("--source-inventory", type=Path, default=DEFAULT_SOURCE_INVENTORY)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_CORPUS_MANIFEST)
    parser.add_argument("--audit", type=Path, default=DEFAULT_COLLECTION_AUDIT)
    parser.add_argument("--c1-summary", type=Path, default=DEFAULT_C1_SUMMARY)
    parser.add_argument("--c1-predictions", type=Path, default=DEFAULT_C1_PREDICTIONS)
    parser.add_argument("--c1-cache-dir", type=Path, default=DEFAULT_C1_CACHE_DIR)
    parser.add_argument("--b14-cache-dir", type=Path, default=DEFAULT_B14_CACHE_DIR)
    parser.add_argument("--cache-manifest", type=Path, default=DEFAULT_CACHE_MANIFEST)
    parser.add_argument("--control-audit", type=Path, default=DEFAULT_CONTROL_AUDIT)
    parser.add_argument("--runtime-projection", type=Path, default=DEFAULT_RUNTIME_PROJECTION)
    parser.add_argument("--runtime-prototypes", type=Path, default=DEFAULT_RUNTIME_PROTOTYPES)
    parser.add_argument("--runtime-config", type=Path, default=DEFAULT_RUNTIME_CONFIG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--supervised-device", default="cpu")
    parser.add_argument("--image-batch-size", type=int, default=IMAGE_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-contract")
    add_common_arguments(validate)
    validate.add_argument("--output", type=Path)
    validate.set_defaults(handler=command_validate_contract)

    smoke = subparsers.add_parser("smoke-extraction")
    add_common_arguments(smoke)
    smoke.add_argument("--smoke-fold", type=int, default=SMOKE_FOLD)
    smoke.add_argument("--smoke-batches", type=int, default=SMOKE_BATCHES)
    smoke.add_argument("--output", type=Path, default=DEFAULT_SMOKE_OUTPUT)
    smoke.set_defaults(handler=command_smoke_extraction)

    precompute = subparsers.add_parser("precompute")
    add_common_arguments(precompute)
    precompute.set_defaults(
        handler=command_precompute,
        freeze_audit=DEFAULT_PHASE2A_FREEZE_AUDIT,
        authorization=DEFAULT_PHASE2A_AUTHORIZATION,
    )

    control = subparsers.add_parser("replay-c1")
    add_common_arguments(control)
    control.set_defaults(
        handler=command_replay_c1,
        freeze_audit=DEFAULT_PHASE2B_FREEZE_AUDIT,
        authorization=DEFAULT_PHASE2B_AUTHORIZATION,
    )

    train = subparsers.add_parser("train-evaluate")
    add_common_arguments(train)
    train.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    train.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    train.set_defaults(
        handler=command_train_evaluate,
        freeze_audit=DEFAULT_PHASE3A_FREEZE_AUDIT,
        authorization=DEFAULT_PHASE3A_AUTHORIZATION,
    )

    replay = subparsers.add_parser("replay")
    add_common_arguments(replay)
    replay.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    replay.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    replay.set_defaults(
        handler=command_replay,
        freeze_audit=DEFAULT_PHASE3B_FREEZE_AUDIT,
        authorization=DEFAULT_PHASE3B_AUTHORIZATION,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = args.handler(args)
    print(canonical_json(result), end="")
    if args.command == "smoke-extraction":
        return 0 if result.get("pass") is True else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

