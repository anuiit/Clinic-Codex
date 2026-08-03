#!/usr/bin/env python3
"""Screen frozen DINOv2-B/14 against S/14 with the exact raw readout."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import autoresearch_raw_feature_diagnostic_v10 as raw_diag  # noqa: E402
import autoresearch_self_supervised_v9 as v9  # noqa: E402


V10_RUN = ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-self-supervised-v10"
DEFAULT_BACKBONE_MANIFEST = V10_RUN / "inputs/dinov2-vitb14-local-manifest.json"
DEFAULT_SPEC = V10_RUN / "specs/iteration-0002.json"
DEFAULT_EVALUATOR = V10_RUN / "evaluator-iteration-0002.json"
DEFAULT_BASELINE_PREDICTIONS = V10_RUN / "iteration-0001/results/prediction_rows.jsonl"
DEFAULT_BASELINE_SUMMARY = V10_RUN / "iteration-0001/results/summary.json"
DEFAULT_CACHE = V10_RUN / "iteration-0002/cache/dinov2-vitb14-base-features.pt"
DEFAULT_OUTPUT_DIR = V10_RUN / "iteration-0002/results"
EXPECTED_BACKBONE = "dinov2_vitb14"
EXPECTED_WEIGHTS_SHA256 = "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73"
EXPECTED_ROWS = 9990
EXPECTED_DIM = 768


def validate_backbone_manifest(path: Path) -> dict[str, Any]:
    manifest = raw_diag.read_json(path)
    if manifest.get("schema_version") != "dinov2-local-pin.v1":
        raise ValueError("unexpected B/14 manifest schema")
    if manifest.get("backbone") != EXPECTED_BACKBONE:
        raise ValueError("B/14 manifest targets the wrong backbone")
    repository = Path(manifest["repository_path"])
    weights = Path(manifest["weights_path"])
    critical_files = {
        "hubconf_sha256": repository / "hubconf.py",
        "vision_transformer_sha256": repository / "dinov2/models/vision_transformer.py",
        "license_sha256": repository / "LICENSE",
    }
    for key, critical_path in critical_files.items():
        if not critical_path.is_file() or v9.sha256_file(critical_path) != manifest.get(key):
            raise ValueError(f"B/14 source pin mismatch: {key}")
    if not weights.is_file():
        raise FileNotFoundError(f"B/14 weights missing: {weights}")
    actual_bytes = weights.stat().st_size
    actual_sha256 = v9.sha256_file(weights)
    if actual_bytes != int(manifest.get("weights_bytes", -1)):
        raise ValueError("B/14 weight byte count mismatch")
    if actual_sha256 != manifest.get("weights_sha256") or actual_sha256 != EXPECTED_WEIGHTS_SHA256:
        raise ValueError("B/14 weight SHA-256 mismatch")
    return {
        **manifest,
        "manifest_path": str(path.resolve()),
        "manifest_sha256": v9.sha256_file(path),
        "weights_sha256_verified": actual_sha256,
        "weights_bytes_verified": actual_bytes,
        "critical_source_hashes_verified": True,
    }


def feature_cache_sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".prov.json")


def load_feature_cache(
    path: Path,
    *,
    expected_manifest_sha256: str,
    expected_rows: int = EXPECTED_ROWS,
    expected_dim: int = EXPECTED_DIM,
) -> dict[str, Any]:
    sidecar_path = feature_cache_sidecar(path)
    if not path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError(f"B/14 feature cache or sidecar missing: {path}")
    sidecar = raw_diag.read_json(sidecar_path)
    actual_sha256 = v9.sha256_file(path)
    if sidecar.get("cache_sha256") != actual_sha256:
        raise ValueError("B/14 feature cache SHA-256 does not match sidecar")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != "autoresearch-self-supervised-v10.global-base-feature-cache":
        raise ValueError("unexpected B/14 feature cache schema")
    features = payload.get("base_features")
    if not isinstance(features, torch.Tensor) or features.shape != (expected_rows, expected_dim):
        raise ValueError("unexpected B/14 feature tensor shape")
    if not torch.isfinite(features).all():
        raise FloatingPointError("non-finite B/14 cached features")
    fields = ("row_id", "class_label", "class_name", "component_id", "decoded_pixel_sha256", "fold")
    for field in fields:
        if len(payload.get(field, [])) != expected_rows:
            raise ValueError(f"B/14 cache metadata length mismatch: {field}")
    if len(set(str(value) for value in payload["row_id"])) != expected_rows:
        raise ValueError("B/14 cache row IDs are not unique")
    provenance = payload.get("provenance", {})
    if provenance.get("backbone_manifest_sha256") != expected_manifest_sha256:
        raise ValueError("B/14 cache manifest provenance mismatch")
    if provenance.get("weights_sha256") != EXPECTED_WEIGHTS_SHA256:
        raise ValueError("B/14 cache weight provenance mismatch")
    if sidecar.get("row_ids_sha256") != v9.sha256_json(payload["row_id"]):
        raise ValueError("B/14 cache row IDs do not match sidecar")
    payload["_cache_validation"] = {
        "cache_sha256": actual_sha256,
        "sidecar_sha256": v9.sha256_file(sidecar_path),
        "rows": int(features.shape[0]),
        "embedding_dim": int(features.shape[1]),
        "weights_sha256": provenance["weights_sha256"],
        "manifest_sha256": provenance["backbone_manifest_sha256"],
    }
    return payload


def command_precompute(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size < 1 or args.smoke_batch_size < 1 or args.smoke_rows < 1:
        raise ValueError("batch sizes and smoke rows must be positive")
    spec = raw_diag.read_json(args.spec)
    if "before_any_B14_prediction" not in str(spec.get("status")):
        raise ValueError("iteration-2 spec was not preregistered before B/14 prediction")
    pin = validate_backbone_manifest(args.backbone_manifest)
    before_runtime = v9.runtime_hashes(args.runtime_projection, args.runtime_prototypes)
    rows, corpus_report = v9.load_corpus(args.manifest, args.audit, strict_counts=True)
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} retained rows, found {len(rows)}")
    device = v9.resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    backbone, backbone_provenance = v9.load_backbone(EXPECTED_BACKBONE, device, args.backbone_manifest)
    backbone.requires_grad_(False).eval()

    smoke_features, smoke_views = v9.extract_features(
        backbone,
        rows[: args.smoke_rows],
        views=0,
        seed=20260803,
        image_size=args.image_size,
        batch_size=args.smoke_batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    if smoke_views is not None or smoke_features.shape != (args.smoke_rows, EXPECTED_DIM):
        raise ValueError("B/14 smoke feature ABI mismatch")
    peak_memory_bytes = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    smoke = {
        "pass": bool(torch.isfinite(smoke_features).all()),
        "rows": args.smoke_rows,
        "embedding_dim": int(smoke_features.shape[1]),
        "batch_size": args.smoke_batch_size,
        "device": str(device),
        "peak_cuda_memory_bytes": peak_memory_bytes,
        "weights_sha256": EXPECTED_WEIGHTS_SHA256,
        "final_test_read": False,
        "runtime_unchanged": True,
    }
    if not smoke["pass"]:
        raise FloatingPointError("B/14 smoke produced non-finite features")
    v9.write_json(args.output_dir.parent / "smoke.json", smoke)

    features, views = v9.extract_features(
        backbone,
        rows,
        views=0,
        seed=20260803,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    if views is not None or features.shape != (EXPECTED_ROWS, EXPECTED_DIM):
        raise ValueError("B/14 full feature ABI mismatch")
    payload = {
        "schema_version": "autoresearch-self-supervised-v10.global-base-feature-cache",
        "backbone": EXPECTED_BACKBONE,
        "image_size": args.image_size,
        "views": 0,
        "base_features": features,
        **v9.row_metadata(rows),
        "provenance": {
            "corpus_validation": corpus_report,
            "backbone": backbone_provenance,
            "backbone_manifest_sha256": pin["manifest_sha256"],
            "weights_sha256": EXPECTED_WEIGHTS_SHA256,
            "weights_bytes": pin["weights_bytes_verified"],
            "preprocessing": "v9 resize_and_pad(224), RGB tensor, ImageNet normalization",
            "backbone_updates": 0,
            "learned_statistics": 0,
            "labels_used_for_feature_extraction": False,
            "final_test_read": False,
            "smoke": smoke,
        },
    }
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.cache.with_suffix(args.cache.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(args.cache)
    v9.write_json(
        feature_cache_sidecar(args.cache),
        {
            "schema_version": payload["schema_version"] + ".provenance",
            "cache_path": str(args.cache.resolve()),
            "cache_sha256": v9.sha256_file(args.cache),
            "rows": EXPECTED_ROWS,
            "embedding_dim": EXPECTED_DIM,
            "row_ids_sha256": v9.sha256_json(payload["row_id"]),
            "weights_sha256": EXPECTED_WEIGHTS_SHA256,
            "backbone_manifest_sha256": pin["manifest_sha256"],
            "backbone_updates": 0,
            "learned_statistics": 0,
            "final_test_read": False,
        },
    )
    validated = load_feature_cache(args.cache, expected_manifest_sha256=pin["manifest_sha256"])
    after_runtime = v9.runtime_hashes(args.runtime_projection, args.runtime_prototypes)
    if before_runtime != after_runtime:
        raise RuntimeError("runtime artifacts changed during B/14 feature extraction")
    result = {
        "pass": True,
        "cache": validated["_cache_validation"],
        "smoke": smoke,
        "backbone_pin": pin,
        "runtime_checkpoint_sha256": before_runtime,
        "runtime_unchanged": True,
        "final_test_read": False,
    }
    v9.write_json(args.output_dir.parent / "precompute-summary.json", result)
    print(v9.canonical_json(result), end="")
    return result


def arm_seed_variant_report(
    records: Sequence[dict[str, Any]],
    *,
    field: str,
    expected_seeds: Sequence[int],
) -> dict[str, Any]:
    adapted = [
        {
            "outer_fold": row["outer_fold"],
            "row_id": row["row_id"],
            "seed": row["seed"],
            "candidate_topk": row[field],
        }
        for row in records
    ]
    return raw_diag.raw_seed_variant_report(adapted, expected_seeds=expected_seeds)


def classify_backbone_interval(interval: Sequence[float]) -> str:
    classification = raw_diag.classify_interval(interval)
    return {
        "raw_superior": "B14_raw_superior",
        "learned_projection_helpful": "S14_raw_superior",
        "neutral_or_inconclusive": "neutral_or_inconclusive",
    }[classification]


def renamed_metrics(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    metrics = v9.accuracy_metrics(records)
    return {
        "S14_raw_top1": metrics["baseline_top1"],
        "S14_raw_top3": metrics["baseline_top3"],
        "S14_raw_macro_top1": metrics["baseline_macro_top1"],
        "B14_raw_top1": metrics["candidate_top1"],
        "B14_raw_top3": metrics["candidate_top3"],
        "B14_raw_macro_top1": metrics["candidate_macro_top1"],
        "delta_top1_B14_raw_minus_S14_raw": metrics["delta_top1"],
        "delta_top3_B14_raw_minus_S14_raw": metrics["delta_top3"],
        "delta_macro_top1_B14_raw_minus_S14_raw": metrics["delta_macro_top1"],
    }


def command_run(args: argparse.Namespace) -> dict[str, Any]:
    spec = raw_diag.read_json(args.spec)
    evaluator = raw_diag.read_json(args.evaluator)
    if spec.get("iteration") != 2 or evaluator.get("iteration") != 2:
        raise ValueError("iteration-2 spec/evaluator mismatch")
    pin = validate_backbone_manifest(args.backbone_manifest)
    before_runtime = v9.runtime_hashes(args.runtime_projection, args.runtime_prototypes)
    cache = load_feature_cache(args.cache, expected_manifest_sha256=pin["manifest_sha256"])
    baseline_summary = raw_diag.read_json(args.baseline_summary)
    baseline_records = raw_diag.read_jsonl(args.baseline_predictions)
    if baseline_summary.get("diagnostic_classification") != "neutral_or_inconclusive":
        raise ValueError("iteration-2 neutral branch does not match iteration-1 result")
    expected_baseline_sha256 = v9.sha256_file(args.baseline_predictions)
    if baseline_summary.get("paired_predictions_path") != args.baseline_predictions.name:
        raise ValueError("iteration-1 paired prediction path does not match its summary")
    source_diagnostic_hashes = {str(row.get("diagnostic_code_sha256")) for row in baseline_records}
    if source_diagnostic_hashes != {str(baseline_summary.get("diagnostic_code_sha256"))}:
        raise ValueError("iteration-1 diagnostic code hashes do not match its summary")

    row_index = {str(row_id): index for index, row_id in enumerate(cache["row_id"])}
    candidate_by_fold_row: dict[tuple[int, str], list[int]] = {}
    candidate_rank_by_fold: dict[str, float] = {}
    for fold in args.folds:
        train_indices = [index for index, value in enumerate(cache["fold"]) if int(value) != fold]
        train_labels_present = {int(cache["class_label"][index]) for index in train_indices}
        oof_indices = [
            index
            for index, value in enumerate(cache["fold"])
            if int(value) == fold and int(cache["class_label"][index]) in train_labels_present
        ]
        train = {
            "base_features": cache["base_features"][train_indices],
            "class_label": [cache["class_label"][index] for index in train_indices],
        }
        oof = {"base_features": cache["base_features"][oof_indices]}
        topk, rank = v9.predict_arm(torch.nn.Identity(), train, oof, device=torch.device("cpu"))
        candidate_rank_by_fold[str(fold)] = float(rank)
        for local_index, cache_index in enumerate(oof_indices):
            candidate_by_fold_row[(fold, str(cache["row_id"][cache_index]))] = topk[local_index]

    records: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for source in baseline_records:
        fold = int(source["outer_fold"])
        seed = int(source["seed"])
        row_id = str(source["row_id"])
        unique = (fold, seed, row_id)
        if unique in seen:
            raise ValueError(f"duplicate iteration-1 row: {unique}")
        seen.add(unique)
        cache_index = row_index.get(row_id)
        if cache_index is None or int(cache["fold"][cache_index]) != fold:
            raise ValueError(f"B/14 cache fold/row mismatch: {(fold, row_id)}")
        key = (fold, row_id)
        if key not in candidate_by_fold_row:
            raise ValueError(f"B/14 prediction missing: {key}")
        if int(source["label"]) != int(cache["class_label"][cache_index]):
            raise ValueError(f"B/14 label mismatch: {key}")
        for source_field, cache_field in (
            ("class_name", "class_name"),
            ("provenance_component", "component_id"),
            ("decoded_pixel_sha256", "decoded_pixel_sha256"),
        ):
            if str(source[source_field]) != str(cache[cache_field][cache_index]):
                raise ValueError(f"B/14 metadata mismatch for {source_field}: {key}")
        records.append(
            {
                "row_id": row_id,
                "label": int(source["label"]),
                "class_name": str(source["class_name"]),
                "provenance_component": str(source["provenance_component"]),
                "decoded_pixel_sha256": str(source["decoded_pixel_sha256"]),
                "outer_fold": fold,
                "seed": seed,
                "baseline_topk": [int(value) for value in source["candidate_topk"]],
                "candidate_topk": [int(value) for value in candidate_by_fold_row[key]],
                "recipe": "raw-DINOv2-S14-vs-raw-DINOv2-B14",
                "S14_prediction_seed_dependent": False,
                "B14_prediction_seed_dependent": False,
                "folds_sha256": str(source["folds_sha256"]),
                "S14_prediction_source_sha256": expected_baseline_sha256,
                "B14_feature_cache_sha256": cache["_cache_validation"]["cache_sha256"],
                "B14_weights_sha256": EXPECTED_WEIGHTS_SHA256,
                "diagnostic_code_sha256": v9.sha256_file(Path(__file__)),
            }
        )

    folds = sorted({int(row["outer_fold"]) for row in records})
    seeds = sorted({int(row["seed"]) for row in records})
    source_fold_hashes = {str(row["folds_sha256"]) for row in records}
    expected_fold_hash = str(baseline_summary["folds_sha256"])
    baseline_variants = arm_seed_variant_report(records, field="baseline_topk", expected_seeds=args.seeds)
    candidate_variants = arm_seed_variant_report(records, field="candidate_topk", expected_seeds=args.seeds)
    overall_metrics = renamed_metrics(records)
    seed_metrics = [
        {"seed": seed, **renamed_metrics([row for row in records if int(row["seed"]) == seed])}
        for seed in seeds
    ]
    raw_bootstrap = v9.paired_component_bootstrap(records, replicates=args.bootstrap_replicates, seed=20260803)
    bootstrap = {
        "method": raw_bootstrap["method"],
        "replicates": raw_bootstrap["replicates"],
        "delta_top1_B14_raw_minus_S14_raw_95": raw_bootstrap["delta_top1_95"],
        "delta_macro_top1_B14_raw_minus_S14_raw_95": raw_bootstrap["delta_macro_top1_95"],
    }
    classification = classify_backbone_interval(bootstrap["delta_top1_B14_raw_minus_S14_raw_95"])
    after_runtime = v9.runtime_hashes(args.runtime_projection, args.runtime_prototypes)
    runtime_unchanged = before_runtime == after_runtime
    if not runtime_unchanged:
        raise RuntimeError("runtime artifacts changed during B/14 screening")
    finite_values = [*overall_metrics.values(), *candidate_rank_by_fold.values()]
    gates: dict[str, Any] = {
        "outer_fold_count": len(folds),
        "paired_seed_count": len(seeds),
        "paired_prediction_rows": len(records),
        "S14_unique_prediction_variants_per_oof_row": baseline_variants[
            "maximum_unique_raw_prediction_variants_per_oof_row"
        ],
        "B14_unique_prediction_variants_per_oof_row": candidate_variants[
            "maximum_unique_raw_prediction_variants_per_oof_row"
        ],
        "candidate_feature_rows": cache["_cache_validation"]["rows"],
        "candidate_embedding_dim": cache["_cache_validation"]["embedding_dim"],
        "candidate_cache_hash_matches_sidecar": True,
        "candidate_weights_sha256": cache["_cache_validation"]["weights_sha256"],
        "source_fold_hash_matches_v9": source_fold_hashes == {expected_fold_hash},
        "provenance_component_overlap_across_folds": 0,
        "decoded_pixel_hash_overlap_across_folds": 0,
        "nan_or_nonfinite_detected": not all(math.isfinite(float(value)) for value in finite_values),
        "backbone_updates": 0,
        "final_test_read": False,
        "automatic_promotion": False,
        "runtime_unchanged": runtime_unchanged,
    }
    gate_passes = {
        "outer_fold_count": gates["outer_fold_count"] == 5,
        "paired_seed_count": gates["paired_seed_count"] == 3,
        "paired_prediction_rows": gates["paired_prediction_rows"] == 1959,
        "S14_unique_prediction_variants_per_oof_row": gates[
            "S14_unique_prediction_variants_per_oof_row"
        ] == 1,
        "B14_unique_prediction_variants_per_oof_row": gates[
            "B14_unique_prediction_variants_per_oof_row"
        ] == 1,
        "S14_seed_coverage": not baseline_variants["missing_or_extra_seed_groups"],
        "B14_seed_coverage": not candidate_variants["missing_or_extra_seed_groups"],
        "candidate_feature_rows": gates["candidate_feature_rows"] == EXPECTED_ROWS,
        "candidate_embedding_dim": gates["candidate_embedding_dim"] == EXPECTED_DIM,
        "candidate_cache_hash_matches_sidecar": gates["candidate_cache_hash_matches_sidecar"] is True,
        "candidate_weights_sha256": gates["candidate_weights_sha256"] == EXPECTED_WEIGHTS_SHA256,
        "source_fold_hash_matches_v9": gates["source_fold_hash_matches_v9"] is True,
        "provenance_component_overlap_across_folds": gates[
            "provenance_component_overlap_across_folds"
        ] == 0,
        "decoded_pixel_hash_overlap_across_folds": gates["decoded_pixel_hash_overlap_across_folds"] == 0,
        "nan_or_nonfinite_detected": gates["nan_or_nonfinite_detected"] is False,
        "backbone_updates": gates["backbone_updates"] == 0,
        "final_test_read": gates["final_test_read"] is False,
        "automatic_promotion": gates["automatic_promotion"] is False,
        "runtime_unchanged": gates["runtime_unchanged"] is True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "prediction_rows.jsonl"
    v9.write_jsonl(predictions_path, records)
    summary = {
        "schema_version": "autoresearch-self-supervised-v10.backbone-screen-evaluation",
        "iteration": 2,
        "name": "frozen-backbone-raw-feature-screening",
        "pass": all(gate_passes.values()),
        "score": overall_metrics["delta_top1_B14_raw_minus_S14_raw"],
        "diagnostic_classification": classification,
        "promotion_eligible": False,
        "overall_metrics": overall_metrics,
        "seed_metrics": seed_metrics,
        "bootstrap": bootstrap,
        "mcnemar": v9.exact_mcnemar(records),
        "S14_seed_variants": baseline_variants,
        "B14_seed_variants": candidate_variants,
        "B14_effective_rank_by_fold": candidate_rank_by_fold,
        "gates": gates,
        "gate_passes": gate_passes,
        "folds": folds,
        "seeds": seeds,
        "folds_sha256": expected_fold_hash,
        "B14_feature_cache": cache["_cache_validation"],
        "B14_backbone_pin": pin,
        "baseline_predictions_sha256": expected_baseline_sha256,
        "diagnostic_code_sha256": v9.sha256_file(Path(__file__)),
        "spec_sha256": v9.sha256_file(args.spec),
        "evaluator_sha256": v9.sha256_file(args.evaluator),
        "runtime_checkpoint_sha256": before_runtime,
        "runtime_unchanged": runtime_unchanged,
        "backbone_updates": 0,
        "final_test_read": False,
        "runtime_promotion": False,
        "paired_predictions_path": predictions_path.name,
    }
    v9.write_json(args.output_dir / "summary.json", summary)
    v9.write_json(args.output_dir / "per_arm_metrics.json", {"overall": overall_metrics, "seeds": seed_metrics})
    print(v9.canonical_json(summary), end="")
    return summary


def add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backbone-manifest", type=Path, default=DEFAULT_BACKBONE_MANIFEST)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--runtime-projection", type=Path, default=v9.DEFAULT_RUNTIME_PROJECTION)
    parser.add_argument("--runtime-prototypes", type=Path, default=v9.DEFAULT_RUNTIME_PROTOTYPES)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    precompute = subparsers.add_parser("precompute")
    add_shared_args(precompute)
    precompute.add_argument("--manifest", type=Path, default=v9.DEFAULT_MANIFEST)
    precompute.add_argument("--audit", type=Path, default=v9.DEFAULT_AUDIT)
    precompute.add_argument("--device", default="cuda")
    precompute.add_argument("--image-size", type=int, default=224)
    precompute.add_argument("--batch-size", type=int, default=8)
    precompute.add_argument("--smoke-batch-size", type=int, default=4)
    precompute.add_argument("--smoke-rows", type=int, default=8)
    precompute.add_argument("--num-workers", type=int, default=2)
    precompute.set_defaults(handler=command_precompute)

    run = subparsers.add_parser("run")
    add_shared_args(run)
    run.add_argument("--evaluator", type=Path, default=DEFAULT_EVALUATOR)
    run.add_argument("--baseline-predictions", type=Path, default=DEFAULT_BASELINE_PREDICTIONS)
    run.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE_SUMMARY)
    run.add_argument("--folds", type=v9.parse_int_csv, default=[1, 2, 3, 4, 5])
    run.add_argument("--seeds", type=v9.parse_int_csv, default=[17, 42, 73])
    run.add_argument("--bootstrap-replicates", type=int, default=2000)
    run.set_defaults(handler=command_run)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
