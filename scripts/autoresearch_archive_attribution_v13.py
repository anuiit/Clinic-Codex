#!/usr/bin/env python3
"""Build an exact archive collision index and recover development provenance.

The runner is intentionally model-free.  It reads the 49 historical ZIP
archives without extracting them, hashes every member, reproduces the v8 RGB
hash contract for raster entries, and attributes retained development rows only
through exact collisions.  It never changes source data or v12 exclusions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import warnings
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = (
    ROOT
    / ".omc/autoresearch/elements-baseline-replacement/runs/"
    "20260803-archive-attribution-v13"
)
SPEC_PATH = RUN_DIR / "specs/iteration-0001.json"
EVALUATOR_PATH = RUN_DIR / "evaluator.json"
DEFAULT_INDEX_DIR = RUN_DIR / "archive-index"
DEFAULT_OUTPUT_DIR = RUN_DIR / "iteration-0001"
DEFAULT_REPLAY_DIR = RUN_DIR / "iteration-0001-replay"
DEFAULT_EVALUATION_PATH = RUN_DIR / "evaluations/iteration-0001.json"

EXPECTED_SPEC_SHA256 = "e407e9ddf6a5da509f9b13e9ee7ed5e90e736a0e67111aa767903bff3a2e890d"
EXPECTED_EVALUATOR_SHA256 = (
    "0d2c55609471282a1299eef657311631872b38edf24fc2ffebffef5cd4812e36"
)
EXPECTED_ARCHIVE_CONTENT_INVENTORY_SHA256 = (
    "71ea5cba9246b70cdae37f04464e6152f9c3962b68bee6f1bc793f1343bd0e9f"
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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_line(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_line(row))
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
    return rows


def decoded_rgb_sha256_bytes(value: bytes) -> tuple[str, int, int]:
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(value)) as image:
            image.load()
            rgb = image.convert("RGB")
            width, height = rgb.size
            digest = hashlib.sha256()
            digest.update(width.to_bytes(8, "big"))
            digest.update(height.to_bytes(8, "big"))
            digest.update(rgb.tobytes())
            return digest.hexdigest(), width, height


def validate_contract(
    spec_path: Path = SPEC_PATH,
    evaluator_path: Path = EVALUATOR_PATH,
) -> dict[str, Any]:
    spec_hash = sha256_file(spec_path)
    evaluator_hash = sha256_file(evaluator_path)
    if spec_hash != EXPECTED_SPEC_SHA256:
        raise ValueError(f"v13 spec hash mismatch: {spec_hash}")
    if evaluator_hash != EXPECTED_EVALUATOR_SHA256:
        raise ValueError(f"v13 evaluator hash mismatch: {evaluator_hash}")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    evaluator = json.loads(evaluator_path.read_text(encoding="utf-8"))
    if spec.get("model_inference_allowed") is not False:
        raise ValueError("v13 must forbid model inference")
    if spec.get("final_test_read_allowed") is not False:
        raise ValueError("v13 must forbid final-test access")
    if spec.get("automatic_promotion_allowed") is not False:
        raise ValueError("v13 must forbid automatic promotion")
    if evaluator.get("exclusion_change_allowed") is not False:
        raise ValueError("v13 must forbid exclusion changes")
    if Image.__version__ != spec["inputs"]["pillow_version"]:
        raise ValueError(
            f"Pillow version mismatch: expected {spec['inputs']['pillow_version']}, "
            f"got {Image.__version__}"
        )
    return {
        "spec": spec,
        "evaluator": evaluator,
        "spec_sha256": spec_hash,
        "evaluator_sha256": evaluator_hash,
    }


def aggregate_archive_hash(archive_hashes: dict[str, str]) -> str:
    inventory = "".join(
        f"{digest}  {name}\n" for name, digest in sorted(archive_hashes.items())
    ).encode("utf-8")
    return sha256_bytes(inventory)


def hash_archives(
    archive_paths: Sequence[Path], *, workers: int
) -> dict[str, str]:
    result: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        pending = {executor.submit(sha256_file, path): path for path in archive_paths}
        for future in as_completed(pending):
            path = pending[future]
            result[path.name] = future.result()
    return dict(sorted(result.items()))


def _member_name(info: zipfile.ZipInfo) -> str:
    return info.filename.replace("\\", "/")


def _index_paths(index_dir: Path, archive_name: str) -> tuple[Path, Path]:
    stem = archive_name.removesuffix(".zip")
    return index_dir / f"{stem}.jsonl", index_dir / f"{stem}.summary.json"


def _valid_cached_index(
    *,
    index_path: Path,
    summary_path: Path,
    archive_sha256: str,
    pillow_version: str,
) -> dict[str, Any] | None:
    if not index_path.is_file() or not summary_path.is_file():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    required = {
        "schema_version": "autoresearch-archive-attribution-v13.archive-index",
        "archive_sha256": archive_sha256,
        "pillow_version": pillow_version,
        "index_sha256": sha256_file(index_path),
    }
    if any(summary.get(key) != value for key, value in required.items()):
        return None
    return summary


def index_archive(
    archive_path: Path,
    *,
    archive_sha256: str,
    index_dir: Path,
    supported_extensions: set[str],
    max_member_bytes: int,
    force: bool,
) -> dict[str, Any]:
    index_path, summary_path = _index_paths(index_dir, archive_path.name)
    if not force:
        cached = _valid_cached_index(
            index_path=index_path,
            summary_path=summary_path,
            archive_sha256=archive_sha256,
            pillow_version=Image.__version__,
        )
        if cached is not None:
            return {**cached, "reused": True}

    rows: list[dict[str, Any]] = []
    raster_success = 0
    raster_failures = 0
    non_raster_entries = 0
    oversized_entries = 0
    with zipfile.ZipFile(archive_path, "r") as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        for member_index, info in enumerate(infos):
            name = _member_name(info)
            extension = PurePosixPath(name).suffix.casefold()
            is_raster = extension in supported_extensions
            if not is_raster:
                non_raster_entries += 1
            row: dict[str, Any] = {
                "archive_name": archive_path.name,
                "member_index": member_index,
                "member_name": name,
                "extension": extension,
                "file_size": info.file_size,
                "compress_size": info.compress_size,
                "crc32": f"{info.CRC:08x}",
                "is_supported_raster": is_raster,
                "byte_sha256": None,
                "decoded_rgb_sha256": None,
                "width": None,
                "height": None,
                "status": None,
                "error": None,
            }
            if info.file_size > max_member_bytes:
                oversized_entries += 1
                if is_raster:
                    raster_failures += 1
                row["status"] = "oversized"
                row["error"] = (
                    f"uncompressed member size {info.file_size} exceeds "
                    f"limit {max_member_bytes}"
                )
                rows.append(row)
                continue
            try:
                value = archive.read(info)
                row["byte_sha256"] = sha256_bytes(value)
            except Exception as error:  # zipfile exposes heterogeneous read errors
                if is_raster:
                    raster_failures += 1
                row["status"] = "read_error"
                row["error"] = f"{type(error).__name__}: {error}"
                rows.append(row)
                continue
            if not is_raster:
                row["status"] = "byte_indexed_non_raster"
                rows.append(row)
                continue
            try:
                pixel_hash, width, height = decoded_rgb_sha256_bytes(value)
                row["decoded_rgb_sha256"] = pixel_hash
                row["width"] = width
                row["height"] = height
                row["status"] = "indexed"
                raster_success += 1
            except Exception as error:  # malformed historical images are evidence
                row["status"] = "decode_error"
                row["error"] = f"{type(error).__name__}: {error}"
                raster_failures += 1
            rows.append(row)

    write_jsonl(index_path, rows)
    summary = {
        "schema_version": "autoresearch-archive-attribution-v13.archive-index",
        "archive_name": archive_path.name,
        "archive_path": str(archive_path.resolve()),
        "archive_sha256": archive_sha256,
        "archive_size": archive_path.stat().st_size,
        "index_path": str(index_path.resolve()),
        "index_sha256": sha256_file(index_path),
        "pillow_version": Image.__version__,
        "non_directory_entries": len(rows),
        "supported_raster_entries": raster_success + raster_failures,
        "raster_decode_successes": raster_success,
        "raster_failures": raster_failures,
        "non_raster_entries": non_raster_entries,
        "oversized_entries": oversized_entries,
        "reused": False,
    }
    write_json(summary_path, summary)
    return summary


def index_all_archives(
    archive_paths: Sequence[Path],
    *,
    archive_hashes: dict[str, str],
    index_dir: Path,
    supported_extensions: set[str],
    max_member_bytes: int,
    workers: int,
    force: bool,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    index_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        pending = {
            executor.submit(
                index_archive,
                path,
                archive_sha256=archive_hashes[path.name],
                index_dir=index_dir,
                supported_extensions=supported_extensions,
                max_member_bytes=max_member_bytes,
                force=force,
            ): path
            for path in archive_paths
        }
        for future in as_completed(pending):
            summaries.append(future.result())
    return sorted(summaries, key=lambda row: row["archive_name"])


def normalize_cote(value: str) -> str:
    return value.strip().replace("\\", "/").casefold()


def load_codex_catalog(
    codex_path: Path, elements_path: Path
) -> tuple[dict[int, str], dict[str, set[int]]]:
    with codex_path.open("r", encoding="utf-8-sig", newline="") as handle:
        codex_rows = list(csv.DictReader(handle))
    titles = {int(row["id"]): row["titre"] for row in codex_rows}
    if sorted(titles) != list(range(1, 51)):
        raise ValueError("codex catalogue ids must be exactly 1 through 50")
    cote_to_codex: dict[str, set[int]] = defaultdict(set)
    with elements_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            cote = normalize_cote(row["cote"])
            if cote:
                cote_to_codex[cote].add(int(row["codexid"]))
    return titles, dict(cote_to_codex)


def build_archive_to_codex_map(
    summaries: Sequence[dict[str, Any]],
    *,
    index_dir: Path,
    codex_titles: dict[int, str],
    cote_to_codex: dict[str, set[int]],
) -> dict[str, Any]:
    archives: list[dict[str, Any]] = []
    for summary in summaries:
        index_path, _ = _index_paths(index_dir, summary["archive_name"])
        hit_members: set[str] = set()
        hit_counts: Counter[int] = Counter()
        for row in read_jsonl(index_path):
            stem = normalize_cote(PurePosixPath(row["member_name"]).stem)
            codex_ids = cote_to_codex.get(stem, set())
            if codex_ids:
                hit_members.add(row["member_name"])
                for codex_id in codex_ids:
                    hit_counts[codex_id] += 1
        matched_codex_ids = sorted(hit_counts)
        unique_codex_id = (
            matched_codex_ids[0] if len(matched_codex_ids) == 1 else None
        )
        archives.append(
            {
                "archive_name": summary["archive_name"],
                "exact_cote_member_hits": len(hit_members),
                "matched_codex_ids": matched_codex_ids,
                "codex_hit_counts": {
                    str(codex_id): hit_counts[codex_id]
                    for codex_id in matched_codex_ids
                },
                "matched_codex_titles": {
                    str(codex_id): codex_titles[codex_id]
                    for codex_id in matched_codex_ids
                },
                "unique_codex_id": unique_codex_id,
                "unique_codex_title": (
                    codex_titles[unique_codex_id]
                    if unique_codex_id is not None
                    else None
                ),
                "mapping_status": (
                    "unique_exact_cote_mapping"
                    if unique_codex_id is not None
                    else "ambiguous_exact_cote_mapping"
                    if matched_codex_ids
                    else "no_exact_cote_mapping"
                ),
            }
        )
    return {
        "schema_version": (
            "autoresearch-archive-attribution-v13.archive-to-codex-map"
        ),
        "mapping_rule": (
            "exact casefolded equality of ZIP member stem and elements.csv cote"
        ),
        "archive_count": len(archives),
        "uniquely_mapped_archives": sum(
            row["unique_codex_id"] is not None for row in archives
        ),
        "ambiguous_archives": sum(
            row["mapping_status"] == "ambiguous_exact_cote_mapping"
            for row in archives
        ),
        "unmapped_archives": sum(
            row["mapping_status"] == "no_exact_cote_mapping" for row in archives
        ),
        "archives": archives,
    }


def load_retained_development_rows(
    manifest_path: Path, audit_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_jsonl(manifest_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    quarantined = set(audit["quarantine"]["row_ids"])
    retained = [row for row in rows if row["row_id"] not in quarantined]
    if len(rows) != 10039:
        raise ValueError(f"unexpected v8 manifest row count: {len(rows)}")
    if len(quarantined) != 49:
        raise ValueError(f"unexpected v8 quarantine count: {len(quarantined)}")
    if len(retained) != 9990:
        raise ValueError(f"unexpected retained row count: {len(retained)}")
    return retained, {
        "manifest_rows": len(rows),
        "quarantined_rows": len(quarantined),
        "retained_rows": len(retained),
    }


def hash_development_artifacts(
    rows: Sequence[dict[str, Any]], *, workers: int
) -> dict[str, str]:
    paths = {str(row["image_path"]): Path(row["image_path"]) for row in rows}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError(f"missing development artifacts: {missing[:3]}")
    hashes: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        pending = {executor.submit(sha256_file, path): key for key, path in paths.items()}
        for future in as_completed(pending):
            hashes[pending[future]] = future.result()
    return hashes


def build_collision_lookups(
    summaries: Sequence[dict[str, Any]], *, index_dir: Path
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    byte_to_archives: dict[str, set[str]] = defaultdict(set)
    rgb_to_archives: dict[str, set[str]] = defaultdict(set)
    for summary in summaries:
        archive_name = summary["archive_name"]
        index_path, _ = _index_paths(index_dir, archive_name)
        for row in read_jsonl(index_path):
            if row["byte_sha256"]:
                byte_to_archives[row["byte_sha256"]].add(archive_name)
            if row["decoded_rgb_sha256"]:
                rgb_to_archives[row["decoded_rgb_sha256"]].add(archive_name)
    return dict(byte_to_archives), dict(rgb_to_archives)


def build_attribution(
    rows: Sequence[dict[str, Any]],
    *,
    development_byte_hashes: dict[str, str],
    byte_to_archives: dict[str, set[str]],
    rgb_to_archives: dict[str, set[str]],
    archive_map: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mapping_by_archive = {
        row["archive_name"]: row for row in archive_map["archives"]
    }
    output: list[dict[str, Any]] = []
    unique_codex_ids: set[int] = set()
    status_counts: Counter[str] = Counter()
    byte_match_rows = 0
    rgb_match_rows = 0
    for row in rows:
        artifact_hash = development_byte_hashes[str(row["image_path"])]
        byte_archives = sorted(byte_to_archives.get(artifact_hash, set()))
        rgb_archives = sorted(
            rgb_to_archives.get(row["decoded_pixel_sha256"], set())
        )
        if byte_archives:
            byte_match_rows += 1
        if rgb_archives:
            rgb_match_rows += 1
        candidate_archives = sorted(set(byte_archives) | set(rgb_archives))
        unique_archive = candidate_archives[0] if len(candidate_archives) == 1 else None
        codex_id: int | None = None
        codex_title: str | None = None
        if unique_archive is None:
            status = "unmatched" if not candidate_archives else "ambiguous_archive"
        else:
            mapping = mapping_by_archive[unique_archive]
            codex_id = mapping["unique_codex_id"]
            codex_title = mapping["unique_codex_title"]
            if codex_id is None:
                status = "unique_archive_without_unique_codex"
            else:
                status = "unique_codex"
                unique_codex_ids.add(codex_id)
        status_counts[status] += 1
        output.append(
            {
                "row_id": row["row_id"],
                "cache_name": row["cache_name"],
                "class_label": row["class_label"],
                "class_name": row["class_name"],
                "component_id": row["component_id"],
                "development_byte_sha256": artifact_hash,
                "decoded_pixel_sha256": row["decoded_pixel_sha256"],
                "byte_match_archives": byte_archives,
                "rgb_match_archives": rgb_archives,
                "candidate_archives": candidate_archives,
                "unique_archive": unique_archive,
                "unique_codex_id": codex_id,
                "unique_codex_title": codex_title,
                "attribution_status": status,
            }
        )
    unique_rows = status_counts["unique_codex"]
    coverage = unique_rows / len(rows) if rows else 0.0
    return output, {
        "retained_rows": len(rows),
        "byte_match_rows": byte_match_rows,
        "rgb_match_rows": rgb_match_rows,
        "unique_codex_attributed_rows": unique_rows,
        "unique_codex_attributed_row_coverage": coverage,
        "unique_codex_ids": sorted(unique_codex_ids),
        "unique_codex_id_count": len(unique_codex_ids),
        "ambiguous_rows": status_counts["ambiguous_archive"],
        "unique_archive_without_unique_codex_rows": status_counts[
            "unique_archive_without_unique_codex"
        ],
        "unmatched_rows": status_counts["unmatched"],
        "status_counts": dict(sorted(status_counts.items())),
    }


def runtime_hashes() -> dict[str, str]:
    paths = {
        "projection_sha256": ROOT / "backend/codex_model/weights/projection.pt",
        "prototypes_sha256": ROOT / "backend/codex_model/weights/prototypes.pt",
        "config_sha256": ROOT / "backend/codex_model/config.json",
    }
    result = {key: sha256_file(path) for key, path in paths.items()}
    expected = {
        "projection_sha256": EXPECTED_RUNTIME_PROJECTION_SHA256,
        "prototypes_sha256": EXPECTED_RUNTIME_PROTOTYPES_SHA256,
        "config_sha256": EXPECTED_RUNTIME_CONFIG_SHA256,
    }
    if result != expected:
        raise ValueError(f"runtime hash mismatch: {result}")
    return result


def build_audit(
    *,
    contract: dict[str, Any],
    archive_hashes: dict[str, str],
    archive_summaries: Sequence[dict[str, Any]],
    archive_map: dict[str, Any],
    attribution_summary: dict[str, Any],
    input_validation: dict[str, Any],
    runtime_before: dict[str, str],
    runtime_after: dict[str, str],
) -> dict[str, Any]:
    spec = contract["spec"]
    total_entries = sum(row["non_directory_entries"] for row in archive_summaries)
    raster_success = sum(row["raster_decode_successes"] for row in archive_summaries)
    raster_failures = sum(row["raster_failures"] for row in archive_summaries)
    non_raster = sum(row["non_raster_entries"] for row in archive_summaries)
    coverage = attribution_summary["unique_codex_attributed_row_coverage"]
    codex_count = attribution_summary["unique_codex_id_count"]
    if coverage >= 0.5 and codex_count >= 20:
        decision_band = "material_recovery"
    elif coverage > 0:
        decision_band = "partial_recovery"
    else:
        decision_band = "no_recovery"
    return {
        "schema_version": "autoresearch-archive-attribution-v13.audit",
        "mission": "elements-baseline-replacement",
        "run_id": "20260803-archive-attribution-v13",
        "iteration": 1,
        "factor": "exact archive collision attribution",
        "contract_hashes": {
            "spec_sha256": contract["spec_sha256"],
            "evaluator_sha256": contract["evaluator_sha256"],
        },
        "archive_inputs": {
            "archive_count": len(archive_summaries),
            "archive_content_inventory_sha256": aggregate_archive_hash(
                archive_hashes
            ),
            "non_directory_entries": total_entries,
            "supported_raster_entries": raster_success + raster_failures,
            "raster_decode_successes": raster_success,
            "raster_failures": raster_failures,
            "non_raster_entries": non_raster,
            "oversized_entries": sum(
                row["oversized_entries"] for row in archive_summaries
            ),
            "archive_index_hashes": {
                row["archive_name"]: row["index_sha256"]
                for row in archive_summaries
            },
        },
        "archive_to_codex": {
            "uniquely_mapped_archives": archive_map["uniquely_mapped_archives"],
            "ambiguous_archives": archive_map["ambiguous_archives"],
            "unmapped_archives": archive_map["unmapped_archives"],
        },
        "input_validation": input_validation,
        "business_identity_coverage_before": 0.0,
        "attribution": attribution_summary,
        "hypothesis": {
            "minimum_coverage": 0.5,
            "minimum_codex_ids": 20,
            "supported": coverage >= 0.5 and codex_count >= 20,
        },
        "decision_band": decision_band,
        "scientific_boundary": {
            "archives_used_as_instrument": False,
            "archives_used_for_training": False,
            "v12_exclusions_modified": 0,
            "record_detail_pages_fetched": 0,
            "external_images_downloaded": 0,
            "model_predictions": 0,
            "holdout_or_final_test_read": False,
            "runtime_before": runtime_before,
            "runtime_after": runtime_after,
            "runtime_unchanged": runtime_before == runtime_after,
            "promotion_eligible": False,
        },
        "next_direction_requires_council": True,
        "decision": (
            "submit_material_provenance_recovery_for_separate_review"
            if decision_band == "material_recovery"
            else "preserve_collision_index_and_continue_external_sourcing"
        ),
        "locked_rules": spec["forbidden_actions"],
    }


def write_audit_outputs(
    output_dir: Path,
    *,
    archive_map: dict[str, Any],
    attribution_rows: Sequence[dict[str, Any]],
    audit: dict[str, Any],
) -> dict[str, str]:
    map_path = output_dir / "archive-to-codex-map.json"
    attribution_path = output_dir / "development-attribution.jsonl"
    audit_path = output_dir / "archive-attribution-audit.json"
    write_json(map_path, archive_map)
    write_jsonl(attribution_path, attribution_rows)
    write_json(audit_path, audit)
    return {
        "archive_map_path": str(map_path.resolve()),
        "archive_map_sha256": sha256_file(map_path),
        "attribution_path": str(attribution_path.resolve()),
        "attribution_sha256": sha256_file(attribution_path),
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
    archive_inputs = audit["archive_inputs"]
    attribution = audit["attribution"]
    boundary = audit["scientific_boundary"]
    replay_exact = all(
        first_outputs[key] == replay_outputs[key]
        for key in (
            "archive_map_sha256",
            "attribution_sha256",
            "audit_sha256",
        )
    )
    integrity_gates = {
        "archive_content_inventory_hash_matches": (
            archive_inputs["archive_content_inventory_sha256"]
            == EXPECTED_ARCHIVE_CONTENT_INVENTORY_SHA256
        ),
        "archive_count_equals_49": archive_inputs["archive_count"] == 49,
        "non_directory_entries_equals_47251": (
            archive_inputs["non_directory_entries"] == 47251
        ),
        "supported_rasters_accounted_for_equals_47050": (
            archive_inputs["supported_raster_entries"] == 47050
        ),
        "non_raster_entries_equals_201": (
            archive_inputs["non_raster_entries"] == 201
        ),
        "retained_rows_equals_9990": attribution["retained_rows"] == 9990,
        "quarantined_rows_equals_49": (
            audit["input_validation"]["quarantined_rows"] == 49
        ),
        "deterministic_replay_same_hashes": replay_exact,
        "v12_exclusions_modified_equals_0": (
            boundary["v12_exclusions_modified"] == 0
        ),
        "model_predictions_equals_0": boundary["model_predictions"] == 0,
        "holdout_or_final_test_unread": (
            boundary["holdout_or_final_test_read"] is False
        ),
        "runtime_unchanged": boundary["runtime_unchanged"],
    }
    execution_integrity_pass = all(integrity_gates.values())
    hypothesis_supported = audit["hypothesis"]["supported"]
    evaluation = {
        "schema_version": "autoresearch-archive-attribution-v13.evaluation",
        "mission": "elements-baseline-replacement",
        "run_id": "20260803-archive-attribution-v13",
        "iteration": 1,
        "pass": execution_integrity_pass and hypothesis_supported,
        "execution_integrity_pass": execution_integrity_pass,
        "hypothesis_supported": hypothesis_supported,
        "promotion_eligible": False,
        "business_identity_coverage_before": 0.0,
        "unique_codex_attributed_row_coverage": attribution[
            "unique_codex_attributed_row_coverage"
        ],
        "unique_codex_attributed_rows": attribution[
            "unique_codex_attributed_rows"
        ],
        "unique_codex_ids": attribution["unique_codex_ids"],
        "unique_codex_id_count": attribution["unique_codex_id_count"],
        "ambiguous_rows": attribution["ambiguous_rows"],
        "unmatched_rows": attribution["unmatched_rows"],
        "decision_band": audit["decision_band"],
        "final_test_read": False,
        "runtime_unchanged": boundary["runtime_unchanged"],
        "model_predictions": 0,
        "exclusions_modified": 0,
        "integrity_gates": integrity_gates,
        "artifacts": {"first": first_outputs, "replay": replay_outputs},
        "runner_sha256": sha256_file(Path(__file__)),
        "decision": audit["decision"],
        "next_direction_requires_council": True,
    }
    write_json(output_path, evaluation)
    return evaluation


def _validate_input_hashes(contract: dict[str, Any]) -> dict[str, Any]:
    inputs = contract["spec"]["inputs"]
    paths = {
        "v8_manifest_sha256": ROOT / inputs["v8_manifest"],
        "v8_audit_sha256": ROOT / inputs["v8_audit"],
        "codex_csv_sha256": Path(inputs["codex_csv"]),
        "elements_csv_sha256": Path(inputs["elements_csv"]),
        "glyph_elements_csv_sha256": Path(inputs["glyph_elements_csv"]),
    }
    result: dict[str, str] = {}
    for key, path in paths.items():
        actual = sha256_file(path)
        if actual != inputs[key]:
            raise ValueError(f"input hash mismatch for {key}: {actual}")
        result[key] = actual
    return result


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    contract = validate_contract()
    inputs = contract["spec"]["inputs"]
    input_hashes = _validate_input_hashes(contract)
    runtime_before = runtime_hashes()
    archive_paths = sorted(args.archive_dir.glob("t_*.zip"), key=lambda path: path.name)
    if len(archive_paths) != inputs["archive_count"]:
        raise ValueError(f"expected 49 archives, got {len(archive_paths)}")
    name_inventory = "".join(f"{path.name}\n" for path in archive_paths).encode()
    if sha256_bytes(name_inventory) != inputs["archive_name_inventory_sha256"]:
        raise ValueError("archive name inventory hash mismatch")

    archive_stats_before = {
        path.name: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in archive_paths
    }
    archive_hashes = hash_archives(archive_paths, workers=args.workers)
    content_inventory_hash = aggregate_archive_hash(archive_hashes)
    if content_inventory_hash != inputs["archive_content_inventory_sha256"]:
        raise ValueError(
            f"archive content inventory hash mismatch: {content_inventory_hash}"
        )
    summaries = index_all_archives(
        archive_paths,
        archive_hashes=archive_hashes,
        index_dir=args.index_dir,
        supported_extensions=set(inputs["supported_extensions"]),
        max_member_bytes=contract["spec"]["hash_contract"][
            "max_member_uncompressed_bytes"
        ],
        workers=args.workers,
        force=args.force,
    )

    codex_titles, cote_to_codex = load_codex_catalog(
        Path(inputs["codex_csv"]), Path(inputs["elements_csv"])
    )
    archive_map = build_archive_to_codex_map(
        summaries,
        index_dir=args.index_dir,
        codex_titles=codex_titles,
        cote_to_codex=cote_to_codex,
    )
    retained_rows, row_validation = load_retained_development_rows(
        ROOT / inputs["v8_manifest"], ROOT / inputs["v8_audit"]
    )
    development_hashes = hash_development_artifacts(
        retained_rows, workers=args.workers
    )
    byte_lookup, rgb_lookup = build_collision_lookups(
        summaries, index_dir=args.index_dir
    )
    attribution_rows, attribution_summary = build_attribution(
        retained_rows,
        development_byte_hashes=development_hashes,
        byte_to_archives=byte_lookup,
        rgb_to_archives=rgb_lookup,
        archive_map=archive_map,
    )
    archive_stats_after = {
        path.name: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in archive_paths
    }
    if archive_stats_before != archive_stats_after:
        raise ValueError("source archive size/mtime changed during read-only audit")
    runtime_after = runtime_hashes()
    input_validation = {
        **row_validation,
        **input_hashes,
        "archive_stats_unchanged": True,
        "archive_content_inventory_sha256": content_inventory_hash,
    }
    audit = build_audit(
        contract=contract,
        archive_hashes=archive_hashes,
        archive_summaries=summaries,
        archive_map=archive_map,
        attribution_summary=attribution_summary,
        input_validation=input_validation,
        runtime_before=runtime_before,
        runtime_after=runtime_after,
    )
    first_outputs = write_audit_outputs(
        args.output_dir,
        archive_map=archive_map,
        attribution_rows=attribution_rows,
        audit=audit,
    )
    replay_outputs = write_audit_outputs(
        args.replay_dir,
        archive_map=archive_map,
        attribution_rows=attribution_rows,
        audit=audit,
    )
    return evaluate(
        audit=audit,
        first_outputs=first_outputs,
        replay_outputs=replay_outputs,
        output_path=args.evaluation_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("all",))
    parser.add_argument("--archive-dir", type=Path, default=Path("/mnt/f/CODEX"))
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    parser.add_argument(
        "--evaluation-path", type=Path, default=DEFAULT_EVALUATION_PATH
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.command == "all":
        evaluation = run_all(args)
        print(json.dumps(evaluation, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if evaluation["execution_integrity_pass"] else 2
    raise AssertionError(args.command)


if __name__ == "__main__":
    sys.exit(main())
