#!/usr/bin/env python3
"""Read-only audit for external CODEx training corpora.

The script intentionally does **not** import, copy, convert, or mutate images.
It inventories candidate element-crop folders, maps their class folders against the
active runtime classifier taxonomy, hashes image bytes and decoded RGB pixels for
both exact and re-encoded duplicate detection, and reports whether the corpus is
feasible for the current episodic training settings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
DEFAULT_SOURCE_PATHS = (
    "AI_clinic_class/Main_Elements",
    "AI_clinic_class/MainElem_Original",
    "Clinic-Codex/data/Elements",
)
VALID_CANONICAL_RE = re.compile(r"^[A-Za-z0-9_-]+$")
NUMERIC_PREFIX_RE = re.compile(r"^\d+[-_](.+)$")
ELEMENT_SUFFIX_RE = re.compile(r"(.+?)(?:[-_]element)$", re.IGNORECASE)
AUDIT_SCHEMA_VERSION = "external-corpus-audit.v2"


@dataclass(frozen=True)
class SourceSpec:
    name: str
    path: Path
    task: str = "element_crop_single_label"


@dataclass
class ImageInventoryRecord:
    source: str
    source_path: str
    class_dir: str
    normalized_name: str
    match_status: str
    matched_active_class: str | None
    path: str
    size_bytes: int
    extension: str
    sha256_bytes: str | None
    sha256_pixels: str | None
    width: int | None
    height: int | None
    mode: str | None
    warnings: list[str] = field(default_factory=list)


@dataclass
class ClassAudit:
    source: str
    source_path: str
    class_dir: str
    normalized_name: str
    match_status: str
    matched_active_class: str | None
    image_count: int
    extensions: dict[str, int]
    duplicate_image_count: int = 0
    duplicate_hashes: list[str] = field(default_factory=list)
    pixel_duplicate_image_count: int = 0
    pixel_duplicate_hashes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class SourceAudit:
    name: str
    path: str
    exists: bool
    task: str
    class_count: int = 0
    image_count: int = 0
    matched_class_count: int = 0
    unmatched_class_count: int = 0
    classes: list[ClassAudit] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class PreflightClass:
    class_name: str
    image_count: int
    train_count: int
    val_count: int
    status: str
    reasons: list[str]


@dataclass
class PreflightAudit:
    status: str
    k_shot: int
    q_queries: int
    min_images_per_class: int
    val_fraction: float
    n_way: int
    matched_class_count: int
    active_class_count: int
    covered_active_class_count: int
    missing_active_class_count: int
    promotable_against_active: bool
    blocking_reasons: list[str]
    class_results: list[PreflightClass]


@dataclass
class AuditReport:
    schema_version: str
    audit_run_id: str
    active_config: str
    active_config_sha256: str
    active_class_count: int
    inventory_count: int
    pixel_decode_failure_count: int
    pixel_decode_failures: list[str]
    sources: list[SourceAudit]
    duplicate_groups: list[dict]
    pixel_duplicate_groups: list[dict]
    aggregate_counts_by_active_class: dict[str, int]
    preflight: PreflightAudit
    warnings: list[str]


def strip_diacritics(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_class_name(raw_name: str) -> str:
    name = raw_name.strip()
    prefix_match = NUMERIC_PREFIX_RE.match(name)
    if prefix_match:
        name = prefix_match.group(1)
    suffix_match = ELEMENT_SUFFIX_RE.match(name)
    if suffix_match:
        name = suffix_match.group(1)
    name = strip_diacritics(name).lower()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^a-z0-9_-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_-")
    return name


def classify_match(raw_name: str, active_by_exact: set[str], active_by_normalized: dict[str, list[str]]) -> tuple[str, str | None]:
    if raw_name in active_by_exact:
        return "exact", raw_name

    without_prefix = NUMERIC_PREFIX_RE.sub(r"\1", raw_name)
    without_suffix = ELEMENT_SUFFIX_RE.sub(r"\1", raw_name)
    without_both = ELEMENT_SUFFIX_RE.sub(r"\1", without_prefix)

    candidates: list[tuple[str, str]] = []
    if without_prefix != raw_name and without_both != without_prefix:
        candidates.append(("strip_prefix_suffix", without_both))
    if without_prefix != raw_name:
        candidates.append(("strip_prefix", without_prefix))
    if without_suffix != raw_name:
        candidates.append(("strip_suffix", without_suffix))

    for status, candidate in candidates:
        if candidate in active_by_exact:
            return status, candidate

    normalized = normalize_class_name(raw_name)
    matches = active_by_normalized.get(normalized, [])
    if len(matches) == 1:
        return "normalized", matches[0]
    if len(matches) > 1:
        return "ambiguous_normalized", None
    return "unmatched", None


def load_active_classes(config_path: Path) -> list[str]:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    classes = data.get("class_names")
    if not isinstance(classes, list) or not all(isinstance(item, str) for item in classes):
        raise ValueError(f"{config_path} does not contain a string list at class_names")
    return classes


def iter_image_files(class_dir: Path) -> Iterable[Path]:
    for child in sorted(class_dir.iterdir()):
        if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS:
            yield child


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def sha256_pixels(path: Path) -> tuple[str | None, int | None, int | None, str | None, str | None]:
    try:
        from PIL import Image, ImageOps
    except Exception as exc:  # pragma: no cover - depends on optional env
        return None, None, None, None, f"pillow_unavailable:{exc.__class__.__name__}"
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            mode = image.mode
            rgb = image.convert("RGB")
            width, height = rgb.size
            h = hashlib.sha256()
            h.update(f"RGB:{width}x{height}:".encode("ascii"))
            h.update(rgb.tobytes())
            return h.hexdigest(), width, height, mode, None
    except Exception as exc:
        return None, None, None, None, f"pixel_hash_failed:{exc.__class__.__name__}"


def build_default_sources(root: Path) -> list[SourceSpec]:
    return [SourceSpec(name=rel.replace("/", "__"), path=root / rel) for rel in DEFAULT_SOURCE_PATHS]


def parse_source_arg(value: str) -> SourceSpec:
    if "=" in value:
        name, path = value.split("=", 1)
        return SourceSpec(name=name.strip() or Path(path).name, path=Path(path).expanduser())
    path = Path(value).expanduser()
    return SourceSpec(name=path.name, path=path)


def duplicate_entries(records: list[ImageInventoryRecord], key: str) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        digest = getattr(record, key)
        if not digest or record.size_bytes <= 0:
            continue
        groups[digest].append(
            {
                "source": record.source,
                "class_dir": record.class_dir,
                "matched_active_class": record.matched_active_class,
                "path": record.path,
            }
        )

    output = []
    for digest, entries in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        if len(entries) <= 1:
            continue
        active_classes = sorted({entry["matched_active_class"] for entry in entries if entry["matched_active_class"]})
        sources = sorted({entry["source"] for entry in entries})
        if len(active_classes) > 1:
            conflict_type = "cross_class"
        elif len(sources) > 1:
            conflict_type = "cross_source"
        else:
            conflict_type = "same_class"
        output.append(
            {
                key: digest,
                "count": len(entries),
                "conflict_type": conflict_type,
                "active_classes": active_classes,
                "sources": sources,
                "entries": entries,
            }
        )
    return output


def audit_sources(
    sources: list[SourceSpec],
    active_classes: list[str],
    hash_files: bool = True,
    pixel_hash_files: bool = True,
) -> tuple[list[SourceAudit], list[dict], list[dict], dict[str, int], list[ImageInventoryRecord]]:
    active_exact = set(active_classes)
    active_normalized: dict[str, list[str]] = defaultdict(list)
    for class_name in active_classes:
        active_normalized[normalize_class_name(class_name)].append(class_name)

    inventory: list[ImageInventoryRecord] = []
    aggregate_counts: Counter[str] = Counter()
    source_reports: list[SourceAudit] = []

    for spec in sources:
        source = SourceAudit(name=spec.name, path=str(spec.path), exists=spec.path.exists(), task=spec.task)
        if not spec.path.exists():
            source.warnings.append("source_path_missing")
            source_reports.append(source)
            continue
        if not spec.path.is_dir():
            source.warnings.append("source_path_not_directory")
            source_reports.append(source)
            continue

        class_dirs = sorted([path for path in spec.path.iterdir() if path.is_dir()], key=lambda p: p.name)
        for class_dir in class_dirs:
            image_files = list(iter_image_files(class_dir))
            if not image_files:
                continue

            extensions = Counter(path.suffix.lower() for path in image_files)
            match_status, matched_active = classify_match(class_dir.name, active_exact, active_normalized)
            warnings: list[str] = []
            normalized_name = normalize_class_name(class_dir.name)

            if not VALID_CANONICAL_RE.match(normalized_name):
                warnings.append("normalized_name_invalid_chars")
            if normalized_name in {"", "?", "??"} or normalized_name.startswith("?"):
                warnings.append("noisy_or_unknown_label")
            if match_status.startswith("ambiguous"):
                warnings.append("ambiguous_mapping_requires_manual_review")
            if match_status == "unmatched":
                warnings.append("unmatched_active_taxonomy")
            if any(ext != ".bmp" for ext in extensions):
                warnings.append("non_bmp_images_present")

            class_audit = ClassAudit(
                source=spec.name,
                source_path=str(spec.path),
                class_dir=class_dir.name,
                normalized_name=normalized_name,
                match_status=match_status,
                matched_active_class=matched_active,
                image_count=len(image_files),
                extensions=dict(sorted(extensions.items())),
                warnings=list(warnings),
            )

            for img_path in image_files:
                record_warnings = list(warnings)
                size_bytes = img_path.stat().st_size
                if size_bytes <= 0:
                    record_warnings.append("empty_file")
                sha_bytes = sha256_file(img_path) if hash_files else None
                sha_pixels, width, height, mode, pixel_warning = sha256_pixels(img_path) if pixel_hash_files else (None, None, None, None, None)
                if pixel_warning:
                    record_warnings.append(pixel_warning)
                inventory.append(
                    ImageInventoryRecord(
                        source=spec.name,
                        source_path=str(spec.path),
                        class_dir=class_dir.name,
                        normalized_name=normalized_name,
                        match_status=match_status,
                        matched_active_class=matched_active,
                        path=str(img_path),
                        size_bytes=size_bytes,
                        extension=img_path.suffix.lower(),
                        sha256_bytes=sha_bytes,
                        sha256_pixels=sha_pixels,
                        width=width,
                        height=height,
                        mode=mode,
                        warnings=record_warnings,
                    )
                )

            if matched_active:
                aggregate_counts[matched_active] += len(image_files)

            source.classes.append(class_audit)
            source.image_count += len(image_files)

        source.class_count = len(source.classes)
        source.matched_class_count = sum(1 for item in source.classes if item.matched_active_class)
        source.unmatched_class_count = source.class_count - source.matched_class_count
        if source.unmatched_class_count:
            source.warnings.append("unmatched_classes_present")
        source_reports.append(source)

    duplicate_groups = duplicate_entries(inventory, "sha256_bytes")
    pixel_duplicate_groups = duplicate_entries(inventory, "sha256_pixels")

    byte_by_class: Counter[tuple[str, str]] = Counter()
    byte_hashes_by_class: dict[tuple[str, str], set[str]] = defaultdict(set)
    pixel_by_class: Counter[tuple[str, str]] = Counter()
    pixel_hashes_by_class: dict[tuple[str, str], set[str]] = defaultdict(set)
    for group in duplicate_groups:
        digest = group["sha256_bytes"]
        for entry in group["entries"]:
            key = (entry["source"], entry["class_dir"])
            byte_by_class[key] += 1
            byte_hashes_by_class[key].add(digest)
    for group in pixel_duplicate_groups:
        digest = group["sha256_pixels"]
        for entry in group["entries"]:
            key = (entry["source"], entry["class_dir"])
            pixel_by_class[key] += 1
            pixel_hashes_by_class[key].add(digest)

    for source in source_reports:
        for class_audit in source.classes:
            key = (source.name, class_audit.class_dir)
            class_audit.duplicate_image_count = byte_by_class.get(key, 0)
            class_audit.duplicate_hashes = sorted(byte_hashes_by_class.get(key, set()))
            class_audit.pixel_duplicate_image_count = pixel_by_class.get(key, 0)
            class_audit.pixel_duplicate_hashes = sorted(pixel_hashes_by_class.get(key, set()))
            if class_audit.duplicate_image_count:
                class_audit.warnings.append("duplicate_images_present")
            if class_audit.pixel_duplicate_image_count:
                class_audit.warnings.append("pixel_duplicate_images_present")

    return source_reports, duplicate_groups, pixel_duplicate_groups, dict(sorted(aggregate_counts.items())), inventory


def deterministic_split_counts(count: int, val_fraction: float) -> tuple[int, int]:
    if count <= 0:
        return 0, 0
    n_val = max(1, int(count * val_fraction))
    n_val = min(n_val, count - 1)
    return count - n_val, n_val


def evaluate_preflight(
    aggregate_counts: dict[str, int],
    active_classes: list[str],
    *,
    k_shot: int,
    q_queries: int,
    min_images_per_class: int,
    val_fraction: float,
    n_way: int,
    require_full_active_coverage: bool = True,
) -> PreflightAudit:
    class_results: list[PreflightClass] = []
    blocking_reasons: list[str] = []
    covered = [class_name for class_name in active_classes if aggregate_counts.get(class_name, 0) > 0]
    missing = [class_name for class_name in active_classes if aggregate_counts.get(class_name, 0) <= 0]
    min_required_episode = k_shot + q_queries

    for class_name in sorted(aggregate_counts):
        count = aggregate_counts[class_name]
        train_count, val_count = deterministic_split_counts(count, val_fraction)
        reasons: list[str] = []
        if count < min_images_per_class:
            reasons.append("below_min_images_per_class")
        if count < min_required_episode:
            reasons.append("below_k_shot_plus_q_queries_for_full_episode")
        if train_count < k_shot:
            reasons.append("train_split_below_k_shot")
        if val_count < 1:
            reasons.append("val_split_empty")
        class_results.append(
            PreflightClass(
                class_name=class_name,
                image_count=count,
                train_count=train_count,
                val_count=val_count,
                status="pass" if not reasons else "fail",
                reasons=reasons,
            )
        )

    passing_episode_classes = [item.class_name for item in class_results if item.image_count >= min_required_episode]
    if len(passing_episode_classes) < n_way:
        blocking_reasons.append(f"insufficient_episode_classes:{len(passing_episode_classes)}<{n_way}")
    if require_full_active_coverage and missing:
        blocking_reasons.append(f"active_taxonomy_not_fully_covered:{len(covered)}/{len(active_classes)}")
    failed_classes = [item for item in class_results if item.status == "fail"]
    if failed_classes:
        blocking_reasons.append(f"classes_failing_preflight:{len(failed_classes)}")

    promotable = not blocking_reasons and len(covered) == len(active_classes)
    return PreflightAudit(
        status="pass" if not blocking_reasons else "fail",
        k_shot=k_shot,
        q_queries=q_queries,
        min_images_per_class=min_images_per_class,
        val_fraction=val_fraction,
        n_way=n_way,
        matched_class_count=len(aggregate_counts),
        active_class_count=len(active_classes),
        covered_active_class_count=len(covered),
        missing_active_class_count=len(missing),
        promotable_against_active=promotable,
        blocking_reasons=blocking_reasons,
        class_results=class_results,
    )


def build_report(
    active_config: Path,
    sources: list[SourceSpec],
    *,
    hash_files: bool,
    pixel_hash_files: bool,
    k_shot: int,
    q_queries: int,
    min_images_per_class: int,
    val_fraction: float,
    n_way: int,
    require_full_active_coverage: bool,
) -> tuple[AuditReport, list[ImageInventoryRecord]]:
    active_classes = load_active_classes(active_config)
    source_reports, duplicate_groups, pixel_duplicate_groups, aggregate_counts, inventory = audit_sources(
        sources, active_classes, hash_files=hash_files, pixel_hash_files=pixel_hash_files
    )
    preflight = evaluate_preflight(
        aggregate_counts,
        active_classes,
        k_shot=k_shot,
        q_queries=q_queries,
        min_images_per_class=min_images_per_class,
        val_fraction=val_fraction,
        n_way=n_way,
        require_full_active_coverage=require_full_active_coverage,
    )
    pixel_failures = [record.path for record in inventory if record.sha256_pixels is None]
    audit_run_id = canonical_json_sha(
        {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "active_config_sha256": sha256_file(active_config),
            "images": [
                {
                    "source": record.source,
                    "path": record.path,
                    "sha256_bytes": record.sha256_bytes,
                    "sha256_pixels": record.sha256_pixels,
                }
                for record in sorted(inventory, key=lambda item: (item.source, item.class_dir, item.path))
            ],
        }
    )
    warnings: list[str] = []
    if duplicate_groups:
        warnings.append("duplicate_hashes_detected")
    if pixel_duplicate_groups:
        warnings.append("pixel_duplicate_hashes_detected")
    if preflight.missing_active_class_count:
        warnings.append("active_taxonomy_missing_classes")
    if any(source.unmatched_class_count for source in source_reports):
        warnings.append("unmatched_source_classes")
    if pixel_failures:
        warnings.append(f"pixel_decode_failures:{len(pixel_failures)}")
    return AuditReport(
        schema_version=AUDIT_SCHEMA_VERSION,
        audit_run_id=audit_run_id,
        active_config=str(active_config),
        active_config_sha256=sha256_file(active_config),
        active_class_count=len(active_classes),
        inventory_count=len(inventory),
        pixel_decode_failure_count=len(pixel_failures),
        pixel_decode_failures=pixel_failures,
        sources=source_reports,
        duplicate_groups=duplicate_groups,
        pixel_duplicate_groups=pixel_duplicate_groups,
        aggregate_counts_by_active_class=aggregate_counts,
        preflight=preflight,
        warnings=warnings,
    ), inventory


def report_to_markdown(report: AuditReport) -> str:
    lines = [
        "# External corpus audit",
        "",
        f"Schema: `{report.schema_version}`",
        f"Audit run id: `{report.audit_run_id}`",
        f"Active config: `{report.active_config}`",
        f"Active config sha256: `{report.active_config_sha256}`",
        f"Active classes: **{report.active_class_count}**",
        f"Inventoried images: **{report.inventory_count}**",
        f"Pixel decode failures: **{report.pixel_decode_failure_count}**",
        f"Preflight: **{report.preflight.status}**",
        f"Promotable against active taxonomy: **{report.preflight.promotable_against_active}**",
        "",
        "## Blocking reasons",
    ]
    lines.extend(f"- `{reason}`" for reason in report.preflight.blocking_reasons) if report.preflight.blocking_reasons else lines.append("- none")

    lines.extend(["", "## Sources", "", "| Source | Exists | Classes | Images | Matched | Unmatched | Warnings |", "|---|---:|---:|---:|---:|---:|---|"])
    for source in report.sources:
        lines.append(
            f"| `{source.name}` | {source.exists} | {source.class_count} | {source.image_count} | "
            f"{source.matched_class_count} | {source.unmatched_class_count} | {', '.join(source.warnings) or '-'} |"
        )

    lines.extend(["", "## Class mapping summary", "", "| Source | Class dir | Normalized | Match | Active class | Count | Ext | Warnings |", "|---|---|---|---|---|---:|---|---|"])
    for source in report.sources:
        for item in source.classes:
            lines.append(
                f"| `{item.source}` | `{item.class_dir}` | `{item.normalized_name}` | `{item.match_status}` | "
                f"`{item.matched_active_class or ''}` | {item.image_count} | `{json.dumps(item.extensions, sort_keys=True)}` | "
                f"{', '.join(item.warnings) or '-'} |"
            )

    lines.extend(["", "## Aggregate active-class counts", "", "| Active class | Images |", "|---|---:|"])
    for class_name, count in sorted(report.aggregate_counts_by_active_class.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{class_name}` | {count} |")

    lines.extend(["", "## Preflight class results", "", "| Class | Images | Train | Val | Status | Reasons |", "|---|---:|---:|---:|---|---|"])
    for item in sorted(report.preflight.class_results, key=lambda value: (value.status, value.class_name)):
        lines.append(f"| `{item.class_name}` | {item.image_count} | {item.train_count} | {item.val_count} | {item.status} | {', '.join(item.reasons) or '-'} |")

    lines.extend(["", "## Duplicate hash groups", ""])
    lines.append(f"Byte-identical groups: **{len(report.duplicate_groups)}**. Pixel-identical groups: **{len(report.pixel_duplicate_groups)}**. Full entries are stored in JSON; Markdown shows group counts only.")
    for group in report.duplicate_groups[:50]:
        lines.append(f"- bytes `{group['sha256_bytes']}`: {group['count']} files")
    for group in report.pixel_duplicate_groups[:50]:
        lines.append(f"- pixels `{group['sha256_pixels']}`: {group['count']} files")
    return "\n".join(lines) + "\n"


def inventory_payload(records: list[ImageInventoryRecord], report: AuditReport) -> dict:
    return {
        "schema_version": "external-corpus-inventory.v1",
        "audit_schema_version": report.schema_version,
        "audit_run_id": report.audit_run_id,
        "active_config": report.active_config,
        "active_config_sha256": report.active_config_sha256,
        "image_count": len(records),
        "images": [asdict(record) for record in sorted(records, key=lambda r: (r.source, r.class_dir, r.path))],
    }


def write_outputs(report: AuditReport, inventory: list[ImageInventoryRecord], json_output: Path | None, markdown_output: Path | None, inventory_output: Path | None) -> None:
    if json_output:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if markdown_output:
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        markdown_output.write_text(report_to_markdown(report), encoding="utf-8")
    if inventory_output:
        inventory_output.parent.mkdir(parents=True, exist_ok=True)
        inventory_output.write_text(json.dumps(inventory_payload(inventory, report), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/mnt/f/CODEX", help="External CODEX root used for default sources")
    parser.add_argument("--active-config", default="backend/codex_model/config.json", help="Runtime model config with class_names")
    parser.add_argument("--source", action="append", default=[], help="Source folder to audit, optionally name=/path. Repeatable.")
    parser.add_argument("--json-output", help="Write full JSON report to this path")
    parser.add_argument("--markdown-output", help="Write human-readable Markdown report to this path")
    parser.add_argument("--inventory-output", help="Write complete image-level inventory JSON to this path")
    parser.add_argument("--no-hash", action="store_true", help="Skip byte SHA256 duplicate detection")
    parser.add_argument("--no-pixel-hash", action="store_true", help="Skip decoded RGB pixel SHA256 duplicate detection")
    parser.add_argument("--k-shot", type=int, default=3)
    parser.add_argument("--q-queries", type=int, default=5)
    parser.add_argument("--min-images-per-class", type=int, default=2)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--n-way", type=int, default=20)
    parser.add_argument("--allow-partial-active-coverage", action="store_true", help="Do not fail preflight solely because source corpus lacks all active classes.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    active_config = Path(args.active_config).expanduser()
    if not active_config.exists():
        print(f"ERROR: active config not found: {active_config}", file=sys.stderr)
        return 2

    sources = [parse_source_arg(value) for value in args.source] if args.source else build_default_sources(Path(args.root).expanduser())
    report, inventory = build_report(
        active_config,
        sources,
        hash_files=not args.no_hash,
        pixel_hash_files=not args.no_pixel_hash,
        k_shot=args.k_shot,
        q_queries=args.q_queries,
        min_images_per_class=args.min_images_per_class,
        val_fraction=args.val_fraction,
        n_way=args.n_way,
        require_full_active_coverage=not args.allow_partial_active_coverage,
    )
    write_outputs(
        report,
        inventory,
        Path(args.json_output).expanduser() if args.json_output else None,
        Path(args.markdown_output).expanduser() if args.markdown_output else None,
        Path(args.inventory_output).expanduser() if args.inventory_output else None,
    )
    print(report_to_markdown(report))
    return 0 if report.preflight.status == "pass" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
