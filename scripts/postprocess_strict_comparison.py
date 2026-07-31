#!/usr/bin/env python3
"""Build the conservative unseen report from a completed model comparison."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from compare_models_unseen import (
    draw_glyph_overview,
    draw_labelled_examples,
    draw_per_class_comparison,
    json_dump,
    save_per_class_csv,
)


def legacy_source_stems(legacy_elements_root: Path) -> set[str]:
    return {
        path.stem.casefold()
        for path in legacy_elements_root.rglob("*")
        if path.is_file()
    }


def filter_strict_holdout(
    rows: list[dict[str, Any]], legacy_stems: set[str]
) -> tuple[list[dict[str, Any]], int]:
    strict: list[dict[str, Any]] = []
    excluded = 0
    for row in rows:
        source_stem = Path(row["source_path"]).stem.casefold()
        if source_stem in legacy_stems:
            excluded += 1
        else:
            strict.append(row)
    return strict, excluded


def exact_mcnemar_p(candidate_only: int, historical_only: int) -> float:
    discordant = candidate_only + historical_only
    if discordant == 0:
        return 1.0
    smaller = min(candidate_only, historical_only)
    tail = sum(math.comb(discordant, index) for index in range(smaller + 1))
    return min(1.0, 2 * tail / (2**discordant))


def metrics(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    count = len(rows)
    if not count:
        raise ValueError("strict holdout is empty")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["class_name"]].append(row)
    per_class: list[dict[str, Any]] = []
    for class_name, class_rows in sorted(grouped.items()):
        class_count = len(class_rows)
        historical_top1 = sum(
            row["historical_top1_correct"] for row in class_rows
        ) / class_count
        candidate_top1 = sum(
            row["candidate_top1_correct"] for row in class_rows
        ) / class_count
        per_class.append(
            {
                "class_name": class_name,
                "count": class_count,
                "historical_top1": historical_top1,
                "candidate_top1": candidate_top1,
                "delta_candidate_minus_historical": (
                    candidate_top1 - historical_top1
                ),
                "historical_top3": sum(
                    row["historical_top3_correct"] for row in class_rows
                )
                / class_count,
                "candidate_top3": sum(
                    row["candidate_top3_correct"] for row in class_rows
                )
                / class_count,
                "top1_agreement": sum(
                    row["top1_agreement"] for row in class_rows
                )
                / class_count,
            }
        )

    candidate_only = sum(
        row["candidate_top1_correct"] and not row["historical_top1_correct"]
        for row in rows
    )
    historical_only = sum(
        row["historical_top1_correct"] and not row["candidate_top1_correct"]
        for row in rows
    )
    summary = {
        "count": count,
        "class_count": len(per_class),
        "historical_top1": sum(
            row["historical_top1_correct"] for row in rows
        )
        / count,
        "candidate_top1": sum(row["candidate_top1_correct"] for row in rows)
        / count,
        "historical_top3": sum(
            row["historical_top3_correct"] for row in rows
        )
        / count,
        "candidate_top3": sum(row["candidate_top3_correct"] for row in rows)
        / count,
        "historical_macro_top1": sum(
            row["historical_top1"] for row in per_class
        )
        / len(per_class),
        "candidate_macro_top1": sum(
            row["candidate_top1"] for row in per_class
        )
        / len(per_class),
        "top1_agreement": sum(row["top1_agreement"] for row in rows) / count,
        "candidate_only_correct": candidate_only,
        "historical_only_correct": historical_only,
        "both_wrong": sum(
            not row["historical_top1_correct"]
            and not row["candidate_top1_correct"]
            for row in rows
        ),
        "historical_rejected_at_0_35": sum(
            row["historical_top_scores"][0] < 0.35 for row in rows
        ),
        "candidate_rejected_at_0_35": sum(
            row["candidate_top_scores"][0] < 0.35 for row in rows
        ),
        "historical_mean_confidence": sum(
            row["historical_top_scores"][0] for row in rows
        )
        / count,
        "candidate_mean_confidence": sum(
            row["candidate_top_scores"][0] for row in rows
        )
        / count,
        "exact_mcnemar_two_sided_p": exact_mcnemar_p(
            candidate_only, historical_only
        ),
    }
    return summary, per_class


def representative_glyphs(glyphs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for glyph in glyphs:
        if glyph["glyph_class"] not in seen:
            selected.append(glyph)
            seen.add(glyph["glyph_class"])
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--legacy-elements-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    element_rows = json.loads(
        (args.comparison_dir / "element-results.json").read_text(encoding="utf-8")
    )
    glyph_rows = json.loads(
        (args.comparison_dir / "glyph-results.json").read_text(encoding="utf-8")
    )
    original_report = json.loads(
        (args.comparison_dir / "comparison-report.json").read_text(encoding="utf-8")
    )
    stems = legacy_source_stems(args.legacy_elements_root)
    strict_rows, excluded_by_stem = filter_strict_holdout(element_rows, stems)
    summary, per_class = metrics(strict_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_dump(args.output_dir / "element-results.json", strict_rows)
    json_dump(args.output_dir / "glyph-results.json", glyph_rows)
    save_per_class_csv(args.output_dir / "per-class-elements.csv", per_class)
    draw_labelled_examples(
        args.output_dir / "labelled-elements-comparison.png", strict_rows
    )
    draw_per_class_comparison(
        args.output_dir / "per-class-elements-comparison.png", per_class
    )
    draw_glyph_overview(
        args.output_dir / "glyph-comparison-overview.png",
        representative_glyphs(glyph_rows),
    )

    report = {
        **original_report,
        "schema_version": "dual-model-strict-unseen-comparison.v1",
        "holdout_audit": {
            **original_report["holdout_audit"],
            "excluded_additional_source_stem_duplicates": excluded_by_stem,
            "strict_holdout_count": len(strict_rows),
            "strict_holdout_class_count": len(per_class),
            "legacy_unique_source_stem_count": len(stems),
            "strict_rule": (
                "exclude decoded-pixel duplicates and any source basename "
                "already present in legacy Elements"
            ),
        },
        "labelled_elements": summary,
    }
    json_dump(args.output_dir / "comparison-report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
