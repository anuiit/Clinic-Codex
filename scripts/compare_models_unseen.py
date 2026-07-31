#!/usr/bin/env python3
"""Compare the deployed and candidate classifiers on genuinely held-out images.

Two complementary protocols are produced:

1. A labelled Elements holdout. Images whose decoded RGB pixel hash occurs in
   the legacy Elements cache are excluded before evaluation.
2. A Glyph integration sample. MobileSAM segments each compound glyph once and
   the exact same crops are classified by both models.

The Glyph protocol measures agreement and weak folder-label retrieval, not
segment accuracy: no segment-level ground truth is available in the archive.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from codex_model.classifier import _ProjectionHead, _preprocess_image  # noqa: E402


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
MODEL_COLORS = {
    "historical": (39, 104, 174),
    "candidate": (44, 160, 101),
    "agree": (34, 139, 94),
    "disagree": (205, 64, 69),
    "neutral": (90, 90, 90),
}


@dataclass(frozen=True)
class RuntimeWeights:
    name: str
    projection: _ProjectionHead
    prototypes: torch.Tensor
    labels: list[int]
    class_names: dict[int, str]


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_pixels(path: Path) -> str:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        digest = hashlib.sha256()
        digest.update(f"RGB:{width}x{height}:".encode("ascii"))
        digest.update(image.tobytes())
        return digest.hexdigest()


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
        if bold
        else Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf")
        if bold
        else Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def load_runtime(name: str, weights_dir: Path, device: torch.device) -> RuntimeWeights:
    projection_state = torch.load(
        weights_dir / "projection.pt", map_location="cpu", weights_only=False
    )
    prototype_payload = torch.load(
        weights_dir / "prototypes.pt", map_location="cpu", weights_only=False
    )
    projection = _ProjectionHead(384, 128).to(device)
    projection.load_state_dict(projection_state)
    projection.eval()

    class_names = {
        int(label): str(class_name)
        for label, class_name in prototype_payload["class_names"].items()
    }
    labels = sorted(class_names)
    if "class_labels" in prototype_payload:
        payload_labels = [int(value) for value in prototype_payload["class_labels"].tolist()]
        if payload_labels != labels:
            raise ValueError(f"{name}: prototype class_labels do not match sorted class_names")
    prototypes = prototype_payload["prototypes"].float().to(device)
    if prototypes.shape != (len(labels), 128):
        raise ValueError(f"{name}: unexpected prototype shape {tuple(prototypes.shape)}")
    return RuntimeWeights(name, projection, prototypes, labels, class_names)


def validate_runtime_contract(a: RuntimeWeights, b: RuntimeWeights) -> None:
    if a.labels != b.labels or a.class_names != b.class_names:
        raise ValueError("historical and candidate runtime taxonomies differ")


def classify_features(
    features: torch.Tensor,
    runtime: RuntimeWeights,
    top_k: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    embeddings = runtime.projection(features)
    similarities = embeddings @ runtime.prototypes.t()
    values, positions = similarities.topk(min(top_k, similarities.shape[1]), dim=1)
    label_tensor = torch.tensor(runtime.labels, device=positions.device, dtype=torch.long)
    return label_tensor[positions], values


def batched(iterable: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(iterable), size):
        yield iterable[start : start + size]


def legacy_pixel_hashes(cache_path: Path) -> tuple[set[str], int]:
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    image_paths = cache.get("image_paths")
    if image_paths is None:
        raise ValueError(f"{cache_path} does not expose image_paths")
    unique_paths = sorted({str(path) for path in image_paths})
    hashes = {sha256_pixels(Path(path)) for path in unique_paths}
    return hashes, len(unique_paths)


def build_external_holdout(
    snapshot_path: Path,
    legacy_hashes: set[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    rows = snapshot["rows"]
    holdout: list[dict[str, Any]] = []
    excluded = 0
    missing = 0
    for row in rows:
        output_path = Path(row["output_path"])
        if not output_path.is_file():
            missing += 1
            continue
        pixel_hash = row.get("source_pixel_sha256") or sha256_pixels(output_path)
        if pixel_hash in legacy_hashes:
            excluded += 1
            continue
        holdout.append(
            {
                "path": str(output_path),
                "class_label": int(row["class_label"]),
                "class_name": str(row["class_name"]),
                "source_pixel_sha256": pixel_hash,
                "source_path": str(row.get("source_path", "")),
            }
        )
    return holdout, {
        "snapshot_count": len(rows),
        "excluded_legacy_pixel_duplicates": excluded,
        "missing_files": missing,
        "holdout_count": len(holdout),
    }


def evaluate_labelled_elements(
    rows: list[dict[str, Any]],
    backbone: torch.nn.Module,
    historical: RuntimeWeights,
    candidate: RuntimeWeights,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in batched(rows, batch_size):
            images = [
                _preprocess_image(Image.open(row["path"]).convert("RGB"), 224)
                for row in batch
            ]
            tensors = torch.cat(images, dim=0).to(device)
            features = backbone(tensors)
            old_labels, old_scores = classify_features(features, historical)
            new_labels, new_scores = classify_features(features, candidate)
            for index, row in enumerate(batch):
                truth = int(candidate.labels[int(row["class_label"])])
                old_top = [int(value) for value in old_labels[index].tolist()]
                new_top = [int(value) for value in new_labels[index].tolist()]
                results.append(
                    {
                        **row,
                        "truth_runtime_label": truth,
                        "historical_top_labels": old_top,
                        "historical_top_names": [
                            historical.class_names[label] for label in old_top
                        ],
                        "historical_top_scores": [
                            float(value) for value in old_scores[index].tolist()
                        ],
                        "candidate_top_labels": new_top,
                        "candidate_top_names": [
                            candidate.class_names[label] for label in new_top
                        ],
                        "candidate_top_scores": [
                            float(value) for value in new_scores[index].tolist()
                        ],
                        "historical_top1_correct": old_top[0] == truth,
                        "candidate_top1_correct": new_top[0] == truth,
                        "historical_top3_correct": truth in old_top,
                        "candidate_top3_correct": truth in new_top,
                        "top1_agreement": old_top[0] == new_top[0],
                    }
                )

    count = len(results)
    if not count:
        raise ValueError("no genuinely held-out labelled Elements remain after deduplication")

    def mean_bool(key: str) -> float:
        return sum(bool(row[key]) for row in results) / count

    per_class_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[row["class_name"]].append(row)
    for class_name, class_rows in sorted(grouped.items()):
        class_count = len(class_rows)
        old_top1 = sum(row["historical_top1_correct"] for row in class_rows) / class_count
        new_top1 = sum(row["candidate_top1_correct"] for row in class_rows) / class_count
        per_class_rows.append(
            {
                "class_name": class_name,
                "count": class_count,
                "historical_top1": old_top1,
                "candidate_top1": new_top1,
                "delta_candidate_minus_historical": new_top1 - old_top1,
                "historical_top3": sum(
                    row["historical_top3_correct"] for row in class_rows
                )
                / class_count,
                "candidate_top3": sum(
                    row["candidate_top3_correct"] for row in class_rows
                )
                / class_count,
                "top1_agreement": sum(row["top1_agreement"] for row in class_rows)
                / class_count,
            }
        )

    macro_old = sum(row["historical_top1"] for row in per_class_rows) / len(per_class_rows)
    macro_new = sum(row["candidate_top1"] for row in per_class_rows) / len(per_class_rows)
    summary = {
        "count": count,
        "class_count": len(per_class_rows),
        "historical_top1": mean_bool("historical_top1_correct"),
        "candidate_top1": mean_bool("candidate_top1_correct"),
        "historical_top3": mean_bool("historical_top3_correct"),
        "candidate_top3": mean_bool("candidate_top3_correct"),
        "historical_macro_top1": macro_old,
        "candidate_macro_top1": macro_new,
        "top1_agreement": mean_bool("top1_agreement"),
        "candidate_only_correct": sum(
            row["candidate_top1_correct"] and not row["historical_top1_correct"]
            for row in results
        ),
        "historical_only_correct": sum(
            row["historical_top1_correct"] and not row["candidate_top1_correct"]
            for row in results
        ),
        "both_wrong": sum(
            not row["historical_top1_correct"] and not row["candidate_top1_correct"]
            for row in results
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    return summary, per_class_rows, results


def save_per_class_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, "white")
    image = image.convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def draw_labelled_examples(path: Path, rows: list[dict[str, Any]]) -> None:
    categories = [
        (
            "Candidat seul correct",
            [
                row
                for row in rows
                if row["candidate_top1_correct"] and not row["historical_top1_correct"]
            ],
        ),
        (
            "Historique seul correct",
            [
                row
                for row in rows
                if row["historical_top1_correct"] and not row["candidate_top1_correct"]
            ],
        ),
        ("Les deux incorrects", [row for row in rows if not row["historical_top1_correct"] and not row["candidate_top1_correct"]]),
        ("Les deux corrects", [row for row in rows if row["historical_top1_correct"] and row["candidate_top1_correct"]]),
    ]
    selected: list[tuple[str, dict[str, Any]]] = []
    for title, candidates in categories:
        for row in sorted(
            candidates,
            key=lambda value: abs(
                value["candidate_top_scores"][0] - value["historical_top_scores"][0]
            ),
            reverse=True,
        )[:3]:
            selected.append((title, row))

    cell_w, cell_h = 420, 330
    columns = 3
    rows_count = max(1, math.ceil(len(selected) / columns))
    image = Image.new("RGB", (columns * cell_w, rows_count * cell_h + 70), "white")
    draw = ImageDraw.Draw(image)
    font = load_font(18)
    small = load_font(15)
    bold = load_font(20, bold=True)
    draw.text((18, 18), "Holdout Elements — exemples comparatifs", fill="black", font=bold)
    for index, (category, row) in enumerate(selected):
        x = (index % columns) * cell_w
        y = 70 + (index // columns) * cell_h
        crop = fit_image(Image.open(row["path"]), (180, 180))
        image.paste(crop, (x + 12, y + 12))
        text_x = x + 205
        draw.text((text_x, y + 12), category, fill=MODEL_COLORS["neutral"], font=small)
        draw.text((text_x, y + 42), f"Vérité: {row['class_name']}", fill="black", font=font)
        old_color = MODEL_COLORS["agree"] if row["historical_top1_correct"] else MODEL_COLORS["disagree"]
        new_color = MODEL_COLORS["agree"] if row["candidate_top1_correct"] else MODEL_COLORS["disagree"]
        draw.text(
            (text_x, y + 82),
            f"Ancien: {row['historical_top_names'][0]}",
            fill=old_color,
            font=small,
        )
        draw.text(
            (text_x, y + 106),
            f"{row['historical_top_scores'][0]:.3f}",
            fill=old_color,
            font=small,
        )
        draw.text(
            (text_x, y + 148),
            f"Nouveau: {row['candidate_top_names'][0]}",
            fill=new_color,
            font=small,
        )
        draw.text(
            (text_x, y + 172),
            f"{row['candidate_top_scores'][0]:.3f}",
            fill=new_color,
            font=small,
        )
        draw.text(
            (x + 12, y + 210),
            Path(row["path"]).name[:48],
            fill=MODEL_COLORS["neutral"],
            font=small,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, optimize=True)


def draw_per_class_comparison(path: Path, per_class: list[dict[str, Any]]) -> None:
    ranked = sorted(
        per_class,
        key=lambda row: (
            abs(row["delta_candidate_minus_historical"]),
            row["count"],
        ),
        reverse=True,
    )[:30]
    width, row_h = 1200, 34
    image = Image.new("RGB", (width, 90 + len(ranked) * row_h), "white")
    draw = ImageDraw.Draw(image)
    font = load_font(15)
    bold = load_font(20, bold=True)
    draw.text((18, 18), "Éléments — plus grands écarts par classe", fill="black", font=bold)
    bar_x, bar_w = 330, 700
    for index, row in enumerate(ranked):
        y = 70 + index * row_h
        draw.text(
            (18, y + 4),
            f"{row['class_name']} (n={row['count']})",
            fill="black",
            font=font,
        )
        old_w = int(bar_w * row["historical_top1"])
        new_w = int(bar_w * row["candidate_top1"])
        draw.rectangle(
            (bar_x, y + 3, bar_x + old_w, y + 13), fill=MODEL_COLORS["historical"]
        )
        draw.rectangle(
            (bar_x, y + 18, bar_x + new_w, y + 28), fill=MODEL_COLORS["candidate"]
        )
        draw.text(
            (bar_x + bar_w + 12, y + 3),
            f"{row['historical_top1']:.0%} / {row['candidate_top1']:.0%}",
            fill="black",
            font=font,
        )
    image.save(path, optimize=True)


def choose_glyph_sample(glyph_root: Path, per_class: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    chosen: list[dict[str, Any]] = []
    for class_dir in sorted(path for path in glyph_root.iterdir() if path.is_dir()):
        files = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        viable: list[Path] = []
        for path in files:
            try:
                with Image.open(path) as image:
                    if image.width >= 32 and image.height >= 32:
                        viable.append(path)
            except Exception:
                continue
        rng.shuffle(viable)
        expected = class_dir.name.removesuffix("-glyph")
        for path in viable[:per_class]:
            chosen.append(
                {
                    "path": str(path),
                    "glyph_class": class_dir.name,
                    "expected_element": expected,
                }
            )
    return chosen


def load_mask_generator(checkpoint: Path, device: torch.device) -> Any:
    legacy_root = REPO_ROOT / "_legacy" / "frontend_integration"
    if str(legacy_root) not in sys.path:
        sys.path.insert(0, str(legacy_root))
    from mobile_sam import SamAutomaticMaskGenerator, sam_model_registry

    sam = sam_model_registry["vit_t"](checkpoint=str(checkpoint)).to(device)
    sam.eval()
    return SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=16,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.92,
        min_mask_region_area=80,
    )


def segment_glyph(mask_generator: Any, image: np.ndarray) -> list[dict[str, Any]]:
    height, width = image.shape[:2]
    masks = mask_generator.generate(image)
    proposals: list[dict[str, Any]] = []
    for mask in masks:
        area = int(mask["area"])
        if area < 50 or area > int(height * width * 0.85):
            continue
        if float(mask["stability_score"]) < 0.8:
            continue
        x, y, box_w, box_h = [int(value) for value in mask["bbox"]]
        x1, y1 = max(0, x - 3), max(0, y - 3)
        x2, y2 = min(width, x + box_w + 3), min(height, y + box_h + 3)
        crop = image[y1:y2, x1:x2].copy()
        crop_mask = mask["segmentation"][y1:y2, x1:x2]
        crop[~crop_mask] = 255
        proposals.append(
            {
                "bbox": (x, y, box_w, box_h),
                "mask": mask["segmentation"],
                "area": area,
                "confidence": float(mask["predicted_iou"]),
                "crop": crop,
            }
        )

    proposals.sort(key=lambda proposal: proposal["confidence"], reverse=True)
    kept: list[dict[str, Any]] = []
    for proposal in proposals:
        overlaps = []
        for other in kept:
            intersection = np.logical_and(proposal["mask"], other["mask"]).sum()
            union = np.logical_or(proposal["mask"], other["mask"]).sum()
            overlaps.append(float(intersection / union) if union else 0.0)
        if all(overlap <= 0.5 for overlap in overlaps):
            kept.append(proposal)
    kept.sort(key=lambda proposal: proposal["area"], reverse=True)
    return kept


def evaluate_glyphs(
    sample: list[dict[str, Any]],
    mask_generator: Any,
    backbone: torch.nn.Module,
    historical: RuntimeWeights,
    candidate: RuntimeWeights,
    device: torch.device,
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    glyph_results: list[dict[str, Any]] = []
    for glyph_index, item in enumerate(sample):
        image = Image.open(item["path"]).convert("RGB")
        image_array = np.asarray(image)
        proposals = segment_glyph(mask_generator, image_array)
        segment_rows: list[dict[str, Any]] = []
        if proposals:
            tensors = torch.cat(
                [
                    _preprocess_image(Image.fromarray(proposal["crop"]), 224)
                    for proposal in proposals
                ],
                dim=0,
            ).to(device)
            with torch.inference_mode():
                features = backbone(tensors)
                old_labels, old_scores = classify_features(features, historical)
                new_labels, new_scores = classify_features(features, candidate)
            for index, proposal in enumerate(proposals):
                old_top = [int(value) for value in old_labels[index].tolist()]
                new_top = [int(value) for value in new_labels[index].tolist()]
                segment_rows.append(
                    {
                        "segment_index": index + 1,
                        "bbox": list(proposal["bbox"]),
                        "area": proposal["area"],
                        "segmentation_confidence": proposal["confidence"],
                        "historical_top_labels": old_top,
                        "historical_top_names": [
                            historical.class_names[label] for label in old_top
                        ],
                        "historical_top_scores": [
                            float(value) for value in old_scores[index].tolist()
                        ],
                        "candidate_top_labels": new_top,
                        "candidate_top_names": [
                            candidate.class_names[label] for label in new_top
                        ],
                        "candidate_top_scores": [
                            float(value) for value in new_scores[index].tolist()
                        ],
                        "top1_agreement": old_top[0] == new_top[0],
                        "top3_overlap": len(set(old_top) & set(new_top)) / 3,
                        "_crop": proposal["crop"],
                    }
                )
        expected = item["expected_element"]
        result = {
            **item,
            "image_sha256": sha256_file(Path(item["path"])),
            "segment_count": len(segment_rows),
            "top1_agreement": (
                sum(row["top1_agreement"] for row in segment_rows) / len(segment_rows)
                if segment_rows
                else None
            ),
            "historical_expected_retrieved_top1": any(
                expected == row["historical_top_names"][0] for row in segment_rows
            ),
            "candidate_expected_retrieved_top1": any(
                expected == row["candidate_top_names"][0] for row in segment_rows
            ),
            "historical_expected_retrieved_top3": any(
                expected in row["historical_top_names"] for row in segment_rows
            ),
            "candidate_expected_retrieved_top3": any(
                expected in row["candidate_top_names"] for row in segment_rows
            ),
            "segments": segment_rows,
        }
        glyph_results.append(result)
        draw_glyph_panel(
            output_dir / "glyph-panels" / f"{glyph_index + 1:02d}-{item['glyph_class']}.png",
            image,
            result,
        )

    segment_rows = [
        segment for glyph in glyph_results for segment in glyph["segments"]
    ]
    glyph_count = len(glyph_results)
    segment_count = len(segment_rows)
    summary = {
        "glyph_count": glyph_count,
        "glyph_class_count": len({row["glyph_class"] for row in glyph_results}),
        "segment_count": segment_count,
        "top1_agreement": (
            sum(row["top1_agreement"] for row in segment_rows) / segment_count
            if segment_count
            else None
        ),
        "mean_top3_overlap": (
            sum(row["top3_overlap"] for row in segment_rows) / segment_count
            if segment_count
            else None
        ),
        "historical_folder_label_retrieval_top1": sum(
            row["historical_expected_retrieved_top1"] for row in glyph_results
        )
        / glyph_count,
        "candidate_folder_label_retrieval_top1": sum(
            row["candidate_expected_retrieved_top1"] for row in glyph_results
        )
        / glyph_count,
        "historical_folder_label_retrieval_top3": sum(
            row["historical_expected_retrieved_top3"] for row in glyph_results
        )
        / glyph_count,
        "candidate_folder_label_retrieval_top3": sum(
            row["candidate_expected_retrieved_top3"] for row in glyph_results
        )
        / glyph_count,
    }
    draw_glyph_overview(output_dir / "glyph-comparison-overview.png", glyph_results)
    return summary, glyph_results


def draw_glyph_panel(path: Path, source: Image.Image, result: dict[str, Any]) -> None:
    segments = result["segments"]
    left_w, left_h = 650, 650
    row_h = 86
    width = 1450
    height = max(760, 120 + len(segments) * row_h)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(22, bold=True)
    font = load_font(16)
    small = load_font(14)
    draw.text(
        (20, 18),
        f"{result['glyph_class']} — {Path(result['path']).name}",
        fill="black",
        font=title_font,
    )
    source_box = fit_image(source, (left_w, left_h))
    canvas.paste(source_box, (20, 70))
    scale = min(left_w / source.width, left_h / source.height)
    offset_x = 20 + (left_w - int(source.width * scale)) // 2
    offset_y = 70 + (left_h - int(source.height * scale)) // 2
    for segment in segments:
        x, y, box_w, box_h = segment["bbox"]
        color = (
            MODEL_COLORS["agree"] if segment["top1_agreement"] else MODEL_COLORS["disagree"]
        )
        box = (
            offset_x + int(x * scale),
            offset_y + int(y * scale),
            offset_x + int((x + box_w) * scale),
            offset_y + int((y + box_h) * scale),
        )
        draw.rectangle(box, outline=color, width=3)
        draw.text(
            (box[0] + 2, box[1] + 2),
            str(segment["segment_index"]),
            fill=color,
            font=font,
        )

    table_x = 700
    draw.text(
        (table_x, 70),
        "Vert = accord top-1 · Rouge = désaccord",
        fill=MODEL_COLORS["neutral"],
        font=font,
    )
    for index, segment in enumerate(segments):
        y = 110 + index * row_h
        crop = fit_image(Image.fromarray(segment["_crop"]), (72, 72))
        canvas.paste(crop, (table_x, y))
        color = (
            MODEL_COLORS["agree"] if segment["top1_agreement"] else MODEL_COLORS["disagree"]
        )
        draw.text((table_x + 82, y), f"#{segment['segment_index']}", fill=color, font=font)
        draw.text(
            (table_x + 125, y),
            f"Ancien  {segment['historical_top_names'][0]}  {segment['historical_top_scores'][0]:.3f}",
            fill=MODEL_COLORS["historical"],
            font=font,
        )
        draw.text(
            (table_x + 125, y + 28),
            f"Nouveau {segment['candidate_top_names'][0]}  {segment['candidate_top_scores'][0]:.3f}",
            fill=MODEL_COLORS["candidate"],
            font=font,
        )
        draw.text(
            (table_x + 125, y + 54),
            "top-3 ancien: "
            + ", ".join(segment["historical_top_names"])
            + " · nouveau: "
            + ", ".join(segment["candidate_top_names"]),
            fill=MODEL_COLORS["neutral"],
            font=small,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)


def draw_glyph_overview(path: Path, glyphs: list[dict[str, Any]]) -> None:
    cell_w, cell_h = 360, 330
    columns = 3
    rows_count = math.ceil(len(glyphs) / columns)
    canvas = Image.new("RGB", (columns * cell_w, 60 + rows_count * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(21, bold=True)
    font = load_font(15)
    draw.text((16, 16), "Glyphs hors split — accord ancien / nouveau", fill="black", font=title_font)
    for index, glyph in enumerate(glyphs):
        x = (index % columns) * cell_w
        y = 60 + (index // columns) * cell_h
        image = fit_image(Image.open(glyph["path"]), (320, 240))
        canvas.paste(image, (x + 20, y + 8))
        agreement = glyph["top1_agreement"]
        agreement_text = "n/a" if agreement is None else f"{agreement:.0%}"
        draw.text(
            (x + 20, y + 255),
            f"{glyph['glyph_class']} · {glyph['segment_count']} régions",
            fill="black",
            font=font,
        )
        draw.text(
            (x + 20, y + 282),
            f"Accord top-1: {agreement_text}",
            fill=MODEL_COLORS["agree"] if agreement == 1 else MODEL_COLORS["disagree"],
            font=font,
        )
    canvas.save(path, optimize=True)


def strip_private_crops(glyphs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean: list[dict[str, Any]] = []
    for glyph in glyphs:
        clean.append(
            {
                **glyph,
                "segments": [
                    {key: value for key, value in segment.items() if key != "_crop"}
                    for segment in glyph["segments"]
                ],
            }
        )
    return clean


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-weights", type=Path, required=True)
    parser.add_argument("--candidate-weights", type=Path, required=True)
    parser.add_argument("--legacy-cache", type=Path, required=True)
    parser.add_argument("--external-snapshot", type=Path, required=True)
    parser.add_argument("--glyph-root", type=Path, required=True)
    parser.add_argument("--mobile-sam-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--glyphs-per-class", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    legacy_hashes, legacy_source_count = legacy_pixel_hashes(args.legacy_cache)
    holdout, holdout_audit = build_external_holdout(
        args.external_snapshot, legacy_hashes
    )
    holdout_audit["legacy_source_count"] = legacy_source_count
    holdout_audit["legacy_unique_pixel_hash_count"] = len(legacy_hashes)

    historical = load_runtime("historical", args.historical_weights, device)
    candidate = load_runtime("candidate", args.candidate_weights, device)
    validate_runtime_contract(historical, candidate)
    backbone = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vits14", pretrained=True
    ).to(device)
    backbone.eval()

    element_summary, per_class, element_rows = evaluate_labelled_elements(
        holdout,
        backbone,
        historical,
        candidate,
        device,
        args.batch_size,
    )
    save_per_class_csv(args.output_dir / "per-class-elements.csv", per_class)
    json_dump(args.output_dir / "element-results.json", element_rows)
    draw_labelled_examples(
        args.output_dir / "labelled-elements-comparison.png", element_rows
    )
    draw_per_class_comparison(
        args.output_dir / "per-class-elements-comparison.png", per_class
    )

    glyph_sample = choose_glyph_sample(
        args.glyph_root, args.glyphs_per_class, args.seed
    )
    mask_generator = load_mask_generator(args.mobile_sam_checkpoint, device)
    glyph_summary, glyph_rows = evaluate_glyphs(
        glyph_sample,
        mask_generator,
        backbone,
        historical,
        candidate,
        device,
        args.output_dir,
    )

    report = {
        "schema_version": "dual-model-unseen-comparison.v1",
        "historical": {
            "weights_dir": str(args.historical_weights.resolve()),
            "projection_sha256": sha256_file(args.historical_weights / "projection.pt"),
            "prototypes_sha256": sha256_file(args.historical_weights / "prototypes.pt"),
        },
        "candidate": {
            "weights_dir": str(args.candidate_weights.resolve()),
            "projection_sha256": sha256_file(args.candidate_weights / "projection.pt"),
            "prototypes_sha256": sha256_file(args.candidate_weights / "prototypes.pt"),
        },
        "holdout_audit": holdout_audit,
        "labelled_elements": element_summary,
        "glyphs": {
            **glyph_summary,
            "ground_truth_scope": (
                "folder-level glyph class only; no segment-level element labels"
            ),
        },
        "seed": args.seed,
    }
    json_dump(args.output_dir / "glyph-results.json", strip_private_crops(glyph_rows))
    json_dump(args.output_dir / "comparison-report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
