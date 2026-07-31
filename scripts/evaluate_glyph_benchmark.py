"""Evaluate glyph benchmark predictions against a manifest.

The evaluator is fail-closed:
- it rejects items whose ground truth is still pending;
- it refuses to inspect locked_test items unless --unlock-locked-test is set;
- it computes greedy IoU matching and classification metrics on the matched pairs.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    def normalized(self) -> "Box":
        x1 = min(self.x1, self.x2)
        y1 = min(self.y1, self.y2)
        x2 = max(self.x1, self.x2)
        y2 = max(self.y1, self.y2)
        return Box(x1, y1, x2, y2)


def box_iou(a: Box, b: Box) -> float:
    a = a.normalized()
    b = b.normalized()
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    if intersection == 0:
        return 0.0
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_box(raw_box) -> Box:
    if isinstance(raw_box, dict):
        coords = raw_box.get("box", raw_box)
    else:
        coords = raw_box
    if len(coords) != 4:
        raise ValueError(f"Expected 4 box coordinates, got {coords!r}")
    return Box(*(float(value) for value in coords))


def _parse_class_names(raw_prediction: dict) -> list[str]:
    for key in ("top_k", "topk", "top_k_class_names"):
        if key in raw_prediction and raw_prediction[key] is not None:
            return [str(value) for value in raw_prediction[key]]
    class_name = raw_prediction.get("class_name", raw_prediction.get("top1"))
    return [str(class_name)] if class_name is not None else []


def _normalize_prediction_box(raw_box) -> dict:
    if not isinstance(raw_box, dict):
        raw_box = {"box": raw_box}
    parsed_box = raw_box.get("box", raw_box)
    if isinstance(parsed_box, Box):
        box = parsed_box
    else:
        box = _parse_box(parsed_box)
    return {
        "box": box,
        "class_name": str(raw_box.get("class_name", raw_box.get("predicted_class", ""))),
        "top_k": _parse_class_names(raw_box),
        "score": float(raw_box.get("score", raw_box.get("similarity", 0.0)) or 0.0),
    }


def _parse_prediction_item(raw_item: dict) -> dict:
    boxes = []
    for raw_box in raw_item.get("predictions", raw_item.get("boxes", [])):
        boxes.append(_normalize_prediction_box(raw_box))
    return {"item_id": str(raw_item["item_id"]), "predictions": boxes}


def _load_predictions(path: Path) -> dict[str, dict]:
    raw = _load_json(path)
    items = raw.get("items", raw)
    if not isinstance(items, list):
        raise ValueError("Predictions JSON must contain a list of items.")
    parsed = {}
    for item in items:
        parsed_item = _parse_prediction_item(item)
        parsed[parsed_item["item_id"]] = parsed_item
    return parsed


def _greedy_match(gt_boxes: list[dict], pred_boxes: list[dict], iou_threshold: float) -> list[tuple[int, int, float]]:
    candidate_pairs: list[tuple[float, int, int]] = []
    for gt_index, gt_box in enumerate(gt_boxes):
        for pred_index, pred_box in enumerate(pred_boxes):
            overlap = box_iou(gt_box["box"], pred_box["box"])
            if overlap >= iou_threshold:
                candidate_pairs.append((overlap, gt_index, pred_index))
    candidate_pairs.sort(reverse=True)

    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for overlap, gt_index, pred_index in candidate_pairs:
        if gt_index in matched_gt or pred_index in matched_pred:
            continue
        matched_gt.add(gt_index)
        matched_pred.add(pred_index)
        matches.append((gt_index, pred_index, overlap))
    return matches


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_manifest(
    manifest: dict,
    predictions: dict[str, dict],
    *,
    splits: Iterable[str] = ("dev",),
    iou_threshold: float = 0.5,
    unlock_locked_test: bool = False,
) -> dict:
    selected_items = [item for item in manifest.get("items", []) if item.get("split") in set(splits)]
    if any(item.get("split") == "locked_test" for item in selected_items) and not unlock_locked_test:
        raise PermissionError("Refusing to evaluate locked_test without --unlock-locked-test.")

    for item in selected_items:
        gt_status = item.get("gt_status")
        gt_boxes = item.get("gt_boxes") or []
        if gt_status == "pending" or not gt_boxes:
            raise ValueError(f"Ground truth pending for item {item.get('id')!r}.")

    total_gt = 0
    total_pred = 0
    total_matched = 0
    total_top1_correct = 0
    total_top3_correct = 0
    per_class = defaultdict(lambda: {"gt": 0, "pred": 0, "matched": 0, "tp": 0, "top1": 0, "top3": 0})

    for item in selected_items:
        gt_boxes = [
            {"box": _parse_box(box["box"]), "class_name": str(box["class_name"])}
            for box in item.get("gt_boxes", [])
        ]
        pred_item = predictions.get(str(item["id"]), {"predictions": []})
        pred_boxes = [_normalize_prediction_box(box) for box in pred_item.get("predictions", [])]

        total_gt += len(gt_boxes)
        total_pred += len(pred_boxes)
        for pred_box in pred_boxes:
            per_class[pred_box["class_name"]]["pred"] += 1

        matches = _greedy_match(gt_boxes, pred_boxes, iou_threshold)
        matched_gt = {gt_index for gt_index, _, _ in matches}
        matched_pred = {pred_index for _, pred_index, _ in matches}

        for gt_index, pred_index, _ in matches:
            gt = gt_boxes[gt_index]
            pred = pred_boxes[pred_index]
            class_name = gt["class_name"]
            per_class[class_name]["gt"] += 1
            per_class[class_name]["matched"] += 1
            total_matched += 1
            if pred["class_name"] == class_name:
                per_class[class_name]["tp"] += 1
                per_class[class_name]["top1"] += 1
                total_top1_correct += 1
            if class_name in pred["top_k"]:
                per_class[class_name]["top3"] += 1
                total_top3_correct += 1

        unmatched_gt = [gt for index, gt in enumerate(gt_boxes) if index not in matched_gt]
        for gt in unmatched_gt:
            per_class[gt["class_name"]]["gt"] += 1

        # Unmatched predictions already counted in the class-wise pred totals.
        for pred_index, pred in enumerate(pred_boxes):
            if pred_index in matched_pred:
                continue

    detection_precision = _safe_div(total_matched, total_pred)
    detection_recall = _safe_div(total_matched, total_gt)
    detection_f1 = _safe_div(2 * detection_precision * detection_recall, detection_precision + detection_recall)
    classification_top1 = _safe_div(total_top1_correct, total_matched)
    classification_top3 = _safe_div(total_top3_correct, total_matched)
    exact_end_to_end = _safe_div(total_top1_correct, total_gt)

    per_class_metrics = {}
    for class_name, stats in sorted(per_class.items()):
        precision = _safe_div(stats["tp"], stats["pred"])
        recall = _safe_div(stats["tp"], stats["gt"])
        f1 = _safe_div(2 * precision * recall, precision + recall)
        per_class_metrics[class_name] = {
            "gt": stats["gt"],
            "pred": stats["pred"],
            "matched": stats["matched"],
            "tp": stats["tp"],
            "top1": stats["top1"],
            "top3": stats["top3"],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "exact_end_to_end": recall if stats["gt"] else 0.0,
        }

    macro_detection_f1 = _safe_div(sum(item["f1"] for item in per_class_metrics.values()), len(per_class_metrics))
    macro_exact_end_to_end = _safe_div(
        sum(item["exact_end_to_end"] for item in per_class_metrics.values()),
        len(per_class_metrics),
    )

    return {
        "schema_version": "1.0",
        "iou_threshold": iou_threshold,
        "splits": list(splits),
        "counts": {
            "ground_truth": total_gt,
            "predictions": total_pred,
            "matched": total_matched,
        },
        "metrics": {
            "detection": {
                "precision": detection_precision,
                "recall": detection_recall,
                "f1": detection_f1,
            },
            "classification_matched": {
                "top1": classification_top1,
                "top3": classification_top3,
            },
            "exact_end_to_end": exact_end_to_end,
            "macro": {
                "detection_f1": macro_detection_f1,
                "exact_end_to_end": macro_exact_end_to_end,
            },
        },
        "per_class": per_class_metrics,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", action="append", dest="splits", default=None)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--unlock-locked-test", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    manifest = _load_json(args.manifest)
    predictions = _load_predictions(args.predictions)
    report = evaluate_manifest(
        manifest,
        predictions,
        splits=args.splits,
        iou_threshold=args.iou_threshold,
        unlock_locked_test=args.unlock_locked_test,
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
