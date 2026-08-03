#!/usr/bin/env python3
"""
Glyph inference script for Phase 1B testing.

Loads the trained InferenceEngine from a prototype checkpoint, scans every
subdirectory under ``glyphs/`` (each subdir = one glyph class), classifies
every image found, and reports per-class breakdowns together with a global
acceptance rate summary.  Optionally saves structured JSON results.

Usage::

    python -m codex_pipeline.scripts.infer_glyph
    python -m codex_pipeline.scripts.infer_glyph --glyphs-dir ./glyphs \\
        --prototypes ./prototypes/prototypes.pt --top-k 5 --threshold 0.35 \\
        --output ./results/glyph_inference.json --device auto
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ensure the project root is importable regardless of the working directory
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from codex_pipeline.inference.engine import InferenceEngine  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Image I/O helpers
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def load_image_rgb(path: Path) -> Optional[np.ndarray]:
    """
    Load an image from *path* and return a uint8 RGB NumPy array (H, W, 3).

    Returns ``None`` and logs a warning if the file cannot be opened.
    """
    try:
        img = Image.open(path).convert("RGB")
        return np.array(img, dtype=np.uint8)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load image %s: %s", path, exc)
        return None


def collect_image_paths(glyph_class_dir: Path) -> List[Path]:
    """
    Return a sorted list of all supported image paths inside *glyph_class_dir*.
    """
    return sorted(
        p
        for p in glyph_class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


# ---------------------------------------------------------------------------
# Per-class inference
# ---------------------------------------------------------------------------

BATCH_SIZE = 32


def classify_glyph_class(
    engine: InferenceEngine,
    glyph_class_name: str,
    image_paths: List[Path],
) -> dict:
    """
    Classify all images belonging to *glyph_class_name* and return a result
    dict that matches the documented JSON output schema::

        {
            "num_images": N,
            "predictions": [
                {
                    "image": "filename.jpg",
                    "prediction": "element_name",
                    "similarity": 0.87,
                    "rejected": false,
                    "top_k": [...]
                },
                ...
            ],
            "summary": {
                "accepted": N,
                "rejected": N,
                "top_elements": {"element_name": count, ...}
            }
        }

    Images that fail to load are skipped (not counted toward accepted/rejected).
    Uses batched inference for ~5-10x speedup over single-image classification.
    """
    predictions: List[dict] = []
    accepted_count = 0
    rejected_count = 0
    element_counter: Counter = Counter()

    # Load all images first, tracking which paths succeeded
    batch_images: List[np.ndarray] = []
    batch_names: List[str] = []

    for img_path in image_paths:
        image_np = load_image_rgb(img_path)
        if image_np is not None:
            batch_images.append(image_np)
            batch_names.append(img_path.name)

    # Process in batches for efficient GPU/MPS utilization
    num_batches = (len(batch_images) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in tqdm(range(num_batches), desc=glyph_class_name, leave=False, unit="batch"):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(batch_images))
        batch = batch_images[start:end]
        names = batch_names[start:end]

        results = engine.classify_batch(batch)

        for img_name, result in zip(names, results):
            top_k_serialisable = [
                {
                    "class_label": int(entry["class_label"]),
                    "class_name": str(entry["class_name"]),
                    "similarity": float(entry["similarity"]),
                }
                for entry in result.top_k
            ]

            predictions.append(
                {
                    "image": img_name,
                    "prediction": result.class_name,
                    "similarity": float(result.similarity),
                    "rejected": bool(result.rejected),
                    "top_k": top_k_serialisable,
                }
            )

            if result.rejected:
                rejected_count += 1
            else:
                accepted_count += 1
                element_counter[result.class_name] += 1

    return {
        "num_images": len(predictions),
        "predictions": predictions,
        "summary": {
            "accepted": accepted_count,
            "rejected": rejected_count,
            "top_elements": dict(element_counter.most_common()),
        },
    }


# ---------------------------------------------------------------------------
# Console reporting
# ---------------------------------------------------------------------------

def print_class_summary(
    glyph_class_name: str,
    class_result: dict,
    top_n: int = 5,
) -> None:
    """
    Print a human-readable summary for one glyph class to stdout.
    """
    summary = class_result["summary"]
    num_images = class_result["num_images"]
    accepted = summary["accepted"]
    rejected = summary["rejected"]
    top_elements = summary["top_elements"]

    accept_pct = (accepted / num_images * 100) if num_images > 0 else 0.0
    reject_pct = (rejected / num_images * 100) if num_images > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"  Glyph class : {glyph_class_name}")
    print(f"  Images      : {num_images}")
    print(f"  Accepted    : {accepted} ({accept_pct:.1f}%)")
    print(f"  Rejected    : {rejected} ({reject_pct:.1f}%)")

    if top_elements:
        print(f"  Top predicted elements (accepted only):")
        for rank, (element, count) in enumerate(
            list(top_elements.items())[:top_n], start=1
        ):
            pct = count / accepted * 100 if accepted > 0 else 0.0
            print(f"    {rank}. {element:<30} {count:>4} images  ({pct:.1f}%)")
    else:
        print("  No accepted predictions.")


def print_global_summary(all_results: Dict[str, dict]) -> None:
    """
    Print a global summary across all glyph classes.
    """
    total_images = sum(v["num_images"] for v in all_results.values())
    total_accepted = sum(v["summary"]["accepted"] for v in all_results.values())
    total_rejected = sum(v["summary"]["rejected"] for v in all_results.values())

    global_accept_pct = (total_accepted / total_images * 100) if total_images > 0 else 0.0
    global_reject_pct = (total_rejected / total_images * 100) if total_images > 0 else 0.0

    print(f"\n{'='*60}")
    print("  GLOBAL SUMMARY")
    print(f"  Glyph classes   : {len(all_results)}")
    print(f"  Total images    : {total_images}")
    print(f"  Total accepted  : {total_accepted} ({global_accept_pct:.1f}%)")
    print(f"  Total rejected  : {total_rejected} ({global_reject_pct:.1f}%)")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Classify glyph images using the trained InferenceEngine and "
            "report per-class element breakdowns."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--glyphs-dir",
        type=str,
        default="./glyphs",
        help="Root directory whose subdirectories each contain glyph images.",
    )
    parser.add_argument(
        "--prototypes",
        type=str,
        default="./prototypes/prototypes.pt",
        help="Path to the exported prototypes.pt checkpoint.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of nearest-prototype candidates in each ClassificationResult.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Cosine similarity below which a prediction is marked as rejected.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save structured JSON results (e.g. results.json).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help='Compute device.  "auto" selects CUDA -> MPS -> CPU.',
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = build_arg_parser()
    args = parser.parse_args()

    glyphs_dir = Path(args.glyphs_dir).resolve()
    prototypes_path = Path(args.prototypes).resolve()

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if not glyphs_dir.is_dir():
        parser.error(f"--glyphs-dir does not exist or is not a directory: {glyphs_dir}")

    # ------------------------------------------------------------------
    # Load InferenceEngine
    # ------------------------------------------------------------------
    print(f"\nLoading InferenceEngine from {prototypes_path} ...")
    engine = InferenceEngine(
        prototype_path=str(prototypes_path),
        device=args.device,
        rejection_threshold=args.threshold,
        top_k=args.top_k,
    )
    print("InferenceEngine ready.\n")

    # ------------------------------------------------------------------
    # Discover glyph class subdirectories
    # ------------------------------------------------------------------
    glyph_class_dirs = sorted(
        d for d in glyphs_dir.iterdir() if d.is_dir()
    )

    if not glyph_class_dirs:
        print(f"No subdirectories found under {glyphs_dir}. Nothing to classify.")
        return

    print(f"Found {len(glyph_class_dirs)} glyph class(es) under {glyphs_dir}:")
    for d in glyph_class_dirs:
        n = len(collect_image_paths(d))
        print(f"  {d.name:<35} ({n} images)")

    # ------------------------------------------------------------------
    # Classify each glyph class
    # ------------------------------------------------------------------
    all_results: Dict[str, dict] = {}

    for glyph_class_dir in tqdm(
        glyph_class_dirs,
        desc="Glyph classes",
        unit="class",
    ):
        glyph_class_name = glyph_class_dir.name
        image_paths = collect_image_paths(glyph_class_dir)

        if not image_paths:
            logger.warning("No images found in %s — skipping.", glyph_class_dir)
            continue

        class_result = classify_glyph_class(engine, glyph_class_name, image_paths)
        all_results[glyph_class_name] = class_result

        print_class_summary(glyph_class_name, class_result)

    # ------------------------------------------------------------------
    # Global summary
    # ------------------------------------------------------------------
    print_global_summary(all_results)

    # ------------------------------------------------------------------
    # Optional JSON output
    # ------------------------------------------------------------------
    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(all_results, fh, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
