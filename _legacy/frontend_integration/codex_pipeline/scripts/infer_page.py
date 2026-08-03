"""
End-to-end page inference script for the Codex pipeline.

Processes one or more manuscript page images through the full pipeline:

    Page image
        → MobileSAM segmentation (region proposals)
        → DINOv2 embedding + prototype classification (InferenceEngine)
        → Spatial grouping into glyphs
        → Export (JSON, CSV, optional annotated PNG)

Usage examples::

    # Single image
    python -m codex_pipeline.scripts.infer_page --image page.jpg

    # Directory of images
    python -m codex_pipeline.scripts.infer_page --image-dir ./pages --output-dir ./results

    # Custom thresholds, skip visualisation
    python -m codex_pipeline.scripts.infer_page \\
        --image page.jpg \\
        --prototypes ./prototypes/prototypes.pt \\
        --threshold 0.40 \\
        --distance-threshold 60.0 \\
        --no-viz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Make the project root importable when the script is called directly or via
# `python -m codex_pipeline.scripts.infer_page`.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from codex_pipeline.inference.engine import InferenceEngine
from codex_pipeline.inference.export import export_csv, export_json, render_annotated_page
from codex_pipeline.inference.grouping import DetectedElement, group_elements_into_glyphs
from codex_pipeline.segmentation.mobilesam import MobileSAMSegmenter


# ---------------------------------------------------------------------------
# Supported image extensions
# ---------------------------------------------------------------------------

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Core pipeline function
# ---------------------------------------------------------------------------


def process_page(
    image: np.ndarray,
    segmenter: MobileSAMSegmenter,
    engine: InferenceEngine,
    distance_threshold: float = 80.0,
) -> Dict:
    """
    Run the full detection pipeline on a single page image.

    The pipeline executes four stages:

    1. **Segmentation** — MobileSAM generates region proposals for the page.
    2. **Classification** — Crops are extracted from each proposal and
       classified in a single batch using DINOv2 + prototypical network.
    3. **Element construction** — Proposals and classification results are
       zipped into :class:`~codex_pipeline.inference.grouping.DetectedElement`
       objects, preserving the rejection flag from the engine.
    4. **Grouping** — Accepted elements are clustered spatially into
       :class:`~codex_pipeline.inference.grouping.DetectedGlyph` objects.

    Args:
        image: Full-page image as a uint8 RGB numpy array of shape (H, W, 3).
        segmenter: Initialised :class:`~codex_pipeline.segmentation.mobilesam.MobileSAMSegmenter`.
        engine: Initialised :class:`~codex_pipeline.inference.engine.InferenceEngine`.
        distance_threshold: Maximum centroid distance (pixels) for merging two
            element clusters into a single glyph.  Passed directly to
            :func:`~codex_pipeline.inference.grouping.group_elements_into_glyphs`.

    Returns:
        Dictionary with the following keys:

        * ``"proposals"`` — list of
          :class:`~codex_pipeline.segmentation.mobilesam.RegionProposal` objects
          returned by :meth:`MobileSAMSegmenter.segment_page`.
        * ``"elements"`` — list of
          :class:`~codex_pipeline.inference.grouping.DetectedElement` objects,
          one per proposal (includes both accepted and rejected elements).
        * ``"glyphs"`` — list of
          :class:`~codex_pipeline.inference.grouping.DetectedGlyph` objects
          sorted left-to-right.
        * ``"timing"`` — dictionary mapping stage names to elapsed seconds:
          ``"segmentation"``, ``"classification"``, ``"grouping"``.
    """
    timing: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Stage 1: Segment the page into region proposals
    # ------------------------------------------------------------------
    t0 = time.time()
    proposals = segmenter.segment_page(image)
    timing["segmentation"] = time.time() - t0

    # ------------------------------------------------------------------
    # Stage 2: Extract crops and classify in a single batch
    # ------------------------------------------------------------------
    t0 = time.time()
    proposals = segmenter.extract_crops(image, proposals)

    crops: List[np.ndarray] = [p.crop for p in proposals if p.crop is not None]
    # Guard: if a crop is somehow missing, substitute a blank white patch
    crops_for_batch: List[np.ndarray] = []
    for p in proposals:
        if p.crop is not None:
            crops_for_batch.append(p.crop)
        else:
            # Fallback: create a minimal white placeholder crop
            crops_for_batch.append(np.full((32, 32, 3), 255, dtype=np.uint8))

    results = engine.classify_batch(crops_for_batch)
    timing["classification"] = time.time() - t0

    # ------------------------------------------------------------------
    # Stage 3: Build DetectedElement list
    # ------------------------------------------------------------------
    elements: List[DetectedElement] = []
    for proposal, result in zip(proposals, results):
        elements.append(
            DetectedElement(
                bbox=proposal.bbox,
                class_name=result.class_name,
                similarity=result.similarity,
                rejected=result.rejected,
            )
        )

    # ------------------------------------------------------------------
    # Stage 4: Group accepted elements into glyphs
    # ------------------------------------------------------------------
    t0 = time.time()
    glyphs = group_elements_into_glyphs(
        elements,
        distance_threshold=distance_threshold,
    )
    timing["grouping"] = time.time() - t0

    return {
        "proposals": proposals,
        "elements": elements,
        "glyphs": glyphs,
        "timing": timing,
    }


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def _collect_image_paths(args: argparse.Namespace) -> List[Path]:
    """
    Build and validate the list of page image paths from CLI arguments.

    Args:
        args: Parsed namespace from :func:`_build_parser`.

    Returns:
        Non-empty list of :class:`pathlib.Path` objects pointing to image files.

    Raises:
        SystemExit: If no images can be found or argument constraints are
            violated.
    """
    paths: List[Path] = []

    if args.image is not None:
        p = Path(args.image)
        if not p.is_file():
            print(f"[ERROR] --image path does not exist or is not a file: {p}", file=sys.stderr)
            sys.exit(1)
        paths.append(p)

    if args.image_dir is not None:
        d = Path(args.image_dir)
        if not d.is_dir():
            print(f"[ERROR] --image-dir path does not exist or is not a directory: {d}", file=sys.stderr)
            sys.exit(1)
        found = sorted(
            f for f in d.iterdir()
            if f.is_file() and f.suffix.lower() in _IMAGE_EXTENSIONS
        )
        if not found:
            print(
                f"[ERROR] No supported image files found in {d} "
                f"(supported extensions: {', '.join(sorted(_IMAGE_EXTENSIONS))})",
                file=sys.stderr,
            )
            sys.exit(1)
        paths.extend(found)

    if not paths:
        print(
            "[ERROR] No images to process.  Provide --image or --image-dir.",
            file=sys.stderr,
        )
        sys.exit(1)

    return paths


def _build_parser() -> argparse.ArgumentParser:
    """
    Construct the argument parser for the infer_page CLI.

    Returns:
        Configured :class:`argparse.ArgumentParser` instance.
    """
    parser = argparse.ArgumentParser(
        prog="python -m codex_pipeline.scripts.infer_page",
        description=(
            "End-to-end codex page inference: "
            "MobileSAM segmentation → DINOv2 classification → glyph grouping → export."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input — mutually exclusive, but at least one must be given
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image",
        metavar="PATH",
        default=None,
        help="Path to a single page image file.",
    )
    input_group.add_argument(
        "--image-dir",
        metavar="DIR",
        default=None,
        help=(
            "Directory of page images.  All files with extensions "
            f"{', '.join(sorted(_IMAGE_EXTENSIONS))} are processed."
        ),
    )

    # Output
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        default="./results",
        help="Directory where JSON, CSV, and PNG outputs are written.",
    )

    # Model / checkpoint
    parser.add_argument(
        "--prototypes",
        metavar="PATH",
        default="./prototypes/prototypes.pt",
        help="Path to the exported prototypes.pt checkpoint.",
    )
    parser.add_argument(
        "--sam-checkpoint",
        metavar="PATH",
        default=None,
        help=(
            "Path to a local MobileSAM checkpoint.  "
            "Omit to use the auto-downloaded default (~/.cache/mobile_sam/mobile_sam.pt)."
        ),
    )

    # Device
    parser.add_argument(
        "--device",
        metavar="DEVICE",
        default="auto",
        help='Compute device: "auto", "cuda", "mps", or "cpu".',
    )

    # Classification thresholds
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        metavar="FLOAT",
        help="Cosine similarity rejection threshold for the InferenceEngine.",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=80.0,
        metavar="FLOAT",
        help="Maximum inter-centroid distance (pixels) for glyph grouping.",
    )

    # Visualisation
    parser.add_argument(
        "--no-viz",
        action="store_true",
        default=False,
        help="Skip generation of annotated PNG visualisations.",
    )

    return parser


# ---------------------------------------------------------------------------
# Per-page summary printing
# ---------------------------------------------------------------------------


def _print_page_summary(
    page_name: str,
    result: Dict,
    output_dir: Path,
    viz: bool,
) -> None:
    """
    Print a concise per-page processing summary to stdout.

    Args:
        page_name: Stem of the source image file (without extension).
        result: Dictionary returned by :func:`process_page`.
        output_dir: Directory where outputs were written.
        viz: Whether a visualisation PNG was produced.
    """
    proposals = result["proposals"]
    elements = result["elements"]
    glyphs = result["glyphs"]
    timing = result["timing"]

    accepted = sum(1 for e in elements if not e.rejected)
    rejected = len(elements) - accepted
    total_t = sum(timing.values())

    print(f"\n{'─' * 60}")
    print(f"  Page : {page_name}")
    print(f"{'─' * 60}")
    print(f"  Segments found        : {len(proposals)}")
    print(f"  Elements accepted     : {accepted}")
    print(f"  Elements rejected     : {rejected}")
    print(f"  Glyphs composed       : {len(glyphs)}")
    print(f"  Timing (s)")
    print(f"    segmentation        : {timing['segmentation']:.3f}")
    print(f"    classification      : {timing['classification']:.3f}")
    print(f"    grouping            : {timing['grouping']:.3f}")
    print(f"    total               : {total_t:.3f}")
    print(f"  Output dir            : {output_dir}")
    print(f"  Visualisation         : {'written' if viz else 'skipped (--no-viz)'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Parse CLI arguments, load models, and run the pipeline over all pages.

    For each page the following output files are written to *output_dir*:

    * ``{page_name}.json`` — structured detection results.
    * ``{page_name}.csv``  — flat element-level table.
    * ``{page_name}_annotated.png`` — annotated image (unless ``--no-viz``).

    A global summary (total pages, elements, glyphs, wall-clock time) is
    printed after all pages have been processed.
    """
    parser = _build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Collect image paths
    # ------------------------------------------------------------------
    image_paths = _collect_image_paths(args)
    print(f"[infer_page] Found {len(image_paths)} image(s) to process.")

    # ------------------------------------------------------------------
    # Load models (once, shared across all pages)
    # ------------------------------------------------------------------
    print("\n[infer_page] Loading InferenceEngine …")
    engine = InferenceEngine(
        prototype_path=args.prototypes,
        device=args.device,
        rejection_threshold=args.threshold,
    )

    print("[infer_page] Loading MobileSAMSegmenter …")
    segmenter = MobileSAMSegmenter(
        checkpoint_path=args.sam_checkpoint,  # None triggers auto-download
        device=args.device,
    )

    # ------------------------------------------------------------------
    # Process each page
    # ------------------------------------------------------------------
    global_start = time.time()
    global_segments = 0
    global_accepted = 0
    global_rejected = 0
    global_glyphs = 0

    for idx, image_path in enumerate(image_paths, start=1):
        page_name = image_path.stem
        print(f"\n[infer_page] Processing {idx}/{len(image_paths)}: {image_path.name}")

        # Load image as RGB numpy array
        try:
            from PIL import Image as PILImage
            pil_img = PILImage.open(image_path).convert("RGB")
            image_np: np.ndarray = np.array(pil_img, dtype=np.uint8)
        except Exception as exc:
            print(f"[WARNING] Could not load {image_path}: {exc} — skipping.", file=sys.stderr)
            continue

        # Run pipeline
        result = process_page(
            image=image_np,
            segmenter=segmenter,
            engine=engine,
            distance_threshold=args.distance_threshold,
        )

        proposals = result["proposals"]
        elements = result["elements"]
        glyphs = result["glyphs"]

        # Accumulate global counters
        global_segments += len(proposals)
        accepted_count = sum(1 for e in elements if not e.rejected)
        global_accepted += accepted_count
        global_rejected += len(elements) - accepted_count
        global_glyphs += len(glyphs)

        # ------------------------------------------------------------------
        # Export outputs
        # ------------------------------------------------------------------
        json_path = output_dir / f"{page_name}.json"
        csv_path = output_dir / f"{page_name}.csv"
        viz_path = output_dir / f"{page_name}_annotated.png"

        export_json(
            glyphs=glyphs,
            output_path=str(json_path),
            page_name=page_name,
            metadata={
                "source_image": str(image_path),
                "threshold": args.threshold,
                "distance_threshold": args.distance_threshold,
                "num_proposals": len(proposals),
                "num_elements_accepted": accepted_count,
                "num_elements_rejected": len(elements) - accepted_count,
                "timing": result["timing"],
            },
        )

        export_csv(
            glyphs=glyphs,
            output_path=str(csv_path),
            page_name=page_name,
        )

        viz_written = False
        if not args.no_viz:
            render_annotated_page(
                image=image_np,
                glyphs=glyphs,
                output_path=str(viz_path),
            )
            viz_written = True

        _print_page_summary(
            page_name=page_name,
            result=result,
            output_dir=output_dir,
            viz=viz_written,
        )

    # ------------------------------------------------------------------
    # Global summary
    # ------------------------------------------------------------------
    global_elapsed = time.time() - global_start
    print(f"\n{'═' * 60}")
    print("  GLOBAL SUMMARY")
    print(f"{'═' * 60}")
    print(f"  Pages processed       : {len(image_paths)}")
    print(f"  Total segments        : {global_segments}")
    print(f"  Total accepted        : {global_accepted}")
    print(f"  Total rejected        : {global_rejected}")
    print(f"  Total glyphs          : {global_glyphs}")
    print(f"  Wall-clock time (s)   : {global_elapsed:.2f}")
    print(f"  Output directory      : {output_dir.resolve()}")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
