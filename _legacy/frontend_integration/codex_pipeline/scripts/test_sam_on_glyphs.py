#!/usr/bin/env python3
"""
Test MobileSAM segmentation on glyph images.

For each selected glyph image:
1. MobileSAM segments the glyph into regions
2. Each region is classified by the trained element classifier
3. An annotated visualization is produced showing detected elements

Usage:
    python -m codex_pipeline.scripts.test_sam_on_glyphs
    python -m codex_pipeline.scripts.test_sam_on_glyphs --num-per-class 3
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'text.usetex': False,
})
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import hsv_to_rgb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator

from codex_pipeline.inference.engine import InferenceEngine


# ── MobileSAM segmentation ──────────────────────────────────

def load_mobilesam(checkpoint_path: str, device: str = "cpu"):
    """Load MobileSAM model and create mask generator."""
    sam = sam_model_registry["vit_t"](checkpoint=checkpoint_path)
    sam.to(device)
    sam.eval()

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=16,          # 16x16=256 prompts (was 32x32=1024)
        pred_iou_thresh=0.88,
        stability_score_thresh=0.92,
        min_mask_region_area=80,
    )
    return mask_generator


def segment_glyph(mask_generator, image: np.ndarray,
                  min_area: int = 50,
                  max_area_ratio: float = 0.85,
                  iou_threshold: float = 0.5) -> list:
    """
    Segment a glyph image into region proposals.

    Returns list of dicts with keys: bbox, mask, area, confidence, crop
    """
    h, w = image.shape[:2]
    page_area = h * w
    max_area = int(page_area * max_area_ratio)

    masks = mask_generator.generate(image)

    # Filter by size and quality
    proposals = []
    for m in masks:
        area = m["area"]
        if area < min_area or area > max_area:
            continue
        if m["stability_score"] < 0.8:
            continue

        bbox = m["bbox"]  # [x, y, w, h]

        # Extract crop with mask
        x, y, bw, bh = bbox
        pad = 3
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)

        crop = image[y1:y2, x1:x2].copy()
        mask_crop = m["segmentation"][y1:y2, x1:x2]
        crop[~mask_crop] = 255  # white background

        proposals.append({
            "bbox": (x, y, bw, bh),
            "mask": m["segmentation"],
            "area": area,
            "confidence": m["predicted_iou"],
            "stability": m["stability_score"],
            "crop": crop,
        })

    # NMS
    proposals.sort(key=lambda p: p["confidence"], reverse=True)
    keep = []
    for prop in proposals:
        should_keep = True
        for kept in keep:
            intersection = np.logical_and(prop["mask"], kept["mask"]).sum()
            union = np.logical_or(prop["mask"], kept["mask"]).sum()
            if union > 0 and intersection / union > iou_threshold:
                should_keep = False
                break
        if should_keep:
            keep.append(prop)

    keep.sort(key=lambda p: p["area"], reverse=True)
    return keep


# ── Visualization ────────────────────────────────────────────

def visualize_glyph_decomposition(
    image: np.ndarray,
    proposals: list,
    classifications: list,
    glyph_class: str,
    output_path: str,
    whole_glyph_pred: str = None,
    whole_glyph_conf: float = None,
):
    """
    Create a visualization showing:
    - Left: original glyph with element bounding boxes
    - Right: individual element crops with their classifications
    """
    n_elements = len(proposals)
    if n_elements == 0:
        return

    # Layout: original on left, up to 8 crops on right
    n_crops = min(n_elements, 8)
    n_rows = (n_crops + 1) // 2

    fig = plt.figure(figsize=(16, max(6, n_rows * 2.5)))

    # Left: annotated glyph
    ax_main = fig.add_axes([0.02, 0.05, 0.45, 0.85])
    ax_main.imshow(image)
    ax_main.set_axis_off()

    title = f"Glyphe : {glyph_class}"
    if whole_glyph_pred:
        title += f"\nClassification globale : {whole_glyph_pred} ({whole_glyph_conf:.3f})"
    ax_main.set_title(title, fontsize=10, fontweight="bold")

    colors = [hsv_to_rgb((i / max(n_elements, 1), 0.8, 0.9)) for i in range(n_elements)]

    for i, (prop, clf) in enumerate(zip(proposals[:8], classifications[:8])):
        if clf["rejected"]:
            continue
        color = colors[i]
        x, y, bw, bh = prop["bbox"]
        rect = patches.Rectangle(
            (x, y), bw, bh,
            linewidth=2, edgecolor=color, facecolor=(*color, 0.15),
        )
        ax_main.add_patch(rect)
        label = f"{clf['class_name']} ({clf['similarity']:.2f})"
        ax_main.text(x, y - 2, label, fontsize=7, color=color,
                     fontweight="bold", backgroundcolor="white")

    # Right: individual crops
    for i in range(n_crops):
        row = i // 2
        col = i % 2
        ax = fig.add_axes([
            0.52 + col * 0.24,
            0.85 - (row + 1) * (0.8 / n_rows),
            0.22,
            0.7 / n_rows,
        ])
        ax.imshow(proposals[i]["crop"])
        ax.set_axis_off()

        clf = classifications[i]
        status = "REJET" if clf["rejected"] else clf["class_name"]
        color = "red" if clf["rejected"] else "black"
        ax.set_title(
            f"{status}\nsim={clf['similarity']:.3f} | area={proposals[i]['area']}",
            fontsize=7, color=color,
        )

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


# ── Main ─────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test MobileSAM on glyph images")
    parser.add_argument("--glyphs-dir", default="./glyphs")
    parser.add_argument("--prototypes", default="./prototypes/prototypes.pt")
    parser.add_argument("--sam-checkpoint", default=str(Path.home() / ".cache/mobile_sam/mobile_sam.pt"))
    parser.add_argument("--output-dir", default="./results/sam_glyph_test")
    parser.add_argument("--num-per-class", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    glyphs_dir = Path(args.glyphs_dir)

    # Load models
    print("Loading InferenceEngine...")
    engine = InferenceEngine(
        prototype_path=args.prototypes,
        device=args.device,
        rejection_threshold=0.35,
        top_k=3,
    )
    print(f"  {len(engine.class_names)} element classes loaded")

    print("Loading MobileSAM...")
    t0 = time.time()
    mask_generator = load_mobilesam(args.sam_checkpoint, device=args.device)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Select test images
    test_classes = ["atl-glyph", "calli-glyph", "cohuatl-glyph", "cuauhtli-glyph",
                    "tochtli-glyph", "pantli-glyph"]

    all_results = {}

    for glyph_class in test_classes:
        class_dir = glyphs_dir / glyph_class
        if not class_dir.exists():
            print(f"Skipping {glyph_class} (not found)")
            continue

        images = sorted(class_dir.glob("*.jpg"))[:args.num_per_class]
        class_name = glyph_class.replace("-glyph", "")
        print(f"\n{'='*60}")
        print(f"Glyph class: {class_name} ({len(images)} images)")
        print(f"{'='*60}")

        class_results = []

        for img_path in images:
            print(f"\n  Image: {img_path.name}")
            image = np.array(Image.open(img_path).convert("RGB"))
            print(f"    Size: {image.shape[1]}x{image.shape[0]}")

            # 1. Whole-glyph classification (baseline)
            whole_result = engine.classify_crop(image)
            print(f"    Whole glyph → {whole_result.class_name} (sim={whole_result.similarity:.3f})")

            # 2. MobileSAM segmentation
            t0 = time.time()
            proposals = segment_glyph(mask_generator, image)
            seg_time = time.time() - t0
            print(f"    SAM segments: {len(proposals)} regions ({seg_time:.2f}s)")

            if not proposals:
                print(f"    No segments found — skipping")
                continue

            # 3. Classify each segment
            crops = [p["crop"] for p in proposals]
            results = engine.classify_batch(crops)

            classifications = []
            for r in results:
                classifications.append({
                    "class_name": r.class_name,
                    "similarity": r.similarity,
                    "rejected": r.rejected,
                    "top_k": r.top_k,
                })

            # 4. Print results
            accepted = [c for c in classifications if not c["rejected"]]
            rejected = [c for c in classifications if c["rejected"]]
            print(f"    Accepted elements: {len(accepted)}, Rejected: {len(rejected)}")

            from collections import Counter
            element_counts = Counter(c["class_name"] for c in accepted)
            print(f"    Detected elements:")
            for elem, count in element_counts.most_common(5):
                print(f"      {elem}: {count}")

            # 5. Visualize
            viz_path = output_dir / f"{class_name}_{img_path.stem}.png"
            visualize_glyph_decomposition(
                image, proposals, classifications,
                glyph_class=class_name,
                output_path=str(viz_path),
                whole_glyph_pred=whole_result.class_name,
                whole_glyph_conf=whole_result.similarity,
            )
            print(f"    Saved: {viz_path.name}")

            class_results.append({
                "image": img_path.name,
                "whole_prediction": whole_result.class_name,
                "whole_confidence": whole_result.similarity,
                "num_segments": len(proposals),
                "num_accepted": len(accepted),
                "num_rejected": len(rejected),
                "elements": dict(element_counts),
            })

        all_results[class_name] = class_results

    # Save JSON summary
    import json
    summary_path = output_dir / "sam_glyph_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"DONE — results saved to {output_dir}/")
    print(f"  Visualizations: {sum(len(v) for v in all_results.values())} images")
    print(f"  Summary: {summary_path.name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
