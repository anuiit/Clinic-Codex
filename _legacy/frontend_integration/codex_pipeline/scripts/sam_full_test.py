#!/usr/bin/env python3
"""
Run MobileSAM segmentation + element classification on ALL glyph images.

Optimized for throughput:
- Batch classification (32 crops at a time)
- Visualizations only for a sample (--viz-per-class)
- JSON results for all images

Usage:
    python -m codex_pipeline.scripts.sam_full_test
    python -m codex_pipeline.scripts.sam_full_test --viz-per-class 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator

from codex_pipeline.inference.engine import InferenceEngine


def load_mobilesam(checkpoint_path: str, device: str = "cpu"):
    sam = sam_model_registry["vit_t"](checkpoint=checkpoint_path)
    sam.to(device)
    sam.eval()
    return SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=16,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.92,
        min_mask_region_area=80,
    )


def segment_and_crop(mask_generator, image: np.ndarray,
                     min_area: int = 50, max_area_ratio: float = 0.85,
                     iou_threshold: float = 0.5) -> list:
    h, w = image.shape[:2]
    max_area = int(h * w * max_area_ratio)

    masks = mask_generator.generate(image)

    proposals = []
    for m in masks:
        area = m["area"]
        if area < min_area or area > max_area:
            continue
        if m["stability_score"] < 0.8:
            continue

        bbox = m["bbox"]
        x, y, bw, bh = bbox
        pad = 3
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)

        crop = image[y1:y2, x1:x2].copy()
        mask_crop = m["segmentation"][y1:y2, x1:x2]
        crop[~mask_crop] = 255

        proposals.append({
            "bbox": (x, y, bw, bh),
            "area": area,
            "confidence": m["predicted_iou"],
            "crop": crop,
        })

    # NMS
    proposals.sort(key=lambda p: p["confidence"], reverse=True)
    keep = []
    for prop in proposals:
        ok = True
        for kept in keep:
            # Simple bbox IoU (faster than mask IoU)
            ax, ay, aw, ah = prop["bbox"]
            bx, by, bbw, bh = kept["bbox"]
            ix = max(0, min(ax+aw, bx+bbw) - max(ax, bx))
            iy = max(0, min(ay+ah, by+bh) - max(ay, by))
            inter = ix * iy
            union = aw * ah + bbw * bh - inter
            if union > 0 and inter / union > iou_threshold:
                ok = False
                break
        if ok:
            keep.append(prop)

    return keep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glyphs-dir", default="./glyphs")
    parser.add_argument("--prototypes", default="./prototypes/prototypes.pt")
    parser.add_argument("--sam-checkpoint", default=str(Path.home() / ".cache/mobile_sam/mobile_sam.pt"))
    parser.add_argument("--output-dir", default="./results/sam_full_test")
    parser.add_argument("--viz-per-class", type=int, default=5,
                        help="Number of visualizations to generate per class")
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
    print(f"  {len(engine.class_names)} element classes")

    print("Loading MobileSAM...")
    mask_gen = load_mobilesam(args.sam_checkpoint, args.device)
    print("  Ready\n")

    # Discover all glyph classes
    all_classes = sorted(
        d for d in glyphs_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )

    global_stats = {
        "total_images": 0,
        "total_segments": 0,
        "total_accepted": 0,
        "total_rejected": 0,
        "avg_segments_per_glyph": 0,
        "avg_time_per_image": 0,
        "classes": {},
    }

    total_time = 0
    total_images = 0

    for class_dir in all_classes:
        class_name = class_dir.name.replace("-glyph", "")
        images = sorted(class_dir.glob("*.jpg"))
        if not images:
            images = sorted(class_dir.glob("*.png")) + sorted(class_dir.glob("*.bmp"))
        if not images:
            print(f"  {class_name}: no images, skipping")
            continue

        print(f"{'='*60}")
        print(f"{class_name}: {len(images)} images")
        print(f"{'='*60}")

        class_segments = 0
        class_accepted = 0
        class_rejected = 0
        class_elements = Counter()
        class_confidences = []
        class_results = []
        viz_count = 0

        for idx, img_path in enumerate(images):
            try:
                image = np.array(Image.open(img_path).convert("RGB"))
            except Exception:
                continue

            t0 = time.time()

            # Segment
            proposals = segment_and_crop(mask_gen, image)

            # Classify
            if proposals:
                crops = [p["crop"] for p in proposals]
                results = engine.classify_batch(crops)

                accepted = sum(1 for r in results if not r.rejected)
                rejected = sum(1 for r in results if r.rejected)
                elements = Counter(r.class_name for r in results if not r.rejected)
                confs = [r.similarity for r in results]
            else:
                results = []
                accepted = 0
                rejected = 0
                elements = Counter()
                confs = []

            elapsed = time.time() - t0
            total_time += elapsed
            total_images += 1

            class_segments += len(proposals)
            class_accepted += accepted
            class_rejected += rejected
            class_elements.update(elements)
            class_confidences.extend(confs)

            class_results.append({
                "image": img_path.name,
                "num_segments": len(proposals),
                "accepted": accepted,
                "rejected": rejected,
                "elements": dict(elements),
                "avg_confidence": float(np.mean(confs)) if confs else 0,
            })

            # Progress every 50 images
            if (idx + 1) % 50 == 0:
                avg_t = total_time / total_images
                remaining = (len(images) - idx - 1) * avg_t
                print(f"  [{idx+1}/{len(images)}] "
                      f"segments={len(proposals)} accepted={accepted} "
                      f"({elapsed:.1f}s) ETA class: {remaining/60:.1f}min")

            # Generate visualization for first N
            if viz_count < args.viz_per_class and proposals and len(proposals) >= 2:
                try:
                    # Whole glyph classification for comparison
                    whole_res = engine.classify_crop(image)
                    classifications = [
                        {"class_name": r.class_name, "similarity": r.similarity,
                         "rejected": r.rejected, "top_k": r.top_k}
                        for r in results
                    ]
                    _save_full_viz(
                        image, proposals, classifications, class_name,
                        str(output_dir / f"{class_name}_{img_path.stem}.png"),
                        whole_glyph_pred=whole_res.class_name,
                        whole_glyph_conf=whole_res.similarity,
                    )
                    viz_count += 1
                except Exception:
                    pass  # Don't crash on viz errors

        # Class summary
        avg_seg = class_segments / len(images) if images else 0
        avg_conf = float(np.mean(class_confidences)) if class_confidences else 0
        print(f"\n  Summary:")
        print(f"    Images processed: {len(images)}")
        print(f"    Total segments: {class_segments} (avg {avg_seg:.1f}/glyph)")
        print(f"    Accepted: {class_accepted}, Rejected: {class_rejected}")
        print(f"    Avg confidence: {avg_conf:.3f}")
        print(f"    Top elements: {class_elements.most_common(5)}")

        global_stats["classes"][class_name] = {
            "num_images": len(images),
            "total_segments": class_segments,
            "avg_segments": round(avg_seg, 1),
            "accepted": class_accepted,
            "rejected": class_rejected,
            "avg_confidence": round(avg_conf, 3),
            "top_elements": dict(class_elements.most_common(10)),
            "predictions": class_results,
        }

    # Global summary
    global_stats["total_images"] = total_images
    global_stats["total_segments"] = sum(
        c["total_segments"] for c in global_stats["classes"].values()
    )
    global_stats["total_accepted"] = sum(
        c["accepted"] for c in global_stats["classes"].values()
    )
    global_stats["total_rejected"] = sum(
        c["rejected"] for c in global_stats["classes"].values()
    )
    global_stats["avg_segments_per_glyph"] = round(
        global_stats["total_segments"] / max(total_images, 1), 1
    )
    global_stats["avg_time_per_image"] = round(total_time / max(total_images, 1), 2)

    # Save
    summary_path = output_dir / "sam_full_results.json"
    with open(summary_path, "w") as f:
        json.dump(global_stats, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"COMPLETE")
    print(f"  Images: {total_images}")
    print(f"  Total segments: {global_stats['total_segments']}")
    print(f"  Avg segments/glyph: {global_stats['avg_segments_per_glyph']}")
    print(f"  Accepted: {global_stats['total_accepted']}")
    print(f"  Rejected: {global_stats['total_rejected']}")
    print(f"  Avg time/image: {global_stats['avg_time_per_image']}s")
    print(f"  Total time: {total_time/60:.1f} min")
    print(f"  Results: {summary_path}")
    print(f"{'='*60}")


def _save_full_viz(image, proposals, classifications, class_name, output_path,
                   whole_glyph_pred=None, whole_glyph_conf=None):
    """
    Full visualization: glyphe annote a gauche + crops segmentes a droite.
    Identique au style de test_sam_on_glyphs.py.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import hsv_to_rgb

    n_elements = len(proposals)
    if n_elements == 0:
        return

    n_crops = min(n_elements, 8)
    n_rows = (n_crops + 1) // 2

    fig = plt.figure(figsize=(16, max(6, n_rows * 2.5)))

    # Left: annotated glyph
    ax_main = fig.add_axes([0.02, 0.05, 0.45, 0.85])
    ax_main.imshow(image)
    ax_main.set_axis_off()

    title = f"Glyphe : {class_name}"
    if whole_glyph_pred:
        title += f"\nClassification globale : {whole_glyph_pred} ({whole_glyph_conf:.3f})"
    ax_main.set_title(title, fontsize=10, fontweight="bold")

    colors = [hsv_to_rgb((i / max(n_elements, 1), 0.8, 0.9)) for i in range(n_elements)]

    for i, (prop, clf) in enumerate(zip(proposals[:8], classifications[:8])):
        if clf["rejected"]:
            continue
        color = colors[i]
        x, y, bw, bh = prop["bbox"]
        rect = mpatches.Rectangle(
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


if __name__ == "__main__":
    main()
