"""
Decompose a glyph into its constituent elements using MobileSAM + CodexClassifier.

Usage:
    python segment_glyph.py path/to/glyph.jpg
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codex_model import CodexClassifier


def main():
    if len(sys.argv) < 2:
        print("Usage: python segment_glyph.py <glyph_image>")
        sys.exit(1)

    from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator

    clf = CodexClassifier()
    sam = sam_model_registry["vit_t"](
        checkpoint=str(Path.home() / ".cache/mobile_sam/mobile_sam.pt")
    )
    sam.eval()
    mask_gen = SamAutomaticMaskGenerator(sam, points_per_side=16,
                                         pred_iou_thresh=0.88,
                                         stability_score_thresh=0.92,
                                         min_mask_region_area=80)

    image = np.array(Image.open(sys.argv[1]).convert("RGB"))
    h, w = image.shape[:2]

    # Whole glyph classification
    whole = clf.classify(image)
    print(f"Whole glyph: {whole['class_name']} (confidence={whole['confidence']:.3f})")

    # Segment + classify each region
    masks = mask_gen.generate(image)
    print(f"\nMobileSAM found {len(masks)} raw masks")

    elements = []
    for m in masks:
        if m["area"] < 50 or m["area"] > h * w * 0.85 or m["stability_score"] < 0.8:
            continue
        x, y, bw, bh = m["bbox"]
        pad = 3
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)
        crop = image[y1:y2, x1:x2].copy()
        crop[~m["segmentation"][y1:y2, x1:x2]] = 255
        result = clf.classify(crop)
        elements.append({**result, "bbox": [x, y, bw, bh], "area": m["area"]})

    elements.sort(key=lambda e: e["area"], reverse=True)

    print(f"After filtering: {len(elements)} elements\n")
    for e in elements:
        status = " [REJECTED]" if e["rejected"] else ""
        print(f"  {e['class_name']:<25} confidence={e['confidence']:.3f}  "
              f"area={e['area']}{status}")


if __name__ == "__main__":
    main()
