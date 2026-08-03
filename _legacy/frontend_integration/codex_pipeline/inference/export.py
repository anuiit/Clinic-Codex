"""
Export utilities for codex pipeline detection results.

Provides three output formats:

JSON
    Structured document with page metadata, glyph bounding boxes, and
    per-element class/similarity information.  Suitable for downstream
    processing and archiving.

CSV
    Flat, one-row-per-element table that is easy to load into pandas or
    spreadsheet tools for statistical analysis.

Annotated image
    Matplotlib figure where glyph enclosing boxes are drawn as thick dashed
    rectangles (one HSV colour per glyph) and element boxes as thin
    semi-transparent filled rectangles in the same colour.  Optional text
    labels show class name and similarity score.
"""

from __future__ import annotations

import colorsys
import csv
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from codex_pipeline.inference.grouping import DetectedElement, DetectedGlyph


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------


def export_json(
    glyphs: List[DetectedGlyph],
    output_path: str,
    page_name: str = "unknown",
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Serialise detection results to a JSON file.

    Output schema::

        {
          "page": "<page_name>",
          "num_glyphs": <int>,
          "num_elements": <int>,
          "metadata": { ... },          // optional, may be null
          "glyphs": [
            {
              "glyph_id": 0,
              "bbox": [x, y, w, h],
              "num_elements": <int>,
              "elements": [
                {
                  "class_name": "atl",
                  "bbox": [x, y, w, h],
                  "similarity": 0.92
                },
                ...
              ]
            },
            ...
          ]
        }

    Args:
        glyphs: Ordered list of DetectedGlyph objects to serialise.
        output_path: Destination file path (created or overwritten).
        page_name: Human-readable identifier for the source page.
        metadata: Optional dictionary of arbitrary extra fields appended to
                  the top-level JSON object.
    """
    total_elements = sum(len(g.elements) for g in glyphs)

    document: Dict[str, Any] = {
        "page": page_name,
        "num_glyphs": len(glyphs),
        "num_elements": total_elements,
        "metadata": metadata,
        "glyphs": [],
    }

    for glyph in glyphs:
        glyph_entry: Dict[str, Any] = {
            "glyph_id": glyph.glyph_id,
            "bbox": list(glyph.bbox),
            "num_elements": len(glyph.elements),
            "elements": [],
        }

        for elem in glyph.elements:
            glyph_entry["elements"].append(
                {
                    "class_name": elem.class_name,
                    "bbox": list(elem.bbox),
                    "similarity": round(float(elem.similarity), 6),
                }
            )

        document["glyphs"].append(glyph_entry)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(document, fh, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def export_csv(
    glyphs: List[DetectedGlyph],
    output_path: str,
    page_name: str = "unknown",
) -> None:
    """
    Write a flat CSV with one row per detected element.

    Column schema:

    +------------------+------------------------------------------------------+
    | Column           | Description                                          |
    +==================+======================================================+
    | page             | Source page identifier.                              |
    | glyph_id         | Integer ID of the parent glyph.                      |
    | glyph_x          | Glyph enclosing bbox x (top-left).                   |
    | glyph_y          | Glyph enclosing bbox y (top-left).                   |
    | glyph_w          | Glyph enclosing bbox width.                          |
    | glyph_h          | Glyph enclosing bbox height.                         |
    | element_class    | Predicted class name.                                |
    | elem_x           | Element bbox x (top-left).                           |
    | elem_y           | Element bbox y (top-left).                           |
    | elem_w           | Element bbox width.                                  |
    | elem_h           | Element bbox height.                                 |
    | similarity       | Prototype similarity score.                          |
    +------------------+------------------------------------------------------+

    Args:
        glyphs: Ordered list of DetectedGlyph objects to export.
        output_path: Destination .csv file path (created or overwritten).
        page_name: Human-readable identifier for the source page.
    """
    fieldnames = [
        "page",
        "glyph_id",
        "glyph_x",
        "glyph_y",
        "glyph_w",
        "glyph_h",
        "element_class",
        "elem_x",
        "elem_y",
        "elem_w",
        "elem_h",
        "similarity",
    ]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for glyph in glyphs:
            gx, gy, gw, gh = glyph.bbox
            for elem in glyph.elements:
                ex, ey, ew, eh = elem.bbox
                writer.writerow(
                    {
                        "page": page_name,
                        "glyph_id": glyph.glyph_id,
                        "glyph_x": gx,
                        "glyph_y": gy,
                        "glyph_w": gw,
                        "glyph_h": gh,
                        "element_class": elem.class_name,
                        "elem_x": ex,
                        "elem_y": ey,
                        "elem_w": ew,
                        "elem_h": eh,
                        "similarity": round(float(elem.similarity), 6),
                    }
                )


# ---------------------------------------------------------------------------
# Annotated image rendering
# ---------------------------------------------------------------------------


def _hsv_palette(n: int) -> List[tuple]:
    """
    Generate *n* visually distinct RGB colours using the HSV colour wheel.

    Colours are evenly spaced in hue with full saturation and value so that
    they remain vivid against both light and dark backgrounds.

    Args:
        n: Number of colours to generate.

    Returns:
        List of (r, g, b) float tuples in [0, 1].
    """
    if n == 0:
        return []
    return [colorsys.hsv_to_rgb(i / n, 0.85, 0.95) for i in range(n)]


def render_annotated_page(
    image: np.ndarray,
    glyphs: List[DetectedGlyph],
    output_path: str,
    show_elements: bool = True,
    show_glyphs: bool = True,
    show_labels: bool = True,
    figsize: tuple = (16, 20),
) -> None:
    """
    Render and save an annotated page image with glyph and element overlays.

    Visual conventions
    ------------------
    * Each glyph is assigned a distinct HSV colour.
    * Glyph enclosing boxes: thick (lw=2) dashed rectangle in the glyph
      colour.
    * Element boxes: thin (lw=1) solid rectangle with a semi-transparent
      (alpha=0.15) filled patch in the same glyph colour.
    * Labels (when *show_labels* is True): ``<class_name> (<similarity>)``
      drawn in white text on a coloured background box near the element's
      top-left corner.

    Args:
        image: Full-page image as a uint8 numpy array (H, W, 3).
        glyphs: Ordered list of DetectedGlyph objects to overlay.
        output_path: Destination image file path (format inferred from
                     extension by matplotlib, e.g. .png, .pdf).
        show_elements: Whether to draw per-element bounding boxes.
        show_glyphs: Whether to draw glyph enclosing bounding boxes.
        show_labels: Whether to annotate elements with class/similarity text.
        figsize: Matplotlib figure size in inches (width, height).
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    colours = _hsv_palette(len(glyphs))

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(image, aspect="equal")
    ax.axis("off")

    for glyph, colour in zip(glyphs, colours):
        # --- Glyph enclosing box ---
        if show_glyphs:
            gx, gy, gw, gh = glyph.bbox
            glyph_rect = mpatches.FancyBboxPatch(
                (gx, gy),
                gw,
                gh,
                boxstyle="square,pad=0",
                linewidth=2.0,
                edgecolor=colour,
                facecolor="none",
                linestyle="--",
                zorder=3,
            )
            ax.add_patch(glyph_rect)

            # Glyph ID label at top-left of enclosing box
            if show_labels:
                ax.text(
                    gx + 2,
                    gy - 4,
                    f"G{glyph.glyph_id}",
                    fontsize=7,
                    fontweight="bold",
                    color="white",
                    bbox=dict(
                        facecolor=colour,
                        edgecolor="none",
                        alpha=0.85,
                        pad=1.5,
                        boxstyle="round,pad=0.2",
                    ),
                    zorder=5,
                )

        # --- Element boxes ---
        if show_elements:
            for elem in glyph.elements:
                ex, ey, ew, eh = elem.bbox

                # Semi-transparent filled patch
                fill_patch = mpatches.Rectangle(
                    (ex, ey),
                    ew,
                    eh,
                    linewidth=0,
                    edgecolor="none",
                    facecolor=colour,
                    alpha=0.15,
                    zorder=2,
                )
                ax.add_patch(fill_patch)

                # Solid thin border
                border_patch = mpatches.Rectangle(
                    (ex, ey),
                    ew,
                    eh,
                    linewidth=1.0,
                    edgecolor=colour,
                    facecolor="none",
                    zorder=4,
                )
                ax.add_patch(border_patch)

                # Text label
                if show_labels:
                    label = f"{elem.class_name} ({elem.similarity:.2f})"
                    ax.text(
                        ex + 2,
                        ey + 2,
                        label,
                        fontsize=5,
                        color="white",
                        verticalalignment="top",
                        bbox=dict(
                            facecolor=colour,
                            edgecolor="none",
                            alpha=0.75,
                            pad=1.0,
                            boxstyle="round,pad=0.15",
                        ),
                        zorder=6,
                    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
