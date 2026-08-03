"""
Spatial grouping of detected elements into glyphs.

After the DINOv2 + prototypical classifier produces a DetectedElement per
region proposal, this module clusters the surviving elements by the
Euclidean distance between their bounding-box centres.  Hierarchical
agglomerative clustering (scipy) with a configurable distance threshold
groups nearby elements into a single DetectedGlyph.

Glyphs are sorted left-to-right by their centre x-coordinate so that the
resulting list follows natural reading order across a manuscript page.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class DetectedElement:
    """
    A single classified region proposal.

    Attributes:
        bbox: Bounding box in (x, y, w, h) image-coordinate format.
        class_name: Predicted element class label (e.g. "atl", "stroke").
        similarity: Cosine similarity score against the nearest prototype.
        rejected: True when the similarity fell below the rejection threshold
                  and the prediction should be treated as unreliable.
    """

    bbox: Tuple[int, int, int, int]
    class_name: str
    similarity: float
    rejected: bool


@dataclass
class DetectedGlyph:
    """
    A spatial cluster of one or more DetectedElement objects.

    Attributes:
        bbox: Tight enclosing bounding box (x, y, w, h) with 5 px padding,
              computed as the union of all member element bounding boxes.
        elements: Detected elements that belong to this glyph cluster.
        glyph_id: Sequential integer identifier assigned after sorting.
    """

    bbox: Tuple[int, int, int, int]
    elements: List[DetectedElement]
    glyph_id: int

    @property
    def center(self) -> Tuple[float, float]:
        """
        Return the geometric centre of the enclosing bounding box.

        Returns:
            (cx, cy) as floats in image coordinates.
        """
        x, y, w, h = self.bbox
        return (x + w / 2.0, y + h / 2.0)

    @property
    def element_names(self) -> List[str]:
        """
        Return an ordered list of class names for all member elements.

        Returns:
            List of class_name strings, one per element in insertion order.
        """
        return [elem.class_name for elem in self.elements]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """Return the centre (cx, cy) of a (x, y, w, h) bounding box."""
    x, y, w, h = bbox
    return (x + w / 2.0, y + h / 2.0)


def _enclosing_bbox(
    elements: List[DetectedElement],
    padding: int = 5,
) -> Tuple[int, int, int, int]:
    """
    Compute the axis-aligned union of all element bounding boxes.

    A uniform *padding* is added to all four sides and clamped to zero so
    the result never goes negative.

    Args:
        elements: Non-empty list of DetectedElement objects.
        padding: Number of pixels to expand the enclosing box on each side.

    Returns:
        Enclosing bounding box in (x, y, w, h) format.
    """
    xs = [e.bbox[0] for e in elements]
    ys = [e.bbox[1] for e in elements]
    x2s = [e.bbox[0] + e.bbox[2] for e in elements]
    y2s = [e.bbox[1] + e.bbox[3] for e in elements]

    x1 = max(0, min(xs) - padding)
    y1 = max(0, min(ys) - padding)
    x2 = max(x2s) + padding
    y2 = max(y2s) + padding

    return (x1, y1, x2 - x1, y2 - y1)


# ---------------------------------------------------------------------------
# Grouping function
# ---------------------------------------------------------------------------


def group_elements_into_glyphs(
    elements: List[DetectedElement],
    distance_threshold: float = 80.0,
    min_elements: int = 1,
    linkage_method: str = "average",
) -> List[DetectedGlyph]:
    """
    Cluster accepted elements into DetectedGlyph objects.

    Algorithm
    ---------
    1. Discard rejected elements.
    2. Compute bbox centres as an (N, 2) float array.
    3. Apply hierarchical agglomerative clustering via
       ``scipy.cluster.hierarchy.linkage`` + ``fcluster`` with
       ``criterion="distance"`` and ``t=distance_threshold``.
    4. Build one DetectedGlyph per cluster, computing its enclosing bbox.
    5. Sort glyphs left-to-right by centre x-coordinate and assign sequential
       IDs starting from 0.

    Edge cases:
    - Zero accepted elements → returns [].
    - Exactly one accepted element → returns a single-element glyph without
      calling scipy (avoids the linkage error for N=1).

    Args:
        elements: All detected elements (accepted *and* rejected); rejected
                  ones are filtered internally.
        distance_threshold: Maximum inter-cluster centroid distance (pixels)
                            for merging two clusters (the ``t`` parameter of
                            ``fcluster``).
        min_elements: Minimum number of elements a cluster must contain to be
                      retained as a glyph.  Defaults to 1 (keep everything).
        linkage_method: Linkage criterion passed to
                        ``scipy.cluster.hierarchy.linkage``.
                        Typical values: "average", "single", "complete",
                        "ward".

    Returns:
        List of DetectedGlyph objects sorted by ascending centre x-coordinate.
    """
    from scipy.cluster.hierarchy import fcluster, linkage

    # --- Step 1: Filter out rejected elements ---
    accepted = [e for e in elements if not e.rejected]

    if len(accepted) == 0:
        return []

    # --- Step 2: Single-element fast path ---
    if len(accepted) == 1:
        glyph = DetectedGlyph(
            bbox=_enclosing_bbox(accepted),
            elements=accepted,
            glyph_id=0,
        )
        return [glyph]

    # --- Step 3: Compute centres and cluster ---
    centres = np.array([_bbox_center(e.bbox) for e in accepted], dtype=np.float64)

    # linkage expects a condensed distance matrix or a (N, M) feature array
    Z = linkage(centres, method=linkage_method, metric="euclidean")
    cluster_ids = fcluster(Z, t=distance_threshold, criterion="distance")
    # cluster_ids are 1-indexed integers

    # --- Step 4: Build glyphs per cluster ---
    cluster_map: dict[int, List[DetectedElement]] = {}
    for elem, cid in zip(accepted, cluster_ids):
        cluster_map.setdefault(int(cid), []).append(elem)

    glyphs: List[DetectedGlyph] = []
    for cid, members in cluster_map.items():
        if len(members) < min_elements:
            continue
        glyphs.append(
            DetectedGlyph(
                bbox=_enclosing_bbox(members),
                elements=members,
                glyph_id=cid,  # temporary; reassigned below
            )
        )

    # --- Step 5: Sort left-to-right, reassign sequential IDs ---
    glyphs.sort(key=lambda g: g.center[0])
    for new_id, glyph in enumerate(glyphs):
        glyph.glyph_id = new_id

    return glyphs
