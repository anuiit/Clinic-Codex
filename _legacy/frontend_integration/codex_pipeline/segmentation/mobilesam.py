"""
MobileSAM-based region proposal generation for codex manuscript pages.

Architecture:
    MobileSAM (ViT-T backbone) → SamAutomaticMaskGenerator → filter + NMS → RegionProposal list

The segmenter runs SAM's automatic mask generation over the full page image,
filters proposals by area and stability score, then applies non-maximum
suppression (NMS) to remove overlapping duplicates before returning clean
RegionProposal objects ready for embedding.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RegionProposal:
    """
    A single segmented region produced by MobileSAM.

    Attributes:
        bbox: Bounding box in (x, y, w, h) format (image coordinates).
        mask: Binary boolean mask of shape (H, W) covering the full page.
        area: Number of foreground pixels in the mask.
        confidence: SAM predicted-IoU score used as confidence proxy.
        crop: Optional cropped image patch with background masked to white.
    """

    bbox: Tuple[int, int, int, int]
    mask: np.ndarray
    area: int
    confidence: float
    crop: Optional[np.ndarray] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CHECKPOINT_URL = (
    "https://huggingface.co/dhkim2810/MobileSAM/resolve/main/mobile_sam.pt"
)
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "mobile_sam"
_DEFAULT_CHECKPOINT = _DEFAULT_CACHE_DIR / "mobile_sam.pt"


def _resolve_device(device: str) -> str:
    """
    Return the best available device string.

    Priority order: cuda → mps → cpu.  Passing an explicit device string
    other than 'auto' bypasses detection entirely.
    """
    if device != "auto":
        return device

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass

    return "cpu"


def _download_checkpoint(dest: Path) -> None:
    """
    Download the MobileSAM checkpoint to *dest* if it does not already exist.

    Args:
        dest: Local path where the checkpoint will be saved.
    """
    if dest.exists():
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[MobileSAMSegmenter] Downloading MobileSAM checkpoint to {dest} …")

    def _progress(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, 100 * downloaded / total_size)
            print(f"\r  {pct:.1f}%", end="", flush=True)

    urllib.request.urlretrieve(_CHECKPOINT_URL, dest, reporthook=_progress)
    print()  # newline after progress
    print(f"[MobileSAMSegmenter] Checkpoint saved to {dest}")


def _compute_iou(
    bbox_a: Tuple[int, int, int, int],
    bbox_b: Tuple[int, int, int, int],
) -> float:
    """
    Compute Intersection-over-Union for two bounding boxes in (x, y, w, h) format.

    Args:
        bbox_a: First bounding box (x, y, w, h).
        bbox_b: Second bounding box (x, y, w, h).

    Returns:
        IoU score in [0, 1].
    """
    ax, ay, aw, ah = bbox_a
    bx, by, bw, bh = bbox_b

    # Convert to (x1, y1, x2, y2)
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = aw * ah
    area_b = bw * bh
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def _nms(
    proposals: List[RegionProposal],
    iou_threshold: float,
) -> List[RegionProposal]:
    """
    Non-maximum suppression over bounding boxes.

    Proposals must already be sorted by descending confidence before calling
    this function.  A proposal is kept if its IoU with every already-kept
    proposal is below *iou_threshold*.

    Args:
        proposals: Candidate proposals sorted by descending confidence.
        iou_threshold: Maximum allowed IoU with any kept proposal.

    Returns:
        Filtered list of kept proposals.
    """
    kept: List[RegionProposal] = []

    for candidate in proposals:
        suppressed = any(
            _compute_iou(candidate.bbox, kept_prop.bbox) >= iou_threshold
            for kept_prop in kept
        )
        if not suppressed:
            kept.append(candidate)

    return kept


# ---------------------------------------------------------------------------
# Segmenter
# ---------------------------------------------------------------------------


class MobileSAMSegmenter:
    """
    Wrapper around MobileSAM for full-page automatic mask generation.

    On first instantiation (or when no checkpoint_path is given) the model
    weights are automatically downloaded to ~/.cache/mobile_sam/mobile_sam.pt.

    Args:
        checkpoint_path: Path to a local MobileSAM .pt checkpoint.
                         Defaults to ~/.cache/mobile_sam/mobile_sam.pt and
                         auto-downloads if absent.
        device: Compute device — "auto", "cuda", "mps", or "cpu".
        min_area: Minimum mask area in pixels; smaller blobs are discarded.
        max_area_ratio: Maximum mask area as a fraction of the page area;
                        masks covering more than this fraction are discarded
                        (catches background bleed-through).
        min_stability_score: SAM stability score threshold; proposals below
                             this value are discarded.
        iou_threshold: IoU threshold for NMS suppression.
        points_per_side: Grid density for SAM's automatic point prompting.
        pred_iou_thresh: SAM internal predicted-IoU filter threshold.
        stability_score_thresh: SAM internal stability score filter threshold
                                (applied inside the MaskGenerator before our
                                own post-filter).
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "auto",
        min_area: int = 200,
        max_area_ratio: float = 0.25,
        min_stability_score: float = 0.8,
        iou_threshold: float = 0.5,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.86,
        stability_score_thresh: float = 0.92,
    ) -> None:
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio
        self.min_stability_score = min_stability_score
        self.iou_threshold = iou_threshold

        # Resolve checkpoint path — download if necessary
        if checkpoint_path is None:
            _download_checkpoint(_DEFAULT_CHECKPOINT)
            checkpoint_path = str(_DEFAULT_CHECKPOINT)

        self.device = _resolve_device(device)

        # Build MobileSAM model
        from mobile_sam import SamAutomaticMaskGenerator, sam_model_registry

        sam = sam_model_registry["vit_t"](checkpoint=checkpoint_path)
        sam.to(self.device)
        sam.eval()

        self._mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment_page(self, image: np.ndarray) -> List[RegionProposal]:
        """
        Run MobileSAM on a full manuscript page image and return filtered proposals.

        Processing pipeline:
        1. Run SamAutomaticMaskGenerator to obtain raw masks.
        2. Filter by area (min_area ≤ area ≤ max_area_ratio × page_area).
        3. Filter by stability_score ≥ min_stability_score.
        4. Sort by predicted_iou (descending) and apply NMS.

        Args:
            image: RGB or BGR uint8 numpy array of shape (H, W, 3).

        Returns:
            List of RegionProposal objects after filtering and NMS.
        """
        page_area = image.shape[0] * image.shape[1]
        max_area = int(self.max_area_ratio * page_area)

        raw_masks = self._mask_generator.generate(image)

        proposals: List[RegionProposal] = []
        for ann in raw_masks:
            area: int = int(ann["area"])

            # --- Area filter ---
            if area < self.min_area or area > max_area:
                continue

            # --- Stability score filter ---
            stability: float = float(ann.get("stability_score", 1.0))
            if stability < self.min_stability_score:
                continue

            # SAM returns bbox as [x, y, w, h]
            x, y, w, h = (int(v) for v in ann["bbox"])
            confidence: float = float(ann.get("predicted_iou", stability))
            mask: np.ndarray = ann["segmentation"].astype(bool)

            proposals.append(
                RegionProposal(
                    bbox=(x, y, w, h),
                    mask=mask,
                    area=area,
                    confidence=confidence,
                )
            )

        # Sort by confidence descending before NMS
        proposals.sort(key=lambda p: p.confidence, reverse=True)
        proposals = _nms(proposals, self.iou_threshold)

        return proposals

    def extract_crops(
        self,
        image: np.ndarray,
        proposals: List[RegionProposal],
        padding: int = 5,
    ) -> List[RegionProposal]:
        """
        Attach cropped image patches to each RegionProposal.

        For every proposal, the bounding-box region (with *padding* pixels on
        each side, clamped to image boundaries) is cropped from *image*.  The
        pixels outside the binary mask are set to 255 (white background) so
        that the DINOv2 encoder sees a clean foreground-only patch.

        The proposals are updated in-place and also returned for convenience.

        Args:
            image: Full-page image as a uint8 numpy array (H, W, C).
            proposals: List of RegionProposal objects — their `.crop` field
                       will be populated.
            padding: Number of extra pixels to include around the bbox on each
                     side (default: 5).

        Returns:
            The same list of proposals with `.crop` fields filled in.
        """
        img_h, img_w = image.shape[:2]

        for prop in proposals:
            x, y, w, h = prop.bbox

            # Expand bbox with padding, clamped to image bounds
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(img_w, x + w + padding)
            y2 = min(img_h, y + h + padding)

            # Crop image and corresponding mask region
            crop = image[y1:y2, x1:x2].copy()
            mask_crop = prop.mask[y1:y2, x1:x2]

            # Set background pixels to white
            background = ~mask_crop
            if crop.ndim == 3:
                crop[background] = 255
            else:
                crop[background] = 255

            prop.crop = crop

        return proposals
