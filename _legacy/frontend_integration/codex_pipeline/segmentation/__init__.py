"""
codex_pipeline.segmentation
============================

Provides MobileSAM-based page segmentation to produce region proposals that
feed into the DINOv2 embedding and prototypical classification stages.

Public API
----------
MobileSAMSegmenter
    Full-page automatic mask generator with area filtering and NMS.
RegionProposal
    Dataclass representing a single segmented region (bbox, mask, crop, …).
"""

from codex_pipeline.segmentation.mobilesam import MobileSAMSegmenter, RegionProposal

__all__ = ["MobileSAMSegmenter", "RegionProposal"]
