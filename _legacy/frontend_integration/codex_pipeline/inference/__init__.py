"""
codex_pipeline.inference
=========================

Inference-time components: classification engine, spatial grouping of detected
elements into glyphs, and multi-format export of detection results.

Public API
----------
InferenceEngine / ClassificationResult
    DINOv2 + prototypical network classification of cropped region proposals.
DetectedElement / DetectedGlyph / group_elements_into_glyphs
    Spatial clustering of classified elements into glyph objects.
export_json / export_csv / render_annotated_page
    Serialise and visualise detection results.

Example::

    from codex_pipeline.inference import (
        InferenceEngine, ClassificationResult,
        DetectedElement, DetectedGlyph, group_elements_into_glyphs,
        export_json, export_csv, render_annotated_page,
    )
"""

from codex_pipeline.inference.engine import ClassificationResult, InferenceEngine
from codex_pipeline.inference.grouping import (
    DetectedElement,
    DetectedGlyph,
    group_elements_into_glyphs,
)
from codex_pipeline.inference.export import (
    export_csv,
    export_json,
    render_annotated_page,
)

__all__ = [
    # engine
    "InferenceEngine",
    "ClassificationResult",
    # grouping
    "DetectedElement",
    "DetectedGlyph",
    "group_elements_into_glyphs",
    # export
    "export_json",
    "export_csv",
    "render_annotated_page",
]
