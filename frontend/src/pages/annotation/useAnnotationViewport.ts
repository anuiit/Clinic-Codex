import { useState } from "react";
import { useImageStageViewport } from "../../components/ImageBBoxStage";
import { useAnnotationPreviewCanvas } from "./useAnnotationPreviewCanvas";
import { useBBoxEditing } from "./useBBoxEditing";
import { useBBoxHistory } from "./useBBoxHistory";
import type { UseAnnotationViewportOptions } from "./annotationViewportTypes";

export function useAnnotationViewport({
  record,
  elements,
  setElements,
  annotationStatus,
  setAnnotationStatus,
  focusedIdx,
  setFocusedIdx,
  setHoveredIdx,
  setNamingFocusToken,
  loading,
}: UseAnnotationViewportOptions) {
  const [showLabelNames, setShowLabelNames] = useState(false);
  const stageViewport = useImageStageViewport({
    imageSize: record?.result.image_size,
    disabled: !record,
    loading,
    resetKey: record?.id ?? null,
  });
  const history = useBBoxHistory({
    resetKey: record?.id ?? null,
    setElements,
    setAnnotationStatus,
    setFocusedIdx,
  });
  const editing = useBBoxEditing({
    record,
    elements,
    setElements,
    annotationStatus,
    setAnnotationStatus,
    focusedIdx,
    setFocusedIdx,
    setHoveredIdx,
    setNamingFocusToken,
    stageViewport,
    pushBboxHistory: history.pushBboxHistory,
    undoBboxHistoryChange: history.undoLastBboxChange,
  });
  const { imageRef, previewCanvasRef, handlePreviewImageLoad } = useAnnotationPreviewCanvas({
    record,
    elements,
    focusedIdx,
    dragState: editing.dragState,
    tempBbox: editing.tempBbox,
  });

  return {
    containerRef: stageViewport.containerRef,
    imageRef,
    previewCanvasRef,
    handlePreviewImageLoad,
    drawMode: editing.drawMode,
    zoom: stageViewport.zoom,
    panOffset: stageViewport.panOffset,
    isPanning: stageViewport.isPanning,
    dragState: editing.dragState,
    bboxHistory: history.bboxHistory,
    tempBbox: editing.tempBbox,
    showLabelNames,
    transformSize: stageViewport.transformSize,
    setDrawMode: editing.setDrawMode,
    setShowLabelNames,
    updateStageSize: stageViewport.updateStageSize,
    zoomIn: stageViewport.zoomIn,
    zoomOut: stageViewport.zoomOut,
    resetAnnotationView: stageViewport.resetView,
    handleStageWheel: stageViewport.handleStageWheel,
    handleSvgPointerDown: editing.handleSvgPointerDown,
    handleSvgPointerMove: editing.handleSvgPointerMove,
    handleSvgPointerUp: editing.handleSvgPointerUp,
    undoLastBboxChange: editing.undoLastBboxChange,
    removeElement: editing.removeElement,
  };
}

export default useAnnotationViewport;
