import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
  type WheelEvent as ReactWheelEvent,
} from "react";
import type { AnnotationStatus } from "../../types";
import { clientToImage } from "../../utils/imageCoords";
import {
  clampZoom,
  nextZoomFromWheel,
  shouldConsumeStageWheel,
} from "../../utils/imageStageZoom";
import {
  hitTestBBoxes,
  hitTestHandles,
  isDragIntent,
  moveBBox,
  resizeBBox,
} from "../../utils/segmentationBoxes";
import { cloneElements } from "./useAnnotationRecord";

import {
  areBboxesEqual,
  isEditableTarget,
  type BboxHistoryEntry,
  type DragState,
  type PendingMoveState,
  type UseAnnotationViewportOptions,
} from "./annotationViewportTypes";
import { useAnnotationPreviewCanvas } from "./useAnnotationPreviewCanvas";
import { useAnnotationStageSize } from "./useAnnotationStageSize";

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
  const [drawMode, setDrawMode] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [panOffset, setPanOffset] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [dragState, setDragState] = useState<DragState>(null);
  const [bboxHistory, setBboxHistory] = useState<BboxHistoryEntry[]>([]);
  const [tempBbox, setTempBbox] = useState<[number, number, number, number] | null>(null);
  const [showLabelNames, setShowLabelNames] = useState(false);

  const { containerRef, stageSize, resolvedStageSize, updateStageSize } = useAnnotationStageSize(record, loading);
  const { imageRef, previewCanvasRef } = useAnnotationPreviewCanvas({ record, elements, focusedIdx, dragState, tempBbox });
  const rafRef = useRef<number | null>(null);
  const dragStateRef = useRef<DragState>(null);
  const bboxHistoryRef = useRef<BboxHistoryEntry[]>([]);
  const pendingMoveRef = useRef<PendingMoveState>(null);
  const pendingTempBboxRef = useRef<[number, number, number, number] | null>(null);
  const panStartRef = useRef<{ clientX: number; clientY: number; offset: { x: number; y: number } } | null>(null);
  const setActiveDragState = (nextDragState: DragState) => {
    dragStateRef.current = nextDragState;
    setDragState(nextDragState);
  };

  const setBboxHistoryEntries = useCallback((nextHistory: BboxHistoryEntry[]) => {
    bboxHistoryRef.current = nextHistory;
    setBboxHistory(nextHistory);
  }, []);

  const pushBboxHistory = useCallback(
    (entry: BboxHistoryEntry) => setBboxHistoryEntries([...bboxHistoryRef.current, entry].slice(-20)),
    [setBboxHistoryEntries],
  );

  const undoLastBboxChange = useCallback(() => {
    const previousHistory = bboxHistoryRef.current;
    const entry = previousHistory[previousHistory.length - 1];
    if (!entry) return;

    setActiveDragState(null);
    setTempBbox(null);
    pendingTempBboxRef.current = null;
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }

    if (entry.type === "create") {
      setElements((prev) => prev.filter((_, idx) => idx !== entry.idx));
      setAnnotationStatus((prev) => {
        const next: Record<number, AnnotationStatus> = {};
        Object.entries(prev).forEach(([key, status]) => {
          const idx = Number(key);
          if (idx < entry.idx) next[idx] = status;
          if (idx > entry.idx) next[idx - 1] = status;
        });
        return next;
      });
      setFocusedIdx(entry.focusedIdx);
    } else if (entry.type === "delete") {
      setElements((prev) => {
        const next = cloneElements(prev);
        next.splice(entry.idx, 0, cloneElements([entry.element])[0]);
        return next;
      });
      setAnnotationStatus((prev) => {
        const next: Record<number, AnnotationStatus> = {};
        Object.entries(prev).forEach(([key, status]) => {
          const idx = Number(key);
          next[idx >= entry.idx ? idx + 1 : idx] = status;
        });
        if (entry.status) next[entry.idx] = entry.status;
        return next;
      });
      setFocusedIdx(entry.idx);
    } else {
      setElements((prev) =>
        prev.map((element, idx) =>
          idx === entry.idx ? { ...element, bbox: [...entry.previousBbox] } : element,
        ),
      );
      setFocusedIdx(entry.idx);
    }

    setBboxHistoryEntries(previousHistory.slice(0, -1));
  }, [setAnnotationStatus, setBboxHistoryEntries, setElements, setFocusedIdx]);

  const removeElement = useCallback(
    (idxToRemove: number) => {
      if (!elements[idxToRemove]) return;
      pushBboxHistory({ type: "delete", idx: idxToRemove, element: cloneElements([elements[idxToRemove]])[0], status: annotationStatus[idxToRemove], focusedIdx });
      setElements((prev) => prev.filter((_, idx) => idx !== idxToRemove));
      setAnnotationStatus((prev) => {
        const next: Record<number, AnnotationStatus> = {};
        Object.entries(prev).forEach(([key, status]) => {
          const idx = Number(key);
          if (idx < idxToRemove) next[idx] = status;
          if (idx > idxToRemove) next[idx - 1] = status;
        });
        return next;
      });
      if (focusedIdx === idxToRemove) setFocusedIdx(null);
      else if (focusedIdx !== null && focusedIdx > idxToRemove) setFocusedIdx(focusedIdx - 1);
    },
    [annotationStatus, elements, focusedIdx, pushBboxHistory, setAnnotationStatus, setElements, setFocusedIdx],
  );

  const scheduleTempBbox = (bbox: [number, number, number, number]) => {
    pendingTempBboxRef.current = bbox;
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(() => {
      if (pendingTempBboxRef.current) setTempBbox(pendingTempBboxRef.current);
      rafRef.current = null;
    });
  };

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- reset edit history when a different record is loaded
    setBboxHistoryEntries([]);
  }, [record?.id, setBboxHistoryEntries]);

  useEffect(() => () => {
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
  }, []);

  useEffect(() => {
    if (zoom <= 1) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- keep pan state clamped when zoom returns to fit view
      setPanOffset({ x: 0, y: 0 });
      setIsPanning(false);
      panStartRef.current = null;
    }
  }, [zoom]);

  const clampPan = useCallback(
    (offset: { x: number; y: number }, zoomLevel = zoom): { x: number; y: number } => {
      if (!containerRef.current || !record || zoomLevel <= 1) return { x: 0, y: 0 };

      const cRect = containerRef.current.getBoundingClientRect();
      const stage = stageSize ?? { width: record.result.image_size[0], height: record.result.image_size[1] };
      const scaledW = stage.width * zoomLevel;
      const scaledH = stage.height * zoomLevel;
      const maxPanX = scaledW / 2 + cRect.width / 2 - scaledW * 0.2;
      const maxPanY = scaledH / 2 + cRect.height / 2 - scaledH * 0.2;

      return { x: Math.max(-maxPanX, Math.min(maxPanX, offset.x)), y: Math.max(-maxPanY, Math.min(maxPanY, offset.y)) };
    },
    [containerRef, record, stageSize, zoom],
  );

  const applyZoom = (nextZoom: number, anchor?: { clientX: number; clientY: number }) => {
    const clampedZoom = clampZoom(nextZoom);

    if (!anchor || !containerRef.current || zoom === 0) {
      setZoom(clampedZoom);
      setPanOffset((prev) => clampPan(prev, clampedZoom));
      return;
    }

    const cRect = containerRef.current.getBoundingClientRect();
    const centerX = cRect.left + cRect.width / 2;
    const centerY = cRect.top + cRect.height / 2;
    const cursorRelX = anchor.clientX - centerX;
    const cursorRelY = anchor.clientY - centerY;
    const zoomRatio = clampedZoom / zoom;

    setZoom(clampedZoom);
    setPanOffset((prev) => clampPan({ x: cursorRelX - (cursorRelX - prev.x) * zoomRatio, y: cursorRelY - (cursorRelY - prev.y) * zoomRatio }, clampedZoom));
  };

  const handleStageWheel = (event: ReactWheelEvent<HTMLDivElement>) => {
    if (!shouldConsumeStageWheel(event.deltaY)) return;
    event.preventDefault();
    applyZoom(nextZoomFromWheel(zoom, event.deltaY), { clientX: event.clientX, clientY: event.clientY });
  };

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- reclamp pan after measured stage or zoom changes
    setPanOffset((prev) => clampPan(prev, zoom));
  }, [clampPan, stageSize, zoom]);

  const handleSvgPointerDown = (e: ReactPointerEvent<SVGSVGElement>) => {
    if (!record) return;
    const [imgW, imgH] = record.result.image_size;
    const { x, y } = clientToImage(e.currentTarget, e.clientX, e.clientY, { width: imgW, height: imgH });

    if (drawMode) {
      const bbox: [number, number, number, number] = [x, y, 0, 0];
      setActiveDragState({ type: "draw", idx: elements.length, startX: x, startY: y, startClientX: e.clientX, startClientY: e.clientY });
      setTempBbox(bbox);
      pendingTempBboxRef.current = bbox;
      pendingMoveRef.current = null;
      e.currentTarget.setPointerCapture(e.pointerId);
      return;
    }

    const handleHit = hitTestHandles({ x, y }, elements.map((el) => el.bbox), 12);
    if (handleHit) {
      const bbox: [number, number, number, number] = [...elements[handleHit.idx].bbox];
      setFocusedIdx(handleHit.idx);
      setActiveDragState({ type: "resize", idx: handleHit.idx, corner: handleHit.handle, startX: x, startY: y, origBbox: bbox });
      setTempBbox(bbox);
      pendingTempBboxRef.current = bbox;
      pendingMoveRef.current = null;
      e.currentTarget.setPointerCapture(e.pointerId);
      return;
    }

    const hitIdx = hitTestBBoxes({ x, y }, elements.map((el) => el.bbox));
    if (hitIdx !== null) {
      const bbox: [number, number, number, number] = [...elements[hitIdx].bbox];
      setFocusedIdx(hitIdx);
      pendingMoveRef.current = { type: "move", idx: hitIdx, startX: x, startY: y, startClientX: e.clientX, startClientY: e.clientY, origBbox: bbox };
      setTempBbox(null);
      pendingTempBboxRef.current = null;
      e.currentTarget.setPointerCapture(e.pointerId);
    } else {
      setFocusedIdx(null);
      pendingMoveRef.current = null;
      if (zoom > 1) {
        setIsPanning(true);
        panStartRef.current = { clientX: e.clientX, clientY: e.clientY, offset: { ...panOffset } };
        e.currentTarget.style.cursor = "grabbing";
        e.currentTarget.setPointerCapture(e.pointerId);
      }
    }
  };

  const handleSvgPointerMove = (e: ReactPointerEvent<SVGSVGElement>) => {
    if (!record) return;

    if (isPanning && panStartRef.current) {
      const dx = e.clientX - panStartRef.current.clientX;
      const dy = e.clientY - panStartRef.current.clientY;
      setPanOffset(clampPan({ x: panStartRef.current.offset.x + dx, y: panStartRef.current.offset.y + dy }));
      e.currentTarget.style.cursor = "grabbing";
      return;
    }

    const [imgW, imgH] = record.result.image_size;
    const { x, y } = clientToImage(e.currentTarget, e.clientX, e.clientY, { width: imgW, height: imgH });
    const pendingMove = pendingMoveRef.current;
    const activeDragState = dragStateRef.current ?? dragState;
    const dragStateToUse = activeDragState ?? (pendingMove && isDragIntent({ x: pendingMove.startClientX, y: pendingMove.startClientY }, { x: e.clientX, y: e.clientY }, 5) ? pendingMove : null);

    if (!activeDragState && pendingMove && dragStateToUse) {
      setActiveDragState({ type: "move", idx: pendingMove.idx, startX: pendingMove.startX, startY: pendingMove.startY, origBbox: [...pendingMove.origBbox] });
      pendingMoveRef.current = null;
    }

    if (dragStateToUse) {
      let nextBbox: [number, number, number, number] | null = null;
      if (dragStateToUse.type === "draw") {
        const minX = Math.min(dragStateToUse.startX, x);
        const minY = Math.min(dragStateToUse.startY, y);
        const maxX = Math.max(dragStateToUse.startX, x);
        const maxY = Math.max(dragStateToUse.startY, y);
        nextBbox = [Math.max(0, minX), Math.max(0, minY), Math.min(imgW - Math.max(0, minX), maxX - minX), Math.min(imgH - Math.max(0, minY), maxY - minY)];
      } else if (dragStateToUse.type === "move" && dragStateToUse.origBbox) {
        nextBbox = moveBBox(dragStateToUse.origBbox, { x: x - dragStateToUse.startX, y: y - dragStateToUse.startY }, { width: imgW, height: imgH });
      } else if (dragStateToUse.type === "resize" && dragStateToUse.origBbox && dragStateToUse.corner) {
        nextBbox = resizeBBox(dragStateToUse.origBbox, dragStateToUse.corner, { x, y }, { width: imgW, height: imgH });
      }
      if (nextBbox) scheduleTempBbox(nextBbox);
    } else if (!drawMode) {
      const hitIdx = hitTestBBoxes({ x, y }, elements.map((el) => el.bbox));
      setHoveredIdx(hitIdx);

      const svg = e.currentTarget;
      let cursor = "default";
      if (focusedIdx !== null) {
        const el = elements[focusedIdx];
        if (el) {
          const handle = hitTestHandles({ x, y }, [el.bbox], 12);
          if (handle?.handle === "tl" || handle?.handle === "br") cursor = "nwse-resize";
          else if (handle?.handle === "tr" || handle?.handle === "bl") cursor = "nesw-resize";
        }
      }
      if (cursor === "default" && hitIdx !== null) cursor = "move";
      else if (cursor === "default" && hitIdx === null && zoom > 1) cursor = "grab";
      svg.style.cursor = cursor;
    } else {
      e.currentTarget.style.cursor = "crosshair";
    }
  };

  const handleSvgPointerUp = (e: ReactPointerEvent<SVGSVGElement>) => {
    if (!record) return;

    if (isPanning) {
      setIsPanning(false);
      panStartRef.current = null;
      if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId);
      e.currentTarget.style.cursor = zoom > 1 ? "grab" : "default";
      return;
    }

    if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId);
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }

    const finalTempBbox = pendingTempBboxRef.current ?? tempBbox;
    const activeDragState = dragStateRef.current ?? dragState;
    if (!activeDragState || !finalTempBbox) {
      if (activeDragState) setActiveDragState(null);
      pendingMoveRef.current = null;
      pendingTempBboxRef.current = null;
      return;
    }

    const newElements = [...elements];
    let shouldCommitElements = false;
    if (activeDragState.type === "draw") {
      if (finalTempBbox[2] > 5 && finalTempBbox[3] > 5) {
        pushBboxHistory({ type: "create", idx: newElements.length, focusedIdx });
        newElements.push({ bbox: finalTempBbox, class_name: "", class_label: 0, confidence: 1.0, top_k: [], rejected: false });
        setAnnotationStatus((prev) => ({ ...prev, [newElements.length - 1]: "draft" }));
        setFocusedIdx(newElements.length - 1);
        setNamingFocusToken((current) => current + 1);
        shouldCommitElements = true;
      }
    } else if ((activeDragState.type === "move" || activeDragState.type === "resize") && activeDragState.idx < newElements.length) {
      const originalBbox = newElements[activeDragState.idx].bbox;
      if (!areBboxesEqual(originalBbox, finalTempBbox)) {
        pushBboxHistory({ type: "update", idx: activeDragState.idx, previousBbox: [...originalBbox], focusedIdx });
        newElements[activeDragState.idx] = { ...newElements[activeDragState.idx], bbox: finalTempBbox };
        setAnnotationStatus((prev) => ({ ...prev, [activeDragState.idx]: "draft" }));
        shouldCommitElements = true;
      }
    }

    if (shouldCommitElements) setElements(newElements);
    setActiveDragState(null);
    setTempBbox(null);
    pendingMoveRef.current = null;
    pendingTempBboxRef.current = null;
  };

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (isEditableTarget(e.target)) return;
      if ((e.metaKey || e.ctrlKey) && !e.shiftKey && e.key.toLowerCase() === "z") {
        e.preventDefault();
        undoLastBboxChange();
        return;
      }
      if ((e.key === "Delete" || e.key === "Backspace") && focusedIdx !== null) removeElement(focusedIdx);
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [focusedIdx, removeElement, undoLastBboxChange]);

  return {
    containerRef,
    imageRef,
    previewCanvasRef,
    drawMode,
    zoom,
    panOffset,
    isPanning,
    dragState,
    bboxHistory,
    tempBbox,
    showLabelNames,
    resolvedStageSize,
    setDrawMode,
    setShowLabelNames,
    updateStageSize,
    applyZoom,
    handleStageWheel,
    handleSvgPointerDown,
    handleSvgPointerMove,
    handleSvgPointerUp,
    undoLastBboxChange,
    removeElement,
  };
}
