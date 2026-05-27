import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
} from "react";
import type { UseImageStageViewportResult } from "../../components/ImageBBoxStage";
import type { AnnotationStatus } from "../../types";
import { clientToImage } from "../../utils/imageCoords";
import {
  hitTestBBoxes,
  hitTestHandle,
  isDragIntent,
  moveBBox,
  resizeBBox,
} from "../../utils/segmentationBoxes";
import {
  areBboxesEqual,
  isEditableTarget,
  type BboxHistoryEntry,
  type DragState,
  type PendingMoveState,
  type UseAnnotationViewportOptions,
} from "./annotationViewportTypes";
import { cloneElements } from "./useAnnotationRecord";

export const RESIZE_HANDLE_VISUAL_SIZE = 10;
export const RESIZE_HANDLE_HIT_RADIUS = 6;

type UseBBoxEditingOptions = Pick<
  UseAnnotationViewportOptions,
  | "record"
  | "elements"
  | "setElements"
  | "annotationStatus"
  | "setAnnotationStatus"
  | "focusedIdx"
  | "setFocusedIdx"
  | "setHoveredIdx"
  | "setNamingFocusToken"
> & {
  stageViewport: UseImageStageViewportResult;
  pushBboxHistory: (entry: BboxHistoryEntry) => void;
  undoBboxHistoryChange: (beforeUndo?: () => void) => void;
};

export function useBBoxEditing({
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
  pushBboxHistory,
  undoBboxHistoryChange,
}: UseBBoxEditingOptions) {
  const [drawMode, setDrawMode] = useState(false);
  const [dragState, setDragState] = useState<DragState>(null);
  const [tempBbox, setTempBbox] = useState<[number, number, number, number] | null>(null);
  const rafRef = useRef<number | null>(null);
  const dragStateRef = useRef<DragState>(null);
  const pendingMoveRef = useRef<PendingMoveState>(null);
  const pendingTempBboxRef = useRef<[number, number, number, number] | null>(null);

  const setActiveDragState = useCallback((nextDragState: DragState) => {
    dragStateRef.current = nextDragState;
    setDragState(nextDragState);
  }, []);

  const resetTransientEditState = useCallback(() => {
    setActiveDragState(null);
    setTempBbox(null);
    pendingTempBboxRef.current = null;
    pendingMoveRef.current = null;
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
  }, [setActiveDragState]);

  const undoLastBboxChange = useCallback(() => {
    undoBboxHistoryChange(resetTransientEditState);
  }, [resetTransientEditState, undoBboxHistoryChange]);

  const removeElement = useCallback(
    (idxToRemove: number) => {
      if (!elements[idxToRemove]) return;
      pushBboxHistory({
        type: "delete",
        idx: idxToRemove,
        element: cloneElements([elements[idxToRemove]])[0],
        status: annotationStatus[idxToRemove],
        focusedIdx,
      });
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
    [
      annotationStatus,
      elements,
      focusedIdx,
      pushBboxHistory,
      setAnnotationStatus,
      setElements,
      setFocusedIdx,
    ],
  );

  const scheduleTempBbox = useCallback((bbox: [number, number, number, number]) => {
    pendingTempBboxRef.current = bbox;
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(() => {
      if (pendingTempBboxRef.current) setTempBbox(pendingTempBboxRef.current);
      rafRef.current = null;
    });
  }, []);

  useEffect(() => () => {
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
  }, []);

  const handleSvgPointerDown = (e: ReactPointerEvent<SVGSVGElement>) => {
    if (!record) return;
    const [imgW, imgH] = record.result.image_size;
    const { x, y } = clientToImage(e.currentTarget, e.clientX, e.clientY, { width: imgW, height: imgH });
    const interactionFocusedIdx = focusedIdx;

    if (drawMode) {
      const bbox: [number, number, number, number] = [x, y, 0, 0];
      setActiveDragState({ type: "draw", idx: elements.length, startX: x, startY: y, startClientX: e.clientX, startClientY: e.clientY });
      setTempBbox(bbox);
      pendingTempBboxRef.current = bbox;
      pendingMoveRef.current = null;
      e.currentTarget.setPointerCapture(e.pointerId);
      return;
    }

    const bboxes = elements.map((el) => el.bbox);
    const hitIdx = hitTestBBoxes({ x, y }, bboxes);
    if (hitIdx !== null && hitIdx !== interactionFocusedIdx) {
      setFocusedIdx(hitIdx);
      setHoveredIdx(hitIdx);
      pendingMoveRef.current = null;
      setTempBbox(null);
      pendingTempBboxRef.current = null;
      return;
    }

    const selectedElement =
      interactionFocusedIdx === null ? null : elements[interactionFocusedIdx];
    const selectedHandle =
      selectedElement && interactionFocusedIdx !== null
        ? hitTestHandle({ x, y }, selectedElement.bbox, RESIZE_HANDLE_HIT_RADIUS)
        : null;
    if (selectedHandle && interactionFocusedIdx !== null && selectedElement) {
      const bbox: [number, number, number, number] = [...selectedElement.bbox];
      setFocusedIdx(interactionFocusedIdx);
      setActiveDragState({ type: "resize", idx: interactionFocusedIdx, corner: selectedHandle, startX: x, startY: y, origBbox: bbox });
      setTempBbox(bbox);
      pendingTempBboxRef.current = bbox;
      pendingMoveRef.current = null;
      e.currentTarget.setPointerCapture(e.pointerId);
      return;
    }

    if (hitIdx !== null && hitIdx === interactionFocusedIdx) {
      const bbox: [number, number, number, number] = [...elements[hitIdx].bbox];
      setFocusedIdx(hitIdx);
      pendingMoveRef.current = { type: "move", idx: hitIdx, startX: x, startY: y, startClientX: e.clientX, startClientY: e.clientY, origBbox: bbox };
      setTempBbox(null);
      pendingTempBboxRef.current = null;
      e.currentTarget.setPointerCapture(e.pointerId);
    } else {
      setFocusedIdx(null);
      pendingMoveRef.current = null;
      if (stageViewport.zoom > 1) {
        stageViewport.beginPan(e);
        e.currentTarget.style.cursor = "grabbing";
      }
    }
  };

  const handleSvgPointerMove = (e: ReactPointerEvent<SVGSVGElement>) => {
    if (!record) return;

    if (stageViewport.movePan(e)) {
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
          const handle = hitTestHandle({ x, y }, el.bbox, RESIZE_HANDLE_HIT_RADIUS);
          if (handle === "tl" || handle === "br") cursor = "nwse-resize";
          else if (handle === "tr" || handle === "bl") cursor = "nesw-resize";
        }
      }
      if (cursor === "default" && hitIdx !== null) cursor = "move";
      else if (cursor === "default" && hitIdx === null && stageViewport.zoom > 1) cursor = "grab";
      svg.style.cursor = cursor;
    } else {
      e.currentTarget.style.cursor = "crosshair";
    }
  };

  const handleSvgPointerUp = (e: ReactPointerEvent<SVGSVGElement>) => {
    if (!record) return;

    if (stageViewport.endPan(e)) {
      e.currentTarget.style.cursor = stageViewport.zoom > 1 ? "grab" : "default";
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
    drawMode,
    dragState,
    tempBbox,
    setDrawMode,
    handleSvgPointerDown,
    handleSvgPointerMove,
    handleSvgPointerUp,
    undoLastBboxChange,
    removeElement,
  };
}

export default useBBoxEditing;
