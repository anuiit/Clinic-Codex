import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
  type WheelEvent as ReactWheelEvent,
} from "react";
import {
  useParams,
  Link,
  useNavigate,
  useSearchParams,
} from "react-router-dom";
import { ArrowLeft, Loader2, ZoomIn, ZoomOut, Maximize2 } from "lucide-react";
import { getAnalysisById, updateElements } from "../services/storage";
import { getClasses, saveAnnotation } from "../services/api";
import { t as translate } from "../i18n/annotation.fr";
import { MainImagePanel } from "../components/MainImagePanel";
import { AnnotationAnalyzerToolbar } from "./AnnotationAnalyzerToolbar";
import { AnnotationElementList } from "./AnnotationElementList";
import { AnnotationPageChrome } from "./AnnotationPageChrome";
import { AnnotationOverlay } from "./annotation/AnnotationOverlay";
import { AnnotationSelectedInspector } from "./annotation/AnnotationSelectedInspector";
import { AnnotationToast } from "./annotation/AnnotationToast";
import { appText } from "../i18n/text";
import type {
  AnalysisRecord,
  AnnotationStatus,
  DetectedElement,
  SaveAnnotationResult,
} from "../types";
import { clientToImage } from "../utils/imageCoords";
import {
  clampZoom,
  nextZoomFromWheel,
  shouldConsumeStageWheel,
} from "../utils/imageStageZoom";
import {
  hitTestBBoxes,
  hitTestHandles,
  isDragIntent,
  moveBBox,
  resizeBBox,
  type BBox,
  type BBoxHandle,
} from "../utils/segmentationBoxes";
import {
  hasExactClassName,
  isUnnamedClass,
  normalizeClassName,
} from "../utils/fuzzyClasses";

type StageSize = { width: number; height: number };
type AnnotationStatusFilter = "all" | "draft" | "submitted" | "rejected";
type AnnotationSortMode =
  | "original"
  | "name"
  | "confidence-asc"
  | "confidence-desc";

type BboxHistoryEntry =
  | { type: "create"; idx: number; focusedIdx: number | null }
  | {
      type: "delete";
      idx: number;
      element: DetectedElement;
      status?: AnnotationStatus;
      focusedIdx: number | null;
    }
  | {
      type: "update";
      idx: number;
      previousBbox: BBox;
      focusedIdx: number | null;
    };

function cloneElements(elements: DetectedElement[]): DetectedElement[] {
  return elements.map((element) => ({
    ...element,
    bbox: [...element.bbox],
    top_k: element.top_k.map((item) => ({ ...item })),
  }));
}

function areBboxesEqual(
  a: [number, number, number, number],
  b: [number, number, number, number],
): boolean {
  return a.every((value, idx) => value === b[idx]);
}

function isEditableTarget(target: EventTarget | null): boolean {
  return (
    target instanceof HTMLInputElement ||
    target instanceof HTMLSelectElement ||
    target instanceof HTMLTextAreaElement ||
    (target instanceof HTMLElement && target.isContentEditable)
  );
}

type DragState = {
  type: "draw" | "move" | "resize";
  idx: number;
  corner?: BBoxHandle;
  startX: number;
  startY: number;
  startClientX?: number;
  startClientY?: number;
  isMoveReady?: boolean;
  origBbox?: BBox;
} | null;

type PendingMoveState = {
  type: "move";
  idx: number;
  startX: number;
  startY: number;
  startClientX: number;
  startClientY: number;
  origBbox: BBox;
} | null;

export default function AnnotationPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const cardRefs = useRef<Array<HTMLElement | null>>([]);

  const elementParam = searchParams.get("element");
  const initialFocusedIdx =
    elementParam !== null && !Number.isNaN(Number(elementParam))
      ? Number(elementParam)
      : null;

  const [record, setRecord] = useState<AnalysisRecord | null>(null);
  const [elements, setElements] = useState<DetectedElement[]>([]);
  const [annotationStatus, setAnnotationStatus] = useState<
    Record<number, AnnotationStatus>
  >({});
  const [classes, setClasses] = useState<string[]>([]);
  const [customClasses, setCustomClasses] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [sending, setSending] = useState(false);
  const [toast, setToast] = useState<{ msg: string; ok: boolean } | null>(null);

  const [focusedIdx, setFocusedIdx] = useState<number | null>(
    initialFocusedIdx,
  );
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const [listHoveredIdx, setListHoveredIdx] = useState<number | null>(null);
  const [drawMode, setDrawMode] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [panOffset, setPanOffset] = useState<{ x: number; y: number }>({
    x: 0,
    y: 0,
  });
  const [isPanning, setIsPanning] = useState(false);
  const [dragState, setDragState] = useState<DragState>(null);
  const [bboxHistory, setBboxHistory] = useState<BboxHistoryEntry[]>([]);
  const [tempBbox, setTempBbox] = useState<
    [number, number, number, number] | null
  >(null);
  const [namingFocusToken, setNamingFocusToken] = useState(0);
  const [stageSize, setStageSize] = useState<StageSize | null>(null);
  const [showLabelNames, setShowLabelNames] = useState(false);
  const [statusFilter, setStatusFilter] =
    useState<AnnotationStatusFilter>("all");
  const [sortMode, setSortMode] = useState<AnnotationSortMode>("original");
  const [listQuery, setListQuery] = useState("");

  const imageRef = useRef<HTMLImageElement>(null);
  const previewCanvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number | null>(null);
  const dragStateRef = useRef<DragState>(null);
  const bboxHistoryRef = useRef<BboxHistoryEntry[]>([]);
  const pendingMoveRef = useRef<PendingMoveState>(null);
  const pendingTempBboxRef = useRef<[number, number, number, number] | null>(
    null,
  );
  const panStartRef = useRef<{
    clientX: number;
    clientY: number;
    offset: { x: number; y: number };
  } | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const t = appText.annotation;

  const getStageSize = useCallback((): StageSize | null => {
    if (!record) return null;
    const [imgW, imgH] = record.result.image_size;
    if (imgW <= 0 || imgH <= 0) return null;

    const rect = containerRef.current?.getBoundingClientRect();
    const availableW = rect?.width && rect.width > 0 ? rect.width : imgW;
    const availableH = rect?.height && rect.height > 0 ? rect.height : imgH;
    const scale = Math.min(1, availableW / imgW, availableH / imgH);

    return {
      width: Math.max(1, Math.round(imgW * scale)),
      height: Math.max(1, Math.round(imgH * scale)),
    };
  }, [record]);

  const updateStageSize = useCallback(() => {
    const nextSize = getStageSize();
    if (!nextSize) {
      setStageSize(null);
      return;
    }

    setStageSize((current) =>
      current?.width === nextSize.width && current?.height === nextSize.height
        ? current
        : nextSize,
    );
  }, [getStageSize]);

  const resolvedStageSize =
    stageSize ??
    (record
      ? {
          width: record.result.image_size[0],
          height: record.result.image_size[1],
        }
      : null);

  const setActiveDragState = (nextDragState: DragState) => {
    dragStateRef.current = nextDragState;
    setDragState(nextDragState);
  };

  const setBboxHistoryEntries = useCallback(
    (nextHistory: BboxHistoryEntry[]) => {
      bboxHistoryRef.current = nextHistory;
      setBboxHistory(nextHistory);
    },
    [],
  );

  const pushBboxHistory = useCallback(
    (entry: BboxHistoryEntry) => {
      setBboxHistoryEntries([...bboxHistoryRef.current, entry].slice(-20));
    },
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
          idx === entry.idx
            ? { ...element, bbox: [...entry.previousBbox] }
            : element,
        ),
      );
      setFocusedIdx(entry.idx);
    }

    setBboxHistoryEntries(previousHistory.slice(0, -1));
  }, [setBboxHistoryEntries]);

  const commitElementName = (idx: number, nextName: string) => {
    const normalizedName = normalizeClassName(nextName);
    if (!normalizedName) return;
    const previousName = normalizeClassName(elements[idx]?.class_name ?? "");
    if (previousName === normalizedName) return;

    setElements((prev) =>
      prev.map((el, elementIdx) =>
        elementIdx === idx ? { ...el, class_name: normalizedName } : el,
      ),
    );
    setAnnotationStatus((prev) => ({ ...prev, [idx]: "draft" }));

    setCustomClasses((prev) =>
      hasExactClassName(normalizedName, [...classes, ...prev])
        ? prev
        : [...prev, normalizedName],
    );
  };

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
      if (focusedIdx === idxToRemove) {
        setFocusedIdx(null);
      } else if (focusedIdx !== null && focusedIdx > idxToRemove) {
        setFocusedIdx(focusedIdx - 1);
      }
    },
    [annotationStatus, elements, focusedIdx, pushBboxHistory],
  );

  const setElementValidation = (idx: number, status: AnnotationStatus) => {
    setAnnotationStatus((prev) => ({ ...prev, [idx]: status }));
  };

  const submitNamedElements = () => {
    setAnnotationStatus((prev) => {
      const next: Record<number, AnnotationStatus> = { ...prev };
      elements.forEach((el, idx) => {
        next[idx] = isUnnamedClass(el.class_name) ? "draft" : "validated";
      });
      return next;
    });
  };

  const scheduleTempBbox = (bbox: [number, number, number, number]) => {
    pendingTempBboxRef.current = bbox;
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
    }
    rafRef.current = requestAnimationFrame(() => {
      if (pendingTempBboxRef.current) {
        setTempBbox(pendingTempBboxRef.current);
      }
      rafRef.current = null;
    });
  };

  useEffect(() => {
    let active = true;

    async function loadData() {
      if (!id) {
        if (active) {
          setLoading(false);
        }
        return;
      }

      setLoading(true);
      const rec = await getAnalysisById(id);
      if (!active) return;
      setRecord(rec);

      if (!rec) {
        setLoading(false);
        return;
      }
      // Deep copy elements so we can mutate safely
      setElements(cloneElements(rec.result.elements));
      setAnnotationStatus(rec.annotationStatus ?? {});
      setBboxHistoryEntries([]);

      try {
        const classesResult = await getClasses();
        if (active) {
          setClasses(classesResult.class_names);
        }
      } catch {
        // failed to load classes
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }
    loadData();

    return () => {
      active = false;
    };
  }, [id, setBboxHistoryEntries]);

  useEffect(() => {
    if (!record) {
      setStageSize(null);
      return;
    }
    if (loading || !containerRef.current) return;

    updateStageSize();

    const container = containerRef.current;
    const handleResize = () => updateStageSize();
    window.addEventListener("resize", handleResize);

    const observer =
      typeof ResizeObserver === "undefined"
        ? null
        : new ResizeObserver(() => updateStageSize());
    observer?.observe(container);

    return () => {
      window.removeEventListener("resize", handleResize);
      observer?.disconnect();
    };
  }, [loading, record, updateStageSize]);

  useEffect(() => {
    if (
      !record ||
      !imageRef.current ||
      !previewCanvasRef.current ||
      focusedIdx === null
    )
      return;
    const el = elements[focusedIdx];
    if (!el) return;

    const [x, y, w, h] =
      dragState?.idx === focusedIdx && tempBbox ? tempBbox : el.bbox;
    const canvas = previewCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (imageRef.current.complete && w > 0 && h > 0) {
      ctx.imageSmoothingEnabled = false;
      const scale = Math.min(canvas.width / w, canvas.height / h);
      const fittedW = w * scale;
      const fittedH = h * scale;
      const drawX = (canvas.width - fittedW) / 2;
      const drawY = (canvas.height - fittedH) / 2;
      ctx.drawImage(
        imageRef.current,
        x,
        y,
        w,
        h,
        drawX,
        drawY,
        fittedW,
        fittedH,
      );
    }
  }, [record, elements, focusedIdx, tempBbox, dragState]);

  useEffect(() => {
    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (zoom <= 1) {
      setPanOffset({ x: 0, y: 0 });
      setIsPanning(false);
      panStartRef.current = null;
    }
  }, [zoom]);

  const clampPan = useCallback(
    (
      offset: { x: number; y: number },
      zoomLevel = zoom,
    ): { x: number; y: number } => {
      if (!containerRef.current || !record || zoomLevel <= 1) {
        return { x: 0, y: 0 };
      }

      const cRect = containerRef.current.getBoundingClientRect();
      const stage = stageSize ?? {
        width: record.result.image_size[0],
        height: record.result.image_size[1],
      };
      const scaledW = stage.width * zoomLevel;
      const scaledH = stage.height * zoomLevel;
      const maxPanX = scaledW / 2 + cRect.width / 2 - scaledW * 0.2;
      const maxPanY = scaledH / 2 + cRect.height / 2 - scaledH * 0.2;

      return {
        x: Math.max(-maxPanX, Math.min(maxPanX, offset.x)),
        y: Math.max(-maxPanY, Math.min(maxPanY, offset.y)),
      };
    },
    [record, stageSize, zoom],
  );

  const applyZoom = (
    nextZoom: number,
    anchor?: { clientX: number; clientY: number },
  ) => {
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
    setPanOffset((prev) => {
      const nextOffset = {
        x: cursorRelX - (cursorRelX - prev.x) * zoomRatio,
        y: cursorRelY - (cursorRelY - prev.y) * zoomRatio,
      };
      return clampPan(nextOffset, clampedZoom);
    });
  };

  const handleStageWheel = (event: ReactWheelEvent<HTMLDivElement>) => {
    if (!shouldConsumeStageWheel(event.deltaY)) return;

    event.preventDefault();
    applyZoom(nextZoomFromWheel(zoom, event.deltaY), {
      clientX: event.clientX,
      clientY: event.clientY,
    });
  };

  useEffect(() => {
    setPanOffset((prev) => clampPan(prev, zoom));
  }, [clampPan, stageSize, zoom]);

  const handleSvgPointerDown = (e: ReactPointerEvent<SVGSVGElement>) => {
    if (!record) return;
    const [imgW, imgH] = record.result.image_size;
    const coords = clientToImage(e.currentTarget, e.clientX, e.clientY, {
      width: imgW,
      height: imgH,
    });
    const { x, y } = coords;

    if (drawMode) {
      const bbox: [number, number, number, number] = [x, y, 0, 0];
      setActiveDragState({
        type: "draw",
        idx: elements.length,
        startX: x,
        startY: y,
        startClientX: e.clientX,
        startClientY: e.clientY,
      });
      setTempBbox(bbox);
      pendingTempBboxRef.current = bbox;
      pendingMoveRef.current = null;
      e.currentTarget.setPointerCapture(e.pointerId);
      return;
    }

    const handleHit = hitTestHandles(
      { x, y },
      elements.map((el) => el.bbox),
      12,
    );

    if (handleHit) {
      const el = elements[handleHit.idx];
      const bbox: [number, number, number, number] = [...el.bbox];
      setFocusedIdx(handleHit.idx);
      setActiveDragState({
        type: "resize",
        idx: handleHit.idx,
        corner: handleHit.handle,
        startX: x,
        startY: y,
        origBbox: bbox,
      });
      setTempBbox(bbox);
      pendingTempBboxRef.current = bbox;
      pendingMoveRef.current = null;
      e.currentTarget.setPointerCapture(e.pointerId);
      return;
    }

    const hitIdx = hitTestBBoxes(
      { x, y },
      elements.map((el) => el.bbox),
    );

    if (hitIdx !== null) {
      const el = elements[hitIdx];
      const bbox: [number, number, number, number] = [...el.bbox];
      setFocusedIdx(hitIdx);
      pendingMoveRef.current = {
        type: "move",
        idx: hitIdx,
        startX: x,
        startY: y,
        startClientX: e.clientX,
        startClientY: e.clientY,
        origBbox: bbox,
      };
      setTempBbox(null);
      pendingTempBboxRef.current = null;
      e.currentTarget.setPointerCapture(e.pointerId);
    } else {
      setFocusedIdx(null);
      pendingMoveRef.current = null;
      if (zoom > 1) {
        setIsPanning(true);
        panStartRef.current = {
          clientX: e.clientX,
          clientY: e.clientY,
          offset: { ...panOffset },
        };
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
      setPanOffset(
        clampPan({
          x: panStartRef.current.offset.x + dx,
          y: panStartRef.current.offset.y + dy,
        }),
      );
      e.currentTarget.style.cursor = "grabbing";
      return;
    }

    const [imgW, imgH] = record.result.image_size;
    const coords = clientToImage(e.currentTarget, e.clientX, e.clientY, {
      width: imgW,
      height: imgH,
    });
    const { x, y } = coords;
    const [origW, origH] = [imgW, imgH];

    const pendingMove = pendingMoveRef.current;
    const activeDragState = dragStateRef.current ?? dragState;
    const dragStateToUse =
      activeDragState ??
      (pendingMove &&
      isDragIntent(
        { x: pendingMove.startClientX, y: pendingMove.startClientY },
        { x: e.clientX, y: e.clientY },
        5,
      )
        ? pendingMove
        : null);

    if (!activeDragState && pendingMove && dragStateToUse) {
      setActiveDragState({
        type: "move",
        idx: pendingMove.idx,
        startX: pendingMove.startX,
        startY: pendingMove.startY,
        origBbox: [...pendingMove.origBbox],
      });
      pendingMoveRef.current = null;
    }

    if (dragStateToUse) {
      let nextBbox: [number, number, number, number] | null = null;
      if (dragStateToUse.type === "draw") {
        const minX = Math.min(dragStateToUse.startX, x);
        const minY = Math.min(dragStateToUse.startY, y);
        const maxX = Math.max(dragStateToUse.startX, x);
        const maxY = Math.max(dragStateToUse.startY, y);
        nextBbox = [
          Math.max(0, minX),
          Math.max(0, minY),
          Math.min(origW - Math.max(0, minX), maxX - minX),
          Math.min(origH - Math.max(0, minY), maxY - minY),
        ];
      } else if (dragStateToUse.type === "move" && dragStateToUse.origBbox) {
        nextBbox = moveBBox(
          dragStateToUse.origBbox,
          { x: x - dragStateToUse.startX, y: y - dragStateToUse.startY },
          { width: origW, height: origH },
        );
      } else if (
        dragStateToUse.type === "resize" &&
        dragStateToUse.origBbox &&
        dragStateToUse.corner
      ) {
        nextBbox = resizeBBox(
          dragStateToUse.origBbox,
          dragStateToUse.corner,
          { x, y },
          { width: origW, height: origH },
        );
      }

      if (nextBbox) {
        scheduleTempBbox(nextBbox);
      }
    } else if (!drawMode) {
      const hitIdx = hitTestBBoxes(
        { x, y },
        elements.map((el) => el.bbox),
      );
      setHoveredIdx(hitIdx);

      const svg = e.currentTarget;
      let cursor = "default";
      if (focusedIdx !== null) {
        const el = elements[focusedIdx];
        if (el) {
          const handle = hitTestHandles({ x, y }, [el.bbox], 12);
          if (handle?.handle === "tl" || handle?.handle === "br")
            cursor = "nwse-resize";
          else if (handle?.handle === "tr" || handle?.handle === "bl")
            cursor = "nesw-resize";
        }
      }
      if (cursor === "default" && hitIdx !== null) {
        cursor = "move";
      } else if (cursor === "default" && hitIdx === null && zoom > 1) {
        cursor = "grab";
      }
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
      if (e.currentTarget.hasPointerCapture(e.pointerId)) {
        e.currentTarget.releasePointerCapture(e.pointerId);
      }
      e.currentTarget.style.cursor = zoom > 1 ? "grab" : "default";
      return;
    }

    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }

    const finalTempBbox = pendingTempBboxRef.current ?? tempBbox;
    const activeDragState = dragStateRef.current ?? dragState;
    if (!activeDragState || !finalTempBbox) {
      if (activeDragState) {
        setActiveDragState(null);
      }
      pendingMoveRef.current = null;
      pendingTempBboxRef.current = null;
      return;
    }

    const newElements = [...elements];
    let shouldCommitElements = false;
    if (activeDragState.type === "draw") {
      if (finalTempBbox[2] > 5 && finalTempBbox[3] > 5) {
        pushBboxHistory({
          type: "create",
          idx: newElements.length,
          focusedIdx,
        });
        newElements.push({
          bbox: finalTempBbox,
          class_name: "",
          class_label: 0,
          confidence: 1.0,
          top_k: [],
          rejected: false,
        });
        setAnnotationStatus((prev) => ({
          ...prev,
          [newElements.length - 1]: "draft",
        }));
        setFocusedIdx(newElements.length - 1);
        setNamingFocusToken((current) => current + 1);
        shouldCommitElements = true;
      }
    } else if (
      (activeDragState.type === "move" || activeDragState.type === "resize") &&
      activeDragState.idx < newElements.length
    ) {
      const originalBbox = newElements[activeDragState.idx].bbox;
      if (!areBboxesEqual(originalBbox, finalTempBbox)) {
        pushBboxHistory({
          type: "update",
          idx: activeDragState.idx,
          previousBbox: [...originalBbox],
          focusedIdx,
        });
        newElements[activeDragState.idx] = {
          ...newElements[activeDragState.idx],
          bbox: finalTempBbox,
        };
        setAnnotationStatus((prev) => ({
          ...prev,
          [activeDragState.idx]: "draft",
        }));
        shouldCommitElements = true;
      }
    }

    if (shouldCommitElements) {
      setElements(newElements);
    }
    setActiveDragState(null);
    setTempBbox(null);
    pendingMoveRef.current = null;
    pendingTempBboxRef.current = null;
  };

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (isEditableTarget(e.target)) return;

      if (
        (e.metaKey || e.ctrlKey) &&
        !e.shiftKey &&
        e.key.toLowerCase() === "z"
      ) {
        e.preventDefault();
        undoLastBboxChange();
        return;
      }

      if (
        (e.key === "Delete" || e.key === "Backspace") &&
        focusedIdx !== null
      ) {
        removeElement(focusedIdx);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [focusedIdx, removeElement, undoLastBboxChange]);

  const submittedCount = elements.filter(
    (el, idx) =>
      annotationStatus[idx] === "validated" && !isUnnamedClass(el.class_name),
  ).length;
  const focusedElement = focusedIdx !== null ? elements[focusedIdx] : null;
  const focusedIsSubmitted =
    focusedIdx !== null && annotationStatus[focusedIdx] === "validated";
  const focusedDisplayName = focusedElement
    ? isUnnamedClass(focusedElement.class_name)
      ? t.unnamedElement
      : focusedElement.class_name
    : null;
  const focusedConfidencePercent = focusedElement
    ? Math.round(focusedElement.confidence * 100)
    : 0;

  const displayedElements = elements
    .map((el, idx) => ({ el, idx }))
    .filter(({ el, idx }) => {
      const normalizedQuery = listQuery.trim().toLocaleLowerCase("fr");
      if (
        normalizedQuery &&
        !`${idx} ${el.class_name}`
          .toLocaleLowerCase("fr")
          .includes(normalizedQuery)
      )
        return false;
      if (statusFilter === "all") return true;
      if (statusFilter === "submitted")
        return annotationStatus[idx] === "validated";
      if (statusFilter === "rejected") return Boolean(el.rejected);
      return annotationStatus[idx] !== "validated" && !el.rejected;
    })
    .sort((a, b) => {
      if (sortMode === "name") {
        return (
          a.el.class_name.localeCompare(b.el.class_name, "fr", {
            sensitivity: "base",
          }) || a.idx - b.idx
        );
      }
      if (sortMode === "confidence-asc")
        return a.el.confidence - b.el.confidence || a.idx - b.idx;
      if (sortMode === "confidence-desc")
        return b.el.confidence - a.el.confidence || a.idx - b.idx;
      return a.idx - b.idx;
    });

  useEffect(() => {
    if (focusedIdx === null) return;
    cardRefs.current[focusedIdx]?.scrollIntoView?.({ block: "nearest" });
  }, [focusedIdx, elements.length]);

  const handleSave = async () => {
    if (!id) return;
    setSaving(true);
    const ok = await updateElements(id, elements, annotationStatus);
    if (!ok) {
      setToast({ msg: translate("save.networkError"), ok: false });
      setSaving(false);
      return;
    }
    setToast({ msg: translate("save.localSuccess"), ok: true });
    setSaving(false);
    setTimeout(() => {
      navigate("/");
    }, 300);
  };

  const handleSendSubmittedForReview = async () => {
    if (!record || !id) return;
    const submittedCandidates = elements
      .map((el, idx) => ({ el, idx }))
      .filter(({ idx }) => annotationStatus[idx] === "validated");
    const unnamedSubmittedIndexes = submittedCandidates
      .map(({ el, idx }) => (isUnnamedClass(el.class_name) ? idx : null))
      .filter((idx): idx is number => idx !== null);
    if (unnamedSubmittedIndexes.length > 0) {
      setToast({
        msg: `${t.submitBlockedUnnamed} (${unnamedSubmittedIndexes.map((idx) => `#${idx}`).join(", ")})`,
        ok: false,
      });
      setTimeout(() => setToast(null), 4000);
      return;
    }
    const submittedElements = submittedCandidates.filter(
      ({ el }) => !isUnnamedClass(el.class_name),
    );
    if (submittedElements.length === 0) {
      const unnamedIndexes = elements
        .map((el, idx) => (isUnnamedClass(el.class_name) ? idx : null))
        .filter((idx): idx is number => idx !== null);
      setToast({
        msg:
          unnamedIndexes.length > 0
            ? `${t.submitBlockedUnnamed} (${unnamedIndexes.map((idx) => `#${idx}`).join(", ")})`
            : t.submitBlockedNone,
        ok: false,
      });
      setTimeout(() => setToast(null), 4000);
      return;
    }

    setSending(true);
    try {
      const payload = {
        analysis_id: id,
        image_name: record.imageName,
        image_data_url: record.imageDataUrl,
        timestamp: record.timestamp,
        annotations: submittedElements.map(({ el, idx }) => ({
          index: idx,
          bbox: el.bbox,
          class_name: el.class_name,
        })),
      };
      const result: SaveAnnotationResult = await saveAnnotation(payload);

      if (result.ok) {
        setToast({ msg: translate("save.remoteSuccess"), ok: true });
      } else {
        let msg = translate("save.networkError");

        switch (result.error_code) {
          case "PERMISSION_DENIED":
            msg = translate("save.permissionDenied");
            break;
          case "DISK_FULL":
            msg = translate("save.diskFull");
            break;
          case "STORAGE_ERROR":
            msg = translate("save.storageError", { message: result.message });
            break;
          case "INTERNAL_ERROR":
            msg = translate("save.internalError", {
              traceId: result.trace_id ?? "?",
            });
            break;
          default:
            msg = translate("save.networkError");
        }

        setToast({ msg, ok: false });
      }
    } finally {
      setSending(false);
      setTimeout(() => setToast(null), 4000);
    }
  };

  if (!record && !loading) {
    return (
      <div className="max-w-2xl mx-auto text-center py-12">
        <h2 className="text-2xl font-bold text-stone-100 mb-4">{t.notFound}</h2>
        <Link
          to="/"
          className="text-amber-400 hover:text-amber-300 inline-flex items-center gap-2"
        >
          <ArrowLeft size={18} /> {t.backToHistory}
        </Link>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <Loader2 className="animate-spin text-amber-500" size={32} />
      </div>
    );
  }

  return (
    <div className="annotation-app flex h-full w-full flex-col gap-1 overflow-hidden p-1">
      <AnnotationPageChrome
        labels={t}
        saving={saving}
        sending={sending}
        onSubmitNamed={submitNamedElements}
        onSave={handleSave}
        onSendSubmittedForReview={handleSendSubmittedForReview}
      />

      <div className="flex min-h-0 flex-1 gap-1.5">
        <MainImagePanel
          tone="annotation"
          className="flex-1"
          testIds={{ controls: "annotation-stage-controls" }}
          stageRef={containerRef}
          stageProps={{
            "data-testid": "annotation-stage-frame",
            onWheel: handleStageWheel,
          }}
          transformProps={{ "data-testid": "annotation-stage" }}
          transformStyle={{
            width: resolvedStageSize
              ? `${resolvedStageSize.width}px`
              : undefined,
            height: resolvedStageSize
              ? `${resolvedStageSize.height}px`
              : undefined,
            transform: `translate(${panOffset.x}px, ${panOffset.y}px) scale(${zoom})`,
            transformOrigin: "center center",
            transition: isPanning ? "none" : "transform 0.1s ease",
            willChange: "transform",
          }}
          transformClassName="annotation-stage shrink-0 overflow-hidden rounded-lg"
          toolbar={
            <AnnotationAnalyzerToolbar
              drawMode={drawMode}
              canUndo={bboxHistory.length > 0}
              showLabelNames={showLabelNames}
              labels={t}
              onToggleDrawMode={() => setDrawMode((current) => !current)}
              onUndo={undoLastBboxChange}
              onToggleLabelNames={() =>
                setShowLabelNames((current) => !current)
              }
            />
          }
          controls={[
            {
              id: "zoom-in",
              label: t.zoomIn,
              title: t.zoomIn,
              onClick: () => applyZoom(zoom + 0.25),
              icon: <ZoomIn size={16} />,
            },
            {
              id: "reset-view",
              label: t.resetView,
              title: t.resetView,
              onClick: () => {
                setZoom(1);
                setPanOffset({ x: 0, y: 0 });
              },
              icon: <Maximize2 size={16} />,
            },
            {
              id: "zoom-out",
              label: t.zoomOut,
              title: t.zoomOut,
              onClick: () => applyZoom(zoom - 0.25),
              icon: <ZoomOut size={16} />,
            },
          ]}
          zoomLabel={`${Math.round(zoom * 100)}%`}
          image={
            <img
              ref={imageRef}
              src={record!.imageDataUrl}
              alt={record!.imageName}
              draggable={false}
              onLoad={updateStageSize}
              className="block h-full w-full object-fill pointer-events-none"
            />
          }
          overlay={
            record && (
              <AnnotationOverlay
                imageSize={record.result.image_size}
                elements={elements}
                annotationStatus={annotationStatus}
                focusedIdx={focusedIdx}
                hoveredIdx={hoveredIdx}
                listHoveredIdx={listHoveredIdx}
                drawMode={drawMode}
                dragState={dragState}
                tempBbox={tempBbox}
                showLabelNames={showLabelNames}
                unnamedLabel={t.unnamedElement}
                onPointerDown={handleSvgPointerDown}
                onPointerMove={handleSvgPointerMove}
                onPointerUp={handleSvgPointerUp}
              />
            )
          }
        />

        <aside
          className="annotation-rail annotation-inspector flex shrink-0 flex-col rounded-2xl p-4"
          aria-label="Inspecteur d’annotation"
        >
          <AnnotationSelectedInspector
            focusedElement={focusedElement}
            focusedIdx={focusedIdx}
            focusedDisplayName={focusedDisplayName}
            focusedConfidencePercent={focusedConfidencePercent}
            focusedIsSubmitted={focusedIsSubmitted}
            previewCanvasRef={previewCanvasRef}
            classes={classes}
            customClasses={customClasses}
            namingFocusToken={namingFocusToken}
            labels={t}
            onCommitElementName={commitElementName}
            onSetElementValidation={(idx, submitted) =>
              setElementValidation(idx, submitted ? "validated" : "draft")
            }
            onRemoveElement={removeElement}
          />

          <AnnotationElementList
            displayedElements={displayedElements}
            annotationStatus={annotationStatus}
            focusedIdx={focusedIdx}
            elementsCount={elements.length}
            submittedCount={submittedCount}
            listQuery={listQuery}
            statusFilter={statusFilter}
            sortMode={sortMode}
            labels={t}
            cardRefs={cardRefs}
            setFocusedIdx={setFocusedIdx}
            onListQueryChange={setListQuery}
            onStatusFilterChange={setStatusFilter}
            onSortModeChange={setSortMode}
            setListHoveredIdx={setListHoveredIdx}
          />
        </aside>
      </div>
      {toast && <AnnotationToast toast={toast} />}
    </div>
  );
}
