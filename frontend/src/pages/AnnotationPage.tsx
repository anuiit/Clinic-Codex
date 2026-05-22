import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
  type WheelEvent as ReactWheelEvent,
} from "react";
import {
  useParams,
  useNavigate,
  Link,
  useSearchParams,
} from "react-router-dom";
import {
  ArrowLeft,
  Save,
  Loader2,
  ZoomIn,
  ZoomOut,
  Maximize2,
  PenTool,
  MousePointer2,
  Trash2,
  Upload,
  Undo2,
  Tags,
} from "lucide-react";
import { getAnalysisById, updateElements } from "../services/storage";
import { getClasses, saveAnnotation } from "../services/api";
import { t as translate } from "../i18n/annotation.fr";
import { MainImagePanel } from "../components/MainImagePanel";
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
  getBoxVisualState,
  hitTestBBoxes,
  hitTestHandles,
  isDragIntent,
  moveBBox,
  resizeBBox,
  type BBox,
  type BBoxHandle,
} from "../utils/segmentationBoxes";
import {
  getFuzzyClassSuggestions,
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

function formatBboxLabel(
  idx: number,
  className: string,
  showName: boolean,
  unnamedLabel: string,
): string {
  if (!showName) return `#${idx}`;
  const displayName = isUnnamedClass(className) ? unnamedLabel : className;
  const clippedName =
    displayName.length > 18 ? `${displayName.slice(0, 17)}…` : displayName;
  return clippedName;
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

interface ElementNameComboboxProps {
  value: string;
  classNames: string[];
  customClassNames: string[];
  topK: DetectedElement["top_k"];
  autoFocusToken: number;
  labels: typeof appText.annotation;
  index: number;
  onCommit: (name: string) => void;
}

function ElementNameCombobox({
  value,
  classNames,
  customClassNames,
  topK,
  autoFocusToken,
  labels,
  index,
  onCommit,
}: ElementNameComboboxProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [inputValue, setInputValue] = useState(() =>
    isUnnamedClass(value) ? "" : value,
  );
  const [isOpen, setIsOpen] = useState(false);
  const [highlightedIdx, setHighlightedIdx] = useState(0);
  const suggestions = getFuzzyClassSuggestions(
    inputValue,
    classNames,
    topK,
    customClassNames,
  );
  const normalizedInput = normalizeClassName(inputValue);
  const allCandidateNames = [
    ...classNames,
    ...customClassNames,
    ...topK.map((item) => item.class_name),
  ];
  const canCreate =
    normalizedInput.length > 0 &&
    !hasExactClassName(normalizedInput, allCandidateNames);

  useEffect(() => {
    setInputValue(isUnnamedClass(value) ? "" : value);
  }, [value]);

  useEffect(() => {
    if (autoFocusToken <= 0) return;
    inputRef.current?.focus();
    inputRef.current?.select();
    setIsOpen(true);
  }, [autoFocusToken]);

  const commitName = (name: string) => {
    const normalizedName = normalizeClassName(name);
    if (!normalizedName) return;
    onCommit(normalizedName);
    setInputValue(normalizedName);
    setIsOpen(false);
  };

  const handleKeyDown = (event: ReactKeyboardEvent<HTMLInputElement>) => {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setIsOpen(true);
      setHighlightedIdx((current) =>
        Math.min(current + 1, Math.max(suggestions.length - 1, 0)),
      );
      return;
    }
    if (event.key === "ArrowUp") {
      event.preventDefault();
      setHighlightedIdx((current) => Math.max(current - 1, 0));
      return;
    }
    if (event.key === "Escape") {
      setIsOpen(false);
      setInputValue(isUnnamedClass(value) ? "" : value);
      return;
    }
    if (event.key === "Enter") {
      event.preventDefault();
      const highlightedSuggestion = suggestions[highlightedIdx];
      commitName(highlightedSuggestion?.name ?? inputValue);
    }
  };

  return (
    <div className="relative" onClick={(event) => event.stopPropagation()}>
      <label
        className="mb-1 block text-xs font-medium uppercase tracking-[0.18em] text-stone-500"
        htmlFor={`element-name-${index}`}
      >
        {labels.renameElement}
      </label>
      <input
        ref={inputRef}
        id={`element-name-${index}`}
        aria-label={`${labels.nameElement} ${index}`}
        value={inputValue}
        onChange={(event) => {
          setInputValue(event.target.value);
          setHighlightedIdx(0);
          setIsOpen(true);
        }}
        onFocus={() => setIsOpen(true)}
        onBlur={() => {
          if (normalizedInput) {
            commitName(inputValue);
          } else {
            setIsOpen(false);
          }
        }}
        onKeyDown={handleKeyDown}
        placeholder={labels.elementNamePlaceholder}
        className="w-full rounded-lg border border-stone-700 bg-stone-900 px-3 py-2 text-sm text-stone-100 outline-none transition-colors placeholder:text-stone-600 focus:border-amber-500"
      />
      {isOpen && (
        <div className="absolute z-20 mt-1 max-h-56 w-full overflow-y-auto rounded-lg border border-stone-700 bg-stone-950 shadow-xl">
          <div className="border-b border-stone-800 px-3 py-1.5 text-[11px] uppercase tracking-[0.2em] text-stone-500">
            {labels.suggestions}
          </div>
          {suggestions.length === 0 && !canCreate && (
            <div className="px-3 py-2 text-sm text-stone-500">
              {labels.noSuggestion}
            </div>
          )}
          {suggestions.map((suggestion, suggestionIdx) => (
            <button
              key={`${suggestion.source}-${suggestion.name}`}
              type="button"
              onMouseDown={(event) => event.preventDefault()}
              onClick={() => commitName(suggestion.name)}
              className={`flex w-full items-center justify-between px-3 py-2 text-left text-sm transition-colors ${suggestionIdx === highlightedIdx ? "bg-amber-500/15 text-amber-100" : "text-stone-100 hover:bg-stone-800"}`}
            >
              <span>{suggestion.name}</span>
              <span className="text-[10px] uppercase tracking-[0.18em] text-stone-500">
                {suggestion.source}
              </span>
            </button>
          ))}
          {canCreate && (
            <button
              type="button"
              onMouseDown={(event) => event.preventDefault()}
              onClick={() => commitName(normalizedInput)}
              className="w-full border-t border-stone-800 px-3 py-2 text-left text-sm font-medium text-emerald-300 transition-colors hover:bg-emerald-500/10"
            >
              {labels.createElementName} « {normalizedInput} »
            </button>
          )}
        </div>
      )}
    </div>
  );
}

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
    async function loadData() {
      if (!id) {
        setLoading(false);
        return;
      }
      const rec = getAnalysisById(id);
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
        setClasses(classesResult.class_names);
      } catch {
        // failed to load classes
      } finally {
        setLoading(false);
      }
    }
    loadData();
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

  const handleSave = () => {
    if (!id) return;
    setSaving(true);
    const ok = updateElements(id, elements, annotationStatus);
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
      <div className="annotation-topbar flex shrink-0 items-center justify-between rounded-xl px-3 py-2">
        <div className="flex min-w-0 items-center gap-4">
          <Link
            to="/"
            className="flex items-center gap-2 rounded-full border border-stone-700/70 bg-stone-950/70 px-3 py-1.5 text-sm font-medium text-stone-300 transition-colors hover:border-amber-500/50 hover:text-stone-50"
          >
            <ArrowLeft size={18} /> {t.back}
          </Link>
          <div className="min-w-0">
            <div className="text-[10px] font-semibold uppercase tracking-[0.28em] text-amber-300/80">
              Clinic Codex
            </div>
            <h1 className="truncate text-lg font-black tracking-tight text-stone-50">
              {t.title}
            </h1>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={submitNamedElements}
            className="rounded-lg border border-emerald-700/60 px-3 py-1.5 text-sm font-semibold text-emerald-200 transition-colors hover:bg-emerald-500/10"
          >
            {t.submitNamed}
          </button>
          <button
            onClick={handleSave}
            disabled={saving}
            className="flex items-center gap-2 rounded-lg bg-amber-500 px-4 py-1.5 text-sm font-semibold text-stone-950 transition-colors hover:bg-amber-400 disabled:opacity-50"
          >
            {saving ? (
              <Loader2 size={18} className="animate-spin" />
            ) : (
              <Save size={18} />
            )}
            {t.saveChanges}
          </button>
          <button
            onClick={handleSendSubmittedForReview}
            disabled={sending}
            className="flex items-center gap-2 rounded-lg bg-emerald-600 px-4 py-1.5 text-sm font-semibold text-white transition-colors hover:bg-emerald-500 disabled:opacity-50"
          >
            {sending ? (
              <Loader2 size={18} className="animate-spin" />
            ) : (
              <Upload size={18} />
            )}
            {t.sendSubmittedForReview}
          </button>
        </div>
      </div>

      <div className="shrink-0 rounded-xl border border-amber-500/20 bg-amber-500/10 px-3 py-1.5 text-xs text-amber-100 shadow-lg shadow-amber-950/20">
        {t.adminApprovalNotice}
      </div>

      <div className="flex min-h-0 flex-1 gap-1.5">
        <MainImagePanel
          ref={containerRef}
          data-testid="annotation-stage-frame"
          tone="annotation"
          onWheel={handleStageWheel}
          toolbar={
            <div className="annotation-floating-toolbar absolute left-3 top-3 z-10 flex items-center gap-1 rounded-2xl p-1">
            <button
              type="button"
              onClick={() => setDrawMode(!drawMode)}
              className={`flex items-center gap-2 rounded-xl px-3 py-2 text-sm font-semibold transition-colors ${drawMode ? "bg-amber-400 text-stone-950 shadow-lg shadow-amber-950/30" : "text-stone-300 hover:bg-stone-800 hover:text-stone-50"}`}
            >
              {drawMode ? <PenTool size={16} /> : <MousePointer2 size={16} />}
              {drawMode ? t.drawMode : t.selectMode}
            </button>
            <button
              type="button"
              onClick={undoLastBboxChange}
              disabled={bboxHistory.length === 0}
              aria-label={t.undoBbox}
              title={`${t.undoBbox} (Ctrl+Z)`}
              className="flex items-center gap-2 rounded-xl px-3 py-2 text-sm font-semibold text-stone-300 transition-colors hover:bg-stone-800 hover:text-stone-50 disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:bg-transparent disabled:hover:text-stone-300"
            >
              <Undo2 size={16} />
              {t.undoBbox}
            </button>
            <button
              type="button"
              onClick={() => setShowLabelNames((current) => !current)}
              className={`flex items-center gap-2 rounded-xl px-3 py-2 text-sm font-semibold transition-colors ${showLabelNames ? "bg-stone-100 text-stone-950" : "text-stone-300 hover:bg-stone-800 hover:text-stone-50"}`}
              aria-pressed={showLabelNames}
              aria-label={
                showLabelNames
                  ? "Masquer les noms des libellés"
                  : "Afficher les noms des libellés"
              }
            >
              {showLabelNames ? (
                <Tags size={16} />
              ) : (
                <span className="text-xs font-black tabular-nums">N°</span>
              )}
              {showLabelNames ? "Noms" : "N°"}
            </button>
            <div className="mx-1 h-6 w-px bg-stone-700/70" />
            <div className="flex items-center gap-1">
              <button
                type="button"
                onClick={() => applyZoom(zoom + 0.25)}
                className="rounded-xl p-1.5 text-stone-300 transition-colors hover:bg-stone-800 hover:text-stone-50"
                aria-label={t.zoomIn}
              >
                <ZoomIn size={16} />
              </button>
              <button
                type="button"
                onClick={() => {
                  setZoom(1);
                  setPanOffset({ x: 0, y: 0 });
                }}
                className="rounded-xl p-1.5 text-stone-300 transition-colors hover:bg-stone-800 hover:text-stone-50"
                title={t.resetView}
              >
                <Maximize2 size={16} />
              </button>
              <button
                type="button"
                onClick={() => applyZoom(zoom - 0.25)}
                className="rounded-xl p-1.5 text-stone-300 transition-colors hover:bg-stone-800 hover:text-stone-50"
                aria-label={t.zoomOut}
              >
                <ZoomOut size={16} />
              </button>
              <span className="px-2 text-xs font-semibold tabular-nums text-stone-400">
                {Math.round(zoom * 100)}%
              </span>
            </div>
            </div>
          }
        >
          <div
            data-testid="annotation-stage"
            style={{
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
            className="annotation-stage relative shrink-0 overflow-hidden rounded-lg"
          >
            <img
              ref={imageRef}
              src={record!.imageDataUrl}
              alt={record!.imageName}
              draggable={false}
              onLoad={updateStageSize}
              className="block h-full w-full object-fill pointer-events-none"
            />
            {record && (
              <svg
                data-testid="annotation-overlay"
                className="absolute left-0 top-0 h-full w-full"
                viewBox={`0 0 ${record.result.image_size[0]} ${record.result.image_size[1]}`}
                preserveAspectRatio="none"
                style={{ width: "100%", height: "100%" }}
                onPointerDown={handleSvgPointerDown}
                onPointerMove={handleSvgPointerMove}
                onPointerUp={handleSvgPointerUp}
              >
                {elements.map((el, idx) => {
                  const [x, y, w, h] =
                    idx === dragState?.idx &&
                    dragState.type !== "draw" &&
                    tempBbox
                      ? tempBbox
                      : el.bbox;
                  const isFocused = idx === focusedIdx;
                  const isHovered = idx === hoveredIdx;
                  const isListHovered = idx === listHoveredIdx;
                  const isSubmitted = annotationStatus[idx] === "validated";
                  const boxVisual = getBoxVisualState({
                    focused: isFocused,
                    hovered: isHovered,
                    listHovered: isListHovered,
                    submitted: isSubmitted,
                    rejected: el.rejected,
                  });
                  const labelY = Math.max(0, y - 28);
                  return (
                    <g key={idx} className="pointer-events-none select-none">
                      <rect
                        x={x}
                        y={y}
                        width={w}
                        height={h}
                        data-testid={`annotation-box-${idx}`}
                        fill={boxVisual.fillColor}
                        stroke={boxVisual.strokeColor}
                        strokeWidth={boxVisual.strokeWidth}
                        strokeDasharray={boxVisual.strokeDasharray}
                        vectorEffect="non-scaling-stroke"
                      />
                      {isFocused && !drawMode && (
                        <>
                          <rect
                            x={x - 6}
                            y={y - 6}
                            width={12}
                            height={12}
                            rx={3}
                            fill="#fef3c7"
                            stroke="#0c0a09"
                            strokeWidth={1.5}
                            className="cursor-nwse-resize"
                          />
                          <rect
                            x={x + w - 6}
                            y={y - 6}
                            width={12}
                            height={12}
                            rx={3}
                            fill="#fef3c7"
                            stroke="#0c0a09"
                            strokeWidth={1.5}
                            className="cursor-nesw-resize"
                          />
                          <rect
                            x={x - 6}
                            y={y + h - 6}
                            width={12}
                            height={12}
                            rx={3}
                            fill="#fef3c7"
                            stroke="#0c0a09"
                            strokeWidth={1.5}
                            className="cursor-nesw-resize"
                          />
                          <rect
                            x={x + w - 6}
                            y={y + h - 6}
                            width={12}
                            height={12}
                            rx={3}
                            fill="#fef3c7"
                            stroke="#0c0a09"
                            strokeWidth={1.5}
                            className="cursor-nwse-resize"
                          />
                        </>
                      )}
                      {(() => {
                        const label = formatBboxLabel(
                          idx,
                          el.class_name,
                          showLabelNames,
                          t.unnamedElement,
                        );
                        const labelWidth = showLabelNames
                          ? Math.min(168, Math.max(64, label.length * 8 + 18))
                          : 44;
                        return (
                          <>
                            <title>{label}</title>
                            <rect
                              x={x}
                              y={labelY}
                              width={labelWidth}
                              height={24}
                              rx={6}
                              fill={boxVisual.strokeColor}
                              opacity={0.95}
                              pointerEvents="none"
                              style={{ userSelect: "none" }}
                            />
                            <text
                              x={x + 10}
                              y={labelY + 12}
                              fill="#0c0a09"
                              fontSize="13"
                              fontWeight="800"
                              fontFamily="sans-serif"
                              textAnchor="start"
                              dominantBaseline="central"
                              pointerEvents="none"
                              style={{ userSelect: "none" }}
                            >
                              {label}
                            </text>
                          </>
                        );
                      })()}
                    </g>
                  );
                })}
                {drawMode && dragState?.type === "draw" && tempBbox && (
                  <rect
                    x={tempBbox[0]}
                    y={tempBbox[1]}
                    width={tempBbox[2]}
                    height={tempBbox[3]}
                    fill="rgba(59, 130, 246, 0.14)"
                    stroke="#60a5fa"
                    strokeWidth={2}
                    strokeDasharray="4 4"
                    vectorEffect="non-scaling-stroke"
                  />
                )}
              </svg>
            )}
          </div>
        </MainImagePanel>

        <aside
          className="annotation-rail annotation-inspector flex shrink-0 flex-col rounded-2xl p-4"
          aria-label="Inspecteur d’annotation"
        >
          <section
            className="annotation-selected-inspector mb-3 flex shrink-0 flex-col gap-3 overflow-hidden rounded-2xl p-3"
            data-testid="selected-element-inspector"
          >
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <div className="text-xs font-semibold uppercase tracking-[0.28em] text-amber-300/80">
                  Inspecteur
                </div>
                <h2 className="mt-1 truncate text-xl font-black text-stone-50">
                  {focusedElement && focusedIdx !== null
                    ? `#${focusedIdx} · ${focusedDisplayName}`
                    : "Sélectionnez un élément"}
                </h2>
              </div>
              {focusedElement && focusedIdx !== null && (
                <span
                  className={`shrink-0 rounded-full px-3 py-1 text-xs font-bold ${focusedElement.rejected ? "bg-red-500/15 text-red-300" : focusedIsSubmitted ? "bg-emerald-500/15 text-emerald-300" : "bg-stone-800 text-stone-400"}`}
                >
                  {focusedIsSubmitted ? t.submitted : t.draft}
                </span>
              )}
            </div>

            <div className="min-h-0 flex-1 overflow-hidden">
              {focusedElement && focusedIdx !== null ? (
                <div className="flex h-full min-h-0 flex-col gap-3">
                  <div className="annotation-selected-overview grid grid-cols-[150px_minmax(0,1fr)] gap-3">
                    <div className="annotation-crop flex h-[150px] items-center justify-center overflow-hidden rounded-xl border border-stone-700/35">
                      <canvas
                        ref={previewCanvasRef}
                        width={200}
                        height={200}
                        className="block h-[140px] w-[140px] rounded-lg object-contain"
                      />
                    </div>
                    <div className="min-w-0 space-y-3">
                      <div>
                        <div className="mb-1 flex items-center justify-between text-xs font-semibold text-stone-400">
                          <span>Confiance</span>
                          <span className="tabular-nums text-stone-200">
                            {focusedConfidencePercent}%
                          </span>
                        </div>
                        <div className="h-2 overflow-hidden rounded-full bg-stone-800">
                          <div
                            className={`h-full rounded-full ${focusedElement.rejected ? "bg-red-400" : focusedIsSubmitted ? "bg-emerald-400" : "bg-amber-400"}`}
                            style={{
                              width: `${Math.max(0, Math.min(100, focusedConfidencePercent))}%`,
                            }}
                          />
                        </div>
                      </div>
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        {(["x", "y", "w", "h"] as const).map(
                          (label, coordIdx) => (
                            <div
                              key={label}
                              className="rounded-xl border border-stone-700/60 bg-stone-950/50 px-3 py-2"
                            >
                              <div className="uppercase tracking-[0.18em] text-stone-500">
                                {label}
                              </div>
                              <div className="mt-1 font-semibold tabular-nums text-stone-100">
                                {Math.round(focusedElement.bbox[coordIdx])}
                              </div>
                            </div>
                          ),
                        )}
                      </div>
                    </div>
                  </div>

                  <div
                    className="annotation-inspector-action-row grid grid-cols-[minmax(0,1fr)_auto_auto] items-end gap-2"
                    data-testid="annotation-inspector-action-row"
                  >
                    <div className="min-w-0">
                      <ElementNameCombobox
                        value={focusedElement.class_name}
                        classNames={[focusedElement.class_name, ...classes]}
                        customClassNames={customClasses}
                        topK={focusedElement.top_k}
                        autoFocusToken={namingFocusToken}
                        labels={t}
                        index={focusedIdx}
                        onCommit={(name) => commitElementName(focusedIdx, name)}
                      />
                    </div>
                    <button
                      type="button"
                      onClick={() =>
                        setElementValidation(
                          focusedIdx,
                          focusedIsSubmitted ? "draft" : "validated",
                        )
                      }
                      disabled={isUnnamedClass(focusedElement.class_name)}
                      className={`shrink-0 rounded-xl px-3 py-2 text-sm font-bold transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${focusedIsSubmitted ? "border border-stone-700 bg-stone-900 text-stone-200 hover:bg-stone-800" : "bg-emerald-500 text-stone-950 hover:bg-emerald-400"}`}
                    >
                      {focusedIsSubmitted ? t.markDraft : t.markSubmitted}
                    </button>
                    <button
                      type="button"
                      onClick={() => removeElement(focusedIdx)}
                      className="shrink-0 rounded-xl border border-red-500/30 px-3 py-2 text-sm font-bold text-red-300 transition-colors hover:bg-red-500/10"
                    >
                      <span className="sr-only">
                        Supprimer l’élément #{focusedIdx}
                      </span>
                      <Trash2 size={18} />
                    </button>
                  </div>
                </div>
              ) : (
                <div className="flex h-full items-center justify-center rounded-xl border border-dashed border-stone-700/60 bg-stone-950/20 px-6 text-center text-sm text-stone-500">
                  {t.selectElementCrop}
                </div>
              )}
            </div>
          </section>

          <section
            className="flex min-h-0 flex-1 flex-col"
            aria-label="Liste compacte des éléments"
          >
            <div
              className="annotation-list-controls mb-3 flex flex-wrap items-end gap-2 xl:flex-nowrap"
              data-testid="annotation-list-controls"
            >
              <div
                className="shrink-0 rounded-full border border-emerald-500/20 bg-emerald-500/10 px-3 py-1.5 text-xs font-semibold text-emerald-300"
                aria-label={`${t.submitted} ${submittedCount}/${elements.length}`}
              >
                {t.submitted} {submittedCount}/{elements.length}
              </div>
              <label className="min-w-[128px] flex-1 text-xs font-semibold uppercase tracking-[0.18em] text-stone-500">
                Filtrer
                <input
                  type="search"
                  aria-label="Filtrer les éléments"
                  value={listQuery}
                  onChange={(event) => setListQuery(event.target.value)}
                  placeholder="Nom ou numéro"
                  className="mt-1 w-full rounded-lg border border-stone-700 bg-stone-950 px-2 py-1.5 text-xs normal-case tracking-normal text-stone-100 outline-none placeholder:text-stone-600 focus:border-amber-500"
                />
              </label>
              <label className="min-w-[112px] text-xs font-semibold uppercase tracking-[0.18em] text-stone-500">
                Statut
                <select
                  value={statusFilter}
                  onChange={(event) =>
                    setStatusFilter(
                      event.target.value as AnnotationStatusFilter,
                    )
                  }
                  className="mt-1 w-full rounded-lg border border-stone-700 bg-stone-950 px-2 py-1.5 text-xs normal-case tracking-normal text-stone-100 outline-none focus:border-amber-500"
                >
                  <option value="all">Tous</option>
                  <option value="draft">Brouillons</option>
                  <option value="submitted">Soumis</option>
                  <option value="rejected">Rejetés</option>
                </select>
              </label>
              <label className="min-w-[122px] text-xs font-semibold uppercase tracking-[0.18em] text-stone-500">
                Tri
                <select
                  value={sortMode}
                  onChange={(event) =>
                    setSortMode(event.target.value as AnnotationSortMode)
                  }
                  className="mt-1 w-full rounded-lg border border-stone-700 bg-stone-950 px-2 py-1.5 text-xs normal-case tracking-normal text-stone-100 outline-none focus:border-amber-500"
                >
                  <option value="original">Original</option>
                  <option value="confidence-asc">Confiance ↑</option>
                  <option value="confidence-desc">Confiance ↓</option>
                  <option value="name">Nom A→Z</option>
                </select>
              </label>
            </div>

            <div className="annotation-scrollbar min-h-0 flex-1 space-y-2 overflow-y-auto pr-2">
              {displayedElements.map(({ el, idx }) => {
                const isFocused = idx === focusedIdx;
                const displayName = isUnnamedClass(el.class_name)
                  ? t.unnamedElement
                  : el.class_name;
                const isSubmitted = annotationStatus[idx] === "validated";
                const confidencePercent = Math.round(el.confidence * 100);

                return (
                  <button
                    key={idx}
                    type="button"
                    ref={(node) => {
                      cardRefs.current[idx] = node;
                    }}
                    onClick={() => setFocusedIdx(idx)}
                    onMouseEnter={() => setListHoveredIdx(idx)}
                    onMouseLeave={() =>
                      setListHoveredIdx((current) =>
                        current === idx ? null : current,
                      )
                    }
                    onFocus={() => setListHoveredIdx(idx)}
                    onBlur={() =>
                      setListHoveredIdx((current) =>
                        current === idx ? null : current,
                      )
                    }
                    className={`annotation-card flex w-full cursor-pointer items-center justify-between gap-3 rounded-2xl p-3 text-left transition-all ${isFocused ? "annotation-card-selected" : ""}`}
                  >
                    <span className="flex min-w-0 items-center gap-3">
                      <span
                        className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-xl text-sm font-black ${isSubmitted ? "bg-emerald-400 text-stone-950" : isFocused ? "bg-amber-400 text-stone-950" : "bg-stone-800 text-stone-300"}`}
                      >
                        #{idx}
                      </span>
                      <span className="min-w-0">
                        <span
                          className={`block truncate text-sm font-bold ${isUnnamedClass(el.class_name) ? "text-amber-300" : "text-stone-100"}`}
                        >
                          {displayName}
                        </span>
                        <span className="mt-1 flex items-center gap-2">
                          <span className="h-1.5 w-20 overflow-hidden rounded-full bg-stone-800">
                            <span
                              className={`block h-full rounded-full ${el.rejected ? "bg-red-400" : isSubmitted ? "bg-emerald-400" : "bg-amber-400"}`}
                              style={{
                                width: `${Math.max(0, Math.min(100, confidencePercent))}%`,
                              }}
                            />
                          </span>
                          <span className="text-[10px] font-semibold tabular-nums text-stone-500">
                            {confidencePercent}%
                          </span>
                        </span>
                      </span>
                    </span>
                    <span
                      className={`shrink-0 rounded-full px-2 py-0.5 text-[11px] font-bold ${el.rejected ? "bg-red-500/15 text-red-300" : isSubmitted ? "bg-emerald-500/15 text-emerald-300" : "bg-stone-800 text-stone-400"}`}
                    >
                      {isSubmitted ? t.submitted : t.draft}
                    </span>
                  </button>
                );
              })}
            </div>
          </section>
        </aside>
      </div>
      {toast && (
        <div
          className={`fixed bottom-6 right-6 z-50 rounded-xl px-5 py-3 text-sm font-medium shadow-lg transition-all ${toast.ok ? "bg-emerald-700 text-white" : "bg-red-700 text-white"}`}
        >
          {toast.msg}
        </div>
      )}
    </div>
  );
}
