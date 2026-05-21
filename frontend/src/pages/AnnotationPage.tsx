import { useCallback, useEffect, useRef, useState, type KeyboardEvent as ReactKeyboardEvent, type PointerEvent as ReactPointerEvent } from 'react';
import { useParams, useNavigate, Link, useSearchParams } from 'react-router-dom';
import { ArrowLeft, Save, Loader2, ZoomIn, ZoomOut, Maximize2, PenTool, MousePointer2, Trash2, Upload } from 'lucide-react';
import { getAnalysisById, updateElements } from '../services/storage';
import { getClasses, saveAnnotation } from '../services/api';
import { t as translate } from '../i18n/annotation.fr';
import { appText } from '../i18n/text';
import type { AnalysisRecord, AnnotationStatus, DetectedElement, SaveAnnotationResult } from '../types';
import { clientToImage } from '../utils/imageCoords';
import { getFuzzyClassSuggestions, hasExactClassName, isUnnamedClass, normalizeClassName } from '../utils/fuzzyClasses';

type StageSize = { width: number; height: number };

type DragState = {
  type: 'draw' | 'move' | 'resize';
  idx: number;
  corner?: 'tl' | 'tr' | 'bl' | 'br';
  startX: number;
  startY: number;
  origBbox?: [number, number, number, number];
} | null;

interface ElementNameComboboxProps {
  value: string;
  classNames: string[];
  customClassNames: string[];
  topK: DetectedElement['top_k'];
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
  const [inputValue, setInputValue] = useState(() => (isUnnamedClass(value) ? '' : value));
  const [isOpen, setIsOpen] = useState(false);
  const [highlightedIdx, setHighlightedIdx] = useState(0);
  const suggestions = getFuzzyClassSuggestions(inputValue, classNames, topK, customClassNames);
  const normalizedInput = normalizeClassName(inputValue);
  const allCandidateNames = [...classNames, ...customClassNames, ...topK.map((item) => item.class_name)];
  const canCreate = normalizedInput.length > 0 && !hasExactClassName(normalizedInput, allCandidateNames);

  useEffect(() => {
    setInputValue(isUnnamedClass(value) ? '' : value);
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
    if (event.key === 'ArrowDown') {
      event.preventDefault();
      setIsOpen(true);
      setHighlightedIdx((current) => Math.min(current + 1, Math.max(suggestions.length - 1, 0)));
      return;
    }
    if (event.key === 'ArrowUp') {
      event.preventDefault();
      setHighlightedIdx((current) => Math.max(current - 1, 0));
      return;
    }
    if (event.key === 'Escape') {
      setIsOpen(false);
      setInputValue(isUnnamedClass(value) ? '' : value);
      return;
    }
    if (event.key === 'Enter') {
      event.preventDefault();
      const highlightedSuggestion = suggestions[highlightedIdx];
      commitName(highlightedSuggestion?.name ?? inputValue);
    }
  };

  return (
    <div className="relative" onClick={(event) => event.stopPropagation()}>
      <label className="mb-1 block text-xs font-medium uppercase tracking-[0.18em] text-stone-500" htmlFor={`element-name-${index}`}>
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
            <div className="px-3 py-2 text-sm text-stone-500">{labels.noSuggestion}</div>
          )}
          {suggestions.map((suggestion, suggestionIdx) => (
            <button
              key={`${suggestion.source}-${suggestion.name}`}
              type="button"
              onMouseDown={(event) => event.preventDefault()}
              onClick={() => commitName(suggestion.name)}
              className={`flex w-full items-center justify-between px-3 py-2 text-left text-sm transition-colors ${suggestionIdx === highlightedIdx ? 'bg-amber-500/15 text-amber-100' : 'text-stone-100 hover:bg-stone-800'}`}
            >
              <span>{suggestion.name}</span>
              <span className="text-[10px] uppercase tracking-[0.18em] text-stone-500">{suggestion.source}</span>
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
  const cardRefs = useRef<Array<HTMLDivElement | null>>([]);

  const elementParam = searchParams.get('element');
  const initialFocusedIdx = elementParam !== null && !Number.isNaN(Number(elementParam))
    ? Number(elementParam)
    : null;
  
  const [record, setRecord] = useState<AnalysisRecord | null>(null);
  const [elements, setElements] = useState<DetectedElement[]>([]);
  const [annotationStatus, setAnnotationStatus] = useState<Record<number, AnnotationStatus>>({});
  const [classes, setClasses] = useState<string[]>([]);
  const [customClasses, setCustomClasses] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [sending, setSending] = useState(false);
  const [toast, setToast] = useState<{ msg: string; ok: boolean } | null>(null);

  const [focusedIdx, setFocusedIdx] = useState<number | null>(initialFocusedIdx);
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const [drawMode, setDrawMode] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [panOffset, setPanOffset] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [dragState, setDragState] = useState<DragState>(null);
  const [tempBbox, setTempBbox] = useState<[number, number, number, number] | null>(null);
  const [namingFocusToken, setNamingFocusToken] = useState(0);
  const [stageSize, setStageSize] = useState<StageSize | null>(null);
  
  const imageRef = useRef<HTMLImageElement>(null);
  const previewCanvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number | null>(null);
  const dragStateRef = useRef<DragState>(null);
  const pendingTempBboxRef = useRef<[number, number, number, number] | null>(null);
  const panStartRef = useRef<{ clientX: number; clientY: number; offset: { x: number; y: number } } | null>(null);
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

    setStageSize((current) => (
      current?.width === nextSize.width && current?.height === nextSize.height ? current : nextSize
    ));
  }, [getStageSize]);

  const resolvedStageSize = stageSize ?? (record ? {
    width: record.result.image_size[0],
    height: record.result.image_size[1],
  } : null);

  const setActiveDragState = (nextDragState: DragState) => {
    dragStateRef.current = nextDragState;
    setDragState(nextDragState);
  };

  const commitElementName = (idx: number, nextName: string) => {
    const normalizedName = normalizeClassName(nextName);
    if (!normalizedName) return;
    const previousName = normalizeClassName(elements[idx]?.class_name ?? '');
    if (previousName === normalizedName) return;

    setElements(prev => prev.map((el, elementIdx) => (
      elementIdx === idx ? { ...el, class_name: normalizedName } : el
    )));
    setAnnotationStatus(prev => ({ ...prev, [idx]: 'draft' }));

    setCustomClasses(prev => (
      hasExactClassName(normalizedName, [...classes, ...prev]) ? prev : [...prev, normalizedName]
    ));
  };

  const removeElement = useCallback((idxToRemove: number) => {
    setElements(prev => prev.filter((_, idx) => idx !== idxToRemove));
    setAnnotationStatus(prev => {
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
  }, [focusedIdx]);

  const setElementValidation = (idx: number, status: AnnotationStatus) => {
    setAnnotationStatus(prev => ({ ...prev, [idx]: status }));
  };

  const submitNamedElements = () => {
    setAnnotationStatus((prev) => {
      const next: Record<number, AnnotationStatus> = { ...prev };
      elements.forEach((el, idx) => {
        next[idx] = isUnnamedClass(el.class_name) ? 'draft' : 'validated';
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
      if (!id) { setLoading(false); return; }
      const rec = getAnalysisById(id);
      setRecord(rec);

      if (!rec) { setLoading(false); return; }
      // Deep copy elements so we can mutate safely
      setElements(JSON.parse(JSON.stringify(rec.result.elements)));
      setAnnotationStatus(rec.annotationStatus ?? {});

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
  }, [id]);

  useEffect(() => {
    if (!record) {
      setStageSize(null);
      return;
    }
    if (loading || !containerRef.current) return;

    updateStageSize();

    const container = containerRef.current;
    const handleResize = () => updateStageSize();
    window.addEventListener('resize', handleResize);

    const observer = typeof ResizeObserver === 'undefined'
      ? null
      : new ResizeObserver(() => updateStageSize());
    observer?.observe(container);

    return () => {
      window.removeEventListener('resize', handleResize);
      observer?.disconnect();
    };
  }, [loading, record, updateStageSize]);

  useEffect(() => {
    if (!record || !imageRef.current || !previewCanvasRef.current || focusedIdx === null) return;
    const el = elements[focusedIdx];
    if (!el) return;
    
    const [x, y, w, h] = dragState?.idx === focusedIdx && tempBbox ? tempBbox : el.bbox;
    const canvas = previewCanvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (imageRef.current.complete && w > 0 && h > 0) {
      ctx.imageSmoothingEnabled = false;
      const scale = Math.min(canvas.width / w, canvas.height / h);
      const fittedW = w * scale;
      const fittedH = h * scale;
      const drawX = (canvas.width - fittedW) / 2;
      const drawY = (canvas.height - fittedH) / 2;
      ctx.drawImage(imageRef.current, x, y, w, h, drawX, drawY, fittedW, fittedH);
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

  const clampPan = useCallback((offset: { x: number; y: number }, zoomLevel = zoom): { x: number; y: number } => {
    if (!containerRef.current || !record || zoomLevel <= 1) {
      return { x: 0, y: 0 };
    }

    const cRect = containerRef.current.getBoundingClientRect();
    const stage = stageSize ?? { width: record.result.image_size[0], height: record.result.image_size[1] };
    const scaledW = stage.width * zoomLevel;
    const scaledH = stage.height * zoomLevel;
    const maxPanX = (scaledW / 2) + (cRect.width / 2) - scaledW * 0.2;
    const maxPanY = (scaledH / 2) + (cRect.height / 2) - scaledH * 0.2;

    return {
      x: Math.max(-maxPanX, Math.min(maxPanX, offset.x)),
      y: Math.max(-maxPanY, Math.min(maxPanY, offset.y)),
    };
  }, [record, stageSize, zoom]);

  const applyZoom = (nextZoom: number, anchor?: { clientX: number; clientY: number }) => {
    const clampedZoom = Math.max(0.25, Math.min(4, nextZoom));

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

  useEffect(() => {
    setPanOffset((prev) => clampPan(prev, zoom));
  }, [clampPan, stageSize, zoom]);

  const getHitHandle = (x: number, y: number, bbox: [number, number, number, number]) => {
    const [bx, by, bw, bh] = bbox;
    const hSize = 12; // slightly larger hit area
    if (Math.abs(x - bx) <= hSize && Math.abs(y - by) <= hSize) return 'tl';
    if (Math.abs(x - (bx + bw)) <= hSize && Math.abs(y - by) <= hSize) return 'tr';
    if (Math.abs(x - bx) <= hSize && Math.abs(y - (by + bh)) <= hSize) return 'bl';
    if (Math.abs(x - (bx + bw)) <= hSize && Math.abs(y - (by + bh)) <= hSize) return 'br';
    return null;
  };

  const getHandleHit = (x: number, y: number) => {
    let bestHit: { idx: number; corner: 'tl' | 'tr' | 'bl' | 'br'; area: number } | null = null;
    for (const [idx, el] of elements.entries()) {
      const handle = getHitHandle(x, y, el.bbox);
      if (!handle) continue;

      const [, , bw, bh] = el.bbox;
      const area = bw * bh;
      if (!bestHit || area < bestHit.area) {
        bestHit = { idx, corner: handle, area };
      }
    }

    return bestHit;
  };

  const handleSvgPointerDown = (e: ReactPointerEvent<SVGSVGElement>) => {
    if (!record) return;
    const [imgW, imgH] = record.result.image_size;
    const coords = clientToImage(e.currentTarget, e.clientX, e.clientY, { width: imgW, height: imgH });
    const { x, y } = coords;

    if (drawMode) {
      const bbox: [number, number, number, number] = [x, y, 0, 0];
      setActiveDragState({ type: 'draw', idx: elements.length, startX: x, startY: y });
      setTempBbox(bbox);
      pendingTempBboxRef.current = bbox;
      e.currentTarget.setPointerCapture(e.pointerId);
      return;
    }

    const handleHit = getHandleHit(x, y);

    if (handleHit) {
      const el = elements[handleHit.idx];
      const bbox: [number, number, number, number] = [...el.bbox];
      setFocusedIdx(handleHit.idx);
      setActiveDragState({ type: 'resize', idx: handleHit.idx, corner: handleHit.corner, startX: x, startY: y, origBbox: bbox });
      setTempBbox(bbox);
      pendingTempBboxRef.current = bbox;
      e.currentTarget.setPointerCapture(e.pointerId);
      return;
    }

    let hitIdx: number | null = null;
    let minArea = Infinity;
    elements.forEach((el, idx) => {
      const [bx, by, bw, bh] = el.bbox;
      if (x >= bx && x <= bx + bw && y >= by && y <= by + bh) {
        const area = bw * bh;
        if (area < minArea) {
          minArea = area;
          hitIdx = idx;
        }
      }
    });

    if (hitIdx !== null) {
      const el = elements[hitIdx];
      const bbox: [number, number, number, number] = [...el.bbox];
      setFocusedIdx(hitIdx);
      setActiveDragState({ type: 'move', idx: hitIdx, startX: x, startY: y, origBbox: bbox });
      setTempBbox(bbox);
      pendingTempBboxRef.current = bbox;
      e.currentTarget.setPointerCapture(e.pointerId);
    } else {
      setFocusedIdx(null);
      if (zoom > 1) {
        setIsPanning(true);
        panStartRef.current = { clientX: e.clientX, clientY: e.clientY, offset: { ...panOffset } };
        e.currentTarget.style.cursor = 'grabbing';
        e.currentTarget.setPointerCapture(e.pointerId);
      }
    }
  };

  const handleSvgPointerMove = (e: ReactPointerEvent<SVGSVGElement>) => {
    if (!record) return;

    if (isPanning && panStartRef.current) {
      const dx = e.clientX - panStartRef.current.clientX;
      const dy = e.clientY - panStartRef.current.clientY;
      setPanOffset(clampPan({
        x: panStartRef.current.offset.x + dx,
        y: panStartRef.current.offset.y + dy,
      }));
      e.currentTarget.style.cursor = 'grabbing';
      return;
    }

    const [imgW, imgH] = record.result.image_size;
    const coords = clientToImage(e.currentTarget, e.clientX, e.clientY, { width: imgW, height: imgH });
    const { x, y } = coords;
    const [origW, origH] = [imgW, imgH];

    const activeDragState = dragStateRef.current ?? dragState;

    if (activeDragState) {
      let nextBbox: [number, number, number, number] | null = null;
      if (activeDragState.type === 'draw') {
        const minX = Math.min(activeDragState.startX, x);
        const minY = Math.min(activeDragState.startY, y);
        const maxX = Math.max(activeDragState.startX, x);
        const maxY = Math.max(activeDragState.startY, y);
        nextBbox = [
          Math.max(0, minX),
          Math.max(0, minY),
          Math.min(origW - Math.max(0, minX), maxX - minX),
          Math.min(origH - Math.max(0, minY), maxY - minY)
        ];
      } else if (activeDragState.type === 'move' && activeDragState.origBbox) {
        const dx = x - activeDragState.startX;
        const dy = y - activeDragState.startY;
        const [origBx, origBy, bw, bh] = activeDragState.origBbox;
        const bx = Math.max(0, Math.min(origW - bw, origBx + dx));
        const by = Math.max(0, Math.min(origH - bh, origBy + dy));
        nextBbox = [bx, by, bw, bh];
      } else if (activeDragState.type === 'resize' && activeDragState.origBbox && activeDragState.corner) {
        let [bx, by, bw, bh] = activeDragState.origBbox;
        if (activeDragState.corner === 'tl') {
          const nx = Math.min(bx + bw - 1, Math.max(0, x));
          const ny = Math.min(by + bh - 1, Math.max(0, y));
          bw = bx + bw - nx;
          bh = by + bh - ny;
          bx = nx;
          by = ny;
        } else if (activeDragState.corner === 'tr') {
          const ny = Math.min(by + bh - 1, Math.max(0, y));
          bw = Math.min(origW - bx, Math.max(1, x - bx));
          bh = by + bh - ny;
          by = ny;
        } else if (activeDragState.corner === 'bl') {
          const nx = Math.min(bx + bw - 1, Math.max(0, x));
          bw = bx + bw - nx;
          bx = nx;
          bh = Math.min(origH - by, Math.max(1, y - by));
        } else if (activeDragState.corner === 'br') {
          bw = Math.min(origW - bx, Math.max(1, x - bx));
          bh = Math.min(origH - by, Math.max(1, y - by));
        }
        nextBbox = [bx, by, bw, bh];
      }

      if (nextBbox) {
        scheduleTempBbox(nextBbox);
      }
    } else if (!drawMode) {
      let hitIdx: number | null = null;
      let minArea = Infinity;
      elements.forEach((el, idx) => {
        const [bx, by, bw, bh] = el.bbox;
        if (x >= bx && x <= bx + bw && y >= by && y <= by + bh) {
          const area = bw * bh;
          if (area < minArea) {
            minArea = area;
            hitIdx = idx;
          }
        }
      });
      setHoveredIdx(hitIdx);
      
      const svg = e.currentTarget;
      let cursor = 'default';
      if (focusedIdx !== null) {
        const el = elements[focusedIdx];
        if (el) {
          const handle = getHitHandle(x, y, el.bbox);
          if (handle === 'tl' || handle === 'br') cursor = 'nwse-resize';
          else if (handle === 'tr' || handle === 'bl') cursor = 'nesw-resize';
        }
      }
      if (cursor === 'default' && hitIdx !== null) {
        cursor = 'move';
      } else if (cursor === 'default' && hitIdx === null && zoom > 1) {
        cursor = 'grab';
      }
      svg.style.cursor = cursor;
    } else {
      e.currentTarget.style.cursor = 'crosshair';
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
      e.currentTarget.style.cursor = zoom > 1 ? 'grab' : 'default';
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
      pendingTempBboxRef.current = null;
      return;
    }

    const newElements = [...elements];
    if (activeDragState.type === 'draw') {
      if (finalTempBbox[2] > 5 && finalTempBbox[3] > 5) {
        newElements.push({
          bbox: finalTempBbox,
          class_name: '',
          class_label: 0,
          confidence: 1.0,
          top_k: [],
          rejected: false
        });
        setAnnotationStatus(prev => ({ ...prev, [newElements.length - 1]: 'draft' }));
        setFocusedIdx(newElements.length - 1);
        setNamingFocusToken((current) => current + 1);
      }
    } else if ((activeDragState.type === 'move' || activeDragState.type === 'resize') && activeDragState.idx < newElements.length) {
      newElements[activeDragState.idx].bbox = finalTempBbox;
      setAnnotationStatus(prev => ({ ...prev, [activeDragState.idx]: 'draft' }));
    }

    setElements(newElements);
    setActiveDragState(null);
    setTempBbox(null);
    pendingTempBboxRef.current = null;
  };

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.key === 'Delete' || e.key === 'Backspace') && focusedIdx !== null) {
        if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement || e.target instanceof HTMLTextAreaElement) return;
        removeElement(focusedIdx);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [focusedIdx, removeElement]);

  const submittedCount = elements.filter((el, idx) => annotationStatus[idx] === 'validated' && !isUnnamedClass(el.class_name)).length;

  useEffect(() => {
    if (focusedIdx === null) return;
    cardRefs.current[focusedIdx]?.scrollIntoView?.({ block: 'nearest' });
  }, [focusedIdx, elements.length]);

  const handleSave = () => {
    if (!id) return;
    setSaving(true);
    const ok = updateElements(id, elements, annotationStatus);
    if (!ok) {
      setToast({ msg: translate('save.networkError'), ok: false });
      setSaving(false);
      return;
    }
    setToast({ msg: translate('save.localSuccess'), ok: true });
    setSaving(false);
    setTimeout(() => {
      navigate('/');
    }, 300);
  };

  const handleSendSubmittedForReview = async () => {
    if (!record || !id) return;
    const submittedCandidates = elements
      .map((el, idx) => ({ el, idx }))
      .filter(({ idx }) => annotationStatus[idx] === 'validated');
    const unnamedSubmittedIndexes = submittedCandidates
      .map(({ el, idx }) => isUnnamedClass(el.class_name) ? idx : null)
      .filter((idx): idx is number => idx !== null);
    if (unnamedSubmittedIndexes.length > 0) {
      setToast({
        msg: `${t.submitBlockedUnnamed} (${unnamedSubmittedIndexes.map((idx) => `#${idx}`).join(', ')})`,
        ok: false,
      });
      setTimeout(() => setToast(null), 4000);
      return;
    }
    const submittedElements = submittedCandidates.filter(({ el }) => !isUnnamedClass(el.class_name));
    if (submittedElements.length === 0) {
      const unnamedIndexes = elements
        .map((el, idx) => isUnnamedClass(el.class_name) ? idx : null)
        .filter((idx): idx is number => idx !== null);
      setToast({
        msg: unnamedIndexes.length > 0
          ? `${t.submitBlockedUnnamed} (${unnamedIndexes.map((idx) => `#${idx}`).join(', ')})`
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
        setToast({ msg: translate('save.remoteSuccess'), ok: true });
      } else {
        let msg = translate('save.networkError');

        switch (result.error_code) {
          case 'PERMISSION_DENIED':
            msg = translate('save.permissionDenied');
            break;
          case 'DISK_FULL':
            msg = translate('save.diskFull');
            break;
          case 'STORAGE_ERROR':
            msg = translate('save.storageError', { message: result.message });
            break;
          case 'INTERNAL_ERROR':
            msg = translate('save.internalError', { traceId: result.trace_id ?? '?' });
            break;
          default:
            msg = translate('save.networkError');
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
        <Link to="/" className="text-amber-400 hover:text-amber-300 inline-flex items-center gap-2">
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
    <div className="annotation-app flex h-full w-full flex-col gap-2 overflow-hidden p-2">
      <div className="annotation-topbar flex shrink-0 items-center justify-between rounded-2xl px-4 py-3">
        <div className="flex min-w-0 items-center gap-4">
          <Link to="/" className="flex items-center gap-2 rounded-full border border-stone-700/70 bg-stone-950/70 px-3 py-1.5 text-sm font-medium text-stone-300 transition-colors hover:border-amber-500/50 hover:text-stone-50">
            <ArrowLeft size={18} /> {t.back}
          </Link>
          <h1 className="text-lg font-bold text-stone-100">{t.title}</h1>
          
          <div className="h-6 w-px bg-stone-700 mx-2" />
          
          <button
            onClick={() => setDrawMode(!drawMode)}
            className={`flex items-center gap-2 rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${drawMode ? 'bg-amber-500 text-stone-950' : 'bg-stone-800 text-stone-300 hover:bg-stone-700'}`}
          >
            {drawMode ? <PenTool size={16} /> : <MousePointer2 size={16} />}
            {drawMode ? t.drawMode : t.selectMode}
          </button>
          
          <div className="flex items-center gap-1 rounded-lg border border-stone-700 bg-stone-900 p-1">
            <button
              onClick={() => applyZoom(zoom + 0.25)}
              className="p-1.5 text-stone-400 hover:text-stone-100 hover:bg-stone-800 rounded"
              aria-label={t.zoomIn}
              title={t.zoomIn}
            >
              <ZoomIn size={16} />
            </button>
            <button
              onClick={() => { setZoom(1); setPanOffset({ x: 0, y: 0 }); }}
              className="p-1.5 text-stone-400 hover:text-stone-100 hover:bg-stone-800 rounded"
              aria-label={t.resetView}
              title={t.resetView}
            >
              <Maximize2 size={16} />
            </button>
            <button
              onClick={() => applyZoom(zoom - 0.25)}
              className="p-1.5 text-stone-400 hover:text-stone-100 hover:bg-stone-800 rounded"
              aria-label={t.zoomOut}
              title={t.zoomOut}
            >
              <ZoomOut size={16} />
            </button>
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
            {saving ? <Loader2 size={18} className="animate-spin" /> : <Save size={18} />}
            {t.saveChanges}
          </button>
          <button
            onClick={handleSendSubmittedForReview}
            disabled={sending}
            className="flex items-center gap-2 rounded-lg bg-emerald-600 px-4 py-1.5 text-sm font-semibold text-white transition-colors hover:bg-emerald-500 disabled:opacity-50"
          >
            {sending ? <Loader2 size={18} className="animate-spin" /> : <Upload size={18} />}
            {t.sendSubmittedForReview}
          </button>
        </div>
      </div>

      <div className="shrink-0 rounded-2xl border border-amber-500/20 bg-amber-500/10 px-3 py-2 text-xs text-amber-100 shadow-lg shadow-amber-950/20">
        {t.adminApprovalNotice}
      </div>

      <div className="flex min-h-0 flex-1 gap-2">
        <div ref={containerRef} className="annotation-stage-frame annotation-scrollbar relative flex flex-1 items-center justify-center overflow-auto rounded-2xl" onWheel={(e) => {
          if (e.ctrlKey) {
            e.preventDefault();
            const svgEl = e.currentTarget.querySelector('svg');
            if (!svgEl || !record) return;
            const [imgW, imgH] = record.result.image_size;
            const pointer = clientToImage(svgEl as SVGSVGElement, e.clientX, e.clientY, { width: imgW, height: imgH });
            if (pointer.x < 0 || pointer.x > imgW || pointer.y < 0 || pointer.y > imgH) return;
            applyZoom(zoom - e.deltaY * 0.01, { clientX: e.clientX, clientY: e.clientY });
          }
        }}>
          <div className="annotation-floating-toolbar absolute left-4 top-4 z-10 flex items-center gap-1 rounded-2xl p-1">
            <button
              type="button"
              onClick={() => setDrawMode(!drawMode)}
              className={`flex items-center gap-2 rounded-xl px-3 py-2 text-sm font-semibold transition-colors ${drawMode ? 'bg-amber-400 text-stone-950 shadow-lg shadow-amber-950/30' : 'text-stone-300 hover:bg-stone-800 hover:text-stone-50'}`}
            >
              {drawMode ? <PenTool size={16} /> : <MousePointer2 size={16} />}
              {drawMode ? t.drawMode : t.selectMode}
            </button>
            <div className="mx-1 h-6 w-px bg-stone-700/70" />
            <div className="flex items-center gap-1 rounded-lg border border-stone-700/70 bg-stone-950/50 p-0.5">
              <button type="button" onClick={() => applyZoom(zoom + 0.25)} className="rounded-xl p-1.5 text-stone-300 transition-colors hover:bg-stone-800 hover:text-stone-50" aria-label="Zoom avant">
                <ZoomIn size={16} />
              </button>
              <button type="button" onClick={() => { setZoom(1); setPanOffset({ x: 0, y: 0 }); }} className="rounded-xl p-1.5 text-stone-300 transition-colors hover:bg-stone-800 hover:text-stone-50" title="Réinitialiser la vue">
                <Maximize2 size={16} />
              </button>
              <button type="button" onClick={() => applyZoom(zoom - 0.25)} className="rounded-xl p-1.5 text-stone-300 transition-colors hover:bg-stone-800 hover:text-stone-50" aria-label="Zoom arrière">
                <ZoomOut size={16} />
              </button>
              <span className="px-2 text-xs font-semibold tabular-nums text-stone-400">{Math.round(zoom * 100)}%</span>
            </div>
          </div>
          <div
            data-testid="annotation-stage"
            style={{
              width: resolvedStageSize ? `${resolvedStageSize.width}px` : undefined,
              height: resolvedStageSize ? `${resolvedStageSize.height}px` : undefined,
              transform: `translate(${panOffset.x}px, ${panOffset.y}px) scale(${zoom})`,
              transformOrigin: 'center center',
              transition: isPanning ? 'none' : 'transform 0.1s ease',
              willChange: 'transform',
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
                style={{ width: '100%', height: '100%' }}
                onPointerDown={handleSvgPointerDown}
                onPointerMove={handleSvgPointerMove}
                onPointerUp={handleSvgPointerUp}
              >
                {elements.map((el, idx) => {
                  const [x, y, w, h] = idx === dragState?.idx && dragState.type !== 'draw' && tempBbox ? tempBbox : el.bbox;
                  const isFocused = idx === focusedIdx;
                  const isHovered = idx === hoveredIdx;
                  const isSubmitted = annotationStatus[idx] === 'validated';
                  const strokeColor = el.rejected ? '#fb7185' : isSubmitted ? '#34d399' : isFocused ? '#fbbf24' : isHovered ? '#f59e0b' : '#a8a29e';
                  const fillColor = el.rejected ? 'rgba(239, 68, 68, 0.16)' : isSubmitted ? 'rgba(16, 185, 129, 0.15)' : isFocused ? 'rgba(245, 158, 11, 0.18)' : isHovered ? 'rgba(245, 158, 11, 0.11)' : 'rgba(168, 162, 158, 0.08)';
                  const labelY = Math.max(0, y - 28);
                  return (
                    <g key={idx}>
                      <rect
                        x={x}
                        y={y}
                        width={w}
                        height={h}
                        fill={fillColor}
                        stroke={strokeColor}
                        strokeWidth={isFocused ? 3 : 2}
                        strokeDasharray={isSubmitted ? undefined : '8 5'}
                        vectorEffect="non-scaling-stroke"
                      />
                      {isFocused && !drawMode && (
                        <>
                          <rect x={x-6} y={y-6} width={12} height={12} rx={3} fill="#fef3c7" stroke="#0c0a09" strokeWidth={1.5} className="cursor-nwse-resize" />
                          <rect x={x+w-6} y={y-6} width={12} height={12} rx={3} fill="#fef3c7" stroke="#0c0a09" strokeWidth={1.5} className="cursor-nesw-resize" />
                          <rect x={x-6} y={y+h-6} width={12} height={12} rx={3} fill="#fef3c7" stroke="#0c0a09" strokeWidth={1.5} className="cursor-nesw-resize" />
                          <rect x={x+w-6} y={y+h-6} width={12} height={12} rx={3} fill="#fef3c7" stroke="#0c0a09" strokeWidth={1.5} className="cursor-nwse-resize" />
                        </>
                      )}
                      <rect x={x} y={labelY} width={44} height={24} rx={6} fill={strokeColor} opacity={0.95} />
                      <text x={x + 22} y={labelY + 12} fill="#0c0a09" fontSize="13" fontWeight="800" fontFamily="sans-serif" textAnchor="middle" dominantBaseline="central">#{idx}</text>
                    </g>
                  );
                })}
                {drawMode && dragState?.type === 'draw' && tempBbox && (
                  <rect x={tempBbox[0]} y={tempBbox[1]} width={tempBbox[2]} height={tempBbox[3]} fill="rgba(59, 130, 246, 0.14)" stroke="#60a5fa" strokeWidth={2} strokeDasharray="4 4" vectorEffect="non-scaling-stroke" />
                )}
              </svg>
            )}
          </div>
        </div>

        <div className="annotation-rail flex w-[360px] shrink-0 flex-col rounded-2xl p-4">
          <div className="annotation-crop mb-4 flex h-[190px] shrink-0 items-center justify-center overflow-hidden rounded-2xl border border-stone-700/50">
            {focusedIdx !== null ? (
              <canvas ref={previewCanvasRef} width={200} height={200} className="block h-[180px] w-[180px] rounded-xl object-contain" />
            ) : (
              <div className="px-4 text-center text-sm text-stone-500">{t.selectElementCrop}</div>
            )}
          </div>
          
          <div className="mb-3 flex items-center justify-between">
            <div>
              <div className="text-[10px] font-semibold uppercase tracking-[0.28em] text-stone-500">{t.elements}</div>
              <div className="text-2xl font-black leading-none text-stone-100">{elements.length}</div>
            </div>
            <div className="rounded-full border border-emerald-500/20 bg-emerald-500/10 px-3 py-1 text-xs font-semibold text-emerald-300">
              {submittedCount} {t.submitted}
            </div>
          </div>

          <div className="annotation-scrollbar flex-1 space-y-2 overflow-y-auto pr-2">
            {elements.map((el, idx) => {
              const isFocused = idx === focusedIdx;
              const displayName = isUnnamedClass(el.class_name) ? t.unnamedElement : el.class_name;
              const isSubmitted = annotationStatus[idx] === 'validated';
              const confidencePercent = Math.round(el.confidence * 100);

              return (
                <div
                  key={idx}
                  ref={(node) => { cardRefs.current[idx] = node; }}
                  onClick={() => setFocusedIdx(idx)}
                  className={`annotation-card flex cursor-pointer flex-col gap-3 rounded-2xl p-3 transition-all ${isFocused ? 'annotation-card-selected' : ''}`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex min-w-0 items-center gap-3">
                      <span className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-xl text-sm font-black ${isSubmitted ? 'bg-emerald-400 text-stone-950' : isFocused ? 'bg-amber-400 text-stone-950' : 'bg-stone-800 text-stone-300'}`}>
                        #{idx}
                      </span>
                      <div className="min-w-0">
                        <div className={`truncate text-sm font-bold ${isUnnamedClass(el.class_name) ? 'text-amber-300' : 'text-stone-100'}`}>
                          {displayName}
                        </div>
                        <div className="mt-1 flex items-center gap-2">
                          <div className="h-1.5 w-20 overflow-hidden rounded-full bg-stone-800">
                            <div
                              className={`h-full rounded-full ${el.rejected ? 'bg-red-400' : isSubmitted ? 'bg-emerald-400' : 'bg-amber-400'}`}
                              style={{ width: `${Math.max(0, Math.min(100, confidencePercent))}%` }}
                            />
                          </div>
                          <span className="text-[10px] font-semibold tabular-nums text-stone-500">{confidencePercent}%</span>
                        </div>
                      </div>
                    </div>
                    <div className="flex shrink-0 items-center gap-1">
                      <span className={`rounded-full px-2 py-0.5 text-[11px] font-bold ${el.rejected ? 'bg-red-500/15 text-red-300' : isSubmitted ? 'bg-emerald-500/15 text-emerald-300' : 'bg-stone-800 text-stone-400'}`}>
                        {isSubmitted ? t.submitted : t.draft}
                      </span>
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          removeElement(idx);
                        }}
                        className="rounded-lg p-1.5 text-stone-500 transition-colors hover:bg-red-500/10 hover:text-red-300"
                      >
                        <Trash2 size={16} />
                      </button>
                    </div>
                  </div>
                  
                  {isFocused && (
                    <>
                      <button
                        type="button"
                        onClick={(event) => {
                          event.stopPropagation();
                          setElementValidation(idx, isSubmitted ? 'draft' : 'validated');
                        }}
                        disabled={isUnnamedClass(el.class_name)}
                        className={`rounded-xl px-3 py-2 text-sm font-bold transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${isSubmitted ? 'border border-stone-700 bg-stone-900 text-stone-200 hover:bg-stone-800' : 'bg-emerald-500 text-stone-950 hover:bg-emerald-400'}`}
                      >
                        {isSubmitted ? t.markDraft : t.markSubmitted}
                      </button>
                      <ElementNameCombobox
                        value={el.class_name}
                        classNames={[el.class_name, ...classes]}
                        customClassNames={customClasses}
                        topK={el.top_k}
                        autoFocusToken={namingFocusToken}
                        labels={t}
                        index={idx}
                        onCommit={(name) => commitElementName(idx, name)}
                      />
                    </>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>
      {toast && (
        <div className={`fixed bottom-6 right-6 z-50 rounded-xl px-5 py-3 text-sm font-medium shadow-lg transition-all ${toast.ok ? 'bg-emerald-700 text-white' : 'bg-red-700 text-white'}`}>
          {toast.msg}
        </div>
      )}
    </div>
  );
}
