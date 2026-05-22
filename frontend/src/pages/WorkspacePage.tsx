import { useCallback, useEffect, useMemo, useRef, useState, type DragEvent, type PointerEvent as ReactPointerEvent, type WheelEvent as ReactWheelEvent } from 'react';
import { useLocation, useNavigate, useSearchParams } from 'react-router-dom';
import {
  AlertCircle,
  CheckCircle2,
  ChevronLeft,
  Edit3,
  ImagePlus,
  Info,
  Loader2,
  Search,
  Trash2,
  Upload,
  PanelLeftClose,
  PanelLeftOpen,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Tags,
} from 'lucide-react';
import { segmentGlyph, getTrust } from '../services/api';
import { appText } from '../i18n/text';
import { deleteAnalysis, getHistory, saveAnalysis } from '../services/storage';
import MainImagePanel from '../components/MainImagePanel';
import type { AnalysisRecord, TrustResult } from '../types';
import { clientToImage } from '../utils/imageCoords';
import {
  nextZoomFromWheel,
  shouldConsumeStageWheel,
} from '../utils/imageStageZoom';
import { getBoxVisualState, hitTestBBoxes } from '../utils/segmentationBoxes';

type OverlayMode = 'all' | 'focused' | 'hidden';
type HoverSource = 'image' | 'list' | null;

const WORKSPACE_WHEEL_ZOOM_SENSITIVITY = 0.0015;

function formatWorkspaceBboxLabel(idx: number, className: string, showName: boolean) {
  if (!showName) return `#${idx}`;
  const clippedName = className.length > 18 ? `${className.slice(0, 17)}…` : className;
  return clippedName;
}

function getCropPreviewSize(bbox: [number, number, number, number], maxSize: number) {
  const [, , boxWidth, boxHeight] = bbox;
  let width = maxSize;
  let height = maxSize;

  if (boxWidth >= boxHeight) {
    height = Math.max(1, maxSize * (boxHeight / boxWidth));
  } else {
    width = Math.max(1, maxSize * (boxWidth / boxHeight));
  }

  return { width, height };
}

function resolveCurrentRecord(records: AnalysisRecord[], preferredId?: string | null) {
  if (preferredId) {
    return records.find((record) => record.id === preferredId) ?? records[0] ?? null;
  }

  return records[0] ?? null;
}

export default function WorkspacePage() {
  const navigate = useNavigate();
  const location = useLocation();
  const [searchParams] = useSearchParams();
  const initialPreferredId = searchParams.get('analysis');

  const [records, setRecords] = useState<AnalysisRecord[]>(() => getHistory());
  const [currentRecord, setCurrentRecord] = useState<AnalysisRecord | null>(() => resolveCurrentRecord(getHistory(), initialPreferredId));
  const [filter, setFilter] = useState('');

  const [dragging, setDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const [hoverSource, setHoverSource] = useState<HoverSource>(null);
  const [focusedIdx, setFocusedIdx] = useState<number | null>(null);
  const [zoom, setZoom] = useState(1);
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(() => !resolveCurrentRecord(getHistory(), initialPreferredId));
  const [overlayMode, setOverlayMode] = useState<OverlayMode>('all');
  const [showLabelNames, setShowLabelNames] = useState(false);
  const [trustData, setTrustData] = useState<TrustResult | null>(null);
  const [contextLoading, setContextLoading] = useState(false);

  const inputRef = useRef<HTMLInputElement>(null);
  const currentFileRef = useRef<File | null>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const workspacePanStartRef = useRef<{ clientX: number; clientY: number; offset: { x: number; y: number } } | null>(null);
  const cropCanvasRefs = useRef<(HTMLCanvasElement | null)[]>([]);
  const detailCanvasRef = useRef<HTMLCanvasElement>(null);
  const t = appText.workspace;

  const resetWorkspaceView = useCallback(() => {
    setZoom(1);
    setPanOffset({ x: 0, y: 0 });
    setIsPanning(false);
    workspacePanStartRef.current = null;
  }, []);

  const resetInspectionState = useCallback(() => {
    cropCanvasRefs.current = [];
    setHoveredIdx(null);
    setHoverSource(null);
    setFocusedIdx(null);
    resetWorkspaceView();
    setOverlayMode('all');
  }, [resetWorkspaceView]);

  const selectRecord = useCallback((record: AnalysisRecord | null) => {
    resetInspectionState();
    setCurrentRecord(record);
  }, [resetInspectionState]);

  const syncRecords = useCallback((preferredId?: string | null) => {
    const nextRecords = getHistory();
    setRecords(nextRecords);
    selectRecord(resolveCurrentRecord(nextRecords, preferredId));
  }, [selectRecord]);

  useEffect(() => {
    if (location.pathname === '/' && searchParams.get('analysis')) {
      navigate('/', { replace: true });
    }
  }, [location.pathname, navigate, searchParams]);

  useEffect(() => {
    if (!currentRecord || !imageRef.current) {
      return;
    }

    const image = imageRef.current;

    const drawCrop = (canvas: HTMLCanvasElement | null, element: AnalysisRecord['result']['elements'][number]) => {
      if (!canvas) {
        return;
      }

      const cropCtx = canvas.getContext('2d');
      if (!cropCtx) {
        return;
      }

      const [x, y, w, h] = element.bbox;
      cropCtx.clearRect(0, 0, canvas.width, canvas.height);
      cropCtx.drawImage(image, x, y, w, h, 0, 0, canvas.width, canvas.height);
    };

    const drawAllCanvases = () => {
      currentRecord.result.elements.forEach((element, idx) => {
        drawCrop(cropCanvasRefs.current[idx], element);
      });

      if (focusedIdx !== null) {
        const focusedElement = currentRecord.result.elements[focusedIdx];
        if (focusedElement) {
          drawCrop(detailCanvasRef.current, focusedElement);
        }
      }
    };

    if (image.complete) {
      drawAllCanvases();
    }

    image.addEventListener('load', drawAllCanvases);
    window.addEventListener('resize', drawAllCanvases);

    return () => {
      image.removeEventListener('load', drawAllCanvases);
      window.removeEventListener('resize', drawAllCanvases);
    };
  }, [currentRecord, focusedIdx]);

  useEffect(() => {
    if (focusedIdx === null || !currentRecord) {
      return;
    }

    const element = currentRecord.result.elements[focusedIdx];
    if (!element) return;

    // eslint-disable-next-line react-hooks/set-state-in-effect -- standard fetch-loading pattern
    setContextLoading(true);

    getTrust(currentRecord.imageDataUrl, element.bbox, element.class_name, 10)
      .then((trust) => {
        setTrustData(trust);
      })
      .catch(() => {
        setTrustData(null);
      })
      .finally(() => {
        setContextLoading(false);
      });
  }, [focusedIdx, currentRecord]);

  const filteredRecords = useMemo(() => {
    if (!filter) {
      return records;
    }

    const query = filter.toLowerCase();
    return records.filter((record) => {
      const names = record.result.elements.map((element) => element.class_name.toLowerCase());
      const annotated = Object.values(record.annotations ?? {}).map((value) => value.toLowerCase());
      return (
        names.some((name) => name.includes(query)) ||
        annotated.some((annotation) => annotation.includes(query)) ||
        record.imageName.toLowerCase().includes(query)
      );
    });
  }, [filter, records]);

  const stats = useMemo(() => {
    if (!currentRecord) {
      return null;
    }

    const elements = currentRecord.result.elements;
    const rejectedCount = elements.filter((element) => element.rejected).length;
    const classCounts: Record<string, number> = {};

    elements.forEach((element, idx) => {
      const finalClass = (currentRecord.annotations ?? {})[idx] ?? element.class_name;
      classCounts[finalClass] = (classCounts[finalClass] || 0) + 1;
    });

    let topClass = 'None';
    let maxCount = 0;
    Object.entries(classCounts).forEach(([className, count]) => {
      if (count > maxCount) {
        maxCount = count;
        topClass = className;
      }
    });

    const annotatedCount = Object.keys(currentRecord.annotations ?? {}).length;
    const topClasses = Object.entries(classCounts)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 3)
      .map(([className, count]) => `${className} ${count}`);

    return {
      total: elements.length,
      rejectedCount,
      annotatedCount,
      submittedCount: annotatedCount,
      topClass,
      topClasses,
      imageSizeLabel: `${currentRecord.result.image_size[0]}×${currentRecord.result.image_size[1]}`,
    };
  }, [currentRecord]);

  const handleFile = useCallback((nextFile: File) => {
    currentFileRef.current = nextFile;
    setFile(nextFile);
    setPreview(null);
    setError(null);

    const reader = new FileReader();
    reader.onload = (event) => {
      if (currentFileRef.current === nextFile) {
        setPreview(event.target?.result as string);
      }
    };
    reader.readAsDataURL(nextFile);
  }, []);

  const clearPendingFile = useCallback(() => {
    currentFileRef.current = null;
    setFile(null);
    setPreview(null);
    setError(null);
    if (inputRef.current) {
      inputRef.current.value = '';
    }
  }, []);

  const onDrop = useCallback((event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    const nextFile = event.dataTransfer.files[0];
    if (nextFile) {
      handleFile(nextFile);
    }
  }, [handleFile]);

  const analyze = async () => {
    if (!file || !preview) {
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await segmentGlyph(file);
      const record: AnalysisRecord = {
        id: crypto.randomUUID(),
        imageName: file.name,
        imageDataUrl: preview,
        timestamp: Date.now(),
        result,
        annotations: {},
      };

      saveAnalysis(record);
      syncRecords(record.id);
      clearPendingFile();
    } catch (issue) {
      setError(issue instanceof Error ? issue.message : t.apiError);
    } finally {
      setLoading(false);
    }
  };

  const removeRecord = (id: string) => {
    deleteAnalysis(id);
    syncRecords(currentRecord?.id === id ? null : currentRecord?.id);
  };

  const handleEditorHandoff = () => {
    if (!currentRecord) {
      return;
    }

    navigate(
      focusedIdx !== null
        ? `/annotate/${currentRecord.id}?element=${focusedIdx}`
        : `/annotate/${currentRecord.id}`,
    );
  };

  const startWorkspacePan = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (zoom <= 1 || event.button !== 0) {
      return;
    }

    const target = event.target;
    if (target instanceof Element && target.closest('[data-overlay-region="true"]')) {
      return;
    }

    setIsPanning(true);
    workspacePanStartRef.current = {
      clientX: event.clientX,
      clientY: event.clientY,
      offset: { ...panOffset },
    };
    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const moveWorkspacePan = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!workspacePanStartRef.current) {
      return;
    }

    setPanOffset({
      x: workspacePanStartRef.current.offset.x + event.clientX - workspacePanStartRef.current.clientX,
      y: workspacePanStartRef.current.offset.y + event.clientY - workspacePanStartRef.current.clientY,
    });
  };

  const stopWorkspacePan = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!workspacePanStartRef.current && !isPanning) {
      return;
    }

    setIsPanning(false);
    workspacePanStartRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  };

  const handleWorkspaceStageWheel = (event: ReactWheelEvent<HTMLDivElement>) => {
    const { deltaY } = event;
    if (!shouldConsumeStageWheel(deltaY)) return;

    event.preventDefault();
    setZoom((currentZoom) => {
      const nextZoom = nextZoomFromWheel(
        currentZoom,
        deltaY,
        undefined,
        WORKSPACE_WHEEL_ZOOM_SENSITIVITY,
      );
      if (nextZoom <= 1) {
        setPanOffset({ x: 0, y: 0 });
      }
      return nextZoom;
    });
  };

  const getWorkspaceOverlayHit = (svg: SVGSVGElement, clientX: number, clientY: number): number | null => {
    if (!currentRecord) {
      return null;
    }

    const [imgW, imgH] = currentRecord.result.image_size;
    const point = clientToImage(svg, clientX, clientY, { width: imgW, height: imgH });
    const visibleElements = currentRecord.result.elements
      .map((element, idx) => ({ idx, bbox: element.bbox }))
      .filter(({ idx }) => overlayMode === 'all' || focusedIdx === null || focusedIdx === idx);
    const hitIdx = hitTestBBoxes(point, visibleElements.map(({ bbox }) => bbox));

    return hitIdx === null ? null : visibleElements[hitIdx].idx;
  };

  const handleWorkspaceOverlayPointerDown = (event: ReactPointerEvent<SVGSVGElement>) => {
    if (event.button !== 0) {
      return;
    }

    const hitIdx = getWorkspaceOverlayHit(event.currentTarget, event.clientX, event.clientY);
    if (hitIdx === null) {
      return;
    }

    event.stopPropagation();
    setFocusedIdx(hitIdx);
    setHoveredIdx(hitIdx);
    setHoverSource(hitIdx === null ? null : 'image');
  };

  const handleWorkspaceOverlayPointerMove = (event: ReactPointerEvent<SVGSVGElement>) => {
    const hitIdx = getWorkspaceOverlayHit(event.currentTarget, event.clientX, event.clientY);
    setHoveredIdx(hitIdx);
    setHoverSource(hitIdx === null ? null : 'image');
    event.currentTarget.style.cursor = hitIdx === null ? (zoom > 1 ? 'grab' : 'default') : 'pointer';
  };

  return (
    <div
      className={`workspace-page flex h-full min-h-0 flex-col gap-3 overflow-hidden rounded-2xl transition-colors ${dragging ? 'ring-2 ring-amber-400/60 ring-offset-2 ring-offset-stone-950' : ''}`}
      onDragEnter={(event) => {
        event.preventDefault();
        setDragging(true);
      }}
      onDragOver={(event) => {
        event.preventDefault();
        setDragging(true);
      }}
      onDragLeave={(event) => {
        if (event.relatedTarget instanceof Node && event.currentTarget.contains(event.relatedTarget)) {
          return;
        }
        setDragging(false);
      }}
      onDrop={onDrop}
    >
      <section className="flex shrink-0 flex-col gap-3 rounded-2xl border border-stone-800 bg-stone-900/80 p-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-3 px-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-amber-500 text-stone-950">
            <ImagePlus size={18} />
          </div>
          <span className="font-semibold tracking-tight text-stone-100">{t.appTitle}</span>
        </div>

        <div className="flex flex-1 items-center justify-end gap-3">
          {error && (
            <div className="flex items-center gap-2 rounded-lg bg-red-400/10 px-3 py-1.5 text-xs text-red-300">
              <AlertCircle size={14} className="shrink-0" />
              <span className="max-w-[300px] truncate">{error}</span>
            </div>
          )}

          <div
            className={`flex items-center gap-3 rounded-xl border border-dashed px-4 py-2 transition-colors ${dragging ? 'border-amber-400 bg-amber-400/10' : 'border-stone-700/80 bg-stone-950/60 hover:border-stone-500'}`}
            onClick={() => inputRef.current?.click()}
            role="button"
            tabIndex={0}
          >
            <input
              ref={inputRef}
              type="file"
              accept="image/png,image/jpeg,image/bmp,image/*"
              className="hidden"
              onChange={(event) => {
                const nextFile = event.target.files?.[0];
                if (nextFile) {
                  handleFile(nextFile);
                }
              }}
            />
            {preview ? (
              <div className="flex items-center gap-3">
                <img src={preview} alt={t.previewAlt} className="h-8 w-8 rounded object-cover" />
                <div className="flex flex-col">
                  <span className="max-w-[120px] truncate text-xs font-medium text-stone-100">{file?.name}</span>
                </div>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    analyze();
                  }}
                  disabled={loading}
                  className="ml-2 inline-flex h-8 items-center justify-center gap-2 rounded-lg bg-amber-500 px-3 text-xs font-semibold text-stone-950 transition-colors hover:bg-amber-400 disabled:opacity-50"
                >
                  {loading ? <><Loader2 size={14} className="animate-spin" /> {t.analyzing}</> : <>{t.analyze}</>}
                </button>
              </div>
            ) : (
              <div className="flex items-center gap-2 text-sm text-stone-400">
                <Upload size={16} />
                <span>{t.uploadPrompt}</span>
              </div>
            )}
          </div>
        </div>
      </section>

      {preview && file && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-stone-950/80 p-6 backdrop-blur-sm">
          <div className="w-full max-w-xl rounded-[28px] border border-stone-800 bg-stone-900 p-5 shadow-[0_30px_90px_rgba(0,0,0,0.45)]">
            <div className="flex items-start justify-between gap-4 border-b border-stone-800 pb-4">
              <div>
                <p className="text-xs uppercase tracking-[0.24em] text-stone-500">{t.uploadModalTitle}</p>
                <h2 className="mt-1 truncate text-lg font-semibold text-stone-100">{file.name}</h2>
                <p className="mt-1 text-sm text-stone-400">{t.uploadModalDescription}</p>
              </div>
              <button
                type="button"
                onClick={clearPendingFile}
                disabled={loading}
                className="rounded-xl border border-stone-700 px-3 py-2 text-sm font-semibold text-stone-300 transition-colors hover:border-stone-500 hover:text-stone-100 disabled:opacity-50"
              >
                {t.cancel}
              </button>
            </div>

            <div className="my-5 flex max-h-[46vh] items-center justify-center overflow-hidden rounded-2xl border border-stone-800 bg-stone-950">
              <img src={preview} alt={t.previewAlt} className="max-h-[46vh] max-w-full object-contain" />
            </div>

            <div className="flex justify-end gap-3">
              <button
                type="button"
                onClick={clearPendingFile}
                disabled={loading}
                className="rounded-xl border border-stone-700 px-4 py-2 text-sm font-semibold text-stone-300 transition-colors hover:border-stone-500 hover:text-stone-100 disabled:opacity-50"
              >
                {t.cancel}
              </button>
              <button
                type="button"
                onClick={analyze}
                disabled={loading}
                className="inline-flex items-center justify-center gap-2 rounded-xl bg-amber-500 px-5 py-2 text-sm font-bold text-stone-950 transition-colors hover:bg-amber-400 disabled:opacity-50"
              >
                {loading ? <><Loader2 size={16} className="animate-spin" /> {t.analyzing}</> : t.analyze}
              </button>
            </div>
          </div>
        </div>
      )}

      <div className={`grid min-h-0 flex-1 gap-3 transition-[grid-template-columns] duration-300 ease-out ${historyOpen ? 'xl:grid-cols-[300px_minmax(0,1fr)]' : 'xl:grid-cols-[56px_minmax(0,1fr)]'}`}>
        <aside
          data-testid="workspace-history-sidebar"
          className={`flex flex-col overflow-hidden transition-[padding,border-color,background-color] duration-200 ease-out ${historyOpen ? 'min-h-0 rounded-2xl border border-stone-800 bg-stone-900/75 p-3' : 'min-h-0 items-center rounded-2xl border border-stone-800 bg-stone-900/75 py-3'}`}
        >
              {!historyOpen ? (
                <div className="flex flex-col items-center w-full h-full overflow-hidden">
                  <button
                    type="button"
                    onClick={() => setHistoryOpen(true)}
                    className="flex h-10 w-10 items-center justify-center rounded-xl bg-stone-950 text-stone-400 transition-colors hover:bg-stone-800 hover:text-stone-100 shrink-0"
                    title={t.expandHistory}
                  >
                    <PanelLeftOpen size={18} />
                  </button>

                  <div className="my-3 h-px w-8 bg-stone-800 shrink-0" />

                  <div className="flex-1 flex flex-col items-center gap-2 overflow-y-auto w-full px-1 py-1">
                    {records.map((record) => (
                      <button
                        key={record.id}
                        type="button"
                        onClick={() => selectRecord(record)}
                        className={`relative rounded-lg overflow-hidden border transition-colors ${
                          currentRecord?.id === record.id
                            ? 'border-amber-500/60 shadow-[0_0_0_1px_rgba(245,158,11,0.25)]'
                            : 'border-stone-800 hover:border-stone-600'
                        }`}
                        title={record.imageName}
                      >
                        <img
                          src={record.imageDataUrl}
                          alt={record.imageName}
                          className="h-10 w-10 object-cover"
                        />
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
            <>
              <div className="mb-3 flex items-center justify-between gap-2 border-b border-stone-800 pb-3" data-testid="workspace-history-header">
                <div className="flex min-w-0 items-center gap-2">
                  <h2 className="text-sm font-semibold text-stone-100">History</h2>
                  <span className="rounded-full border border-stone-800 bg-stone-950/70 px-2 py-0.5 text-xs font-medium tabular-nums text-stone-400">
                    {filteredRecords.length} total
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => setHistoryOpen(false)}
                    className="flex h-10 w-10 items-center justify-center rounded-xl bg-stone-950 text-stone-400 transition-colors hover:bg-stone-800 hover:text-stone-100"
                    title={t.collapseHistory}
                  >
                    <PanelLeftClose size={18} />
                  </button>
                </div>
              </div>

              <div className="relative mb-3">
                <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-stone-500" />
                <input
                  value={filter}
                  onChange={(event) => setFilter(event.target.value)}
                  placeholder={t.filterPlaceholder}
                  className="w-full rounded-xl border border-stone-700 bg-stone-950 px-10 py-2.5 text-sm text-stone-100 outline-none transition-colors focus:border-amber-400"
                />
              </div>

              <div className="min-h-0 flex-1 space-y-2 overflow-y-auto pr-1" data-testid="workspace-history-list">
                {records.length === 0 ? (
                  <div className="rounded-2xl border border-dashed border-stone-700 bg-stone-950/70 px-4 py-8 text-center text-sm text-stone-500">
                    {t.noAnalyses}
                  </div>
                ) : filteredRecords.length === 0 ? (
                  <div className="rounded-2xl border border-dashed border-stone-700 bg-stone-950/70 px-4 py-8 text-center text-sm text-stone-500">
                    {t.noFilterMatch}
                  </div>
                ) : (
              filteredRecords.map((record) => {
                const isActive = currentRecord?.id === record.id;
                const badges = [
                  ...new Set(
                    record.result.elements
                      .map((element, idx) => (!element.rejected ? (record.annotations ?? {})[idx] ?? element.class_name : null))
                      .filter((value): value is string => Boolean(value)),
                  ),
                ].slice(0, 3);

                return (
                  <div
                    key={record.id}
                    role="button"
                    tabIndex={0}
                    onClick={() => selectRecord(record)}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter' || event.key === ' ') {
                        event.preventDefault();
                        selectRecord(record);
                      }
                    }}
                    className={`group rounded-xl border p-2.5 transition-colors ${isActive ? 'border-amber-500/60 bg-amber-500/10 shadow-[0_0_0_1px_rgba(245,158,11,0.25)]' : 'border-stone-800 bg-stone-950/70 hover:border-stone-700'}`}
                  >
                    <div className="flex items-start gap-3">
                      <img src={record.imageDataUrl} alt={record.imageName} className="h-16 w-16 rounded-lg border border-stone-800 object-cover" />
                      <div className="min-w-0 flex-1">
                        <div className="flex items-start justify-between gap-2">
                          <div>
                            <p className="truncate text-sm font-medium text-stone-100">{record.imageName}</p>
                            <p className="mt-1 text-xs text-stone-500">{new Date(record.timestamp).toLocaleDateString()} · {record.result.num_elements} {t.elementsSuffix}</p>
                          </div>
                          <button
                            type="button"
                            onClick={(event) => {
                              event.stopPropagation();
                              removeRecord(record.id);
                            }}
                            className="rounded-lg p-1.5 text-stone-500 transition-colors hover:bg-stone-800 hover:text-red-300"
                            aria-label={`${t.deleteLabel} ${record.imageName}`}
                          >
                            <Trash2 size={14} />
                          </button>
                        </div>
                        <div className="mt-3 flex flex-wrap gap-1.5">
                          {badges.map((badge) => (
                            <span key={badge} className="rounded-md bg-amber-400/10 px-2 py-1 text-[11px] text-amber-300">
                              {badge}
                            </span>
                          ))}
                          {record.result.elements.some((element) => element.rejected) && (
                            <span className="rounded-md bg-red-400/10 px-2 py-1 text-[11px] text-red-300">
                              {record.result.elements.filter((element) => element.rejected).length} {t.rejectedSuffix}
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })
            )}
          </div>
            </>
          )}
        </aside>

        <section className="min-h-0 overflow-hidden">
          {currentRecord ? (
            <>
              <div className="grid h-full min-h-0 gap-3 xl:grid-cols-[minmax(0,1.45fr)_minmax(280px,0.55fr)] 2xl:grid-cols-[minmax(0,1.55fr)_minmax(320px,0.45fr)]" data-testid="workspace-content-grid">
                <MainImagePanel
                  tone="workspace"
                  className="h-full"
                  title={(
                    <h2 className="truncate text-lg font-semibold text-stone-100 max-w-[200px] sm:max-w-[300px]" title={currentRecord.imageName}>
                      {currentRecord.imageName}
                    </h2>
                  )}
                  badges={focusedIdx !== null && (
                    <span className="rounded-md border border-amber-500/20 bg-amber-500/10 px-2 py-1 text-xs font-medium text-amber-500">
                      {t.focusLabel}: {focusedIdx}
                    </span>
                  )}
                  headerActions={(
                    <>
                      <div className="inline-flex rounded-lg border border-stone-700 bg-stone-950 p-1">
                        {(['all', 'focused', 'hidden'] as OverlayMode[]).map((mode) => {
                          const isActive = overlayMode === mode;
                          return (
                            <button
                              key={mode}
                              type="button"
                              onClick={() => setOverlayMode(mode)}
                              className={`rounded-md px-3 py-1.5 text-sm font-medium capitalize transition-colors ${isActive ? 'bg-amber-500 text-stone-950' : 'text-stone-400 hover:text-stone-100'}`}
                            >
                              {mode === 'all' ? t.overlayAll : mode === 'focused' ? t.overlayFocused : t.overlayHidden}
                            </button>
                          );
                        })}
                      </div>
                      <button
                        type="button"
                        onClick={() => setShowLabelNames((current) => !current)}
                        aria-pressed={showLabelNames}
                        aria-label={showLabelNames ? 'Masquer les noms des libellés' : 'Afficher les noms des libellés'}
                        title={showLabelNames ? 'Masquer les noms des libellés' : 'Afficher les noms des libellés'}
                        className={`inline-flex h-10 w-10 items-center justify-center rounded-lg border text-sm font-semibold transition-colors ${showLabelNames ? 'border-stone-100 bg-stone-100 text-stone-950' : 'border-stone-700 bg-stone-950 text-stone-300 hover:text-stone-100'}`}
                      >
                        {showLabelNames ? <Tags size={16} aria-hidden="true" /> : <span aria-hidden="true" className="text-base font-black leading-none">#</span>}
                      </button>
                    </>
                  )}
                  stageClassName={`workspace-stage ${zoom > 1 ? (isPanning ? 'cursor-grabbing' : 'cursor-grab') : ''}`}
                  stageProps={{
                    onPointerDown: startWorkspacePan,
                    onPointerMove: moveWorkspacePan,
                    onPointerUp: stopWorkspacePan,
                    onPointerCancel: stopWorkspacePan,
                    onWheel: handleWorkspaceStageWheel,
                  }}
                  transformStyle={{
                    transform: `translate(${panOffset.x}px, ${panOffset.y}px) scale(${zoom})`,
                    transformOrigin: 'center center',
                    transition: isPanning ? 'none' : 'transform 0.1s ease',
                    willChange: 'transform',
                  }}
                  image={(
                    <img
                      ref={imageRef}
                      src={currentRecord.imageDataUrl}
                      alt={currentRecord.imageName}
                      draggable={false}
                      className="block max-h-full max-w-full rounded-lg object-contain"
                    />
                  )}
                  overlay={currentRecord && overlayMode !== 'hidden' && (
                    <svg
                      className="absolute left-0 top-0 h-full w-full"
                      data-testid="workspace-overlay"
                      viewBox={`0 0 ${currentRecord.result.image_size[0]} ${currentRecord.result.image_size[1]}`}
                      preserveAspectRatio="none"
                      style={{ width: '100%', height: '100%' }}
                      onPointerDown={handleWorkspaceOverlayPointerDown}
                      onPointerMove={handleWorkspaceOverlayPointerMove}
                      onPointerLeave={() => { setHoveredIdx(null); setHoverSource(null); }}
                    >
                      {currentRecord.result.elements.map((el, idx) => {
                        if (overlayMode === 'focused' && focusedIdx !== null && focusedIdx !== idx) return null;
                        const [x, y, w, h] = el.bbox;
                        const visualState = getBoxVisualState({
                          focused: idx === focusedIdx,
                          hovered: idx === hoveredIdx && hoverSource === 'image',
                          listHovered: idx === hoveredIdx && hoverSource === 'list',
                          submitted: (currentRecord.annotations ?? {})[idx] !== undefined,
                          rejected: el.rejected,
                        });
                        const labelY = Math.max(0, y - 28);
                        return (
                          <g key={idx} data-overlay-region="true" className="pointer-events-none select-none">
                            <rect
                              x={x}
                              y={y}
                              width={w}
                              height={h}
                              fill={visualState.fillColor}
                              stroke={visualState.strokeColor}
                              strokeWidth={visualState.strokeWidth}
                              strokeDasharray={visualState.strokeDasharray}
                              vectorEffect="non-scaling-stroke"
                            />
                            {(() => {
                              const label = formatWorkspaceBboxLabel(idx, el.class_name, showLabelNames);
                              const labelWidth = showLabelNames ? Math.min(168, Math.max(64, label.length * 8 + 18)) : 44;
                              return (
                                <>
                                  <title>{label}</title>
                                  <rect x={x} y={labelY} width={labelWidth} height={24} rx={6} fill={visualState.strokeColor} opacity={0.95} />
                                  <text x={x + 10} y={labelY + 12} fill={visualState.labelColor} fontSize="13" fontWeight="800" fontFamily="sans-serif" textAnchor="start" dominantBaseline="central">{label}</text>
                                </>
                              );
                            })()}
                          </g>
                        );
                      })}
                    </svg>
                  )}
                  controls={[
                    {
                      id: 'zoom-in',
                      label: t.zoomIn,
                      title: t.zoomIn,
                      onClick: () => setZoom((z) => Math.min(4, z + 0.25)),
                      icon: <ZoomIn size={16} />,
                    },
                    {
                      id: 'fit-to-view',
                      label: t.fitToView,
                      title: t.fitToView,
                      onClick: resetWorkspaceView,
                      icon: <Maximize2 size={16} />,
                    },
                    {
                      id: 'zoom-out',
                      label: t.zoomOut,
                      title: t.zoomOut,
                      onClick: () => {
                        setZoom((currentZoom) => {
                          const nextZoom = Math.max(0.25, currentZoom - 0.25);
                          if (nextZoom <= 1) {
                            setPanOffset({ x: 0, y: 0 });
                          }
                          return nextZoom;
                        });
                      },
                      icon: <ZoomOut size={16} />,
                    },
                  ]}
                  testIds={{ stage: 'workspace-stage' }}
                />

                <section className="flex min-h-0 flex-col overflow-hidden rounded-2xl bg-stone-900/35 sidebar-shell" data-testid="workspace-detected-panel">
                  {focusedIdx !== null ? (
                    <div className="flex h-full flex-col">
                      <div className="flex items-center justify-between sidebar-header px-5 py-4 border-b border-stone-800/60">
                        <button 
                          onClick={() => setFocusedIdx(null)}
                          className="flex items-center gap-2 text-stone-400 hover:text-stone-100 transition-colors"
                        >
                          <ChevronLeft size={16} />
                        <span className="text-sm font-medium">{t.backToRegions}</span>
                        </button>
                      </div>
                      
                      <div className="annotation-scrollbar flex-1 overflow-y-auto sidebar-body p-5 space-y-6">
                          {(() => {
                          const element = currentRecord.result.elements[focusedIdx];
                          const trust = trustData?.trust;
                          const summaryClass = trust?.top1_class ?? element.class_name;
                          const topPrediction = trust?.top1_similarity ?? element.confidence;
                          const runnerUp = trust?.top_k?.[1]?.confidence ?? element.top_k[1]?.confidence ?? 0;
                          const margin = trust?.margin_to_second ?? (topPrediction - runnerUp);
                          const isRejected = trust ? !trust.above_rejection_threshold : element.rejected;
                          const isAmbiguous = trust?.ambiguous ?? (margin < 0.05);
                          const detailPreviewSize = getCropPreviewSize(element.bbox, 180);
                          const initialPredictionDiffers = Boolean(trust && trust.top1_class !== element.class_name);
                          
                          return (
                            <>
                              <div className="flex flex-col gap-3 p-1">
                                <div className="flex items-center justify-between">
                                  <h4 className="text-base font-semibold text-stone-300">{t.segmentPreview}</h4>
                                </div>
                                <div className="flex min-h-[180px] items-center justify-center rounded-2xl bg-stone-950/55 p-3">
                                  <canvas
                                    ref={detailCanvasRef}
                                    width={detailPreviewSize.width}
                                    height={detailPreviewSize.height}
                                    className="max-h-[180px] max-w-full rounded-lg bg-stone-800"
                                  />
                                </div>
                              </div>

                              <div className="flex flex-col gap-3 p-1">
                                <div className="flex items-center justify-between">
                                  <h4 className="text-base font-semibold text-stone-300">{t.trustSummary}</h4>
                                  {contextLoading && <Loader2 size={14} className="animate-spin text-stone-500" />}
                                </div>
                                <div className="flex items-center justify-between">
                                  <span className="text-2xl font-bold text-stone-100 truncate">{summaryClass}</span>
                                  <span className={`shrink-0 px-3 py-1.5 rounded-md text-base font-bold ${
                                    isRejected
                                      ? 'bg-red-500/10 text-red-400 border border-red-500/20' 
                                      : 'bg-green-500/10 text-green-400 border border-green-500/20'
                                  }`}>
                                    {(topPrediction * 100).toFixed(1)}%
                                  </span>
                                </div>
                                {initialPredictionDiffers && trust && (
                                  <div className="rounded-lg border border-amber-500/20 bg-amber-500/10 p-2 text-xs text-amber-300">
                                    <div className="font-semibold">{t.recalculatedPrediction}</div>
                                    <div className="mt-1 text-amber-200/80">
                                      {t.initialProposal}: <span className="font-medium text-amber-200">{element.class_name}</span>
                                      {' '}#{trust.predicted_class_rank} · {(trust.predicted_class_similarity * 100).toFixed(1)}%
                                    </div>
                                  </div>
                                )}
                                {isAmbiguous && !isRejected && (
                                  <div className="flex items-start gap-2 rounded-lg border border-amber-500/20 bg-amber-500/10 p-3 text-amber-400">
                                    <AlertCircle size={16} className="shrink-0 mt-0.5" />
                                    <div className="text-sm">
                                      <strong>{t.ambiguousPrediction}</strong>
                                      <p className="mt-1 text-xs opacity-80">{t.ambiguousDetails} {(margin * 100).toFixed(1)}%. {t.alternativesExist}</p>
                                    </div>
                                  </div>
                                )}
                                {isRejected && (
                                  <div className="flex items-start gap-2 rounded-lg border border-red-500/20 bg-red-500/10 p-3 text-red-400">
                                    <AlertCircle size={16} className="shrink-0 mt-0.5" />
                                    <div className="text-sm">
                                      <strong>{t.lowConfidenceFlag}</strong>
                                      <p className="mt-1 text-xs opacity-80">{t.thresholdDetails}</p>
                                    </div>
                                  </div>
                                )}
                                {trust && (
                                  <div className="flex gap-2 text-sm">
                                    <div className="flex-1 rounded-lg bg-stone-950/50 p-2">
                                      <span className="text-stone-500 block">{t.rank}</span>
                                      <span className="text-stone-200 font-semibold">#1</span>
                                    </div>
                                    <div className="flex-1 rounded-lg bg-stone-950/50 p-2">
                                      <span className="text-stone-500 block">{t.margin}</span>
                                      <span className={`font-semibold ${isAmbiguous ? 'text-amber-400' : 'text-stone-200'}`}>{(margin * 100).toFixed(1)}%</span>
                                    </div>
                                  </div>
                                )}
                              </div>

                              <div className="flex flex-col gap-3 p-1">
                                <div className="flex items-center justify-between">
                                  <h4 className="text-base font-semibold text-stone-300">{t.topPredictions}</h4>
                                </div>
                                <div className="space-y-2">
                                  {(trust?.top_k ?? element.top_k).map((item, i) => (
                                    <div key={i} className="flex items-center gap-2">
                                      <span className="w-4 text-[10px] text-stone-500 text-right">{i + 1}</span>
                                      <div className="flex-1">
                                        <div className="flex justify-between text-sm mb-0.5">
                                          <span className={i === 0 ? "text-stone-200 font-medium" : "text-stone-400"}>{item.class_name}</span>
                                          <span className={i === 0 ? "text-stone-300 font-medium" : "text-stone-500"}>{(item.confidence * 100).toFixed(1)}%</span>
                                        </div>
                                        <div className="h-1 w-full overflow-hidden rounded-full bg-stone-800">
                                          <div 
                                            className={`h-full rounded-full ${i === 0 ? (isRejected ? 'bg-amber-500' : 'bg-green-500') : 'bg-stone-600'}`}
                                            style={{ width: `${item.confidence * 100}%` }}
                                          />
                                        </div>
                                      </div>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            </>
                          );
                        })()}
                      </div>
                      
                      <div className="p-5 border-t border-stone-800 sidebar-header">
                         <button
                           type="button"
                           onClick={handleEditorHandoff}
                           className="w-full flex items-center justify-center gap-2 rounded-xl bg-amber-500 px-4 py-3 text-sm font-bold text-stone-950 transition-all hover:bg-amber-400 active:scale-[0.98]"
                         >
                           <Edit3 size={16} />
                            {currentRecord.result.elements[focusedIdx].rejected ? t.correctElement : t.annotateRegion}
                         </button>
                      </div>
                    </div>
                  ) : (
                    <div className="flex h-full flex-col p-4 lg:p-5">
                  <div className="mb-3 flex items-start justify-between gap-3 border-b border-stone-800 pb-3">
                    <div>
                      <p className="text-xs uppercase tracking-[0.24em] text-stone-500">{t.proposalPanel}</p>
                      <h2 className="mt-1 text-base font-semibold text-stone-100">{t.detectedElements}</h2>
                    </div>
                    <div className="text-right">
                      <button
                        type="button"
                        onClick={handleEditorHandoff}
                        className="inline-flex items-center gap-2 rounded-xl bg-amber-500 px-3 py-2 text-sm font-semibold text-stone-950 transition-colors hover:bg-amber-400"
                      >
                        <Edit3 size={16} /> {t.annotateRecord}
                      </button>
                    </div>
                  </div>

                  {stats && stats.rejectedCount === stats.total && stats.total > 0 && (
                    <div className="mb-4 flex items-start gap-3 rounded-2xl border border-amber-500/20 bg-amber-500/10 p-3">
                      <Info className="mt-0.5 shrink-0 text-amber-500" size={18} />
                      <div>
                        <p className="text-sm text-stone-200">{t.allRejected}</p>
                        <button type="button" onClick={handleEditorHandoff} className="mt-1 text-xs font-medium text-amber-400 hover:text-amber-300">
                          {t.goToAnnotation}
                        </button>
                      </div>
                    </div>
                  )}

                  {stats && (
                    <div className="mb-3 grid grid-cols-2 gap-2 text-xs">
                      <div className="rounded-xl border border-stone-800 bg-stone-950/55 p-2">
                        <span className="block text-stone-500">Image</span>
                        <span className="mt-1 block truncate font-semibold text-stone-200" title={currentRecord.imageName}>{currentRecord.imageName}</span>
                      </div>
                      <div className="rounded-xl border border-stone-800 bg-stone-950/55 p-2">
                        <span className="block text-stone-500">Dimensions</span>
                        <span className="mt-1 block font-semibold tabular-nums text-stone-200">{stats.imageSizeLabel}</span>
                      </div>
                      <div className="rounded-xl border border-stone-800 bg-stone-950/55 p-2">
                        <span className="block text-stone-500">Annotés / rejetés</span>
                        <span className="mt-1 block font-semibold tabular-nums text-stone-200">{stats.annotatedCount}/{stats.total} · {stats.rejectedCount}</span>
                      </div>
                      <div className="rounded-xl border border-stone-800 bg-stone-950/55 p-2">
                        <span className="block text-stone-500">Classes</span>
                        <span className="mt-1 block truncate font-semibold text-stone-200" title={stats.topClasses.join(', ')}>{stats.topClasses.join(', ') || stats.topClass}</span>
                      </div>
                    </div>
                  )}

                  <div
                    className="annotation-scrollbar workspace-detected-grid grid flex-1 auto-rows-min grid-cols-1 gap-2 overflow-y-auto pr-2"
                    data-testid="workspace-detected-list"
                    tabIndex={0}
                    onKeyDown={(e) => {
                      if (!currentRecord) return;
                      const total = currentRecord.result.elements.length;
                      if (total === 0) return;
                      if (e.key === 'ArrowDown' || e.key === 'ArrowRight') {
                        e.preventDefault();
                        setFocusedIdx((prev) => (prev === null ? 0 : (prev + 1) % total));
                      } else if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') {
                        e.preventDefault();
                        setFocusedIdx((prev) => (prev === null ? total - 1 : (prev - 1 + total) % total));
                      } else if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        if (focusedIdx !== null) setFocusedIdx(focusedIdx);
                      } else if (e.key === 'Escape') {
                        setFocusedIdx(null);
                      }
                    }}
                  >
                    {currentRecord.result.elements.length === 0 ? (
                      <div className="flex h-full flex-col items-center justify-center rounded-2xl border border-dashed border-stone-800 bg-stone-950/50 p-6 text-center">
                        <Info className="mb-3 text-stone-600" size={32} />
                        <p className="text-stone-400">{t.noElements}</p>
                      </div>
                    ) : (
                      currentRecord.result.elements.map((element, idx) => {
                        const hasAnnotation = (currentRecord.annotations ?? {})[idx] !== undefined;
                        const displayClass = hasAnnotation ? (currentRecord.annotations ?? {})[idx] : element.class_name;
                        const isHovered = hoveredIdx === idx;
                        const badgeClasses = element.rejected
                          ? 'bg-red-400/10 text-red-400 border border-red-400/20'
                          : 'bg-amber-400/10 text-amber-400 border border-amber-400/20';
                        const indexBadgeClasses = element.rejected ? 'bg-red-500 text-stone-950' : 'bg-amber-500 text-stone-950';

                        const [, , boxWidth, boxHeight] = element.bbox;
                        let destinationWidth = 48;
                        let destinationHeight = 48;
                        if (boxWidth >= boxHeight) {
                          destinationHeight = Math.max(1, 48 * (boxHeight / boxWidth));
                        } else {
                          destinationWidth = Math.max(1, 48 * (boxWidth / boxHeight));
                        }

                        return (
                          <div key={idx} className="flex flex-col">
                             <div
                              role="button"
                              aria-label={`${displayClass} région ${idx}`}
                              className={`flex cursor-pointer items-center gap-2.5 rounded-xl border px-2.5 py-2 text-left transition-colors ${
                                focusedIdx === idx ? 'border-amber-500/60 bg-amber-500/5' : isHovered ? 'border-stone-700 bg-stone-900' : 'border-stone-800 bg-stone-950'
                              }`}
                              onClick={() => setFocusedIdx(idx)}
                              onMouseEnter={() => { setHoveredIdx(idx); setHoverSource('list'); }}
                              onMouseLeave={() => {
                                setHoveredIdx((current) => (current === idx ? null : current));
                                setHoverSource((current) => (current === 'list' ? null : current));
                              }}
                              onKeyDown={(event) => {
                                if (event.key === 'Enter' || event.key === ' ') {
                                  event.preventDefault();
                                  setFocusedIdx(idx);
                                }
                              }}
                              tabIndex={0}
                            >
                              <div className="flex h-9 w-9 shrink-0 items-center justify-center overflow-hidden rounded-lg border border-stone-800/50 bg-stone-900">
                                <canvas
                                  ref={(canvas) => {
                                    cropCanvasRefs.current[idx] = canvas;
                                  }}
                                  width={destinationWidth}
                                  height={destinationHeight}
                                  className="block rounded bg-stone-800"
                                />
                              </div>
                              <span className={`flex h-5 w-5 shrink-0 items-center justify-center rounded text-[10px] font-bold ${indexBadgeClasses}`}>
                                {idx}
                              </span>
                              <span className="flex-1 truncate text-sm text-stone-100">{displayClass}</span>
                              {hasAnnotation && <CheckCircle2 size={12} className="text-green-400 shrink-0" />}
                              <div className="flex shrink-0 flex-col items-end gap-0.5">
                                <span className={`rounded px-1.5 py-0.5 text-[10px] font-semibold leading-none ${badgeClasses}`}>
                                  {(element.confidence * 100).toFixed(1)}%
                                </span>
                              </div>
                            </div>
                          </div>
                        );
                      })
                    )}
                  </div>
                </div>
                  )}
                </section>
              </div>
            </>
          ) : (
            <div className="flex min-h-[520px] flex-col items-center justify-center rounded-[28px] border border-dashed border-stone-700 bg-stone-900/60 px-6 text-center">
              <div className="flex h-16 w-16 items-center justify-center rounded-2xl border border-amber-400/20 bg-amber-400/10 text-amber-300">
                <Info size={28} />
              </div>
              <h2 className="mt-5 text-xl font-semibold text-stone-100">{t.noAnalysisSelected}</h2>
              <p className="mt-2 text-sm text-stone-400">
                {t.noAnalysisDetails}
              </p>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
