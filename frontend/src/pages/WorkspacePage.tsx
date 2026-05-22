import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type DragEvent,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
  type WheelEvent as ReactWheelEvent,
} from "react";
import { useLocation, useNavigate, useSearchParams } from "react-router-dom";
import { Maximize2, ZoomIn, ZoomOut } from 'lucide-react';
import { segmentGlyph, getTrust } from '../services/api';
import { appText } from '../i18n/text';
import { deleteAnalysis, getHistory, saveAnalysis } from '../services/storage';
import MainImagePanel from '../components/MainImagePanel';
import WorkspaceHistoryPanel from '../components/WorkspaceHistoryPanel';
import WorkspaceDetectedPanel from './workspace/WorkspaceDetectedPanel';
import WorkspaceEmptyState from './workspace/WorkspaceEmptyState';
import WorkspaceHeader from './workspace/WorkspaceHeader';
import WorkspaceOverlay from './workspace/WorkspaceOverlay';
import WorkspaceOverlayToolbar from './workspace/WorkspaceOverlayToolbar';
import WorkspaceUploadModal from './workspace/WorkspaceUploadModal';
import type { WorkspaceHoverSource, WorkspaceOverlayMode } from './workspace/workspaceViewUtils';
import type { AnalysisRecord, TrustResult } from '../types';
import { clientToImage } from '../utils/imageCoords';
import {
  nextZoomFromWheel,
  shouldConsumeStageWheel,
} from "../utils/imageStageZoom";
import { hitTestBBoxes } from "../utils/segmentationBoxes";

type OverlayMode = WorkspaceOverlayMode;
type HoverSource = WorkspaceHoverSource;

const WORKSPACE_WHEEL_ZOOM_SENSITIVITY = 0.0015;

function formatWorkspaceBboxLabel(
  idx: number,
  className: string,
  showName: boolean,
) {
  if (!showName) return `#${idx}`;
  const clippedName =
    className.length > 18 ? `${className.slice(0, 17)}…` : className;
  return clippedName;
}

function getCropPreviewSize(
  bbox: [number, number, number, number],
  maxSize: number,
) {
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

function resolveCurrentRecord(
  records: AnalysisRecord[],
  preferredId?: string | null,
) {
  if (preferredId) {
    return (
      records.find((record) => record.id === preferredId) ?? records[0] ?? null
    );
  }

  return records[0] ?? null;
}

export default function WorkspacePage() {
  const navigate = useNavigate();
  const location = useLocation();
  const [searchParams] = useSearchParams();
  const initialPreferredId = searchParams.get("analysis");

  const [records, setRecords] = useState<AnalysisRecord[]>(() => getHistory());
  const [currentRecord, setCurrentRecord] = useState<AnalysisRecord | null>(
    () => resolveCurrentRecord(getHistory(), initialPreferredId),
  );
  const [filter, setFilter] = useState("");

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
  const [historyOpen, setHistoryOpen] = useState(
    () => !resolveCurrentRecord(getHistory(), initialPreferredId),
  );
  const [overlayMode, setOverlayMode] = useState<OverlayMode>("all");
  const [showLabelNames, setShowLabelNames] = useState(false);
  const [trustData, setTrustData] = useState<TrustResult | null>(null);
  const [contextLoading, setContextLoading] = useState(false);

  const inputRef = useRef<HTMLInputElement>(null);
  const currentFileRef = useRef<File | null>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const workspacePanStartRef = useRef<{
    clientX: number;
    clientY: number;
    offset: { x: number; y: number };
  } | null>(null);
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
    setOverlayMode("all");
  }, [resetWorkspaceView]);

  const selectRecord = useCallback(
    (record: AnalysisRecord | null) => {
      resetInspectionState();
      setCurrentRecord(record);
    },
    [resetInspectionState],
  );

  const syncRecords = useCallback(
    (preferredId?: string | null) => {
      const nextRecords = getHistory();
      setRecords(nextRecords);
      selectRecord(resolveCurrentRecord(nextRecords, preferredId));
    },
    [selectRecord],
  );

  useEffect(() => {
    if (location.pathname === "/" && searchParams.get("analysis")) {
      navigate("/", { replace: true });
    }
  }, [location.pathname, navigate, searchParams]);

  useEffect(() => {
    if (!currentRecord || !imageRef.current) {
      return;
    }

    const image = imageRef.current;

    const drawCrop = (
      canvas: HTMLCanvasElement | null,
      element: AnalysisRecord["result"]["elements"][number],
    ) => {
      if (!canvas) {
        return;
      }

      const cropCtx = canvas.getContext("2d");
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

    image.addEventListener("load", drawAllCanvases);
    window.addEventListener("resize", drawAllCanvases);

    return () => {
      image.removeEventListener("load", drawAllCanvases);
      window.removeEventListener("resize", drawAllCanvases);
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
      const names = record.result.elements.map((element) =>
        element.class_name.toLowerCase(),
      );
      const annotated = Object.values(record.annotations ?? {}).map((value) =>
        value.toLowerCase(),
      );
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
      const finalClass =
        (currentRecord.annotations ?? {})[idx] ?? element.class_name;
      classCounts[finalClass] = (classCounts[finalClass] || 0) + 1;
    });

    let topClass = "None";
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
      inputRef.current.value = "";
    }
  }, []);

  const onDrop = useCallback(
    (event: DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      setDragging(false);
      const nextFile = event.dataTransfer.files[0];
      if (nextFile) {
        handleFile(nextFile);
      }
    },
    [handleFile],
  );

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
    if (
      target instanceof Element &&
      target.closest('[data-overlay-region="true"]')
    ) {
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
      x:
        workspacePanStartRef.current.offset.x +
        event.clientX -
        workspacePanStartRef.current.clientX,
      y:
        workspacePanStartRef.current.offset.y +
        event.clientY -
        workspacePanStartRef.current.clientY,
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

  const handleWorkspaceStageWheel = (
    event: ReactWheelEvent<HTMLDivElement>,
  ) => {
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

  const getWorkspaceOverlayHit = (
    svg: SVGSVGElement,
    clientX: number,
    clientY: number,
  ): number | null => {
    if (!currentRecord) {
      return null;
    }

    const [imgW, imgH] = currentRecord.result.image_size;
    const point = clientToImage(svg, clientX, clientY, {
      width: imgW,
      height: imgH,
    });
    const visibleElements = currentRecord.result.elements
      .map((element, idx) => ({ idx, bbox: element.bbox }))
      .filter(
        ({ idx }) =>
          overlayMode === "all" || focusedIdx === null || focusedIdx === idx,
      );
    const hitIdx = hitTestBBoxes(
      point,
      visibleElements.map(({ bbox }) => bbox),
    );

    return hitIdx === null ? null : visibleElements[hitIdx].idx;
  };

  const handleWorkspaceOverlayPointerDown = (
    event: ReactPointerEvent<SVGSVGElement>,
  ) => {
    if (event.button !== 0) {
      return;
    }

    const hitIdx = getWorkspaceOverlayHit(
      event.currentTarget,
      event.clientX,
      event.clientY,
    );
    if (hitIdx === null) {
      return;
    }

    event.stopPropagation();
    setFocusedIdx(hitIdx);
    setHoveredIdx(hitIdx);
    setHoverSource(hitIdx === null ? null : "image");
  };

  const handleWorkspaceOverlayPointerMove = (
    event: ReactPointerEvent<SVGSVGElement>,
  ) => {
    const hitIdx = getWorkspaceOverlayHit(
      event.currentTarget,
      event.clientX,
      event.clientY,
    );
    setHoveredIdx(hitIdx);
    setHoverSource(hitIdx === null ? null : "image");
    event.currentTarget.style.cursor =
      hitIdx === null ? (zoom > 1 ? "grab" : "default") : "pointer";
  };

  return (
    <div
      className={`workspace-page flex h-full min-h-0 flex-col gap-3 overflow-hidden rounded-2xl transition-colors ${dragging ? "ring-2 ring-amber-400/60 ring-offset-2 ring-offset-stone-950" : ""}`}
      onDragEnter={(event) => {
        event.preventDefault();
        setDragging(true);
      }}
      onDragOver={(event) => {
        event.preventDefault();
        setDragging(true);
      }}
      onDragLeave={(event) => {
        if (
          event.relatedTarget instanceof Node &&
          event.currentTarget.contains(event.relatedTarget)
        ) {
          return;
        }
        setDragging(false);
      }}
      onDrop={onDrop}
    >
      <WorkspaceHeader
        title={t.appTitle}
        error={error}
        dragging={dragging}
        preview={preview}
        fileName={file?.name ?? null}
        loading={loading}
        uploadPrompt={t.uploadPrompt}
        previewAlt={t.previewAlt}
        analyzeLabel={t.analyze}
        analyzingLabel={t.analyzing}
        onUploadClick={() => inputRef.current?.click()}
        onAnalyze={analyze}
        onFileChange={handleFile}
        inputRef={inputRef}
      />

      <WorkspaceUploadModal
        open={Boolean(preview && file)}
        preview={preview}
        fileName={file?.name ?? null}
        loading={loading}
        title={t.uploadModalTitle}
        description={t.uploadModalDescription}
        previewAlt={t.previewAlt}
        cancelLabel={t.cancel}
        analyzeLabel={t.analyze}
        analyzingLabel={t.analyzing}
        onClose={clearPendingFile}
        onAnalyze={analyze}
      />

      <div className={`grid min-h-0 flex-1 gap-3 transition-[grid-template-columns] duration-300 ease-out ${historyOpen ? 'xl:grid-cols-[300px_minmax(0,1fr)]' : 'xl:grid-cols-[56px_minmax(0,1fr)]'}`}>
        <WorkspaceHistoryPanel
          records={records}
          filteredRecords={filteredRecords}
          currentRecordId={currentRecord?.id ?? null}
          filter={filter}
          historyOpen={historyOpen}
          labels={{
            expandHistory: t.expandHistory,
            collapseHistory: t.collapseHistory,
            filterPlaceholder: t.filterPlaceholder,
            noAnalyses: t.noAnalyses,
            noFilterMatch: t.noFilterMatch,
            elementsSuffix: t.elementsSuffix,
            rejectedSuffix: t.rejectedSuffix,
            historyTitle: 'History',
            totalSuffix: 'total',
            deleteLabel: t.deleteLabel,
          }}
          onFilterChange={setFilter}
          onToggleHistoryOpen={setHistoryOpen}
          onSelectRecord={selectRecord}
          onRemoveRecord={removeRecord}
        />

        <section className="min-h-0 overflow-hidden">
          {currentRecord ? (
            <>
              <div
                className="grid h-full min-h-0 gap-3 xl:grid-cols-[minmax(0,1.45fr)_minmax(280px,0.55fr)] 2xl:grid-cols-[minmax(0,1.55fr)_minmax(320px,0.45fr)]"
                data-testid="workspace-content-grid"
              >
                <MainImagePanel
                  tone="workspace"
                  className="h-full"
                  title={
                    <h2
                      className="truncate text-lg font-semibold text-stone-100 max-w-[200px] sm:max-w-[300px]"
                      title={currentRecord.imageName}
                    >
                      {currentRecord.imageName}
                    </h2>
                  }
                  badges={
                    focusedIdx !== null && (
                      <span className="rounded-md border border-amber-500/20 bg-amber-500/10 px-2 py-1 text-xs font-medium text-amber-500">
                        {t.focusLabel}: {focusedIdx}
                      </span>
                    )
                  }
                  toolbar={
                    <AnalyzerToolbar>
                      <div className="inline-flex rounded-lg border border-stone-700 bg-stone-950 p-1">
                        {(["all", "focused", "hidden"] as OverlayMode[]).map(
                          (mode) => {
                            const isActive = overlayMode === mode;
                            return (
                              <AnalyzerToolbarButton
                                key={mode}
                                type="button"
                                onClick={() => setOverlayMode(mode)}
                                active={isActive}
                                className="rounded-md px-3 py-1.5 capitalize"
                              >
                                {mode === "all"
                                  ? t.overlayAll
                                  : mode === "focused"
                                    ? t.overlayFocused
                                    : t.overlayHidden}
                              </AnalyzerToolbarButton>
                            );
                          },
                        )}
                      </div>
                      <AnalyzerToolbarButton
                        type="button"
                        onClick={() => setShowLabelNames((current) => !current)}
                        active={showLabelNames}
                        aria-pressed={showLabelNames}
                        aria-label={
                          showLabelNames
                            ? "Masquer les noms des libellés"
                            : "Afficher les noms des libellés"
                        }
                        title={
                          showLabelNames
                            ? "Masquer les noms des libellés"
                            : "Afficher les noms des libellés"
                        }
                        className="w-10 justify-center px-0"
                      >
                        {showLabelNames ? (
                          <Tags size={16} aria-hidden="true" />
                        ) : (
                          <span
                            aria-hidden="true"
                            className="text-base font-black leading-none"
                          >
                            #
                          </span>
                        )}
                      </AnalyzerToolbarButton>
                    </AnalyzerToolbar>
                  }
                  stageClassName={`workspace-stage ${zoom > 1 ? (isPanning ? "cursor-grabbing" : "cursor-grab") : ""}`}
                  stageProps={{
                    onPointerDown: startWorkspacePan,
                    onPointerMove: moveWorkspacePan,
                    onPointerUp: stopWorkspacePan,
                    onPointerCancel: stopWorkspacePan,
                    onWheel: handleWorkspaceStageWheel,
                  }}
                  transformStyle={{
                    transform: `translate(${panOffset.x}px, ${panOffset.y}px) scale(${zoom})`,
                    transformOrigin: "center center",
                    transition: isPanning ? "none" : "transform 0.1s ease",
                    willChange: "transform",
                  }}
                  image={
                    <img
                      ref={imageRef}
                      src={currentRecord.imageDataUrl}
                      alt={currentRecord.imageName}
                      draggable={false}
                      className="block max-h-full max-w-full rounded-lg object-contain"
                    />
                  }
                  overlay={
                    currentRecord &&
                    overlayMode !== "hidden" && (
                        <WorkspaceOverlay
                          imageSize={currentRecord.result.image_size}
                          elements={currentRecord.result.elements}
                          annotations={currentRecord.annotations}
                          focusedIdx={focusedIdx}
                          hoveredIdx={hoveredIdx}
                          hoverSource={hoverSource}
                          overlayMode={overlayMode}
                          showLabelNames={showLabelNames}
                          zoom={zoom}
                          onPointerDown={handleWorkspaceOverlayPointerDown}
                          onPointerMove={handleWorkspaceOverlayPointerMove}
                          onPointerLeave={() => {
                            setHoveredIdx(null);
                            setHoverSource(null);
                          }}
                          formatLabel={formatWorkspaceBboxLabel}
                          testId="workspace-overlay"
                        />
                    )
                  }
                  controls={[
                    {
                      id: "zoom-in",
                      label: t.zoomIn,
                      title: t.zoomIn,
                      onClick: () => setZoom((z) => Math.min(4, z + 0.25)),
                      icon: <ZoomIn size={16} />,
                    },
                    {
                      id: "fit-to-view",
                      label: t.fitToView,
                      title: t.fitToView,
                      onClick: resetWorkspaceView,
                      icon: <Maximize2 size={16} />,
                    },
                    {
                      id: "zoom-out",
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
                  testIds={{ stage: "workspace-stage" }}
                />

                <WorkspaceDetectedPanel
                  currentRecord={currentRecord}
                  focusedIdx={focusedIdx}
                  trustData={trustData}
                  contextLoading={contextLoading}
                  showLabelNames={showLabelNames}
                  onBack={() => setFocusedIdx(null)}
                  onHandoff={handleEditorHandoff}
                  onSelectRegion={setFocusedIdx}
                  onHoverRegion={(idx) => {
                    setHoveredIdx(idx);
                    setHoverSource("list");
                  }}
                  onLeaveRegion={() => {
                    setHoveredIdx(null);
                    setHoverSource(null);
                  }}
                  formatCropSize={getCropPreviewSize}
                  labels={{
                    backToRegions: t.backToRegions,
                    segmentPreview: t.segmentPreview,
                    trustSummary: t.trustSummary,
                    recalculatedPrediction: t.recalculatedPrediction,
                    initialProposal: t.initialProposal,
                    ambiguousPrediction: t.ambiguousPrediction,
                    ambiguousDetails: t.ambiguousDetails,
                    alternativesExist: t.alternativesExist,
                    lowConfidenceFlag: t.lowConfidenceFlag,
                    thresholdDetails: t.thresholdDetails,
                    rank: t.rank,
                    margin: t.margin,
                    topPredictions: t.topPredictions,
                    proposalPanel: t.proposalPanel,
                    detectedElements: t.detectedElements,
                    annotateRecord: t.annotateRecord,
                    allRejected: t.allRejected,
                    goToAnnotation: t.goToAnnotation,
                    noElements: t.noElements,
                    annotateRegion: t.annotateRegion,
                    correctElement: t.correctElement,
                  }}
                />
              </div>
            </>
          ) : (
            <WorkspaceEmptyState
              title={t.noAnalysisSelected}
              description={t.noAnalysisDetails}
            />
          )}
        </section>
      </div>
    </div>
  );
}
