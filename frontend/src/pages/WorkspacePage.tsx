import { useNavigate } from "react-router-dom";
import { Maximize2, ZoomIn, ZoomOut } from "lucide-react";
import { appText } from "../i18n/text";
import MainImagePanel from "../components/MainImagePanel";
import WorkspaceHistoryPanel from "../components/WorkspaceHistoryPanel";
import WorkspaceDetectedPanel from "./workspace/WorkspaceDetectedPanel";
import WorkspaceEmptyState from "./workspace/WorkspaceEmptyState";
import WorkspaceHeader from "./workspace/WorkspaceHeader";
import WorkspaceOverlay from "./workspace/WorkspaceOverlay";
import WorkspaceOverlayToolbar from "./workspace/WorkspaceOverlayToolbar";
import WorkspaceUploadModal from "./workspace/WorkspaceUploadModal";
import { useWorkspaceHistory } from "./workspace/useWorkspaceHistory";
import { useWorkspaceUpload } from "./workspace/useWorkspaceUpload";
import { useWorkspaceViewport } from "./workspace/useWorkspaceViewport";

export default function WorkspacePage() {
  const navigate = useNavigate();
  const t = appText.workspace;
  const history = useWorkspaceHistory();
  const {
    imageRef,
    cropCanvasRefs,
    detailCanvasRef,
    hoveredIdx,
    hoverSource,
    focusedIdx,
    zoom,
    panOffset,
    isPanning,
    overlayMode,
    showLabelNames,
    trustData,
    contextLoading,
    setHoveredIdx,
    setHoverSource,
    setFocusedIdx,
    setZoom,
    setPanOffset,
    setOverlayMode,
    setShowLabelNames,
    resetWorkspaceView,
    startWorkspacePan,
    moveWorkspacePan,
    stopWorkspacePan,
    handleWorkspaceStageWheel,
    handleWorkspaceOverlayPointerDown,
    handleWorkspaceOverlayPointerMove,
    handleWorkspaceDetectedListKeyDown,
  } = useWorkspaceViewport(history.currentRecord);
  const upload = useWorkspaceUpload({
    apiErrorLabel: t.apiError,
    syncRecords: history.syncRecords,
  });

  const handleEditorHandoff = () => {
    if (!history.currentRecord) {
      return;
    }

    navigate(
      focusedIdx !== null
        ? `/annotate/${history.currentRecord.id}?element=${focusedIdx}`
        : `/annotate/${history.currentRecord.id}`,
    );
  };

  return (
    <div
      className={`workspace-page flex h-full min-h-0 flex-col gap-3 overflow-hidden rounded-2xl transition-colors ${upload.dragging ? "ring-2 ring-amber-400/60 ring-offset-2 ring-offset-stone-950" : ""}`}
      onDragEnter={(event) => {
        event.preventDefault();
        upload.setDragging(true);
      }}
      onDragOver={(event) => {
        event.preventDefault();
        upload.setDragging(true);
      }}
      onDragLeave={(event) => {
        if (
          event.relatedTarget instanceof Node &&
          event.currentTarget.contains(event.relatedTarget)
        ) {
          return;
        }
        upload.setDragging(false);
      }}
      onDrop={upload.onDrop}
    >
      <WorkspaceHeader
        inputRef={upload.inputRef}
        dragging={upload.dragging}
        preview={upload.preview}
        file={upload.file}
        loading={upload.loading}
        error={upload.error}
        labels={{
          appTitle: t.appTitle,
          previewAlt: t.previewAlt,
          uploadPrompt: t.uploadPrompt,
          analyze: t.analyze,
          analyzing: t.analyzing,
        }}
        onFileSelected={upload.handleFile}
        onAnalyze={upload.analyze}
      />

      {upload.preview && upload.file && (
        <WorkspaceUploadModal
          preview={upload.preview}
          file={upload.file}
          loading={upload.loading}
          labels={{
            uploadModalTitle: t.uploadModalTitle,
            uploadModalDescription: t.uploadModalDescription,
            previewAlt: t.previewAlt,
            cancel: t.cancel,
            analyze: t.analyze,
            analyzing: t.analyzing,
          }}
          onCancel={upload.clearPendingFile}
          onAnalyze={upload.analyze}
        />
      )}

      <div
        className={`grid min-h-0 flex-1 gap-3 transition-[grid-template-columns] duration-300 ease-out ${history.historyOpen ? "xl:grid-cols-[300px_minmax(0,1fr)]" : "xl:grid-cols-[56px_minmax(0,1fr)]"}`}
      >
        <WorkspaceHistoryPanel
          records={history.records}
          filteredRecords={history.filteredRecords}
          currentRecordId={history.currentRecord?.id ?? null}
          filter={history.filter}
          historyOpen={history.historyOpen}
          labels={{
            expandHistory: t.expandHistory,
            collapseHistory: t.collapseHistory,
            filterPlaceholder: t.filterPlaceholder,
            noAnalyses: history.storageLoading ? "Chargement de l’historique…" : t.noAnalyses,
            noFilterMatch: t.noFilterMatch,
            elementsSuffix: t.elementsSuffix,
            rejectedSuffix: t.rejectedSuffix,
            historyTitle: "History",
            totalSuffix: "total",
            deleteLabel: t.deleteLabel,
          }}
          onFilterChange={history.setFilter}
          onToggleHistoryOpen={history.setHistoryOpen}
          onSelectRecord={history.selectRecord}
          onRemoveRecord={history.removeRecord}
        />

        <section className="min-h-0 overflow-hidden">
          {history.currentRecord ? (
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
                    title={history.currentRecord.imageName}
                  >
                    {history.currentRecord.imageName}
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
                  <WorkspaceOverlayToolbar
                    overlayMode={overlayMode}
                    showLabelNames={showLabelNames}
                    labels={{
                      overlayAll: t.overlayAll,
                      overlayFocused: t.overlayFocused,
                      overlayHidden: t.overlayHidden,
                    }}
                    onOverlayModeChange={setOverlayMode}
                    onToggleLabelNames={() =>
                      setShowLabelNames((current) => !current)
                    }
                  />
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
                    src={history.currentRecord.imageDataUrl}
                    alt={history.currentRecord.imageName}
                    draggable={false}
                    className="block max-h-full max-w-full rounded-lg object-contain"
                  />
                }
                overlay={
                  <WorkspaceOverlay
                    record={history.currentRecord}
                    focusedIdx={focusedIdx}
                    hoveredIdx={hoveredIdx}
                    hoverSource={hoverSource}
                    overlayMode={overlayMode}
                    showLabelNames={showLabelNames}
                    onPointerDown={handleWorkspaceOverlayPointerDown}
                    onPointerMove={handleWorkspaceOverlayPointerMove}
                    onPointerLeave={() => {
                      setHoveredIdx(null);
                      setHoverSource(null);
                    }}
                  />
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
                record={history.currentRecord}
                focusedIdx={focusedIdx}
                hoveredIdx={hoveredIdx}
                stats={history.stats}
                trustData={trustData}
                contextLoading={contextLoading}
                cropCanvasRefs={cropCanvasRefs}
                detailCanvasRef={detailCanvasRef}
                onBackToRegions={() => setFocusedIdx(null)}
                onEditorHandoff={handleEditorHandoff}
                onDetectedListKeyDown={handleWorkspaceDetectedListKeyDown}
                onFocusRegion={setFocusedIdx}
                onListRegionEnter={(idx) => {
                  setHoveredIdx(idx);
                  setHoverSource("list");
                }}
                onListRegionLeave={(idx) => {
                  setHoveredIdx((current) =>
                    current === idx ? null : current,
                  );
                  setHoverSource((current) =>
                    current === "list" ? null : current,
                  );
                }}
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
          ) : (
            <WorkspaceEmptyState
              title={t.noAnalysisSelected}
              details={t.noAnalysisDetails}
            />
          )}
        </section>
      </div>
    </div>
  );
}
