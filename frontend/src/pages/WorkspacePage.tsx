import { useNavigate } from "react-router-dom";
import { Edit3 } from "lucide-react";
import { appText } from "../i18n/text";
import { ImageBBoxStage } from "../components/ImageBBoxStage";
import WorkspaceHistoryPanel from "../components/WorkspaceHistoryPanel";
import type { ThemeMode } from "../components/ThemeToggle";
import WorkspaceDetectedPanel from "./workspace/WorkspaceDetectedPanel";
import WorkspaceEmptyState from "./workspace/WorkspaceEmptyState";
import WorkspaceHeader from "./workspace/WorkspaceHeader";
import WorkspaceOverlayToolbar from "./workspace/WorkspaceOverlayToolbar";
import WorkspaceUploadModal from "./workspace/WorkspaceUploadModal";
import { useWorkspaceHistory } from "./workspace/useWorkspaceHistory";
import { useWorkspaceUpload } from "./workspace/useWorkspaceUpload";
import { useWorkspaceViewport } from "./workspace/useWorkspaceViewport";
import {
  formatWorkspaceBboxLabel,
  hasWorkspaceSubmittedAnnotation,
} from "./workspace/workspaceViewUtils";

type WorkspacePageProps = {
  themeMode?: ThemeMode;
  onToggleTheme?: () => void;
};

export default function WorkspacePage({
  themeMode = "dark",
  onToggleTheme = () => undefined,
}: WorkspacePageProps = {}) {
  const navigate = useNavigate();
  const t = appText.workspace;
  const history = useWorkspaceHistory();
  const {
    imageRef,
    setCropCanvasRef,
    setDetailCanvasRef,
    containerRef,
    transformSize,
    hoveredIdx,
    hoverSource,
    focusedIdx,
    zoom,
    panOffset,
    isPanning,
    overlayMode,
    showLabelNames,
    trustState,
    trustData,
    contextLoading,
    setHoveredIdx,
    setHoverSource,
    setFocusedIdx,
    setOverlayMode,
    setShowLabelNames,
    handleWorkspaceImageLoad,
    zoomIn,
    zoomOut,
    resetWorkspaceView,
    clearWorkspaceSelection,
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
  const workspaceHeaderMeta =
    history.currentRecord && history.stats ? (
      <div
        className="ui-text-meta flex min-w-0 flex-wrap items-center gap-1.5"
        data-testid="workspace-image-header-meta"
      >
        <span className="truncate" title={history.currentRecord.imageName}>
          {history.currentRecord.imageName}
        </span>
        <span aria-hidden="true" className="text-[var(--divider)]">
          ·
        </span>
        <span>{history.stats.imageSizeLabel}</span>
        <span aria-hidden="true" className="text-[var(--divider)]">
          ·
        </span>
        <span>
          Annotés / rejetés {history.stats.annotatedCount}/{history.stats.total} ·{" "}
          {history.stats.rejectedCount}
        </span>
        <span aria-hidden="true" className="text-[var(--divider)]">
          ·
        </span>
        <span
          className="min-w-0 truncate"
          title={history.stats.topClasses.join(", ")}
        >
          Classes {history.stats.topClasses.join(", ") || history.stats.topClass}
        </span>
      </div>
    ) : null;

  return (
    <div
      className={`workspace-page flex h-full min-h-0 flex-col gap-3 overflow-hidden rounded-2xl transition-colors ${upload.dragging ? "ring-2 ring-[color:var(--border-strong)] ring-offset-2 ring-offset-[var(--app-bg)]" : ""}`}
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
        themeMode={themeMode}
        onToggleTheme={onToggleTheme}
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
              <ImageBBoxStage
                tone="workspace"
                mode="inspect"
                className="h-full"
                imageDataUrl={history.currentRecord.imageDataUrl}
                imageName={history.currentRecord.imageName}
                imageSize={history.currentRecord.result.image_size}
                boxes={history.currentRecord.result.elements.map((element, idx) => ({
                  id: idx,
                  bbox: element.bbox,
                  label: element.class_name,
                  confidence: element.confidence,
                  rejected: element.rejected,
                  status:
                    history.currentRecord &&
                    hasWorkspaceSubmittedAnnotation(history.currentRecord, idx)
                      ? "validated"
                      : "draft",
                }))}
                selectedId={focusedIdx}
                showLabelNames={showLabelNames}
                overlayMode={overlayMode}
                viewport={{ zoom, panOffset, isPanning }}
                transformSize={transformSize}
                imageFit="fill"
                boxStateById={Object.fromEntries(
                  history.currentRecord.result.elements.map((_, idx) => [
                    idx,
                    {
                      imageHovered:
                        idx === hoveredIdx && hoverSource === "image",
                      listHovered: idx === hoveredIdx && hoverSource === "list",
                      submitted: history.currentRecord
                        ? hasWorkspaceSubmittedAnnotation(history.currentRecord, idx)
                        : false,
                    },
                  ]),
                )}
                renderLabel={(box) =>
                  formatWorkspaceBboxLabel(
                    Number(box.id),
                    box.label ?? "",
                    showLabelNames,
                  )
                }
                title={
                  <h2
                    className="ui-title-md max-w-[200px] truncate text-lg sm:max-w-[300px]"
                    title={history.currentRecord.imageName}
                  >
                    {history.currentRecord.imageName}
                  </h2>
                }
                headerMeta={workspaceHeaderMeta}
                headerActions={
                  <button
                    type="button"
                    onClick={handleEditorHandoff}
                    className="ui-action-primary inline-flex items-center gap-2 rounded-xl px-3 py-2 text-sm"
                    data-testid="workspace-header-annotate-action"
                  >
                    <Edit3 size={16} /> {t.annotateRecord}
                  </button>
                }
                badges={
                  focusedIdx !== null && (
                    <span className="ui-chip ui-chip--accent rounded-md px-2 py-1 text-xs font-medium">
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
                      zoomOut: t.zoomOut,
                      fitToView: t.fitToView,
                      zoomIn: t.zoomIn,
                      deselect: t.deselect,
                    }}
                    onOverlayModeChange={setOverlayMode}
                    hasSelection={focusedIdx !== null}
                    zoomLabel={`${Math.round(zoom * 100)}%`}
                    onToggleLabelNames={() =>
                      setShowLabelNames((current) => !current)
                    }
                    onDeselect={clearWorkspaceSelection}
                    onZoomOut={zoomOut}
                    onFitToView={resetWorkspaceView}
                    onZoomIn={zoomIn}
                  />
                }
                toolbarPlacement="bottom-center"
                stageClassName={`workspace-stage ${zoom > 1 ? (isPanning ? "cursor-grabbing" : "cursor-grab") : ""}`}
                stageProps={{
                  onPointerDown: startWorkspacePan,
                  onPointerMove: moveWorkspacePan,
                  onPointerUp: stopWorkspacePan,
                  onPointerCancel: stopWorkspacePan,
                  onWheel: handleWorkspaceStageWheel,
                }}
                stageRef={containerRef}
                imageProps={{ ref: imageRef, onLoad: handleWorkspaceImageLoad }}
                svgProps={{
                  onPointerDown: handleWorkspaceOverlayPointerDown,
                  onPointerMove: handleWorkspaceOverlayPointerMove,
                  onPointerLeave: () => {
                    setHoveredIdx(null);
                    setHoverSource(null);
                  },
                }}
                testIds={{
                  header: "workspace-image-header",
                  stage: "workspace-stage",
                  overlay: "workspace-overlay",
                }}
              />

              <WorkspaceDetectedPanel
                record={history.currentRecord}
                focusedIdx={focusedIdx}
                hoveredIdx={hoveredIdx}
                stats={history.stats}
                trustState={trustState}
                trustData={trustData}
                contextLoading={contextLoading}
                setCropCanvasRef={setCropCanvasRef}
                setDetailCanvasRef={setDetailCanvasRef}
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
