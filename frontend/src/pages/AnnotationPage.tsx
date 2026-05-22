import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom";
import { ArrowLeft, Loader2, Maximize2, ZoomIn, ZoomOut } from "lucide-react";
import { MainImagePanel } from "../components/MainImagePanel";
import { appText } from "../i18n/text";
import { AnnotationAnalyzerToolbar } from "./AnnotationAnalyzerToolbar";
import { AnnotationElementList } from "./AnnotationElementList";
import { AnnotationPageChrome } from "./AnnotationPageChrome";
import { AnnotationOverlay } from "./annotation/AnnotationOverlay";
import { AnnotationSelectedInspector } from "./annotation/AnnotationSelectedInspector";
import { AnnotationToast } from "./annotation/AnnotationToast";
import { useAnnotationElementModel } from "./annotation/useAnnotationElementModel";
import { useAnnotationRecord } from "./annotation/useAnnotationRecord";
import { useAnnotationSubmission } from "./annotation/useAnnotationSubmission";
import { useAnnotationViewport } from "./annotation/useAnnotationViewport";

export default function AnnotationPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const t = appText.annotation;

  const elementParam = searchParams.get("element");
  const initialFocusedIdx =
    elementParam !== null && !Number.isNaN(Number(elementParam))
      ? Number(elementParam)
      : null;

  const annotation = useAnnotationRecord(id, initialFocusedIdx);
  const model = useAnnotationElementModel({
    elements: annotation.elements,
    setElements: annotation.setElements,
    annotationStatus: annotation.annotationStatus,
    setAnnotationStatus: annotation.setAnnotationStatus,
    classes: annotation.classes,
    setCustomClasses: annotation.setCustomClasses,
    focusedIdx: annotation.focusedIdx,
    cardRefs: annotation.cardRefs,
  });
  const {
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
  } = useAnnotationViewport({
    record: annotation.record,
    elements: annotation.elements,
    setElements: annotation.setElements,
    annotationStatus: annotation.annotationStatus,
    setAnnotationStatus: annotation.setAnnotationStatus,
    focusedIdx: annotation.focusedIdx,
    setFocusedIdx: annotation.setFocusedIdx,
    setHoveredIdx: annotation.setHoveredIdx,
    setNamingFocusToken: model.setNamingFocusToken,
    loading: annotation.loading,
  });
  const submission = useAnnotationSubmission({
    id,
    record: annotation.record,
    elements: annotation.elements,
    annotationStatus: annotation.annotationStatus,
    navigate,
    labels: {
      submitBlockedUnnamed: t.submitBlockedUnnamed,
      submitBlockedNone: t.submitBlockedNone,
    },
  });

  if (!annotation.record && !annotation.loading) {
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

  if (annotation.loading) {
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
        saving={submission.saving}
        sending={submission.sending}
        onSubmitNamed={model.submitNamedElements}
        onSave={submission.handleSave}
        onSendSubmittedForReview={submission.handleSendSubmittedForReview}
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
                applyZoom(1);
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
              src={annotation.record!.imageDataUrl}
              alt={annotation.record!.imageName}
              draggable={false}
              onLoad={updateStageSize}
              className="block h-full w-full object-fill pointer-events-none"
            />
          }
          overlay={
            <AnnotationOverlay
              imageSize={annotation.record!.result.image_size}
              elements={annotation.elements}
              annotationStatus={annotation.annotationStatus}
              focusedIdx={annotation.focusedIdx}
              hoveredIdx={annotation.hoveredIdx}
              listHoveredIdx={annotation.listHoveredIdx}
              drawMode={drawMode}
              dragState={dragState}
              tempBbox={tempBbox}
              showLabelNames={showLabelNames}
              unnamedLabel={t.unnamedElement}
              onPointerDown={handleSvgPointerDown}
              onPointerMove={handleSvgPointerMove}
              onPointerUp={handleSvgPointerUp}
            />
          }
        />

        <aside
          className="annotation-rail annotation-inspector flex shrink-0 flex-col rounded-2xl p-4"
          aria-label="Inspecteur d’annotation"
        >
          <AnnotationSelectedInspector
            focusedElement={model.focusedElement}
            focusedIdx={annotation.focusedIdx}
            focusedDisplayName={model.focusedDisplayName ?? t.unnamedElement}
            focusedConfidencePercent={model.focusedConfidencePercent}
            focusedIsSubmitted={model.focusedIsSubmitted}
            previewCanvasRef={previewCanvasRef}
            classes={annotation.classes}
            customClasses={annotation.customClasses}
            namingFocusToken={model.namingFocusToken}
            labels={t}
            onCommitElementName={model.commitElementName}
            onSetElementValidation={(idx, submitted) =>
              model.setElementValidation(idx, submitted ? "validated" : "draft")
            }
            onRemoveElement={removeElement}
          />

          <AnnotationElementList
            displayedElements={model.displayedElements}
            annotationStatus={annotation.annotationStatus}
            focusedIdx={annotation.focusedIdx}
            elementsCount={annotation.elements.length}
            submittedCount={model.submittedCount}
            listQuery={model.listQuery}
            statusFilter={model.statusFilter}
            sortMode={model.sortMode}
            labels={t}
            cardRefs={annotation.cardRefs}
            setFocusedIdx={annotation.setFocusedIdx}
            onListQueryChange={model.setListQuery}
            onStatusFilterChange={model.setStatusFilter}
            onSortModeChange={model.setSortMode}
            setListHoveredIdx={annotation.setListHoveredIdx}
          />
        </aside>
      </div>
      {submission.toast && <AnnotationToast toast={submission.toast} />}
    </div>
  );
}
