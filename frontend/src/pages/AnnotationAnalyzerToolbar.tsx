import {
  Maximize2,
  MousePointer2,
  PenTool,
  Tags,
  Undo2,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import { AnalyzerToolbar, AnalyzerToolbarButton } from "../components/AnalyzerToolbar";
import { appText } from "../i18n/text";

type AnnotationLabels = typeof appText.annotation;

type AnnotationAnalyzerToolbarProps = {
  drawMode: boolean;
  canUndo: boolean;
  showLabelNames: boolean;
  labels: AnnotationLabels;
  onToggleDrawMode: () => void;
  onUndo: () => void;
  onToggleLabelNames: () => void;
  onZoomOut?: () => void;
  onFitToView?: () => void;
  onZoomIn?: () => void;
  zoomLabel?: string;
};

export function AnnotationAnalyzerToolbar({
  drawMode,
  canUndo,
  showLabelNames,
  labels,
  onToggleDrawMode,
  onUndo,
  onToggleLabelNames,
  onZoomOut,
  onFitToView,
  onZoomIn,
  zoomLabel,
}: AnnotationAnalyzerToolbarProps) {
  const hasZoomControls = Boolean(onZoomOut && onFitToView && onZoomIn);

  return (
    <div data-testid="annotation-analyzer-toolbar">
      <AnalyzerToolbar className="annotation-analyzer-toolbar__dock rounded-3xl">
        <AnalyzerToolbarButton
          type="button"
          onClick={onToggleDrawMode}
          active={drawMode}
          aria-pressed={drawMode}
        >
          {drawMode ? (
            <PenTool size={16} aria-hidden="true" />
          ) : (
            <MousePointer2 size={16} aria-hidden="true" />
          )}
          {labels.drawBbox}
        </AnalyzerToolbarButton>
        <AnalyzerToolbarButton
          type="button"
          onClick={onUndo}
          disabled={!canUndo}
          aria-label={labels.undoBbox}
          title={labels.undoBboxTitle}
        >
          <Undo2 size={16} aria-hidden="true" />
          {labels.undoBbox}
        </AnalyzerToolbarButton>
        <AnalyzerToolbarButton
          type="button"
          onClick={onToggleLabelNames}
          active={showLabelNames}
          aria-pressed={showLabelNames}
          aria-label={labels.labelsToggle}
        >
          {showLabelNames ? (
            <Tags size={16} aria-hidden="true" />
          ) : (
            <span className="text-xs font-black tabular-nums">N°</span>
          )}
          {labels.labelsToggle}
        </AnalyzerToolbarButton>
        {hasZoomControls && (
          <>
            <span
              aria-hidden="true"
              className="analyzer-toolbar__separator mx-1 h-6 w-px"
            />
            <AnalyzerToolbarButton
              type="button"
              onClick={onZoomOut}
              aria-label={labels.zoomOut}
              title={labels.zoomOut}
              className="justify-center px-2.5"
            >
              <ZoomOut size={16} aria-hidden="true" />
            </AnalyzerToolbarButton>
            <AnalyzerToolbarButton
              type="button"
              onClick={onFitToView}
              aria-label={labels.fitToView}
              title={labels.fitToView}
              className="justify-center px-2.5"
            >
              <Maximize2 size={16} aria-hidden="true" />
            </AnalyzerToolbarButton>
            <AnalyzerToolbarButton
              type="button"
              onClick={onZoomIn}
              aria-label={labels.zoomIn}
              title={labels.zoomIn}
              className="justify-center px-2.5"
            >
              <ZoomIn size={16} aria-hidden="true" />
            </AnalyzerToolbarButton>
            {zoomLabel && (
              <span className="annotation-analyzer-toolbar__zoom-label ui-text-meta min-w-12 px-2 text-center font-semibold tabular-nums">
                {zoomLabel}
              </span>
            )}
          </>
        )}
      </AnalyzerToolbar>
    </div>
  );
}

export default AnnotationAnalyzerToolbar;
