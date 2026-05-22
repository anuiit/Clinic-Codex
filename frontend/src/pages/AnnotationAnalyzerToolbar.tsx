import { MousePointer2, PenTool, Tags, Undo2 } from "lucide-react";
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
};

export function AnnotationAnalyzerToolbar({
  drawMode,
  canUndo,
  showLabelNames,
  labels,
  onToggleDrawMode,
  onUndo,
  onToggleLabelNames,
}: AnnotationAnalyzerToolbarProps) {
  return (
    <div data-testid="annotation-analyzer-toolbar">
      <AnalyzerToolbar>
        <AnalyzerToolbarButton
          type="button"
          onClick={onToggleDrawMode}
          active={drawMode}
        >
          {drawMode ? <PenTool size={16} /> : <MousePointer2 size={16} />}
          {drawMode ? labels.drawMode : labels.selectMode}
        </AnalyzerToolbarButton>
        <AnalyzerToolbarButton
          type="button"
          onClick={onUndo}
          disabled={!canUndo}
          aria-label={labels.undoBbox}
          title={`${labels.undoBbox} (Ctrl+Z)`}
        >
          <Undo2 size={16} />
          {labels.undoBbox}
        </AnalyzerToolbarButton>
        <AnalyzerToolbarButton
          type="button"
          onClick={onToggleLabelNames}
          active={showLabelNames}
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
        </AnalyzerToolbarButton>
      </AnalyzerToolbar>
    </div>
  );
}

export default AnnotationAnalyzerToolbar;
