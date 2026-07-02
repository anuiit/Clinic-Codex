import { Eye, EyeOff, Focus, Maximize2, Tags, XCircle, ZoomIn, ZoomOut } from "lucide-react";
import { AnalyzerToolbar, AnalyzerToolbarButton } from "../../components/AnalyzerToolbar";
import type { WorkspaceOverlayMode } from "./workspaceViewUtils";

type WorkspaceOverlayToolbarLabels = {
  overlayAll: string;
  overlayFocused: string;
  overlayHidden: string;
  zoomOut: string;
  fitToView: string;
  zoomIn: string;
  deselect: string;
};

type WorkspaceOverlayToolbarProps = {
  overlayMode: WorkspaceOverlayMode;
  showLabelNames: boolean;
  hasSelection: boolean;
  zoomLabel: string;
  labels: WorkspaceOverlayToolbarLabels;
  onOverlayModeChange: (mode: WorkspaceOverlayMode) => void;
  onToggleLabelNames: () => void;
  onDeselect: () => void;
  onZoomOut: () => void;
  onFitToView: () => void;
  onZoomIn: () => void;
};

const overlayModes: WorkspaceOverlayMode[] = ["all", "focused", "hidden"];

function getOverlayLabel(mode: WorkspaceOverlayMode, labels: WorkspaceOverlayToolbarLabels) {
  if (mode === "all") return labels.overlayAll;
  if (mode === "focused") return labels.overlayFocused;
  return labels.overlayHidden;
}

function getOverlayIcon(mode: WorkspaceOverlayMode) {
  if (mode === "all") return <Eye size={16} aria-hidden="true" />;
  if (mode === "focused") return <Focus size={16} aria-hidden="true" />;
  return <EyeOff size={16} aria-hidden="true" />;
}

export default function WorkspaceOverlayToolbar({
  overlayMode,
  showLabelNames,
  hasSelection,
  zoomLabel,
  labels,
  onOverlayModeChange,
  onToggleLabelNames,
  onDeselect,
  onZoomOut,
  onFitToView,
  onZoomIn,
}: WorkspaceOverlayToolbarProps) {
  return (
    <div data-testid="workspace-analyzer-toolbar">
      <AnalyzerToolbar className="workspace-analyzer-toolbar__dock rounded-none">
        <div className="inline-flex gap-0 rounded-none bg-transparent p-0">
          {overlayModes.map((mode) => (
            <AnalyzerToolbarButton
              key={mode}
              type="button"
              onClick={() => onOverlayModeChange(mode)}
              active={overlayMode === mode}
              aria-pressed={overlayMode === mode}
              aria-label={getOverlayLabel(mode, labels)}
              title={getOverlayLabel(mode, labels)}
              className="workspace-analyzer-toolbar__segment min-w-24 justify-center capitalize"
            >
              <span className="analyzer-toolbar__icon-slot">{getOverlayIcon(mode)}</span>
              <span className="analyzer-toolbar__label-slot">{getOverlayLabel(mode, labels)}</span>
            </AnalyzerToolbarButton>
          ))}
        </div>
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
          title={
            showLabelNames
              ? "Masquer les noms des libellés"
              : "Afficher les noms des libellés"
          }
          className="w-11 justify-center px-0"
        >
          <span className="analyzer-toolbar__icon-slot">
            {showLabelNames ? (
              <Tags size={16} aria-hidden="true" />
            ) : (
              <span aria-hidden="true" className="text-base font-black leading-none tabular-nums">
                #
              </span>
            )}
          </span>
        </AnalyzerToolbarButton>
        <AnalyzerToolbarButton
          type="button"
          onClick={onDeselect}
          disabled={!hasSelection}
          aria-label={labels.deselect}
          title={labels.deselect}
          className="workspace-analyzer-toolbar__deselect min-w-36 justify-center"
        >
          <span className="analyzer-toolbar__icon-slot"><XCircle size={16} aria-hidden="true" /></span>
          <span className="analyzer-toolbar__label-slot">{labels.deselect}</span>
        </AnalyzerToolbarButton>
        <span aria-hidden="true" className="analyzer-toolbar__separator mx-1 h-6 w-px" />
        <AnalyzerToolbarButton type="button" onClick={onZoomOut} aria-label={labels.zoomOut} title={labels.zoomOut} className="justify-center px-2.5">
          <ZoomOut size={16} aria-hidden="true" />
        </AnalyzerToolbarButton>
        <AnalyzerToolbarButton type="button" onClick={onFitToView} aria-label={labels.fitToView} title={labels.fitToView} className="justify-center px-2.5">
          <Maximize2 size={16} aria-hidden="true" />
        </AnalyzerToolbarButton>
        <AnalyzerToolbarButton type="button" onClick={onZoomIn} aria-label={labels.zoomIn} title={labels.zoomIn} className="justify-center px-2.5">
          <ZoomIn size={16} aria-hidden="true" />
        </AnalyzerToolbarButton>
        <span className="workspace-analyzer-toolbar__zoom-label min-w-12 px-2 text-center text-xs font-semibold tabular-nums">
          {zoomLabel}
        </span>
      </AnalyzerToolbar>
    </div>
  );
}
