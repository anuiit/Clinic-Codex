import { Tags } from "lucide-react";
import { AnalyzerToolbar, AnalyzerToolbarButton } from "../../components/AnalyzerToolbar";
import type { WorkspaceOverlayMode } from "./workspaceViewUtils";

type WorkspaceOverlayToolbarLabels = {
  overlayAll: string;
  overlayFocused: string;
  overlayHidden: string;
};

type WorkspaceOverlayToolbarProps = {
  overlayMode: WorkspaceOverlayMode;
  showLabelNames: boolean;
  labels: WorkspaceOverlayToolbarLabels;
  onOverlayModeChange: (mode: WorkspaceOverlayMode) => void;
  onToggleLabelNames: () => void;
};

const overlayModes: WorkspaceOverlayMode[] = ["all", "focused", "hidden"];

function getOverlayLabel(mode: WorkspaceOverlayMode, labels: WorkspaceOverlayToolbarLabels) {
  if (mode === "all") return labels.overlayAll;
  if (mode === "focused") return labels.overlayFocused;
  return labels.overlayHidden;
}

export default function WorkspaceOverlayToolbar({
  overlayMode,
  showLabelNames,
  labels,
  onOverlayModeChange,
  onToggleLabelNames,
}: WorkspaceOverlayToolbarProps) {
  return (
    <AnalyzerToolbar>
      <div className="inline-flex rounded-lg border border-stone-700 bg-stone-950 p-1">
        {overlayModes.map((mode) => (
          <AnalyzerToolbarButton
            key={mode}
            type="button"
            onClick={() => onOverlayModeChange(mode)}
            active={overlayMode === mode}
            className="rounded-md px-3 py-1.5 capitalize"
          >
            {getOverlayLabel(mode, labels)}
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
        className="w-10 justify-center px-0"
      >
        {showLabelNames ? (
          <Tags size={16} aria-hidden="true" />
        ) : (
          <span aria-hidden="true" className="text-base font-black leading-none">
            #
          </span>
        )}
      </AnalyzerToolbarButton>
    </AnalyzerToolbar>
  );
}
