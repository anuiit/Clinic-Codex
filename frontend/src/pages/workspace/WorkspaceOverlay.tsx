import type { PointerEvent as ReactPointerEvent } from "react";
import type { AnalysisRecord } from "../../types";
import { getBoxVisualState } from "../../utils/segmentationBoxes";
import {
  formatWorkspaceBboxLabel,
  type WorkspaceHoverSource,
  type WorkspaceOverlayMode,
} from "./workspaceViewUtils";

type WorkspaceOverlayProps = {
  record: AnalysisRecord;
  overlayMode: WorkspaceOverlayMode;
  focusedIdx: number | null;
  hoveredIdx: number | null;
  hoverSource: WorkspaceHoverSource;
  showLabelNames: boolean;
  onPointerDown: (event: ReactPointerEvent<SVGSVGElement>) => void;
  onPointerMove: (event: ReactPointerEvent<SVGSVGElement>) => void;
  onPointerLeave: () => void;
};

export default function WorkspaceOverlay({
  record,
  overlayMode,
  focusedIdx,
  hoveredIdx,
  hoverSource,
  showLabelNames,
  onPointerDown,
  onPointerMove,
  onPointerLeave,
}: WorkspaceOverlayProps) {
  if (overlayMode === "hidden") return null;

  return (
    <svg
      className="absolute left-0 top-0 h-full w-full"
      data-testid="workspace-overlay"
      viewBox={`0 0 ${record.result.image_size[0]} ${record.result.image_size[1]}`}
      preserveAspectRatio="none"
      style={{ width: "100%", height: "100%" }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerLeave={onPointerLeave}
    >
      {record.result.elements.map((element, idx) => {
        if (overlayMode === "focused" && focusedIdx !== null && focusedIdx !== idx) {
          return null;
        }

        const [x, y, w, h] = element.bbox;
        const visualState = getBoxVisualState({
          focused: idx === focusedIdx,
          hovered: idx === hoveredIdx && hoverSource === "image",
          listHovered: idx === hoveredIdx && hoverSource === "list",
          submitted: (record.annotations ?? {})[idx] !== undefined,
          rejected: element.rejected,
        });
        const label = formatWorkspaceBboxLabel(idx, element.class_name, showLabelNames);
        const labelWidth = showLabelNames
          ? Math.min(168, Math.max(64, label.length * 8 + 18))
          : 44;
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
            <title>{label}</title>
            <rect
              x={x}
              y={labelY}
              width={labelWidth}
              height={24}
              rx={6}
              fill={visualState.strokeColor}
              opacity={0.95}
            />
            <text
              x={x + 10}
              y={labelY + 12}
              fill={visualState.labelColor}
              fontSize="13"
              fontWeight="800"
              fontFamily="sans-serif"
              textAnchor="start"
              dominantBaseline="central"
            >
              {label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
