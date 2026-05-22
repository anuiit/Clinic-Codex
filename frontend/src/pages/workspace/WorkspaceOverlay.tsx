import type { PointerEvent as ReactPointerEvent } from "react";
import { getBoxVisualState } from "../../utils/segmentationBoxes";

type OverlayMode = "all" | "focused" | "hidden";

type WorkspaceOverlayProps = {
  imageSize: [number, number];
  elements: Array<{
    bbox: [number, number, number, number];
    class_name: string;
    rejected: boolean;
  }>;
  annotations: Record<number, string> | undefined;
  focusedIdx: number | null;
  hoveredIdx: number | null;
  hoverSource: "image" | "list" | null;
  overlayMode: OverlayMode;
  showLabelNames: boolean;
  zoom: number;
  onPointerDown: (event: ReactPointerEvent<SVGSVGElement>) => void;
  onPointerMove: (event: ReactPointerEvent<SVGSVGElement>) => void;
  onPointerLeave: () => void;
  getHitIdx: (clientX: number, clientY: number) => number | null;
  formatLabel: (idx: number, className: string, showName: boolean) => string;
  testId?: string;
};

export function WorkspaceOverlay({
  imageSize,
  elements,
  annotations,
  focusedIdx,
  hoveredIdx,
  hoverSource,
  overlayMode,
  showLabelNames,
  zoom,
  onPointerDown,
  onPointerMove,
  onPointerLeave,
  formatLabel,
  testId,
}: WorkspaceOverlayProps) {
  return (
    <svg
      className="absolute left-0 top-0 h-full w-full"
      data-testid={testId}
      viewBox={`0 0 ${imageSize[0]} ${imageSize[1]}`}
      preserveAspectRatio="none"
      style={{ width: "100%", height: "100%" }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerLeave={onPointerLeave}
    >
      {elements.map((el, idx) => {
        if (overlayMode === "focused" && focusedIdx !== null && focusedIdx !== idx) {
          return null;
        }
        const [x, y, w, h] = el.bbox;
        const visualState = getBoxVisualState({
          focused: idx === focusedIdx,
          hovered: idx === hoveredIdx && hoverSource === "image",
          listHovered: idx === hoveredIdx && hoverSource === "list",
          submitted: annotations?.[idx] !== undefined,
          rejected: el.rejected,
        });
        const labelY = Math.max(0, y - 28);
        const label = formatLabel(idx, el.class_name, showLabelNames);
        const labelWidth = showLabelNames
          ? Math.min(168, Math.max(64, label.length * 8 + 18))
          : 44;

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
