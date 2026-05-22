import type { PointerEvent as ReactPointerEvent } from "react";
import type { AnnotationStatus, DetectedElement } from "../../types";
import { getBoxVisualState } from "../../utils/segmentationBoxes";
import { formatBboxLabel } from "./annotationUtils";

type OverlayDragState = {
  type: "draw" | "move" | "resize";
  idx: number;
} | null;

interface AnnotationOverlayProps {
  imageSize: [number, number];
  elements: DetectedElement[];
  annotationStatus: Record<number, AnnotationStatus>;
  focusedIdx: number | null;
  hoveredIdx: number | null;
  listHoveredIdx: number | null;
  drawMode: boolean;
  dragState: OverlayDragState;
  tempBbox: [number, number, number, number] | null;
  showLabelNames: boolean;
  unnamedLabel: string;
  onPointerDown: (event: ReactPointerEvent<SVGSVGElement>) => void;
  onPointerMove: (event: ReactPointerEvent<SVGSVGElement>) => void;
  onPointerUp: (event: ReactPointerEvent<SVGSVGElement>) => void;
}

export function AnnotationOverlay({
  imageSize,
  elements,
  annotationStatus,
  focusedIdx,
  hoveredIdx,
  listHoveredIdx,
  drawMode,
  dragState,
  tempBbox,
  showLabelNames,
  unnamedLabel,
  onPointerDown,
  onPointerMove,
  onPointerUp,
}: AnnotationOverlayProps) {
  const [imageWidth, imageHeight] = imageSize;

  return (
    <svg
      data-testid="annotation-overlay"
      className="absolute left-0 top-0 h-full w-full"
      viewBox={`0 0 ${imageWidth} ${imageHeight}`}
      preserveAspectRatio="none"
      style={{ width: "100%", height: "100%" }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
    >
      {elements.map((el, idx) => {
        const [x, y, w, h] =
          idx === dragState?.idx && dragState.type !== "draw" && tempBbox
            ? tempBbox
            : el.bbox;
        const isFocused = idx === focusedIdx;
        const isHovered = idx === hoveredIdx;
        const isListHovered = idx === listHoveredIdx;
        const isSubmitted = annotationStatus[idx] === "validated";
        const boxVisual = getBoxVisualState({
          focused: isFocused,
          hovered: isHovered,
          listHovered: isListHovered,
          submitted: isSubmitted,
          rejected: el.rejected,
        });
        const label = formatBboxLabel(
          idx,
          el.class_name,
          showLabelNames,
          unnamedLabel,
        );
        const labelY = Math.max(0, y - 28);
        const labelWidth = showLabelNames
          ? Math.min(168, Math.max(64, label.length * 8 + 18))
          : 44;

        return (
          <g key={idx} className="pointer-events-none select-none">
            <rect
              x={x}
              y={y}
              width={w}
              height={h}
              data-testid={`annotation-box-${idx}`}
              fill={boxVisual.fillColor}
              stroke={boxVisual.strokeColor}
              strokeWidth={boxVisual.strokeWidth}
              strokeDasharray={boxVisual.strokeDasharray}
              vectorEffect="non-scaling-stroke"
            />
            {isFocused && !drawMode && (
              <>
                <rect
                  x={x - 6}
                  y={y - 6}
                  width={12}
                  height={12}
                  rx={3}
                  fill="#fef3c7"
                  stroke="#0c0a09"
                  strokeWidth={1.5}
                  className="cursor-nwse-resize"
                />
                <rect
                  x={x + w - 6}
                  y={y - 6}
                  width={12}
                  height={12}
                  rx={3}
                  fill="#fef3c7"
                  stroke="#0c0a09"
                  strokeWidth={1.5}
                  className="cursor-nesw-resize"
                />
                <rect
                  x={x - 6}
                  y={y + h - 6}
                  width={12}
                  height={12}
                  rx={3}
                  fill="#fef3c7"
                  stroke="#0c0a09"
                  strokeWidth={1.5}
                  className="cursor-nesw-resize"
                />
                <rect
                  x={x + w - 6}
                  y={y + h - 6}
                  width={12}
                  height={12}
                  rx={3}
                  fill="#fef3c7"
                  stroke="#0c0a09"
                  strokeWidth={1.5}
                  className="cursor-nwse-resize"
                />
              </>
            )}
            <title>{label}</title>
            <rect
              x={x}
              y={labelY}
              width={labelWidth}
              height={24}
              rx={6}
              fill={boxVisual.strokeColor}
              opacity={0.95}
              pointerEvents="none"
              style={{ userSelect: "none" }}
            />
            <text
              x={x + 10}
              y={labelY + 12}
              fill="#0c0a09"
              fontSize="13"
              fontWeight="800"
              fontFamily="sans-serif"
              textAnchor="start"
              dominantBaseline="central"
              pointerEvents="none"
              style={{ userSelect: "none" }}
            >
              {label}
            </text>
          </g>
        );
      })}
      {drawMode && dragState?.type === "draw" && tempBbox && (
        <rect
          x={tempBbox[0]}
          y={tempBbox[1]}
          width={tempBbox[2]}
          height={tempBbox[3]}
          fill="rgba(59, 130, 246, 0.14)"
          stroke="#60a5fa"
          strokeWidth={2}
          strokeDasharray="4 4"
          vectorEffect="non-scaling-stroke"
        />
      )}
    </svg>
  );
}
