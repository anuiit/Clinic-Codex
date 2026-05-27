import type { PointerEvent as ReactPointerEvent } from "react";
import { getBoxVisualState } from "../../utils/segmentationBoxes";
import type {
  ImageBBoxOverlayProps,
  ImageBBoxStageBox,
} from "./imageBBoxStage.types";
import {
  getBBoxOverlayRenderState,
  useBBoxOverlayHitTest,
} from "./useBBoxOverlayHitTest";

function cx(...values: Array<string | false | null | undefined>) {
  return values.filter(Boolean).join(" ");
}

function getDefaultLabel(
  box: ImageBBoxStageBox,
  showLabelNames: boolean,
): string {
  if (showLabelNames && box.label) {
    return `#${box.id} ${box.label}`;
  }

  return `#${box.id}`;
}

function getLabelWidth(label: string, showLabelNames: boolean) {
  if (!showLabelNames) return 44;
  return Math.min(168, Math.max(64, label.length * 8 + 18));
}

export function ImageBBoxOverlay({
  imageSize,
  boxes,
  selectedId = null,
  hoveredId = null,
  showLabelNames = false,
  overlayMode = "all",
  onSelectBox,
  onHoverBox,
  displayBBoxById,
  boxStateById,
  renderLabel,
  renderBoxExtras,
  overlayChildren,
  boxTestIdPrefix = "image-bbox-stage-box",
  svgProps,
  testId,
}: ImageBBoxOverlayProps) {
  const getOverlayHit = useBBoxOverlayHitTest({
    imageSize,
    boxes,
    selectedId,
    hoveredId,
    overlayMode,
    displayBBoxById,
    boxStateById,
  });

  if (overlayMode === "hidden") {
    return null;
  }

  const [imageWidth, imageHeight] = imageSize;
  const {
    className,
    style,
    children: svgChildren,
    onPointerDown,
    onPointerMove,
    onPointerLeave,
    ...svgRest
  } = svgProps ?? {};
  const handlePointerDown = (event: ReactPointerEvent<SVGSVGElement>) => {
    onPointerDown?.(event);
    if (!onSelectBox || event.defaultPrevented || event.button !== 0) {
      return;
    }

    onSelectBox(
      getOverlayHit(
        event.currentTarget,
        event.clientX,
        event.clientY,
      ),
    );
  };
  const handlePointerMove = (event: ReactPointerEvent<SVGSVGElement>) => {
    onPointerMove?.(event);
    if (!onHoverBox || event.defaultPrevented) {
      return;
    }

    onHoverBox(
      getOverlayHit(
        event.currentTarget,
        event.clientX,
        event.clientY,
      ),
    );
  };
  const handlePointerLeave = (event: ReactPointerEvent<SVGSVGElement>) => {
    onPointerLeave?.(event);
    onHoverBox?.(null);
  };

  return (
    <svg
      {...svgRest}
      className={cx("absolute left-0 top-0 h-full w-full", className)}
      data-testid={testId}
      viewBox={`0 0 ${imageWidth} ${imageHeight}`}
      preserveAspectRatio="none"
      style={{ width: "100%", height: "100%", ...style }}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerLeave={handlePointerLeave}
    >
      {boxes.map((box) => {
        const state = getBBoxOverlayRenderState(
          box,
          selectedId,
          hoveredId,
          boxStateById,
        );
        if (overlayMode === "focused" && selectedId !== null && !state.focused) {
          return null;
        }

        const bbox = displayBBoxById?.[box.id] ?? box.bbox;
        const [x, y, width, height] = bbox;
        const visualState = getBoxVisualState({
          focused: state.focused,
          hovered: state.imageHovered,
          listHovered: state.listHovered,
          submitted: state.submitted,
          rejected: state.rejected,
        });
        const fallbackLabel = getDefaultLabel(box, showLabelNames);
        const label = renderLabel?.(box, state) ?? fallbackLabel;
        const titleLabel =
          typeof label === "string" ? label : fallbackLabel;
        const labelY = Math.max(0, y - 28);
        const labelWidth = getLabelWidth(titleLabel, showLabelNames);

        return (
          <g
            key={box.id}
            data-overlay-region="true"
            data-box-id={box.id}
            data-selected={state.focused || undefined}
            data-hovered={
              state.imageHovered || state.listHovered || undefined
            }
            data-status={state.submitted ? "validated" : "draft"}
            data-rejected={state.rejected || undefined}
            className="pointer-events-none select-none"
          >
            <title>{titleLabel}</title>
            <rect
              x={x}
              y={y}
              width={width}
              height={height}
              data-testid={`${boxTestIdPrefix}-${box.id}`}
              fill={visualState.fillColor}
              stroke={visualState.strokeColor}
              strokeWidth={visualState.strokeWidth}
              strokeDasharray={visualState.strokeDasharray}
              vectorEffect="non-scaling-stroke"
            />
            {renderBoxExtras?.(box, state, bbox)}
            <rect
              x={x}
              y={labelY}
              width={labelWidth}
              height={24}
              rx={6}
              fill={visualState.strokeColor}
              opacity={0.95}
              pointerEvents="none"
              style={{ userSelect: "none" }}
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
              pointerEvents="none"
              style={{ userSelect: "none" }}
            >
              {label}
            </text>
          </g>
        );
      })}
      {overlayChildren}
      {svgChildren}
    </svg>
  );
}

export default ImageBBoxOverlay;
