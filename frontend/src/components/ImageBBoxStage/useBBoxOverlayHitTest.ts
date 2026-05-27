import { useCallback } from "react";
import { clientToImage } from "../../utils/imageCoords";
import { hitTestBBoxes } from "../../utils/segmentationBoxes";
import type {
  ImageBBox,
  ImageBBoxBoxState,
  ImageBBoxOverlayMode,
  ImageBBoxStageBox,
  ImageBBoxStageBoxId,
  ImageBBoxStageBoxRenderState,
} from "./imageBBoxStage.types";

type BBoxOverlayHitTestOptions = {
  imageSize?: readonly [number, number] | null;
  boxes: ImageBBoxStageBox[];
  selectedId?: ImageBBoxStageBoxId | null;
  hoveredId?: ImageBBoxStageBoxId | null;
  overlayMode?: ImageBBoxOverlayMode;
  displayBBoxById?: Partial<Record<ImageBBoxStageBoxId, ImageBBox>>;
  boxStateById?: Partial<Record<ImageBBoxStageBoxId, ImageBBoxBoxState>>;
};

export function bboxIdsMatch(
  left: ImageBBoxStageBoxId | null | undefined,
  right: ImageBBoxStageBoxId | null | undefined,
) {
  return left !== null && left !== undefined && right !== null && right !== undefined
    ? String(left) === String(right)
    : left === right;
}

export function getBBoxOverlayRenderState(
  box: ImageBBoxStageBox,
  selectedId: ImageBBoxStageBoxId | null | undefined,
  hoveredId: ImageBBoxStageBoxId | null | undefined,
  boxStateById?: Partial<Record<ImageBBoxStageBoxId, ImageBBoxBoxState>>,
): ImageBBoxStageBoxRenderState {
  const overrideState = boxStateById?.[box.id] ?? {};
  const focused = overrideState.focused ?? bboxIdsMatch(box.id, selectedId);
  const imageHovered =
    overrideState.imageHovered ?? bboxIdsMatch(box.id, hoveredId);
  const listHovered = overrideState.listHovered ?? false;
  const submitted = overrideState.submitted ?? box.status === "validated";
  const rejected = overrideState.rejected ?? box.rejected ?? false;

  return {
    ...overrideState,
    focused,
    imageHovered,
    listHovered,
    submitted,
    rejected,
  };
}

export function getBBoxOverlayHit(
  svg: SVGSVGElement,
  clientX: number,
  clientY: number,
  {
    imageSize,
    boxes,
    selectedId = null,
    hoveredId = null,
    overlayMode = "all",
    displayBBoxById,
    boxStateById,
  }: BBoxOverlayHitTestOptions,
): ImageBBoxStageBoxId | null {
  if (!imageSize || overlayMode === "hidden") {
    return null;
  }

  const [imageWidth, imageHeight] = imageSize;
  const point = clientToImage(svg, clientX, clientY, {
    width: imageWidth,
    height: imageHeight,
  });
  const visibleBoxes = boxes
    .map((box) => ({
      box,
      bbox: displayBBoxById?.[box.id] ?? box.bbox,
      state: getBBoxOverlayRenderState(box, selectedId, hoveredId, boxStateById),
    }))
    .filter(
      ({ state }) =>
        overlayMode !== "focused" || selectedId === null || state.focused,
    );
  const hitIdx = hitTestBBoxes(
    point,
    visibleBoxes.map(({ bbox }) => bbox),
  );

  return hitIdx === null ? null : visibleBoxes[hitIdx].box.id;
}

export function useBBoxOverlayHitTest(options: BBoxOverlayHitTestOptions) {
  return useCallback(
    (svg: SVGSVGElement, clientX: number, clientY: number) =>
      getBBoxOverlayHit(svg, clientX, clientY, options),
    [options],
  );
}

export type { BBoxOverlayHitTestOptions };
export default useBBoxOverlayHitTest;
