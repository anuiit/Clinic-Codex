import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
} from "react";
import { ActionButton, StatusPill } from "../../components/ui/AdminPrimitives";
import {
  createImageBBoxZoomControls,
  ImageBBoxStage,
  ImageBBoxToolbar,
  useImageStageViewport,
  type ImageBBox,
  type ImageBBoxStageBox,
  type ImageBBoxStageBoxRenderState,
} from "../../components/ImageBBoxStage";
import { adminAnnotationMediaUrl } from "../../services/api";
import { clientToImage } from "../../utils/imageCoords";
import {
  clampBBox,
  hitTestHandle,
  moveBBox,
  resizeBBox,
  type BBoxHandle,
} from "../../utils/segmentationBoxes";
import type {
  AdminAnnotationAnalysis,
  AdminAnnotationElement,
} from "../../types";

export type AdminCorrectionBBox = [number, number, number, number];

export type AdminCorrectionStageProps = {
  analysis: AdminAnnotationAnalysis;
  selectedElement: AdminAnnotationElement;
  bbox: AdminCorrectionBBox;
  onBboxChange: (bbox: AdminCorrectionBBox) => void;
  onImageSizeChange?: (size: [number, number] | null) => void;
  mutating: boolean;
  className?: string;
};

type InteractionMode = "edit" | "pan";
type DragState =
  | { type: "draw"; start: { x: number; y: number } }
  | {
      type: "move";
      start: { x: number; y: number };
      original: AdminCorrectionBBox;
    }
  | {
      type: "resize";
      handle: BBoxHandle;
      original: AdminCorrectionBBox;
    };

function toBBox(values: number[]): AdminCorrectionBBox {
  return [
    Math.round(values[0] ?? 0),
    Math.round(values[1] ?? 0),
    Math.max(1, Math.round(values[2] ?? 1)),
    Math.max(1, Math.round(values[3] ?? 1)),
  ];
}

function roundAndClamp(
  bbox: AdminCorrectionBBox,
  [width, height]: [number, number],
): AdminCorrectionBBox {
  const clamped = clampBBox(
    bbox.map((value) => Math.round(value)) as AdminCorrectionBBox,
    { width, height },
  );
  return [
    Math.round(clamped[0]),
    Math.round(clamped[1]),
    Math.max(1, Math.round(clamped[2])),
    Math.max(1, Math.round(clamped[3])),
  ];
}

function containsPoint(
  point: { x: number; y: number },
  [x, y, width, height]: AdminCorrectionBBox,
) {
  return (
    point.x >= x &&
    point.x <= x + width &&
    point.y >= y &&
    point.y <= y + height
  );
}

function sameBBox(left: AdminCorrectionBBox, right: AdminCorrectionBBox) {
  return left.every((value, index) => value === right[index]);
}

export function AdminCorrectionStage({
  analysis,
  selectedElement,
  bbox,
  onBboxChange,
  onImageSizeChange,
  mutating,
  className,
}: AdminCorrectionStageProps) {
  const [imageSize, setImageSize] = useState<[number, number] | null>(null);
  const [interactionMode, setInteractionMode] =
    useState<InteractionMode>("edit");
  const dragRef = useRef<DragState | null>(null);
  const imageUrl = adminAnnotationMediaUrl(analysis.image_url);
  const originalBBox = useMemo(
    () => toBBox(selectedElement.bbox),
    [selectedElement.bbox],
  );
  const currentBBox = imageSize ? roundAndClamp(bbox, imageSize) : bbox;
  const selectedKey = selectedElement.key;
  const viewport = useImageStageViewport({
    imageSize,
    disabled: !analysis.image_exists || !imageSize,
    loading: !imageSize,
    resetKey: `${analysis.analysis_id}:${selectedKey}`,
  });

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- a new source needs a fresh natural-size measurement
    setImageSize(null);
    setInteractionMode("edit");
    dragRef.current = null;
    onImageSizeChange?.(null);
  }, [analysis.analysis_id, analysis.image_url, onImageSizeChange]);

  const recordNaturalSize = useCallback(
    (image: HTMLImageElement) => {
      const nextSize: [number, number] = [
        image.naturalWidth,
        image.naturalHeight,
      ];
      if (nextSize[0] <= 0 || nextSize[1] <= 0) return;
      setImageSize(nextSize);
      onImageSizeChange?.(nextSize);
    },
    [onImageSizeChange],
  );

  const boxes = useMemo<ImageBBoxStageBox[]>(
    () =>
      analysis.elements.map((element) => ({
        id: element.key,
        bbox: toBBox(element.bbox),
        label: `#${element.index} ${element.class_name || "Sans nom"}`,
        rejected: element.review_status === "rejected",
        status: element.review_status === "approved" ? "validated" : "draft",
      })),
    [analysis.elements],
  );

  const displayBBoxById = useMemo(
    () => ({ [selectedKey]: currentBBox }),
    [currentBBox, selectedKey],
  );

  const commitBBox = useCallback(
    (nextBBox: AdminCorrectionBBox) => {
      if (!imageSize) return;
      onBboxChange(roundAndClamp(nextBBox, imageSize));
    },
    [imageSize, onBboxChange],
  );

  const pointFromEvent = useCallback(
    (event: ReactPointerEvent<SVGSVGElement>) => {
      if (!imageSize) return null;
      return clientToImage(
        event.currentTarget,
        event.clientX,
        event.clientY,
        { width: imageSize[0], height: imageSize[1] },
      );
    },
    [imageSize],
  );

  const handlePointerDown = useCallback(
    (event: ReactPointerEvent<SVGSVGElement>) => {
      if (mutating || !imageSize || event.button > 0) return;
      if (interactionMode === "pan") {
        viewport.beginPan(event);
        return;
      }

      const point = pointFromEvent(event);
      if (!point) return;
      const handle = hitTestHandle(
        point,
        currentBBox,
        Math.max(5, 10 / viewport.zoom),
      );
      if (handle) {
        dragRef.current = { type: "resize", handle, original: currentBBox };
      } else if (containsPoint(point, currentBBox)) {
        dragRef.current = {
          type: "move",
          start: point,
          original: currentBBox,
        };
      } else {
        dragRef.current = { type: "draw", start: point };
        commitBBox([Math.round(point.x), Math.round(point.y), 1, 1]);
      }
      event.preventDefault();
      event.currentTarget.setPointerCapture(event.pointerId);
    },
    [
      commitBBox,
      currentBBox,
      imageSize,
      interactionMode,
      mutating,
      pointFromEvent,
      viewport,
    ],
  );

  const handlePointerMove = useCallback(
    (event: ReactPointerEvent<SVGSVGElement>) => {
      if (interactionMode === "pan") {
        viewport.movePan(event);
        return;
      }
      const drag = dragRef.current;
      const point = pointFromEvent(event);
      if (!drag || !point || !imageSize) return;

      if (drag.type === "draw") {
        commitBBox([
          Math.min(drag.start.x, point.x),
          Math.min(drag.start.y, point.y),
          Math.max(1, Math.abs(point.x - drag.start.x)),
          Math.max(1, Math.abs(point.y - drag.start.y)),
        ]);
      } else if (drag.type === "move") {
        commitBBox(
          moveBBox(
            drag.original,
            { x: point.x - drag.start.x, y: point.y - drag.start.y },
            { width: imageSize[0], height: imageSize[1] },
          ) as AdminCorrectionBBox,
        );
      } else {
        commitBBox(
          resizeBBox(
            drag.original,
            drag.handle,
            point,
            { width: imageSize[0], height: imageSize[1] },
          ) as AdminCorrectionBBox,
        );
      }
    },
    [commitBBox, imageSize, interactionMode, pointFromEvent, viewport],
  );

  const endPointerInteraction = useCallback(
    (event: ReactPointerEvent<SVGSVGElement>) => {
      if (interactionMode === "pan") {
        viewport.endPan(event);
      }
      dragRef.current = null;
      if (event.currentTarget.hasPointerCapture(event.pointerId)) {
        event.currentTarget.releasePointerCapture(event.pointerId);
      }
    },
    [interactionMode, viewport],
  );

  const renderBoxExtras = useCallback(
    (
      box: ImageBBoxStageBox,
      _state: ImageBBoxStageBoxRenderState,
      renderedBBox: ImageBBox,
    ) => {
      if (box.id !== selectedKey) return null;
      const [x, y, width, height] = renderedBBox;
      const corners = [
        [x, y],
        [x + width, y],
        [x, y + height],
        [x + width, y + height],
      ];
      return (
        <>
          {!sameBBox(originalBBox, renderedBBox as AdminCorrectionBBox) ? (
            <rect
              data-testid="admin-correction-original-bbox"
              x={originalBBox[0]}
              y={originalBBox[1]}
              width={originalBBox[2]}
              height={originalBBox[3]}
              fill="none"
              stroke="var(--text-soft)"
              strokeDasharray="6 4"
              strokeWidth={2}
              vectorEffect="non-scaling-stroke"
              pointerEvents="none"
            />
          ) : null}
          {corners.map(([cornerX, cornerY], index) => (
            <rect
              key={index}
              className="admin-segmentation-handle"
              x={cornerX - 5}
              y={cornerY - 5}
              width={10}
              height={10}
              rx={2}
              fill="var(--accent)"
              stroke="var(--app-bg)"
              strokeWidth={2}
              vectorEffect="non-scaling-stroke"
              pointerEvents="none"
            />
          ))}
        </>
      );
    },
    [originalBBox, selectedKey],
  );

  const zoomControls = useMemo(
    () =>
      createImageBBoxZoomControls({
        labels: {
          zoomOut: "Dézoomer",
          fitToView: "Ajuster à la vue",
          zoomIn: "Zoomer",
        },
        onZoomOut: viewport.zoomOut,
        onFitToView: viewport.resetView,
        onZoomIn: viewport.zoomIn,
      }),
    [viewport.resetView, viewport.zoomIn, viewport.zoomOut],
  );

  return (
    <div
      className={`admin-segmentation-stage ${className ?? ""}`.trim()}
      data-testid="admin-segmentation-stage"
      data-natural-width={imageSize?.[0]}
      data-natural-height={imageSize?.[1]}
    >
      {!analysis.image_exists ? (
        <div className="ui-empty-state h-full">Image source manquante</div>
      ) : !imageSize ? (
        <img
          src={imageUrl}
          alt={`Image à corriger ${analysis.analysis_id}`}
          className="admin-segmentation-preloader"
          onLoad={(event) => recordNaturalSize(event.currentTarget)}
        />
      ) : (
        <ImageBBoxStage
          imageDataUrl={imageUrl}
          imageName={`Image à corriger ${analysis.analysis_id}`}
          imageSize={imageSize}
          boxes={boxes}
          selectedId={selectedKey}
          showLabelNames
          mode="edit"
          title={`Correction · élément #${selectedElement.index}`}
          headerMeta={
            <div className="flex flex-wrap items-center gap-2">
              <StatusPill>{analysis.elements.length} éléments en contexte</StatusPill>
              <StatusPill tone="ready">{Math.round(viewport.zoom * 100)} %</StatusPill>
            </div>
          }
          toolbar={
            <ImageBBoxToolbar label="Outils de correction">
              <ActionButton
                tone={interactionMode === "edit" ? "primary" : "ghost"}
                className="px-3 py-1.5 text-xs"
                disabled={mutating}
                aria-pressed={interactionMode === "edit"}
                onClick={() => setInteractionMode("edit")}
              >
                Modifier la zone
              </ActionButton>
              <ActionButton
                tone={interactionMode === "pan" ? "primary" : "ghost"}
                className="px-3 py-1.5 text-xs"
                disabled={mutating || viewport.zoom <= 1}
                aria-pressed={interactionMode === "pan"}
                onClick={() => setInteractionMode("pan")}
              >
                Déplacer l’image
              </ActionButton>
            </ImageBBoxToolbar>
          }
          panelControls={zoomControls}
          controlsPlacement="bottom-right"
          zoomLabel={`${Math.round(viewport.zoom * 100)} %`}
          viewport={viewport}
          transformSize={viewport.transformSize}
          displayBBoxById={displayBBoxById}
          boxStateById={{ [selectedKey]: { focused: true } }}
          renderLabel={(box) => box.label ?? "Sans nom"}
          renderBoxExtras={renderBoxExtras}
          imageProps={{
            alt: `Image à corriger ${analysis.analysis_id}`,
            onLoad: (event) => recordNaturalSize(event.currentTarget),
          }}
          stageRef={viewport.containerRef}
          stageProps={{ onWheel: viewport.handleStageWheel }}
          svgProps={{
            "aria-label": `Redessiner la segmentation de l'élément ${selectedElement.index}`,
            onPointerDown: handlePointerDown,
            onPointerMove: handlePointerMove,
            onPointerUp: endPointerInteraction,
            onPointerCancel: endPointerInteraction,
          }}
          testIds={{
            root: "admin-correction-image-shell",
            stage: "admin-correction-image-stage",
            transform: "admin-correction-image-transform",
            overlay: "admin-correction-image-overlay",
            controls: "admin-correction-image-controls",
          }}
        />
      )}
    </div>
  );
}

export default AdminCorrectionStage;
