import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
  type WheelEvent as ReactWheelEvent,
} from "react";
import {
  clampZoom,
  nextZoomFromWheel,
  shouldConsumeStageWheel,
  type ImageStageZoomBounds,
} from "../../utils/imageStageZoom";
import {
  useImageBBoxStageSize,
  type UseImageBBoxStageSizeOptions,
} from "./useImageBBoxStageSize";

export const IMAGE_BBOX_STAGE_WHEEL_SENSITIVITY = 0.0015;
export const IMAGE_BBOX_STAGE_ZOOM_STEP = 0.25;

type PanOffset = { x: number; y: number };

export type UseImageBBoxStageViewportOptions = UseImageBBoxStageSizeOptions &
  ImageStageZoomBounds & {
    resetKey?: string | number | null;
    wheelSensitivity?: number;
    zoomStep?: number;
  };

export type UseImageBBoxStageViewportResult = ReturnType<
  typeof useImageBBoxStageViewport
>;

function samePan(left: PanOffset, right: PanOffset) {
  return left.x === right.x && left.y === right.y;
}

function getImageFallback(
  imageSize: UseImageBBoxStageSizeOptions["imageSize"],
) {
  const [width, height] = imageSize ?? [];
  return width && height && width > 0 && height > 0 ? { width, height } : null;
}

export function useImageBBoxStageViewport({
  imageSize,
  disabled = false,
  loading = false,
  resetKey = null,
  min,
  max,
  wheelSensitivity = IMAGE_BBOX_STAGE_WHEEL_SENSITIVITY,
  zoomStep = IMAGE_BBOX_STAGE_ZOOM_STEP,
}: UseImageBBoxStageViewportOptions) {
  const stageSize = useImageBBoxStageSize({ imageSize, disabled, loading });
  const [zoom, setZoom] = useState(1);
  const [panOffset, setPanOffset] = useState<PanOffset>({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const panStartRef = useRef<{
    clientX: number;
    clientY: number;
    offset: PanOffset;
  } | null>(null);
  const { getContainerElement, transformSize } = stageSize;
  const fallbackSize = useMemo(() => getImageFallback(imageSize), [imageSize]);
  const zoomBounds = useMemo(() => ({ min, max }), [max, min]);

  const clampPan = useCallback(
    (offset: PanOffset, zoomLevel: number): PanOffset => {
      const container = getContainerElement();
      const stage = transformSize ?? fallbackSize;

      if (!container || !stage || zoomLevel <= 1) {
        return { x: 0, y: 0 };
      }

      const containerRect = container.getBoundingClientRect();
      const scaledWidth = stage.width * zoomLevel;
      const scaledHeight = stage.height * zoomLevel;
      const maxPanX =
        scaledWidth / 2 + containerRect.width / 2 - scaledWidth * 0.2;
      const maxPanY =
        scaledHeight / 2 + containerRect.height / 2 - scaledHeight * 0.2;

      return {
        x: Math.max(-maxPanX, Math.min(maxPanX, offset.x)),
        y: Math.max(-maxPanY, Math.min(maxPanY, offset.y)),
      };
    },
    [fallbackSize, getContainerElement, transformSize],
  );

  const resetView = useCallback(() => {
    setZoom(1);
    setPanOffset({ x: 0, y: 0 });
    setIsPanning(false);
    panStartRef.current = null;
  }, []);

  const applyZoom = useCallback(
    (nextZoom: number) => {
      const clampedZoom = clampZoom(nextZoom, zoomBounds);
      setZoom(clampedZoom);
      setPanOffset((current) => {
        const nextPan =
          clampedZoom <= 1 ? { x: 0, y: 0 } : clampPan(current, clampedZoom);
        return samePan(current, nextPan) ? current : nextPan;
      });
    },
    [clampPan, zoomBounds],
  );

  const zoomBy = useCallback(
    (delta: number) => {
      applyZoom(zoom + delta);
    },
    [applyZoom, zoom],
  );

  const zoomIn = useCallback(() => zoomBy(zoomStep), [zoomBy, zoomStep]);
  const zoomOut = useCallback(() => zoomBy(-zoomStep), [zoomBy, zoomStep]);

  const handleStageWheel = useCallback(
    (event: ReactWheelEvent<HTMLDivElement>) => {
      if (!shouldConsumeStageWheel(event.deltaY)) return;

      event.preventDefault();
      applyZoom(
        nextZoomFromWheel(zoom, event.deltaY, zoomBounds, wheelSensitivity),
      );
    },
    [applyZoom, wheelSensitivity, zoom, zoomBounds],
  );

  const beginPan = useCallback(
    <T extends Element>(event: ReactPointerEvent<T>) => {
      if (zoom <= 1 || event.button > 0) {
        return false;
      }

      setIsPanning(true);
      panStartRef.current = {
        clientX: event.clientX,
        clientY: event.clientY,
        offset: { ...panOffset },
      };
      event.preventDefault();
      event.currentTarget.setPointerCapture(event.pointerId);
      return true;
    },
    [panOffset, zoom],
  );

  const movePan = useCallback(
    <T extends Element>(event: ReactPointerEvent<T>) => {
      if (!panStartRef.current) {
        return false;
      }

      const nextPan = clampPan({
        x:
          panStartRef.current.offset.x +
          event.clientX -
          panStartRef.current.clientX,
        y:
          panStartRef.current.offset.y +
          event.clientY -
          panStartRef.current.clientY,
      }, zoom);
      setPanOffset((current) => (samePan(current, nextPan) ? current : nextPan));
      return true;
    },
    [clampPan, zoom],
  );

  const endPan = useCallback(<T extends Element>(event: ReactPointerEvent<T>) => {
    if (!panStartRef.current && !isPanning) {
      return false;
    }

    setIsPanning(false);
    panStartRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    return true;
  }, [isPanning]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- reset shared viewer state when consumers switch images/records
    resetView();
  }, [resetKey, resetView]);

  useEffect(() => {
    if (zoom <= 1) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- keep pan state fit-to-view when zoom reaches/below 1
      setPanOffset((current) =>
        samePan(current, { x: 0, y: 0 }) ? current : { x: 0, y: 0 },
      );
      setIsPanning(false);
      panStartRef.current = null;
    }
  }, [zoom]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- reclamp pan after stage measurement changes
    setPanOffset((current) => {
      const nextPan = clampPan(current, zoom);
      return samePan(current, nextPan) ? current : nextPan;
    });
  }, [clampPan, transformSize, zoom]);

  return {
    ...stageSize,
    zoom,
    panOffset,
    isPanning,
    resetView,
    applyZoom,
    zoomIn,
    zoomOut,
    handleStageWheel,
    beginPan,
    movePan,
    endPan,
  };
}

export default useImageBBoxStageViewport;
