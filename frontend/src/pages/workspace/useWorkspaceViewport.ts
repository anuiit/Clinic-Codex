import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
  type WheelEvent as ReactWheelEvent,
} from "react";
import { getTrust } from "../../services/api";
import type { AnalysisRecord, TrustResult } from "../../types";
import { clientToImage } from "../../utils/imageCoords";
import {
  nextZoomFromWheel,
  shouldConsumeStageWheel,
} from "../../utils/imageStageZoom";
import { hitTestBBoxes } from "../../utils/segmentationBoxes";
import type { WorkspaceHoverSource, WorkspaceOverlayMode } from "./workspaceViewUtils";

const WORKSPACE_WHEEL_ZOOM_SENSITIVITY = 0.0015;

type OverlayMode = WorkspaceOverlayMode;
type HoverSource = WorkspaceHoverSource;

export function useWorkspaceViewport(currentRecord: AnalysisRecord | null) {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const [hoverSource, setHoverSource] = useState<HoverSource>(null);
  const [focusedIdx, setFocusedIdx] = useState<number | null>(null);
  const [zoom, setZoom] = useState(1);
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [overlayMode, setOverlayMode] = useState<OverlayMode>("all");
  const [showLabelNames, setShowLabelNames] = useState(false);
  const [trustData, setTrustData] = useState<TrustResult | null>(null);
  const [contextLoading, setContextLoading] = useState(false);

  const imageRef = useRef<HTMLImageElement>(null);
  const workspacePanStartRef = useRef<{
    clientX: number;
    clientY: number;
    offset: { x: number; y: number };
  } | null>(null);
  const cropCanvasRefs = useRef<(HTMLCanvasElement | null)[]>([]);
  const detailCanvasRef = useRef<HTMLCanvasElement>(null);
  const trustRequestIdRef = useRef(0);

  const resetWorkspaceView = useCallback(() => {
    setZoom(1);
    setPanOffset({ x: 0, y: 0 });
    setIsPanning(false);
    workspacePanStartRef.current = null;
  }, []);

  const resetInspectionState = useCallback(() => {
    cropCanvasRefs.current = [];
    setHoveredIdx(null);
    setHoverSource(null);
    setFocusedIdx(null);
    resetWorkspaceView();
    setOverlayMode("all");
  }, [resetWorkspaceView]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- reset image inspection state when the active record changes
    resetInspectionState();
  }, [currentRecord?.id, resetInspectionState]);

  useEffect(() => {
    if (!currentRecord || !imageRef.current) {
      return;
    }

    const image = imageRef.current;

    const drawCrop = (
      canvas: HTMLCanvasElement | null,
      element: AnalysisRecord["result"]["elements"][number],
    ) => {
      if (!canvas) {
        return;
      }

      const cropCtx = canvas.getContext("2d");
      if (!cropCtx) {
        return;
      }

      const [x, y, w, h] = element.bbox;
      cropCtx.clearRect(0, 0, canvas.width, canvas.height);
      cropCtx.drawImage(image, x, y, w, h, 0, 0, canvas.width, canvas.height);
    };

    const drawAllCanvases = () => {
      currentRecord.result.elements.forEach((element, idx) => {
        drawCrop(cropCanvasRefs.current[idx], element);
      });

      if (focusedIdx !== null) {
        const focusedElement = currentRecord.result.elements[focusedIdx];
        if (focusedElement) {
          drawCrop(detailCanvasRef.current, focusedElement);
        }
      }
    };

    if (image.complete) {
      drawAllCanvases();
    }

    image.addEventListener("load", drawAllCanvases);
    window.addEventListener("resize", drawAllCanvases);

    return () => {
      image.removeEventListener("load", drawAllCanvases);
      window.removeEventListener("resize", drawAllCanvases);
    };
  }, [currentRecord, focusedIdx]);

  useEffect(() => {
    if (focusedIdx === null || !currentRecord) {
      trustRequestIdRef.current += 1;
      // eslint-disable-next-line react-hooks/set-state-in-effect -- reset trust inspection state when no element is focused
      setContextLoading(false);
      setTrustData(null);
      return;
    }

    const element = currentRecord.result.elements[focusedIdx];
    if (!element) return;
    const controller = new AbortController();
    const requestId = trustRequestIdRef.current + 1;
    trustRequestIdRef.current = requestId;

    setContextLoading(true);

    const isActiveRequest = () =>
      trustRequestIdRef.current === requestId && !controller.signal.aborted;

    getTrust(currentRecord.imageDataUrl, element.bbox, element.class_name, 10, {
      signal: controller.signal,
    })
      .then((trust) => {
        if (isActiveRequest()) {
          setTrustData(trust);
        }
      })
      .catch(() => {
        if (isActiveRequest()) {
          setTrustData(null);
        }
      })
      .finally(() => {
        if (isActiveRequest()) {
          setContextLoading(false);
        }
      });

    return () => {
      controller.abort();
    };
  }, [focusedIdx, currentRecord]);

  const startWorkspacePan = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (zoom <= 1 || event.button !== 0) {
      return;
    }

    const target = event.target;
    if (
      target instanceof Element &&
      target.closest('[data-overlay-region="true"]')
    ) {
      return;
    }

    setIsPanning(true);
    workspacePanStartRef.current = {
      clientX: event.clientX,
      clientY: event.clientY,
      offset: { ...panOffset },
    };
    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const moveWorkspacePan = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!workspacePanStartRef.current) {
      return;
    }

    setPanOffset({
      x:
        workspacePanStartRef.current.offset.x +
        event.clientX -
        workspacePanStartRef.current.clientX,
      y:
        workspacePanStartRef.current.offset.y +
        event.clientY -
        workspacePanStartRef.current.clientY,
    });
  };

  const stopWorkspacePan = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!workspacePanStartRef.current && !isPanning) {
      return;
    }

    setIsPanning(false);
    workspacePanStartRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  };

  const handleWorkspaceStageWheel = (
    event: ReactWheelEvent<HTMLDivElement>,
  ) => {
    const { deltaY } = event;
    if (!shouldConsumeStageWheel(deltaY)) return;

    event.preventDefault();
    setZoom((currentZoom) => {
      const nextZoom = nextZoomFromWheel(
        currentZoom,
        deltaY,
        undefined,
        WORKSPACE_WHEEL_ZOOM_SENSITIVITY,
      );
      if (nextZoom <= 1) {
        setPanOffset({ x: 0, y: 0 });
      }
      return nextZoom;
    });
  };

  const getWorkspaceOverlayHit = (
    svg: SVGSVGElement,
    clientX: number,
    clientY: number,
  ): number | null => {
    if (!currentRecord) {
      return null;
    }

    const [imgW, imgH] = currentRecord.result.image_size;
    const point = clientToImage(svg, clientX, clientY, {
      width: imgW,
      height: imgH,
    });
    const visibleElements = currentRecord.result.elements
      .map((element, idx) => ({ idx, bbox: element.bbox }))
      .filter(
        ({ idx }) =>
          overlayMode === "all" || focusedIdx === null || focusedIdx === idx,
      );
    const hitIdx = hitTestBBoxes(
      point,
      visibleElements.map(({ bbox }) => bbox),
    );

    return hitIdx === null ? null : visibleElements[hitIdx].idx;
  };

  const handleWorkspaceOverlayPointerDown = (
    event: ReactPointerEvent<SVGSVGElement>,
  ) => {
    if (event.button !== 0) {
      return;
    }

    const hitIdx = getWorkspaceOverlayHit(
      event.currentTarget,
      event.clientX,
      event.clientY,
    );
    if (hitIdx === null) {
      return;
    }

    event.stopPropagation();
    setFocusedIdx(hitIdx);
    setHoveredIdx(hitIdx);
    setHoverSource(hitIdx === null ? null : "image");
  };

  const handleWorkspaceOverlayPointerMove = (
    event: ReactPointerEvent<SVGSVGElement>,
  ) => {
    const hitIdx = getWorkspaceOverlayHit(
      event.currentTarget,
      event.clientX,
      event.clientY,
    );
    setHoveredIdx(hitIdx);
    setHoverSource(hitIdx === null ? null : "image");
    event.currentTarget.style.cursor =
      hitIdx === null ? (zoom > 1 ? "grab" : "default") : "pointer";
  };

  const handleWorkspaceDetectedListKeyDown = (
    event: ReactKeyboardEvent<HTMLDivElement>,
  ) => {
    if (!currentRecord) return;
    const total = currentRecord.result.elements.length;
    if (total === 0) return;

    if (event.key === "ArrowDown" || event.key === "ArrowRight") {
      event.preventDefault();
      setFocusedIdx((previous) =>
        previous === null ? 0 : (previous + 1) % total,
      );
    } else if (event.key === "ArrowUp" || event.key === "ArrowLeft") {
      event.preventDefault();
      setFocusedIdx((previous) =>
        previous === null ? total - 1 : (previous - 1 + total) % total,
      );
    } else if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      if (focusedIdx !== null) setFocusedIdx(focusedIdx);
    } else if (event.key === "Escape") {
      setFocusedIdx(null);
    }
  };

  return {
    imageRef,
    cropCanvasRefs,
    detailCanvasRef,
    hoveredIdx,
    hoverSource,
    focusedIdx,
    zoom,
    panOffset,
    isPanning,
    overlayMode,
    showLabelNames,
    trustData,
    contextLoading,
    setHoveredIdx,
    setHoverSource,
    setFocusedIdx,
    setZoom,
    setPanOffset,
    setOverlayMode,
    setShowLabelNames,
    resetWorkspaceView,
    startWorkspacePan,
    moveWorkspacePan,
    stopWorkspacePan,
    handleWorkspaceStageWheel,
    handleWorkspaceOverlayPointerDown,
    handleWorkspaceOverlayPointerMove,
    handleWorkspaceDetectedListKeyDown,
  };
}
