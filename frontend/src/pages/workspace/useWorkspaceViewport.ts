import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
} from "react";
import {
  useBBoxOverlayHitTest,
  useBBoxSelection,
  useImageStageViewport,
} from "../../components/ImageBBoxStage";
import { getTrust } from "../../services/api";
import type { AnalysisRecord, TrustResult } from "../../types";
import type { WorkspaceHoverSource, WorkspaceOverlayMode } from "./workspaceViewUtils";

type OverlayMode = WorkspaceOverlayMode;
type HoverSource = WorkspaceHoverSource;
export type WorkspaceTrustState = {
  recordId: string | null;
  focusedIdx: number | null;
  status: "idle" | "loading" | "ready" | "error";
  data: TrustResult | null;
};

export function useWorkspaceViewport(currentRecord: AnalysisRecord | null) {
  const {
    hoveredId: hoveredIdx,
    hoverSource,
    focusedId: focusedIdx,
    setHoveredId: setHoveredIdx,
    setHoverSource,
    setFocusedId: setFocusedIdx,
    clearSelection,
  } = useBBoxSelection<number, HoverSource>();
  const [overlayMode, setOverlayMode] = useState<OverlayMode>("all");
  const [showLabelNames, setShowLabelNames] = useState(false);
  const [trustState, setTrustState] = useState<WorkspaceTrustState>({
    recordId: null,
    focusedIdx: null,
    status: "idle",
    data: null,
  });
  const stageViewport = useImageStageViewport({
    imageSize: currentRecord?.result.image_size,
    disabled: !currentRecord,
    resetKey: currentRecord?.id ?? null,
  });
  const updateStageSize = stageViewport.updateStageSize;
  const overlayHitBoxes = useMemo(
    () =>
      currentRecord?.result.elements.map((element, idx) => ({
        id: idx,
        bbox: element.bbox,
      })) ?? [],
    [currentRecord?.result.elements],
  );
  const getWorkspaceOverlayHit = useBBoxOverlayHitTest({
    imageSize: currentRecord?.result.image_size,
    boxes: overlayHitBoxes,
    selectedId: focusedIdx,
    overlayMode,
  });

  const imageRef = useRef<HTMLImageElement>(null);
  const cropCanvasRefs = useRef<(HTMLCanvasElement | null)[]>([]);
  const detailCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const cropDrawFrameRef = useRef<number | null>(null);
  const latestRecordRef = useRef(currentRecord);
  const latestFocusedIdxRef = useRef(focusedIdx);
  const trustRequestIdRef = useRef(0);
  const pendingEmptyClickRef = useRef<{
    pointerId: number;
    startX: number;
    startY: number;
    canceled: boolean;
  } | null>(null);
  const resetWorkspaceView = stageViewport.resetView;

  useLayoutEffect(() => {
    latestRecordRef.current = currentRecord;
    latestFocusedIdxRef.current = focusedIdx;
  });

  const cancelScheduledCropDraw = useCallback(() => {
    if (cropDrawFrameRef.current !== null) {
      cancelAnimationFrame(cropDrawFrameRef.current);
      cropDrawFrameRef.current = null;
    }
  }, []);

  const scheduleCropDraw = useCallback(() => {
    cancelScheduledCropDraw();
    const scheduledRecordId = latestRecordRef.current?.id ?? null;

    cropDrawFrameRef.current = requestAnimationFrame(() => {
      cropDrawFrameRef.current = null;

      const record = latestRecordRef.current;
      if (!record || record.id !== scheduledRecordId) {
        return;
      }

      const image = imageRef.current;
      if (!image || !image.complete || image.naturalWidth <= 0) {
        return;
      }

      const drawCrop = (
        canvas: HTMLCanvasElement | null,
        element: AnalysisRecord["result"]["elements"][number] | undefined,
      ) => {
        if (!canvas || !element) {
          return;
        }

        const cropCtx = canvas.getContext("2d");
        if (!cropCtx) {
          return;
        }

        const [x, y, w, h] = element.bbox;
        cropCtx.clearRect(0, 0, canvas.width, canvas.height);
        if (w <= 0 || h <= 0) {
          return;
        }
        cropCtx.drawImage(image, x, y, w, h, 0, 0, canvas.width, canvas.height);
      };

      record.result.elements.forEach((element, idx) => {
        drawCrop(cropCanvasRefs.current[idx] ?? null, element);
      });

      const focusedElement =
        latestFocusedIdxRef.current === null
          ? undefined
          : record.result.elements[latestFocusedIdxRef.current];
      drawCrop(detailCanvasRef.current, focusedElement);
    });
  }, [cancelScheduledCropDraw]);

  const setCropCanvasRef = useCallback(
    (idx: number, canvas: HTMLCanvasElement | null) => {
      cropCanvasRefs.current[idx] = canvas;
      scheduleCropDraw();
    },
    [scheduleCropDraw],
  );

  const setDetailCanvasRef = useCallback(
    (canvas: HTMLCanvasElement | null) => {
      detailCanvasRef.current = canvas;
      scheduleCropDraw();
    },
    [scheduleCropDraw],
  );

  const clearWorkspaceSelection = useCallback(() => {
    clearSelection();
    pendingEmptyClickRef.current = null;
  }, [clearSelection]);

  const resetInspectionState = useCallback(() => {
    cancelScheduledCropDraw();
    cropCanvasRefs.current = [];
    detailCanvasRef.current = null;
    clearWorkspaceSelection();
    resetWorkspaceView();
    setOverlayMode("all");
  }, [
    cancelScheduledCropDraw,
    clearWorkspaceSelection,
    resetWorkspaceView,
  ]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- reset image inspection state when the active record changes
    resetInspectionState();
  }, [currentRecord?.id, resetInspectionState]);

  useEffect(() => {
    scheduleCropDraw();
  }, [currentRecord, focusedIdx, scheduleCropDraw]);

  useEffect(() => {
    if (!currentRecord || !imageRef.current) {
      return;
    }

    const image = imageRef.current;
    image.addEventListener("load", scheduleCropDraw);
    window.addEventListener("resize", scheduleCropDraw);

    return () => {
      image.removeEventListener("load", scheduleCropDraw);
      window.removeEventListener("resize", scheduleCropDraw);
    };
  }, [currentRecord, scheduleCropDraw]);

  useEffect(() => cancelScheduledCropDraw, [cancelScheduledCropDraw]);

  const handleWorkspaceImageLoad = useCallback(() => {
    updateStageSize();
    scheduleCropDraw();
  }, [scheduleCropDraw, updateStageSize]);

  useEffect(() => {
    if (focusedIdx === null || !currentRecord) {
      trustRequestIdRef.current += 1;
      // eslint-disable-next-line react-hooks/set-state-in-effect -- reset trust inspection state when no element is focused
      setTrustState({
        recordId: currentRecord?.id ?? null,
        focusedIdx: null,
        status: "idle",
        data: null,
      });
      return;
    }

    const element = currentRecord.result.elements[focusedIdx];
    if (!element) return;
    const controller = new AbortController();
    const requestId = trustRequestIdRef.current + 1;
    trustRequestIdRef.current = requestId;

    setTrustState({
      recordId: currentRecord.id,
      focusedIdx,
      status: "loading",
      data: null,
    });

    const isActiveRequest = () =>
      trustRequestIdRef.current === requestId && !controller.signal.aborted;

    getTrust(currentRecord.imageDataUrl, element.bbox, element.class_name, 10, {
      signal: controller.signal,
    })
      .then((trust) => {
        if (isActiveRequest()) {
          setTrustState({
            recordId: currentRecord.id,
            focusedIdx,
            status: "ready",
            data: trust,
          });
        }
      })
      .catch(() => {
        if (isActiveRequest()) {
          setTrustState({
            recordId: currentRecord.id,
            focusedIdx,
            status: "error",
            data: null,
          });
        }
      })

    return () => {
      controller.abort();
    };
  }, [focusedIdx, currentRecord]);

  const startWorkspacePan = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) {
      return;
    }

    const target = event.target;
    if (
      target instanceof Element &&
      target.closest(
        '[data-overlay-region="true"], [data-stage-interactive="true"], button, a, input, select, textarea, [role="button"]',
      )
    ) {
      pendingEmptyClickRef.current = null;
      return;
    }

    pendingEmptyClickRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      canceled: false,
    };

    if (stageViewport.zoom > 1) {
      stageViewport.beginPan(event);
    }
  };

  const moveWorkspacePan = (event: ReactPointerEvent<HTMLDivElement>) => {
    const pending = pendingEmptyClickRef.current;
    if (pending && pending.pointerId === event.pointerId) {
      const moved = Math.hypot(
        event.clientX - pending.startX,
        event.clientY - pending.startY,
      );
      if (moved >= 5) {
        pending.canceled = true;
      }
    }

    stageViewport.movePan(event);
  };

  const stopWorkspacePan = (event: ReactPointerEvent<HTMLDivElement>) => {
    stageViewport.endPan(event);

    const pending = pendingEmptyClickRef.current;
    if (pending && pending.pointerId === event.pointerId) {
      pendingEmptyClickRef.current = null;
      if (!pending.canceled) {
        clearSelection();
      }
    }
  };

  const handleWorkspaceOverlayPointerDown = (
    event: ReactPointerEvent<SVGSVGElement>,
  ) => {
    if (event.button !== 0) {
      return;
    }

    const hitId = getWorkspaceOverlayHit(
      event.currentTarget,
      event.clientX,
      event.clientY,
    );
    const hitIdx = typeof hitId === "number" ? hitId : null;
    if (hitIdx === null) {
      return;
    }

    event.stopPropagation();
    setFocusedIdx(hitIdx);
    setHoveredIdx(hitIdx);
    setHoverSource("image");
  };

  const handleWorkspaceOverlayPointerMove = (
    event: ReactPointerEvent<SVGSVGElement>,
  ) => {
    const hitId = getWorkspaceOverlayHit(
      event.currentTarget,
      event.clientX,
      event.clientY,
    );
    const hitIdx = typeof hitId === "number" ? hitId : null;
    setHoveredIdx(hitIdx);
    setHoverSource(hitIdx === null ? null : "image");
    event.currentTarget.style.cursor =
      hitIdx === null ? (stageViewport.zoom > 1 ? "grab" : "default") : "pointer";
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
  const trustMatchesFocusedElement =
    trustState.recordId === (currentRecord?.id ?? null) &&
    trustState.focusedIdx === focusedIdx;
  const trustData =
    trustMatchesFocusedElement && trustState.status === "ready"
      ? trustState.data
      : null;
  const contextLoading =
    trustMatchesFocusedElement && trustState.status === "loading";

  return {
    imageRef,
    setCropCanvasRef,
    setDetailCanvasRef,
    containerRef: stageViewport.containerRef,
    transformSize: stageViewport.transformSize,
    hoveredIdx,
    hoverSource,
    focusedIdx,
    zoom: stageViewport.zoom,
    panOffset: stageViewport.panOffset,
    isPanning: stageViewport.isPanning,
    overlayMode,
    showLabelNames,
    trustState,
    trustData,
    contextLoading,
    setHoveredIdx,
    setHoverSource,
    setFocusedIdx,
    setOverlayMode,
    setShowLabelNames,
    handleWorkspaceImageLoad,
    zoomIn: stageViewport.zoomIn,
    zoomOut: stageViewport.zoomOut,
    resetWorkspaceView,
    clearWorkspaceSelection,
    startWorkspacePan,
    moveWorkspacePan,
    stopWorkspacePan,
    handleWorkspaceStageWheel: stageViewport.handleStageWheel,
    handleWorkspaceOverlayPointerDown,
    handleWorkspaceOverlayPointerMove,
    handleWorkspaceDetectedListKeyDown,
  };
}
