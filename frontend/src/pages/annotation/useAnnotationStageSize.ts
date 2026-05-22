import { useCallback, useEffect, useRef, useState } from "react";
import type { AnalysisRecord } from "../../types";
import type { StageSize } from "./annotationViewportTypes";

export function useAnnotationStageSize(
  record: AnalysisRecord | null,
  loading: boolean,
) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [stageSize, setStageSize] = useState<StageSize | null>(null);

  const getStageSize = useCallback((): StageSize | null => {
    if (!record) return null;
    const [imgW, imgH] = record.result.image_size;
    if (imgW <= 0 || imgH <= 0) return null;

    const rect = containerRef.current?.getBoundingClientRect();
    const availableW = rect?.width && rect.width > 0 ? rect.width : imgW;
    const availableH = rect?.height && rect.height > 0 ? rect.height : imgH;
    const scale = Math.min(1, availableW / imgW, availableH / imgH);

    return {
      width: Math.max(1, Math.round(imgW * scale)),
      height: Math.max(1, Math.round(imgH * scale)),
    };
  }, [record]);

  const updateStageSize = useCallback(() => {
    const nextSize = getStageSize();
    if (!nextSize) {
      setStageSize(null);
      return;
    }

    setStageSize((current) =>
      current?.width === nextSize.width && current?.height === nextSize.height
        ? current
        : nextSize,
    );
  }, [getStageSize]);

  useEffect(() => {
    if (!record) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- clear measured stage when the record disappears
      setStageSize(null);
      return;
    }
    if (loading || !containerRef.current) return;

    updateStageSize();
    const container = containerRef.current;
    const handleResize = () => updateStageSize();
    window.addEventListener("resize", handleResize);

    const observer =
      typeof ResizeObserver === "undefined"
        ? null
        : new ResizeObserver(() => updateStageSize());
    observer?.observe(container);

    return () => {
      window.removeEventListener("resize", handleResize);
      observer?.disconnect();
    };
  }, [loading, record, updateStageSize]);

  const resolvedStageSize =
    stageSize ??
    (record
      ? { width: record.result.image_size[0], height: record.result.image_size[1] }
      : null);

  return { containerRef, stageSize, resolvedStageSize, updateStageSize };
}
