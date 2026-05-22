import { useEffect, useRef } from "react";
import type { AnalysisRecord, DetectedElement } from "../../types";
import type { DragState } from "./annotationViewportTypes";

export function useAnnotationPreviewCanvas({
  record,
  elements,
  focusedIdx,
  dragState,
  tempBbox,
}: {
  record: AnalysisRecord | null;
  elements: DetectedElement[];
  focusedIdx: number | null;
  dragState: DragState;
  tempBbox: [number, number, number, number] | null;
}) {
  const imageRef = useRef<HTMLImageElement>(null);
  const previewCanvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!record || !imageRef.current || !previewCanvasRef.current || focusedIdx === null) return;
    const el = elements[focusedIdx];
    if (!el) return;

    const [x, y, w, h] = dragState?.idx === focusedIdx && tempBbox ? tempBbox : el.bbox;
    const canvas = previewCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (imageRef.current.complete && w > 0 && h > 0) {
      ctx.imageSmoothingEnabled = false;
      const scale = Math.min(canvas.width / w, canvas.height / h);
      const fittedW = w * scale;
      const fittedH = h * scale;
      const drawX = (canvas.width - fittedW) / 2;
      const drawY = (canvas.height - fittedH) / 2;
      ctx.drawImage(imageRef.current, x, y, w, h, drawX, drawY, fittedW, fittedH);
    }
  }, [record, elements, focusedIdx, tempBbox, dragState]);

  return { imageRef, previewCanvasRef };
}
