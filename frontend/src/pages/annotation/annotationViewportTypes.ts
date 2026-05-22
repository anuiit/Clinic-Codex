import type { Dispatch, SetStateAction } from "react";
import type { AnalysisRecord, AnnotationStatus, DetectedElement } from "../../types";
import type { BBox, BBoxHandle } from "../../utils/segmentationBoxes";

export type StageSize = { width: number; height: number };

export type BboxHistoryEntry =
  | { type: "create"; idx: number; focusedIdx: number | null }
  | {
      type: "delete";
      idx: number;
      element: DetectedElement;
      status?: AnnotationStatus;
      focusedIdx: number | null;
    }
  | {
      type: "update";
      idx: number;
      previousBbox: BBox;
      focusedIdx: number | null;
    };

export type DragState = {
  type: "draw" | "move" | "resize";
  idx: number;
  corner?: BBoxHandle;
  startX: number;
  startY: number;
  startClientX?: number;
  startClientY?: number;
  isMoveReady?: boolean;
  origBbox?: BBox;
} | null;

export type PendingMoveState = {
  type: "move";
  idx: number;
  startX: number;
  startY: number;
  startClientX: number;
  startClientY: number;
  origBbox: BBox;
} | null;

export type UseAnnotationViewportOptions = {
  record: AnalysisRecord | null;
  elements: DetectedElement[];
  setElements: Dispatch<SetStateAction<DetectedElement[]>>;
  annotationStatus: Record<number, AnnotationStatus>;
  setAnnotationStatus: Dispatch<SetStateAction<Record<number, AnnotationStatus>>>;
  focusedIdx: number | null;
  setFocusedIdx: Dispatch<SetStateAction<number | null>>;
  setHoveredIdx: Dispatch<SetStateAction<number | null>>;
  setNamingFocusToken: Dispatch<SetStateAction<number>>;
  loading: boolean;
};

export function areBboxesEqual(
  a: [number, number, number, number],
  b: [number, number, number, number],
): boolean {
  return a.every((value, idx) => value === b[idx]);
}

export function isEditableTarget(target: EventTarget | null): boolean {
  return (
    target instanceof HTMLInputElement ||
    target instanceof HTMLSelectElement ||
    target instanceof HTMLTextAreaElement ||
    (target instanceof HTMLElement && target.isContentEditable)
  );
}
