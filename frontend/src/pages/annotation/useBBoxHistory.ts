import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type Dispatch,
  type SetStateAction,
} from "react";
import type { AnnotationStatus, DetectedElement } from "../../types";
import { cloneElements } from "./useAnnotationRecord";
import type { BboxHistoryEntry } from "./annotationViewportTypes";

type UseBBoxHistoryOptions = {
  resetKey?: string | number | null;
  setElements: Dispatch<SetStateAction<DetectedElement[]>>;
  setAnnotationStatus: Dispatch<SetStateAction<Record<number, AnnotationStatus>>>;
  setFocusedIdx: Dispatch<SetStateAction<number | null>>;
};

export function useBBoxHistory({
  resetKey = null,
  setElements,
  setAnnotationStatus,
  setFocusedIdx,
}: UseBBoxHistoryOptions) {
  const [bboxHistory, setBboxHistory] = useState<BboxHistoryEntry[]>([]);
  const bboxHistoryRef = useRef<BboxHistoryEntry[]>([]);

  const setBboxHistoryEntries = useCallback((nextHistory: BboxHistoryEntry[]) => {
    bboxHistoryRef.current = nextHistory;
    setBboxHistory(nextHistory);
  }, []);

  const pushBboxHistory = useCallback(
    (entry: BboxHistoryEntry) =>
      setBboxHistoryEntries([...bboxHistoryRef.current, entry].slice(-20)),
    [setBboxHistoryEntries],
  );

  const undoLastBboxChange = useCallback(
    (beforeUndo?: () => void) => {
      const previousHistory = bboxHistoryRef.current;
      const entry = previousHistory[previousHistory.length - 1];
      if (!entry) return;

      beforeUndo?.();

      if (entry.type === "create") {
        setElements((prev) => prev.filter((_, idx) => idx !== entry.idx));
        setAnnotationStatus((prev) => {
          const next: Record<number, AnnotationStatus> = {};
          Object.entries(prev).forEach(([key, status]) => {
            const idx = Number(key);
            if (idx < entry.idx) next[idx] = status;
            if (idx > entry.idx) next[idx - 1] = status;
          });
          return next;
        });
        setFocusedIdx(entry.focusedIdx);
      } else if (entry.type === "delete") {
        setElements((prev) => {
          const next = cloneElements(prev);
          next.splice(entry.idx, 0, cloneElements([entry.element])[0]);
          return next;
        });
        setAnnotationStatus((prev) => {
          const next: Record<number, AnnotationStatus> = {};
          Object.entries(prev).forEach(([key, status]) => {
            const idx = Number(key);
            next[idx >= entry.idx ? idx + 1 : idx] = status;
          });
          if (entry.status) next[entry.idx] = entry.status;
          return next;
        });
        setFocusedIdx(entry.idx);
      } else {
        setElements((prev) =>
          prev.map((element, idx) =>
            idx === entry.idx ? { ...element, bbox: [...entry.previousBbox] } : element,
          ),
        );
        setFocusedIdx(entry.idx);
      }

      setBboxHistoryEntries(previousHistory.slice(0, -1));
    },
    [setAnnotationStatus, setBboxHistoryEntries, setElements, setFocusedIdx],
  );

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- reset annotation undo stack when a different record is loaded
    setBboxHistoryEntries([]);
  }, [resetKey, setBboxHistoryEntries]);

  return {
    bboxHistory,
    pushBboxHistory,
    undoLastBboxChange,
  };
}

export default useBBoxHistory;
