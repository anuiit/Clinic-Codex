import { useEffect, useRef, useState } from "react";
import { getClasses } from "../../services/api";
import { getAnalysisById } from "../../services/storage";
import type { AnalysisRecord, AnnotationStatus, DetectedElement } from "../../types";

export function cloneElements(elements: DetectedElement[]): DetectedElement[] {
  return elements.map((element) => ({
    ...element,
    bbox: [...element.bbox],
    top_k: element.top_k.map((item) => ({ ...item })),
  }));
}

export function useAnnotationRecord(id: string | undefined, initialFocusedIdx: number | null) {
  const cardRefs = useRef<Array<HTMLElement | null>>([]);
  const [record, setRecord] = useState<AnalysisRecord | null>(null);
  const [elements, setElements] = useState<DetectedElement[]>([]);
  const [annotationStatus, setAnnotationStatus] = useState<
    Record<number, AnnotationStatus>
  >({});
  const [classes, setClasses] = useState<string[]>([]);
  const [customClasses, setCustomClasses] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [focusedIdx, setFocusedIdx] = useState<number | null>(initialFocusedIdx);
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const [listHoveredIdx, setListHoveredIdx] = useState<number | null>(null);

  useEffect(() => {
    let active = true;

    async function loadData() {
      setLoading(true);
      setRecord(null);
      setElements([]);
      setAnnotationStatus({});

      if (!id) {
        if (active) setLoading(false);
        return;
      }

      const rec = await getAnalysisById(id);
      if (!active) return;
      setRecord(rec);

      if (!rec) {
        setLoading(false);
        return;
      }
      setElements(cloneElements(rec.result.elements));
      setAnnotationStatus(rec.annotationStatus ?? {});

      try {
        const classesResult = await getClasses();
        if (active) setClasses(classesResult.class_names);
      } catch {
        // failed to load classes
      } finally {
        if (active) setLoading(false);
      }
    }
    void loadData();
    return () => {
      active = false;
    };
  }, [id]);

  return {
    cardRefs,
    record,
    elements,
    annotationStatus,
    classes,
    customClasses,
    loading,
    focusedIdx,
    hoveredIdx,
    listHoveredIdx,
    setElements,
    setAnnotationStatus,
    setCustomClasses,
    setFocusedIdx,
    setHoveredIdx,
    setListHoveredIdx,
  };
}
