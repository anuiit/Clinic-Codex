import { useEffect, useState, type Dispatch, type MutableRefObject, type SetStateAction } from "react";
import type { AnnotationStatus, DetectedElement } from "../../types";
import {
  hasExactClassName,
  isUnnamedClass,
  normalizeClassName,
} from "../../utils/fuzzyClasses";

type AnnotationStatusFilter = "all" | "draft" | "submitted" | "rejected";
type AnnotationSortMode =
  | "original"
  | "name"
  | "confidence-asc"
  | "confidence-desc";

type UseAnnotationElementModelOptions = {
  elements: DetectedElement[];
  setElements: Dispatch<SetStateAction<DetectedElement[]>>;
  annotationStatus: Record<number, AnnotationStatus>;
  setAnnotationStatus: Dispatch<SetStateAction<Record<number, AnnotationStatus>>>;
  classes: string[];
  setCustomClasses: Dispatch<SetStateAction<string[]>>;
  focusedIdx: number | null;
  cardRefs: MutableRefObject<Array<HTMLElement | null>>;
};

export function useAnnotationElementModel({
  elements,
  setElements,
  annotationStatus,
  setAnnotationStatus,
  classes,
  setCustomClasses,
  focusedIdx,
  cardRefs,
}: UseAnnotationElementModelOptions) {
  const [statusFilter, setStatusFilter] =
    useState<AnnotationStatusFilter>("all");
  const [sortMode, setSortMode] = useState<AnnotationSortMode>("original");
  const [listQuery, setListQuery] = useState("");
  const [namingFocusToken, setNamingFocusToken] = useState(0);

  const commitElementName = (idx: number, nextName: string) => {
    const normalizedName = normalizeClassName(nextName);
    if (!normalizedName) return;
    const previousName = normalizeClassName(elements[idx]?.class_name ?? "");
    if (previousName === normalizedName) return;

    setElements((prev) =>
      prev.map((el, elementIdx) =>
        elementIdx === idx ? { ...el, class_name: normalizedName } : el,
      ),
    );
    setAnnotationStatus((prev) => ({ ...prev, [idx]: "draft" }));

    setCustomClasses((prev) =>
      hasExactClassName(normalizedName, [...classes, ...prev])
        ? prev
        : [...prev, normalizedName],
    );
  };

  const setElementValidation = (idx: number, status: AnnotationStatus) => {
    setAnnotationStatus((prev) => ({ ...prev, [idx]: status }));
  };

  const commitElementNote = (idx: number, note: string) => {
    const trimmed = note.trim();
    setElements((prev) =>
      prev.map((el, elementIdx) => {
        if (elementIdx !== idx) return el;
        if (trimmed) {
          return el.note === trimmed ? el : { ...el, note: trimmed };
        }
        if (el.note === undefined) return el;
        const next = { ...el };
        delete next.note;
        return next;
      }),
    );
  };

  const submitNamedElements = () => {
    setAnnotationStatus((prev) => {
      const next: Record<number, AnnotationStatus> = { ...prev };
      elements.forEach((el, idx) => {
        next[idx] = isUnnamedClass(el.class_name) ? "draft" : "validated";
      });
      return next;
    });
  };

  const submittedCount = elements.filter(
    (el, idx) =>
      annotationStatus[idx] === "validated" && !isUnnamedClass(el.class_name),
  ).length;
  const focusedElement = focusedIdx !== null ? elements[focusedIdx] : null;
  const focusedIsSubmitted =
    focusedIdx !== null && annotationStatus[focusedIdx] === "validated";
  const focusedDisplayName = focusedElement
    ? isUnnamedClass(focusedElement.class_name)
      ? null
      : focusedElement.class_name
    : null;
  const focusedConfidencePercent = focusedElement
    ? Math.round(focusedElement.confidence * 100)
    : 0;

  const displayedElements = elements
    .map((el, idx) => ({ el, idx }))
    .filter(({ el, idx }) => {
      const normalizedQuery = listQuery.trim().toLocaleLowerCase("fr");
      if (
        normalizedQuery &&
        !`${idx} ${el.class_name}`
          .toLocaleLowerCase("fr")
          .includes(normalizedQuery)
      )
        return false;
      if (statusFilter === "all") return true;
      if (statusFilter === "submitted")
        return annotationStatus[idx] === "validated";
      if (statusFilter === "rejected") return Boolean(el.rejected);
      return annotationStatus[idx] !== "validated" && !el.rejected;
    })
    .sort((a, b) => {
      if (sortMode === "name") {
        return (
          a.el.class_name.localeCompare(b.el.class_name, "fr", {
            sensitivity: "base",
          }) || a.idx - b.idx
        );
      }
      if (sortMode === "confidence-asc")
        return a.el.confidence - b.el.confidence || a.idx - b.idx;
      if (sortMode === "confidence-desc")
        return b.el.confidence - a.el.confidence || a.idx - b.idx;
      return a.idx - b.idx;
    });

  useEffect(() => {
    if (focusedIdx === null) return;
    cardRefs.current[focusedIdx]?.scrollIntoView?.({ block: "nearest" });
  }, [cardRefs, focusedIdx, elements.length]);

  return {
    statusFilter,
    sortMode,
    listQuery,
    namingFocusToken,
    submittedCount,
    focusedElement,
    focusedIsSubmitted,
    focusedDisplayName,
    focusedConfidencePercent,
    displayedElements,
    setStatusFilter,
    setSortMode,
    setListQuery,
    setNamingFocusToken,
    commitElementName,
    commitElementNote,
    setElementValidation,
    submitNamedElements,
  };
}
