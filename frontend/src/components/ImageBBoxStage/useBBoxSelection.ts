import { useCallback, useState } from "react";
import type { ImageBBoxStageBoxId } from "./imageBBoxStage.types";

export type BBoxHoverSource = "image" | "list" | null;

export function useBBoxSelection<
  Id extends ImageBBoxStageBoxId = ImageBBoxStageBoxId,
  Source extends string | null = BBoxHoverSource,
>() {
  const [hoveredId, setHoveredId] = useState<Id | null>(null);
  const [hoverSource, setHoverSource] = useState<Source | null>(null);
  const [focusedId, setFocusedId] = useState<Id | null>(null);

  const clearHover = useCallback(() => {
    setHoveredId(null);
    setHoverSource(null);
  }, []);

  const clearFocus = useCallback(() => {
    setFocusedId(null);
  }, []);

  const clearSelection = useCallback(() => {
    clearHover();
    clearFocus();
  }, [clearFocus, clearHover]);

  return {
    hoveredId,
    hoverSource,
    focusedId,
    setHoveredId,
    setHoverSource,
    setFocusedId,
    clearHover,
    clearFocus,
    clearSelection,
  };
}

export default useBBoxSelection;
