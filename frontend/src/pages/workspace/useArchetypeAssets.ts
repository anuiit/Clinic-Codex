import { useEffect, useState } from "react";
import type { SimilarResult } from "../../types";
import { getSimilar } from "../../services/api";
import { isArchetypeGalleryEnabled } from "./archetypeFlag";

export type ArchetypeAssetsState =
  | { status: "disabled" }
  | { status: "loading" }
  | { status: "error" }
  | {
      status: "ready";
      // class_name -> exemplar image path (null when the class has no sample)
      assets: Map<string, string | null>;
      covered: number;
      total: number;
    };

// F1-v0: fetch one exemplar asset per top-K class from /similar (whose
// `asset` field is filled by the backend sample index). No-op when the
// archetype flag is off.
export function useArchetypeAssets(
  recordId: string | null,
  imageDataUrl: string | null,
  bbox: [number, number, number, number] | null,
  classNames: string[],
): ArchetypeAssetsState {
  const [state, setState] = useState<ArchetypeAssetsState>({ status: "disabled" });
  const classKey = classNames.join("");

  useEffect(() => {
    if (!isArchetypeGalleryEnabled() || !recordId || !imageDataUrl || !bbox) {
      setState({ status: "disabled" });
      return;
    }
    const controller = new AbortController();
    setState({ status: "loading" });
    getSimilar(imageDataUrl, bbox, Math.max(classNames.length, 5), {
      signal: controller.signal,
    })
      .then((result: SimilarResult) => {
        if (controller.signal.aborted) return;
        const assets = new Map<string, string | null>();
        for (const item of result.results) {
          assets.set(item.class_name, item.asset);
        }
        const covered = classNames.filter((name) => assets.get(name)).length;
        setState({
          status: "ready",
          assets,
          covered,
          total: classNames.length,
        });
      })
      .catch(() => {
        if (!controller.signal.aborted) {
          setState({ status: "error" });
        }
      });
    return () => controller.abort();
    // classKey captures the class list identity without a deep-compare lint hit.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [recordId, imageDataUrl, bbox, classKey]);

  return state;
}
