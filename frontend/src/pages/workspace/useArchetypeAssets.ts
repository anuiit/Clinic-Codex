import { useEffect, useMemo, useState } from "react";
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
  const enabled = isArchetypeGalleryEnabled();
  const bboxKey = JSON.stringify(bbox);
  const classKey = JSON.stringify(classNames);
  const query = useMemo(() => enabled && recordId && imageDataUrl && bboxKey !== "null" ? {
    recordId, imageDataUrl,
    bbox: JSON.parse(bboxKey) as [number, number, number, number],
    classNames: JSON.parse(classKey) as string[],
  } : null, [enabled, recordId, imageDataUrl, bboxKey, classKey]);
  const [state, setState] = useState<{ query: typeof query; result: ArchetypeAssetsState } | null>(null);

  useEffect(() => {
    if (!query) return;
    const controller = new AbortController();
    getSimilar(query.imageDataUrl, query.bbox, Math.max(query.classNames.length, 5), {
      signal: controller.signal,
    })
      .then((result: SimilarResult) => {
        if (controller.signal.aborted) return;
        const assets = new Map<string, string | null>();
        for (const item of result.results) {
          assets.set(item.class_name, item.asset);
        }
        const covered = query.classNames.filter((name) => assets.get(name)).length;
        setState({
          query,
          result: { status: "ready", assets, covered, total: query.classNames.length },
        });
      })
      .catch(() => {
        if (!controller.signal.aborted) {
          setState({ query, result: { status: "error" } });
        }
      });
    return () => controller.abort();
  }, [query]);

  return !query ? { status: "disabled" } : state?.query === query ? state.result : { status: "loading" };
}
