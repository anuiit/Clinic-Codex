import type { Dispatch, MutableRefObject, SetStateAction } from "react";
import type { AnnotationStatus, DetectedElement } from "../types";
import { appText } from "../i18n/text";
import { isUnnamedClass } from "../utils/fuzzyClasses";

type AnnotationLabels = typeof appText.annotation;
type AnnotationStatusFilter = "all" | "draft" | "submitted" | "rejected";
type AnnotationSortMode =
  | "original"
  | "name"
  | "confidence-asc"
  | "confidence-desc";

type DisplayedAnnotationElement = {
  el: DetectedElement;
  idx: number;
};

type AnnotationElementListProps = {
  displayedElements: DisplayedAnnotationElement[];
  annotationStatus: Record<number, AnnotationStatus>;
  focusedIdx: number | null;
  elementsCount: number;
  submittedCount: number;
  listQuery: string;
  statusFilter: AnnotationStatusFilter;
  sortMode: AnnotationSortMode;
  labels: AnnotationLabels;
  cardRefs: MutableRefObject<Array<HTMLElement | null>>;
  setFocusedIdx: Dispatch<SetStateAction<number | null>>;
  onListQueryChange: (value: string) => void;
  onStatusFilterChange: (value: AnnotationStatusFilter) => void;
  onSortModeChange: (value: AnnotationSortMode) => void;
  setListHoveredIdx: Dispatch<SetStateAction<number | null>>;
};

export function AnnotationElementList({
  displayedElements,
  annotationStatus,
  focusedIdx,
  elementsCount,
  submittedCount,
  listQuery,
  statusFilter,
  sortMode,
  labels,
  cardRefs,
  setFocusedIdx,
  onListQueryChange,
  onStatusFilterChange,
  onSortModeChange,
  setListHoveredIdx,
}: AnnotationElementListProps) {
  return (
    <section
      className="flex min-h-0 flex-1 flex-col"
      aria-label="Liste compacte des éléments"
      data-testid="annotation-element-list"
    >
      <div
        className="annotation-list-controls mb-3 flex flex-wrap items-end gap-2 xl:flex-nowrap"
        data-testid="annotation-list-controls"
      >
        <div
          className="shrink-0 rounded-full border border-emerald-500/20 bg-emerald-500/10 px-3 py-1.5 text-xs font-semibold text-emerald-300"
          aria-label={`${labels.submitted} ${submittedCount}/${elementsCount}`}
        >
          {labels.submitted} {submittedCount}/{elementsCount}
        </div>
        <label className="min-w-[128px] flex-1 text-xs font-semibold uppercase tracking-[0.18em] text-stone-500">
          Filtrer
          <input
            type="search"
            aria-label="Filtrer les éléments"
            value={listQuery}
            onChange={(event) => onListQueryChange(event.target.value)}
            placeholder="Nom ou numéro"
            className="mt-1 w-full rounded-lg border border-stone-700 bg-stone-950 px-2 py-1.5 text-xs normal-case tracking-normal text-stone-100 outline-none placeholder:text-stone-600 focus:border-amber-500"
          />
        </label>
        <label className="min-w-[112px] text-xs font-semibold uppercase tracking-[0.18em] text-stone-500">
          Statut
          <select
            value={statusFilter}
            onChange={(event) =>
              onStatusFilterChange(event.target.value as AnnotationStatusFilter)
            }
            className="mt-1 w-full rounded-lg border border-stone-700 bg-stone-950 px-2 py-1.5 text-xs normal-case tracking-normal text-stone-100 outline-none focus:border-amber-500"
          >
            <option value="all">Tous</option>
            <option value="draft">Brouillons</option>
            <option value="submitted">Soumis</option>
            <option value="rejected">Rejetés</option>
          </select>
        </label>
        <label className="min-w-[122px] text-xs font-semibold uppercase tracking-[0.18em] text-stone-500">
          Tri
          <select
            value={sortMode}
            onChange={(event) =>
              onSortModeChange(event.target.value as AnnotationSortMode)
            }
            className="mt-1 w-full rounded-lg border border-stone-700 bg-stone-950 px-2 py-1.5 text-xs normal-case tracking-normal text-stone-100 outline-none focus:border-amber-500"
          >
            <option value="original">Original</option>
            <option value="confidence-asc">Confiance ↑</option>
            <option value="confidence-desc">Confiance ↓</option>
            <option value="name">Nom A→Z</option>
          </select>
        </label>
      </div>

      <div className="annotation-scrollbar min-h-0 flex-1 space-y-2 overflow-y-auto pr-2">
        {displayedElements.map(({ el, idx }) => {
          const isFocused = idx === focusedIdx;
          const actualDisplayName = isUnnamedClass(el.class_name)
            ? labels.unnamedElement
            : el.class_name;
          const isSubmitted = annotationStatus[idx] === "validated";
          const confidencePercent = Math.round(el.confidence * 100);

          return (
            <button
              key={idx}
              type="button"
              ref={(node) => {
                cardRefs.current[idx] = node;
              }}
              onClick={() => setFocusedIdx(idx)}
              onMouseEnter={() => setListHoveredIdx(idx)}
              onMouseLeave={() =>
                setListHoveredIdx((current) =>
                  current === idx ? null : current,
                )
              }
              onFocus={() => setListHoveredIdx(idx)}
              onBlur={() =>
                setListHoveredIdx((current) =>
                  current === idx ? null : current,
                )
              }
              className={`annotation-card flex w-full cursor-pointer items-center justify-between gap-3 rounded-2xl p-3 text-left transition-all ${isFocused ? "annotation-card-selected" : ""}`}
            >
              <span className="flex min-w-0 items-center gap-3">
                <span
                  className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-xl text-sm font-black ${isSubmitted ? "bg-emerald-400 text-stone-950" : isFocused ? "bg-amber-400 text-stone-950" : "bg-stone-800 text-stone-300"}`}
                >
                  #{idx}
                </span>
                <span className="min-w-0">
                  <span
                    className={`block truncate text-sm font-bold ${isUnnamedClass(el.class_name) ? "text-amber-300" : "text-stone-100"}`}
                  >
                    {actualDisplayName}
                  </span>
                  <span className="mt-1 flex items-center gap-2">
                    <span className="h-1.5 w-20 overflow-hidden rounded-full bg-stone-800">
                      <span
                        className={`block h-full rounded-full ${el.rejected ? "bg-red-400" : isSubmitted ? "bg-emerald-400" : "bg-amber-400"}`}
                        style={{
                          width: `${Math.max(0, Math.min(100, confidencePercent))}%`,
                        }}
                      />
                    </span>
                    <span className="text-[10px] font-semibold tabular-nums text-stone-500">
                      {confidencePercent}%
                    </span>
                  </span>
                </span>
              </span>
              <span
                className={`shrink-0 rounded-full px-2 py-0.5 text-[11px] font-bold ${el.rejected ? "bg-red-500/15 text-red-300" : isSubmitted ? "bg-emerald-500/15 text-emerald-300" : "bg-stone-800 text-stone-400"}`}
              >
                {isSubmitted ? labels.submitted : labels.draft}
              </span>
            </button>
          );
        })}
      </div>
    </section>
  );
}

export default AnnotationElementList;
