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
        className="annotation-list-controls ui-section mb-3 flex flex-wrap items-end gap-2 p-3 xl:flex-nowrap"
        data-testid="annotation-list-controls"
      >
        <div
          className="annotation-status-chip annotation-status-chip--validated shrink-0 px-2 py-1.5"
          aria-label={`${labels.submitted} ${submittedCount}/${elementsCount}`}
        >
          {labels.submitted} {submittedCount}/{elementsCount}
        </div>
        <label className="ui-text-eyebrow min-w-[128px] flex-1">
          Filtrer
          <input
            type="search"
            aria-label="Filtrer les éléments"
            value={listQuery}
            onChange={(event) => onListQueryChange(event.target.value)}
            placeholder="Nom ou numéro"
            className="ui-input mt-1 w-full rounded-lg px-2 py-1.5 normal-case tracking-normal"
          />
        </label>
        <label className="ui-text-eyebrow min-w-[112px]">
          Statut
          <select
            value={statusFilter}
            onChange={(event) =>
              onStatusFilterChange(event.target.value as AnnotationStatusFilter)
            }
            className="ui-select mt-1 w-full rounded-lg px-2 py-1.5 normal-case tracking-normal"
          >
            <option value="all">Tous</option>
            <option value="draft">Brouillons</option>
            <option value="submitted">Prêts pour revue</option>
            <option value="rejected">Rejetés</option>
          </select>
        </label>
        <label className="ui-text-eyebrow min-w-[122px]">
          Tri
          <select
            value={sortMode}
            onChange={(event) =>
              onSortModeChange(event.target.value as AnnotationSortMode)
            }
            className="ui-select mt-1 w-full rounded-lg px-2 py-1.5 normal-case tracking-normal"
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
          const progressTone = el.rejected
            ? "ui-progress-value--danger"
            : isSubmitted
              ? "ui-progress-value--ready"
              : "ui-progress-value--accent";

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
              className={`annotation-card flex w-full cursor-pointer items-center justify-between gap-3 rounded-2xl p-3 text-left transition-colors ${isFocused ? "annotation-card-selected" : ""}`}
            >
              <span className="flex min-w-0 items-center gap-3">
                <span
                  className={`annotation-index-badge ${isSubmitted ? "annotation-index-badge--validated" : isFocused ? "annotation-index-badge--focused" : "annotation-index-badge--draft"}`}
                >
                  #{idx}
                </span>
                <span className="min-w-0">
                  <span
                    className={`block truncate text-sm font-bold ${isUnnamedClass(el.class_name) ? "text-[var(--accent)]" : "text-[var(--text-main)]"}`}
                  >
                    {actualDisplayName}
                  </span>
                  <span className="mt-1 flex items-center gap-2">
                    <span className="ui-progress-track inline-block h-1.5 w-20">
                      <span
                        className={`ui-progress-value ${progressTone} block`}
                        style={{
                          width: `${Math.max(0, Math.min(100, confidencePercent))}%`,
                        }}
                      />
                    </span>
                    <span className="ui-text-meta font-semibold tabular-nums">
                      {confidencePercent}%
                    </span>
                  </span>
                </span>
              </span>
              <span
                className={`annotation-status-chip shrink-0 ${el.rejected ? "annotation-status-chip--rejected" : isSubmitted ? "annotation-status-chip--validated" : "annotation-status-chip--draft"}`}
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
