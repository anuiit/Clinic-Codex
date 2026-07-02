import type { Dispatch, MutableRefObject, SetStateAction } from "react";
import type { AnnotationStatus, DetectedElement } from "../types";
import { appText } from "../i18n/text";
import { isUnnamedClass } from "../utils/fuzzyClasses";
import { StatusPill, type BadgeTone } from "../components/ui/Primitives";
import annotationStyles from "./annotation/AnnotationChrome.module.css";

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
      className={`${annotationStyles.owner} flex min-h-0 flex-1 flex-col`}
      aria-label="Liste compacte des éléments"
      data-testid="annotation-element-list"
    >
      <div
        className="annotation-list-controls mb-0 border-b border-[color:var(--border-subtle)]"
        data-testid="annotation-list-controls"
      >
        <StatusPill
          tone="ready"
          className="annotation-list-count shrink-0 px-1.5 py-1"
          label={`Compteur prêt pour revue ${submittedCount}/${elementsCount}`}
        >
          Prêts {submittedCount}/{elementsCount}
        </StatusPill>
        <label className="annotation-filter-field annotation-filter-field--search ui-text-eyebrow">
          <span className="annotation-filter-label">Filtrer</span>
          <input
            type="search"
            aria-label="Filtrer les éléments"
            value={listQuery}
            onChange={(event) => onListQueryChange(event.target.value)}
            placeholder="Nom ou #"
            className="ui-input annotation-filter-control w-full rounded-none px-2 py-1 normal-case tracking-normal"
          />
        </label>
        <label className="annotation-filter-field ui-text-eyebrow">
          <span className="annotation-filter-label">Statut</span>
          <select
            value={statusFilter}
            onChange={(event) =>
              onStatusFilterChange(event.target.value as AnnotationStatusFilter)
            }
            className="ui-select annotation-filter-control w-full rounded-none px-2 py-1 normal-case tracking-normal"
          >
            <option value="all">Tous</option>
            <option value="draft">Brouillons</option>
            <option value="submitted">Prêts pour revue</option>
            <option value="rejected">Rejetés</option>
          </select>
        </label>
        <label className="annotation-filter-field ui-text-eyebrow">
          <span className="annotation-filter-label">Tri</span>
          <select
            value={sortMode}
            onChange={(event) =>
              onSortModeChange(event.target.value as AnnotationSortMode)
            }
            className="ui-select annotation-filter-control w-full rounded-none px-2 py-1 normal-case tracking-normal"
          >
            <option value="original">Original</option>
            <option value="confidence-asc">Confiance ↑</option>
            <option value="confidence-desc">Confiance ↓</option>
            <option value="name">Nom A→Z</option>
          </select>
        </label>
      </div>

      <div className="min-h-0 flex-1 space-y-0 overflow-y-auto pr-0">
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
          const badgeTone: BadgeTone = el.rejected
            ? "danger"
            : isSubmitted
              ? "ready"
              : "neutral";

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
              className={`annotation-card flex w-full cursor-pointer items-center justify-between gap-2.5 rounded-none border-b border-[color:var(--border-subtle)] p-2 text-left transition-colors ${isFocused ? "annotation-card-selected" : ""}`}
            >
              <span className="flex min-w-0 items-center gap-2.5">
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
              <StatusPill tone={badgeTone} className="shrink-0 px-1.5 py-0.5 text-[0.66rem]">
                {isSubmitted ? labels.submitted : labels.draft}
              </StatusPill>
            </button>
          );
        })}
      </div>
    </section>
  );
}

export default AnnotationElementList;
