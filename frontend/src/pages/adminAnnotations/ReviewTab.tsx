import styles from "./ReviewTab.module.css";
import { useMemo, useState, type ReactNode } from "react";
import { ActionButton, PillButton } from "../../components/ui/AdminPrimitives";
import { ReferenceThumb } from "./ReferenceGlyphArt";
import { adminAnnotationMediaUrl } from "../../services/api";
import type { AdminAnnotationElement, AdminAnnotationModifyPayload, AdminAnnotationQueue, AdminAnnotationReviewStatus } from "../../types";
import { formatBbox, reviewRowSignal, reviewRows, REVIEW_STATUS_FILTER_LABEL, STATUS_LABEL, type ReviewRow, type ReviewStatusFilter } from "./model";
import { StatusBadge } from "./shared";
import { ReviewElementInspector } from "./ReviewInspector";

function AdminElementRow({
  element,
  selected,
  rowLabel,
  cropAlt,
  detail,
  diagnostic,
  accessibleSummary,
  onSelect,
}: {
  element: AdminAnnotationElement;
  selected: boolean;
  rowLabel: string;
  cropAlt: string;
  detail: ReactNode;
  diagnostic: ReactNode;
  accessibleSummary: string;
  onSelect: (element: AdminAnnotationElement) => void;
}) {
  return (
    <li>
      <button
        type="button"
        role="option"
        aria-selected={selected}
        aria-label={`Ouvrir ${rowLabel}. ${accessibleSummary}${selected ? " (sélectionné)" : ""}`}
        aria-current={selected ? "true" : undefined}
        className={`admin-table-row w-full text-left ${selected ? "admin-table-row--active" : ""}`}
        onClick={() => onSelect(element)}
      >
        <ReferenceThumb>
          {element.crop_exists ? (
            <img
              src={adminAnnotationMediaUrl(element.crop_url)}
              alt={cropAlt}
            />
          ) : (
            <span className="sr-only">Découpe manquante</span>
          )}
        </ReferenceThumb>
        <span className="admin-row-index">#{element.index}</span>
        <span className="admin-row-title">
          {element.class_name || "Sans nom"}
        </span>
        <StatusBadge status={element.review_status} label={rowLabel} />
        <span className="admin-row-detail">{detail}</span>
        <span className="admin-row-diagnostic">{diagnostic}</span>
      </button>
    </li>
  );
}

function ReviewQueueRow({
  row,
  selected,
  onSelect,
}: {
  row: ReviewRow;
  selected: boolean;
  onSelect: (element: AdminAnnotationElement) => void;
}) {
  const { analysis, element } = row;
  const rowLabel = `l'élément ${element.index} ${element.class_name || "Sans nom"} du triage`;
  const cropState = element.crop_exists ? "découpe présente" : "découpe manquante";
  const signal = reviewRowSignal(row);
  const detailSummary = `Statut ${STATUS_LABEL[element.review_status]}. ${analysis.analysis_id}. Zone ${formatBbox(element.bbox)}. ${cropState}. ${signal}.`;
  return (
    <AdminElementRow
      element={element}
      selected={selected}
      rowLabel={rowLabel}
      cropAlt={`Découpe de triage ${element.index} pour ${element.class_name}`}
      detail={<>{element.bbox[2]}×{element.bbox[3]}</>}
      diagnostic={signal}
      accessibleSummary={detailSummary}
      onSelect={onSelect}
    />
  );
}

function ReviewFilterSummary({
  total,
  filtered,
  statusFilter,
  classFilter,
  searchQuery,
  filteredRows,
  onClear,
}: {
  total: number;
  filtered: number;
  statusFilter: ReviewStatusFilter;
  classFilter: string;
  searchQuery: string;
  filteredRows: ReviewRow[];
  onClear: () => void;
}) {
  const hasActiveFilters =
    statusFilter !== "all" ||
    classFilter !== "all" ||
    Boolean(searchQuery.trim());
  const visibleStatusCounts = filteredRows.reduce<
    Record<AdminAnnotationReviewStatus, number>
  >(
    (counts, row) => ({
      ...counts,
      [row.element.review_status]: counts[row.element.review_status] + 1,
    }),
    { pending: 0, approved: 0, rejected: 0 },
  );

  return (
    <section className="admin-list-meta" aria-label="Filtres actifs du triage">
      <div className="admin-list-meta__main">
        <span>
          {filtered} / {total} élément{total === 1 ? "" : "s"} affiché{filtered === 1 ? "" : "s"}.
        </span>
        <span>
          Affiché : {visibleStatusCounts.pending} à vérifier ·{" "}
          {visibleStatusCounts.approved} validé(s) ·{" "}
          {visibleStatusCounts.rejected} rejeté(s)
        </span>
        {hasActiveFilters ? (
          <ul
            className="admin-inline-list"
            aria-label="Filtres de triage appliqués"
          >
            {statusFilter !== "all" ? (
              <li>Statut : {REVIEW_STATUS_FILTER_LABEL[statusFilter]}</li>
            ) : null}
            {classFilter !== "all" ? <li>Classe : {classFilter}</li> : null}
            {searchQuery.trim() ? <li>Recherche : {searchQuery.trim()}</li> : null}
          </ul>
        ) : (
          <span>Aucun filtre : toute la file est visible.</span>
        )}
      </div>
      <ActionButton
        tone="ghost"
        className="min-h-7 px-2.5 py-1"
        disabled={!hasActiveFilters}
        onClick={onClear}
      >
        Effacer les filtres
      </ActionButton>
    </section>
  );
}

export function ReviewTab({
  queue,
  selectedKey,
  mutatingKey,
  editingKey,
  onSelect,
  onReview,
  onModify,
  onEdit,
  onAnnulerEdit,
}: {
  queue: AdminAnnotationQueue;
  selectedKey: string | null;
  mutatingKey: string | null;
  editingKey: string | null;
  onSelect: (element: AdminAnnotationElement) => void;
  onReview: (
    element: AdminAnnotationElement,
    status: AdminAnnotationReviewStatus,
  ) => void;
  onModify: (
    element: AdminAnnotationElement,
    payload: AdminAnnotationModifyPayload,
  ) => void;
  onEdit: (element: AdminAnnotationElement) => void;
  onAnnulerEdit: () => void;
}) {
  const [statusFilter, setStatusFilter] = useState<ReviewStatusFilter>("all");
  const [classFilter, setClassFilter] = useState("all");
  const [searchQuery, setSearchQuery] = useState("");
  const rows = useMemo(() => reviewRows(queue), [queue]);
  const classOptions = useMemo(
    () =>
      [
        ...new Set(rows.map((row) => row.element.class_name || "Sans nom")),
      ].sort(),
    [rows],
  );
  const filteredRows = useMemo(() => {
    const normalizedQuery = searchQuery.trim().toLowerCase();
    return rows.filter((row) => {
      const className = row.element.class_name || "Sans nom";
      const statusMatches =
        statusFilter === "all" || row.element.review_status === statusFilter;
      const classMatches = classFilter === "all" || className === classFilter;
      const queryMatches =
        !normalizedQuery ||
        [
          row.analysis.analysis_id,
          String(row.element.index),
          className,
          formatBbox(row.element.bbox),
          ...row.diagnostics,
        ]
          .join(" ")
          .toLowerCase()
          .includes(normalizedQuery);
      return statusMatches && classMatches && queryMatches;
    });
  }, [classFilter, rows, searchQuery, statusFilter]);

  const clearReviewFilters = () => {
    setStatusFilter("all");
    setClassFilter("all");
    setSearchQuery("");
  };

  if (!rows.length) {
    return (
      <div className="ui-empty-state p-6">
        Aucun élément soumis n'attend une décision.
      </div>
    );
  }

  const selectedRow =
    filteredRows.find((row) => row.element.key === selectedKey) ??
    filteredRows[0] ??
    (selectedKey
      ? rows.find((row) => row.element.key === selectedKey)
      : null) ??
    rows[0] ??
    null;
  const selectedIndex = filteredRows.findIndex(
    (row) => row.element.key === selectedRow?.element.key,
  );

  return (
    <section className={`${styles.owner} admin-split-grid admin-review-workspace`}>
      <div className="admin-list-pane">
        <div className="admin-toolbar" aria-label="Filtres de triage">
          <div className="admin-status-pills" aria-label="Filtrer par statut">
            {Object.entries(REVIEW_STATUS_FILTER_LABEL).map(([value, label]) => (
              <PillButton
                key={value}
                active={statusFilter === value}
                onClick={() => setStatusFilter(value as ReviewStatusFilter)}
              >
                {value === "all" ? "Tous" : label}
              </PillButton>
            ))}
          </div>

          <label className="admin-field admin-native-filter">
            <span>Statut</span>
            <select
              className="ui-select px-2 py-1"
              value={statusFilter}
              onChange={(event) =>
                setStatusFilter(event.target.value as ReviewStatusFilter)
              }
            >
              {Object.entries(REVIEW_STATUS_FILTER_LABEL).map(
                ([value, label]) => (
                  <option key={value} value={value}>
                    {label}
                  </option>
                ),
              )}
            </select>
          </label>
          <label className="admin-field admin-native-filter">
            <span>Classe</span>
            <select
              className="ui-select px-2 py-1"
              value={classFilter}
              onChange={(event) => setClassFilter(event.target.value)}
            >
              <option value="all">Toutes les classes</option>
              {classOptions.map((className) => (
                <option key={className} value={className}>
                  {className}
                </option>
              ))}
            </select>
          </label>
          <label className="admin-field admin-field--grow admin-search-field">
            <span>Rechercher</span>
            <input
              className="ui-input px-2 py-1"
              type="search"
              value={searchQuery}
              onChange={(event) => setSearchQuery(event.target.value)}
              placeholder="classe, #, statut"
            />
          </label>
        </div>

        <ReviewFilterSummary
          total={rows.length}
          filtered={filteredRows.length}
          statusFilter={statusFilter}
          classFilter={classFilter}
          searchQuery={searchQuery}
          filteredRows={filteredRows}
          onClear={clearReviewFilters}
        />

        {filteredRows.length ? (
          <ul
            role="listbox"
            aria-label="File de triage"
            className="admin-list-scroll"
          >
            {filteredRows.map((row) => (
              <ReviewQueueRow
                key={row.element.key}
                row={row}
                selected={row.element.key === selectedRow?.element.key}
                onSelect={onSelect}
              />
            ))}
          </ul>
        ) : (
          <div className="ui-empty-state p-6">
            <p>Aucun élément ne correspond aux filtres actifs.</p>
            <ActionButton
              tone="ghost"
              className="mt-3 px-3 py-2 text-sm"
              onClick={clearReviewFilters}
            >
              Effacer les filtres
            </ActionButton>
          </div>
        )}
      </div>

      <ReviewElementInspector
        row={selectedRow}
        filteredRows={filteredRows}
        selectedIndex={selectedIndex}
        mutating={mutatingKey === selectedRow?.element.key}
        editing={editingKey === selectedRow?.element.key}
        onSelect={onSelect}
        onReview={onReview}
        onModify={onModify}
        onEdit={onEdit}
        onAnnulerEdit={onAnnulerEdit}
      />
    </section>
  );
}
