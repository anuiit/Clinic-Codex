import styles from "./DatasetTab.module.css";
import { useMemo, useState } from "react";
import { ActionButton, PillButton } from "../../components/ui/AdminPrimitives";
import { ReferenceTileArt } from "./ReferenceGlyphArt";
import { adminAnnotationMediaUrl } from "../../services/api";
import type { AdminAnnotationElement, AdminAnnotationQueue } from "../../types";
import { DATASET_SPLIT_LABEL, TRAINABLE_DATASET_SPLITS, datasetRows, STATUS_LABEL, type DatasetRow, type DatasetSplitFilter, type TrainableDatasetSplit } from "./model";

type TrainableSplitCounts = Record<TrainableDatasetSplit, number>;

function DatasetFilterSummary({
  total,
  filtered,
  splitFilter,
  classFilter,
  rows,
  onClear,
}: {
  total: number;
  filtered: number;
  splitFilter: DatasetSplitFilter;
  classFilter: string;
  rows: DatasetRow[];
  onClear: () => void;
}) {
  const hasActiveFilters = splitFilter !== "all" || classFilter !== "all";
  const visibleSplitCounts = rows.reduce<TrainableSplitCounts>(
    (counts, row) => ({
      ...counts,
      [row.element.dataset_split as TrainableDatasetSplit]:
        counts[row.element.dataset_split as TrainableDatasetSplit] + 1,
    }),
    { train: 0, val: 0, test: 0 },
  );

  return (
    <section className="admin-list-meta" aria-label="Filtres actifs du dataset">
      <div className="admin-list-meta__main">
        <span>
          {filtered} / {total} élément{total === 1 ? "" : "s"} du dataset affiché{filtered === 1 ? "" : "s"}.
        </span>
        <span>
          Affiché : {visibleSplitCounts.train} train ·{" "}
          {visibleSplitCounts.val} val · {visibleSplitCounts.test} test
        </span>
        {hasActiveFilters ? (
          <ul
            className="admin-inline-list"
            aria-label="Filtres dataset appliqués"
          >
            {splitFilter !== "all" ? (
              <li>Split : {DATASET_SPLIT_LABEL[splitFilter]}</li>
            ) : null}
            {classFilter !== "all" ? <li>Classe : {classFilter}</li> : null}
          </ul>
        ) : (
          <span>
            Aucun filtre : toutes les images validées et utilisables sont visibles.
          </span>
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

export function DatasetTab({
  queue,
  onJumpToReview,
}: {
  queue: AdminAnnotationQueue;
  onJumpToReview: (element: AdminAnnotationElement) => void;
}) {
  const [splitFilter, setSplitFilter] = useState<DatasetSplitFilter>("all");
  const [classFilter, setClassFilter] = useState("all");
  const [selectedDatasetKey, setSelectedDatasetKey] = useState<string | null>(
    null,
  );
  const rows = useMemo(() => datasetRows(queue), [queue]);
  const classCounts = useMemo(() => {
    const counts = new Map<string, number>();
    for (const row of rows) {
      const className = row.element.class_name || "Sans nom";
      counts.set(className, (counts.get(className) ?? 0) + 1);
    }
    return [...counts.entries()].sort(([a], [b]) => a.localeCompare(b));
  }, [rows]);
  const filteredRows = useMemo(
    () =>
      rows.filter((row) => {
        const splitMatches =
          splitFilter === "all" || row.element.dataset_split === splitFilter;
        const classMatches =
          classFilter === "all" ||
          (row.element.class_name || "Sans nom") === classFilter;
        return splitMatches && classMatches;
      }),
    [classFilter, rows, splitFilter],
  );
  const splitCounts = useMemo(
    () =>
      rows.reduce<TrainableSplitCounts>(
        (counts, row) => ({
          ...counts,
          [row.element.dataset_split as TrainableDatasetSplit]:
            counts[row.element.dataset_split as TrainableDatasetSplit] + 1,
        }),
        { train: 0, val: 0, test: 0 },
      ),
    [rows],
  );
  const selectedRow =
    filteredRows.find((row) => row.element.key === selectedDatasetKey) ??
    filteredRows[0] ??
    null;

  const clearDatasetFilters = () => {
    setSplitFilter("all");
    setClassFilter("all");
  };

  return (
    <section className={`${styles.owner} admin-dataset-reference-shell`}>
      <header className="admin-dataset-reference-top">
        <div className="summary">
          <span><strong>{rows.length}</strong> images incluses</span>
          <span><strong>{classCounts.length}</strong> classes</span>
          <span>
            <strong>{splitCounts.train}/{splitCounts.val}/{splitCounts.test}</strong> train/val/test
          </span>
        </div>
        {selectedRow ? (
          <ActionButton
            tone="primary"
            className="min-h-8 px-3 py-1.5"
            onClick={() => onJumpToReview(selectedRow.element)}
          >
            Ouvrir dans le triage
          </ActionButton>
        ) : null}
      </header>

      <div className="dataset-shell-reference">
        <aside className="classes-reference" aria-label="Classes dataset">
          <div className="mini-toolbar-reference">
            <PillButton active={classFilter === "all"} onClick={() => setClassFilter("all")}>Toutes</PillButton>
            <PillButton disabled>Faibles</PillButton>
          </div>
          <div className="class-list-reference">
            {classCounts.map(([className, count]) => (
              <button
                key={className}
                type="button"
                className={`class-row-reference ${classFilter === className ? "active" : ""}`}
                onClick={() => setClassFilter(className)}
              >
                <b>{className}</b>
                <span className="count">{count}</span>
                <span className={`health ${count >= 4 ? "ok" : count >= 2 ? "low" : "bad"}`}>
                  {count >= 4 ? "ok" : count >= 2 ? "faible" : "bas"}
                </span>
              </button>
            ))}
          </div>
        </aside>

        <section className="gallery-area-reference">
          <div className="gallery-toolbar-reference" aria-label="Filtres split dataset">
            {(["all", ...TRAINABLE_DATASET_SPLITS] as DatasetSplitFilter[]).map((value) => (
              <PillButton
                key={value}
                active={splitFilter === value}
                onClick={() => setSplitFilter(value as DatasetSplitFilter)}
              >
                {DATASET_SPLIT_LABEL[value]}
              </PillButton>
            ))}
            <PillButton
              className="ml-auto"
              disabled={splitFilter === "all" && classFilter === "all"}
              onClick={clearDatasetFilters}
            >
              Effacer
            </PillButton>
          </div>

          <DatasetFilterSummary
            total={rows.length}
            filtered={filteredRows.length}
            splitFilter={splitFilter}
            classFilter={classFilter}
            rows={filteredRows}
            onClear={clearDatasetFilters}
          />

          {filteredRows.length ? (
            <div className="gallery-reference" role="listbox" aria-label="Vue dataset">
              {filteredRows.map((row) => {
                const { element } = row;
                const selected = selectedRow?.element.key === element.key;
                return (
                  <button
                    key={element.key}
                    type="button"
                    role="option"
                    aria-selected={selected}
                    aria-label={`Ouvrir élément dataset ${element.index} ${element.class_name || "Sans nom"}. Split ${DATASET_SPLIT_LABEL[element.dataset_split]}. Statut ${STATUS_LABEL[element.review_status]}.`}
                    className={`tile-reference ${selected ? "selected" : ""}`}
                    onClick={() => setSelectedDatasetKey(element.key)}
                    onDoubleClick={() => onJumpToReview(element)}
                  >
                    <ReferenceTileArt>
                      {element.crop_exists ? (
                        <img
                          src={adminAnnotationMediaUrl(element.crop_url)}
                          alt={`Découpe dataset ${element.index} pour ${element.class_name}`}
                        />
                      ) : (
                        <span>Découpe manquante</span>
                      )}
                    </ReferenceTileArt>
                    <span className="tile-foot-reference">
                      <span>#{element.index}</span>
                      <span className={`split ${element.dataset_split}`}>{DATASET_SPLIT_LABEL[element.dataset_split]}</span>
                    </span>
                  </button>
                );
              })}
            </div>
          ) : (
            <div className="ui-empty-state p-6">
              <p>Aucune image validée du dataset ne correspond aux filtres.</p>
              <ActionButton
                tone="ghost"
                className="mt-3 px-3 py-2 text-sm"
                onClick={clearDatasetFilters}
              >
                Effacer les filtres
              </ActionButton>
            </div>
          )}
        </section>
      </div>
    </section>
  );
}
