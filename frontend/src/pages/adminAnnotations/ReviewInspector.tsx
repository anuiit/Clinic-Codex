import {
  ActionButton,
  AdminSection,
  StatusPill,
} from "../../components/ui/AdminPrimitives";
import type { KeyboardEvent as ReactKeyboardEvent } from "react";
import { adminAnnotationMediaUrl } from "../../services/api";
import type {
  AdminAnnotationElement,
  AdminAnnotationModifyPayload,
  AdminAnnotationReviewStatus,
} from "../../types";
import {
  DATASET_SPLIT_LABEL,
  DATASET_SPLIT_TONE,
  trainabilityCopy,
  type ReviewRow,
} from "./model";
import { StatusBadge } from "./shared";
import { ElementEditor } from "./ReviewEditor";
import { ReferenceDecisionPane } from "./ReferenceGlyphArt";
import { AdminReviewStage } from "./AdminReviewStage";

export function ReviewElementInspector({
  row,
  filteredRows,
  selectedIndex,
  mutating,
  editing,
  onSelect,
  onReview,
  onModify,
  onEdit,
  onAnnulerEdit,
  classNames,
}: {
  row: ReviewRow | null;
  filteredRows: ReviewRow[];
  selectedIndex: number;
  mutating: boolean;
  editing: boolean;
  onSelect: (element: AdminAnnotationElement) => void;
  onReview: (
    element: AdminAnnotationElement,
    status: AdminAnnotationReviewStatus,
  ) => void;
  onModify: (
    element: AdminAnnotationElement,
    payload: AdminAnnotationModifyPayload,
  ) => void;
  classNames: string[];
  onEdit: (element: AdminAnnotationElement) => void;
  onAnnulerEdit: () => void;
}) {
  if (!row) {
    return (
      <aside className="ui-empty-state p-6" aria-label="Inspecteur de décision">
        Sélectionnez un élément de la file pour décider quoi en faire.
      </aside>
    );
  }

  const { analysis, element, diagnostics } = row;
  const canGoPrevious = selectedIndex > 0;
  const canGoNext =
    selectedIndex >= 0 && selectedIndex < filteredRows.length - 1;
  const handleInspectorKeyDown = (
    event: ReactKeyboardEvent<HTMLElement>,
  ) => {
    if (
      editing ||
      selectedIndex < 0 ||
      (event.key !== "ArrowLeft" && event.key !== "ArrowRight")
    ) {
      return;
    }

    const target = event.target;
    if (
      target instanceof HTMLInputElement ||
      target instanceof HTMLTextAreaElement ||
      target instanceof HTMLSelectElement ||
      (target instanceof HTMLElement && target.isContentEditable)
    ) {
      return;
    }

    const nextIndex = selectedIndex + (event.key === "ArrowRight" ? 1 : -1);
    const nextRow = filteredRows[nextIndex];
    if (!nextRow) return;

    event.preventDefault();
    onSelect(nextRow.element);
  };

  return (
    <aside
      className={`admin-inspector min-h-0 ${editing ? "admin-inspector--editing" : ""}`}
      aria-labelledby="review-inspector-heading"
      onKeyDown={handleInspectorKeyDown}
    >
      <div className="admin-inspector-grid">
        <div className="admin-decision-header">
          <div>
            <p className="ui-text-eyebrow">Décision visuelle</p>
            <h2
              id="review-inspector-heading"
              className="text-lg font-semibold text-[color:var(--text-heading)]"
            >
              Élément #{element.index} · {element.class_name || "Sans nom"}
            </h2>
            <p className="mt-1 break-all ui-text-caption">
              Regardez la découpe, puis validez, rejetez ou corrigez.
            </p>
          </div>
          <StatusBadge
            status={element.review_status}
            label={`Élément sélectionné ${element.index} ${element.class_name || "Sans nom"}`}
          />
        </div>

        <div
          className="admin-inspector-flags flex flex-wrap gap-2"
          aria-label="Contexte de l'annotation sélectionnée"
        >
          <StatusPill>
            File {selectedIndex >= 0 ? selectedIndex + 1 : "—"} /{" "}
            {filteredRows.length}
          </StatusPill>
          <StatusPill>
            Zone {element.bbox[2]} × {element.bbox[3]} px
          </StatusPill>
          <StatusPill tone={DATASET_SPLIT_TONE[element.dataset_split]}>
            Split {DATASET_SPLIT_LABEL[element.dataset_split]}
          </StatusPill>
          {element.trainable ? (
            <StatusPill tone="ready">Prêt pour entraînement</StatusPill>
          ) : (
            <StatusPill tone="warning">Pas encore prêt</StatusPill>
          )}
          {element.stale_decision ? (
            <StatusPill tone="warning">Décision à refaire</StatusPill>
          ) : null}
        </div>

        <div className="admin-decision-media">
          <div className="admin-decision-media__context">
            <AdminReviewStage
              analysis={analysis}
              selectedElement={element}
              onSelectElement={onSelect}
            />
          </div>
          <div className="flex min-h-0 flex-col gap-3 overflow-hidden">
            <div className="flex min-h-0 shrink-0 flex-col">
              <div className="mb-1 ui-text-caption">Découpe à décider</div>
              <ReferenceDecisionPane type="crop">
                {element.crop_exists ? (
                  <img
                    src={adminAnnotationMediaUrl(element.crop_url)}
                    alt={`Découpe ${element.index} pour ${element.class_name}`}
                  />
                ) : (
                  <div>Découpe manquante</div>
                )}
              </ReferenceDecisionPane>
            </div>

            <AdminSection
              className="min-h-0 flex-1 overflow-y-auto"
              aria-label="Trainability diagnostics"
            >
              <h3 className="ui-title-sm">Effet sur le dataset</h3>
              <p className="mt-1 ui-text-body-sm">{trainabilityCopy(row)}</p>
              {diagnostics.length ? (
                <ul className="mt-1 list-disc space-y-0.5 pl-5 ui-text-caption">
                  {diagnostics.map((diagnostic) => (
                    <li key={diagnostic}>{diagnostic}</li>
                  ))}
                </ul>
              ) : null}
            </AdminSection>
          </div>
        </div>

        <nav
          className="flex flex-wrap gap-2"
          aria-label="Navigation dans la file"
          hidden
          aria-hidden="true"
        >
          <ActionButton
            tone="ghost"
            className="px-3 py-2 text-sm"
            disabled={!canGoPrevious}
            onClick={() => onSelect(filteredRows[selectedIndex - 1].element)}
          >
            Précédent
          </ActionButton>
          <ActionButton
            tone="ghost"
            className="px-3 py-2 text-sm"
            disabled={!canGoNext}
            onClick={() => onSelect(filteredRows[selectedIndex + 1].element)}
          >
            Suivant
          </ActionButton>
        </nav>

        {selectedIndex < 0 ? (
          <div className="ui-alert ui-alert--accent p-3 text-sm">
            L'élément sélectionné est hors des filtres actifs. Effacez les
            filtres pour naviguer dans la file.
          </div>
        ) : null}

        <AdminSection
          className="admin-decision-actions"
          aria-label="Actions de décision"
        >
          <h3 className="sr-only">Actions de décision</h3>
          <p className="sr-only">
            Valider ajoute l'élément au dataset si sa découpe est utilisable.
            Rejeter l'exclut de l'entraînement tout en gardant une trace.
            Corriger permet de modifier le nom ou la zone avant validation.
          </p>
          <div className="admin-action-bar">
            <div className="admin-key-hints" aria-hidden="true">
              <kbd>←</kbd>
              <kbd>→</kbd>
              <span>navigation</span>
            </div>
            <ActionButton
              tone="ghost"
              className="admin-decision-action admin-prev-action"
              disabled={!canGoPrevious}
              onClick={() => onSelect(filteredRows[selectedIndex - 1].element)}
            >
              Précédent
            </ActionButton>
            <div className="admin-main-actions">
              <ActionButton
                tone="primary"
                className="admin-decision-action"
                disabled={mutating || element.review_status === "approved"}
                title="Valide cet élément pour le dataset si la découpe est utilisable."
                onClick={() => onReview(element, "approved")}
              >
                V · Valider
              </ActionButton>
              <ActionButton
                tone="danger"
                className="admin-decision-action"
                disabled={mutating || element.review_status === "rejected"}
                title="Écarte cet élément de l'entraînement tout en gardant une trace."
                onClick={() => onReview(element, "rejected")}
              >
                R · Rejeter
              </ActionButton>
              <ActionButton
                tone="neutral"
                className="admin-decision-action"
                disabled={mutating}
                title="Corrige le nom ou la zone avant de valider."
                onClick={() => onEdit(element)}
              >
                C · Corriger
              </ActionButton>
            </div>
            <ActionButton
              tone="ghost"
              className="admin-decision-action"
              disabled={!canGoNext}
              onClick={() => onSelect(filteredRows[selectedIndex + 1].element)}
            >
              Suivant
            </ActionButton>
          </div>
        </AdminSection>

        {editing ? (
          <ElementEditor
            key={[element.key, element.class_name, ...element.bbox].join(":")}
            analysis={analysis}
            element={element}
            classNames={classNames}
            mutating={mutating}
            onAnnuler={onAnnulerEdit}
            onModify={onModify}
          />
        ) : null}
      </div>
    </aside>
  );
}
