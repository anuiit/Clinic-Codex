import { useId, useState } from "react";
import styles from "./ReviewEditor.module.css";
import {
  ActionButton,
  AdminSection,
} from "../../components/ui/AdminPrimitives";
import {
  ClassNameCombobox,
  type ClassNameComboboxLabels,
} from "../../components/ui/ClassNameCombobox";
import { adminAnnotationMediaUrl } from "../../services/api";
import type {
  AdminAnnotationAnalysis,
  AdminAnnotationElement,
  AdminAnnotationModifyPayload,
} from "../../types";
import { normalizeClassName } from "../../utils/fuzzyClasses";
import {
  AdminCorrectionStage,
  type AdminCorrectionBBox,
} from "./AdminCorrectionStage";

const CLASS_NAME_LABELS: ClassNameComboboxLabels = {
  suggestions: "Classes existantes",
  noSuggestion: "Aucune classe correspondante",
  createElementName: "Créer la classe",
  renameElement: "Nom de l'élément",
  nameElement: "Nom de l'élément",
  elementNamePlaceholder: "Rechercher ou créer une classe",
};

function toBboxTuple(values: number[]): AdminCorrectionBBox {
  return [values[0] ?? 0, values[1] ?? 0, values[2] ?? 0, values[3] ?? 0];
}

function sameBbox(left: AdminCorrectionBBox, right: AdminCorrectionBBox) {
  return left.every((value, index) => value === right[index]);
}

function versionedMediaUrl(path: string, fingerprint: string) {
  const url = adminAnnotationMediaUrl(path);
  const separator = url.includes("?") ? "&" : "?";
  return `${url}${separator}v=${encodeURIComponent(fingerprint)}`;
}

export function ElementEditor({
  analysis,
  element,
  classNames,
  mutating,
  onAnnuler,
  onModify,
}: {
  analysis: AdminAnnotationAnalysis;
  element: AdminAnnotationElement;
  classNames: string[];
  mutating: boolean;
  onAnnuler: () => void;
  onModify: (
    element: AdminAnnotationElement,
    payload: AdminAnnotationModifyPayload,
  ) => void;
}) {
  const [className, setClassName] = useState(element.class_name);
  const [bbox, setBbox] = useState<AdminCorrectionBBox>(
    toBboxTuple(element.bbox),
  );
  const [clientError, setClientError] = useState<string | null>(null);
  const [imageSize, setImageSize] = useState<[number, number] | null>(null);
  const previewClipId = `admin-correction-preview-${useId().replace(/:/g, "")}`;
  const originalBbox = toBboxTuple(element.bbox);
  const normalizedClassName = normalizeClassName(className);
  const bboxChanged = !sameBbox(bbox, originalBbox);
  const classChanged = normalizedClassName !== element.class_name.trim();
  const imageUrl = adminAnnotationMediaUrl(analysis.image_url);
  const cropUrl = versionedMediaUrl(
    element.crop_url,
    element.source_fingerprint,
  );
  const contextualClassNames = analysis.elements
    .map((candidate) => candidate.class_name)
    .filter(Boolean);

  const setBboxValue = (position: number, value: string) => {
    const next = [...bbox] as AdminCorrectionBBox;
    next[position] = Number(value);
    setBbox(next);
  };

  const submit = (approveAfterSave: boolean) => {
    setClientError(null);
    if (!normalizedClassName) {
      setClientError("Le nom est requis avant d'enregistrer.");
      return;
    }
    if (bbox.some((value) => !Number.isFinite(value))) {
      setClientError("Les valeurs de zone doivent être des nombres valides.");
      return;
    }
    if (bbox[2] <= 0 || bbox[3] <= 0) {
      setClientError(
        "La largeur et la hauteur de zone doivent être positives.",
      );
      return;
    }

    onModify(element, {
      class_name: normalizedClassName,
      bbox,
      approve_after_save: approveAfterSave || undefined,
    });
  };

  return (
    <AdminSection
      as="form"
      className={`${styles.owner} admin-element-editor`}
      onSubmit={(event) => event.preventDefault()}
    >
      <div className="admin-element-editor__layout">
        <div className="admin-segmentation-editor">
          <AdminCorrectionStage
            analysis={analysis}
            selectedElement={element}
            bbox={bbox}
            onBboxChange={setBbox}
            onImageSizeChange={setImageSize}
            mutating={mutating}
          />

          <aside
            className="admin-segmentation-side"
            aria-label="Paramètres de correction"
          >
            <div className="admin-element-editor__header">
              <ClassNameCombobox
                value={className}
                classNames={classNames}
                customClassNames={contextualClassNames}
                topK={[]}
                autoFocusToken={1}
                labels={CLASS_NAME_LABELS}
                index={element.index}
                inputId={`class-${element.key}`}
                ariaLabel="Nom de l'élément"
                disabled={mutating}
                onInputChange={setClassName}
                onCommit={setClassName}
              />
            </div>

            <fieldset className="admin-bbox-advanced">
              <legend className="ui-text-caption">Zone en pixels</legend>
              {[
                ["x", "X"],
                ["y", "Y"],
                ["w", "Largeur"],
                ["h", "Hauteur"],
              ].map(([key, label], index) => (
                <label key={key} className="admin-bbox-field">
                  <span>{label}</span>
                  <input
                    className="ui-input"
                    type="number"
                    min={index < 2 ? 0 : 1}
                    step="1"
                    value={bbox[index]}
                    disabled={mutating}
                    onChange={(event) =>
                      setBboxValue(index, event.target.value)
                    }
                    aria-label={`Zone ${key} pour l'élément ${element.index}`}
                  />
                </label>
              ))}
            </fieldset>

            <div
              className="admin-correction-context"
              aria-label="Comparaison avant et après correction"
            >
              <figure>
                <figcaption>Découpe actuelle</figcaption>
                {element.crop_exists ? (
                  <img
                    src={cropUrl}
                    alt={`Découpe actuelle ${element.index} pour ${element.class_name}`}
                  />
                ) : (
                  <div className="ui-empty-state">Découpe manquante</div>
                )}
              </figure>
              <figure>
                <figcaption>Aperçu corrigé</figcaption>
                {!bboxChanged && element.crop_exists ? (
                  <img
                    src={cropUrl}
                    alt={`Aperçu corrigé de l'élément ${element.index}`}
                  />
                ) : imageSize ? (
                  <svg
                    viewBox={`${bbox[0]} ${bbox[1]} ${Math.max(1, bbox[2])} ${Math.max(1, bbox[3])}`}
                    preserveAspectRatio="xMidYMid meet"
                    overflow="hidden"
                    role="img"
                    aria-label={`Aperçu corrigé de l'élément ${element.index}`}
                  >
                    <defs>
                      <clipPath
                        id={previewClipId}
                        clipPathUnits="userSpaceOnUse"
                      >
                        <rect
                          data-testid="admin-correction-preview-clip"
                          x={bbox[0]}
                          y={bbox[1]}
                          width={Math.max(1, bbox[2])}
                          height={Math.max(1, bbox[3])}
                        />
                      </clipPath>
                    </defs>
                    <image
                      href={imageUrl}
                      x="0"
                      y="0"
                      width={imageSize[0]}
                      height={imageSize[1]}
                      clipPath={`url(#${previewClipId})`}
                    />
                  </svg>
                ) : (
                  <div className="ui-empty-state">
                    Chargement de l'aperçu…
                  </div>
                )}
              </figure>
            </div>

            <dl
              className="admin-correction-summary"
              aria-label="Résumé des modifications"
            >
              <div>
                <dt>Classe</dt>
                <dd>
                  {classChanged
                    ? `${element.class_name || "Sans nom"} → ${normalizedClassName || "Sans nom"}`
                    : `${element.class_name || "Sans nom"} · inchangée`}
                </dd>
              </div>
              <div>
                <dt>Zone</dt>
                <dd>
                  {bboxChanged
                    ? `[${originalBbox.join(", ")}] → [${bbox.join(", ")}]`
                    : `[${bbox.join(", ")}] · inchangée`}
                </dd>
              </div>
            </dl>

            <p className="admin-correction-help ui-text-caption">
              Enregistrer conserve cette correction ouverte. Enregistrer et
              valider marque directement la nouvelle découpe comme prête.
            </p>

            {clientError ? (
              <div role="alert" className="ui-alert ui-alert--danger p-3">
                {clientError}
              </div>
            ) : null}

            <div className="admin-element-editor__actions">
              <ActionButton
                tone="neutral"
                className="px-3 py-2 text-sm"
                disabled={mutating}
                onClick={() => submit(false)}
              >
                Enregistrer
              </ActionButton>
              <ActionButton
                tone="primary"
                className="px-3 py-2 text-sm"
                disabled={mutating}
                onClick={() => submit(true)}
              >
                Enregistrer et valider
              </ActionButton>
              <ActionButton
                tone="ghost"
                className="px-3 py-2 text-sm"
                disabled={mutating}
                onClick={onAnnuler}
              >
                Annuler
              </ActionButton>
            </div>
          </aside>
        </div>
      </div>
    </AdminSection>
  );
}
