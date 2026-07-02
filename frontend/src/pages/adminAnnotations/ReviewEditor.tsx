import { useRef, useState, type PointerEvent } from "react";
import { ActionButton, AdminSection } from "../../components/ui/AdminPrimitives";
import { adminAnnotationMediaUrl } from "../../services/api";
import type {
  AdminAnnotationAnalysis,
  AdminAnnotationElement,
  AdminAnnotationModifyPayload,
} from "../../types";

function toBboxTuple(values: number[]): [number, number, number, number] {
  return [values[0] ?? 0, values[1] ?? 0, values[2] ?? 0, values[3] ?? 0];
}

export function ElementEditor({
  analysis,
  element,
  mutating,
  onAnnuler,
  onModify,
}: {
  analysis: AdminAnnotationAnalysis;
  element: AdminAnnotationElement;
  mutating: boolean;
  onAnnuler: () => void;
  onModify: (
    element: AdminAnnotationElement,
    payload: AdminAnnotationModifyPayload,
  ) => void;
}) {
  const [className, setClassName] = useState(element.class_name);
  const [bbox, setBbox] = useState<[number, number, number, number]>(
    toBboxTuple(element.bbox),
  );
  const [clientError, setClientError] = useState<string | null>(null);
  const [imageSize, setImageSize] = useState<{ width: number; height: number } | null>(
    null,
  );
  const dragStartRef = useRef<{ x: number; y: number } | null>(null);

  const setBboxValue = (position: number, value: string) => {
    const next = [...bbox] as [number, number, number, number];
    next[position] = Number(value);
    setBbox(next);
  };

  const pointFromPointer = (event: PointerEvent<SVGSVGElement>) => {
    if (!imageSize) {
      return null;
    }

    const rect = event.currentTarget.getBoundingClientRect();
    if (!rect.width || !rect.height) {
      return null;
    }

    return {
      x: Math.max(
        0,
        Math.min(imageSize.width, ((event.clientX - rect.left) / rect.width) * imageSize.width),
      ),
      y: Math.max(
        0,
        Math.min(imageSize.height, ((event.clientY - rect.top) / rect.height) * imageSize.height),
      ),
    };
  };

  const setBboxFromDrag = (
    start: { x: number; y: number },
    point: { x: number; y: number },
  ) => {
    const x = Math.round(Math.min(start.x, point.x));
    const y = Math.round(Math.min(start.y, point.y));
    const w = Math.max(1, Math.round(Math.abs(point.x - start.x)));
    const h = Math.max(1, Math.round(Math.abs(point.y - start.y)));
    setBbox([x, y, w, h]);
  };

  const handleSegmentationPointerDown = (event: PointerEvent<SVGSVGElement>) => {
    if (mutating || !analysis.image_exists || !imageSize) {
      return;
    }

    const point = pointFromPointer(event);
    if (!point) {
      return;
    }

    event.currentTarget.setPointerCapture(event.pointerId);
    dragStartRef.current = point;
    setBbox([Math.round(point.x), Math.round(point.y), 1, 1]);
  };

  const handleSegmentationPointerMove = (event: PointerEvent<SVGSVGElement>) => {
    const start = dragStartRef.current;
    if (!start) {
      return;
    }

    const point = pointFromPointer(event);
    if (point) {
      setBboxFromDrag(start, point);
    }
  };

  const stopSegmentationDrag = (event: PointerEvent<SVGSVGElement>) => {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    dragStartRef.current = null;
  };

  const submit = (approveAfterSave: boolean) => {
    setClientError(null);
    if (!className.trim()) {
      setClientError("Le nom est requis avant d'enregistrer.");
      return;
    }
    if (bbox.some((value) => !Number.isFinite(value))) {
      setClientError("Les valeurs de zone doivent être des nombres valides.");
      return;
    }
    if (bbox[2] <= 0 || bbox[3] <= 0) {
      setClientError("La largeur et la hauteur de zone doivent être positives.");
      return;
    }

    onModify(element, {
      class_name: className,
      bbox,
      approve_after_save: approveAfterSave || undefined,
    });
  };

  return (
    <AdminSection
      as="form"
      className="admin-element-editor"
      onSubmit={(event) => event.preventDefault()}
    >
      <div className="admin-element-editor__layout">
        <div className="admin-element-editor__header">
          <label className="ui-title-sm" htmlFor={`class-${element.key}`}>
            Nom de l'élément
          </label>
          <input
            id={`class-${element.key}`}
            className="ui-input mt-1 w-full px-3 py-2"
            value={className}
            disabled={mutating}
            onChange={(event) => setClassName(event.target.value)}
          />
        </div>

        <div className="admin-segmentation-editor">
          <div
            className="admin-segmentation-stage"
            style={
              imageSize
                ? { aspectRatio: `${imageSize.width} / ${imageSize.height}` }
                : undefined
            }
          >
            {analysis.image_exists ? (
              <img
                src={adminAnnotationMediaUrl(analysis.image_url)}
                alt={`Image à corriger ${analysis.analysis_id}`}
                onLoad={(event) => {
                  setImageSize({
                    width: event.currentTarget.naturalWidth,
                    height: event.currentTarget.naturalHeight,
                  });
                }}
              />
            ) : (
              <div className="ui-empty-state h-full">Image source manquante</div>
            )}
            {imageSize ? (
              <svg
                className="admin-segmentation-overlay"
                viewBox={`0 0 ${imageSize.width} ${imageSize.height}`}
                preserveAspectRatio="none"
                onPointerDown={handleSegmentationPointerDown}
                onPointerMove={handleSegmentationPointerMove}
                onPointerUp={stopSegmentationDrag}
                onPointerCancel={stopSegmentationDrag}
                aria-label={`Redessiner la segmentation de l'élément ${element.index}`}
              >
                <rect
                  className="admin-segmentation-rect"
                  x={bbox[0]}
                  y={bbox[1]}
                  width={Math.max(1, bbox[2])}
                  height={Math.max(1, bbox[3])}
                />
                {[
                  [bbox[0], bbox[1]],
                  [bbox[0] + bbox[2], bbox[1]],
                  [bbox[0], bbox[1] + bbox[3]],
                  [bbox[0] + bbox[2], bbox[1] + bbox[3]],
                ].map(([x, y], index) => (
                  <rect
                    key={index}
                    className="admin-segmentation-handle"
                    x={x - 6}
                    y={y - 6}
                    width={12}
                    height={12}
                  />
                ))}
              </svg>
            ) : null}
          </div>

          <div className="admin-segmentation-side">
            <div>
              <h3 className="ui-title-sm">Correction visuelle</h3>
              <p className="mt-1 ui-text-caption">
                Glissez sur l'image complète pour redessiner la zone. Les champs
                restent disponibles pour une retouche fine.
              </p>
            </div>
            <fieldset className="admin-bbox-advanced">
              <legend className="ui-text-caption">Coordonnées avancées</legend>
              {["x", "y", "w", "h"].map((label, index) => (
                <label
                  key={label}
                  className="ui-text-caption flex flex-col gap-1 uppercase tracking-wide"
                >
                  {label}
                  <input
                    className="ui-input"
                    type="number"
                    value={bbox[index]}
                    disabled={mutating}
                    onChange={(event) => setBboxValue(index, event.target.value)}
                    aria-label={`Zone ${label} pour l'élément ${element.index}`}
                  />
                </label>
              ))}
            </fieldset>
          </div>
        </div>

        <p className="ui-text-caption">
          L'enregistrement régénère la découpe. « Enregistrer » remet l'élément à vérifier ; « Enregistrer et valider » le marque prêt une fois la découpe créée.
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
      </div>
    </AdminSection>
  );
}
