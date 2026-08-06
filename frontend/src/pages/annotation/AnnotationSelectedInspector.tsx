import { useEffect, useState, type RefObject } from "react";
import { Trash2 } from "lucide-react";
import type { DetectedElement } from "../../types";
import { appText } from "../../i18n/text";
import { isUnnamedClass } from "../../utils/fuzzyClasses";
import sidebarStyles from "../../components/SidebarChrome.module.css";
import { ActionButton, StatusPill } from "../../components/ui/Primitives";
import annotationStyles from "./AnnotationChrome.module.css";
import { ElementNameCombobox } from "./ElementNameCombobox";

interface AnnotationSelectedInspectorProps {
  focusedElement: DetectedElement | null;
  focusedIdx: number | null;
  focusedDisplayName: string | null;
  focusedConfidencePercent: number;
  focusedIsSubmitted: boolean;
  previewCanvasRef: RefObject<HTMLCanvasElement | null>;
  classes: string[];
  customClasses: string[];
  namingFocusToken: number;
  labels: typeof appText.annotation;
  onCommitElementName: (idx: number, name: string) => void;
  onCommitElementNote: (idx: number, note: string) => void;
  onSetElementValidation: (idx: number, submitted: boolean) => void;
  onRemoveElement: (idx: number) => void;
}

export function AnnotationSelectedInspector({
  focusedElement,
  focusedIdx,
  focusedDisplayName,
  focusedConfidencePercent,
  focusedIsSubmitted,
  previewCanvasRef,
  classes,
  customClasses,
  namingFocusToken,
  labels,
  onCommitElementName,
  onCommitElementNote,
  onSetElementValidation,
  onRemoveElement,
}: AnnotationSelectedInspectorProps) {
  const [noteDraft, setNoteDraft] = useState(focusedElement?.note ?? "");

  useEffect(() => {
    setNoteDraft(focusedElement?.note ?? "");
    // Reset the draft only when the selected element changes, not on every
    // keystroke-driven elements update.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focusedIdx]);

  const commitNote = () => {
    if (focusedIdx === null) return;
    if (noteDraft.trim() === (focusedElement?.note ?? "")) return;
    onCommitElementNote(focusedIdx, noteDraft);
  };

  return (
    <section
      className={`${annotationStyles.owner} ${sidebarStyles.owner} annotation-selected-inspector relative flex shrink-0 flex-col overflow-visible rounded-none p-0`}
      data-testid="selected-element-inspector"
    >
      <div className="sidebar-header flex items-start justify-between gap-3 px-3 py-2">
        <div className="min-w-0">
          <div className="ui-text-eyebrow">
            Inspecteur
          </div>
          <h2 className="ui-title-md mt-0.5 truncate text-base normal-case tracking-tight">
            {focusedElement && focusedIdx !== null
              ? `#${focusedIdx} · ${focusedDisplayName}`
              : "Sélectionnez un élément"}
          </h2>
        </div>
        {focusedElement && focusedIdx !== null && (
          <StatusPill
            tone={focusedElement.rejected ? "danger" : focusedIsSubmitted ? "ready" : "neutral"}
            className="shrink-0 px-2 py-1"
          >
            {focusedIsSubmitted ? labels.submitted : labels.draft}
          </StatusPill>
        )}
      </div>

      <div className="annotation-selected-inspector__body sidebar-body min-h-0 overflow-visible p-0">
        {focusedElement && focusedIdx !== null ? (
          <div className="flex h-full min-h-0 flex-col gap-0">
            <div className="annotation-selected-overview grid grid-cols-[118px_minmax(0,1fr)] gap-0">
              <div className="annotation-crop flex h-[116px] items-center justify-center overflow-hidden rounded-none border-0 border-r border-[color:var(--border-subtle)]">
                <canvas
                  ref={previewCanvasRef}
                  width={200}
                  height={200}
                  className="block h-[108px] w-[108px] rounded-none object-contain"
                />
              </div>
              <div className="annotation-selected-metrics min-w-0">
                <div className="px-3 py-2">
                  <div className="ui-text-meta mb-1 flex items-center justify-between font-semibold">
                    <span>Confiance</span>
                    <span className="tabular-nums text-[var(--text-body)]">
                      {focusedConfidencePercent}%
                    </span>
                  </div>
                  <div className="ui-progress-track h-2">
                    <div
                      className={`ui-progress-value ${focusedElement.rejected ? "ui-progress-value--danger" : focusedIsSubmitted ? "ui-progress-value--ready" : "ui-progress-value--accent"}`}
                      style={{
                        width: `${Math.max(0, Math.min(100, focusedConfidencePercent))}%`,
                      }}
                    />
                  </div>
                </div>
                <dl className="annotation-bbox-meta" aria-label="Coordonnées de segmentation">
                  {(["x", "y", "w", "h"] as const).map((label, coordIdx) => (
                    <div
                      key={label}
                    >
                      <dt>
                        {label}
                      </dt>
                      <dd>
                        {Math.round(focusedElement.bbox[coordIdx])}
                      </dd>
                    </div>
                  ))}
                </dl>
              </div>
            </div>

            <div
              className="annotation-inspector-action-row grid grid-cols-[minmax(0,1fr)_auto_auto] items-end gap-1.5"
              data-testid="annotation-inspector-action-row"
            >
              <div className="min-w-0">
                <ElementNameCombobox
                  value={focusedElement.class_name}
                  classNames={[focusedElement.class_name, ...classes]}
                  customClassNames={customClasses}
                  topK={focusedElement.top_k}
                  autoFocusToken={namingFocusToken}
                  labels={labels}
                  index={focusedIdx}
                  onCommit={(name) => onCommitElementName(focusedIdx, name)}
                />
              </div>
              <ActionButton
                type="button"
                onClick={() =>
                onSetElementValidation(focusedIdx, !focusedIsSubmitted)
                }
                disabled={isUnnamedClass(focusedElement.class_name)}
                tone={focusedIsSubmitted ? "ghost" : "ready"}
                className="shrink-0 px-2.5 py-1.5 text-xs"
              >
                {focusedIsSubmitted ? labels.markDraft : labels.markSubmitted}
              </ActionButton>
              <ActionButton
                type="button"
                onClick={() => onRemoveElement(focusedIdx)}
                tone="danger"
                className="shrink-0 px-2.5 py-1.5"
              >
                <span className="sr-only">
                  Supprimer l’élément #{focusedIdx}
                </span>
                <Trash2 size={18} />
              </ActionButton>
            </div>

            <div className="annotation-inspector-note flex flex-col gap-1 px-3 py-2">
              <label
                htmlFor="annotation-element-note"
                className="ui-text-meta font-semibold"
              >
                {labels.elementNote}
              </label>
              <textarea
                id="annotation-element-note"
                data-testid="annotation-element-note"
                value={noteDraft}
                onChange={(event) => setNoteDraft(event.target.value)}
                onBlur={commitNote}
                placeholder={labels.elementNotePlaceholder}
                rows={2}
                maxLength={2000}
                className="ui-input min-h-[3rem] w-full resize-y px-2 py-1.5 text-sm"
              />
            </div>
          </div>
        ) : (
          <div className="ui-empty-state flex h-full items-center justify-center px-6 text-center">
            {labels.selectElementCrop}
          </div>
        )}
      </div>
    </section>
  );
}
