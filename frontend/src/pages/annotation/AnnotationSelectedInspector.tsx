import type { RefObject } from "react";
import { Trash2 } from "lucide-react";
import type { DetectedElement } from "../../types";
import { appText } from "../../i18n/text";
import { isUnnamedClass } from "../../utils/fuzzyClasses";
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
  onSetElementValidation,
  onRemoveElement,
}: AnnotationSelectedInspectorProps) {
  return (
    <section
      className="annotation-selected-inspector relative mb-3 flex shrink-0 flex-col overflow-visible rounded-xl p-0"
      data-testid="selected-element-inspector"
    >
      <div className="sidebar-header flex items-start justify-between gap-3 px-3 py-2">
        <div className="min-w-0">
          <div className="ui-text-eyebrow">
            Inspecteur
          </div>
          <h2 className="ui-title-md mt-1 truncate text-xl normal-case tracking-tight">
            {focusedElement && focusedIdx !== null
              ? `#${focusedIdx} · ${focusedDisplayName}`
              : "Sélectionnez un élément"}
          </h2>
        </div>
        {focusedElement && focusedIdx !== null && (
          <span
            className={`annotation-status-chip shrink-0 px-2 py-1 ${focusedElement.rejected ? "annotation-status-chip--rejected" : focusedIsSubmitted ? "annotation-status-chip--validated" : "annotation-status-chip--draft"}`}
          >
            {focusedIsSubmitted ? labels.submitted : labels.draft}
          </span>
        )}
      </div>

      <div className="annotation-selected-inspector__body sidebar-body min-h-0 overflow-visible p-3">
        {focusedElement && focusedIdx !== null ? (
          <div className="flex h-full min-h-0 flex-col gap-3">
            <div className="annotation-selected-overview grid grid-cols-[150px_minmax(0,1fr)] gap-3">
              <div className="annotation-crop flex h-[150px] items-center justify-center overflow-hidden rounded-xl border">
                <canvas
                  ref={previewCanvasRef}
                  width={200}
                  height={200}
                  className="block h-[140px] w-[140px] rounded-lg object-contain"
                />
              </div>
              <div className="min-w-0 space-y-3">
                <div>
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
                <div className="grid grid-cols-2 gap-2 text-xs">
                  {(["x", "y", "w", "h"] as const).map((label, coordIdx) => (
                    <div
                      key={label}
                      className="ui-section px-3 py-2"
                    >
                      <div className="ui-text-eyebrow">
                        {label}
                      </div>
                      <div className="mt-1 font-semibold tabular-nums text-[var(--text-main)]">
                        {Math.round(focusedElement.bbox[coordIdx])}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div
              className="annotation-inspector-action-row grid grid-cols-[minmax(0,1fr)_auto_auto] items-end gap-2"
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
              <button
                type="button"
                onClick={() =>
                onSetElementValidation(focusedIdx, !focusedIsSubmitted)
                }
                disabled={isUnnamedClass(focusedElement.class_name)}
                className={`annotation-action-button shrink-0 ${focusedIsSubmitted ? "annotation-action-button--ghost" : "annotation-action-button--success"}`}
              >
                {focusedIsSubmitted ? labels.markDraft : labels.markSubmitted}
              </button>
              <button
                type="button"
                onClick={() => onRemoveElement(focusedIdx)}
                className="annotation-action-button annotation-action-button--danger shrink-0 px-3 py-2"
              >
                <span className="sr-only">
                  Supprimer l’élément #{focusedIdx}
                </span>
                <Trash2 size={18} />
              </button>
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
