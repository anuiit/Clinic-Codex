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
      className="annotation-selected-inspector mb-3 flex shrink-0 flex-col gap-3 overflow-hidden rounded-2xl p-3"
      data-testid="selected-element-inspector"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="text-xs font-semibold uppercase tracking-[0.28em] text-amber-300/80">
            Inspecteur
          </div>
          <h2 className="mt-1 truncate text-xl font-black text-stone-50">
            {focusedElement && focusedIdx !== null
              ? `#${focusedIdx} · ${focusedDisplayName}`
              : "Sélectionnez un élément"}
          </h2>
        </div>
        {focusedElement && focusedIdx !== null && (
          <span
            className={`shrink-0 rounded-full px-3 py-1 text-xs font-bold ${focusedElement.rejected ? "bg-red-500/15 text-red-300" : focusedIsSubmitted ? "bg-emerald-500/15 text-emerald-300" : "bg-stone-800 text-stone-400"}`}
          >
            {focusedIsSubmitted ? labels.submitted : labels.draft}
          </span>
        )}
      </div>

      <div className="min-h-0 flex-1 overflow-hidden">
        {focusedElement && focusedIdx !== null ? (
          <div className="flex h-full min-h-0 flex-col gap-3">
            <div className="annotation-selected-overview grid grid-cols-[150px_minmax(0,1fr)] gap-3">
              <div className="annotation-crop flex h-[150px] items-center justify-center overflow-hidden rounded-xl border border-stone-700/35">
                <canvas
                  ref={previewCanvasRef}
                  width={200}
                  height={200}
                  className="block h-[140px] w-[140px] rounded-lg object-contain"
                />
              </div>
              <div className="min-w-0 space-y-3">
                <div>
                  <div className="mb-1 flex items-center justify-between text-xs font-semibold text-stone-400">
                    <span>Confiance</span>
                    <span className="tabular-nums text-stone-200">
                      {focusedConfidencePercent}%
                    </span>
                  </div>
                  <div className="h-2 overflow-hidden rounded-full bg-stone-800">
                    <div
                      className={`h-full rounded-full ${focusedElement.rejected ? "bg-red-400" : focusedIsSubmitted ? "bg-emerald-400" : "bg-amber-400"}`}
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
                      className="rounded-xl border border-stone-700/60 bg-stone-950/50 px-3 py-2"
                    >
                      <div className="uppercase tracking-[0.18em] text-stone-500">
                        {label}
                      </div>
                      <div className="mt-1 font-semibold tabular-nums text-stone-100">
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
                className={`shrink-0 rounded-xl px-3 py-2 text-sm font-bold transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${focusedIsSubmitted ? "border border-stone-700 bg-stone-900 text-stone-200 hover:bg-stone-800" : "bg-emerald-500 text-stone-950 hover:bg-emerald-400"}`}
              >
                {focusedIsSubmitted ? labels.markDraft : labels.markSubmitted}
              </button>
              <button
                type="button"
                onClick={() => onRemoveElement(focusedIdx)}
                className="shrink-0 rounded-xl border border-red-500/30 px-3 py-2 text-sm font-bold text-red-300 transition-colors hover:bg-red-500/10"
              >
                <span className="sr-only">
                  Supprimer l’élément #{focusedIdx}
                </span>
                <Trash2 size={18} />
              </button>
            </div>
          </div>
        ) : (
          <div className="flex h-full items-center justify-center rounded-xl border border-dashed border-stone-700/60 bg-stone-950/20 px-6 text-center text-sm text-stone-500">
            {labels.selectElementCrop}
          </div>
        )}
      </div>
    </section>
  );
}
