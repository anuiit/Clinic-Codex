import type { RefObject } from "react";
import { AlertCircle, ImagePlus, Loader2, Upload } from "lucide-react";

type WorkspaceHeaderLabels = {
  appTitle: string;
  previewAlt: string;
  uploadPrompt: string;
  analyze: string;
  analyzing: string;
};

type WorkspaceHeaderProps = {
  inputRef: RefObject<HTMLInputElement | null>;
  dragging: boolean;
  preview: string | null;
  file: File | null;
  loading: boolean;
  error: string | null;
  labels: WorkspaceHeaderLabels;
  onFileSelected: (file: File) => void;
  onAnalyze: () => void;
};

export default function WorkspaceHeader({
  inputRef,
  dragging,
  preview,
  file,
  loading,
  error,
  labels,
  onFileSelected,
  onAnalyze,
}: WorkspaceHeaderProps) {
  return (
    <section className="flex shrink-0 flex-col gap-3 rounded-2xl border border-stone-800 bg-stone-900/80 p-3 sm:flex-row sm:items-center sm:justify-between">
      <div className="flex items-center gap-3 px-2">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-amber-500 text-stone-950">
          <ImagePlus size={18} />
        </div>
        <span className="font-semibold tracking-tight text-stone-100">
          {labels.appTitle}
        </span>
      </div>

      <div className="flex flex-1 items-center justify-end gap-3">
        {error && (
          <div className="flex items-center gap-2 rounded-lg bg-red-400/10 px-3 py-1.5 text-xs text-red-300">
            <AlertCircle size={14} className="shrink-0" />
            <span className="max-w-[300px] truncate">{error}</span>
          </div>
        )}

        <div
          className={`flex items-center gap-3 rounded-xl border border-dashed px-4 py-2 transition-colors ${dragging ? "border-amber-400 bg-amber-400/10" : "border-stone-700/80 bg-stone-950/60 hover:border-stone-500"}`}
          onClick={() => inputRef.current?.click()}
          role="button"
          tabIndex={0}
        >
          <input
            ref={inputRef}
            type="file"
            accept="image/png,image/jpeg,image/bmp,image/*"
            className="hidden"
            onChange={(event) => {
              const nextFile = event.target.files?.[0];
              if (nextFile) onFileSelected(nextFile);
            }}
          />
          {preview ? (
            <div className="flex items-center gap-3">
              <img
                src={preview}
                alt={labels.previewAlt}
                className="h-8 w-8 rounded object-cover"
              />
              <div className="flex flex-col">
                <span className="max-w-[120px] truncate text-xs font-medium text-stone-100">
                  {file?.name}
                </span>
              </div>
              <button
                type="button"
                onClick={(event) => {
                  event.stopPropagation();
                  onAnalyze();
                }}
                disabled={loading}
                className="ml-2 inline-flex h-8 items-center justify-center gap-2 rounded-lg bg-amber-500 px-3 text-xs font-semibold text-stone-950 transition-colors hover:bg-amber-400 disabled:opacity-50"
              >
                {loading ? (
                  <>
                    <Loader2 size={14} className="animate-spin" /> {labels.analyzing}
                  </>
                ) : (
                  labels.analyze
                )}
              </button>
            </div>
          ) : (
            <div className="flex items-center gap-2 text-sm text-stone-400">
              <Upload size={16} />
              <span>{labels.uploadPrompt}</span>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
