import { Loader2 } from "lucide-react";

type WorkspaceUploadModalProps = {
  open: boolean;
  preview: string | null;
  fileName: string | null;
  loading: boolean;
  title: string;
  description: string;
  previewAlt: string;
  cancelLabel: string;
  analyzeLabel: string;
  analyzingLabel: string;
  onClose: () => void;
  onAnalyze: () => void;
};

export function WorkspaceUploadModal({
  open,
  preview,
  fileName,
  loading,
  title,
  description,
  previewAlt,
  cancelLabel,
  analyzeLabel,
  analyzingLabel,
  onClose,
  onAnalyze,
}: WorkspaceUploadModalProps) {
  if (!open || !preview || !fileName) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-stone-950/80 p-6 backdrop-blur-sm">
      <div className="w-full max-w-xl rounded-[28px] border border-stone-800 bg-stone-900 p-5 shadow-[0_30px_90px_rgba(0,0,0,0.45)]">
        <div className="flex items-start justify-between gap-4 border-b border-stone-800 pb-4">
          <div>
            <p className="text-xs uppercase tracking-[0.24em] text-stone-500">
              {title}
            </p>
            <h2 className="mt-1 truncate text-lg font-semibold text-stone-100">
              {fileName}
            </h2>
            <p className="mt-1 text-sm text-stone-400">{description}</p>
          </div>
          <button
            type="button"
            onClick={onClose}
            disabled={loading}
            className="rounded-xl border border-stone-700 px-3 py-2 text-sm font-semibold text-stone-300 transition-colors hover:border-stone-500 hover:text-stone-100 disabled:opacity-50"
          >
            {cancelLabel}
          </button>
        </div>

        <div className="my-5 flex max-h-[46vh] items-center justify-center overflow-hidden rounded-2xl border border-stone-800 bg-stone-950">
          <img
            src={preview}
            alt={previewAlt}
            className="max-h-[46vh] max-w-full object-contain"
          />
        </div>

        <div className="flex justify-end gap-3">
          <button
            type="button"
            onClick={onClose}
            disabled={loading}
            className="rounded-xl border border-stone-700 px-4 py-2 text-sm font-semibold text-stone-300 transition-colors hover:border-stone-500 hover:text-stone-100 disabled:opacity-50"
          >
            {cancelLabel}
          </button>
          <button
            type="button"
            onClick={onAnalyze}
            disabled={loading}
            className="inline-flex items-center justify-center gap-2 rounded-xl bg-amber-500 px-5 py-2 text-sm font-bold text-stone-950 transition-colors hover:bg-amber-400 disabled:opacity-50"
          >
            {loading ? (
              <>
                <Loader2 size={16} className="animate-spin" /> {analyzingLabel}
              </>
            ) : (
              analyzeLabel
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
