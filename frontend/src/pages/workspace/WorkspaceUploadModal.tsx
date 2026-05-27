import { Loader2 } from "lucide-react";

type WorkspaceUploadModalLabels = {
  uploadModalTitle: string;
  uploadModalDescription: string;
  previewAlt: string;
  cancel: string;
  analyze: string;
  analyzing: string;
};

type WorkspaceUploadModalProps = {
  preview: string;
  file: File;
  loading: boolean;
  labels: WorkspaceUploadModalLabels;
  onAnalyze: () => void;
  onCancel: () => void;
};

export default function WorkspaceUploadModal({
  preview,
  file,
  loading,
  labels,
  onAnalyze,
  onCancel,
}: WorkspaceUploadModalProps) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-[var(--modal-scrim)] p-6 backdrop-blur-sm">
      <div className="ui-panel w-full max-w-xl rounded-[28px] p-5 shadow-[var(--shadow-soft)]">
        <div className="ui-divider flex items-start justify-between gap-4 border-b pb-4">
          <div>
            <p className="ui-text-eyebrow">
              {labels.uploadModalTitle}
            </p>
            <h2 className="ui-title-md mt-1 truncate text-lg">
              {file.name}
            </h2>
            <p className="ui-text-body-sm mt-1">
              {labels.uploadModalDescription}
            </p>
          </div>
          <button
            type="button"
            onClick={onCancel}
            disabled={loading}
            className="ui-action-ghost rounded-xl px-3 py-2 text-sm font-semibold disabled:opacity-50"
          >
            {labels.cancel}
          </button>
        </div>

        <div className="ui-crop-shell my-5 flex max-h-[46vh] items-center justify-center overflow-hidden rounded-2xl">
          <img
            src={preview}
            alt={labels.previewAlt}
            className="max-h-[46vh] max-w-full object-contain"
          />
        </div>

        <div className="flex justify-end gap-3">
          <button
            type="button"
            onClick={onCancel}
            disabled={loading}
            className="ui-action-ghost rounded-xl px-4 py-2 text-sm font-semibold disabled:opacity-50"
          >
            {labels.cancel}
          </button>
          <button
            type="button"
            onClick={onAnalyze}
            disabled={loading}
            className="ui-action-primary inline-flex items-center justify-center gap-2 rounded-xl px-5 py-2 text-sm disabled:opacity-50"
          >
            {loading ? (
              <>
                <Loader2 size={16} className="animate-spin" /> {labels.analyzing}
              </>
            ) : (
              labels.analyze
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
