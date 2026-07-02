import type { KeyboardEvent, RefObject } from "react";
import { AlertCircle, ImagePlus, Loader2, Upload } from "lucide-react";
import { ThemeToggle, type ThemeMode } from "../../components/ThemeToggle";
import styles from "./WorkspaceChrome.module.css";

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
  themeMode: ThemeMode;
  onToggleTheme: () => void;
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
  themeMode,
  onToggleTheme,
}: WorkspaceHeaderProps) {
  const handleUploadKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.target !== event.currentTarget) return;
    if (event.key !== "Enter" && event.key !== " ") return;

    event.preventDefault();
    inputRef.current?.click();
  };

  return (
    <section className={`${styles.owner} app-header workspace-header flex shrink-0 flex-col gap-3 rounded-2xl p-3 sm:flex-row sm:items-center sm:justify-between`}>
      <div className="flex items-center gap-3 px-2">
        <div className="app-header__icon flex h-8 w-8 items-center justify-center rounded-lg">
          <ImagePlus size={18} />
        </div>
        <span className="app-header__title font-semibold tracking-tight">
          {labels.appTitle}
        </span>
      </div>

      <div className="flex flex-1 items-center justify-end gap-3">
        <ThemeToggle mode={themeMode} onToggle={onToggleTheme} />
        {error && (
          <div className="ui-alert ui-alert--danger flex items-center gap-2 px-3 py-1.5 text-xs">
            <AlertCircle size={14} className="shrink-0" />
            <span className="max-w-[300px] truncate">{error}</span>
          </div>
        )}

        <div
          className={`flex items-center gap-3 rounded-xl border border-dashed px-4 py-2 transition-colors ${dragging ? "border-[color:var(--border-strong)] bg-[var(--accent-soft)]" : "border-[color:var(--field-border)] bg-[var(--field-bg)] hover:border-[color:var(--border-strong)]"}`}
          onClick={() => inputRef.current?.click()}
          onKeyDown={handleUploadKeyDown}
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
                <span className="ui-text-meta max-w-[120px] truncate font-medium">
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
                className="ui-action-primary ml-2 h-8 gap-2 rounded-lg px-3 text-xs disabled:opacity-50"
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
            <div className="flex items-center gap-2 text-sm text-[color:var(--text-muted)]">
              <Upload size={16} />
              <span>{labels.uploadPrompt}</span>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
