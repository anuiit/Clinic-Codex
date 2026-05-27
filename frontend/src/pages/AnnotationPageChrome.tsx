import { ArrowLeft, Loader2, Save, Upload } from "lucide-react";
import { Link } from "react-router-dom";
import { appText } from "../i18n/text";
import { ThemeToggle, type ThemeMode } from "../components/ThemeToggle";

type AnnotationLabels = typeof appText.annotation;

type AnnotationPageChromeProps = {
  labels: AnnotationLabels;
  imageName?: string | null;
  saving: boolean;
  sending: boolean;
  onSubmitNamed: () => void;
  onSave: () => void;
  onSendSubmittedForReview: () => void;
  themeMode: ThemeMode;
  onToggleTheme: () => void;
};

export function AnnotationPageChrome({
  labels,
  imageName,
  saving,
  sending,
  onSubmitNamed,
  onSave,
  onSendSubmittedForReview,
  themeMode,
  onToggleTheme,
}: AnnotationPageChromeProps) {
  return (
    <div data-testid="annotation-page-chrome">
      <div className="annotation-topbar flex shrink-0 items-center justify-between rounded-2xl px-3 py-2">
        <div className="flex min-w-0 items-center gap-4">
          <Link
            to="/"
            className="ui-action-ghost flex items-center gap-2 rounded-full px-3 py-1.5 text-sm font-medium"
          >
            <ArrowLeft size={18} /> {labels.back}
          </Link>
          <div className="min-w-0">
            <div className="ui-text-eyebrow">
              {labels.title}
            </div>
            <h1 className="ui-title-md truncate text-lg tracking-tight" title={imageName ?? labels.title}>
              {imageName ?? labels.title}
            </h1>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <ThemeToggle mode={themeMode} onToggle={onToggleTheme} />
          <button
            type="button"
            onClick={onSubmitNamed}
            className="annotation-action-button annotation-action-button--ghost"
          >
            {labels.submitNamed}
          </button>
          <button
            type="button"
            onClick={onSave}
            disabled={saving}
            className="annotation-action-button annotation-action-button--primary px-4"
          >
            {saving ? (
              <Loader2 size={18} className="animate-spin" />
            ) : (
              <Save size={18} />
            )}
            {labels.saveChanges}
          </button>
          <button
            type="button"
            onClick={onSendSubmittedForReview}
            disabled={sending}
            className="annotation-action-button annotation-action-button--success px-4"
          >
            {sending ? (
              <Loader2 size={18} className="animate-spin" />
            ) : (
              <Upload size={18} />
            )}
            {labels.sendSubmittedForReview}
          </button>
        </div>
      </div>

      <div
        data-testid="annotation-admin-notice"
        className="ui-alert ui-alert--accent shrink-0 px-3 py-1.5 text-xs"
      >
        {labels.adminApprovalNotice}
      </div>
    </div>
  );
}

export default AnnotationPageChrome;
