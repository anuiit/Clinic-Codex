import { ArrowLeft, Loader2, Save, Upload } from "lucide-react";
import { Link } from "react-router-dom";
import { appText } from "../i18n/text";

type AnnotationLabels = typeof appText.annotation;

type AnnotationPageChromeProps = {
  labels: AnnotationLabels;
  saving: boolean;
  sending: boolean;
  onSubmitNamed: () => void;
  onSave: () => void;
  onSendSubmittedForReview: () => void;
};

export function AnnotationPageChrome({
  labels,
  saving,
  sending,
  onSubmitNamed,
  onSave,
  onSendSubmittedForReview,
}: AnnotationPageChromeProps) {
  return (
    <div data-testid="annotation-page-chrome">
      <div className="annotation-topbar flex shrink-0 items-center justify-between rounded-xl px-3 py-2">
        <div className="flex min-w-0 items-center gap-4">
          <Link
            to="/"
            className="flex items-center gap-2 rounded-full border border-stone-700/70 bg-stone-950/70 px-3 py-1.5 text-sm font-medium text-stone-300 transition-colors hover:border-amber-500/50 hover:text-stone-50"
          >
            <ArrowLeft size={18} /> {labels.back}
          </Link>
          <div className="min-w-0">
            <div className="text-[10px] font-semibold uppercase tracking-[0.28em] text-amber-300/80">
              Clinic Codex
            </div>
            <h1 className="truncate text-lg font-black tracking-tight text-stone-50">
              {labels.title}
            </h1>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={onSubmitNamed}
            className="rounded-lg border border-emerald-700/60 px-3 py-1.5 text-sm font-semibold text-emerald-200 transition-colors hover:bg-emerald-500/10"
          >
            {labels.submitNamed}
          </button>
          <button
            type="button"
            onClick={onSave}
            disabled={saving}
            className="flex items-center gap-2 rounded-lg bg-amber-500 px-4 py-1.5 text-sm font-semibold text-stone-950 transition-colors hover:bg-amber-400 disabled:opacity-50"
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
            className="flex items-center gap-2 rounded-lg bg-emerald-600 px-4 py-1.5 text-sm font-semibold text-white transition-colors hover:bg-emerald-500 disabled:opacity-50"
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
        className="shrink-0 rounded-xl border border-amber-500/20 bg-amber-500/10 px-3 py-1.5 text-xs text-amber-100 shadow-lg shadow-amber-950/20"
      >
        {labels.adminApprovalNotice}
      </div>
    </div>
  );
}

export default AnnotationPageChrome;
