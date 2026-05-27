import { Info } from "lucide-react";

type WorkspaceEmptyStateProps = {
  title: string;
  details: string;
};

export default function WorkspaceEmptyState({ title, details }: WorkspaceEmptyStateProps) {
  return (
    <div className="ui-empty-state flex min-h-[520px] flex-col items-center justify-center rounded-[28px] px-6 text-center">
      <div className="flex h-16 w-16 items-center justify-center rounded-2xl border border-[color:var(--border-strong)] bg-[var(--accent-soft)] text-[var(--accent)]">
        <Info size={28} />
      </div>
      <h2 className="ui-title-md mt-5 text-xl">{title}</h2>
      <p className="ui-text-body-sm mt-2">{details}</p>
    </div>
  );
}
