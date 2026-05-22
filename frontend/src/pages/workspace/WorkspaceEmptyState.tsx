import { Info } from "lucide-react";

type WorkspaceEmptyStateProps = {
  title: string;
  details: string;
};

export default function WorkspaceEmptyState({ title, details }: WorkspaceEmptyStateProps) {
  return (
    <div className="flex min-h-[520px] flex-col items-center justify-center rounded-[28px] border border-dashed border-stone-700 bg-stone-900/60 px-6 text-center">
      <div className="flex h-16 w-16 items-center justify-center rounded-2xl border border-amber-400/20 bg-amber-400/10 text-amber-300">
        <Info size={28} />
      </div>
      <h2 className="mt-5 text-xl font-semibold text-stone-100">{title}</h2>
      <p className="mt-2 text-sm text-stone-400">{details}</p>
    </div>
  );
}
