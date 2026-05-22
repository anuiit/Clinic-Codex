import type { ButtonHTMLAttributes, HTMLAttributes } from "react";

type AnalyzerToolbarProps = HTMLAttributes<HTMLDivElement>;

type AnalyzerToolbarButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  active?: boolean;
};

function cx(...values: Array<string | false | null | undefined>) {
  return values.filter(Boolean).join(" ");
}

export function AnalyzerToolbar({
  children,
  className,
  ...props
}: AnalyzerToolbarProps) {
  return (
    <div
      {...props}
      className={cx(
        "analyzer-toolbar flex items-center gap-1 rounded-2xl p-1",
        className,
      )}
    >
      {children}
    </div>
  );
}

export function AnalyzerToolbarButton({
  active,
  className,
  children,
  type = "button",
  ...props
}: AnalyzerToolbarButtonProps) {
  return (
    <button
      {...props}
      type={type}
      className={cx(
        "analyzer-toolbar__button flex items-center gap-2 rounded-xl px-3 py-2 text-sm font-semibold transition-colors disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:bg-transparent",
        active
          ? "bg-amber-400 text-stone-950 shadow-lg shadow-amber-950/30"
          : "text-stone-300 hover:bg-stone-800 hover:text-stone-50",
        className,
      )}
    >
      {children}
    </button>
  );
}

export default AnalyzerToolbar;
