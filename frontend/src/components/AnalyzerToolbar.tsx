import type { ButtonHTMLAttributes, HTMLAttributes } from "react";
import styles from "./AnalyzerToolbar.module.css";

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
        "analyzer-toolbar flex items-center gap-0 rounded-none p-0",
        styles.owner,
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
        "analyzer-toolbar__button inline-flex min-h-9 min-w-9 items-center justify-center gap-2 rounded-none px-3 py-2 text-sm font-semibold transition-colors disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:bg-transparent",
        active ? "analyzer-toolbar__button--active" : "analyzer-toolbar__button--idle",
        className,
      )}
    >
      {children}
    </button>
  );
}

export default AnalyzerToolbar;
