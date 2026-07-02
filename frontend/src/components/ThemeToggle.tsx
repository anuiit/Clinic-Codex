import { Moon, Sun } from "lucide-react";
import { ActionButton } from "./ui/Primitives";

export type ThemeMode = "dark" | "light";

type ThemeToggleProps = {
  mode: ThemeMode;
  onToggle: () => void;
  className?: string;
};

export function ThemeToggle({ mode, onToggle, className }: ThemeToggleProps) {
  const isLight = mode === "light";
  const label = isLight ? "Activer le mode sombre" : "Activer le mode clair";

  return (
    <ActionButton
      type="button"
      tone="ghost"
      className={["min-h-9 px-2.5 py-1.5", className].filter(Boolean).join(" ")}
      onClick={onToggle}
      aria-label={label}
      title={label}
      data-theme-mode={mode}
    >
      <span className="inline-flex w-4 items-center justify-center" aria-hidden="true">
        {isLight ? <Moon size={16} /> : <Sun size={16} />}
      </span>
      <span className="min-w-12 text-left">{isLight ? "Sombre" : "Clair"}</span>
    </ActionButton>
  );
}

export default ThemeToggle;
