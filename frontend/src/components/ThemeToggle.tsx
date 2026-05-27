import { Moon, Sun } from "lucide-react";

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
    <button
      type="button"
      className={["theme-toggle", className].filter(Boolean).join(" ")}
      onClick={onToggle}
      aria-label={label}
      title={label}
      data-theme-mode={mode}
    >
      <span className="theme-toggle__icon" aria-hidden="true">
        {isLight ? <Moon size={16} /> : <Sun size={16} />}
      </span>
      <span className="theme-toggle__label">{isLight ? "Sombre" : "Clair"}</span>
    </button>
  );
}

export default ThemeToggle;
