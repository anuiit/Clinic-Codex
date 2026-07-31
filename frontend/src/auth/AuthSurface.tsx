import type { ReactNode } from "react";
import { ThemeToggle, type ThemeMode } from "../components/ThemeToggle";

type AuthSurfaceProps = {
  eyebrow: string;
  title: string;
  subtitle: string;
  children: ReactNode;
  themeMode?: ThemeMode;
  onToggleTheme?: () => void;
};

export function AuthSurface({ eyebrow, title, subtitle, children, themeMode = "dark", onToggleTheme }: AuthSurfaceProps) {
  return <main className="mx-auto flex min-h-screen w-full max-w-md flex-col justify-center gap-6 p-6"><div className="flex justify-end">{onToggleTheme ? <ThemeToggle mode={themeMode} onToggle={onToggleTheme} /> : null}</div><section className="rounded-2xl border border-[color:var(--field-border)] bg-[var(--field-bg)] p-6 shadow-sm"><p className="ui-text-eyebrow">{eyebrow}</p><h1 className="mt-2 text-2xl font-semibold text-[color:var(--text-heading)]">{title}</h1><p className="mt-2 text-sm text-[color:var(--text-muted)]">{subtitle}</p><div className="mt-6">{children}</div></section></main>;
}
