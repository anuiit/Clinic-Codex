import type { ButtonHTMLAttributes, ElementType, HTMLAttributes, KeyboardEvent, ReactNode } from "react";

type MetricTone = "neutral" | "accent" | "ready" | "danger";

export type ButtonTone = "neutral" | "primary" | "danger" | "ghost" | "ready";
export type BadgeTone = "neutral" | "ready" | "warning" | "danger";

export type MetricStripItem = {
  label: string;
  value: ReactNode;
  helper?: ReactNode;
  tone?: MetricTone;
};

function cx(...parts: Array<string | false | null | undefined>) {
  return parts.filter(Boolean).join(" ");
}

const controlBase =
  "inline-flex items-center justify-center gap-1.5 rounded-full border text-xs font-medium tracking-[-0.01em] transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[color:var(--accent)] disabled:cursor-not-allowed disabled:opacity-50";

const buttonToneClass: Record<ButtonTone, string> = {
  primary:
    "border-[color:var(--accent)] bg-[color:var(--accent)] text-[color:var(--accent-text)] hover:bg-[color:var(--accent-hover)]",
  neutral:
    "border-[color:var(--border-subtle)] bg-[color:var(--surface-muted)] text-[color:var(--text-heading)] hover:border-[color:var(--border-strong)] hover:bg-[color:var(--surface-hover)]",
  danger:
    "border-[color:var(--danger)] bg-transparent text-[color:var(--danger-text)] hover:bg-[color:var(--danger-soft)]",
  ghost:
    "border-transparent bg-transparent text-[color:var(--text-soft)] hover:border-[color:var(--border-subtle)] hover:bg-[color:var(--surface-hover)] hover:text-[color:var(--text-heading)]",
  ready:
    "border-[color:var(--status-ready-border)] bg-[color:var(--status-ready)] text-[color:var(--status-ready-on-solid)] hover:bg-[color:var(--status-ready-text)]",
};

const badgeToneClass: Record<BadgeTone, string> = {
  neutral:
    "border-[color:var(--border-subtle)] bg-[color:var(--surface-muted)] text-[color:var(--text-soft)]",
  ready:
    "border-[color:var(--status-ready-border)] bg-[color:var(--status-ready-soft)] text-[color:var(--status-ready-text)]",
  warning:
    "border-[color:color-mix(in_srgb,var(--orange)_34%,transparent)] bg-[color:var(--warning-soft)] text-[color:var(--warning-text)]",
  danger:
    "border-[color:color-mix(in_srgb,var(--danger)_34%,transparent)] bg-[color:var(--danger-soft)] text-[color:var(--danger-text)]",
};


export function AdminSection<T extends ElementType = "section">({
  as,
  variant = "flat",
  className = "",
  children,
  ...props
}: HTMLAttributes<HTMLElement> & {
  as?: T;
  variant?: "flat" | "summary";
  children: ReactNode;
}) {
  const Component = (as ?? "section") as ElementType;
  return (
    <Component
      className={cx(
        "admin-flat-section border-0 border-b border-[color:var(--border-subtle)] bg-transparent px-[0.45rem] py-[0.35rem]",
        variant === "summary" && "admin-page-summary",
        className,
      )}
      {...props}
    >
      {children}
    </Component>
  );
}

export function WorkstationMark({ className = "" }: { className?: string }) {
  return <span className={cx("admin-brand-mark", className)} aria-hidden="true" />;
}

export function PillButton({
  active = false,
  className = "",
  children,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  active?: boolean;
}) {
  return (
    <button
      type="button"
      data-active={active ? "true" : "false"}
      className={cx(
        controlBase,
        "h-7 px-2.5",
        active
          ? "border-[color:var(--border-subtle)] bg-[color:var(--surface-muted)] text-[color:var(--text-heading)]"
          : "border-transparent bg-transparent text-[color:var(--text-soft)] hover:border-[color:var(--border-subtle)] hover:bg-[color:var(--surface-hover)] hover:text-[color:var(--text-heading)]",
        className,
      )}
      {...props}
    >
      {children}
    </button>
  );
}

export function ActionButton({
  tone = "neutral",
  className = "",
  children,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  tone?: ButtonTone;
}) {
  return (
    <button
      type="button"
      data-variant={tone}
      className={cx(controlBase, "min-h-9 px-3 py-1.5", buttonToneClass[tone], className)}
      {...props}
    >
      {children}
    </button>
  );
}

export function StatusPill({
  tone = "neutral",
  label,
  children,
  className = "",
}: {
  tone?: BadgeTone;
  label?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <span
      aria-label={label}
      data-tone={tone}
      className={cx(
        "inline-flex min-h-5 items-center justify-center rounded-full border px-2 text-xs font-medium leading-none tracking-[-0.01em]",
        badgeToneClass[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

export function MetricPane({
  label,
  value,
  children,
  testId,
}: {
  label: ReactNode;
  value: ReactNode;
  children: ReactNode;
  testId?: string;
}) {
  return (
    <div className="admin-metric-pane" data-testid={testId}>
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      <div className="chart">{children}</div>
    </div>
  );
}

export function MetricStrip({
  items,
  "aria-label": ariaLabel,
  className = "",
  variant = "bar",
}: {
  items: MetricStripItem[];
  "aria-label": string;
  className?: string;
  variant?: "bar" | "cards";
}) {
  return (
    <section
      aria-label={ariaLabel}
      className={`ui-metric-strip ui-metric-strip--${variant} ${className}`}
    >
      {items.map((item) => (
        <div
          key={item.label}
          className={`ui-metric ui-metric--${item.tone ?? "neutral"}`}
          title={
            typeof item.helper === "string"
              ? `${item.label}: ${item.value} — ${item.helper}`
              : undefined
          }
        >
          <div className="ui-metric__label">{item.label}</div>
          <div className="ui-metric__value">{item.value}</div>
          {item.helper ? (
            <div className="ui-metric__helper">{item.helper}</div>
          ) : null}
        </div>
      ))}
    </section>
  );
}

export type PageTabItem<TId extends string> = {
  id: TId;
  label: string;
  hint?: ReactNode;
};

export function PageTabs<TId extends string>({
  items,
  activeId,
  onSelect,
  ariaLabel,
  panelIdPrefix,
  variant = "underline",
}: {
  items: PageTabItem<TId>[];
  activeId: TId;
  onSelect: (id: TId) => void;
  ariaLabel: string;
  panelIdPrefix: string;
  variant?: "underline" | "pill";
}) {
  const activeIndex = Math.max(
    items.findIndex((item) => item.id === activeId),
    0,
  );
  const selectByOffset = (offset: number) => {
    const nextIndex = (activeIndex + offset + items.length) % items.length;
    onSelect(items[nextIndex].id);
  };
  const handleKeyDown = (event: KeyboardEvent<HTMLButtonElement>) => {
    if (event.key === "ArrowRight") {
      event.preventDefault();
      selectByOffset(1);
    }
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      selectByOffset(-1);
    }
  };

  return (
    <div className={`ui-tabs-shell ui-tabs-shell--${variant}`}>
      <div
        role="tablist"
        aria-label={ariaLabel}
        className={`ui-tabs ui-tabs--${variant}`}
      >
        {items.map((item) => {
          const selected = item.id === activeId;
          return (
            <button
              key={item.id}
              id={`${panelIdPrefix}-${item.id}-tab`}
              type="button"
              role="tab"
              aria-selected={selected}
              aria-controls={`${panelIdPrefix}-${item.id}-panel`}
              aria-label={item.label}
              tabIndex={selected ? 0 : -1}
              className={`ui-tab ${selected ? "ui-tab--active" : ""}`}
              onClick={() => onSelect(item.id)}
              onKeyDown={handleKeyDown}
            >
              <span className="ui-tab-label">
                <b>{item.label}</b>
                {item.hint ? <small>{item.hint}</small> : null}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
