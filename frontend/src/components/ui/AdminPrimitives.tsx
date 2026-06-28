import type { KeyboardEvent, ReactNode } from "react";

type MetricTone = "neutral" | "accent" | "ready" | "danger";

export type MetricStripItem = {
  label: string;
  value: ReactNode;
  helper?: ReactNode;
  tone?: MetricTone;
};

export function MetricStrip({
  items,
  "aria-label": ariaLabel,
  className = "",
}: {
  items: MetricStripItem[];
  "aria-label": string;
  className?: string;
}) {
  return (
    <section aria-label={ariaLabel} className={`ui-metric-strip ${className}`}>
      {items.map((item) => (
        <div
          key={item.label}
          className={`ui-metric ui-metric--${item.tone ?? "neutral"}`}
        >
          <div className="ui-text-eyebrow text-[0.65rem]">{item.label}</div>
          <div className="ui-metric__value">{item.value}</div>
          {item.helper ? (
            <div className="mt-1 ui-text-caption">{item.helper}</div>
          ) : null}
        </div>
      ))}
    </section>
  );
}

export type PageTabItem<TId extends string> = {
  id: TId;
  label: string;
};

export function PageTabs<TId extends string>({
  items,
  activeId,
  onSelect,
  ariaLabel,
  panelIdPrefix,
}: {
  items: PageTabItem<TId>[];
  activeId: TId;
  onSelect: (id: TId) => void;
  ariaLabel: string;
  panelIdPrefix: string;
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
    <div className="ui-tabs-shell">
      <div role="tablist" aria-label={ariaLabel} className="ui-tabs">
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
              {item.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}
