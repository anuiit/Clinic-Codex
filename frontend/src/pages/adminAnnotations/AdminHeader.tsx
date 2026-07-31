import styles from "./AdminHeader.module.css";
import { useMemo } from "react";
import ThemeToggle from "../../components/ThemeToggle";
import { ActionButton, PageTabs, WorkstationMark } from "../../components/ui/AdminPrimitives";
import type { AdminAnnotationQueue } from "../../types";
import { ADMIN_TABS, type AdminAnnotationsPageProps, type AdminTab, formatTimestamp } from "./model";
import { QueueCounters } from "./shared";

export function AdminHeader({
  themeMode = "dark",
  onToggleTheme,
  queue,
  refreshing,
  refreshDisabled,
  refreshDisabledReason,
  lastRefreshedAt,
  onRefresh,
  activeTab,
  onSelectTab,
  authSlot,
}: AdminAnnotationsPageProps & {
  queue: AdminAnnotationQueue | null;
  refreshing: boolean;
  refreshDisabled: boolean;
  refreshDisabledReason: string;
  lastRefreshedAt: Date | null;
  onRefresh: () => void;
  activeTab: AdminTab;
  onSelectTab: (tab: AdminTab) => void;
}) {
  return (
    <header className={`${styles.owner} admin-command-bar admin-chrome`}>
      <div className="admin-command-main">
        <div className="admin-command-title admin-brand">
          <WorkstationMark />
          <div>
            <h1 className="text-base font-semibold tracking-tight text-[color:var(--text-heading)]">
              Poste de triage
            </h1>
            <span className="admin-station-id">codex-014</span>
          </div>
        </div>
        <AdminTabs activeTab={activeTab} onSelect={onSelectTab} queue={queue} />
        {queue ? <QueueCounters queue={queue} /> : null}
        <div className="admin-command-actions">
          {authSlot}
          <span className="admin-timestamp">
            Actualisé {formatTimestamp(lastRefreshedAt)}
          </span>
          <ActionButton
            tone="ghost"
            className="px-2.5 py-1 text-xs disabled:cursor-not-allowed disabled:opacity-50"
            disabled={refreshDisabled}
            onClick={onRefresh}
            title={refreshDisabled ? refreshDisabledReason : "Actualiser la file"}
          >
            {refreshing ? "Actualisation…" : "Actualiser"}
          </ActionButton>
          {onToggleTheme ? (
            <ThemeToggle
              mode={themeMode}
              onToggle={onToggleTheme}
              className="shrink-0"
            />
          ) : null}
        </div>
      </div>
    </header>
  );
}
function AdminTabs({
  activeTab,
  onSelect,
  queue,
}: {
  activeTab: AdminTab;
  onSelect: (tab: AdminTab) => void;
  queue: AdminAnnotationQueue | null;
}) {
  const tabItems = useMemo(() => {
    if (!queue) {
      return ADMIN_TABS;
    }
    const reviewed = queue.counts.approved + queue.counts.rejected;
    return ADMIN_TABS.map((tab) => ({
      ...tab,
      hint:
        tab.id === "review"
          ? `${reviewed}/${queue.counts.total}`
          : tab.id === "dataset"
            ? queue.counts.trainable
            : queue.counts.trainable > 0
              ? "prêt"
              : "bloqué",
    }));
  }, [queue]);
  return (
    <PageTabs
      items={tabItems}
      activeId={activeTab}
      onSelect={onSelect}
      ariaLabel="Étapes du poste de triage"
      panelIdPrefix="admin"
      variant="underline"
    />
  );
}
