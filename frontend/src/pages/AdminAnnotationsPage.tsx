import { useCallback, useEffect, useState } from "react";
import sharedStyles from "./adminAnnotations/AdminShared.module.css";
import "./AdminAnnotationsPage.css";
import { AdminHeader } from "./adminAnnotations/AdminHeader";
import { DatasetTab } from "./adminAnnotations/DatasetTab";
import { ReviewTab } from "./adminAnnotations/ReviewTab";
import { TrainingTab } from "./adminAnnotations/TrainingTab";
import { PanelSkeleton } from "./adminAnnotations/shared";
import { ADMIN_QUEUE_AUTO_REFRESH_MS, ADMIN_TABS, STATUS_LABEL, type AdminAnnotationsPageProps, type AdminTab } from "./adminAnnotations/model";
import {
  getAdminAnnotationQueue,
  getClasses,
  modifyAdminAnnotationElement,
  setAdminAnnotationReviewStatus,
} from "../services/api";
import type {
  AdminAnnotationElement,
  AdminAnnotationModifyPayload,
  AdminAnnotationQueue,
  AdminAnnotationReviewStatus,
} from "../types";

function AdminAnnotationsPage({
  themeMode = "dark",
  onToggleTheme,
  initialTab = "review",
  onNavigateTab,
  authSlot,
}: AdminAnnotationsPageProps) {
  const [queue, setQueue] = useState<AdminAnnotationQueue | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [actionMessage, setActionMessage] = useState<string | null>(null);
  const [mutatingKey, setMutatingKey] = useState<string | null>(null);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [editingKey, setEditingKey] = useState<string | null>(null);
  const [classNames, setClassNames] = useState<string[]>([]);
  const [localActiveTab, setLocalActiveTab] = useState<AdminTab>(initialTab);
  const activeTab = onNavigateTab ? initialTab : localActiveTab;
  const [queueRefreshing, setQueueRefreshing] = useState(false);
  const [lastQueueRefreshAt, setLastQueueRefreshAt] = useState<Date | null>(
    null,
  );

  const syncQueue = useCallback(async () => {
    const nextQueue = await getAdminAnnotationQueue();
    setQueue(nextQueue);
    setLastQueueRefreshAt(new Date());
    return nextQueue;
  }, []);

  const handleSelectTab = useCallback(
    (tab: AdminTab) => {
      if (onNavigateTab) {
        onNavigateTab(tab);
        return;
      }
      setLocalActiveTab(tab);
    },
    [onNavigateTab],
  );

  const loadQueue = useCallback(
    async ({ showLoading = true }: { showLoading?: boolean } = {}) => {
      if (showLoading) {
        setLoading(true);
      } else {
        setQueueRefreshing(true);
      }
      setLoadError(null);
      try {
        await syncQueue();
      } catch {
        setLoadError("Impossible de charger la file locale de triage.");
      } finally {
        if (showLoading) {
          setLoading(false);
        } else {
          setQueueRefreshing(false);
        }
      }
    },
    [syncQueue],
  );

  useEffect(() => {
    const timeout = window.setTimeout(() => {
      void loadQueue();
    }, 0);
    return () => window.clearTimeout(timeout);
  }, [loadQueue]);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const classes = await getClasses();
        if (!cancelled) {
          setClassNames(classes.class_names);
        }
      } catch {
        if (!cancelled) {
          setClassNames([]);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  const canRefreshQueue =
    !loading && !queueRefreshing && !mutatingKey && !editingKey;
  const refreshDisabledReason = editingKey
    ? "Terminez ou annulez la correction avant d'actualiser la file."
    : mutatingKey
      ? "Attendez la fin de l'action de triage avant d'actualiser."
      : loading || queueRefreshing
        ? "Actualisation déjà en cours."
        : "Actualiser la file";

  const handleManualRefresh = useCallback(() => {
    if (!canRefreshQueue) {
      return;
    }
    void loadQueue({ showLoading: false });
  }, [canRefreshQueue, loadQueue]);

  useEffect(() => {
    const interval = window.setInterval(() => {
      if (document.visibilityState === "hidden" || !canRefreshQueue) {
        return;
      }
      void loadQueue({ showLoading: false });
    }, ADMIN_QUEUE_AUTO_REFRESH_MS);
    return () => window.clearInterval(interval);
  }, [canRefreshQueue, loadQueue]);

  const handleSelectElement = useCallback((element: AdminAnnotationElement) => {
    setSelectedKey(element.key);
    setEditingKey((current) => (current === element.key ? current : null));
  }, []);

  const handleReview = async (
    element: AdminAnnotationElement,
    status: AdminAnnotationReviewStatus,
  ) => {
    setActionError(null);
    setActionMessage(null);
    setSelectedKey(element.key);
    setMutatingKey(element.key);
    try {
      await setAdminAnnotationReviewStatus(
        element.analysis_id,
        element.index,
        status,
      );
      try {
        await syncQueue();
        setActionMessage(
          `Élément ${element.index} marqué comme ${STATUS_LABEL[status]}.`,
        );
      } catch {
        setActionMessage(
          `Élément ${element.index} marqué comme ${STATUS_LABEL[status]}, mais la file n'a pas pu être actualisée. Utilisez Actualiser pour resynchroniser.`,
        );
      }
    } catch {
      setActionError(
        `Impossible de marquer l'élément ${element.index} comme ${STATUS_LABEL[status]}. Le statut visible n'a pas été modifié.`,
      );
    } finally {
      setMutatingKey(null);
    }
  };

  const handleModify = async (
    element: AdminAnnotationElement,
    payload: AdminAnnotationModifyPayload,
  ) => {
    setActionError(null);
    setActionMessage(null);
    setSelectedKey(element.key);
    setMutatingKey(element.key);
    try {
      await modifyAdminAnnotationElement(
        element.analysis_id,
        element.index,
        payload,
      );
      const successMessage = payload.approve_after_save
        ? `Élément ${element.index} enregistré et validé.`
        : `Changements de l'élément ${element.index} enregistrés.`;
      try {
        await syncQueue();
        setActionMessage(successMessage);
      } catch {
        setActionMessage(
          `${successMessage} La file n'a pas pu être actualisée ; utilisez Actualiser pour resynchroniser.`,
        );
      }
      if (payload.approve_after_save) {
        setEditingKey(null);
      }
    } catch {
      setActionError(
        `Impossible d'enregistrer les changements de l'élément ${element.index}. L'annotation visible n'a pas été modifiée.`,
      );
    } finally {
      setMutatingKey(null);
    }
  };

  const queueElementKeys =
    queue?.analyses.flatMap((analysis) =>
      analysis.elements.map((element) => element.key),
    ) ?? [];
  const effectiveSelectedKey =
    selectedKey && queueElementKeys.includes(selectedKey)
      ? selectedKey
      : (queueElementKeys[0] ?? null);
  const effectiveEditingKey =
    editingKey && queueElementKeys.includes(editingKey) ? editingKey : null;

  return (
    <div
      className={`${sharedStyles.owner} admin-console flex h-full min-h-0 flex-col overflow-hidden`}
      data-theme={themeMode}
    >
      <AdminHeader
        themeMode={themeMode}
        onToggleTheme={onToggleTheme}
        queue={queue}
        refreshing={queueRefreshing}
        refreshDisabled={!canRefreshQueue}
        refreshDisabledReason={refreshDisabledReason}
        lastRefreshedAt={lastQueueRefreshAt}
        onRefresh={handleManualRefresh}
        activeTab={activeTab}
        onSelectTab={handleSelectTab}
        authSlot={authSlot}
      />

      <div className="admin-alert-stack">
        {loading ? <PanelSkeleton label="Chargement de la file de triage" /> : null}

        {loadError ? (
          <div role="alert" className="ui-alert ui-alert--danger p-3">
            {loadError}
          </div>
        ) : null}

        {actionError ? (
          <div role="alert" className="ui-alert ui-alert--danger p-3">
            {actionError}
          </div>
        ) : null}

        {actionMessage ? (
          <div role="status" className="ui-alert ui-alert--success p-3">
            {actionMessage}
          </div>
        ) : null}

      </div>

      {queue ? (
        <div
          className={`admin-tab-content ${activeTab === "review" ? "admin-tab-content--review" : ""}`}
        >
          {ADMIN_TABS.map((tab) => (
            <section
              key={tab.id}
              id={`admin-${tab.id}-panel`}
              role="tabpanel"
              aria-label={`Panneau ${tab.label}`}
              hidden={activeTab !== tab.id}
            >
              {activeTab === tab.id && tab.id === "review" ? (
                <ReviewTab
                  queue={queue}
                  selectedKey={effectiveSelectedKey}
                  mutatingKey={mutatingKey}
                  editingKey={effectiveEditingKey}
                  onSelect={handleSelectElement}
                  classNames={classNames}
                  onReview={handleReview}
                  onModify={handleModify}
                  onEdit={(element) => {
                    setSelectedKey(element.key);
                    setEditingKey(element.key);
                  }}
                  onAnnulerEdit={() => setEditingKey(null)}
                />
              ) : null}
              {activeTab === tab.id && tab.id === "dataset" ? (
                <DatasetTab
                  queue={queue}
                  onJumpToReview={(element) => {
                    handleSelectTab("review");
                    setSelectedKey(element.key);
                    setEditingKey(null);
                  }}
                />
              ) : null}
              {activeTab === tab.id && tab.id === "training" ? (
                <TrainingTab />
              ) : null}
            </section>
          ))}
        </div>
      ) : null}
    </div>
  );
}

export default AdminAnnotationsPage;
