import { AdminSection, StatusPill } from "../../components/ui/AdminPrimitives";
import skeletonStyles from "../../components/LoadingSkeleton.module.css";
import type { AdminAnnotationQueue, AdminAnnotationReviewStatus } from "../../types";
import { STATUS_LABEL, STATUS_TONE } from "./model";

export function StatusBadge({
  status,
  label,
}: {
  status: AdminAnnotationReviewStatus;
  label?: string;
}) {
  const accessibleLabel = label
    ? `${label} : statut ${STATUS_LABEL[status]}`
    : `Statut ${STATUS_LABEL[status]}`;
  return (
    <StatusPill label={accessibleLabel} tone={STATUS_TONE[status]}>
      {STATUS_LABEL[status]}
    </StatusPill>
  );
}

export function QueueCounters({ queue }: { queue: AdminAnnotationQueue }) {
  const remaining = queue.counts.pending;
  return (
    <div className="admin-topstats" aria-label="Compteurs du triage">
      <span><strong>{remaining}</strong> restants</span>
      <span><strong>{queue.counts.trainable}</strong> inclus</span>
      <span><span className="admin-live-dot" aria-hidden="true" />local</span>
    </div>
  );
}

export function PanelSkeleton({ label }: { label: string }) {
  return (
    <AdminSection
      as="div"
      className={`${skeletonStyles.owner} min-h-20`}
      role="status"
      aria-label={label}
    >
      <div className="workspace-trust-skeleton h-3 w-40 rounded-full" />
      <div className="workspace-trust-skeleton mt-4 h-8 rounded-xl" />
      <div className="workspace-trust-skeleton mt-3 h-8 w-2/3 rounded-xl" />
    </AdminSection>
  );
}
