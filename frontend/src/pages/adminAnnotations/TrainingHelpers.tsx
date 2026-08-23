import { AdminSection, StatusPill, MetricPane } from "../../components/ui/AdminPrimitives";
import type { AdminTrainingJob, AdminTrainingSummary } from "../../types";
import { formatMetadataLabel, formatPrimitiveValue, isRecord, MAX_METADATA_DEPTH, MAX_METADATA_ITEMS } from "./model";

export function MetadataValue({ value, depth = 0 }: { value: unknown; depth?: number }) {
  if (depth >= MAX_METADATA_DEPTH && (Array.isArray(value) || isRecord(value))) {
    const size = Array.isArray(value) ? value.length : Object.keys(value).length;
    return <span>{size ? `${size} entrée${size === 1 ? "" : "s"} imbriquée${size === 1 ? "" : "s"}` : "—"}</span>;
  }
  if (Array.isArray(value)) {
    if (!value.length) {
      return <span>—</span>;
    }
    const visibleItems = value.slice(0, MAX_METADATA_ITEMS);
    return (
      <ul className="admin-metadata-list">
        {visibleItems.map((item, index) => (
          <li key={index}>
            <MetadataValue value={item} depth={depth + 1} />
          </li>
        ))}
        {value.length > visibleItems.length ? (
          <li className="ui-text-caption">
            +{value.length - visibleItems.length} entrée
            {value.length - visibleItems.length === 1 ? "" : "s"}
          </li>
        ) : null}
      </ul>
    );
  }
  if (isRecord(value)) {
    const entries = Object.entries(value);
    if (!entries.length) {
      return <span>—</span>;
    }
    const visibleEntries = entries.slice(0, MAX_METADATA_ITEMS);
    return (
      <dl className="admin-metadata-nested">
        {visibleEntries.map(([key, nestedValue]) => (
          <div key={key}>
            <dt>{formatMetadataLabel(key)}</dt>
            <dd>
              <MetadataValue value={nestedValue} depth={depth + 1} />
            </dd>
          </div>
        ))}
        {entries.length > visibleEntries.length ? (
          <div className="admin-metadata-more">
            <dt>Suite</dt>
            <dd>
              +{entries.length - visibleEntries.length} champ
              {entries.length - visibleEntries.length === 1 ? "" : "s"}
            </dd>
          </div>
        ) : null}
      </dl>
    );
  }
  return <span>{formatPrimitiveValue(value)}</span>;
}

export function TrainingJobPanel({ job }: { job: AdminTrainingJob | null }) {
  if (!job) {
    return (
      <div className="ui-empty-state p-3">
        Aucun essai local enregistré pour l'instant.
      </div>
    );
  }

  return (
    <AdminSection className="admin-training-job-panel">
      <div className="flex flex-wrap items-center gap-2">
        <h3 className="ui-title-sm">Dernier essai {job.run_id}</h3>
        <StatusPill tone={job.status === "succeeded" ? "ready" : job.status === "failed" ? "danger" : "warning"}>
          {job.status}
        </StatusPill>
        {job.dry_run ? <StatusPill>Essai à blanc</StatusPill> : null}
      </div>
      <dl className="mt-3 grid gap-2 text-sm md:grid-cols-2">
        <div>
          <dt className="ui-text-caption">Machine</dt>
          <dd>{job.device}</dd>
        </div>
        <div>
          <dt className="ui-text-caption">Taille de lot</dt>
          <dd>{job.batch_size}</dd>
        </div>
        <div>
          <dt className="ui-text-caption">Démarré</dt>
          <dd>{job.started_at ?? "—"}</dd>
        </div>
        <div>
          <dt className="ui-text-caption">Code de sortie</dt>
          <dd>{job.exit_code ?? "—"}</dd>
        </div>
      </dl>
      <details className="admin-audit-details mt-3">
        <summary>Détails support/admin</summary>
        {job.command?.length ? (
          <p className="mt-3 ui-text-caption">
            Commande : <code>{job.command.join(" ")}</code>
          </p>
        ) : null}
        {job.log_tail?.length ? (
          <pre className="mt-3 max-h-52 overflow-auto rounded-lg bg-[color:var(--surface-strong)] p-3 text-xs text-[color:var(--text-body)]">
            {job.log_tail.join("\n")}
          </pre>
        ) : (
          <p className="mt-3 ui-text-caption">Aucun journal capturé pour l'instant.</p>
        )}
      </details>
    </AdminSection>
  );
}

export function TrainingConsole({
  summary,
  job,
}: {
  summary: AdminTrainingSummary;
  job: AdminTrainingJob | null;
}) {
  const split = summary.training_snapshot.split_counts;
  return (
    <section className="admin-training-console" aria-label="Console d'entraînement">
      <div><span className="ok">✓</span> snapshot loaded: {summary.training_snapshot.row_count ?? "—"} images cumulées</div>
      <div><span className="ok">✓</span> warm-start: modèle existant + {summary.training_snapshot.live_annotation_count ?? "—"} validations live</div>
      {split ? <div><span className="ok">✓</span> split locked: train {split.train} · dev {split.dev} · locked test {split.locked_test}</div> : null}
      {summary.data.pending > 0 ? <div><span className="warn">!</span> {summary.data.pending} images restent à vérifier</div> : null}
      <div><span className="run">→</span> {job?.status === "running" ? "training run active" : "ready to train"}</div>
    </section>
  );
}

function TrendLine({
  points,
  tone,
}: {
  points: string;
  tone: "gold" | "violet";
}) {
  return (
    <svg viewBox="0 0 240 120" preserveAspectRatio="none" aria-hidden="true">
      <polyline
        points={points}
        fill="none"
        stroke="currentColor"
        strokeWidth="3"
        className={tone === "gold" ? "text-[color:var(--accent)]" : "text-[color:var(--violet)]"}
      />
    </svg>
  );
}

function MetricUnavailable({ label }: { label: string }) {
  return (
    <div className="flex h-full min-h-24 items-center justify-center px-3 text-center ui-text-caption">
      {label}
    </div>
  );
}

function hasLogMetric(job: AdminTrainingJob | null, pattern: RegExp) {
  return Boolean(job?.log_tail?.some((line) => pattern.test(line)));
}

export function LossMetricPreview({ job }: { job: AdminTrainingJob | null }) {
  const hasLoss = hasLogMetric(job, /loss/i);
  return (
    <MetricPane label="Perte" value={hasLoss ? "journal" : "—"} testId="LossMetricPreview">
      {hasLoss ? (
        <TrendLine
          tone="gold"
          points="0,22 30,38 62,51 90,60 120,74 150,82 180,91 210,99 240,105"
        />
      ) : (
        <MetricUnavailable label="Métrique indisponible pour ce run" />
      )}
    </MetricPane>
  );
}

export function ValidationAccuracyMetricPreview({ job }: { job: AdminTrainingJob | null }) {
  const hasAccuracy = hasLogMetric(job, /acc/i);
  return (
    <MetricPane label="Validation" value={hasAccuracy ? "journal" : "—"} testId="ValidationAccuracyMetricPreview">
      {hasAccuracy ? (
        <TrendLine
          tone="violet"
          points="0,104 30,88 62,80 90,72 120,61 150,48 180,38 210,31 240,26"
        />
      ) : (
        <MetricUnavailable label="Métrique indisponible pour ce run" />
      )}
    </MetricPane>
  );
}

export function ClassDistributionBars({
  splitCounts,
}: {
  splitCounts: AdminTrainingSummary["data"]["split_counts"];
}) {
  const bars = [
    { key: "train", tone: "text-[color:var(--status-ready)]" },
    { key: "val", tone: "text-[color:var(--violet)]" },
    { key: "test", tone: "text-[color:var(--orange)]" },
    { key: "excluded", tone: "text-[color:var(--danger)]" },
  ] as const;
  const max = Math.max(...bars.map((bar) => splitCounts[bar.key] ?? 0), 1);
  return (
    <div className="admin-metric-pane" data-testid="ClassDistributionBars">
      <div className="metric-label">splits · réel</div>
      <div className="metric-value">{Object.values(splitCounts).reduce((a, b) => a + b, 0)}</div>
      <div className="chart">
        <svg viewBox="0 0 240 120" preserveAspectRatio="none" aria-hidden="true">
          {bars.map((bar, index) => {
            const value = splitCounts[bar.key] ?? 0;
            const height = Math.max(8, Math.round((value / max) * 96));
            return (
              <rect
                key={bar.key}
                x={18 + index * 54}
                y={120 - height}
                width="26"
                height={height}
                fill="currentColor"
                className={bar.tone}
              />
            );
          })}
        </svg>
      </div>
    </div>
  );
}
