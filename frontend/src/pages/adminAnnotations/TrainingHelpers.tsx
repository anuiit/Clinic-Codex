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
      {job.error ? <p role="alert" className="ui-alert ui-alert--danger p-3">{job.error}</p> : null}
      {job.result ? (
        <section aria-label="Résultat du candidat" className="mt-3">
          <h4 className="ui-title-sm">Candidat créé — non activé</h4>
          <p>{job.result.unique_count} images uniques, {job.result.duplicate_count} doublons ignorés ; {job.result.updated_classes.length} classes mises à jour.</p>
          <p>Sur les images apprises : base {job.result.base_correct}/{job.result.unique_count}, candidat {job.result.candidate_correct}/{job.result.unique_count}.</p>
          <p className="ui-alert ui-alert--accent p-2">Diagnostic d'apprentissage uniquement : aucune amélioration sur de nouvelles images n'est démontrée.</p>
          <p>Version : <code>{job.model_version_id}</code></p>
          <p>Fichiers locaux : <code>{job.candidate_version_dir}</code></p>
        </section>
      ) : null}
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
      <div><span className={summary.training_snapshot.valid ? "ok" : "warn"}>{summary.training_snapshot.valid ? "✓" : "!"}</span> {summary.training_snapshot.mode === "local_prior" ? "Capture automatique au lancement" : "Snapshot cumulatif"} : {summary.training_snapshot.row_count ?? "—"} images</div>
      <div><span className={summary.training_snapshot.live_train_count ? "ok" : "warn"}>{summary.training_snapshot.live_train_count ? "✓" : "!"}</span> projection conservée · {summary.training_snapshot.live_train_count ?? "—"} validations utilisées pour les prototypes</div>
      {split ? <div><span className="ok">✓</span> split locked: train {split.train} · dev {split.dev} · locked test {split.locked_test}</div> : null}
      {summary.data.pending > 0 ? <div><span className="warn">!</span> {summary.data.pending} images restent à vérifier</div> : null}
      <div><span className="run">→</span> {job?.status === "running" ? "training run active" : summary.launch_allowed_for_request ? "ready to train" : "lancement bloqué"}</div>
    </section>
  );
}

export function LossMetricPreview({ job }: { job: AdminTrainingJob | null }) {
  return (
    <MetricPane label="Images apprises" value={job?.result?.unique_count ?? "—"} testId="LossMetricPreview">
      <p className="ui-text-caption p-3">Projection figée : aucune courbe de perte d'optimisation.</p>
    </MetricPane>
  );
}

export function ValidationAccuracyMetricPreview({ job }: { job: AdminTrainingJob | null }) {
  return (
    <MetricPane label="Validation indépendante" value="—" testId="ValidationAccuracyMetricPreview">
      <p className="ui-text-caption p-3">{job?.result
        ? "Non mesurée : les annotations servent à l'apprentissage."
        : "Métrique indisponible pour ce run"}</p>
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
            const height = Math.round((value / max) * 96);
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
