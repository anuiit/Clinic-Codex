import styles from "./TrainingTab.module.css";
import { useCallback, useEffect, useState } from "react";
import { ActionButton, AdminSection, MetricStrip, StatusPill } from "../../components/ui/AdminPrimitives";
import { getAdminTrainingSummary, getLatestAdminTrainingJob, startAdminTrainingJob } from "../../services/api";
import type { AdminTrainingJob, AdminTrainingSummary } from "../../types";
import { formatClassSummary, formatPrimitiveValue, isRecord, TRAINING_JOB_POLL_INTERVAL_MS } from "./model";
import { PanelSkeleton } from "./shared";
import { ClassDistributionBars, LossMetricPreview, MetadataValue, TrainingConsole, TrainingJobPanel, ValidationAccuracyMetricPreview } from "./TrainingHelpers";

function trainingLaunchReasonLabel(reason: string) {
  if (reason.startsWith("disabled_by_default:")) {
    return "Protection locale active : l'entraînement doit être ouvert volontairement par un administrateur technique.";
  }
  if (
    reason.includes("no_trainable_annotations") ||
    reason.includes("not enough") ||
    reason.includes("insufficient")
  ) {
    return "Dataset insuffisant : validez au moins un élément à jour avant de lancer un essai.";
  }
  if (reason.includes("running")) {
    return "Un essai est déjà en cours : attendez sa fin avant d'en démarrer un autre.";
  }
  if (reason.startsWith("training_snapshot_stale:")) {
    return "Le snapshot cumulatif ne contient pas les dernières validations : reconstruisez-le avant le réentraînement.";
  }
  if (reason.startsWith("training_model_dir_override_unsupported:")) {
    return "MODEL_DIR est actif : désactivez cet override avant le réentraînement admin.";
  }
  if (reason.includes("training_snapshot")) {
    return "Snapshot cumulatif indisponible ou invalide : vérifiez sa configuration locale.";
  }
  if (reason.includes("training_backbone")) {
    return "Le pin local DINOv2 requis pour un réentraînement reproductible est absent.";
  }
  if (reason.includes("training_warmstart")) {
    return "Le modèle existant ne fournit pas la projection nécessaire au warm-start.";
  }
  return reason.replaceAll("_", " ");
}

export function TrainingTab() {
  const [summary, setSummary] = useState<AdminTrainingSummary | null>(null);
  const [latestJob, setLatestJob] = useState<AdminTrainingJob | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [dryRun, setDryRun] = useState(true);
  const [device, setDevice] = useState("auto");
  const [batchSize, setBatchSize] = useState(16);
  const [notes, setNotes] = useState("");

  const loadSummary = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const next = await getAdminTrainingSummary();
      setSummary(next);
      setLatestJob(next.latest_job ?? null);
      setDevice(next.parameters.editable.device[0] ?? "auto");
      setBatchSize(next.parameters.editable.batch_size.default);
    } catch {
      setError("Impossible de charger le résumé d’entraînement local.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const timeout = window.setTimeout(() => {
      void loadSummary();
    }, 0);
    return () => window.clearTimeout(timeout);
  }, [loadSummary]);

  useEffect(() => {
    if (latestJob?.status !== "running") {
      return;
    }
    const interval = window.setInterval(async () => {
      try {
        const response = await getLatestAdminTrainingJob();
        setLatestJob(response.job);
      } catch {
        setError("Impossible d’actualiser le dernier essai local.");
      }
    }, TRAINING_JOB_POLL_INTERVAL_MS);
    return () => window.clearInterval(interval);
  }, [latestJob?.status]);

  const startJob = async () => {
    if (!summary) {
      return;
    }
    if (latestJob?.status === "running") {
      setError("Un essai local est déjà en cours.");
      return;
    }
    if (!Number.isInteger(batchSize) || batchSize < batchBounds.min || batchSize > batchBounds.max) {
      setError(`La taille de lot doit être un entier entre ${batchBounds.min} et ${batchBounds.max}.`);
      return;
    }
    if (notes.length > 200) {
      setError("Les notes de lancement doivent contenir 200 caractères ou moins.");
      return;
    }
    setError(null);
    setStarting(true);
    try {
      const response = await startAdminTrainingJob({
        dry_run: dryRun,
        device,
        batch_size: batchSize,
        notes,
      });
      setLatestJob(response.job);
    } catch {
      setError(
        "Impossible de lancer l'essai local. Vérifiez que l'option d'entraînement est activée sur le backend local.",
      );
    } finally {
      setStarting(false);
    }
  };

  if (loading) {
    return <PanelSkeleton label="Chargement de l'entraînement" />;
  }

  if (!summary) {
    return (
      <div role="alert" className="ui-alert ui-alert--danger p-4">
        {error ?? "Résumé d'entraînement indisponible."}
      </div>
    );
  }

  const jobRunning = latestJob?.status === "running";
  const disabled =
    !summary.launch_allowed_for_request || starting || jobRunning;
  const batchBounds = summary.parameters.editable.batch_size;
  const disabledByDefault = summary.launch_disabled_reasons.some((reason) =>
    reason.startsWith("disabled_by_default:"),
  );
  const modelDirOverrideActive = Boolean(
    summary.paths.model_dir_override_active,
  );
  const launchState = !summary.launch_allowed_for_request
    ? "Lancement bloqué"
    : "Prêt pour essai local";
  const launchMode = dryRun ? "Essai à blanc sélectionné" : "Entraînement complet sélectionné";
  const launchModeCopy = dryRun
    ? "L'essai à blanc vérifie le snapshot cumulatif, le modèle existant et le chemin de génération du candidat sans lancer l'entraînement."
    : "Le réentraînement part du modèle existant et du snapshot cumulatif incluant les validations. Il crée uniquement un candidat local.";
  const modelConfig = isRecord(summary.parameters.config.model)
    ? summary.parameters.config.model
    : {};
  const trainingConfig = isRecord(summary.parameters.config.training)
    ? summary.parameters.config.training
    : {};
  const backbone = formatPrimitiveValue(modelConfig.backbone ?? "ResNet18 · baseline");
  const epochs = formatPrimitiveValue(trainingConfig.num_epochs ?? 30);
  const splitCounts = summary.data.split_counts as AdminTrainingSummary["data"]["split_counts"] | undefined;
  const snapshotSplits = summary.training_snapshot.split_counts;
  if (!splitCounts) {
    return (
      <section className={`${styles.owner} admin-tab-stack`}>
        <div role="alert" className="ui-alert ui-alert--danger p-4">
          Résumé d'entraînement incomplet : split train/val/test indisponible. Actualisez la page ou vérifiez le backend local.
        </div>
      </section>
    );
  }

  return (
    <section className={`${styles.owner} admin-tab-stack`}>
      <AdminSection as="div" variant="summary">
        <p className="ui-text-eyebrow">Entraînement</p>
        <h2 className="text-lg font-semibold text-[color:var(--text-heading)]">
          Réentraînement cumulatif local
        </h2>
        <p className="ui-text-caption">
          Le lancement reste volontairement protégé. Cette page sert d'assistant guidé : un administrateur technique ouvre
          le mode entraînement local, puis l'opérateur vérifie le dataset et lance d'abord un essai à blanc.
        </p>
      </AdminSection>

      <section className="admin-train-config" aria-label="Configuration d'entraînement">
        <div className="admin-train-config-line"><span>Modèle</span><strong>{backbone}</strong></div>
        <div className="admin-train-config-line"><span>Époques</span><strong>{epochs}</strong></div>
        <div className="admin-train-config-line"><span>Batch</span><strong>{batchSize}</strong></div>
        <div className="admin-train-config-line"><span>Sortie</span><strong>{summary.paths.runs_dir}</strong></div>
      </section>

      <section className="admin-train-checks" aria-label="Préflight d'entraînement">
        <div className={`admin-train-check ${summary.training_snapshot.valid ? "ok" : "warn"}`}><i>{summary.training_snapshot.valid ? "✓" : "!"}</i><b>Snapshot cumulatif</b><small>{summary.training_snapshot.row_count ?? "—"}</small></div>
        <div className="admin-train-check ok"><i>✓</i><b>Nouvelles validations prises en compte</b><small>{summary.training_snapshot.live_annotation_count ?? "—"}</small></div>
        <div className={`admin-train-check ${snapshotSplits ? "ok" : "warn"}`}><i>{snapshotSplits ? "✓" : "!"}</i><b>Split train/dev/test verrouillé</b><small>{snapshotSplits ? `${snapshotSplits.train}/${snapshotSplits.dev}/${snapshotSplits.locked_test}` : "—"}</small></div>
        <div className="admin-train-check warn"><i>!</i><b>À vérifier</b><small>{summary.data.pending}</small></div>
        <div className="admin-train-check warn"><i>!</i><b>Exclues</b><small>{splitCounts.excluded}</small></div>
        <div className="admin-train-check ok"><i>✓</i><b>GPU local</b><small>{device}</small></div>
      </section>

      {!summary.launch_allowed_for_request ? (
        <div role="alert" className="ui-alert ui-alert--accent p-2">
          <strong>Lancement bloqué : non autorisé pour cette requête.</strong>
          <ul className="mt-2 list-disc pl-5 text-sm">
            {summary.launch_disabled_reasons.map((reason) => (
              <li key={reason}>{trainingLaunchReasonLabel(reason)}</li>
            ))}
          </ul>
          {disabledByDefault ? (
            <p className="mt-2 text-sm">
              Demandez à l'administrateur technique d'ouvrir une session d'entraînement locale. Le réglage par défaut du
              projet reste fermé pour éviter tout lancement involontaire.
            </p>
          ) : null}
        </div>
      ) : null}

      {modelDirOverrideActive ? (
        <div role="alert" className="ui-alert ui-alert--accent p-2">
          <strong>Réentraînement bloqué : chemin de modèle personnalisé actif.</strong>
          <p className="mt-2 text-sm">
            Retirez <code>MODEL_DIR</code> avant le lancement pour que la configuration et la projection viennent du même
            package runtime.
          </p>
        </div>
      ) : null}

      {error ? (
        <div role="alert" className="ui-alert ui-alert--danger p-2">
          {error}
        </div>
      ) : null}

      <ol className="admin-workflow-steps" aria-label="Parcours d'entraînement">
        <li>
          <strong>1. Vérifier les éléments</strong>
          <span>Le snapshot doit inclure le corpus existant et toutes les validations à jour.</span>
        </li>
        <li>
          <strong>2. Essai à blanc d'abord</strong>
          <span>Tester la préparation locale sans remplacer le modèle.</span>
        </li>
        <li>
          <strong>3. Promouvoir explicitement</strong>
          <span>Contrôler le candidat avant redémarrage.</span>
        </li>
      </ol>

      <MetricStrip
        aria-label="Compteurs d'entraînement"
        items={[
          { label: "Prêts", value: summary.data.trainable, tone: "ready" },
          { label: "Validés", value: summary.data.approved },
          {
            label: "Rejetés",
            value: summary.data.rejected,
            tone: summary.data.rejected > 0 ? "danger" : "neutral",
          },
          { label: "À vérifier", value: summary.data.pending },
          { label: "Classes", value: summary.data.classes.length },
        ]}
      />

      <section
        className="admin-training-metrics"
        aria-label="Aperçu des métriques d'entraînement"
      >
        <LossMetricPreview job={latestJob} />
        <ValidationAccuracyMetricPreview job={latestJob} />
        <ClassDistributionBars splitCounts={splitCounts} />
      </section>
      <p className="ui-text-caption px-1">
        Les courbes de perte et validation s’activent quand le journal du run expose ces métriques. Les barres de split utilisent
        la distribution réelle du résumé local.
      </p>

      <TrainingConsole summary={summary} job={latestJob} />

      <AdminSection className="admin-class-summary">
        <h3 className="ui-title-sm">Validés par classe</h3>
        {Object.keys(summary.data.per_class).length ? (
          <ul className="mt-3 flex flex-wrap gap-2">
            {Object.entries(summary.data.per_class).map(
              ([className, count]) => (
                <li key={className}>
                  <StatusPill tone="ready">{className}: {count}</StatusPill>
                </li>
              ),
            )}
          </ul>
        ) : (
          <p className="mt-2 ui-text-caption">
            Aucun élément prêt pour l'instant.
          </p>
        )}
      </AdminSection>

      <AdminSection
        className="admin-launch-summary"
        aria-label="Résumé avant lancement de l'entraînement"
      >
        <div className="flex flex-wrap items-center gap-2">
          <h3 className="ui-title-sm">Résumé avant lancement</h3>
          <StatusPill tone={summary.launch_allowed_for_request ? "ready" : "warning"}>
            {launchState}
          </StatusPill>
        </div>
        <p className="mt-2 ui-text-body-sm">
          <strong>{launchMode}:</strong> {launchModeCopy}
        </p>
        <dl className="mt-3 grid gap-2 text-sm md:grid-cols-2 lg:grid-cols-3">
          <div>
            <dt className="ui-text-caption">Utilisera</dt>
            <dd>
              {summary.training_snapshot.row_count ?? "—"} images cumulées
            </dd>
          </div>
          <div>
            <dt className="ui-text-caption">Snapshot</dt>
            <dd>{summary.training_snapshot.snapshot_id ?? "Non configuré"}</dd>
          </div>
          <div>
            <dt className="ui-text-caption">Validations live prises en compte</dt>
            <dd>{summary.training_snapshot.live_annotation_count ?? "—"}</dd>
          </div>
          <div>
            <dt className="ui-text-caption">Classes</dt>
            <dd>{formatClassSummary(summary.data.classes)}</dd>
          </div>
          <div>
            <dt className="ui-text-caption">Machine</dt>
            <dd>{device}</dd>
          </div>
          <div>
            <dt className="ui-text-caption">Taille de lot</dt>
            <dd>{batchSize}</dd>
          </div>
          <div>
            <dt className="ui-text-caption">Dernier essai</dt>
            <dd>
              {latestJob
                ? `${latestJob.run_id} (${latestJob.status})`
                : "Aucun essai enregistré"}
            </dd>
          </div>
          <div>
            <dt className="ui-text-caption">Permission de lancement</dt>
            <dd>
              {summary.launch_allowed_for_request
                ? "Autorisée pour cette session locale"
                : "Bloquée par la protection backend"}
            </dd>
          </div>
          <div>
            <dt className="ui-text-caption">Activation</dt>
            <dd>Bloquée tant que le contrat d'évaluation promotion n'est pas complet</dd>
          </div>
        </dl>
      </AdminSection>

      <AdminSection
        className="admin-training-launch"
        aria-label="Contrôles de lancement de l'entraînement"
      >
        <div className="admin-training-form">
          <label className="admin-field">
            <span>Essai à blanc</span>
            <select
              className="ui-select px-2 py-1"
              value={dryRun ? "yes" : "no"}
              onChange={(event) => setDryRun(event.target.value === "yes")}
            >
              <option value="yes">Oui</option>
              <option value="no">Non, entraînement complet</option>
            </select>
          </label>
          <label className="admin-field">
            <span>Machine</span>
            <select
              className="ui-select px-2 py-1"
              value={device}
              onChange={(event) => setDevice(event.target.value)}
            >
              {summary.parameters.editable.device.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </label>
          <label className="admin-field">
            <span>Taille de lot</span>
            <input
              className="ui-input px-2 py-1"
              type="number"
              min={batchBounds.min}
              max={batchBounds.max}
              value={batchSize}
              onChange={(event) => setBatchSize(Number(event.target.value))}
            />
          </label>
          <label className="admin-field admin-field--grow">
            <span>Notes</span>
            <input
              className="ui-input px-2 py-1"
              value={notes}
              onChange={(event) => setNotes(event.target.value)}
            />
          </label>
        </div>
        <div className="admin-training-submit">
          <div>
            <p className="ui-text-eyebrow">Action de lancement</p>
            <p className="ui-text-body-sm">
              <strong>{launchMode}.</strong> Les fichiers candidats ne remplacent jamais le modèle actif sans promotion explicite et redémarrage.
            </p>
          </div>
          <ActionButton
            tone="primary"
            className="admin-training-primary"
            disabled={disabled}
            onClick={startJob}
          >
            {starting
              ? "Démarrage…"
              : jobRunning
                ? "Entraînement en cours"
                : dryRun
                  ? "Lancer l'essai à blanc"
                  : "Lancer l'entraînement"}
          </ActionButton>
        </div>
      </AdminSection>

      <TrainingJobPanel job={latestJob} />

      <details className="admin-audit-details">
        <summary>Détails support/admin</summary>
        <p className="mt-2 ui-text-caption">
          Informations techniques utiles au support local, masquées pendant le
          triage opérateur.
        </p>
        <section className="admin-metadata-grid mt-3">
          <AdminSection as="div">
            <h3 className="ui-title-sm">Fichiers produits</h3>
            <dl className="mt-3 space-y-2 text-sm">
              {Object.entries(summary.artifacts).map(([key, value]) => (
                <div key={key}>
                  <dt className="ui-text-caption">{key}</dt>
                  <dd className="break-all"><MetadataValue value={value} /></dd>
                </div>
              ))}
            </dl>
          </AdminSection>
          <AdminSection as="div">
            <h3 className="ui-title-sm">Chemins résolus</h3>
            <dl className="mt-3 space-y-2 text-sm">
              {Object.entries(summary.paths).map(([key, value]) => (
                <div key={key}>
                  <dt className="ui-text-caption">{key}</dt>
                  <dd className="break-all"><MetadataValue value={value} /></dd>
                </div>
              ))}
            </dl>
          </AdminSection>
          <AdminSection as="div">
            <h3 className="ui-title-sm">Configuration en lecture seule</h3>
            <dl className="mt-3 space-y-2 text-sm">
              {Object.entries(summary.parameters.config).flatMap(
                ([section, value]) =>
                  Object.entries((value as Record<string, unknown>) ?? {}).map(
                    ([key, entry]) => (
                      <div key={`${section}.${key}`}>
                        <dt className="ui-text-caption">
                          {section}.{key}
                        </dt>
                        <dd><MetadataValue value={entry} /></dd>
                      </div>
                    ),
                  ),
              )}
            </dl>
          </AdminSection>
        </section>
      </details>
    </section>
  );
}
