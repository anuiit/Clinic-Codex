import type { ThemeMode } from "../../components/ThemeToggle";
import type {
  AdminAnnotationAnalysis,
  AdminAnnotationElement,
  AdminAnnotationQueue,
  AdminAnnotationReviewStatus,
  AdminDatasetSplit,
} from "../../types";

export type AdminTab = "review" | "dataset" | "training";

export type AdminAnnotationsPageProps = {
  themeMode?: ThemeMode;
  onToggleTheme?: () => void;
  initialTab?: AdminTab;
  onNavigateTab?: (tab: AdminTab) => void;
};

export const ADMIN_TABS: Array<{ id: AdminTab; label: string; description: string }> =
  [
    {
      id: "review",
      label: "Trier",
      description: "Vérifier, valider, rejeter ou corriger les éléments.",
    },
    {
      id: "dataset",
      label: "Dataset",
      description: "Comprendre ce qui est prêt ou bloqué pour l'entraînement.",
    },
    {
      id: "training",
      label: "Entraîner",
      description:
        "Lancer prudemment un essai local à partir des éléments prêts.",
    },
  ];

export const STATUS_LABEL: Record<AdminAnnotationReviewStatus, string> = {
  pending: "À vérifier",
  approved: "Validé",
  rejected: "Rejeté",
};

export const STATUS_TONE: Record<AdminAnnotationReviewStatus, "warning" | "ready" | "danger"> = {
  pending: "warning",
  approved: "ready",
  rejected: "danger",
};


export function formatBbox(bbox: number[]) {
  return bbox.join(", ");
}

export function formatTimestamp(value: Date | null) {
  if (!value) {
    return "pas encore actualisé";
  }
  return value.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}
export type ReviewStatusFilter = AdminAnnotationReviewStatus | "all";

export type ReviewRow = {
  analysis: AdminAnnotationAnalysis;
  element: AdminAnnotationElement;
  diagnostics: string[];
};

export const REVIEW_STATUS_FILTER_LABEL: Record<ReviewStatusFilter, string> = {
  all: "Tous les statuts",
  pending: "À vérifier",
  approved: "Validés",
  rejected: "Rejetés",
};

export function diagnosticsByElementKey(queue: AdminAnnotationQueue) {
  const diagnosticsByKey = new Map<string, string[]>();
  for (const diagnostic of queue.diagnostics) {
    if (!diagnostic.key) {
      continue;
    }
    diagnosticsByKey.set(diagnostic.key, [
      ...(diagnosticsByKey.get(diagnostic.key) ?? []),
      `${diagnostic.code}: ${diagnostic.message}`,
    ]);
  }
  return diagnosticsByKey;
}

export function reviewRows(queue: AdminAnnotationQueue): ReviewRow[] {
  const diagnosticsByKey = diagnosticsByElementKey(queue);
  return queue.analyses.flatMap((analysis) =>
    analysis.elements
      .filter((element) => element.review_status !== "approved")
      .map((element) => ({
        analysis,
        element,
        diagnostics: diagnosticsByKey.get(element.key) ?? [],
      })),
  );
}

export function trainabilityCopy(row: ReviewRow) {
  const { element, diagnostics } = row;
  if (element.trainable) {
    return "Cet élément est validé, à jour, et son image de découpe existe : il est prêt pour l'entraînement.";
  }
  if (element.review_status === "rejected") {
    return "Cet élément est rejeté : il restera visible pour contrôle mais ne sera pas utilisé pour l'entraînement.";
  }
  if (element.review_status === "pending") {
    return "Cet élément attend une décision. Il ne sera pas entraîné tant qu'il n'est pas validé.";
  }
  if (!element.crop_exists) {
    return "L'élément est validé mais son image de découpe manque : il est bloqué pour l'entraînement.";
  }
  if (element.stale_decision) {
    return "La décision doit être refaite après régénération avant d'être utilisée pour l'entraînement.";
  }
  if (diagnostics.length) {
    return diagnostics[0];
  }
  return "L'élément est validé mais pas encore utilisable : vérifiez le blocage indiqué.";
}

export function reviewRowSignal(row: ReviewRow) {
  const { element, diagnostics } = row;
  if (element.trainable) {
    return "Prêt pour l'entraînement";
  }
  if (element.review_status === "approved") {
    return `À corriger · ${diagnostics[0] ?? "vérifier la découpe"}`;
  }
  if (element.review_status === "rejected") {
    return "Écarté du dataset";
  }
  return "À décider";
}
export type DatasetBucket =
  "trainable" | "approved_nontrainable" | "rejected" | "pending";
export type TrainableDatasetSplit = Exclude<AdminDatasetSplit, "excluded">;
export type DatasetSplitFilter = TrainableDatasetSplit | "all";

export type DatasetRow = {
  element: AdminAnnotationElement;
  bucket: DatasetBucket;
  diagnostics: string[];
};


export const TRAINABLE_DATASET_SPLITS: TrainableDatasetSplit[] = [
  "train",
  "val",
  "test",
];

export const DATASET_SPLIT_LABEL: Record<AdminDatasetSplit | "all", string> = {
  all: "Tous les splits",
  train: "Train",
  val: "Val",
  test: "Test",
  excluded: "Exclus",
};

export const DATASET_SPLIT_TONE: Record<AdminDatasetSplit, "ready" | "warning" | "danger"> = {
  train: "ready",
  val: "warning",
  test: "warning",
  excluded: "danger",
};

export function datasetBucketFor(element: AdminAnnotationElement): DatasetBucket {
  if (element.trainable) {
    return "trainable";
  }
  if (element.review_status === "approved") {
    return "approved_nontrainable";
  }
  return element.review_status;
}

export function datasetRows(queue: AdminAnnotationQueue): DatasetRow[] {
  const diagnosticsByKey = diagnosticsByElementKey(queue);

  return queue.analyses.flatMap((analysis) =>
    analysis.elements
      .filter(
        (element) => element.trainable && element.dataset_split !== "excluded",
      )
      .map((element) => ({
        element,
        bucket: datasetBucketFor(element),
        diagnostics: diagnosticsByKey.get(element.key) ?? [],
      })),
  );
}

export function formatPrimitiveValue(value: unknown) {
  if (value === null || value === undefined || value === "") {
    return "—";
  }
  if (typeof value === "boolean") {
    return value ? "Oui" : "Non";
  }
  return String(value);
}

export function formatMetadataLabel(key: string) {
  return key.replaceAll("_", " ");
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

export const MAX_METADATA_DEPTH = 3;
export const MAX_METADATA_ITEMS = 8;

export const TRAINING_JOB_POLL_INTERVAL_MS = 1000;
export const ADMIN_QUEUE_AUTO_REFRESH_MS = 30_000;

export function formatClassSummary(classes: string[]) {
  return classes.length ? classes.join(", ") : "Aucune classe prête pour l'instant";
}
