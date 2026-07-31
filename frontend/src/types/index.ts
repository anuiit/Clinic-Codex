export type ClassLabel = string | number;

export interface AuthUser {
  id: string;
  email: string;
  role?: string;
  roles: string[];
  permissions: string[];
}

export interface AuthSession {
  auth_enabled: boolean;
  user: AuthUser | null;
  status?: string;
  csrf_token?: string;
}

export interface LoginPayload {
  email: string;
  password: string;
}

export interface BootstrapStatus {
  status: 'ok';
  auth_enabled: boolean;
  bootstrap_available: boolean;
}

export interface TopKItem {
  class_name: string;
  class_label?: ClassLabel | null;
  confidence: number;
}

export interface ClassifyResult {
  class_name: string;
  class_label?: ClassLabel | null;
  confidence: number;
  rejected: boolean;
  top_k: TopKItem[];
}

export interface DetectedElement extends ClassifyResult {
  bbox: [number, number, number, number]; // [x, y, w, h]
}

export interface SegmentResult {
  num_elements: number;
  image_size: [number, number]; // [w, h]
  elements: DetectedElement[];
}

export interface ClassesResult {
  num_classes: number;
  class_names: string[];
}

export interface SimilarItem {
  rank: number;
  match_type: string;
  class_name: string;
  class_label: ClassLabel | null;
  similarity: number;
  band: 'high' | 'moderate' | 'low';
  asset: string | null;
}

export interface SimilarResult {
  query: { bbox: [number, number, number, number]; mode: string };
  best_match: { class_name: string; similarity: number; rejected: boolean };
  results: SimilarItem[];
}

export interface TrustSignals {
  predicted_class_rank: number;
  predicted_class_similarity: number;
  top1_class: string;
  top1_similarity: number;
  margin_to_second: number;
  above_rejection_threshold: boolean;
  rejection_threshold: number;
  ambiguous: boolean;
  entropy: number;
  top_k: TopKItem[];
}

export interface TrustResult {
  query: { bbox: [number, number, number, number]; predicted_class: string };
  trust: TrustSignals;
}

export type AnnotationStatus = 'draft' | 'validated';

export interface AnalysisRecord {
  id: string;
  imageName: string;
  imageDataUrl: string;
  timestamp: number;
  result: SegmentResult;
  annotations: Record<number, string>;
  annotationStatus?: Record<number, AnnotationStatus>;
}

export interface SaveAnnotationPayload {
  analysis_id: string;
  image_name: string;
  image_data_url: string;
  timestamp: number;
  annotations: Array<{
    index: number;
    bbox: [number, number, number, number];
    class_name: string;
  }>;
}

export interface SaveAnnotationResponse {
  status: "ok" | "error";
  analysis_id: string;
  saved_count: number;
  classes: string[];
  saved_at?: string;
  error?: string;
}

export type SaveAnnotationErrorCode =
  | 'VALIDATION_ERROR'
  | 'PERMISSION_DENIED'
  | 'DISK_FULL'
  | 'STORAGE_ERROR'
  | 'INTERNAL_ERROR'
  | 'NETWORK_ERROR';

export interface SaveAnnotationError {
  ok: false;
  error_code: SaveAnnotationErrorCode;
  message: string;
  hint?: string;
  trace_id?: string;
}

export type SaveAnnotationSuccess = SaveAnnotationResponse & { ok: true };

export type SaveAnnotationResult = SaveAnnotationSuccess | SaveAnnotationError;

export type AdminAnnotationReviewStatus = 'pending' | 'approved' | 'rejected';
export type AdminDatasetSplit = 'train' | 'val' | 'test' | 'excluded';

export interface AdminAnnotationDiagnostic {
  code: string;
  message: string;
  analysis_id?: string;
  index?: number;
  key?: string;
}

export interface AdminAnnotationCounts {
  total: number;
  pending: number;
  approved: number;
  rejected: number;
  trainable: number;
}

export interface AdminAnnotationElement {
  key: string;
  analysis_id: string;
  index: number;
  class_name: string;
  bbox: [number, number, number, number] | number[];
  crop_path: string;
  crop_url: string;
  crop_exists: boolean;
  review_status: AdminAnnotationReviewStatus;
  trainable: boolean;
  dataset_split: AdminDatasetSplit;
  split_reason: string;
  source_fingerprint: string;
  stale_decision: boolean;
}

export interface AdminAnnotationAnalysis {
  analysis_id: string;
  uploaded_at?: string;
  image_path: string;
  image_url: string;
  image_exists: boolean;
  elements: AdminAnnotationElement[];
}

export interface AdminAnnotationQueue {
  status: 'ok';
  schema_version: number;
  local_only: boolean;
  warning: string;
  counts: AdminAnnotationCounts;
  analyses: AdminAnnotationAnalysis[];
  diagnostics: AdminAnnotationDiagnostic[];
}

export interface AdminAnnotationMutationResponse {
  status: 'ok';
  local_only: boolean;
  warning: string;
  element: AdminAnnotationElement;
  counts?: AdminAnnotationCounts;
}

export interface AdminAnnotationModifyPayload {
  class_name: string;
  bbox: [number, number, number, number];
  approve_after_save?: boolean;
  status?: AdminAnnotationReviewStatus;
}

export interface AdminTrainingFileInfo {
  path: string;
  exists: boolean;
  size?: number;
  mtime?: string;
  sha256?: string | null;
}

export interface AdminTrainingJob {
  run_id: string;
  status: 'running' | 'succeeded' | 'failed' | 'disabled' | 'rejected';
  local_only?: boolean;
  dry_run: boolean;
  device: string;
  batch_size: number;
  notes?: string;
  started_at?: string;
  finished_at?: string | null;
  exit_code?: number | null;
  pid?: number | null;
  process_identity?: string | null;
  command?: string[];
  cwd?: string;
  env?: Record<string, string>;
  log_path?: string;
  log_tail?: string[];
  artifacts?: Record<string, unknown>;
}

export interface AdminTrainingSummary {
  status: 'ok';
  local_only: boolean;
  warning: string;
  training_jobs_enabled: boolean;
  launch_allowed_for_request: boolean;
  launch_disabled_reasons: string[];
  data: {
    total: number;
    pending: number;
    approved: number;
    rejected: number;
    trainable: number;
    classes: string[];
    per_class: Record<string, number>;
    split_counts: Record<AdminDatasetSplit, number>;
    diagnostics: AdminAnnotationDiagnostic[];
  };
  parameters: {
    editable: {
      dry_run: boolean;
      device: string[];
      batch_size: { default: number; min: number; max: number };
    };
    script_env_defaults: Record<string, string>;
    config: Record<string, unknown>;
  };
  paths: Record<string, string | boolean | null>;
  artifacts: Record<string, unknown>;
  latest_job?: AdminTrainingJob | null;
}

export interface AdminTrainingJobResponse {
  status: 'ok';
  local_only: boolean;
  job: AdminTrainingJob | null;
}

export interface AdminTrainingStartPayload {
  dry_run: boolean;
  device: string;
  batch_size: number;
  notes?: string;
}
