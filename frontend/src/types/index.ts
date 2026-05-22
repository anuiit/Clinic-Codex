export type ClassLabel = string | number;

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
