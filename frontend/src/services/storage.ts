import type { AnalysisRecord, AnnotationStatus } from '../types';

const STORAGE_KEY = 'codex_analyses';

function normalizeAnnotationStatus(status: unknown, elementCount: number): Record<number, AnnotationStatus> {
  if (!status || typeof status !== 'object') {
    return {};
  }

  const normalized: Record<number, AnnotationStatus> = {};
  Object.entries(status as Record<string, unknown>).forEach(([key, value]) => {
    const idx = Number(key);
    if (!Number.isInteger(idx) || idx < 0 || idx >= elementCount) return;
    if (value === 'validated' || value === 'draft') {
      normalized[idx] = value;
    }
  });
  return normalized;
}

export function normalizeAnalysisRecord(record: AnalysisRecord): AnalysisRecord {
  const elements = record.result?.elements ?? [];
  return {
    ...record,
    annotations: record.annotations ?? {},
    annotationStatus: normalizeAnnotationStatus(record.annotationStatus, elements.length),
  };
}

export function saveAnalysis(record: AnalysisRecord): void {
  const history = getHistory();
  history.unshift(normalizeAnalysisRecord(record));
  localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
}

export function getHistory(): AnalysisRecord[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) as AnalysisRecord[] : [];
    return Array.isArray(parsed) ? parsed.map(normalizeAnalysisRecord) : [];
  } catch {
    return [];
  }
}

export function getAnalysisById(id: string): AnalysisRecord | null {
  return getHistory().find((r) => r.id === id) ?? null;
}

export function updateAnnotations(id: string, annotations: Record<number, string>): boolean {
  const history = getHistory();
  const idx = history.findIndex((r) => r.id === id);
  if (idx === -1) return false;
  try {
    history[idx] = { ...history[idx], annotations };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
    return true;
  } catch {
    return false;
  }
}

export function updateElements(
  id: string,
  elements: AnalysisRecord['result']['elements'],
  annotationStatus?: Record<number, AnnotationStatus>,
): boolean {
  const history = getHistory();
  const idx = history.findIndex((r) => r.id === id);
  if (idx === -1) return false;
  try {
    const nextStatus = normalizeAnnotationStatus(annotationStatus ?? history[idx].annotationStatus, elements.length);
    history[idx] = {
      ...history[idx],
      annotationStatus: nextStatus,
      result: {
        ...history[idx].result,
        elements,
        num_elements: elements.length,
      },
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
    return true;
  } catch {
    return false;
  }
}

export function deleteAnalysis(id: string): void {
  const filtered = getHistory().filter((r) => r.id !== id);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(filtered));
}
