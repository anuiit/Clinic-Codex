import type { AnalysisRecord } from '../types';

export const LEGACY_STORAGE_KEY = 'codex_analyses';
export const LEGACY_MIGRATION_MARKER_KEY = 'codex_analyses_migrated_to_idb_v1';

function getLocalStorage(): Storage | null {
  try {
    return globalThis.localStorage ?? null;
  } catch {
    return null;
  }
}

function isObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object';
}

function isAnalysisRecord(value: unknown): value is AnalysisRecord {
  if (!isObject(value)) return false;
  const result = value.result;
  return (
    typeof value.id === 'string' &&
    typeof value.imageName === 'string' &&
    typeof value.imageDataUrl === 'string' &&
    typeof value.timestamp === 'number' &&
    isObject(result) &&
    typeof result.num_elements === 'number' &&
    Array.isArray(result.image_size) &&
    Array.isArray(result.elements) &&
    isObject(value.annotations)
  );
}

export function readLegacyAnalysisRecords(): AnalysisRecord[] {
  const storage = getLocalStorage();
  if (!storage) return [];

  try {
    const raw = storage.getItem(LEGACY_STORAGE_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isAnalysisRecord);
  } catch {
    return [];
  }
}

export function hasLegacyPayload(): boolean {
  const storage = getLocalStorage();
  if (!storage) return false;
  try {
    return storage.getItem(LEGACY_STORAGE_KEY) !== null;
  } catch {
    return false;
  }
}

export function markLegacyMigrationComplete(importedCount: number): void {
  const storage = getLocalStorage();
  if (!storage) return;
  try {
    storage.setItem(
      LEGACY_MIGRATION_MARKER_KEY,
      JSON.stringify({ version: 1, importedCount, migratedAt: Date.now() }),
    );
  } catch {
    // Advisory marker only; never fail storage initialization because it cannot be written.
  }
}
