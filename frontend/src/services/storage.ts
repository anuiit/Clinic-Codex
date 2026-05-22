import type { AnalysisRecord, AnnotationStatus } from '../types';
import {
  createIndexedDbStorageDb,
  type AnalysisStorageDb,
} from './storageDb';
import {
  hasLegacyPayload,
  markLegacyMigrationComplete,
  readLegacyAnalysisRecords,
} from './storageLegacy';

export type StorageInitResult = {
  ok: boolean;
  migrated: boolean;
  legacyFound: boolean;
  legacyRecordCount: number;
  importedCount: number;
  error?: Error;
};

type StorageDbFactory = () => AnalysisStorageDb;

type StorageTestOptions = {
  dbFactory?: StorageDbFactory;
};

let dbFactory: StorageDbFactory = () => createIndexedDbStorageDb();
let db: AnalysisStorageDb | null = null;
let initPromise: Promise<StorageInitResult> | null = null;
let lastInitResult: StorageInitResult | null = null;

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
    result: {
      ...record.result,
      num_elements: elements.length,
      elements,
    },
  };
}

function toError(issue: unknown) {
  return issue instanceof Error ? issue : new Error(String(issue));
}

function getDb() {
  if (!db) {
    db = dbFactory();
  }
  return db;
}

function legacyHistory() {
  return readLegacyAnalysisRecords().map(normalizeAnalysisRecord);
}

export async function initializeStorage(): Promise<StorageInitResult> {
  if (initPromise) return initPromise;

  initPromise = (async () => {
    const legacyFound = hasLegacyPayload();
    const legacyRecords = legacyHistory();

    try {
      const storageDb = getDb();
      const importedCount = await storageDb.importMissingRecords(legacyRecords);
      markLegacyMigrationComplete(importedCount);
      const result: StorageInitResult = {
        ok: true,
        migrated: true,
        legacyFound,
        legacyRecordCount: legacyRecords.length,
        importedCount,
      };
      lastInitResult = result;
      return result;
    } catch (issue) {
      db?.close();
      db = null;
      initPromise = null;
      const result: StorageInitResult = {
        ok: false,
        migrated: false,
        legacyFound,
        legacyRecordCount: legacyRecords.length,
        importedCount: 0,
        error: toError(issue),
      };
      lastInitResult = result;
      return result;
    }
  })();

  return initPromise;
}

async function getWritableDb() {
  const result = await initializeStorage();
  if (!result.ok) {
    throw result.error ?? new Error('Browser storage is unavailable');
  }
  return getDb();
}

export function getLastStorageInitResult() {
  return lastInitResult;
}

export async function getHistory(): Promise<AnalysisRecord[]> {
  const result = await initializeStorage();
  if (!result.ok) {
    return legacyHistory();
  }

  try {
    const records = await getDb().listRecords();
    return records.map((stored) => normalizeAnalysisRecord(stored.record));
  } catch (issue) {
    lastInitResult = {
      ...result,
      ok: false,
      error: toError(issue),
    };
    return legacyHistory();
  }
}

export async function getAnalysisById(id: string): Promise<AnalysisRecord | null> {
  const result = await initializeStorage();
  if (!result.ok) {
    return legacyHistory().find((record) => record.id === id) ?? null;
  }

  try {
    const stored = await getDb().getRecord(id);
    return stored ? normalizeAnalysisRecord(stored.record) : null;
  } catch (issue) {
    lastInitResult = {
      ...result,
      ok: false,
      error: toError(issue),
    };
    return legacyHistory().find((record) => record.id === id) ?? null;
  }
}

export async function saveAnalysis(record: AnalysisRecord): Promise<void> {
  const storageDb = await getWritableDb();
  await storageDb.saveRecord(normalizeAnalysisRecord(record));
}

export async function updateAnnotations(id: string, annotations: Record<number, string>): Promise<boolean> {
  try {
    const storageDb = await getWritableDb();
    return storageDb.updateAnnotations(id, annotations);
  } catch {
    return false;
  }
}

export async function updateElements(
  id: string,
  elements: AnalysisRecord['result']['elements'],
  annotationStatus?: Record<number, AnnotationStatus>,
): Promise<boolean> {
  try {
    const storageDb = await getWritableDb();
    const existing = await storageDb.getRecord(id);
    if (!existing) return false;
    const nextStatus = normalizeAnnotationStatus(
      annotationStatus ?? existing.record.annotationStatus,
      elements.length,
    );
    return storageDb.updateElements(id, elements, nextStatus);
  } catch {
    return false;
  }
}

export async function deleteAnalysis(id: string): Promise<void> {
  const storageDb = await getWritableDb();
  await storageDb.deleteRecord(id);
}

export function __resetStorageForTests(options: StorageTestOptions = {}): void {
  db?.close();
  db = null;
  initPromise = null;
  lastInitResult = null;
  dbFactory = options.dbFactory ?? (() => createIndexedDbStorageDb());
}
