import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { AnalysisRecord } from '../types';
import { createMemoryStorageDb, type AnalysisStorageDb } from './storageDb';
import { LEGACY_MIGRATION_MARKER_KEY, LEGACY_STORAGE_KEY } from './storageLegacy';
import {
  __resetStorageForTests,
  deleteAnalysis,
  getAnalysisById,
  getHistory,
  getLastStorageInitResult,
  initializeStorage,
  normalizeAnalysisRecord,
  saveAnalysis,
  updateElements,
} from './storage';

const RECORD: AnalysisRecord = {
  id: 'analysis-id',
  imageName: 'sample.png',
  imageDataUrl: 'data:image/png;base64,abc',
  timestamp: 1704067200000,
  result: {
    num_elements: 2,
    image_size: [800, 600],
    elements: [
      { bbox: [1, 2, 3, 4], class_name: 'atl', class_label: 1, confidence: 0.9, rejected: false, top_k: [] },
      { bbox: [5, 6, 7, 8], class_name: 'beta', class_label: 2, confidence: 0.8, rejected: false, top_k: [] },
    ],
  },
  annotations: {},
};

const SECOND_RECORD: AnalysisRecord = {
  ...RECORD,
  id: 'second-id',
  imageName: 'second.png',
  timestamp: 1704153600000,
};

function cloneRecord(record: AnalysisRecord): AnalysisRecord {
  return structuredClone(record) as AnalysisRecord;
}

function resetWithDb(db: AnalysisStorageDb = createMemoryStorageDb()) {
  __resetStorageForTests({ dbFactory: () => db });
  return db;
}

describe('storage IndexedDB boundary compatibility', () => {
  beforeEach(() => {
    localStorage.clear();
    vi.restoreAllMocks();
    resetWithDb();
  });

  it('getHistory returns [] for missing legacy payload and empty IndexedDB', async () => {
    await expect(getHistory()).resolves.toEqual([]);
  });

  it('getHistory returns [] for invalid legacy JSON', async () => {
    localStorage.setItem(LEGACY_STORAGE_KEY, '{not json');

    await expect(getHistory()).resolves.toEqual([]);
  });

  it('getHistory returns [] for valid JSON that is not an array', async () => {
    localStorage.setItem(LEGACY_STORAGE_KEY, JSON.stringify({ id: 'not-array' }));

    await expect(getHistory()).resolves.toEqual([]);
  });

  it('migrates old records without annotationStatus as draft-compatible records', async () => {
    localStorage.setItem(LEGACY_STORAGE_KEY, JSON.stringify([RECORD]));

    expect((await getHistory())[0]).toEqual(expect.objectContaining({ annotationStatus: {} }));
  });

  it('retains legacy localStorage and writes an advisory marker after migration', async () => {
    localStorage.setItem(LEGACY_STORAGE_KEY, JSON.stringify([RECORD]));

    await initializeStorage();

    expect(localStorage.getItem(LEGACY_STORAGE_KEY)).toBe(JSON.stringify([RECORD]));
    expect(localStorage.getItem(LEGACY_MIGRATION_MARKER_KEY)).toContain('importedCount');
  });

  it('migrates legacy arrays in original order and does not duplicate on repeated initialization', async () => {
    const db = createMemoryStorageDb();
    resetWithDb(db);
    localStorage.setItem(LEGACY_STORAGE_KEY, JSON.stringify([RECORD, SECOND_RECORD]));

    await initializeStorage();
    __resetStorageForTests({ dbFactory: () => db });
    await initializeStorage();

    expect((await getHistory()).map((record) => record.id)).toEqual(['analysis-id', 'second-id']);
  });

  it('shares concurrent initialization and imports legacy records once', async () => {
    const db = createMemoryStorageDb();
    const importSpy = vi.spyOn(db, 'importMissingRecords');
    resetWithDb(db);
    localStorage.setItem(LEGACY_STORAGE_KEY, JSON.stringify([RECORD, SECOND_RECORD]));

    const [first, second, third] = await Promise.all([getHistory(), initializeStorage(), getHistory()]);

    expect(first.map((record) => record.id)).toEqual(['analysis-id', 'second-id']);
    expect(second.ok).toBe(true);
    expect(third.map((record) => record.id)).toEqual(['analysis-id', 'second-id']);
    expect(importSpy).toHaveBeenCalledTimes(1);
  });

  it('merges missing legacy ids without overwriting existing IndexedDB records', async () => {
    const newerA = { ...RECORD, imageName: 'indexeddb-a.png' };
    resetWithDb(createMemoryStorageDb([newerA]));
    localStorage.setItem(LEGACY_STORAGE_KEY, JSON.stringify([RECORD, SECOND_RECORD]));

    const history = await getHistory();

    expect(history.map((record) => record.id)).toEqual(['analysis-id', 'second-id']);
    expect(history[0].imageName).toBe('indexeddb-a.png');
  });

  it('saveAnalysis prepends normalized new records and replaces duplicate ids in place', async () => {
    await saveAnalysis(RECORD);
    await saveAnalysis(SECOND_RECORD);
    await saveAnalysis({ ...RECORD, imageName: 'updated.png', annotationStatus: { 0: 'validated', 9: 'validated' } });

    const history = await getHistory();

    expect(history.map((record) => record.id)).toEqual(['second-id', 'analysis-id']);
    expect(history[1].imageName).toBe('updated.png');
    expect(history[1].annotationStatus).toEqual({ 0: 'validated' });
  });

  it('getAnalysisById returns matching normalized record or null', async () => {
    await saveAnalysis(RECORD);

    await expect(getAnalysisById('analysis-id')).resolves.toEqual(expect.objectContaining({ id: 'analysis-id', annotationStatus: {} }));
    await expect(getAnalysisById('missing')).resolves.toBeNull();
  });

  it('updateElements updates elements, num_elements, and normalized status', async () => {
    await saveAnalysis({ ...RECORD, annotationStatus: { 0: 'validated', 1: 'draft' } });

    const nextElements = [{ ...RECORD.result.elements[0], class_name: 'renamed' }];
    await expect(updateElements('analysis-id', nextElements)).resolves.toBe(true);

    const [stored] = await getHistory();
    expect(stored.result.elements).toHaveLength(1);
    expect(stored.result.num_elements).toBe(1);
    expect(stored.annotationStatus).toEqual({ 0: 'validated' });
  });

  it('updateElements accepts explicit annotation status updates and returns false for missing ids', async () => {
    await saveAnalysis(RECORD);

    await expect(updateElements('analysis-id', RECORD.result.elements, { 0: 'validated' })).resolves.toBe(true);
    await expect(updateElements('missing', RECORD.result.elements)).resolves.toBe(false);

    expect((await getHistory())[0].annotationStatus).toEqual({ 0: 'validated' });
  });

  it('deleteAnalysis removes only requested ids and is idempotent', async () => {
    await saveAnalysis(RECORD);
    await saveAnalysis(SECOND_RECORD);

    await deleteAnalysis('second-id');
    await deleteAnalysis('missing');

    expect((await getHistory()).map((record) => record.id)).toEqual(['analysis-id']);
  });

  it('falls back to read-only legacy history when IndexedDB initialization fails', async () => {
    localStorage.setItem(LEGACY_STORAGE_KEY, JSON.stringify([RECORD]));
    __resetStorageForTests({
      dbFactory: () => {
        throw new Error('open failed');
      },
    });

    await expect(getHistory()).resolves.toEqual([normalizeAnalysisRecord(RECORD)]);
    await expect(saveAnalysis(cloneRecord(RECORD))).rejects.toThrow('open failed');
    expect(getLastStorageInitResult()).toEqual(expect.objectContaining({ ok: false }));
  });
});
