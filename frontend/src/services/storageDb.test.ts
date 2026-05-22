import { beforeEach, describe, expect, it } from 'vitest';
import type { AnalysisRecord } from '../types';
import { createIndexedDbStorageDb, createMemoryStorageDb } from './storageDb';

const RECORD_A: AnalysisRecord = {
  id: 'a',
  imageName: 'a.png',
  imageDataUrl: 'data:image/png;base64,a',
  timestamp: 1704067200000,
  result: {
    num_elements: 2,
    image_size: [800, 600],
    elements: [
      { bbox: [1, 2, 3, 4], class_name: 'alpha', class_label: 1, confidence: 0.9, rejected: false, top_k: [] },
      { bbox: [5, 6, 7, 8], class_name: 'beta', class_label: 2, confidence: 0.8, rejected: false, top_k: [] },
    ],
  },
  annotations: {},
  annotationStatus: { 0: 'validated', 1: 'draft' },
};

const RECORD_B: AnalysisRecord = {
  ...RECORD_A,
  id: 'b',
  imageName: 'b.png',
  timestamp: 1704153600000,
  annotationStatus: {},
};

let db = createMemoryStorageDb();

beforeEach(() => {
  db.close();
  db = createMemoryStorageDb();
});

describe('storageDb adapter contract', () => {
  it('documents the Vitest/jsdom native IndexedDB blocker without adding packages', () => {
    expect(globalThis.indexedDB).toBeUndefined();
    expect(() => createIndexedDbStorageDb()).toThrow('IndexedDB is unavailable');
  });

  it('opens an empty store equivalent and lists no records', async () => {
    await expect(db.listRecords()).resolves.toEqual([]);
  });

  it('saves and lists one record', async () => {
    await db.saveRecord(RECORD_A);

    await expect(db.listRecords()).resolves.toEqual([
      expect.objectContaining({ id: 'a', record: expect.objectContaining({ id: 'a' }) }),
    ]);
  });

  it('saves duplicate ids without creating duplicates and preserves order', async () => {
    await db.saveRecord(RECORD_A);
    await db.saveRecord(RECORD_B);
    await db.saveRecord({ ...RECORD_A, imageName: 'updated-a.png' });

    const records = await db.listRecords();

    expect(records.map((record) => record.id)).toEqual(['b', 'a']);
    expect(records[1].record.imageName).toBe('updated-a.png');
  });

  it('preserves explicit order when importing legacy records and prepends newer saves', async () => {
    await db.importMissingRecords([RECORD_A, RECORD_B]);
    await db.saveRecord({ ...RECORD_A, id: 'newest', imageName: 'newest.png' });

    expect((await db.listRecords()).map((record) => record.id)).toEqual(['newest', 'a', 'b']);
  });

  it('gets existing records and returns null for missing ids', async () => {
    await db.saveRecord(RECORD_A);

    await expect(db.getRecord('a')).resolves.toEqual(expect.objectContaining({ id: 'a' }));
    await expect(db.getRecord('missing')).resolves.toBeNull();
  });

  it('updates elements and annotation status for one existing record', async () => {
    await db.saveRecord(RECORD_A);
    await db.saveRecord(RECORD_B);

    const nextElements = [{ ...RECORD_A.result.elements[0], class_name: 'renamed' }];
    await expect(db.updateElements('a', nextElements, { 0: 'validated' })).resolves.toBe(true);

    const stored = await db.getRecord('a');
    expect(stored?.record.result.elements).toEqual(nextElements);
    expect(stored?.record.result.num_elements).toBe(1);
    expect(stored?.record.annotationStatus).toEqual({ 0: 'validated' });
    expect((await db.getRecord('b'))?.record.imageName).toBe('b.png');
  });

  it('returns false for a missing update and leaves the store unchanged', async () => {
    await db.saveRecord(RECORD_A);

    await expect(db.updateElements('missing', [], {})).resolves.toBe(false);

    expect((await db.listRecords()).map((record) => record.id)).toEqual(['a']);
  });

  it('updates annotations independently', async () => {
    await db.saveRecord(RECORD_A);

    await expect(db.updateAnnotations('a', { 0: 'aleph' })).resolves.toBe(true);
    expect((await db.getRecord('a'))?.record.annotations).toEqual({ 0: 'aleph' });
  });

  it('deletes only matching ids and tolerates repeated opens/resets', async () => {
    await db.saveRecord(RECORD_A);
    await db.saveRecord(RECORD_B);

    await db.deleteRecord('a');
    await db.deleteRecord('a');

    expect((await db.listRecords()).map((record) => record.id)).toEqual(['b']);
  });

  it('merges missing legacy ids after existing IndexedDB records', async () => {
    await db.saveRecord({ ...RECORD_A, imageName: 'idb-a.png' });

    await expect(db.importMissingRecords([RECORD_A, RECORD_B])).resolves.toBe(1);

    const history = await db.listRecords();
    expect(history.map((record) => record.id)).toEqual(['a', 'b']);
    expect(history[0].record.imageName).toBe('idb-a.png');
  });
});
