import type { AnalysisRecord, AnnotationStatus } from '../types';

export const STORAGE_DB_NAME = 'clinic-codex-storage';
export const STORAGE_DB_VERSION = 1;
export const ANALYSIS_STORE_NAME = 'analysisRecords';

export type StoredAnalysisRecord = {
  id: string;
  order: number;
  record: AnalysisRecord;
  updatedAt: number;
};

export type AnalysisStorageDb = {
  listRecords(): Promise<StoredAnalysisRecord[]>;
  getRecord(id: string): Promise<StoredAnalysisRecord | null>;
  saveRecord(record: AnalysisRecord): Promise<void>;
  updateAnnotations(id: string, annotations: Record<number, string>): Promise<boolean>;
  updateElements(
    id: string,
    elements: AnalysisRecord['result']['elements'],
    annotationStatus: Record<number, AnnotationStatus>,
  ): Promise<boolean>;
  deleteRecord(id: string): Promise<void>;
  importMissingRecords(records: AnalysisRecord[]): Promise<number>;
  close(): void;
};

type IndexedDbStorageOptions = {
  dbName?: string;
  indexedDBFactory?: IDBFactory;
};

function cloneRecord(record: AnalysisRecord): AnalysisRecord {
  return structuredClone(record) as AnalysisRecord;
}

function cloneElements(elements: AnalysisRecord['result']['elements']) {
  return structuredClone(elements) as AnalysisRecord['result']['elements'];
}

function ordered(records: StoredAnalysisRecord[]) {
  return [...records].sort((a, b) => a.order - b.order || b.updatedAt - a.updatedAt);
}

function requestToPromise<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error('IndexedDB request failed'));
  });
}

function transactionDone(transaction: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    transaction.oncomplete = () => resolve();
    transaction.onabort = () => reject(transaction.error ?? new Error('IndexedDB transaction aborted'));
    transaction.onerror = () => reject(transaction.error ?? new Error('IndexedDB transaction failed'));
  });
}

export function createIndexedDbStorageDb({
  dbName = STORAGE_DB_NAME,
  indexedDBFactory,
}: IndexedDbStorageOptions = {}): AnalysisStorageDb {
  const factory = indexedDBFactory ?? globalThis.indexedDB;
  if (!factory) {
    throw new Error('IndexedDB is unavailable in this environment');
  }

  let dbPromise: Promise<IDBDatabase> | null = null;

  const openDb = (): Promise<IDBDatabase> => {
    if (dbPromise) return dbPromise;

    dbPromise = new Promise<IDBDatabase>((resolve, reject) => {
      const request = factory.open(dbName, STORAGE_DB_VERSION);

      request.onupgradeneeded = () => {
        const db = request.result;
        if (!db.objectStoreNames.contains(ANALYSIS_STORE_NAME)) {
          db.createObjectStore(ANALYSIS_STORE_NAME, { keyPath: 'id' });
        }
      };

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error ?? new Error('Unable to open IndexedDB storage'));
      request.onblocked = () => reject(new Error('IndexedDB storage upgrade was blocked'));
    }).catch((error) => {
      dbPromise = null;
      throw error;
    });

    return dbPromise;
  };

  async function withStore<T>(
    mode: IDBTransactionMode,
    callback: (store: IDBObjectStore) => Promise<T>,
  ): Promise<T> {
    const db = await openDb();
    const transaction = db.transaction(ANALYSIS_STORE_NAME, mode);
    const store = transaction.objectStore(ANALYSIS_STORE_NAME);
    const result = await callback(store);
    await transactionDone(transaction);
    return result;
  }

  return {
    async listRecords() {
      const records = await withStore('readonly', (store) =>
        requestToPromise<StoredAnalysisRecord[]>(store.getAll()),
      );
      return ordered(records);
    },

    async getRecord(id) {
      const record = await withStore('readonly', (store) =>
        requestToPromise<StoredAnalysisRecord | undefined>(store.get(id)),
      );
      return record ?? null;
    },

    async saveRecord(record) {
      await withStore('readwrite', async (store) => {
        const now = Date.now();
        const [existing, allRecords] = await Promise.all([
          requestToPromise<StoredAnalysisRecord | undefined>(store.get(record.id)),
          requestToPromise<StoredAnalysisRecord[]>(store.getAll()),
        ]);
        const minOrder = allRecords.length > 0 ? Math.min(...allRecords.map((item) => item.order)) : 1;
        const next: StoredAnalysisRecord = {
          id: record.id,
          order: existing?.order ?? minOrder - 1,
          record: cloneRecord(record),
          updatedAt: now,
        };
        await requestToPromise(store.put(next));
      });
    },

    async updateAnnotations(id, annotations) {
      return withStore('readwrite', async (store) => {
        const existing = await requestToPromise<StoredAnalysisRecord | undefined>(store.get(id));
        if (!existing) return false;
        await requestToPromise(
          store.put({
            ...existing,
            record: { ...existing.record, annotations },
            updatedAt: Date.now(),
          }),
        );
        return true;
      });
    },

    async updateElements(id, elements, annotationStatus) {
      return withStore('readwrite', async (store) => {
        const existing = await requestToPromise<StoredAnalysisRecord | undefined>(store.get(id));
        if (!existing) return false;
        await requestToPromise(
          store.put({
            ...existing,
            record: {
              ...existing.record,
              annotationStatus,
              result: {
                ...existing.record.result,
                elements: cloneElements(elements),
                num_elements: elements.length,
              },
            },
            updatedAt: Date.now(),
          }),
        );
        return true;
      });
    },

    async deleteRecord(id) {
      await withStore('readwrite', async (store) => {
        await requestToPromise(store.delete(id));
      });
    },

    async importMissingRecords(records) {
      if (records.length === 0) return 0;
      return withStore('readwrite', async (store) => {
        const existing = await requestToPromise<StoredAnalysisRecord[]>(store.getAll());
        const existingIds = new Set(existing.map((item) => item.id));
        const storeIsEmpty = existing.length === 0;
        const nextAppendOrder = storeIsEmpty ? 0 : Math.max(...existing.map((item) => item.order)) + 1;
        let imported = 0;

        for (const [index, record] of records.entries()) {
          if (existingIds.has(record.id)) continue;
          const order = storeIsEmpty ? index : nextAppendOrder + imported;
          await requestToPromise(
            store.put({
              id: record.id,
              order,
              record: cloneRecord(record),
              updatedAt: Date.now(),
            } satisfies StoredAnalysisRecord),
          );
          imported += 1;
        }

        return imported;
      });
    },

    close() {
      if (dbPromise) {
        void dbPromise.then((db) => db.close()).catch(() => undefined);
        dbPromise = null;
      }
    },
  };
}

export function createMemoryStorageDb(initialRecords: AnalysisRecord[] = []): AnalysisStorageDb {
  const records = new Map<string, StoredAnalysisRecord>();
  initialRecords.forEach((record, index) => {
    records.set(record.id, {
      id: record.id,
      order: index,
      record: cloneRecord(record),
      updatedAt: Date.now(),
    });
  });

  const listStored = () => ordered([...records.values()]).map((record) => ({
    ...record,
    record: cloneRecord(record.record),
  }));

  return {
    async listRecords() {
      return listStored();
    },
    async getRecord(id) {
      const record = records.get(id);
      return record ? { ...record, record: cloneRecord(record.record) } : null;
    },
    async saveRecord(record) {
      const existing = records.get(record.id);
      const allRecords = [...records.values()];
      const minOrder = allRecords.length > 0 ? Math.min(...allRecords.map((item) => item.order)) : 1;
      records.set(record.id, {
        id: record.id,
        order: existing?.order ?? minOrder - 1,
        record: cloneRecord(record),
        updatedAt: Date.now(),
      });
    },
    async updateAnnotations(id, annotations) {
      const existing = records.get(id);
      if (!existing) return false;
      records.set(id, {
        ...existing,
        record: { ...cloneRecord(existing.record), annotations },
        updatedAt: Date.now(),
      });
      return true;
    },
    async updateElements(id, elements, annotationStatus) {
      const existing = records.get(id);
      if (!existing) return false;
      const record = cloneRecord(existing.record);
      records.set(id, {
        ...existing,
        record: {
          ...record,
          annotationStatus,
          result: {
            ...record.result,
            elements: cloneElements(elements),
            num_elements: elements.length,
          },
        },
        updatedAt: Date.now(),
      });
      return true;
    },
    async deleteRecord(id) {
      records.delete(id);
    },
    async importMissingRecords(nextRecords) {
      const existing = [...records.values()];
      const storeIsEmpty = existing.length === 0;
      const nextAppendOrder = storeIsEmpty ? 0 : Math.max(...existing.map((item) => item.order)) + 1;
      let imported = 0;
      nextRecords.forEach((record, index) => {
        if (records.has(record.id)) return;
        records.set(record.id, {
          id: record.id,
          order: storeIsEmpty ? index : nextAppendOrder + imported,
          record: cloneRecord(record),
          updatedAt: Date.now(),
        });
        imported += 1;
      });
      return imported;
    },
    close() {
      records.clear();
    },
  };
}
