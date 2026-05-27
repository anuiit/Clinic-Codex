import type { Page } from 'playwright/test';
import {
  LEGACY_MIGRATION_MARKER_KEY,
  LEGACY_STORAGE_KEY,
} from '../../src/services/storageLegacy';

const DB_NAME = 'clinic-codex-storage';
const STORE_NAME = 'analysisRecords';

type SeedRecord = { id: string } & Record<string, unknown>;

export async function seedIndexedDbRecord(page: Page, record: SeedRecord) {
  await page.evaluate(
    async ({ dbName, storeName, seededRecord }) => {
      await new Promise<void>((resolve, reject) => {
        const request = indexedDB.open(dbName, 1);
        request.onupgradeneeded = () => {
          const db = request.result;
          if (!db.objectStoreNames.contains(storeName)) {
            db.createObjectStore(storeName, { keyPath: 'id' });
          }
        };
        request.onerror = () => reject(request.error ?? new Error('Unable to open IndexedDB'));
        request.onsuccess = () => {
          const db = request.result;
          const tx = db.transaction(storeName, 'readwrite');
          const store = tx.objectStore(storeName);
          store.clear();
          store.put({ id: seededRecord.id, order: 0, record: seededRecord, updatedAt: Date.now() });
          tx.oncomplete = () => {
            db.close();
            resolve();
          };
          tx.onerror = () => reject(tx.error ?? new Error('Unable to seed IndexedDB'));
          tx.onabort = () => reject(tx.error ?? new Error('IndexedDB seed aborted'));
        };
      });
    },
    { dbName: DB_NAME, storeName: STORE_NAME, seededRecord: record },
  );
}

export async function clearIndexedDbRecords(page: Page) {
  await page.evaluate(
    async ({ dbName, legacyMigrationMarkerKey, legacyStorageKey, storeName }) => {
      window.localStorage.removeItem(legacyStorageKey);
      window.localStorage.removeItem(legacyMigrationMarkerKey);
      await new Promise<void>((resolve, reject) => {
        const request = indexedDB.open(dbName, 1);
        request.onupgradeneeded = () => {
          const db = request.result;
          if (!db.objectStoreNames.contains(storeName)) {
            db.createObjectStore(storeName, { keyPath: 'id' });
          }
        };
        request.onerror = () => reject(request.error ?? new Error('Unable to open IndexedDB'));
        request.onsuccess = () => {
          const db = request.result;
          const tx = db.transaction(storeName, 'readwrite');
          tx.objectStore(storeName).clear();
          tx.oncomplete = () => {
            db.close();
            resolve();
          };
          tx.onerror = () => reject(tx.error ?? new Error('Unable to clear IndexedDB'));
          tx.onabort = () => reject(tx.error ?? new Error('IndexedDB clear aborted'));
        };
      });
    },
    {
      dbName: DB_NAME,
      legacyMigrationMarkerKey: LEGACY_MIGRATION_MARKER_KEY,
      legacyStorageKey: LEGACY_STORAGE_KEY,
      storeName: STORE_NAME,
    },
  );
}
