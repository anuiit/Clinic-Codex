import { useEffect, useState, type ReactNode } from 'react';
import { getRuntimeVersion } from '../services/api';
import type { RuntimeVersionInfo } from '../types';
import {
  DEFAULT_RUNTIME_VERSION,
  RuntimeVersionContext,
} from './RuntimeVersionContext';

export function RuntimeVersionProvider({ children }: { children: ReactNode }) {
  const [runtime, setRuntime] = useState<RuntimeVersionInfo>(
    DEFAULT_RUNTIME_VERSION,
  );

  useEffect(() => {
    const controller = new AbortController();
    getRuntimeVersion({ signal: controller.signal })
      .then(setRuntime)
      .catch(() => {
        // The app remains usable if version metadata is temporarily unavailable.
      });
    return () => controller.abort();
  }, []);

  return (
    <RuntimeVersionContext.Provider value={runtime}>
      {children}
    </RuntimeVersionContext.Provider>
  );
}
