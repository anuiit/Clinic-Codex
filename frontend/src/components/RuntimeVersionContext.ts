import { createContext } from 'react';
import type { RuntimeVersionInfo } from '../types';

export const DEFAULT_RUNTIME_VERSION: RuntimeVersionInfo = {
  app_name: 'Clinic Codex',
  app_version: '',
  model_version: null,
};

export const RuntimeVersionContext = createContext<RuntimeVersionInfo>(
  DEFAULT_RUNTIME_VERSION,
);
