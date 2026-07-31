import { defineConfig, type UserConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import type { InlineConfig } from 'vitest'
import { configDefaults } from 'vitest/config'

// https://vite.dev/config/
const config: UserConfig & { test: InlineConfig } = {
  server: { port: 7118, strictPort: false, host: '127.0.0.1' },
  plugins: [react(), tailwindcss()],
  test: {
    environment: 'jsdom',
    exclude: [...configDefaults.exclude, 'tests/e2e/**', '.omx-artifacts/**'],
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
    testTimeout: 10_000,
  },
}

export default defineConfig(config)
