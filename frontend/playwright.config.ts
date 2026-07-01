import { defineConfig, devices } from 'playwright/test';

export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  // The app uses one shared Vite dev server plus IndexedDB-heavy flows.
  // Keep browser e2e deterministic locally and in CI; previous fully parallel
  // runs produced false-red page/context timeouts while the same suite passed
  // consistently with one worker.
  workers: 1,
  reporter: 'list',
  use: {
    baseURL: 'http://localhost:7118',
    trace: 'on-first-retry',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
  webServer: {
    command: 'npm run dev',
    url: 'http://localhost:7118',
    reuseExistingServer: true,
    timeout: 30000,
  },
});
