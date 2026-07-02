import { expect, test, type Page, type Route } from 'playwright/test';
import { clearIndexedDbRecords, seedIndexedDbRecord } from './storageSeed';

const onePxPng =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';

const smokeRecord = {
  id: 'full-smoke-record',
  imageName: 'full-smoke-glyph.png',
  imageDataUrl: onePxPng,
  timestamp: 1700000000000,
  result: {
    num_elements: 2,
    image_size: [400, 300] as [number, number],
    elements: [
      {
        bbox: [50, 60, 80, 70] as [number, number, number, number],
        class_name: 'glyph-a',
        class_label: 0,
        confidence: 0.91,
        rejected: false,
        top_k: [{ class_name: 'glyph-alpha', confidence: 0.88 }],
      },
      {
        bbox: [190, 120, 90, 75] as [number, number, number, number],
        class_name: '',
        class_label: 1,
        confidence: 0.32,
        rejected: false,
        top_k: [{ class_name: 'beta', confidence: 0.44 }],
      },
    ],
  },
  annotations: {},
  annotationStatus: { 0: 'validated', 1: 'draft' },
};

const adminQueue = {
  status: 'ok',
  schema_version: 1,
  local_only: true,
  warning: 'local queue',
  counts: { total: 4, pending: 1, approved: 2, rejected: 1, trainable: 2 },
  analyses: [
    {
      analysis_id: 'admin-a',
      uploaded_at: '2026-06-30T20:00:00+00:00',
      image_path: '/tmp/admin-a/image.png',
      image_url: '/admin/annotations/admin-a/image',
      image_exists: true,
      elements: [
        {
          key: 'admin-a:0',
          analysis_id: 'admin-a',
          index: 0,
          class_name: 'atl',
          bbox: [10, 20, 30, 40],
          crop_path: '/tmp/admin-a/0.png',
          crop_url: '/admin/annotations/admin-a/0/crop',
          crop_exists: true,
          review_status: 'pending',
          trainable: false,
          source_fingerprint: 'p0',
          stale_decision: false,
          dataset_split: 'excluded',
          split_reason: 'pending_review',
        },
        {
          key: 'admin-a:1',
          analysis_id: 'admin-a',
          index: 1,
          class_name: 'bet',
          bbox: [40, 50, 60, 70],
          crop_path: '/tmp/admin-a/1.png',
          crop_url: '/admin/annotations/admin-a/1/crop',
          crop_exists: true,
          review_status: 'approved',
          trainable: true,
          source_fingerprint: 'p1',
          stale_decision: false,
          dataset_split: 'train',
          split_reason: 'trainable_hash_80_10_10',
        },
        {
          key: 'admin-a:2',
          analysis_id: 'admin-a',
          index: 2,
          class_name: 'rej',
          bbox: [70, 80, 90, 100],
          crop_path: '/tmp/admin-a/2.png',
          crop_url: '/admin/annotations/admin-a/2/crop',
          crop_exists: true,
          review_status: 'rejected',
          trainable: false,
          source_fingerprint: 'p2',
          stale_decision: false,
          dataset_split: 'excluded',
          split_reason: 'rejected_review',
        },
        {
          key: 'admin-a:3',
          analysis_id: 'admin-a',
          index: 3,
          class_name: 'gimel',
          bbox: [100, 110, 120, 130],
          crop_path: '/tmp/admin-a/3.png',
          crop_url: '/admin/annotations/admin-a/3/crop',
          crop_exists: true,
          review_status: 'approved',
          trainable: true,
          source_fingerprint: 'p3',
          stale_decision: false,
          dataset_split: 'val',
          split_reason: 'trainable_hash_80_10_10',
        },
      ],
    },
  ],
  diagnostics: [],
};

const trainingSummary = {
  status: 'ok',
  local_only: true,
  warning: 'local only',
  training_jobs_enabled: true,
  launch_allowed_for_request: true,
  launch_disabled_reasons: [],
  data: {
    total: 4,
    pending: 1,
    approved: 2,
    rejected: 1,
    trainable: 2,
    classes: ['bet', 'gimel'],
    per_class: { bet: 1, gimel: 1 },
    split_counts: { train: 1, val: 1, test: 0, excluded: 2 },
    diagnostics: [],
  },
  parameters: {
    editable: {
      dry_run: true,
      device: ['auto', 'cpu'],
      batch_size: { default: 16, min: 1, max: 256 },
    },
    script_env_defaults: { BATCH_SIZE: '16', DEVICE: 'auto' },
    config: { training: { num_epochs: 10 }, model: { backbone: 'dinov2_vits14' } },
  },
  paths: {
    script: '/repo/scripts/retrain.sh',
    runs_dir: '/repo/backend/training_runs',
    model_dir_override_active: false,
  },
  artifacts: {
    approved_export_manifest: { path: '/repo/manifest.json', exists: true, sha256: 'abc' },
    model_registry: { status: 'ok', promoted_version: 'demo' },
  },
  latest_job: null,
};

async function mockApi(page: Page) {
  await page.route('**/classes', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        num_classes: 4,
        class_names: ['glyph-a', 'glyph-alpha', 'beta', 'atl'],
      }),
    }),
  );
  await page.route('**/trust', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        query: {},
        trust: {
          top1_class: 'glyph-a',
          top1_similarity: 0.9,
          margin_to_second: 0.3,
          above_rejection_threshold: true,
          ambiguous: false,
          entropy: 0.1,
          top_k: [],
        },
      }),
    }),
  );
  await page.route('**/save-annotation', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        status: 'ok',
        analysis_id: smokeRecord.id,
        saved_count: 1,
        classes: ['glyph-a'],
      }),
    }),
  );
  await page.route('**/admin/annotations', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(adminQueue) }),
  );
  await page.route(/.*\/admin\/annotations\/.*\/(image|crop)$/, (route: Route) =>
    route.fulfill({
      status: 200,
      contentType: 'image/png',
      body: Buffer.from(onePxPng.split(',')[1], 'base64'),
    }),
  );
  await page.route(/.*\/admin\/annotations\/.*\/(review|modify)$/, (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ status: 'ok' }) }),
  );
  await page.route('**/admin/training/summary', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(trainingSummary) }),
  );
  await page.route('**/admin/training/jobs/latest', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ status: 'ok', job: null }) }),
  );
  await page.route('**/admin/training/jobs', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        status: 'ok',
        job: {
          run_id: 'dry-1',
          status: 'running',
          dry_run: true,
          device: 'auto',
          batch_size: 16,
          started_at: '2026-07-01T00:00:00Z',
          exit_code: null,
          log_tail: ['started'],
        },
      }),
    }),
  );
}

test.beforeEach(async ({ page }) => {
  await mockApi(page);
});

test('full app smoke: workspace, annotation, and admin tabs stay wired', async ({ page }) => {
  const consoleErrors: string[] = [];
  page.on('console', (message) => {
    if (message.type() === 'error') {
      consoleErrors.push(message.text());
    }
  });
  page.on('pageerror', (error) => consoleErrors.push(error.message));

  await page.goto('/');
  await clearIndexedDbRecords(page);
  await seedIndexedDbRecord(page, smokeRecord);
  await page.goto(`/?analysis=${smokeRecord.id}`);

  await expect(page.getByText('Analyseur de glyphes Codex')).toBeVisible();
  await expect(
    page.getByTestId('workspace-stage').getByRole('img', { name: smokeRecord.imageName }),
  ).toBeVisible();
  await expect(page.getByTestId('image-bbox-stage-box-0')).toBeVisible();

  await page.getByRole('button', { name: /glyph-a région 0/i }).click();
  await expect(page.getByTestId('workspace-focused-selected-card')).toBeVisible();
  await page.getByRole('button', { name: 'Annoter la région' }).click();
  await expect(page).toHaveURL(new RegExp(`/annotate/${smokeRecord.id}\\?element=0$`));

  await expect(
    page.getByTestId('annotation-stage').getByRole('img', { name: smokeRecord.imageName }),
  ).toBeVisible();
  const renameInput = page.getByLabel(/Nommer l’élément 0/);
  await expect(renameInput).toBeVisible();
  await renameInput.fill('gl');
  const suggestionMenu = page.getByTestId('element-name-suggestions');
  await expect(suggestionMenu).toBeVisible();
  await expect(suggestionMenu).toContainText('glyph-alpha');
  await expect
    .poll(() => suggestionMenu.evaluate((node) => node.parentElement?.tagName.toLowerCase()))
    .toBe('body');
  await page.keyboard.press('Escape');
  await page.getByRole('button', { name: 'Zoom avant' }).click();
  await expect(page.getByText('125%')).toBeVisible();
  await page.getByRole('button', { name: 'Ajuster à la vue' }).click();
  await expect(page.getByText('100%')).toBeVisible();

  await page.goto('/admin/annotations/review');
  await expect(page.getByRole('tab', { name: /Trier/i })).toHaveAttribute('aria-selected', 'true');
  const triageList = page.getByRole('listbox', { name: /file de triage/i });
  await expect(triageList).toBeVisible();
  await expect(triageList.getByRole('option', { name: /atl/i })).toBeVisible();
  await expect(triageList.getByRole('option', { name: /rej/i })).toBeVisible();
  await expect(triageList.getByText(/bet/)).toHaveCount(0);

  await page.getByRole('tab', { name: /Dataset/i }).click();
  await expect(page).toHaveURL(/\/admin\/annotations\/dataset/);
  const datasetList = page.getByRole('listbox', { name: /Vue dataset/i });
  await expect(datasetList).toBeVisible();
  await expect(datasetList.getByRole('option', { name: /bet/i })).toBeVisible();
  await expect(datasetList.getByRole('option', { name: /gimel/i })).toBeVisible();
  await expect(datasetList.getByText(/rej/)).toHaveCount(0);

  await page.getByRole('tab', { name: /Entraîner/i }).click();
  await expect(page).toHaveURL(/\/admin\/annotations\/training/);
  await expect(page.getByRole('heading', { name: /Assistant d'entraînement local/i })).toBeVisible();
  await expect(page.getByLabel(/Compteurs d'entraînement/i)).toContainText('Prêts');
  await page.getByRole('button', { name: /Lancer l'essai à blanc/i }).click();
  await expect(page.getByRole('button', { name: /Entraînement en cours/i })).toBeVisible();

  expect(consoleErrors.filter((line) => !line.includes('favicon'))).toEqual([]);
});
