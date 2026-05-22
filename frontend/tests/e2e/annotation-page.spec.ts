import { test, expect, type Page } from 'playwright/test';
import { seedIndexedDbRecord } from './storageSeed';

const TEST_RECORD = {
  id: 'test-e2e-001',
  imageName: 'test-glyph.png',
  imageDataUrl:
    'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
  timestamp: 1700000000000,
  result: {
    num_elements: 1,
    image_size: [400, 300] as [number, number],
    elements: [
      {
        bbox: [50, 60, 80, 70] as [number, number, number, number],
        class_name: 'glyph-a',
        class_label: 0,
        confidence: 0.9,
        rejected: false,
        top_k: [],
      },
    ],
  },
  annotations: {},
  annotationStatus: { 0: 'validated' },
};

async function seedAndNavigate(page: Page, record = TEST_RECORD) {
  await page.goto('/');
  await seedIndexedDbRecord(page, record);
  await page.goto(`/annotate/${record.id}`);
}

test.beforeEach(async ({ page }) => {
  await page.route('**/classes', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ num_classes: 2, class_names: ['glyph-a', 'glyph-b'] }),
    });
  });
});

test('Cas 1 — local save shows FR success toast', async ({ page }) => {
  await seedAndNavigate(page);

  const saveBtn = page.getByRole('button', { name: /Enregistrer/i });
  await expect(saveBtn).toBeVisible({ timeout: 5000 });

  await saveBtn.click();

  const toast = page.locator('[class*="fixed"]').filter({ hasText: 'Annotations enregistrées localement' });
  await expect(toast).toBeVisible({ timeout: 3000 });
});

test('Cas 2 — remote save success shows FR toast', async ({ page }) => {
  await page.route('**/save-annotation', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        status: 'ok',
        analysis_id: 'test-e2e-001',
        saved_count: 2,
        classes: ['a', 'b'],
      }),
    });
  });

  await seedAndNavigate(page);

  const sendBtn = page.getByRole('button', { name: /Envoyer/i });
  await expect(sendBtn).toBeVisible({ timeout: 5000 });

  await sendBtn.click();

  const toast = page.locator('[class*="fixed"]').filter({ hasText: 'Annotations envoyées au serveur' });
  await expect(toast).toBeVisible({ timeout: 5000 });
});

test('Cas 3 — mode switch does not misalign boxes', async ({ page }) => {
  await seedAndNavigate(page);

  const bboxRect = page.locator('svg rect').first();
  await expect(bboxRect).toBeVisible({ timeout: 5000 });

  const beforeBox = await bboxRect.boundingBox();
  expect(beforeBox).not.toBeNull();

  const modeBtn = page.getByRole('button', { name: /Mode (dessin|sélection)/i });
  await expect(modeBtn).toBeVisible();
  await modeBtn.click();

  const afterBox = await bboxRect.boundingBox();
  expect(afterBox).not.toBeNull();

  expect(Math.abs(afterBox!.x - beforeBox!.x)).toBeLessThanOrEqual(2);
  expect(Math.abs(afterBox!.y - beforeBox!.y)).toBeLessThanOrEqual(2);
});

test('Cas 4 — zoom does not misalign boxes', async ({ page }) => {
  await seedAndNavigate(page);

  const bboxRect = page.locator('svg rect').first();
  const imageEl = page.locator('img').first();

  await expect(bboxRect).toBeVisible({ timeout: 5000 });
  await expect(imageEl).toBeVisible();

  const imgBefore = await imageEl.boundingBox();
  const rectBefore = await bboxRect.boundingBox();
  expect(imgBefore).not.toBeNull();
  expect(rectBefore).not.toBeNull();

  const relXBefore = rectBefore!.x - imgBefore!.x;
  const relYBefore = rectBefore!.y - imgBefore!.y;
  const normalizedXBefore = relXBefore / imgBefore!.width;
  const normalizedYBefore = relYBefore / imgBefore!.height;

  const zoomInButton = page.getByRole('button', { name: 'Zoom avant' });
  await expect(zoomInButton).toBeVisible();

  await zoomInButton.click();
  await zoomInButton.click();

  await page.waitForTimeout(200);

  const imgAfter = await imageEl.boundingBox();
  const rectAfter = await bboxRect.boundingBox();
  expect(imgAfter).not.toBeNull();
  expect(rectAfter).not.toBeNull();

  const relXAfter = rectAfter!.x - imgAfter!.x;
  const relYAfter = rectAfter!.y - imgAfter!.y;
  const normalizedXAfter = relXAfter / imgAfter!.width;
  const normalizedYAfter = relYAfter / imgAfter!.height;

  expect(Math.abs(normalizedXAfter - normalizedXBefore)).toBeLessThanOrEqual(0.01);
  expect(Math.abs(normalizedYAfter - normalizedYBefore)).toBeLessThanOrEqual(0.01);
});

test('Cas 5 — pan while zoomed moves image+boxes as one unit', async ({ page }) => {
  await seedAndNavigate(page);

  const bboxRect = page.locator('svg rect').first();
  const imageEl = page.locator('img').first();

  await expect(bboxRect).toBeVisible({ timeout: 5000 });
  await expect(imageEl).toBeVisible();

  const zoomInButton = page.getByRole('button', { name: 'Zoom avant' });
  await zoomInButton.click();
  await zoomInButton.click();
  await page.waitForTimeout(200);

  const imgBefore = await imageEl.boundingBox();
  const rectBefore = await bboxRect.boundingBox();
  expect(imgBefore).not.toBeNull();
  expect(rectBefore).not.toBeNull();

  const relXBefore = rectBefore!.x - imgBefore!.x;
  const relYBefore = rectBefore!.y - imgBefore!.y;

  const svgEl = page.locator('svg.absolute').first();
  await expect(svgEl).toBeVisible();

  const svgBox = await svgEl.boundingBox();
  expect(svgBox).not.toBeNull();

  const startX = svgBox!.x + svgBox!.width / 2;
  const startY = svgBox!.y + svgBox!.height / 2;

  await page.mouse.move(startX, startY);
  await page.mouse.down();
  await page.mouse.move(startX + 100, startY + 50, { steps: 10 });
  await page.mouse.up();

  await page.waitForTimeout(200);

  const imgAfter = await imageEl.boundingBox();
  const rectAfter = await bboxRect.boundingBox();
  expect(imgAfter).not.toBeNull();
  expect(rectAfter).not.toBeNull();

  const relXAfter = rectAfter!.x - imgAfter!.x;
  const relYAfter = rectAfter!.y - imgAfter!.y;

  expect(Math.abs(relXAfter - relXBefore)).toBeLessThanOrEqual(2);
  expect(Math.abs(relYAfter - relYBefore)).toBeLessThanOrEqual(2);

  const imgMovedX = Math.abs(imgAfter!.x - imgBefore!.x);
  const imgMovedY = Math.abs(imgAfter!.y - imgBefore!.y);
  expect(imgMovedX + imgMovedY).toBeGreaterThan(0);
});
