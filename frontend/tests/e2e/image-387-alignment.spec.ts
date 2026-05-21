import { readFileSync } from 'node:fs';
import { test, expect, type Page } from 'playwright/test';

const imageBytes = readFileSync(new URL('../../src/test/fixtures/387_769v.jpg', import.meta.url));
const imageDataUrl = `data:image/jpeg;base64,${imageBytes.toString('base64')}`;

const RECORD = {
  id: 'image-387-769v-e2e',
  imageName: '387_769v.jpg',
  imageDataUrl,
  timestamp: 1704067200000,
  result: {
    num_elements: 1,
    image_size: [750, 1210] as [number, number],
    elements: [
      {
        bbox: [120, 240, 150, 180] as [number, number, number, number],
        class_name: 'atl',
        class_label: 1,
        confidence: 0.9,
        rejected: false,
        top_k: [],
      },
    ],
  },
  annotations: {},
  annotationStatus: { 0: 'validated' },
};

async function seedAndNavigate(page: Page) {
  await page.goto('/');
  await page.evaluate((record) => {
    localStorage.setItem('codex_analyses', JSON.stringify([record]));
  }, RECORD);
  await page.goto(`/annotate/${RECORD.id}`);
}

test.beforeEach(async ({ page }) => {
  await page.route('**/classes', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ num_classes: 1, class_names: ['atl'] }),
    });
  });
});

test('387_769v image and bbox stay aligned through zoom and pan', async ({ page }) => {
  await seedAndNavigate(page);

  const imageEl = page.getByAltText('387_769v.jpg');
  const bboxRect = page.locator('svg.absolute > g rect').first();
  await expect(imageEl).toBeVisible({ timeout: 5000 });
  await expect(bboxRect).toBeVisible({ timeout: 5000 });

  const imageBefore = await imageEl.boundingBox();
  const rectBefore = await bboxRect.boundingBox();
  expect(imageBefore).not.toBeNull();
  expect(rectBefore).not.toBeNull();

  const relXBefore = rectBefore!.x - imageBefore!.x;
  const relYBefore = rectBefore!.y - imageBefore!.y;

  const zoomGroup = page.locator('.flex.items-center.gap-1.rounded-lg.border');
  const zoomInButton = zoomGroup.locator('button').first();
  await zoomInButton.click();
  await zoomInButton.click();

  const svgEl = page.locator('svg.absolute').first();
  const svgBox = await svgEl.boundingBox();
  expect(svgBox).not.toBeNull();

  const startX = svgBox!.x + svgBox!.width - 20;
  const startY = svgBox!.y + svgBox!.height - 20;
  await page.mouse.move(startX, startY);
  await page.mouse.down();
  await page.mouse.move(startX + 80, startY + 40, { steps: 10 });
  await page.mouse.up();

  const imageAfter = await imageEl.boundingBox();
  const rectAfter = await bboxRect.boundingBox();
  expect(imageAfter).not.toBeNull();
  expect(rectAfter).not.toBeNull();

  const relXAfter = rectAfter!.x - imageAfter!.x;
  const relYAfter = rectAfter!.y - imageAfter!.y;

  expect(Math.abs(relXAfter - relXBefore * 1.5)).toBeLessThanOrEqual(3);
  expect(Math.abs(relYAfter - relYBefore * 1.5)).toBeLessThanOrEqual(3);
  expect(Math.abs(imageAfter!.x - imageBefore!.x) + Math.abs(imageAfter!.y - imageBefore!.y)).toBeGreaterThan(0);
});
