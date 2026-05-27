import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { expect, test, type Locator, type Page } from 'playwright/test';
import { clearIndexedDbRecords, seedIndexedDbRecord } from './storageSeed';

const imageFixturePath = fileURLToPath(
  new URL('../../src/test/fixtures/387_769v.jpg', import.meta.url),
);
const imageDataUrl = `data:image/jpeg;base64,${readFileSync(imageFixturePath).toString('base64')}`;

const imageSize = [750, 1210] as [number, number];
const targetBbox = [120, 240, 150, 180] as [number, number, number, number];
const expectedNormalized = {
  x: targetBbox[0] / imageSize[0],
  y: targetBbox[1] / imageSize[1],
  width: targetBbox[2] / imageSize[0],
  height: targetBbox[3] / imageSize[1],
};

const phase4Record = {
  id: 'phase4-handoff-record',
  imageName: '387_769v.jpg',
  imageDataUrl,
  timestamp: 1704067200000,
  result: {
    num_elements: 1,
    image_size: imageSize,
    elements: [
      {
        bbox: targetBbox,
        class_name: 'atl',
        class_label: 1,
        confidence: 0.9,
        rejected: false,
        top_k: [{ class_name: 'atl', confidence: 0.9 }],
      },
    ],
  },
  annotations: {},
  annotationStatus: { 0: 'validated' },
};

type NormalizedBBox = {
  x: number;
  y: number;
  width: number;
  height: number;
};

async function normalizedBBoxRelativeToImage(
  imageLocator: Locator,
  bboxLocator: Locator,
): Promise<NormalizedBBox> {
  const imageBox = await imageLocator.boundingBox();
  expect(imageBox).not.toBeNull();
  const overlayBox = await bboxLocator.evaluate((node) => {
    const element = node as SVGGraphicsElement;
    const svg = element.ownerSVGElement;
    if (!svg) {
      throw new Error('BBox rect is not attached to an SVG overlay');
    }

    const geometry = element.getBBox();
    const svgRect = svg.getBoundingClientRect();
    const viewBox = svg.viewBox.baseVal;
    return {
      bbox: {
        x: geometry.x,
        y: geometry.y,
        width: geometry.width,
        height: geometry.height,
      },
      svg: {
        x: svgRect.x,
        y: svgRect.y,
        width: svgRect.width,
        height: svgRect.height,
      },
      viewBox: {
        width: viewBox.width,
        height: viewBox.height,
      },
    };
  });

  return {
    x:
      (overlayBox.svg.x -
        imageBox!.x +
        (overlayBox.bbox.x / overlayBox.viewBox.width) * overlayBox.svg.width) /
      imageBox!.width,
    y:
      (overlayBox.svg.y -
        imageBox!.y +
        (overlayBox.bbox.y / overlayBox.viewBox.height) * overlayBox.svg.height) /
      imageBox!.height,
    width:
      ((overlayBox.bbox.width / overlayBox.viewBox.width) * overlayBox.svg.width) /
      imageBox!.width,
    height:
      ((overlayBox.bbox.height / overlayBox.viewBox.height) * overlayBox.svg.height) /
      imageBox!.height,
  };
}

function maxNormalizedDelta(actual: NormalizedBBox, expected: NormalizedBBox) {
  return Math.max(
    Math.abs(actual.x - expected.x),
    Math.abs(actual.y - expected.y),
    Math.abs(actual.width - expected.width),
    Math.abs(actual.height - expected.height),
  );
}

async function expectNormalizedCloseEventually(
  imageLocator: Locator,
  bboxLocator: Locator,
  expected: NormalizedBBox,
  tolerance = 0.02,
) {
  await expect
    .poll(
      async () =>
        maxNormalizedDelta(
          await normalizedBBoxRelativeToImage(imageLocator, bboxLocator),
          expected,
        ),
      { timeout: 5000 },
    )
    .toBeLessThanOrEqual(tolerance);
}

async function clearStorageAndOpenWorkspace(page: Page) {
  await page.goto('/');
  await clearIndexedDbRecords(page);
  await page.goto('/');
}

test.beforeEach(async ({ page }) => {
  await page.route('**/classes', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ num_classes: 1, class_names: ['atl'] }),
    });
  });

  await page.route('**/trust', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        query: { bbox: targetBbox, predicted_class: 'atl' },
        trust: {
          predicted_class_rank: 1,
          predicted_class_similarity: 0.91,
          top1_class: 'atl',
          top1_similarity: 0.91,
          margin_to_second: 0.4,
          above_rejection_threshold: true,
          rejection_threshold: 0.35,
          ambiguous: false,
          entropy: 0.12,
          top_k: [{ class_name: 'atl', confidence: 0.91 }],
        },
      }),
    });
  });
});

test('upload analysis renders Workspace bbox at normalized image ratios', async ({ page }) => {
  await page.route('**/segment', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(phase4Record.result),
    });
  });
  await clearStorageAndOpenWorkspace(page);

  await page.locator('input[type="file"]').setInputFiles(imageFixturePath);
  await expect(page.getByText('Image prête à analyser')).toBeVisible({ timeout: 5000 });
  await page.getByRole('button', { name: 'Analyser' }).last().click();

  const image = page.getByTestId('workspace-stage').getByRole('img', { name: '387_769v.jpg' });
  const bbox = page.getByTestId('image-bbox-stage-box-0');
  await expect(image).toBeVisible({ timeout: 5000 });
  await expect(bbox).toBeVisible({ timeout: 5000 });

  await expectNormalizedCloseEventually(image, bbox, expectedNormalized);
});

test('Workspace to Annotation handoff preserves normalized bbox ratios', async ({ page }) => {
  await clearStorageAndOpenWorkspace(page);
  await seedIndexedDbRecord(page, phase4Record);
  await page.goto(`/?analysis=${phase4Record.id}`);

  const workspaceImage = page.getByTestId('workspace-stage').getByRole('img', { name: '387_769v.jpg' });
  const workspaceBbox = page.getByTestId('image-bbox-stage-box-0');
  await expect(workspaceImage).toBeVisible({ timeout: 5000 });
  await expect(workspaceBbox).toBeVisible({ timeout: 5000 });
  await expectNormalizedCloseEventually(
    workspaceImage,
    workspaceBbox,
    expectedNormalized,
  );
  const workspaceNormalized = await normalizedBBoxRelativeToImage(
    workspaceImage,
    workspaceBbox,
  );

  await page.getByRole('button', { name: /atl région 0/i }).click();
  await page.getByRole('button', { name: 'Annoter la région' }).click();
  await expect(page).toHaveURL(new RegExp(`/annotate/${phase4Record.id}\\?element=0$`));

  const annotationImage = page.getByTestId('annotation-stage').getByRole('img', { name: '387_769v.jpg' });
  const annotationBbox = page.getByTestId('annotation-box-0');
  await expect(annotationImage).toBeVisible({ timeout: 5000 });
  await expect(annotationBbox).toBeVisible({ timeout: 5000 });

  await expectNormalizedCloseEventually(
    annotationImage,
    annotationBbox,
    workspaceNormalized,
  );
  await expectNormalizedCloseEventually(
    annotationImage,
    annotationBbox,
    expectedNormalized,
  );
});
