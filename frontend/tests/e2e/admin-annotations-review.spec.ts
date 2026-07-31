import { expect, test, type Page } from "playwright/test";

const onePxPng =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
const wideSvg =
  '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="200" viewBox="0 0 400 200"><rect width="400" height="200" fill="#eee0c7"/><rect x="80" y="40" width="180" height="100" fill="#8a4f2d"/></svg>';

const adminQueue = {
  status: "ok",
  schema_version: 1,
  local_only: true,
  warning: "local queue",
  counts: { total: 2, pending: 2, approved: 0, rejected: 0, trainable: 0 },
  diagnostics: [],
  analyses: [
    {
      analysis_id: "review-smoke",
      uploaded_at: "2026-07-01T10:00:00+00:00",
      image_path: "/tmp/review-smoke.png",
      image_url: "/admin/annotations/review-smoke/image",
      image_exists: true,
      elements: [
        {
          key: "review-smoke:0",
          analysis_id: "review-smoke",
          index: 0,
          class_name: "Alpha",
          bbox: [80, 40, 180, 100],
          crop_path: "/tmp/review-smoke-0.png",
          crop_url: "/admin/annotations/review-smoke/0/crop",
          crop_exists: true,
          review_status: "pending",
          trainable: false,
          source_fingerprint: "review-smoke-fp-0",
          stale_decision: false,
          dataset_split: "excluded",
          split_reason: "pending_review",
        },
        {
          key: "review-smoke:1",
          analysis_id: "review-smoke",
          index: 1,
          class_name: "Beta",
          bbox: [20, 120, 50, 40],
          crop_path: "/tmp/review-smoke-1.png",
          crop_url: "/admin/annotations/review-smoke/1/crop",
          crop_exists: true,
          review_status: "pending",
          trainable: false,
          source_fingerprint: "review-smoke-fp-1",
          stale_decision: false,
          dataset_split: "excluded",
          split_reason: "pending_review",
        },
      ],
    },
  ],
};

function headers(type = "application/json") {
  return {
    "content-type": type,
    "access-control-allow-origin": "http://localhost:7118",
    "access-control-allow-credentials": "true",
    "access-control-allow-headers": "*",
    "access-control-allow-methods": "GET,POST,OPTIONS",
  };
}

async function installAdminRoutes(page: Page) {
  const png = Buffer.from(onePxPng.split(",")[1], "base64");

  await page.route("**/*", async (route) => {
    const request = route.request();
    const url = new URL(request.url());

    if (request.method() === "OPTIONS") {
      await route.fulfill({ status: 200, headers: headers() });
      return;
    }

    if (
      url.origin === "http://localhost:7117" ||
      url.origin === "http://127.0.0.1:7117"
    ) {
      if (url.pathname === "/auth/me") {
        await route.fulfill({
          status: 200,
          headers: headers(),
          body: JSON.stringify({ auth_enabled: false, user: null }),
        });
        return;
      }
      if (url.pathname === "/version") {
        await route.fulfill({
          status: 200,
          headers: headers(),
          body: JSON.stringify({
            app_name: "Clinic Codex",
            app_version: "0.1.0",
            model_version: "1.0.0",
          }),
        });
        return;
      }
      if (url.pathname === "/admin/annotations") {
        await route.fulfill({
          status: 200,
          headers: headers(),
          body: JSON.stringify(adminQueue),
        });
        return;
      }
      if (url.pathname === "/classes") {
        await route.fulfill({
          status: 200,
          headers: headers(),
          body: JSON.stringify({
            num_classes: 4,
            class_names: ["Alpha", "Beta", "Gamma", "Tochtli"],
          }),
        });
        return;
      }
      if (url.pathname === "/admin/annotations/review-smoke/image") {
        await route.fulfill({
          status: 200,
          headers: headers("image/svg+xml"),
          body: wideSvg,
        });
        return;
      }
      if (
        /^\/admin\/annotations\/review-smoke\/\d+\/crop$/.test(url.pathname)
      ) {
        await route.fulfill({
          status: 200,
          headers: headers("image/png"),
          body: png,
        });
        return;
      }
      if (
        /^\/admin\/annotations\/review-smoke\/\d+\/(review|modify)$/.test(
          url.pathname,
        )
      ) {
        await route.fulfill({
          status: 200,
          headers: headers(),
          body: JSON.stringify({
            status: "ok",
            local_only: true,
            warning: "",
            element: adminQueue.analyses[0].elements[0],
            counts: adminQueue.counts,
          }),
        });
        return;
      }
    }

    await route.continue();
  });
}

test.beforeEach(async ({ page }) => {
  await installAdminRoutes(page);
});

test("keeps the review workstation compact, bounded, and zoomable", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("/admin/annotations/review", { waitUntil: "networkidle" });

  await expect(page.getByRole("tab", { name: /trier/i })).toHaveAttribute(
    "aria-selected",
    "true",
  );
  await expect(page.getByRole("heading", { name: /#0.*alpha/i })).toBeVisible();

  const selectedRow = page.getByRole("option", {
    name: /ouvrir l'élément 0 alpha/i,
  });
  const rowThumb = selectedRow.locator('[data-reference-art="thumb"]');
  const rowTitle = selectedRow.locator(".admin-row-title");
  const selectedRowBox = await selectedRow.boundingBox();
  const rowThumbBox = await rowThumb.boundingBox();
  const rowTitleBox = await rowTitle.boundingBox();

  expect(selectedRowBox).not.toBeNull();
  expect(rowThumbBox).not.toBeNull();
  expect(rowTitleBox).not.toBeNull();
  expect(selectedRowBox!.height).toBeLessThanOrEqual(72);
  expect(rowThumbBox!.width).toBeGreaterThanOrEqual(44);
  expect(rowThumbBox!.height).toBeGreaterThanOrEqual(44);
  expect(rowTitleBox!.height).toBeLessThanOrEqual(24);
  await expect(rowTitle).toHaveCSS("white-space", "nowrap");

  await selectedRow.focus();
  await page.keyboard.press("ArrowDown");
  await expect(page.getByRole("heading", { name: /#1.*beta/i })).toBeVisible();
  await page.keyboard.press("ArrowUp");
  await expect(page.getByRole("heading", { name: /#0.*alpha/i })).toBeVisible();

  const imageShell = page.getByTestId("admin-review-image-shell");
  const imageHeader = page.getByTestId("admin-review-image-header");
  const imageStage = page.getByTestId("admin-review-image-stage");
  const imageTransform = page.getByTestId("admin-review-image-transform");
  const contextImage = page.getByRole("img", {
    name: /image complète review-smoke/i,
  });
  const cropPane = page.locator(
    '[data-reference-art="crop"][data-real-media="true"]',
  );
  const cropImage = cropPane.locator("img");

  await expect(imageShell).toBeVisible();
  await expect(imageStage).toBeVisible();
  await expect(cropPane).toBeVisible();
  await expect(contextImage).toBeVisible();
  await expect(cropImage).toBeVisible();
  await expect(cropImage).toHaveCSS("object-fit", "contain");
  await expect(imageHeader.getByText(/détails techniques \/ audit/i)).toBeVisible();
  await expect(
    page.locator(".admin-inspector-grid > .admin-audit-details"),
  ).toHaveCount(0);

  const stageBox = await imageStage.boundingBox();
  const transformBox = await imageTransform.boundingBox();
  expect(stageBox).not.toBeNull();
  expect(transformBox).not.toBeNull();
  expect(stageBox!.height).toBeGreaterThan(480);
  expect(Math.abs(transformBox!.width / transformBox!.height - 2)).toBeLessThan(
    0.02,
  );

  await page.setViewportSize({ width: 2048, height: 1109 });
  const wideStageBox = await imageStage.boundingBox();
  const wideTabMetrics = await page.locator(".admin-tab-content").evaluate(
    (element) => ({
      clientHeight: element.clientHeight,
      scrollHeight: element.scrollHeight,
    }),
  );
  expect(wideStageBox).not.toBeNull();
  expect(wideStageBox!.height).toBeGreaterThan(700);
  expect(wideTabMetrics.scrollHeight).toBeLessThanOrEqual(
    wideTabMetrics.clientHeight + 1,
  );
  await page.setViewportSize({ width: 1440, height: 900 });

  const inspectorMetrics = await page.locator(".admin-inspector-grid").evaluate(
    (element) => ({
      clientHeight: element.clientHeight,
      scrollHeight: element.scrollHeight,
      overflowY: getComputedStyle(element).overflowY,
    }),
  );
  const tabMetrics = await page.locator(".admin-tab-content").evaluate(
    (element) => ({
      clientHeight: element.clientHeight,
      scrollHeight: element.scrollHeight,
      overflowY: getComputedStyle(element).overflowY,
    }),
  );
  expect(inspectorMetrics.scrollHeight).toBeLessThanOrEqual(
    inspectorMetrics.clientHeight + 1,
  );
  expect(inspectorMetrics.overflowY).toBe("hidden");
  expect(tabMetrics.scrollHeight).toBeLessThanOrEqual(tabMetrics.clientHeight + 1);
  expect(tabMetrics.overflowY).toBe("hidden");

  await expect(imageShell).toContainText("100 %");
  await page.getByRole("button", { name: "Zoomer", exact: true }).click();
  await expect(imageShell).toContainText("125 %");
  await expect(imageTransform).toHaveCSS("transform", /matrix\(1\.25/);

  await page.getByRole("option", {
    name: /ouvrir l'élément 1 beta/i,
  }).click();
  await expect(page.getByRole("heading", { name: /#1.*beta/i })).toBeVisible();
  await expect(imageShell).toContainText("125 %");
  await expect(imageTransform).toHaveCSS("transform", /matrix\(1\.25/);

  await page.getByRole("button", { name: "Zoomer", exact: true }).focus();
  await page.keyboard.press("ArrowLeft");
  await expect(page.getByRole("heading", { name: /#0.*alpha/i })).toBeVisible();
  await expect(imageShell).toContainText("125 %");

  await page.getByRole("button", { name: "Ajuster à la vue" }).click();
  await expect(imageShell).toContainText("100 %");

  if (process.env.CAPTURE_ADMIN_VISUALS === "1") {
    for (const viewport of [
      { name: "wide", width: 2048, height: 1109 },
      { name: "desktop", width: 1440, height: 900 },
      { name: "tablet", width: 768, height: 1024 },
      { name: "mobile", width: 390, height: 844 },
    ]) {
      await page.setViewportSize(viewport);
      await expect(imageShell).toBeVisible();
      await page.evaluate(() => document.fonts.ready);
      await page.screenshot({
        path: `.omx-artifacts/admin-ui-harmonization/admin-review-${viewport.name}.png`,
        fullPage: false,
      });
    }
  }
});

test("keeps the admin decision context copy visible for the selected analysis", async ({
  page,
}) => {
  await page.goto("/admin/annotations/review", { waitUntil: "networkidle" });

  await expect(
    page.getByRole("heading", { name: /élément #0 · alpha/i }),
  ).toBeVisible();
  await expect(page.getByText("Image complète")).toBeVisible();
  await expect(page.getByText("Découpe à décider")).toBeVisible();
  await expect(page.locator(".admin-inspector-flags")).toBeVisible();
  await expect(
    page.getByRole("button", { name: /v · valider/i }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: /r · rejeter/i }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: /c · corriger/i }),
  ).toBeVisible();
});

test("preserves the natural image ratio while redrawing and shows correction context", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("/admin/annotations/review", { waitUntil: "networkidle" });

  const contextStrip = page.getByLabel("Contexte de l'annotation sélectionnée");
  await expect(contextStrip).toContainText("File 1 / 2");
  await expect(contextStrip).toContainText("Zone 180 × 100 px");
  await expect(contextStrip).toContainText("Pas encore prêt");

  await page.getByRole("button", { name: /c · corriger/i }).click();
  const stage = page.getByTestId("admin-segmentation-stage");
  const imageShell = page.getByTestId("admin-correction-image-shell");
  const imageTransform = page.getByTestId("admin-correction-image-transform");
  const sourceImage = page.getByRole("img", {
    name: /image à corriger review-smoke/i,
  });
  const overlay = page.getByLabel(/redessiner la segmentation de l'élément 0/i);
  const classInput = page.getByLabel("Nom de l'élément");

  await expect(contextStrip).toBeVisible();
  await expect(classInput).toBeFocused();
  await expect(imageShell).toContainText("2 éléments en contexte");
  await expect(
    imageShell.locator("svg text").filter({ hasText: "#0 Alpha" }),
  ).toBeVisible();
  await expect(
    imageShell.locator("svg text").filter({ hasText: "#1 Beta" }),
  ).toBeVisible();

  await expect(stage).toHaveAttribute("data-natural-width", "400");
  await expect(stage).toHaveAttribute("data-natural-height", "200");
  await expect(sourceImage).toHaveCSS("object-fit", "contain");
  const currentCrop = page.getByRole("img", {
    name: /découpe actuelle 0 pour alpha/i,
  });
  const unchangedPreview = page.getByRole("img", {
    name: /aperçu corrigé de l'élément 0/i,
  });
  await expect(currentCrop).toBeVisible();
  await expect(unchangedPreview).toBeVisible();
  const currentCropSrc = await currentCrop.getAttribute("src");
  expect(currentCropSrc).toContain("v=review-smoke-fp-0");
  await expect(unchangedPreview).toHaveAttribute("src", currentCropSrc!);

  const correctionSide = page.getByRole("complementary", {
    name: "Paramètres de correction",
  });
  const editor = page.locator(".admin-element-editor");
  const editorActions = page.locator(".admin-element-editor__actions");
  for (const viewport of [
    { width: 1440, height: 900 },
    { width: 2048, height: 1109 },
  ]) {
    await page.setViewportSize(viewport);
    const responsiveStageBox = await stage.boundingBox();
    const responsiveSideBox = await correctionSide.boundingBox();
    const editorBox = await editor.boundingBox();
    const actionsBox = await editorActions.boundingBox();
    const metrics = await page.evaluate(() => ({
      clientWidth: document.documentElement.clientWidth,
      scrollWidth: document.documentElement.scrollWidth,
      clientHeight: document.documentElement.clientHeight,
      scrollHeight: document.documentElement.scrollHeight,
    }));
    const sideMetrics = await correctionSide.evaluate((element) => ({
      clientHeight: element.clientHeight,
      scrollHeight: element.scrollHeight,
    }));

    expect(responsiveStageBox).not.toBeNull();
    expect(responsiveSideBox).not.toBeNull();
    expect(editorBox).not.toBeNull();
    expect(actionsBox).not.toBeNull();
    expect(metrics.scrollWidth).toBeLessThanOrEqual(metrics.clientWidth + 1);
    expect(metrics.scrollHeight).toBeLessThanOrEqual(metrics.clientHeight + 1);
    expect(sideMetrics.scrollHeight).toBeLessThanOrEqual(
      sideMetrics.clientHeight + 1,
    );
    expect(Math.abs(responsiveStageBox!.y - responsiveSideBox!.y)).toBeLessThanOrEqual(1);
    expect(
      Math.abs(responsiveStageBox!.height - responsiveSideBox!.height),
    ).toBeLessThanOrEqual(1);
    expect(responsiveStageBox!.width + responsiveSideBox!.width).toBeGreaterThan(
      editorBox!.width - 24,
    );
    expect(actionsBox!.y + actionsBox!.height).toBeLessThanOrEqual(
      responsiveSideBox!.y + responsiveSideBox!.height + 1,
    );
  }
  await page.setViewportSize({ width: 1440, height: 900 });

  const stageBox = await stage.boundingBox();
  const transformBox = await imageTransform.boundingBox();
  const imageBox = await sourceImage.boundingBox();
  const overlayBox = await overlay.boundingBox();
  expect(stageBox).not.toBeNull();
  expect(transformBox).not.toBeNull();
  expect(imageBox).not.toBeNull();
  expect(overlayBox).not.toBeNull();
  expect(Math.abs(transformBox!.width / transformBox!.height - 2)).toBeLessThan(
    0.02,
  );
  expect(Math.abs(imageBox!.width - transformBox!.width)).toBeLessThanOrEqual(1);
  expect(Math.abs(imageBox!.height - transformBox!.height)).toBeLessThanOrEqual(1);
  expect(Math.abs(overlayBox!.width - transformBox!.width)).toBeLessThanOrEqual(1);
  expect(Math.abs(overlayBox!.height - transformBox!.height)).toBeLessThanOrEqual(1);
  expect(stageBox!.width).toBeGreaterThanOrEqual(transformBox!.width);

  await page.getByRole("button", { name: "Zoomer", exact: true }).click();
  await expect(imageShell).toContainText("125 %");
  await page.getByRole("button", { name: "Ajuster à la vue" }).click();
  await expect(imageShell).toContainText("100 %");

  await classInput.fill("Gam");
  const suggestions = page.getByRole("listbox", { name: "Classes existantes" });
  await expect(suggestions).toBeVisible();
  await expect(suggestions.getByRole("option").filter({ hasText: "Gamma" })).toBeVisible();
  const suggestionSurface = await suggestions.evaluate((element) => {
    const style = getComputedStyle(element);
    return {
      backgroundColor: style.backgroundColor,
      opacity: style.opacity,
    };
  });
  expect(suggestionSurface.backgroundColor).not.toBe("transparent");
  expect(suggestionSurface.backgroundColor).not.toBe("rgba(0, 0, 0, 0)");
  expect(suggestionSurface.opacity).toBe("1");

  await classInput.fill("Delta");
  await suggestions
    .getByRole("option", { name: /créer la classe.*delta/i })
    .click();
  await expect(classInput).toHaveValue("Delta");

  await page.mouse.move(
    overlayBox!.x + overlayBox!.width * 0.75,
    overlayBox!.y + overlayBox!.height * 0.1,
  );
  await page.mouse.down();
  await page.mouse.move(
    overlayBox!.x + overlayBox!.width * 0.95,
    overlayBox!.y + overlayBox!.height * 0.4,
  );
  await page.mouse.up();

  await expect(page.getByLabel("Zone x pour l'élément 0")).toHaveValue("300");
  await expect(page.getByLabel("Zone y pour l'élément 0")).toHaveValue("20");
  await expect(page.getByLabel("Zone w pour l'élément 0")).toHaveValue("80");
  await expect(page.getByLabel("Zone h pour l'élément 0")).toHaveValue("60");
  await expect(
    page.getByTestId("admin-correction-preview-clip"),
  ).toHaveAttribute("x", "300");
  await expect(
    page.getByTestId("admin-correction-preview-clip"),
  ).toHaveAttribute("y", "20");
  await expect(
    page.getByTestId("admin-correction-preview-clip"),
  ).toHaveAttribute("width", "80");
  await expect(
    page.getByTestId("admin-correction-preview-clip"),
  ).toHaveAttribute("height", "60");

  if (process.env.CAPTURE_ADMIN_VISUALS === "1") {
    for (const viewport of [
      { name: "desktop", width: 1440, height: 1000 },
      { name: "tablet", width: 768, height: 1024 },
      { name: "mobile", width: 390, height: 844 },
    ]) {
      await page.setViewportSize(viewport);
      await expect(stage).toBeVisible();
      await page.evaluate(() => document.fonts.ready);
      const responsiveStageBox = await stage.boundingBox();
      const responsiveSideBox = await page
        .getByRole("complementary", { name: "Paramètres de correction" })
        .boundingBox();
      const documentWidth = await page.evaluate(
        () => document.documentElement.scrollWidth,
      );
      expect(responsiveStageBox).not.toBeNull();
      expect(responsiveSideBox).not.toBeNull();
      expect(documentWidth).toBeLessThanOrEqual(viewport.width + 1);

      if (viewport.width > 1100) {
        expect(Math.abs(responsiveStageBox!.y - responsiveSideBox!.y)).toBeLessThanOrEqual(1);
        expect(
          Math.abs(responsiveStageBox!.height - responsiveSideBox!.height),
        ).toBeLessThanOrEqual(1);
        expect(responsiveStageBox!.height).toBeGreaterThan(600);
      } else {
        expect(responsiveSideBox!.y).toBeGreaterThanOrEqual(
          responsiveStageBox!.y + responsiveStageBox!.height,
        );
      }

      const editorActionButtons = page.locator(
        ".admin-element-editor__actions button",
      );
      await expect(editorActionButtons).toHaveCount(3);
      for (let index = 0; index < 3; index += 1) {
        const actionBox = await editorActionButtons.nth(index).boundingBox();
        expect(actionBox).not.toBeNull();
        expect(actionBox!.width).toBeGreaterThanOrEqual(80);
        expect(actionBox!.height).toBeGreaterThanOrEqual(32);
      }
      await page.screenshot({
        path: `.omx-artifacts/admin-ui-harmonization/admin-correction-${viewport.name}.png`,
        fullPage: true,
      });
    }
  }

  const modifyRequest = page.waitForRequest(
    (request) =>
      request.method() === "POST" &&
      new URL(request.url()).pathname ===
        "/admin/annotations/review-smoke/0/modify",
  );
  await page.getByRole("button", { name: /^enregistrer$/i }).click();
  expect((await modifyRequest).postDataJSON()).toMatchObject({
    class_name: "Delta",
    bbox: [300, 20, 80, 60],
  });
  await expect(classInput).toBeVisible();
  await expect(
    page.getByText(/enregistrer conserve cette correction ouverte/i),
  ).toBeVisible();
});
