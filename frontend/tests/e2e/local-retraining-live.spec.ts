import { expect, test } from "playwright/test";
import { createHash, randomUUID } from "node:crypto";
import { readFileSync } from "node:fs";
import path from "node:path";

test("new local user annotates, approves and creates a candidate with real models", async ({ page }) => {
  test.skip(process.env.CLINIC_LIVE_E2E !== "1", "opt-in: requires a fresh installed local backend");
  test.setTimeout(240_000);
  page.setDefaultTimeout(20_000);
  const root = path.resolve("..");
  const protectedFiles = ["config.json", "weights/projection.pt", "weights/prototypes.pt"];
  const hashes = () => protectedFiles.map(file => createHash("sha256")
    .update(readFileSync(path.join(root, "backend/codex_model", file))).digest("hex"));
  const before = hashes();
  const failures: string[] = [];
  page.on("pageerror", error => failures.push(error.message));
  await page.goto("/login");
  await expect(page.getByRole("heading", { name: "Créer le premier compte administrateur" })).toBeVisible();
  const password = randomUUID();
  await page.getByLabel("Email", { exact: true }).fill("new-user@example.test");
  await page.getByLabel("Password", { exact: true }).fill(password);
  await page.getByLabel("Confirm password", { exact: true }).fill(password);
  await page.getByRole("button", { name: "Create administrator account", exact: true }).click();
  await expect(page).toHaveURL(/admin\/annotations/);
  await page.goto("/");
  await page.locator('input[type="file"]').setInputFiles(
    path.join(root, "backend/data/elements_sample/0015-cacahuatl/03_04_22-27.bmp"),
  );
  const segmented = page.waitForResponse(response => response.url().endsWith("/segment"), { timeout: 120_000 });
  await page.getByRole("button", { name: "Analyser", exact: true }).last().click();
  expect((await segmented).status()).toBe(200);
  await page.getByRole("button", { name: "Annoter l’analyse", exact: true }).click();
  await expect(page).toHaveURL(/\/annotate\//);
  await page.locator(".annotation-card").first().click();
  const name = page.getByRole("combobox", { name: /Nommer l’élément/ });
  await name.fill("cacahuatl");
  await name.press("Enter");
  await page.getByRole("button", { name: "Marquer comme prêt", exact: true }).click();
  const submitted = page.waitForResponse(response => response.url().endsWith("/save-annotation"));
  await page.getByRole("button", { name: "Envoyer pour revue", exact: true }).click();
  expect((await submitted).status()).toBe(200);
  await page.goto("/admin/annotations/review");
  const approved = page.waitForResponse(response =>
    /\/admin\/annotations\/[^/]+\/\d+\/review$/.test(response.url()));
  await page.getByRole("button", { name: "V · Valider", exact: true }).click();
  expect((await approved).status()).toBe(200);
  await page.getByRole("tab", { name: /Entraîner/ }).click();
  await page.getByRole("combobox", { name: "Machine", exact: true }).selectOption("cpu");
  await page.getByRole("spinbutton", { name: "Taille de lot", exact: true }).fill("1");
  await page.getByRole("button", { name: "Lancer l'essai à blanc", exact: true }).click();
  await expect(page.locator(".admin-training-job-panel").getByText("succeeded", { exact: true })).toBeVisible({ timeout: 120_000 });
  await page.getByRole("combobox", { name: "Essai à blanc", exact: true }).selectOption("no");
  await page.getByRole("button", { name: "Lancer l'entraînement", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Candidat créé — non activé", exact: true })).toBeVisible({ timeout: 120_000 });
  await expect(page.locator(".admin-training-job-panel").getByText("succeeded", { exact: true })).toBeVisible();
  await page.reload();
  await expect(page.getByRole("heading", { name: "Candidat créé — non activé", exact: true })).toBeVisible();
  expect(hashes()).toEqual(before);
  expect(failures).toEqual([]);
  const resultPanel = page.getByRole("region", { name: "Résultat du candidat", exact: true });
  await resultPanel.scrollIntoViewIfNeeded();
  await expect(resultPanel).toBeInViewport();
  await page.screenshot({ path: "test-results/local-retraining-candidate.png", fullPage: true });
});
