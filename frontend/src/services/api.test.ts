import { beforeEach, describe, expect, it, vi } from "vitest";

const axiosMock = vi.hoisted(() => ({
  get: vi.fn(),
  post: vi.fn(),
}));

vi.mock("axios", () => ({
  default: axiosMock,
}));

async function loadApi(baseUrl?: string) {
  vi.resetModules();
  vi.unstubAllEnvs();
  if (baseUrl !== undefined) {
    vi.stubEnv("VITE_API_BASE_URL", baseUrl);
  }
  return import("./api");
}

describe("API client contract", () => {
  beforeEach(() => {
    axiosMock.get.mockReset();
    axiosMock.post.mockReset();
    vi.unstubAllGlobals();
    vi.unstubAllEnvs();
  });

  it("uses the default base URL for GET /classes", async () => {
    axiosMock.get.mockResolvedValueOnce({ data: { num_classes: 2, class_names: ["atl", "tochtli"] } });
    const api = await loadApi();

    await expect(api.getClasses()).resolves.toEqual({ num_classes: 2, class_names: ["atl", "tochtli"] });

    expect(axiosMock.get).toHaveBeenCalledWith("http://localhost:7117/classes");
  });

  it("uses VITE_API_BASE_URL overrides and forwards AbortSignal for GET requests", async () => {
    const controller = new AbortController();
    axiosMock.get.mockResolvedValueOnce({ data: { num_classes: 1, class_names: ["atl"] } });
    const api = await loadApi("http://api.test");

    await api.getClasses({ signal: controller.signal });

    expect(axiosMock.get).toHaveBeenCalledWith("http://api.test/classes", { signal: controller.signal });
  });

  it("posts image FormData for /segment and preserves response data", async () => {
    const file = new File(["png"], "glyph.png", { type: "image/png" });
    const segment = { num_elements: 0, image_size: [8, 6], elements: [] };
    axiosMock.post.mockResolvedValueOnce({ data: segment });
    const api = await loadApi();

    await expect(api.segmentGlyph(file)).resolves.toEqual(segment);

    const [url, form] = axiosMock.post.mock.calls[0];
    expect(url).toBe("http://localhost:7117/segment");
    expect(form).toBeInstanceOf(FormData);
    expect((form as FormData).get("image")).toBe(file);
  });

  it("forwards AbortSignal for multipart image requests", async () => {
    const file = new File(["png"], "glyph.png", { type: "image/png" });
    const controller = new AbortController();
    axiosMock.post.mockResolvedValueOnce({ data: { class_name: "atl", confidence: 0.7, rejected: false, top_k: [] } });
    const api = await loadApi();

    await api.classifyElement(file, { signal: controller.signal });

    expect(axiosMock.post).toHaveBeenCalledWith(
      "http://localhost:7117/classify",
      expect.any(FormData),
      { signal: controller.signal },
    );
  });

  it("posts /similar JSON using base64 content, default limit, and prototype mode", async () => {
    const similar = { query: { bbox: [1, 2, 3, 4], mode: "prototype" }, best_match: { class_name: "atl", similarity: 0.7, rejected: false }, results: [] };
    axiosMock.post.mockResolvedValueOnce({ data: similar });
    const api = await loadApi();

    await expect(api.getSimilar("data:image/png;base64,abc123", [1, 2, 3, 4])).resolves.toEqual(similar);

    expect(axiosMock.post).toHaveBeenCalledWith("http://localhost:7117/similar", {
      image_base64: "abc123",
      bbox: [1, 2, 3, 4],
      limit: 5,
      mode: "prototype",
    });
  });

  it("posts /trust JSON using base64 content, predicted class, top_k, and AbortSignal", async () => {
    const trust = { query: { bbox: [1, 2, 3, 4], predicted_class: "atl" }, trust: { top_k: [] } };
    const controller = new AbortController();
    axiosMock.post.mockResolvedValueOnce({ data: trust });
    const api = await loadApi();

    await api.getTrust("data:image/png;base64,abc123", [1, 2, 3, 4], "atl", 7, { signal: controller.signal });

    expect(axiosMock.post).toHaveBeenCalledWith(
      "http://localhost:7117/trust",
      {
        image_base64: "abc123",
        bbox: [1, 2, 3, 4],
        predicted_class: "atl",
        top_k: 7,
      },
      { signal: controller.signal },
    );
  });

  it("normalizes saveAnnotation success responses without throwing", async () => {
    const payload = makeSavePayload();
    vi.stubGlobal("fetch", vi.fn(() => Promise.resolve(jsonResponse({ status: "ok", analysis_id: "a1", saved_count: 1, classes: ["atl"], saved_at: "now" }))));
    const api = await loadApi();

    await expect(api.saveAnnotation(payload)).resolves.toEqual({ ok: true, status: "ok", analysis_id: "a1", saved_count: 1, classes: ["atl"], saved_at: "now" });

    expect(fetch).toHaveBeenCalledWith("http://localhost:7117/save-annotation", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  });

  it("normalizes saveAnnotation storage/internal error responses", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.resolve(jsonResponse({ error_code: "INTERNAL_ERROR", message: "Erreur interne", trace_id: "trace-123" }, 500))));
    const api = await loadApi();

    await expect(api.saveAnnotation(makeSavePayload())).resolves.toEqual({
      ok: false,
      error_code: "INTERNAL_ERROR",
      message: "Erreur interne",
      hint: undefined,
      trace_id: "trace-123",
    });
  });

  it("normalizes canonical saveAnnotation validation errors", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() =>
        Promise.resolve(jsonResponse({ status: "error", error_code: "VALIDATION_ERROR", message: "annotations[0].bbox required", error: "annotations[0].bbox required" }, 400)),
      ),
    );
    const api = await loadApi();

    await expect(api.saveAnnotation(makeSavePayload())).resolves.toEqual({
      ok: false,
      error_code: "VALIDATION_ERROR",
      message: "annotations[0].bbox required",
      hint: undefined,
      trace_id: undefined,
    });
  });

  it("normalizes legacy saveAnnotation 400 validation errors as actionable failures", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() =>
        Promise.resolve(jsonResponse({ status: "error", error: "missing field: annotations" }, 400)),
      ),
    );
    const api = await loadApi();

    await expect(api.saveAnnotation(makeSavePayload())).resolves.toEqual({
      ok: false,
      error_code: "VALIDATION_ERROR",
      message: "missing field: annotations",
      hint: undefined,
      trace_id: undefined,
    });
  });

  it("normalizes saveAnnotation permission and disk-full error responses", async () => {
    const api = await loadApi();

    vi.stubGlobal(
      "fetch",
      vi.fn(() =>
        Promise.resolve(
          jsonResponse(
            {
              error_code: "PERMISSION_DENIED",
              message: "Droits insuffisants",
              hint: "check permissions",
            },
            409,
          ),
        ),
      ),
    );
    await expect(api.saveAnnotation(makeSavePayload())).resolves.toEqual({
      ok: false,
      error_code: "PERMISSION_DENIED",
      message: "Droits insuffisants",
      hint: "check permissions",
      trace_id: undefined,
    });

    vi.stubGlobal(
      "fetch",
      vi.fn(() =>
        Promise.resolve(
          jsonResponse(
            {
              error_code: "DISK_FULL",
              message: "Espace disque insuffisant",
            },
            507,
          ),
        ),
      ),
    );
    await expect(api.saveAnnotation(makeSavePayload())).resolves.toEqual({
      ok: false,
      error_code: "DISK_FULL",
      message: "Espace disque insuffisant",
      hint: undefined,
      trace_id: undefined,
    });
  });

  it("falls back safely when saveAnnotation receives a malformed internal error body", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() => Promise.resolve(jsonResponse({ trace_id: "trace-only" }, 500))),
    );
    const api = await loadApi();

    await expect(api.saveAnnotation(makeSavePayload())).resolves.toEqual({
      ok: false,
      error_code: "NETWORK_ERROR",
      message: "save-annotation failed: 500",
      hint: undefined,
      trace_id: "trace-only",
    });
  });

  it("uses NETWORK_ERROR fallback for non-JSON saveAnnotation failures", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.resolve(new Response("not-json", { status: 503 }))));
    const api = await loadApi();

    await expect(api.saveAnnotation(makeSavePayload())).resolves.toEqual({
      ok: false,
      error_code: "NETWORK_ERROR",
      message: "save-annotation failed: 503",
      hint: undefined,
      trace_id: undefined,
    });
  });

  it("uses NETWORK_ERROR fallback for thrown saveAnnotation network failures", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.reject(new Error("offline"))));
    const api = await loadApi();

    await expect(api.saveAnnotation(makeSavePayload())).resolves.toEqual({
      ok: false,
      error_code: "NETWORK_ERROR",
      message: "Network error while saving annotation",
    });
  });

  it("forwards AbortSignal for saveAnnotation fetch requests", async () => {
    const controller = new AbortController();
    vi.stubGlobal("fetch", vi.fn(() => Promise.resolve(jsonResponse({ status: "ok", analysis_id: "a1", saved_count: 1, classes: ["atl"] }))));
    const api = await loadApi();

    await api.saveAnnotation(makeSavePayload(), { signal: controller.signal });

    expect(fetch).toHaveBeenCalledWith(
      "http://localhost:7117/save-annotation",
      expect.objectContaining({ signal: controller.signal }),
    );
  });

  it("loads the local admin annotation queue", async () => {
    const queue = {
      status: "ok",
      schema_version: 1,
      local_only: true,
      warning: "local only",
      counts: { total: 1, pending: 1, approved: 0, rejected: 0, trainable: 0 },
      analyses: [],
      diagnostics: [],
    };
    const controller = new AbortController();
    axiosMock.get.mockResolvedValueOnce({ data: queue });
    const api = await loadApi("http://api.test");

    await expect(api.getAdminAnnotationQueue({ signal: controller.signal })).resolves.toEqual(queue);

    expect(axiosMock.get).toHaveBeenCalledWith(
      "http://api.test/admin/annotations",
      { signal: controller.signal },
    );
  });

  it("posts element-level local admin review decisions", async () => {
    const mutation = { status: "ok", local_only: true, warning: "local only", element: { key: "analysis 1:0" } };
    axiosMock.post.mockResolvedValueOnce({ data: mutation });
    const api = await loadApi("http://api.test");

    await expect(api.setAdminAnnotationReviewStatus("analysis 1", 0, "approved")).resolves.toEqual(mutation);

    expect(axiosMock.post).toHaveBeenCalledWith(
      "http://api.test/admin/annotations/analysis%201/0/review",
      { status: "approved" },
    );
  });

  it("posts element-level local admin modify payloads", async () => {
    const mutation = { status: "ok", local_only: true, warning: "local only", element: { key: "analysis 1:0" } };
    axiosMock.post.mockResolvedValueOnce({ data: mutation });
    const api = await loadApi("http://api.test");
    const payload = { class_name: "new-atl", bbox: [1, 2, 3, 4] as [number, number, number, number], approve_after_save: true };

    await expect(api.modifyAdminAnnotationElement("analysis 1", 0, payload)).resolves.toEqual(mutation);

    expect(axiosMock.post).toHaveBeenCalledWith(
      "http://api.test/admin/annotations/analysis%201/0/modify",
      payload,
    );
  });

  it("loads local admin training summary and latest job", async () => {
    const summary = { status: "ok", training_jobs_enabled: false };
    const latest = { status: "ok", local_only: true, job: null };
    axiosMock.get.mockResolvedValueOnce({ data: summary }).mockResolvedValueOnce({ data: latest });
    const api = await loadApi("http://api.test");

    await expect(api.getAdminTrainingSummary()).resolves.toEqual(summary);
    await expect(api.getLatestAdminTrainingJob()).resolves.toEqual(latest);

    expect(axiosMock.get).toHaveBeenNthCalledWith(1, "http://api.test/admin/training/summary");
    expect(axiosMock.get).toHaveBeenNthCalledWith(2, "http://api.test/admin/training/jobs/latest");
  });

  it("posts guarded local admin training jobs", async () => {
    const response = { status: "ok", local_only: true, job: { run_id: "r1", status: "running" } };
    const payload = { dry_run: true, device: "cpu", batch_size: 8, notes: "smoke" };
    axiosMock.post.mockResolvedValueOnce({ data: response });
    const api = await loadApi("http://api.test");

    await expect(api.startAdminTrainingJob(payload)).resolves.toEqual(response);

    expect(axiosMock.post).toHaveBeenCalledWith("http://api.test/admin/training/jobs", payload);
  });

  it("builds backend media URLs for local admin images", async () => {
    const api = await loadApi("http://api.test");

    expect(api.adminAnnotationMediaUrl("/admin/annotations/a1/image")).toBe(
      "http://api.test/admin/annotations/a1/image",
    );
    expect(api.adminAnnotationMediaUrl("https://cdn.example/crop.png")).toBe("https://cdn.example/crop.png");
  });
});

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function makeSavePayload() {
  return {
    analysis_id: "a1",
    image_name: "glyph.png",
    image_data_url: "data:image/png;base64,abc123",
    timestamp: 1770000000000,
    annotations: [{ index: 0, bbox: [0, 0, 5, 5] as [number, number, number, number], class_name: "atl" }],
  };
}
