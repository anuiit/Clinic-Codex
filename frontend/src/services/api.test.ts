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
