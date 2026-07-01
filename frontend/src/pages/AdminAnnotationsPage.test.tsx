import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AdminAnnotationsPage from "./AdminAnnotationsPage";
import type {
  AdminAnnotationQueue,
  AdminTrainingJob,
  AdminTrainingSummary,
} from "../types";

const apiMock = vi.hoisted(() => ({
  adminAnnotationMediaUrl: vi.fn((path: string) => `http://api.test${path}`),
  getAdminAnnotationQueue: vi.fn(),
  getAdminTrainingSummary: vi.fn(),
  getLatestAdminTrainingJob: vi.fn(),
  modifyAdminAnnotationElement: vi.fn(),
  setAdminAnnotationReviewStatus: vi.fn(),
  startAdminTrainingJob: vi.fn(),
}));

vi.mock("../services/api", () => apiMock);

function queueWithStatuses(
  status0: "pending" | "approved" | "rejected",
  status1: "pending" | "approved" | "rejected",
): AdminAnnotationQueue {
  const trainable0 = status0 === "approved";
  const trainable1 = status1 === "approved";
  return {
    status: "ok",
    schema_version: 1,
    local_only: true,
    warning:
      "Local/dev-only annotation review endpoint. It is not production-secured.",
    counts: {
      total: 2,
      pending: [status0, status1].filter((status) => status === "pending")
        .length,
      approved: [status0, status1].filter((status) => status === "approved")
        .length,
      rejected: [status0, status1].filter((status) => status === "rejected")
        .length,
      trainable: [trainable0, trainable1].filter(Boolean).length,
    },
    analyses: [
      {
        analysis_id: "analysis-1",
        uploaded_at: "2026-05-26T13:00:00+00:00",
        image_path: "/tmp/annotations/analysis-1/image.png",
        image_url: "/admin/annotations/analysis-1/image",
        image_exists: true,
        elements: [
          {
            key: "analysis-1:0",
            analysis_id: "analysis-1",
            index: 0,
            class_name: "atl",
            bbox: [0, 1, 2, 3],
            crop_path: "/tmp/annotations/analysis-1/elements/0.png",
            crop_url: "/admin/annotations/analysis-1/0/crop",
            crop_exists: true,
            review_status: status0,
            trainable: trainable0,
            source_fingerprint: "fingerprint-0",
            stale_decision: false,
            dataset_split: trainable0 ? "train" : "excluded",
            split_reason: trainable0 ? "trainable_hash_80_10_10" : status0 === "rejected" ? "rejected_review" : "pending_review",
          },
          {
            key: "analysis-1:1",
            analysis_id: "analysis-1",
            index: 1,
            class_name: "calli",
            bbox: [4, 5, 6, 7],
            crop_path: "/tmp/annotations/analysis-1/elements/1.png",
            crop_url: "/admin/annotations/analysis-1/1/crop",
            crop_exists: true,
            review_status: status1,
            trainable: trainable1,
            source_fingerprint: "fingerprint-1",
            stale_decision: false,
            dataset_split: trainable1 ? "val" : "excluded",
            split_reason: trainable1 ? "trainable_hash_80_10_10" : status1 === "rejected" ? "rejected_review" : "pending_review",
          },
        ],
      },
    ],
    diagnostics: [],
  };
}

function trainingJob(
  overrides: Partial<AdminTrainingJob> = {},
): AdminTrainingJob {
  return {
    run_id: "run-1",
    status: "running",
    dry_run: true,
    device: "cpu",
    batch_size: 8,
    started_at: "2026-05-26T14:00:00+00:00",
    exit_code: null,
    log_tail: ["stage 1"],
    ...overrides,
  };
}

function trainingSummary(
  overrides: Partial<AdminTrainingSummary> = {},
): AdminTrainingSummary {
  return {
    status: "ok",
    local_only: true,
    warning: "local only",
    training_jobs_enabled: false,
    launch_allowed_for_request: false,
    launch_disabled_reasons: [
      "disabled_by_default: set ENABLE_ADMIN_TRAINING_JOBS=1 to allow local launches",
    ],
    data: {
      total: 4,
      pending: 1,
      approved: 2,
      rejected: 1,
      trainable: 1,
      classes: ["atl"],
      per_class: { atl: 1 },
      split_counts: { train: 1, val: 0, test: 0, excluded: 3 },
      diagnostics: [],
    },
    parameters: {
      editable: {
        dry_run: true,
        device: ["auto", "cpu", "mps", "cuda"],
        batch_size: { default: 16, min: 1, max: 256 },
      },
      script_env_defaults: { BATCH_SIZE: "16", DEVICE: "auto" },
      config: {
        training: { num_epochs: 100 },
        model: { backbone: "dinov2_vits14" },
      },
    },
    paths: {
      script: "/repo/scripts/retrain.sh",
      runs_dir: "/repo/backend/training_runs",
      model_dir_override_active: false,
    },
    artifacts: {
      approved_export_manifest: {
        path: "/repo/backend/training_data/approved/Elements/_approved_export_manifest.json",
        exists: true,
        sha256: "abc123",
      },
      model_registry: {
        status: "ok",
        promoted_version: "20260527T010203Z-demo",
      },
    },
    latest_job: null,
    ...overrides,
  };
}

async function renderLoaded(queue = queueWithStatuses("pending", "rejected")) {
  apiMock.getAdminAnnotationQueue.mockResolvedValueOnce(queue);
  render(<AdminAnnotationsPage themeMode="light" onToggleTheme={vi.fn()} />);
  await screen.findByRole("heading", { name: /poste de triage/i });
  return screen.findByRole("listbox", { name: /file de triage/i });
}

describe("AdminAnnotationsPage", () => {
  beforeEach(() => {
    apiMock.adminAnnotationMediaUrl.mockClear();
    apiMock.getAdminAnnotationQueue.mockReset();
    apiMock.getAdminTrainingSummary.mockReset();
    apiMock.getLatestAdminTrainingJob.mockReset();
    apiMock.modifyAdminAnnotationElement.mockReset();
    apiMock.setAdminAnnotationReviewStatus.mockReset();
    apiMock.startAdminTrainingJob.mockReset();
  });

  it("renders the reference visual triage workstation with rows and decision inspector", async () => {
    await renderLoaded();

    expect(
      screen.getByRole("tablist", { name: /étapes du poste de triage/i }),
    ).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /trier/i })).toHaveAttribute(
      "aria-selected",
      "true",
    );
    expect(screen.getByRole("tab", { name: /dataset/i })).toHaveAttribute(
      "aria-selected",
      "false",
    );
    expect(screen.getByRole("tab", { name: /entraîner/i })).toHaveAttribute(
      "aria-selected",
      "false",
    );
    expect(screen.getByText(/codex-014/i)).toBeInTheDocument();
    expect(screen.getByText(/restants/i)).toBeInTheDocument();
    expect(screen.getByText(/inclus/i)).toBeInTheDocument();
    expect(screen.getByText(/local/i)).toBeInTheDocument();

    expect(
      screen.getByRole("listbox", { name: /file de triage/i }),
    ).toBeInTheDocument();
    const selectedReviewRow = screen.getByRole("option", {
      name: /ouvrir l'élément 0 atl du triage/i,
    });
    expect(selectedReviewRow).toHaveAttribute("aria-selected", "true");
    expect(selectedReviewRow).toHaveAccessibleName(/statut À vérifier/i);
    expect(
      screen.getByRole("heading", { name: /élément #0 · atl/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("img", { name: /image complète analysis-1/i }),
    ).toHaveAttribute(
      "src",
      "http://api.test/admin/annotations/analysis-1/image",
    );
    expect(
      screen.getByRole("img", { name: /découpe 0 pour atl/i }),
    ).toBeInTheDocument();
    expect(screen.getByText(/effet sur le dataset/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /valider/i })).toBeEnabled();
    expect(screen.queryByRole("button", { name: /retrain/i })).not.toBeInTheDocument();
  });

  it("filters triage rows, recovers empty filters, and manually refreshes the queue", async () => {
    const user = userEvent.setup();
    apiMock.getAdminAnnotationQueue
      .mockResolvedValueOnce(queueWithStatuses("pending", "rejected"))
      .mockResolvedValueOnce(queueWithStatuses("approved", "rejected"));

    render(<AdminAnnotationsPage />);

    expect(await screen.findByText(/2 \/ 2 éléments affichés/i)).toBeInTheDocument();

    await user.selectOptions(screen.getByRole("combobox", { name: /^statut$/i }), "rejected");
    expect(screen.getByText(/1 \/ 2 éléments affiché/i)).toBeInTheDocument();
    expect(
      screen.getByRole("list", { name: /filtres de triage appliqués/i }),
    ).toHaveTextContent(/statut : rejetés/i);

    await user.selectOptions(screen.getByRole("combobox", { name: /^classe$/i }), "atl");
    expect(
      screen.getByText(/aucun élément ne correspond aux filtres actifs/i),
    ).toBeInTheDocument();
    await user.click(screen.getAllByRole("button", { name: /effacer les filtres/i }).at(-1)!);
    expect(
      screen.getByRole("option", { name: /ouvrir l'élément 0 atl/i }),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /actualiser/i }));
    await waitFor(() =>
      expect(apiMock.getAdminAnnotationQueue).toHaveBeenCalledTimes(2),
    );
    expect(
      await screen.findByLabelText(/élément sélectionné 1 calli : statut Rejeté/i),
    ).toBeInTheDocument();
  });

  it("keeps current queue visible when manual refresh fails", async () => {
    const user = userEvent.setup();
    apiMock.getAdminAnnotationQueue
      .mockResolvedValueOnce(queueWithStatuses("pending", "rejected"))
      .mockRejectedValueOnce(new Error("offline"));

    render(<AdminAnnotationsPage />);

    expect(
      await screen.findByRole("option", { name: /ouvrir l'élément 0 atl/i }),
    ).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /actualiser/i }));

    expect(
      await screen.findByText(/impossible de charger la file locale de triage/i),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("option", { name: /ouvrir l'élément 0 atl/i }),
    ).toBeInTheDocument();
  });

  it("selects triage rows and navigates the inspector within active filters", async () => {
    const user = userEvent.setup();
    await renderLoaded();

    const firstRow = screen.getByRole("option", {
      name: /ouvrir l'élément 0 atl/i,
    });
    const secondRow = screen.getByRole("option", {
      name: /ouvrir l'élément 1 calli/i,
    });
    expect(firstRow).toHaveAttribute("aria-current", "true");

    await user.click(secondRow);

    expect(
      screen.getByRole("heading", { name: /élément #1 · calli/i }),
    ).toBeInTheDocument();
    expect(secondRow).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("button", { name: /suivant/i })).toBeDisabled();

    await user.click(screen.getByRole("button", { name: /précédent/i }));
    expect(
      screen.getByRole("heading", { name: /élément #0 · atl/i }),
    ).toBeInTheDocument();

    await user.selectOptions(screen.getByRole("combobox", { name: /^statut$/i }), "rejected");
    expect(
      screen.getByRole("option", { name: /ouvrir l'élément 1 calli/i }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("option", { name: /ouvrir l'élément 0 atl/i }),
    ).not.toBeInTheDocument();
  });

  it("separates dataset buckets with filters and jumps back to triage", async () => {
    const user = userEvent.setup();
    const mixed = queueWithStatuses("approved", "approved");
    mixed.analyses[0].elements[1] = {
      ...mixed.analyses[0].elements[1],
      trainable: false,
      stale_decision: true,
      dataset_split: "excluded",
      split_reason: "stale_decision",
    };
    mixed.analyses[0].elements.push(
      {
        ...mixed.analyses[0].elements[0],
        key: "analysis-1:2",
        index: 2,
        class_name: "atl",
        bbox: [2, 2, 4, 4],
        crop_url: "/admin/annotations/analysis-1/2/crop",
        review_status: "rejected",
        trainable: false,
        dataset_split: "excluded",
        split_reason: "rejected_review",
        source_fingerprint: "fingerprint-2",
      },
      {
        ...mixed.analyses[0].elements[0],
        key: "analysis-1:3",
        index: 3,
        class_name: "maya",
        bbox: [3, 3, 4, 4],
        crop_url: "/admin/annotations/analysis-1/3/crop",
        review_status: "pending",
        trainable: false,
        dataset_split: "excluded",
        split_reason: "pending_review",
        source_fingerprint: "fingerprint-3",
      },
    );
    mixed.counts = {
      total: 4,
      pending: 1,
      approved: 2,
      rejected: 1,
      trainable: 1,
    };
    mixed.diagnostics = [
      {
        code: "stale_decision",
        message: "fingerprint mismatch",
        key: "analysis-1:1",
        analysis_id: "analysis-1",
        index: 1,
      },
    ];
    await renderLoaded(mixed);
    await user.click(screen.getByRole("tab", { name: /dataset/i }));

    expect(screen.getByLabelText(/classes dataset/i)).toBeInTheDocument();
    const datasetList = screen.getByRole("listbox", { name: /vue dataset/i });
    expect(datasetList).toBeInTheDocument();
    expect(
      within(datasetList).getByRole("option", { name: /ouvrir élément dataset 0 atl/i }),
    ).toHaveAttribute("aria-selected", "true");
    expect(
      screen.getAllByRole("img", { name: /découpe dataset 0 pour atl/i })[0],
    ).toBeInTheDocument();
    expect(
      within(datasetList).queryByRole("option", { name: /ouvrir élément dataset 1 calli/i }),
    ).not.toBeInTheDocument();
    expect(
      within(screen.getByLabelText(/filtres split dataset/i)).queryByRole("button", { name: /exclus/i }),
    ).not.toBeInTheDocument();
    expect(screen.getByText(/1 \/ 1 élément du dataset affiché/i)).toBeInTheDocument();

    await user.click(within(screen.getByLabelText(/classes dataset/i)).getByRole("button", { name: /atl/i }));
    expect(screen.getByText(/1 \/ 1 élément du dataset affiché/i)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /ouvrir dans le triage/i }));

    expect(screen.getByRole("tab", { name: /trier/i })).toHaveAttribute(
      "aria-selected",
      "true",
    );
    expect(
      screen.getByRole("heading", { name: /élément #2 · atl/i }),
    ).toBeInTheDocument();
  });

  it("renders Training disabled summary, guardrails, and readable metadata", async () => {
    const user = userEvent.setup();
    apiMock.getAdminAnnotationQueue.mockResolvedValueOnce(
      queueWithStatuses("approved", "rejected"),
    );
    apiMock.getAdminTrainingSummary.mockResolvedValueOnce(trainingSummary());

    render(<AdminAnnotationsPage />);
    await user.click(await screen.findByRole("tab", { name: /entraîner/i }));

    expect(
      await screen.findByRole("heading", { name: /assistant d'entraînement local/i }),
    ).toBeInTheDocument();
    expect(screen.getAllByText(/lancement bloqué/i).length).toBeGreaterThan(0);
    expect(screen.queryByText(/disabled_by_default/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/enable_admin_training_jobs=1/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/bash scripts\/retrain\.sh --dry-run/i)).not.toBeInTheDocument();
    expect(screen.getAllByText(/protection locale active/i).length).toBeGreaterThan(0);
    expect(document.body).not.toHaveTextContent(
      /placeholder|LossSketch|ValidationAccuracySketch/i,
    );
    expect(
      screen.getByRole("region", { name: /résumé avant lancement de l'entraînement/i }),
    ).toHaveTextContent(/bloquée par la protection backend/i);
    expect(screen.getByText(/l'essai à blanc vérifie/i)).toBeInTheDocument();
    expect(screen.getByText(/validés par classe/i)).toBeInTheDocument();
    expect(screen.getByText(/atl: 1/i)).toBeInTheDocument();
    expect(screen.getByText(/fichiers produits/i)).toBeInTheDocument();
    expect(screen.getAllByText(/approved_export_manifest/i).length).toBeGreaterThan(0);
    expect(
      screen.getByRole("button", { name: /lancer l'essai à blanc/i }),
    ).toBeDisabled();
  });

  it("surfaces incomplete Training split metadata instead of showing fake zero counts", async () => {
    const user = userEvent.setup();
    const incompleteSummary = trainingSummary();
    delete (incompleteSummary.data as Partial<typeof incompleteSummary.data>).split_counts;
    apiMock.getAdminAnnotationQueue.mockResolvedValueOnce(
      queueWithStatuses("approved", "rejected"),
    );
    apiMock.getAdminTrainingSummary.mockResolvedValueOnce(incompleteSummary);

    render(<AdminAnnotationsPage />);
    await user.click(await screen.findByRole("tab", { name: /entraîner/i }));

    expect(
      await screen.findByRole("alert", { name: "" }),
    ).toHaveTextContent(/résumé d'entraînement incomplet/i);
    expect(screen.queryByText(/split locked: train 0/i)).not.toBeInTheDocument();
  });

  it("updates Training pre-action summary for dry-run versus full-training choices", async () => {
    const user = userEvent.setup();
    apiMock.getAdminAnnotationQueue.mockResolvedValueOnce(
      queueWithStatuses("approved", "rejected"),
    );
    apiMock.getAdminTrainingSummary.mockResolvedValueOnce(
      trainingSummary({
        training_jobs_enabled: true,
        launch_allowed_for_request: true,
        launch_disabled_reasons: [],
      }),
    );

    render(<AdminAnnotationsPage />);
    await user.click(await screen.findByRole("tab", { name: /entraîner/i }));

    const summary = await screen.findByRole("region", {
      name: /résumé avant lancement de l'entraînement/i,
    });
    expect(within(summary).getByText(/prêt pour essai local/i)).toBeInTheDocument();
    expect(within(summary).getByText(/essai à blanc sélectionné/i)).toBeInTheDocument();
    expect(within(summary).getByText(/sans écrire d'artefact/i)).toBeInTheDocument();
    expect(within(summary).getAllByText(/auto/i).length).toBeGreaterThan(0);
    expect(within(summary).getByText("16")).toBeInTheDocument();

    await user.selectOptions(screen.getByLabelText(/essai à blanc/i), "no");
    await user.selectOptions(screen.getByLabelText(/machine/i), "cpu");
    await user.clear(screen.getByLabelText(/taille de lot/i));
    await user.type(screen.getByLabelText(/taille de lot/i), "8");

    expect(within(summary).getByText(/entraînement complet sélectionné/i)).toBeInTheDocument();
    expect(within(summary).getByText(/paquet candidat local/i)).toBeInTheDocument();
    expect(within(summary).getAllByText(/promotion explicite/i).length).toBeGreaterThan(0);
    expect(within(summary).getByText(/cpu/i)).toBeInTheDocument();
    expect(within(summary).getByText("8")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /lancer l'entraînement/i }),
    ).toBeEnabled();
  });

  it("starts a guarded dry-run training job and hides log tail behind support details", async () => {
    const user = userEvent.setup();
    const started = trainingJob({
      run_id: "run-started",
      status: "running",
      command: ["bash", "scripts/retrain.sh", "--dry-run"],
      log_tail: ["mock dry run started"],
    });
    apiMock.getAdminAnnotationQueue.mockResolvedValueOnce(
      queueWithStatuses("approved", "rejected"),
    );
    apiMock.getAdminTrainingSummary.mockResolvedValueOnce(
      trainingSummary({
        training_jobs_enabled: true,
        launch_allowed_for_request: true,
        launch_disabled_reasons: [],
      }),
    );
    apiMock.startAdminTrainingJob.mockResolvedValueOnce({
      status: "ok",
      local_only: true,
      job: started,
    });

    render(<AdminAnnotationsPage />);
    await user.click(await screen.findByRole("tab", { name: /entraîner/i }));
    await screen.findByRole("button", { name: /lancer l'essai à blanc/i });
    await user.selectOptions(screen.getByLabelText(/machine/i), "cpu");
    await user.clear(screen.getByLabelText(/taille de lot/i));
    await user.type(screen.getByLabelText(/taille de lot/i), "8");
    await user.type(screen.getByLabelText(/notes/i), "smoke");

    await user.click(screen.getByRole("button", { name: /lancer l'essai à blanc/i }));

    await waitFor(() => {
      expect(apiMock.startAdminTrainingJob).toHaveBeenCalledWith({
        dry_run: true,
        device: "cpu",
        batch_size: 8,
        notes: "smoke",
      });
    });
    expect(await screen.findByText(/dernier essai run-started/i)).toBeInTheDocument();
    expect(screen.getByText(/mock dry run started/i)).not.toBeVisible();
    expect(screen.getByText(/bash scripts\/retrain\.sh --dry-run/i)).not.toBeVisible();
    await user.click(screen.getAllByText(/détails support\/admin/i)[0]);
    expect(screen.getByText(/mock dry run started/i)).toBeVisible();
    expect(
      screen.getByRole("button", { name: /entraînement en cours/i }),
    ).toBeDisabled();
  });

  it("approves an element, reloads the queue, and reports refresh-failure success separately", async () => {
    const user = userEvent.setup();
    apiMock.getAdminAnnotationQueue
      .mockResolvedValueOnce(queueWithStatuses("pending", "rejected"))
      .mockResolvedValueOnce(queueWithStatuses("approved", "pending"))
      .mockRejectedValueOnce(new Error("offline"));
    apiMock.setAdminAnnotationReviewStatus.mockResolvedValue({
      status: "ok",
      local_only: true,
      warning: "local only",
      element: queueWithStatuses("approved", "rejected").analyses[0].elements[0],
    });

    render(<AdminAnnotationsPage />);
    await screen.findByRole("button", { name: /valider/i });

    await user.click(screen.getByRole("button", { name: /valider/i }));

    await waitFor(() => {
      expect(apiMock.setAdminAnnotationReviewStatus).toHaveBeenCalledWith(
        "analysis-1",
        0,
        "approved",
      );
    });
    expect(await screen.findByText(/élément 0 marqué comme Validé/i)).toBeInTheDocument();
    expect(await screen.findByLabelText(/élément sélectionné 1 calli : statut À vérifier/i)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /rejeter/i }));
    expect(
      await screen.findByText(/mais la file n'a pas pu être actualisée/i),
    ).toBeInTheDocument();
  });

  it("pauses automatic refresh while correction is open and saves through modify endpoint", async () => {
    let autoRefresh: (() => void) | undefined;
    const intervalId = 1 as unknown as ReturnType<typeof window.setInterval>;
    const setIntervalSpy = vi
      .spyOn(window, "setInterval")
      .mockImplementation((handler: TimerHandler) => {
        autoRefresh = handler as () => void;
        return intervalId;
      });
    const clearIntervalSpy = vi
      .spyOn(window, "clearInterval")
      .mockImplementation(() => undefined);
    const user = userEvent.setup();
    const initial = queueWithStatuses("pending", "rejected");
    const updated = queueWithStatuses("pending", "rejected");
    updated.analyses[0].elements[0] = {
      ...updated.analyses[0].elements[0],
      class_name: "new-atl",
      bbox: [1, 2, 5, 6],
      trainable: false,
    };
    apiMock.getAdminAnnotationQueue
      .mockResolvedValueOnce(initial)
      .mockResolvedValueOnce(updated)
      .mockResolvedValue(updated);
    apiMock.modifyAdminAnnotationElement.mockResolvedValueOnce({
      status: "ok",
      local_only: true,
      warning: "local only",
      element: updated.analyses[0].elements[0],
    });

    try {
      render(<AdminAnnotationsPage />);
      const naturalWidthSpy = vi
        .spyOn(HTMLImageElement.prototype, "naturalWidth", "get")
        .mockReturnValue(100);
      const naturalHeightSpy = vi
        .spyOn(HTMLImageElement.prototype, "naturalHeight", "get")
        .mockReturnValue(100);
      const rectSpy = vi
        .spyOn(SVGElement.prototype, "getBoundingClientRect")
        .mockReturnValue({
          x: 0,
          y: 0,
          left: 0,
          top: 0,
          right: 100,
          bottom: 100,
          width: 100,
          height: 100,
          toJSON: () => ({}),
        } as DOMRect);
      Object.defineProperty(SVGElement.prototype, "setPointerCapture", {
        configurable: true,
        value: vi.fn(),
      });
      Object.defineProperty(SVGElement.prototype, "releasePointerCapture", {
        configurable: true,
        value: vi.fn(),
      });
      Object.defineProperty(SVGElement.prototype, "hasPointerCapture", {
        configurable: true,
        value: vi.fn(() => true),
      });

      await user.click(await screen.findByRole("button", { name: /corriger/i }));
      expect(screen.getByLabelText(/nom de l'élément/i)).toBeInTheDocument();
      const refreshButton = screen.getByRole("button", { name: /actualiser/i });
      expect(refreshButton).toBeDisabled();
      await act(async () => {
        autoRefresh?.();
      });
      expect(apiMock.getAdminAnnotationQueue).toHaveBeenCalledTimes(1);

      await act(async () => {
        fireEvent.load(screen.getByRole("img", { name: /image à corriger analysis-1/i }));
      });
      const visualEditor = await screen.findByLabelText(
        /redessiner la segmentation de l'élément 0/i,
      );
      const dispatchPointer = (type: string, clientX: number, clientY: number) => {
        const event = new Event(type, { bubbles: true, cancelable: true });
        Object.defineProperties(event, {
          clientX: { value: clientX },
          clientY: { value: clientY },
          pointerId: { value: 1 },
        });
        fireEvent(visualEditor, event);
      };
      dispatchPointer("pointerdown", 10, 20);
      dispatchPointer("pointermove", 60, 80);
      dispatchPointer("pointerup", 60, 80);
      expect(screen.getByLabelText(/zone x pour l'élément 0/i)).toHaveValue(10);
      expect(screen.getByLabelText(/zone y pour l'élément 0/i)).toHaveValue(20);
      expect(screen.getByLabelText(/zone w pour l'élément 0/i)).toHaveValue(50);
      expect(screen.getByLabelText(/zone h pour l'élément 0/i)).toHaveValue(60);

      await user.clear(screen.getByLabelText(/nom de l'élément/i));
      await user.type(screen.getByLabelText(/nom de l'élément/i), "new-atl");
      await user.click(screen.getByRole("button", { name: /^enregistrer$/i }));

      await waitFor(() => {
        expect(apiMock.modifyAdminAnnotationElement).toHaveBeenCalledWith(
          "analysis-1",
          0,
          {
            class_name: "new-atl",
            bbox: [10, 20, 50, 60],
            approve_after_save: undefined,
          },
        );
      });
      naturalWidthSpy.mockRestore();
      naturalHeightSpy.mockRestore();
      rectSpy.mockRestore();
      expect(
        await screen.findByRole("heading", { name: /élément #0 · new-atl/i }),
      ).toBeInTheDocument();
    } finally {
      setIntervalSpy.mockRestore();
      clearIntervalSpy.mockRestore();
    }
  });

  it("preserves visible status and shows an error when a triage mutation fails", async () => {
    const user = userEvent.setup();
    apiMock.getAdminAnnotationQueue.mockResolvedValueOnce(
      queueWithStatuses("pending", "pending"),
    );
    apiMock.setAdminAnnotationReviewStatus.mockRejectedValueOnce(
      new Error("offline"),
    );

    render(<AdminAnnotationsPage />);
    await screen.findByRole("button", { name: /rejeter/i });

    await user.click(screen.getByRole("button", { name: /rejeter/i }));

    expect(
      await screen.findByText(/impossible de marquer l'élément 0 comme Rejeté/i),
    ).toBeInTheDocument();
    expect(screen.getAllByLabelText(/statut À vérifier/i).length).toBeGreaterThan(1);
    expect(apiMock.getAdminAnnotationQueue).toHaveBeenCalledTimes(1);
  });
});
