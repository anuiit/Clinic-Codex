import { render, fireEvent, act, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { AnalysisRecord } from "../types";

vi.mock("../services/storage", () => ({
  getAnalysisById: vi.fn(),
  updateElements: vi.fn(() => true),
}));

vi.mock("../services/api", () => ({
  getClasses: vi.fn(() =>
    Promise.resolve({ class_names: ["aleph", "alpha", "beta", "lamed"] }),
  ),
  saveAnnotation: vi.fn(),
}));

import { getClasses, saveAnnotation } from "../services/api";
import { getAnalysisById, updateElements } from "../services/storage";
import AnnotationPage from "./AnnotationPage";

const BASE_RECORD: AnalysisRecord = {
  id: "test-id",
  imageDataUrl: "data:image/png;base64,abc",
  imageName: "test.png",
  timestamp: 1704067200000,
  result: {
    num_elements: 1,
    image_size: [800, 600],
    elements: [
      {
        bbox: [100, 100, 50, 40],
        class_name: "atl",
        class_label: 1,
        confidence: 0.9,
        rejected: false,
        top_k: [{ class_name: "aleph", confidence: 0.8 }],
      },
    ],
  },
  annotations: {},
};

const CONTAINER_RECT = {
  left: 0,
  top: 0,
  width: 800,
  height: 600,
  right: 800,
  bottom: 600,
  x: 0,
  y: 0,
  toJSON: () => {},
} as DOMRect;

function renderPage(record: AnalysisRecord = BASE_RECORD) {
  vi.mocked(getAnalysisById).mockReturnValue(record);
  return render(
    <MemoryRouter initialEntries={["/annotation/test-id"]}>
      <Routes>
        <Route path="/annotation/:id" element={<AnnotationPage />} />
        <Route path="/" element={<div>home</div>} />
      </Routes>
    </MemoryRouter>,
  );
}

function dispatchPointer(
  target: Element,
  type: "pointerdown" | "pointermove" | "pointerup",
  init: {
    clientX: number;
    clientY: number;
    pointerId?: number;
    buttons?: number;
  },
) {
  const event = new MouseEvent(type, {
    bubbles: true,
    cancelable: true,
    clientX: init.clientX,
    clientY: init.clientY,
  });
  Object.defineProperty(event, "pointerId", { value: init.pointerId ?? 1 });
  Object.defineProperty(event, "buttons", { value: init.buttons ?? 0 });
  fireEvent(target, event);
}

beforeEach(() => {
  vi.mocked(getClasses).mockResolvedValue({
    num_classes: 4,
    class_names: ["aleph", "alpha", "beta", "lamed"],
  });
  vi.mocked(saveAnnotation).mockResolvedValue({
    ok: true,
    status: "ok",
    analysis_id: "test-id",
    saved_count: 1,
    classes: ["aleph"],
  });
  vi.mocked(updateElements).mockClear();
  vi.mocked(saveAnnotation).mockClear();
  Element.prototype.getBoundingClientRect = vi.fn(() => CONTAINER_RECT);
  HTMLCanvasElement.prototype.getContext = vi.fn(() => ({
    clearRect: vi.fn(),
    drawImage: vi.fn(),
    imageSmoothingEnabled: false,
  })) as unknown as typeof HTMLCanvasElement.prototype.getContext;
});

describe("AnnotationPage element naming UX", () => {
  it("renames from fuzzy suggestions while preserving the bbox", async () => {
    const user = userEvent.setup();
    const { container } = renderPage();

    await user.click(await screen.findByText("atl"));
    const input = await screen.findByLabelText("Nommer l’élément 0");

    await user.clear(input);
    await user.type(input, "al");
    await user.click(await screen.findByText("aleph"));

    const saveButton = Array.from(container.querySelectorAll("button")).find(
      (button) => button.textContent?.includes("Enregistrer les modifications"),
    ) as HTMLElement;
    await user.click(saveButton);

    expect(updateElements).toHaveBeenCalledWith(
      "test-id",
      [
        expect.objectContaining({
          class_name: "aleph",
          bbox: [100, 100, 50, 40],
        }),
      ],
      { 0: "draft" },
    );
  });

  it("creates a missing element name from typed text", async () => {
    const user = userEvent.setup();
    const { container } = renderPage({
      ...BASE_RECORD,
      result: {
        ...BASE_RECORD.result,
        elements: [{ ...BASE_RECORD.result.elements[0], class_name: "" }],
      },
    });

    await user.click(await screen.findByText("À nommer"));
    const input = await screen.findByLabelText("Nommer l’élément 0");
    await user.type(input, "nouveau glyphe");
    await user.click(await screen.findByText("Créer « nouveau glyphe »"));

    const saveButton = Array.from(container.querySelectorAll("button")).find(
      (button) => button.textContent?.includes("Enregistrer les modifications"),
    ) as HTMLElement;
    await user.click(saveButton);

    expect(updateElements).toHaveBeenCalledWith(
      "test-id",
      [
        expect.objectContaining({
          class_name: "nouveau glyphe",
          bbox: [100, 100, 50, 40],
        }),
      ],
      { 0: "draft" },
    );
  });

  it("commits a typed missing name with Enter", async () => {
    const user = userEvent.setup();
    const { container } = renderPage();

    await user.click(await screen.findByText("atl"));
    const input = await screen.findByLabelText("Nommer l’élément 0");
    await user.clear(input);
    await user.type(input, "signe rare{Enter}");

    const saveButton = Array.from(container.querySelectorAll("button")).find(
      (button) => button.textContent?.includes("Enregistrer les modifications"),
    ) as HTMLElement;
    await user.click(saveButton);

    expect(updateElements).toHaveBeenCalledWith(
      "test-id",
      [
        expect.objectContaining({
          class_name: "signe rare",
          bbox: [100, 100, 50, 40],
        }),
      ],
      { 0: "draft" },
    );
  });

  it("bulk-submits named elements while leaving unnamed elements as drafts", async () => {
    const user = userEvent.setup();
    const { container } = renderPage({
      ...BASE_RECORD,
      result: {
        ...BASE_RECORD.result,
        num_elements: 3,
        elements: [
          BASE_RECORD.result.elements[0],
          {
            bbox: [200, 200, 30, 20],
            class_name: "",
            class_label: 0,
            confidence: 1,
            rejected: false,
            top_k: [],
          },
          {
            bbox: [240, 220, 35, 25],
            class_name: "beta",
            class_label: 2,
            confidence: 0.7,
            rejected: false,
            top_k: [],
          },
        ],
      },
      annotationStatus: { 0: "draft", 1: "draft", 2: "draft" },
    });

    await user.click(await screen.findByText("Soumettre les éléments nommés"));

    const saveButton = Array.from(container.querySelectorAll("button")).find(
      (button) => button.textContent?.includes("Enregistrer les modifications"),
    ) as HTMLElement;
    await user.click(saveButton);

    expect(updateElements).toHaveBeenCalledWith(
      "test-id",
      [
        expect.objectContaining({ class_name: "atl" }),
        expect.objectContaining({ class_name: "" }),
        expect.objectContaining({ class_name: "beta" }),
      ],
      { 0: "validated", 1: "draft", 2: "validated" },
    );
  });

  it("blocks review send while a submitted element is unnamed", async () => {
    const user = userEvent.setup();
    renderPage({
      ...BASE_RECORD,
      result: {
        ...BASE_RECORD.result,
        elements: [
          { ...BASE_RECORD.result.elements[0], class_name: "unknown" },
        ],
      },
    });

    await user.click(await screen.findByText("Envoyer les soumis"));

    expect(
      await screen.findByText(/Nommez les éléments soumis/),
    ).toBeInTheDocument();
    expect(saveAnnotation).not.toHaveBeenCalled();
  });

  it("blocks review send while a submitted element name is empty", async () => {
    const user = userEvent.setup();
    renderPage({
      ...BASE_RECORD,
      result: {
        ...BASE_RECORD.result,
        elements: [{ ...BASE_RECORD.result.elements[0], class_name: "" }],
      },
    });

    await user.click(await screen.findByText("Envoyer les soumis"));

    expect(
      await screen.findByText(/Nommez les éléments soumis/),
    ).toBeInTheDocument();
    expect(saveAnnotation).not.toHaveBeenCalled();
  });

  it("requires at least one submitted named element before review send", async () => {
    const user = userEvent.setup();
    renderPage();

    await user.click(await screen.findByText("Envoyer les soumis"));

    expect(
      await screen.findByText(/Soumettez au moins un élément nommé/),
    ).toBeInTheDocument();
    expect(saveAnnotation).not.toHaveBeenCalled();
  });

  it("sends only submitted named elements for review", async () => {
    const user = userEvent.setup();
    renderPage({
      ...BASE_RECORD,
      result: {
        ...BASE_RECORD.result,
        num_elements: 2,
        elements: [
          BASE_RECORD.result.elements[0],
          {
            bbox: [200, 200, 30, 20],
            class_name: "beta",
            class_label: 2,
            confidence: 0.7,
            rejected: false,
            top_k: [],
          },
        ],
      },
      annotationStatus: { 0: "validated", 1: "draft" },
    });

    await user.click(await screen.findByText("Envoyer les soumis"));

    expect(saveAnnotation).toHaveBeenCalledWith(
      expect.objectContaining({
        analysis_id: "test-id",
        image_name: "test.png",
        image_data_url: "data:image/png;base64,abc",
        timestamp: 1704067200000,
        annotations: [
          { index: 0, bbox: [100, 100, 50, 40], class_name: "atl" },
        ],
      }),
    );
  });

  it("does not block review send because a draft element is unnamed", async () => {
    const user = userEvent.setup();
    renderPage({
      ...BASE_RECORD,
      result: {
        ...BASE_RECORD.result,
        num_elements: 2,
        elements: [
          BASE_RECORD.result.elements[0],
          {
            bbox: [200, 200, 30, 20],
            class_name: "",
            class_label: 0,
            confidence: 1,
            rejected: false,
            top_k: [],
          },
        ],
      },
      annotationStatus: { 0: "validated", 1: "draft" },
    });

    await user.click(await screen.findByText("Envoyer les soumis"));

    expect(saveAnnotation).toHaveBeenCalledWith(
      expect.objectContaining({
        annotations: [
          { index: 0, bbox: [100, 100, 50, 40], class_name: "atl" },
        ],
      }),
    );
  });

  it("focuses the naming input after drawing a new bbox", async () => {
    const { container } = renderPage({
      ...BASE_RECORD,
      result: { ...BASE_RECORD.result, num_elements: 0, elements: [] },
    });
    await screen.findByText("Mode sélection");

    const drawButton = Array.from(container.querySelectorAll("button")).find(
      (button) => button.textContent?.includes("Mode sélection"),
    ) as HTMLElement;
    await act(async () => {
      fireEvent.click(drawButton);
    });

    const svg = container.querySelector("svg.absolute") as SVGSVGElement;
    svg.getBoundingClientRect = vi.fn(() => CONTAINER_RECT);
    svg.setPointerCapture = vi.fn();
    svg.releasePointerCapture = vi.fn();
    svg.hasPointerCapture = vi.fn(() => false);

    await act(async () => {
      dispatchPointer(svg, "pointerdown", {
        clientX: 100,
        clientY: 100,
        pointerId: 1,
        buttons: 1,
      });
    });
    await act(async () => {
      dispatchPointer(svg, "pointermove", {
        clientX: 155,
        clientY: 165,
        pointerId: 1,
        buttons: 1,
      });
    });
    await act(async () => {
      dispatchPointer(svg, "pointerup", {
        clientX: 155,
        clientY: 165,
        pointerId: 1,
      });
    });

    const input = await screen.findByLabelText("Nommer l’élément 0");
    expect(input).toHaveFocus();
  });
  it("uses the compact list for selection while editing in the main inspector", async () => {
    const user = userEvent.setup();
    renderPage({
      ...BASE_RECORD,
      result: {
        ...BASE_RECORD.result,
        num_elements: 2,
        elements: [
          BASE_RECORD.result.elements[0],
          {
            bbox: [220, 160, 80, 70],
            class_name: "beta",
            class_label: 2,
            confidence: 0.72,
            rejected: false,
            top_k: [],
          },
        ],
      },
    });

    await user.click(await screen.findByRole("button", { name: /#1 beta/i }));

    const inspector = screen.getByTestId("selected-element-inspector");
    expect(inspector).toHaveTextContent("#1 · beta");
    const input = await screen.findByLabelText("Nommer l’élément 1");
    expect(inspector).toContainElement(input);
  });

  it("keeps compact chrome, analyzer toolbar, and list controls in stable regions", async () => {
    const user = userEvent.setup();
    const { container } = renderPage({
      ...BASE_RECORD,
      result: {
        ...BASE_RECORD.result,
        num_elements: 2,
        elements: [
          BASE_RECORD.result.elements[0],
          {
            bbox: [220, 160, 80, 70],
            class_name: "beta",
            class_label: 2,
            confidence: 0.72,
            rejected: false,
            top_k: [],
          },
        ],
      },
    });

    await user.click(await screen.findByRole("button", { name: /#1 beta/i }));

    const chrome = screen.getByTestId("annotation-page-chrome");
    const topbar = container.querySelector(".annotation-topbar") as HTMLElement;
    expect(chrome).toContainElement(topbar);
    expect(
      within(chrome).getByRole("button", { name: "Retour" }),
    ).toBeInTheDocument();
    expect(
      within(chrome).getByRole("button", {
        name: "Soumettre les éléments nommés",
      }),
    ).toBeInTheDocument();
    expect(
      within(chrome).getByRole("button", {
        name: "Enregistrer les modifications",
      }),
    ).toBeInTheDocument();
    expect(
      within(chrome).getByRole("button", { name: "Envoyer les soumis" }),
    ).toBeInTheDocument();
    expect(screen.getByTestId("annotation-admin-notice")).toHaveTextContent(
      "Gardez",
    );
    expect(
      within(chrome).queryByRole("searchbox", { name: /filtrer/i }),
    ).not.toBeInTheDocument();

    const compactList = screen.getByLabelText("Liste compacte des éléments");
    const toolbar = container.querySelector(".main-image-panel__toolbar");
    expect(toolbar).toBeInTheDocument();
    expect(within(toolbar as HTMLElement).getByRole("button", { name: "Mode sélection" })).toBeInTheDocument();
    expect(within(toolbar as HTMLElement).getByRole("button", { name: "Annuler bbox" })).toBeInTheDocument();
    expect(within(toolbar as HTMLElement).getByRole("button", { name: "N°" })).toBeInTheDocument();
    expect(toolbar).toContainElement(screen.getByTestId("annotation-analyzer-toolbar"));

    const controls = screen.getByTestId("annotation-stage-controls");
    expect(within(controls).getByRole("button", { name: "Zoom avant" })).toBeInTheDocument();
    expect(within(controls).getByRole("button", { name: "Réinitialiser la vue" })).toBeInTheDocument();
    expect(within(controls).getByRole("button", { name: "Zoom arrière" })).toBeInTheDocument();
    expect(within(controls).getByText("100%")).toBeInTheDocument();

    expect(
      within(compactList).getByRole("searchbox", { name: /filtrer/i }),
    ).toBeInTheDocument();
    const controls = screen.getByTestId("annotation-list-controls");
    expect(controls).toHaveClass("xl:flex-nowrap");
    expect(controls).toContainElement(
      within(compactList).getByLabelText(/statut/i),
    );
    expect(controls).toContainElement(
      within(compactList).getByLabelText(/tri/i),
    );

    const inspector = screen.getByTestId("selected-element-inspector");
    expect(inspector).toHaveClass("annotation-selected-inspector");
    expect(
      inspector.querySelector(".annotation-selected-overview"),
    ).toHaveClass("grid-cols-[150px_minmax(0,1fr)]");
    expect(inspector).toContainElement(
      screen.getByLabelText("Nommer l’élément 1"),
    );
  });

  it("highlights the linked bbox when hovering a compact list item", async () => {
    renderPage({
      ...BASE_RECORD,
      result: {
        ...BASE_RECORD.result,
        num_elements: 2,
        elements: [
          BASE_RECORD.result.elements[0],
          {
            bbox: [220, 160, 80, 70],
            class_name: "beta",
            class_label: 2,
            confidence: 0.72,
            rejected: false,
            top_k: [],
          },
        ],
      },
    });

    const betaRow = await screen.findByRole("button", { name: /#1 beta/i });
    const betaBox = await screen.findByTestId("annotation-box-1");

    expect(betaBox).toHaveAttribute("stroke", "#a8a29e");
    fireEvent.mouseEnter(betaRow);
    expect(betaBox).toHaveAttribute("stroke", "#38bdf8");
    fireEvent.mouseLeave(betaRow);
    expect(betaBox).toHaveAttribute("stroke", "#a8a29e");
  });

  it("moves the submitted summary out of the top bar and into the list badge", async () => {
    const { container } = renderPage({
      ...BASE_RECORD,
      result: {
        ...BASE_RECORD.result,
        num_elements: 2,
        elements: [
          BASE_RECORD.result.elements[0],
          {
            bbox: [220, 160, 80, 70],
            class_name: "beta",
            class_label: 2,
            confidence: 0.72,
            rejected: false,
            top_k: [],
          },
        ],
      },
      annotationStatus: { 0: "validated", 1: "draft" },
    });

    await screen.findByText("atl");

    expect(container.querySelector(".annotation-topbar")).not.toHaveTextContent(
      /Soumis\s*:/i,
    );
    const compactList = screen.getByLabelText("Liste compacte des éléments");
    expect(compactList).toHaveTextContent(
      /(?:Soumis\s*)?1\s*\/\s*2(?:\s*Soumis)?/i,
    );
  });

  it("keeps the compact list header badge visible without the old elements count block", async () => {
    const elements = Array.from({ length: 38 }, (_, idx) => ({
      bbox: [100 + idx, 100, 50, 40] as [number, number, number, number],
      class_name: `glyphe-${idx}`,
      class_label: idx,
      confidence: 0.9,
      rejected: false,
      top_k: [],
    }));

    renderPage({
      ...BASE_RECORD,
      result: {
        ...BASE_RECORD.result,
        num_elements: elements.length,
        elements,
      },
      annotationStatus: Object.fromEntries(
        elements.map((_, idx) => [idx, "validated"]),
      ) as AnalysisRecord["annotationStatus"],
    });

    await screen.findByText("glyphe-0");

    const compactList = screen.getByLabelText("Liste compacte des éléments");
    expect(screen.getByLabelText("Soumis 38/38")).toBeInTheDocument();
    expect(compactList).not.toHaveTextContent(/Éléments\s*38\s*\/\s*38/i);
  });

  it('keeps selected and empty inspector shells on the reduced shared size contract', async () => {
    const user = userEvent.setup();
    renderPage();

    const inspector = await screen.findByTestId('selected-element-inspector');
    expect(inspector).toHaveClass('annotation-selected-inspector');
    expect(inspector).not.toHaveClass('h-[360px]');
    expect(inspector).not.toHaveClass('annotation-panel');

    await user.click(await screen.findByText("atl"));

    expect(inspector).toHaveClass('annotation-selected-inspector');
    expect(inspector).not.toHaveClass('h-[360px]');
    expect(inspector).not.toHaveClass('annotation-panel');
  });

  it('keeps filter, status, and Tri controls in one readable compact row contract', async () => {
    renderPage();

    const controls = await screen.findByTestId('annotation-list-controls');
    expect(controls).toHaveClass('annotation-list-controls');
    expect(controls).toContainElement(screen.getByRole('searchbox', { name: /filtrer/i }));
    expect(controls).toContainElement(screen.getByLabelText(/statut/i));
    expect(controls).toContainElement(screen.getByLabelText(/tri/i));

    for (const labelText of ['Filtrer', 'Statut', 'Tri']) {
      const label = within(controls).getByText(labelText).closest('label');
      expect(label).toHaveClass('text-xs');
      expect(label).not.toHaveClass('text-[10px]');
    }
  });

  it('groups rename, submit, and delete controls in one inspector action row', async () => {
    const user = userEvent.setup();
    renderPage();

    await user.click(await screen.findByText("atl"));

    const actionRow = screen.getByTestId("annotation-inspector-action-row");
    expect(actionRow).toContainElement(
      screen.getByLabelText("Nommer l’élément 0"),
    );
    expect(
      within(actionRow).getByRole("button", { name: "Soumettre" }),
    ).toBeInTheDocument();
    expect(
      within(actionRow).getByRole("button", { name: "Supprimer l’élément #0" }),
    ).toBeInTheDocument();
  });

  it("filters and sorts the compact annotation list without changing bbox data", async () => {
    const user = userEvent.setup();
    const { container } = renderPage({
      ...BASE_RECORD,
      result: {
        ...BASE_RECORD.result,
        num_elements: 3,
        elements: [
          {
            ...BASE_RECORD.result.elements[0],
            class_name: "zeta",
            confidence: 0.9,
          },
          {
            bbox: [220, 160, 80, 70],
            class_name: "beta",
            class_label: 2,
            confidence: 0.15,
            rejected: false,
            top_k: [],
          },
          {
            bbox: [320, 260, 90, 80],
            class_name: "alpha",
            class_label: 3,
            confidence: 0.6,
            rejected: true,
            top_k: [],
          },
        ],
      },
      annotationStatus: { 0: "validated", 1: "draft", 2: "draft" },
    });

    await user.type(
      await screen.findByRole("searchbox", { name: /filtrer/i }),
      "beta",
    );
    const compactList = screen.getByLabelText("Liste compacte des éléments");
    expect(
      within(compactList).getByRole("button", { name: /#1 beta/i }),
    ).toBeInTheDocument();
    expect(
      within(compactList).queryByRole("button", { name: /#0 zeta/i }),
    ).not.toBeInTheDocument();

    await user.clear(screen.getByRole("searchbox", { name: /filtrer/i }));
    await user.selectOptions(screen.getByLabelText(/tri/i), "confidence-asc");

    const rows = within(compactList).getAllByRole("button", { name: /#\d/i });
    expect(rows[0]).toHaveAccessibleName(/#1 beta/i);

    const saveButton = Array.from(container.querySelectorAll("button")).find(
      (button) => button.textContent?.includes("Enregistrer les modifications"),
    ) as HTMLElement;
    await user.click(saveButton);

    expect(updateElements).toHaveBeenCalledWith(
      "test-id",
      [
        expect.objectContaining({
          class_name: "zeta",
          bbox: [100, 100, 50, 40],
        }),
        expect.objectContaining({
          class_name: "beta",
          bbox: [220, 160, 80, 70],
        }),
        expect.objectContaining({
          class_name: "alpha",
          bbox: [320, 260, 90, 80],
        }),
      ],
      { 0: "validated", 1: "draft", 2: "draft" },
    );
  });

  it("toggles annotation canvas bbox labels from numbers to names", async () => {
    const user = userEvent.setup();
    const { container } = renderPage({
      ...BASE_RECORD,
      result: {
        ...BASE_RECORD.result,
        num_elements: 2,
        elements: [
          BASE_RECORD.result.elements[0],
          {
            bbox: [220, 160, 80, 70],
            class_name: "beta",
            class_label: 2,
            confidence: 0.72,
            rejected: false,
            top_k: [],
          },
        ],
      },
    });

    await screen.findByTestId("annotation-box-1");
    const overlay = container.querySelector("svg.absolute") as SVGSVGElement;
    expect(overlay).toHaveTextContent("#1");
    expect(overlay).not.toHaveTextContent("#1 · beta");

    await user.click(
      screen.getByRole("button", {
        name: /(?:afficher|masquer).*(?:noms|libellés)/i,
      }),
    );

    expect(overlay).toHaveTextContent("atl");
    expect(overlay).toHaveTextContent("beta");
    expect(overlay).not.toHaveTextContent("#0 · atl");
    expect(overlay).not.toHaveTextContent("#1 · beta");
  });
});
