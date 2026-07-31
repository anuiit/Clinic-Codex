import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { AnalysisRecord, SegmentResult, TrustResult } from '../types';
import WorkspacePage from './WorkspacePage';

const RECORDS: AnalysisRecord[] = [
  {
    id: 'alpha-run',
    imageName: 'alpha.png',
    imageDataUrl: 'data:image/png;base64,alpha',
    timestamp: 1704067200000,
    result: {
      num_elements: 2,
      image_size: [800, 600],
      elements: [
        {
          bbox: [100, 120, 50, 40],
          class_name: 'aleph',
          class_label: 1,
          confidence: 0.92,
          rejected: false,
          top_k: [
            { class_name: 'aleph', confidence: 0.92 },
            { class_name: 'ayin', confidence: 0.13 },
          ],
        },
        {
          bbox: [260, 200, 35, 60],
          class_name: 'lamed',
          class_label: 2,
          confidence: 0.31,
          rejected: true,
          top_k: [
            { class_name: 'lamed', confidence: 0.31 },
            { class_name: 'nun', confidence: 0.28 },
          ],
        },
      ],
    },
    annotations: { 0: 'annotated aleph' },
  },
  {
    id: 'beta-run',
    imageName: 'beta.png',
    imageDataUrl: 'data:image/png;base64,beta',
    timestamp: 1704153600000,
    result: {
      num_elements: 1,
      image_size: [640, 480],
      elements: [
        {
          bbox: [30, 40, 24, 24],
          class_name: 'bet',
          class_label: 3,
          confidence: 0.81,
          rejected: false,
          top_k: [{ class_name: 'bet', confidence: 0.81 }],
        },
      ],
    },
    annotations: {},
  },
];

const SEGMENT_RESULT: SegmentResult = {
  num_elements: 1,
  image_size: [320, 240],
  elements: [
    {
      bbox: [10, 20, 30, 40],
      class_name: 'fresh',
      class_label: 4,
      confidence: 0.77,
      rejected: false,
      top_k: [{ class_name: 'fresh', confidence: 0.77 }],
    },
  ],
};

const TRUST_RESULT: TrustResult = {
  query: { bbox: [100, 120, 50, 40], predicted_class: 'aleph' },
  trust: {
    predicted_class_rank: 1,
    predicted_class_similarity: 0.91,
    top1_class: 'aleph',
    top1_similarity: 0.91,
    margin_to_second: 0.2,
    above_rejection_threshold: true,
    rejection_threshold: 0.35,
    ambiguous: false,
    entropy: 0.23,
    top_k: [
      { class_name: 'aleph', confidence: 0.91 },
      { class_name: 'ayin', confidence: 0.71 },
    ],
  },
};

let historyRecords: AnalysisRecord[] = [];
let drawImageMock = vi.fn();
let clearRectMock = vi.fn();

vi.mock('../services/storage', () => ({
  deleteAnalysis: vi.fn(async (id: string) => {
    historyRecords = historyRecords.filter((record) => record.id !== id);
  }),
  getHistory: vi.fn(async () => historyRecords),
  saveAnalysis: vi.fn(async (record: AnalysisRecord) => {
    historyRecords = [record, ...historyRecords];
  }),
}));

vi.mock('../services/api', () => ({
  getTrust: vi.fn(() => Promise.resolve(TRUST_RESULT)),
  segmentGlyph: vi.fn(() => Promise.resolve(SEGMENT_RESULT)),
}));

import { getTrust, segmentGlyph } from '../services/api';
import { deleteAnalysis, saveAnalysis } from '../services/storage';

function cloneRecords(records: AnalysisRecord[]) {
  return structuredClone(records) as AnalysisRecord[];
}

function AnnotationHandoffProbe() {
  const location = useLocation();
  return <div>annotation handoff {location.pathname}{location.search}</div>;
}

function renderPage(initialRecords = RECORDS, initialEntry = '/') {
  historyRecords = cloneRecords(initialRecords);
  return render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <Routes>
        <Route path="/" element={<WorkspacePage />} />
        <Route path="/annotate/:id" element={<AnnotationHandoffProbe />} />
      </Routes>
    </MemoryRouter>,
  );
}

function dispatchPointer(
  target: Element,
  type: 'pointerdown' | 'pointermove' | 'pointerup',
  init: { clientX: number; clientY: number; pointerId?: number; buttons?: number },
) {
  const event = new MouseEvent(type, {
    bubbles: true,
    cancelable: true,
    clientX: init.clientX,
    clientY: init.clientY,
  });
  Object.defineProperty(event, 'pointerId', { value: init.pointerId ?? 1 });
  Object.defineProperty(event, 'buttons', { value: init.buttons ?? 0 });
  fireEvent(target, event);
}

function stubSvgRect(svg: SVGSVGElement, width: number, height: number) {
  svg.getBoundingClientRect = vi.fn().mockReturnValue({
    left: 0,
    top: 0,
    width,
    height,
    right: width,
    bottom: height,
    x: 0,
    y: 0,
    toJSON: () => {},
  });
}


async function waitForWorkspaceHistory() {
  await screen.findAllByText('alpha.png');
}

function stubCanvas() {
  drawImageMock = vi.fn();
  clearRectMock = vi.fn();
  HTMLCanvasElement.prototype.getContext = vi.fn(() => ({
    clearRect: clearRectMock,
    drawImage: drawImageMock,
  })) as unknown as typeof HTMLCanvasElement.prototype.getContext;
}

function drawImageSourceRects() {
  return drawImageMock.mock.calls.map((call) => call.slice(1, 5));
}

describe('WorkspacePage interaction coverage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    stubCanvas();
    Object.defineProperty(HTMLImageElement.prototype, 'complete', {
      configurable: true,
      get: () => true,
    });
    Object.defineProperty(HTMLImageElement.prototype, 'naturalWidth', {
      configurable: true,
      get: () => 800,
    });
    vi.stubGlobal('crypto', { randomUUID: () => 'new-analysis-id' });
  });

  it('filters history results, selects a matching run, and deletes the active run', async () => {
    const user = userEvent.setup();
    renderPage();
    await waitForWorkspaceHistory();

    await user.click(screen.getByTitle('Déplier l’historique'));
    await user.type(screen.getByPlaceholderText('Filtrer par glyphe ou classe'), 'beta');
    expect(screen.getByText('beta.png')).toBeInTheDocument();

    await user.click(screen.getByText('beta.png'));
    expect(screen.getAllByText('beta.png').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('bet').length).toBeGreaterThanOrEqual(1);

    await user.click(screen.getByLabelText('Supprimer beta.png'));

    expect(deleteAnalysis).toHaveBeenCalledWith('beta-run');
    expect(screen.queryByText('beta.png')).not.toBeInTheDocument();
    expect(screen.getByText('Aucun résultat ne correspond au filtre.')).toBeInTheDocument();
  });

  it('reflects AnnotationPage-saved element edits and validation status instead of stale legacy annotations', async () => {
    renderPage([
      {
        ...RECORDS[0],
        result: {
          ...RECORDS[0].result,
          elements: [
            {
              ...RECORDS[0].result.elements[0],
              class_name: 'edited aleph',
              bbox: [110, 130, 55, 45],
            },
            RECORDS[0].result.elements[1],
          ],
        },
        annotations: { 0: 'stale legacy aleph' },
        annotationStatus: { 0: 'validated' },
      },
    ]);

    await screen.findByTestId('workspace-image-header-meta');

    expect(screen.getByTestId('workspace-image-header-meta')).toHaveTextContent(
      'Annotés / rejetés 1/2 · 1',
    );
    expect(screen.getByRole('button', { name: /edited aleph région 0/i })).toBeInTheDocument();
    expect(screen.queryByText(/stale legacy aleph/i)).not.toBeInTheDocument();
  });

  it('toggles overlays and opens focused region details without starting a pan', async () => {
    const user = userEvent.setup();
    const { container } = renderPage();

    await screen.findByTestId('workspace-overlay');
    expect(container.querySelector('svg.absolute')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'masqué' }));
    expect(container.querySelector('svg.absolute')).not.toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'tout' }));
    expect(container.querySelector('svg.absolute')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'tout' })).toHaveClass(
      'analyzer-toolbar__button',
      'rounded-none',
    );
    expect(screen.getByRole('button', { name: 'tout' }).parentElement).not.toHaveClass('border');

    await user.click(screen.getByRole('button', { name: /annotated aleph région 0/i }));

    expect(await screen.findByText('Retour aux régions')).toBeInTheDocument();
    expect(screen.getByTestId('workspace-focused-selected-card')).toHaveClass('workspace-inspector-section');
    expect(screen.getByText('Aperçu du segment')).toBeInTheDocument();
    await waitFor(() => expect(getTrust).toHaveBeenCalledWith('data:image/png;base64,alpha', [100, 120, 50, 40], 'aleph', 10, expect.objectContaining({ signal: expect.any(AbortSignal) })));

    await user.click(screen.getByRole('button', { name: /Retour aux régions/ }));
    expect(screen.getByText('Éléments détectés')).toBeInTheDocument();
  });


  it('clears the focused workspace region with the explicit deselect tool', async () => {
    const user = userEvent.setup();
    renderPage();

    await screen.findByTestId('workspace-overlay');
    expect(screen.getByRole('button', { name: 'Désélectionner' })).toBeDisabled();

    await user.click(await screen.findByRole('button', { name: /annotated aleph région 0/i }));
    expect(await screen.findByText('Retour aux régions')).toBeInTheDocument();
    const deselect = screen.getByRole('button', { name: 'Désélectionner' });
    expect(deselect).toBeEnabled();

    await user.click(deselect);

    expect(screen.queryByText('Retour aux régions')).not.toBeInTheDocument();
    expect(await screen.findByText('Éléments détectés')).toBeInTheDocument();
  });

  it('draws workspace list and focused-detail crop canvases from attached refs', async () => {
    const user = userEvent.setup();
    renderPage();

    expect(await screen.findByTestId('workspace-list-crop-canvas-0')).toBeInTheDocument();
    await waitFor(() =>
      expect(drawImageSourceRects()).toEqual(
        expect.arrayContaining([
          [100, 120, 50, 40],
          [260, 200, 35, 60],
        ]),
      ),
    );

    drawImageMock.mockClear();
    await user.click(screen.getByRole('button', { name: /lamed région 1/i }));

    const detailCanvas = await screen.findByTestId('workspace-detail-crop-canvas') as HTMLCanvasElement;
    await waitFor(() =>
      expect(drawImageMock).toHaveBeenCalledWith(
        expect.any(HTMLImageElement),
        260,
        200,
        35,
        60,
        0,
        0,
        detailCanvas.width,
        detailCanvas.height,
      ),
    );

    drawImageMock.mockClear();
    await user.click(screen.getByRole('button', { name: /Retour aux régions/ }));

    expect(await screen.findByTestId('workspace-list-crop-canvas-0')).toBeInTheDocument();
    await waitFor(() =>
      expect(drawImageSourceRects()).toEqual(
        expect.arrayContaining([
          [100, 120, 50, 40],
          [260, 200, 35, 60],
        ]),
      ),
    );
  });

  it('keeps tiny and invalid workspace crop previews visible without drawing empty sources', async () => {
    renderPage([
      {
        ...RECORDS[0],
        result: {
          ...RECORDS[0].result,
          num_elements: 2,
          elements: [
            {
              ...RECORDS[0].result.elements[0],
              bbox: [100, 120, 1, 80],
              class_name: 'thin',
            },
            {
              ...RECORDS[0].result.elements[1],
              bbox: [260, 200, 0, 60],
              class_name: 'empty-source',
              rejected: false,
            },
          ],
        },
      },
    ]);

    const tinyCanvas = await screen.findByTestId('workspace-list-crop-canvas-0') as HTMLCanvasElement;
    const invalidCanvas = await screen.findByTestId('workspace-list-crop-canvas-1') as HTMLCanvasElement;

    expect(tinyCanvas.width).toBeGreaterThanOrEqual(18);
    expect(tinyCanvas.height).toBe(48);
    expect(invalidCanvas.width).toBe(48);
    expect(invalidCanvas.height).toBe(48);
    await waitFor(() =>
      expect(drawImageSourceRects()).toEqual(
        expect.arrayContaining([[100, 120, 1, 80]]),
      ),
    );
    expect(drawImageSourceRects()).not.toEqual(
      expect.arrayContaining([[260, 200, 0, 60]]),
    );
  });

  it('redraws crop previews for a newly selected record without stale previous-record draws', async () => {
    const user = userEvent.setup();
    renderPage();

    await screen.findByText('annotated aleph');
    await user.click(screen.getByTitle('Déplier l’historique'));
    const betaRun = await screen.findByText('beta.png');
    drawImageMock.mockClear();
    await user.click(betaRun);

    await waitFor(() =>
      expect(drawImageSourceRects()).toEqual(
        expect.arrayContaining([[30, 40, 24, 24]]),
      ),
    );
    expect(drawImageSourceRects().at(-1)).toEqual([30, 40, 24, 24]);
  });

  it('renders selected workspace identity before delayed trust recalculation resolves', async () => {
    const user = userEvent.setup();
    vi.mocked(getTrust).mockReturnValueOnce(new Promise<TrustResult>(() => undefined));
    renderPage();

    await user.click(await screen.findByRole('button', { name: /lamed région 1/i }));

    expect(await screen.findByText('Retour aux régions')).toBeInTheDocument();
    expect(screen.getAllByText('lamed').length).toBeGreaterThanOrEqual(1);
    expect(screen.getByTestId('workspace-trust-summary')).toHaveAttribute('aria-busy', 'true');
    expect(screen.getByTestId('workspace-trust-loading')).toBeInTheDocument();
    expect(screen.queryByText('31.0%')).not.toBeInTheDocument();
    expect(screen.getByText('Résumé de confiance')).toBeInTheDocument();
    expect(getTrust).toHaveBeenCalledWith('data:image/png;base64,alpha', [260, 200, 35, 60], 'lamed', 10, expect.objectContaining({ signal: expect.any(AbortSignal) }));
  });

  it('focuses a region from the full row click without rendering a separate details button', async () => {
    const user = userEvent.setup();
    renderPage();

    await screen.findByText('annotated aleph');
    expect(screen.queryByRole('button', { name: 'Voir plus de détails' })).not.toBeInTheDocument();

    await user.click(screen.getByText('annotated aleph'));

    expect(await screen.findByText('Retour aux régions')).toBeInTheDocument();
    expect(screen.queryByText('Région 0')).not.toBeInTheDocument();
    await waitFor(() => expect(getTrust).toHaveBeenCalledWith('data:image/png;base64,alpha', [100, 120, 50, 40], 'aleph', 10, expect.objectContaining({ signal: expect.any(AbortSignal) })));
  });

  it('hands off the selected workspace region to the annotation editor query param', async () => {
    const user = userEvent.setup();
    renderPage();

    await user.click(await screen.findByRole('button', { name: /annotated aleph région 0/i }));
    await user.click(await screen.findByRole('button', { name: 'Annoter la région' }));

    expect(await screen.findByText('annotation handoff /annotate/alpha-run?element=0')).toBeInTheDocument();
  });

  it('hands off the whole active workspace record without a selected element query param', async () => {
    const user = userEvent.setup();
    renderPage();

    await user.click(await screen.findByRole('button', { name: 'Annoter l’analyse' }));

    expect(await screen.findByText('annotation handoff /annotate/alpha-run')).toBeInTheDocument();
  });

  it('moves workspace analysis action and summary metadata into the image header', async () => {
    renderPage();

    const header = await screen.findByTestId('workspace-image-header');
    const annotateAction = within(header).getByRole('button', { name: 'Annoter l’analyse' });
    expect(annotateAction).toHaveClass('workspace-annotate-action');
    expect(annotateAction).not.toHaveClass('ui-action-primary');
    expect(screen.getByTestId('workspace-image-header-meta')).toHaveTextContent('800×600');
    expect(screen.getByTestId('workspace-image-header-meta')).toHaveTextContent('Annotés / rejetés 1/2 · 1');
    expect(screen.getByTestId('workspace-image-header-meta')).toHaveTextContent('Classes');

    const panel = screen.getByTestId('workspace-detected-panel');
    expect(within(panel).queryByRole('button', { name: 'Annoter l’analyse' })).not.toBeInTheDocument();
    expect(within(panel).queryByText('Dimensions')).not.toBeInTheDocument();
  });

  it('toggles workspace canvas bbox labels with an icon-only accessible control', async () => {
    const user = userEvent.setup();
    const { container } = renderPage();

    await screen.findByTestId('workspace-overlay');
    const toggle = screen.getByRole('button', { name: /(?:afficher|masquer).*(?:noms|libellés)/i });
    expect(toggle).toHaveAccessibleName(/(?:afficher|masquer).*(?:noms|libellés)/i);
    expect(toggle.textContent?.trim()).not.toMatch(/(?:Noms|N°)/i);

    const overlay = await screen.findByTestId('workspace-overlay');
    expect(overlay).toHaveTextContent('#1');
    expect(overlay).not.toHaveTextContent('#1 · lamed');

    const labelToggle = screen.getByRole('button', { name: /(?:afficher|masquer).*(?:noms|libellés)/i });
    expect(labelToggle).toHaveAttribute('title', 'Afficher les noms des libellés');
    expect(labelToggle).not.toHaveTextContent(/Noms|N°/i);

    await user.click(labelToggle);

    expect(screen.getByRole('button', { name: /masquer.*noms.*libellés/i })).toHaveAttribute('title', 'Masquer les noms des libellés');
    expect(screen.getByRole('button', { name: /masquer.*noms.*libellés/i })).not.toHaveTextContent(/Noms|N°/i);

    const namedOverlay = container.querySelector('[data-testid="workspace-overlay"]');
    expect(namedOverlay).toHaveTextContent('aleph');
    expect(namedOverlay).toHaveTextContent('lamed');
    expect(namedOverlay).not.toHaveTextContent('#0 · aleph');
    expect(namedOverlay).not.toHaveTextContent('#1 · lamed');
  });

  it('uses a flatter history sidebar while preserving search and delete affordances', async () => {
    const user = userEvent.setup();
    renderPage();
    await waitForWorkspaceHistory();

    await user.click(screen.getByTitle('Déplier l’historique'));
    const sidebar = screen.getByTestId('workspace-history-sidebar');
    const header = screen.getByTestId('workspace-history-header');

    expect(sidebar).not.toHaveClass('border', 'border-stone-800');
    expect(header).not.toHaveClass('border-b', 'border-stone-800');
    expect(screen.getByPlaceholderText('Filtrer par glyphe ou classe')).toHaveClass('ui-input');
    expect(screen.getByLabelText('Supprimer alpha.png')).toBeInTheDocument();

    const activeRow = within(screen.getByTestId('workspace-history-list'))
      .getByText('alpha.png')
      .closest('[role="button"]') as HTMLElement;
    expect(activeRow).toHaveClass('selection-card--active');
    expect(activeRow).not.toHaveClass('border', 'shadow-[0_0_0_1px_rgba(245,158,11,0.25)]');
  });

  it('mirrors hover state between the workspace overlay and detected list', async () => {
    const { container } = renderPage();

    const overlay = await screen.findByTestId('workspace-overlay') as unknown as SVGSVGElement;
    stubSvgRect(overlay, 800, 600);
    const firstRow = await screen.findByRole('button', { name: /annotated aleph région 0/i });
    const firstRegion = () => container.querySelector('[data-box-id="0"]') as SVGGElement;

    expect(firstRegion()).not.toHaveAttribute('data-hovered');

    await act(async () => {
      dispatchPointer(overlay, 'pointermove', { clientX: 125, clientY: 140, pointerId: 1, buttons: 0 });
    });

    expect(firstRow).toHaveClass('selection-card--active');
    expect(firstRegion()).toHaveAttribute('data-hovered', 'true');

    await act(async () => {
      fireEvent.pointerLeave(overlay);
    });
    expect(firstRegion()).not.toHaveAttribute('data-hovered');

    await act(async () => {
      fireEvent.mouseEnter(firstRow);
    });
    expect(firstRow).toHaveClass('selection-card--active');
    expect(firstRegion()).toHaveAttribute('data-hovered', 'true');

    await act(async () => {
      fireEvent.mouseLeave(firstRow);
    });
    expect(firstRegion()).not.toHaveAttribute('data-hovered');
  });

  it('keeps workspace list and detail panels in the narrowed image-dominant scroll layout without duplicate focus lists', async () => {
    const user = userEvent.setup();
    renderPage();

    await screen.findByText('annotated aleph');
    const contentGrid = screen.getByTestId('workspace-content-grid');
    expect(contentGrid).toHaveClass('xl:grid-cols-[minmax(0,1.45fr)_minmax(280px,0.55fr)]');
    expect(contentGrid).toHaveClass('2xl:grid-cols-[minmax(0,1.55fr)_minmax(320px,0.45fr)]');
    expect(contentGrid).not.toHaveClass('2xl:grid-cols-[minmax(0,1.08fr)_minmax(360px,0.92fr)]');

    const stage = screen.getByTestId('workspace-stage');
    expect(stage).toHaveClass('image-stage-frame', 'image-stage-scrollbar', 'image-stage-grid');

    const stageToolbar = within(stage).getByTestId('workspace-analyzer-toolbar');
    expect(stage.querySelector('.main-image-panel__controls')).not.toBeInTheDocument();
    expect(stage.querySelector('.main-image-panel__toolbar')).toHaveClass(
      'main-image-panel__toolbar--bottom-center',
    );
    expect(
      within(stageToolbar).getAllByRole('button').map((button) =>
        button.getAttribute('aria-label') ?? button.textContent?.trim(),
      ),
    ).toEqual([
      'tout',
      'focus',
      'masqué',
      'Afficher les noms des libellés',
      'Désélectionner',
      'Zoom arrière',
      'Ajuster à la vue',
      'Zoom avant',
    ]);

    const sidebar = screen.getByTestId('workspace-detected-panel');
    expect(sidebar).toBeInTheDocument();
    const proposalList = within(sidebar).getByTestId('workspace-detected-list');
    expect(proposalList).toHaveClass('grid-cols-1', 'overflow-y-auto', 'pr-0');
    expect(proposalList).not.toHaveClass('2xl:gap-x-4');

    await user.click(screen.getByRole('button', { name: /annotated aleph région 0/i }));

    expect(await screen.findByText('Retour aux régions')).toBeInTheDocument();
    expect(screen.queryByText('Éléments détectés')).not.toBeInTheDocument();
    expect(screen.getAllByText('Aperçu du segment')).toHaveLength(1);
    expect(document.querySelector('.sidebar-body')).toHaveClass('overflow-y-auto');
    expect(screen.queryByRole('button', { name: /annotated aleph région 0/i })).not.toBeInTheDocument();
  });

  it('keeps workspace zoom/pan contained and read-only while focusing rows', async () => {
    const user = userEvent.setup();
    const { container } = renderPage();

    const image = (await screen.findAllByAltText('alpha.png')).find((candidate) =>
      candidate.className.includes('object-fill'),
    ) as HTMLImageElement;
    const wrapper = image.parentElement as HTMLElement;
    const viewport = wrapper.parentElement as HTMLElement;
    viewport.setPointerCapture = vi.fn();
    viewport.releasePointerCapture = vi.fn();
    viewport.hasPointerCapture = vi.fn(() => true);

    await act(async () => {
      fireEvent.wheel(viewport, { deltaY: -100 });
    });
    await act(async () => {
      dispatchPointer(viewport, 'pointerdown', { clientX: 100, clientY: 100, pointerId: 1, buttons: 1 });
    });
    await act(async () => {
      dispatchPointer(viewport, 'pointermove', { clientX: 130, clientY: 145, pointerId: 1, buttons: 1 });
    });
    await act(async () => {
      dispatchPointer(viewport, 'pointerup', { clientX: 130, clientY: 145, pointerId: 1 });
    });

    expect(wrapper.style.transform).toBe('translate(30px, 45px) scale(1.15)');
    expect(viewport).toHaveClass('overflow-hidden');

    await user.click(screen.getByText('annotated aleph'));
    expect(await screen.findByText('Retour aux régions')).toBeInTheDocument();
    expect(saveAnalysis).not.toHaveBeenCalled();
    expect(container.querySelector('[data-testid="workspace-overlay"]')).toBeInTheDocument();
  });


  it('uses smallest-area overlay hit priority and keeps workspace overlays read-only', async () => {
    const overlappingRecord: AnalysisRecord = {
      ...RECORDS[0],
      result: {
        ...RECORDS[0].result,
        num_elements: 2,
        elements: [
          {
            ...RECORDS[0].result.elements[0],
            bbox: [80, 80, 240, 220],
            class_name: 'outer',
          },
          {
            ...RECORDS[0].result.elements[1],
            bbox: [100, 120, 50, 40],
            class_name: 'inner',
            rejected: false,
          },
        ],
      },
      annotations: {},
    };
    const initialSnapshot = cloneRecords([overlappingRecord]);
    renderPage(initialSnapshot);

    const overlay = await screen.findByTestId('workspace-overlay') as unknown as SVGSVGElement;
    stubSvgRect(overlay, 800, 600);

    await act(async () => {
      dispatchPointer(overlay, 'pointermove', { clientX: 110, clientY: 130, pointerId: 1, buttons: 0 });
      dispatchPointer(overlay, 'pointerdown', { clientX: 110, clientY: 130, pointerId: 1, buttons: 1 });
    });

    expect(await screen.findByText('Retour aux régions')).toBeInTheDocument();
    expect(screen.queryByText('Région 1')).not.toBeInTheDocument();
    await waitFor(() => expect(getTrust).toHaveBeenCalledWith('data:image/png;base64,alpha', [100, 120, 50, 40], 'inner', 10, expect.objectContaining({ signal: expect.any(AbortSignal) })));
    expect(saveAnalysis).not.toHaveBeenCalled();
    expect(historyRecords).toEqual(initialSnapshot);
  });

  it('keeps the pending upload visible when browser storage save fails', async () => {
    const user = userEvent.setup();
    vi.mocked(saveAnalysis).mockRejectedValueOnce(new Error('IndexedDB quota exceeded'));
    const { container } = renderPage([]);
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    const file = new File(['glyph pixels'], 'glyph.png', { type: 'image/png' });

    await user.upload(fileInput, file);
    await user.click((await screen.findAllByRole('button', { name: 'Analyser' })).at(-1) as HTMLElement);

    await waitFor(() => expect(segmentGlyph).toHaveBeenCalledWith(file));
    expect(await screen.findByText('IndexedDB quota exceeded')).toBeInTheDocument();
    expect(screen.getByText('Image prête à analyser')).toBeInTheDocument();
    expect(screen.getAllByText('glyph.png').length).toBeGreaterThanOrEqual(1);
  });

  it('opens the upload preview, cancels cleanly, and analyzes the selected image', async () => {
    const user = userEvent.setup();
    const { container } = renderPage([]);
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    const file = new File(['glyph pixels'], 'glyph.png', { type: 'image/png' });

    await user.upload(fileInput, file);
    expect(await screen.findByText('Image prête à analyser')).toBeInTheDocument();
    expect(screen.getAllByText('glyph.png').length).toBeGreaterThanOrEqual(1);

    const dialog = screen.getByText('Image prête à analyser').closest('div')?.parentElement as HTMLElement;
    await user.click(within(dialog).getAllByRole('button', { name: 'Annuler' })[0]);
    expect(screen.queryByText('Image prête à analyser')).not.toBeInTheDocument();

    await user.upload(fileInput, file);
    expect(await screen.findByText('Image prête à analyser')).toBeInTheDocument();
    await user.click((await screen.findAllByRole('button', { name: 'Analyser' })).at(-1) as HTMLElement);

    await waitFor(() => expect(segmentGlyph).toHaveBeenCalledWith(file));
    expect(saveAnalysis).toHaveBeenCalledWith(expect.objectContaining({
      id: 'new-analysis-id',
      imageName: 'glyph.png',
      result: SEGMENT_RESULT,
      annotations: {},
    }));
    await waitFor(() => expect(screen.queryByText('Image prête à analyser')).not.toBeInTheDocument());
    expect(screen.getAllByText('glyph.png').length).toBeGreaterThanOrEqual(1);
  });
});
