import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
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

vi.mock('../services/storage', () => ({
  deleteAnalysis: vi.fn((id: string) => {
    historyRecords = historyRecords.filter((record) => record.id !== id);
  }),
  getHistory: vi.fn(() => historyRecords),
  saveAnalysis: vi.fn((record: AnalysisRecord) => {
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

function renderPage(initialRecords = RECORDS, initialEntry = '/') {
  historyRecords = cloneRecords(initialRecords);
  return render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <Routes>
        <Route path="/" element={<WorkspacePage />} />
        <Route path="/annotate/:id" element={<div>annotation handoff</div>} />
      </Routes>
    </MemoryRouter>,
  );
}

function stubCanvas() {
  HTMLCanvasElement.prototype.getContext = vi.fn(() => ({
    clearRect: vi.fn(),
    drawImage: vi.fn(),
  })) as unknown as typeof HTMLCanvasElement.prototype.getContext;
}

describe('WorkspacePage interaction coverage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    stubCanvas();
    vi.stubGlobal('crypto', { randomUUID: () => 'new-analysis-id' });
  });

  it('filters history results, selects a matching run, and deletes the active run', async () => {
    const user = userEvent.setup();
    renderPage();

    await user.type(screen.getByPlaceholderText('Filtrer par glyphe ou classe'), 'beta');
    expect(screen.getByText('1')).toBeInTheDocument();
    expect(screen.getByText('beta.png')).toBeInTheDocument();

    await user.click(screen.getByText('beta.png'));
    expect(screen.getAllByText('beta.png').length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('bet')).toBeInTheDocument();

    await user.click(screen.getByLabelText('Supprimer beta.png'));

    expect(deleteAnalysis).toHaveBeenCalledWith('beta-run');
    expect(screen.getByText('alpha.png')).toBeInTheDocument();
    expect(screen.queryByText('beta.png')).not.toBeInTheDocument();
  });

  it('toggles overlays and opens focused region details without starting a pan', async () => {
    const user = userEvent.setup();
    const { container } = renderPage();

    expect(container.querySelector('svg.absolute')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'masqué' }));
    expect(container.querySelector('svg.absolute')).not.toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'tout' }));
    expect(container.querySelector('svg.absolute')).toBeInTheDocument();

    await user.click(screen.getAllByRole('button', { name: 'Voir plus de détails' })[0]);

    expect(await screen.findByText('Retour aux régions')).toBeInTheDocument();
    expect(screen.getByText('Aperçu du segment')).toBeInTheDocument();
    await waitFor(() => expect(getTrust).toHaveBeenCalledWith('data:image/png;base64,alpha', [100, 120, 50, 40], 'aleph', 10));

    await user.click(screen.getByRole('button', { name: /Retour aux régions/ }));
    expect(screen.getByText('Éléments détectés')).toBeInTheDocument();
  });

  it('opens the upload preview, cancels cleanly, and analyzes the selected image', async () => {
    const user = userEvent.setup();
    const { container } = renderPage([]);
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    const file = new File(['glyph pixels'], 'glyph.png', { type: 'image/png' });

    await user.upload(fileInput, file);
    expect(await screen.findByText('Image prête à analyser')).toBeInTheDocument();
    expect(screen.getByText('glyph.png')).toBeInTheDocument();

    const dialog = screen.getByText('Image prête à analyser').closest('div')?.parentElement as HTMLElement;
    await user.click(within(dialog).getAllByRole('button', { name: 'Annuler' })[0]);
    expect(screen.queryByText('Image prête à analyser')).not.toBeInTheDocument();

    await user.upload(fileInput, file);
    expect(await screen.findByText('Image prête à analyser')).toBeInTheDocument();
    await user.click(screen.getAllByRole('button', { name: 'Analyser' }).at(-1) as HTMLElement);

    await waitFor(() => expect(segmentGlyph).toHaveBeenCalledWith(file));
    expect(saveAnalysis).toHaveBeenCalledWith(expect.objectContaining({
      id: 'new-analysis-id',
      imageName: 'glyph.png',
      result: SEGMENT_RESULT,
      annotations: {},
    }));
    await waitFor(() => expect(screen.queryByText('Image prête à analyser')).not.toBeInTheDocument());
    expect(screen.getByText('glyph.png')).toBeInTheDocument();
  });
});
