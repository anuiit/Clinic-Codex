import { render, fireEvent, act, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { AnalysisRecord } from '../types';
import WorkspacePage from './WorkspacePage';

const RECORD: AnalysisRecord = {
  id: 'analysis-id',
  imageName: 'workspace-test.png',
  imageDataUrl: 'data:image/png;base64,abc',
  timestamp: 1704067200000,
  result: {
    num_elements: 1,
    image_size: [800, 600],
    elements: [
      {
        bbox: [100, 100, 50, 40],
        class_name: 'atl',
        class_label: 1,
        confidence: 0.9,
        rejected: false,
        top_k: [],
      },
    ],
  },
  annotations: {},
};

vi.mock('../services/storage', () => ({
  deleteAnalysis: vi.fn(),
  getHistory: vi.fn(() => [RECORD]),
  saveAnalysis: vi.fn(),
}));

vi.mock('../services/api', () => ({
  getTrust: vi.fn(() => Promise.resolve(null)),
  segmentGlyph: vi.fn(),
}));

function renderPage() {
  return render(
    <MemoryRouter initialEntries={['/']}>
      <Routes>
        <Route path="/" element={<WorkspacePage />} />
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

describe('WorkspacePage image pan behavior', () => {
  beforeEach(() => {
    HTMLCanvasElement.prototype.getContext = vi.fn(() => ({
      clearRect: vi.fn(),
      drawImage: vi.fn(),
    })) as unknown as typeof HTMLCanvasElement.prototype.getContext;
  });

  it('pans the image and overlay wrapper together after wheel zoom', async () => {
    const { container } = renderPage();
    const images = await screen.findAllByAltText('workspace-test.png');
    const image = images.find((candidate) => candidate.className.includes('object-contain')) as HTMLImageElement;
    const wrapper = image.parentElement as HTMLElement;
    const viewport = wrapper.parentElement as HTMLElement;

    viewport.setPointerCapture = vi.fn();
    viewport.releasePointerCapture = vi.fn();
    viewport.hasPointerCapture = vi.fn(() => true);

    expect(wrapper.style.transform).toBe('translate(0px, 0px) scale(1)');

    await act(async () => {
      fireEvent.wheel(viewport, { deltaY: -100 });
    });
    expect(wrapper.style.transform).toContain('scale(1.15)');

    await act(async () => {
      dispatchPointer(viewport, 'pointerdown', { clientX: 100, clientY: 100, pointerId: 1, buttons: 1 });
    });
    await act(async () => {
      dispatchPointer(viewport, 'pointermove', { clientX: 135, clientY: 150, pointerId: 1, buttons: 1 });
    });
    await act(async () => {
      dispatchPointer(viewport, 'pointerup', { clientX: 135, clientY: 150, pointerId: 1 });
    });

    expect(wrapper.style.transform).toBe('translate(35px, 50px) scale(1.15)');

    const overlay = container.querySelector('svg.absolute') as SVGSVGElement;
    expect(overlay.getAttribute('preserveAspectRatio')).toBe('none');
  });

  it('collapses and expands the history sidebar via its toggle buttons', async () => {
    const user = userEvent.setup();
    renderPage();

    const expandButton = await screen.findByTitle('Déplier l’historique');
    expect(screen.queryByPlaceholderText('Filtrer par glyphe ou classe')).not.toBeInTheDocument();

    await user.click(expandButton);
    expect(await screen.findByPlaceholderText('Filtrer par glyphe ou classe')).toBeInTheDocument();
    expect(screen.getByText('workspace-test.png')).toBeInTheDocument();

    await user.click(screen.getByTitle('Replier l’historique'));
    expect(screen.queryByPlaceholderText('Filtrer par glyphe ou classe')).not.toBeInTheDocument();
  });

  it('focuses a single overlay region and returns to the full overlay view', async () => {
    const user = userEvent.setup();
    const { container } = renderPage();

    const overlayRegion = container.querySelector('[data-overlay-region="true"]') as SVGGElement;
    expect(overlayRegion).toBeTruthy();

    await user.click(overlayRegion);
    expect(await screen.findByText('Retour aux régions')).toBeInTheDocument();
    expect(screen.getByText('Region 0')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Retour aux régions' }));
    expect(screen.queryByText('Retour aux régions')).not.toBeInTheDocument();
  });
});
