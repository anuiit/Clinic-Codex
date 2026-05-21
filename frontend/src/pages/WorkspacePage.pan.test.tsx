import { render, fireEvent, act, screen, within } from '@testing-library/react';
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
    expect(viewport).toHaveClass('image-stage-frame', 'image-stage-scrollbar', 'image-stage-grid');

    viewport.setPointerCapture = vi.fn();
    viewport.releasePointerCapture = vi.fn();
    viewport.hasPointerCapture = vi.fn(() => true);

    expect(wrapper.style.transform).toBe('translate(0px, 0px) scale(1)');

    const wheelEvent = new WheelEvent('wheel', { bubbles: true, cancelable: true, deltaY: -100 });
    const preventDefaultSpy = vi.spyOn(wheelEvent, 'preventDefault');
    await act(async () => {
      viewport.dispatchEvent(wheelEvent);
    });
    expect(preventDefaultSpy).toHaveBeenCalled();
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

  it('collapses and expands the history sidebar via stable container animation and compact header', async () => {
    const user = userEvent.setup();
    renderPage();

    const sidebar = screen.getByTestId('workspace-history-sidebar');
    expect(sidebar).toHaveClass('transition-[padding,border-color,background-color]');
    expect(sidebar).not.toHaveClass('transition-all');

    const expandButton = await screen.findByTitle('Déplier l’historique');
    expect(screen.queryByPlaceholderText('Filtrer par glyphe ou classe')).not.toBeInTheDocument();

    await user.click(expandButton);
    expect(await screen.findByPlaceholderText('Filtrer par glyphe ou classe')).toBeInTheDocument();
    const header = screen.getByTestId('workspace-history-header');
    expect(within(header).getByRole('heading', { name: 'History' })).toBeInTheDocument();
    expect(header).toHaveTextContent('1 total');
    expect(screen.queryByText('Historique des analyses')).not.toBeInTheDocument();
    expect(screen.queryByText('Analyses enregistrées')).not.toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'workspace-test.png' })).toBeInTheDocument();

    await user.click(screen.getByTitle('Replier l’historique'));
    expect(screen.queryByPlaceholderText('Filtrer par glyphe ou classe')).not.toBeInTheDocument();
  });

  it('focuses a single overlay region and returns to the full overlay view', async () => {
    const user = userEvent.setup();
    const { container } = renderPage();

    const overlay = await screen.findByTestId('workspace-overlay') as unknown as SVGSVGElement;
    overlay.getBoundingClientRect = vi.fn().mockReturnValue({
      left: 0,
      top: 0,
      width: 800,
      height: 600,
      right: 800,
      bottom: 600,
      x: 0,
      y: 0,
      toJSON: () => {},
    });
    expect(container.querySelector('[data-overlay-region="true"]')).toBeTruthy();

    await act(async () => {
      dispatchPointer(overlay, 'pointerdown', { clientX: 125, clientY: 120, pointerId: 1, buttons: 1 });
    });
    expect(await screen.findByText('Retour aux régions')).toBeInTheDocument();
    expect(screen.queryByText('Région 0')).not.toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Retour aux régions' }));
    expect(screen.queryByText('Retour aux régions')).not.toBeInTheDocument();
  });
});
