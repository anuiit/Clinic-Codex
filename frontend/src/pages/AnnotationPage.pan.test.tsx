import { render, fireEvent, act, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { vi, describe, it, expect, beforeEach } from 'vitest';

vi.mock('../services/storage', () => ({
  getAnalysisById: vi.fn(),
  updateElements: vi.fn(() => true),
}));

vi.mock('../services/api', () => ({
  getClasses: vi.fn(() => Promise.resolve({ class_names: [] })),
  saveAnnotation: vi.fn(),
}));

import { getAnalysisById, updateElements } from '../services/storage';
import AnnotationPage from './AnnotationPage';
import type { AnalysisRecord } from '../types';

const STUB_RECORD: AnalysisRecord = {
  id: 'test-id',
  imageDataUrl: 'data:image/png;base64,abc',
  imageName: 'test.png',
  timestamp: 1704067200000,
  result: {
    num_elements: 0,
    image_size: [800, 600] as [number, number],
    elements: [],
  },
  annotations: {},
};

function renderPage(record: AnalysisRecord = STUB_RECORD) {
  vi.mocked(getAnalysisById).mockReturnValue(record);
  return render(
    <MemoryRouter initialEntries={['/annotation/test-id']}>
      <Routes>
        <Route path="/annotation/:id" element={<AnnotationPage />} />
        <Route path="/" element={<div>home</div>} />
      </Routes>
    </MemoryRouter>,
  );
}

const CONTAINER_RECT = {
  left: 0, top: 0, width: 800, height: 600,
  right: 800, bottom: 600, x: 0, y: 0, toJSON: () => {},
} as DOMRect;

let measuredRect: DOMRect = CONTAINER_RECT;
let drawImageMock = vi.fn();
let clearRectMock = vi.fn();

beforeEach(() => {
  vi.mocked(getAnalysisById).mockReturnValue(STUB_RECORD);
  vi.mocked(updateElements).mockClear();
  measuredRect = CONTAINER_RECT;
  drawImageMock = vi.fn();
  clearRectMock = vi.fn();
  Element.prototype.getBoundingClientRect = vi.fn(() => measuredRect);
  HTMLElement.prototype.getBoundingClientRect = vi.fn(() => measuredRect);
  SVGElement.prototype.getBoundingClientRect = vi.fn(() => measuredRect);
  Object.defineProperty(HTMLImageElement.prototype, 'complete', {
    configurable: true,
    get: () => true,
  });
  HTMLCanvasElement.prototype.getContext = vi.fn(() => ({
    clearRect: clearRectMock,
    drawImage: drawImageMock,
    imageSmoothingEnabled: false,
  })) as unknown as typeof HTMLCanvasElement.prototype.getContext;
});

function getSvgAndWrapper(container: HTMLElement): { svg: SVGSVGElement; wrapper: HTMLElement } {
  const svg = container.querySelector('svg.absolute') as SVGSVGElement;
  svg.getBoundingClientRect = vi.fn(() => CONTAINER_RECT);
  svg.setPointerCapture = vi.fn();
  svg.releasePointerCapture = vi.fn();
  svg.hasPointerCapture = vi.fn(() => false);
  return { svg, wrapper: svg.parentElement as HTMLElement };
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

async function clickZoomIn(container: HTMLElement, times: number) {
  const buttons = container.querySelectorAll('button');
  const zoomButtons = Array.from(buttons).filter(b => b.className.includes('p-1.5') || b.className.includes('p-1'));
  const zoomInButton = zoomButtons[0] as HTMLElement;
  for (let i = 0; i < times; i++) {
    await act(async () => { fireEvent.click(zoomInButton); });
  }
}

describe('AnnotationPage pan behavior', () => {
  it('renders SVG overlays without preserveAspectRatio letterboxing', async () => {
    const { container } = renderPage();
    await act(async () => {});

    const { svg } = getSvgAndWrapper(container);
    expect(svg.getAttribute('preserveAspectRatio')).toBe('none');
  });

  it('keeps image and SVG overlay locked to the same measured stage after resize', async () => {
    measuredRect = {
      left: 0, top: 0, width: 400, height: 300,
      right: 400, bottom: 300, x: 0, y: 0, toJSON: () => {},
    } as DOMRect;

    renderPage();
    await act(async () => {});

    const stage = screen.getByTestId('annotation-stage');
    const overlay = screen.getByTestId('annotation-overlay');

    await waitFor(() => {
      expect(stage).toHaveStyle({ width: '400px', height: '300px' });
      expect(overlay).toHaveStyle({ width: '100%', height: '100%' });
    });

    measuredRect = {
      left: 0, top: 0, width: 300, height: 300,
      right: 300, bottom: 300, x: 0, y: 0, toJSON: () => {},
    } as DOMRect;

    await act(async () => {
      window.dispatchEvent(new Event('resize'));
    });

    await waitFor(() => {
      expect(stage).toHaveStyle({ width: '300px', height: '225px' });
      expect(overlay).toHaveStyle({ width: '100%', height: '100%' });
    });
  });

  it('fits the selected segment preview crop into the canvas without clipping', async () => {
    renderPage({
      ...STUB_RECORD,
      result: {
        ...STUB_RECORD.result,
        num_elements: 1,
        elements: [
          {
            bbox: [100, 120, 500, 100],
            class_name: 'wide',
            class_label: 1,
            confidence: 0.9,
            rejected: false,
            top_k: [],
          },
        ],
      },
    });

    const card = await screen.findByText('wide');
    await act(async () => {
      fireEvent.click(card);
    });

    await waitFor(() => expect(drawImageMock).toHaveBeenCalled());
    const [, sourceX, sourceY, sourceW, sourceH, drawX, drawY, drawW, drawH] = drawImageMock.mock.calls.at(-1)!;

    expect([sourceX, sourceY, sourceW, sourceH]).toEqual([100, 120, 500, 100]);
    expect(drawX).toBe(0);
    expect(drawY).toBe(80);
    expect(drawW).toBe(200);
    expect(drawH).toBe(40);
  });

  it('at zoom=1, pointerdown+move on background does NOT change wrapper transform', async () => {
    const { container } = renderPage();
    await act(async () => {});

    const { svg, wrapper } = getSvgAndWrapper(container);
    const transformBefore = wrapper.style.transform;

    await act(async () => {
      fireEvent.pointerDown(svg, { clientX: 100, clientY: 100, pointerId: 1 });
    });
    await act(async () => {
      fireEvent.pointerMove(svg, { clientX: 250, clientY: 250, pointerId: 1 });
    });
    await act(async () => {
      fireEvent.pointerUp(svg, { clientX: 250, clientY: 250, pointerId: 1 });
    });

    expect(wrapper.style.transform).toBe(transformBefore);
  });

  it('at zoom=2, pointerdown+move on background updates wrapper transform with translate', async () => {
    const { container } = renderPage();
    await act(async () => {});

    await clickZoomIn(container, 4);

    const { svg, wrapper } = getSvgAndWrapper(container);

    await act(async () => {
      fireEvent.pointerDown(svg, { clientX: 100, clientY: 100, pointerId: 1 });
    });
    await act(async () => {
      fireEvent.pointerMove(svg, { clientX: 250, clientY: 250, pointerId: 1 });
    });
    await act(async () => {
      fireEvent.pointerUp(svg, { clientX: 250, clientY: 250, pointerId: 1 });
    });

    expect(wrapper.style.transform).toMatch(/translate\(/);
    expect(wrapper.style.transform).not.toMatch(/translate\(0px,\s*0px\)/);
  });

  it('reset-view button resets pan and zoom', async () => {
    const { container } = renderPage();
    await act(async () => {});

    await clickZoomIn(container, 4);

    const { svg, wrapper } = getSvgAndWrapper(container);

    await act(async () => {
      fireEvent.pointerDown(svg, { clientX: 100, clientY: 100, pointerId: 1 });
    });
    await act(async () => {
      fireEvent.pointerMove(svg, { clientX: 250, clientY: 250, pointerId: 1 });
    });
    await act(async () => {
      fireEvent.pointerUp(svg, { clientX: 250, clientY: 250, pointerId: 1 });
    });

    const buttons = container.querySelectorAll('button');
    const resetBtn = Array.from(buttons).find(b => b.title === 'Réinitialiser la vue') as HTMLElement;

    await act(async () => {
      fireEvent.click(resetBtn);
    });

    expect(wrapper.style.transform).toMatch(/translate\(0px,\s*0px\)\s*scale\(1\)/);
  });

  it('dragging an existing bbox changes the bbox without panning the image wrapper', async () => {
    const recordWithElement: AnalysisRecord = {
      ...STUB_RECORD,
      result: {
        ...STUB_RECORD.result,
        num_elements: 1,
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
    };

    const { container } = renderPage(recordWithElement);
    await screen.findByText('atl');

    const { svg, wrapper } = getSvgAndWrapper(container);
    const transformBefore = wrapper.style.transform;

    await act(async () => {
      dispatchPointer(svg, 'pointerdown', { clientX: 125, clientY: 125, pointerId: 1, buttons: 1 });
    });
    await act(async () => {});
    await act(async () => {
      dispatchPointer(svg, 'pointermove', { clientX: 155, clientY: 165, pointerId: 1, buttons: 1 });
    });
    await act(async () => {
      dispatchPointer(svg, 'pointerup', { clientX: 155, clientY: 165, pointerId: 1 });
    });
    await act(async () => {});

    expect(wrapper.style.transform).toBe(transformBefore);

    const saveButton = Array.from(container.querySelectorAll('button')).find((button) =>
      button.textContent?.includes('Enregistrer les modifications'),
    ) as HTMLElement;
    await act(async () => {
      fireEvent.click(saveButton);
    });

    expect(updateElements).toHaveBeenCalledWith('test-id', [
      expect.objectContaining({ bbox: [130, 140, 50, 40] }),
    ], { 0: 'draft' });
  });
});
