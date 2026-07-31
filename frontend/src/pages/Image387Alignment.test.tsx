import { readFileSync } from 'node:fs';
import { render, fireEvent, act, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { AnalysisRecord } from '../types';
import AnnotationPage from './AnnotationPage';
import WorkspacePage from './WorkspacePage';

const imageBytes = readFileSync('src/test/fixtures/387_769v.jpg');
const imageDataUrl = `data:image/jpeg;base64,${imageBytes.toString('base64')}`;
const IMAGE_SIZE: [number, number] = [750, 1210];

const record387: AnalysisRecord = {
  id: 'image-387-769v',
  imageName: '387_769v.jpg',
  imageDataUrl,
  timestamp: 1704067200000,
  result: {
    num_elements: 1,
    image_size: IMAGE_SIZE,
    elements: [
      {
        bbox: [120, 240, 150, 180],
        class_name: 'atl',
        class_label: 1,
        confidence: 0.9,
        rejected: false,
        top_k: [],
      },
    ],
  },
  annotations: {},
  annotationStatus: { 0: 'validated' },
};

let historyRecords: AnalysisRecord[] = [record387];
let annotationRecord: AnalysisRecord | null = record387;

vi.mock('../services/storage', () => ({
  deleteAnalysis: vi.fn(async () => undefined),
  getAnalysisById: vi.fn(async () => annotationRecord),
  getHistory: vi.fn(async () => historyRecords),
  saveAnalysis: vi.fn(async () => undefined),
  updateElements: vi.fn(async () => true),
}));

vi.mock('../services/api', () => ({
  getClasses: vi.fn(() => Promise.resolve({ num_classes: 1, class_names: ['atl'] })),
  getTrust: vi.fn(() => Promise.resolve(null)),
  saveAnnotation: vi.fn(),
  segmentGlyph: vi.fn(),
}));

import { updateElements } from '../services/storage';

const FIXTURE_RECT = {
  left: 25,
  top: 10,
  width: 375,
  height: 605,
  right: 400,
  bottom: 615,
  x: 25,
  y: 10,
  toJSON: () => {},
} as DOMRect;

const PADDED_STAGE_RECT = {
  left: 0,
  top: 0,
  width: 395,
  height: 625,
  right: 395,
  bottom: 625,
  x: 0,
  y: 0,
  toJSON: () => {},
} as DOMRect;

function applyPaddedStageRect(stage: HTMLElement) {
  stage.style.paddingLeft = '5px';
  stage.style.paddingRight = '5px';
  stage.style.paddingTop = '5px';
  stage.style.paddingBottom = '5px';
  stage.style.borderLeftWidth = '5px';
  stage.style.borderRightWidth = '5px';
  stage.style.borderTopWidth = '5px';
  stage.style.borderBottomWidth = '5px';
  stage.getBoundingClientRect = vi.fn(() => PADDED_STAGE_RECT);
}

async function measureWorkspaceTransform() {
  const view = renderWorkspacePage();
  const images = await screen.findAllByAltText('387_769v.jpg');
  const image = images.find((candidate) => candidate.className.includes('object-fill')) as HTMLImageElement;
  const wrapper = image.parentElement as HTMLElement;
  const stage = await screen.findByTestId('workspace-stage');
  applyPaddedStageRect(stage);

  await act(async () => {
    fireEvent.load(image);
    window.dispatchEvent(new Event('resize'));
  });
  await waitFor(() => expect(wrapper.style.width).toBe('375px'));

  const measured = { width: wrapper.style.width, height: wrapper.style.height };
  view.unmount();
  return measured;
}

async function measureAnnotationTransform() {
  const view = renderAnnotationPage();
  const image = await screen.findByAltText('387_769v.jpg') as HTMLImageElement;
  const wrapper = image.parentElement as HTMLElement;
  const stage = await screen.findByTestId('annotation-stage-frame');
  applyPaddedStageRect(stage);

  await act(async () => {
    fireEvent.load(image);
    window.dispatchEvent(new Event('resize'));
  });
  await waitFor(() => expect(wrapper.style.width).toBe('375px'));

  const measured = { width: wrapper.style.width, height: wrapper.style.height };
  view.unmount();
  return measured;
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

function renderAnnotationPage() {
  return render(
    <MemoryRouter initialEntries={['/annotation/image-387-769v']}>
      <Routes>
        <Route path="/annotation/:id" element={<AnnotationPage />} />
        <Route path="/" element={<div>home</div>} />
      </Routes>
    </MemoryRouter>,
  );
}

function renderWorkspacePage() {
  return render(
    <MemoryRouter initialEntries={['/']}>
      <Routes>
        <Route path="/" element={<WorkspacePage />} />
      </Routes>
    </MemoryRouter>,
  );
}

describe('387_769v.jpg visual geometry regression', () => {
  beforeEach(() => {
    historyRecords = [record387];
    annotationRecord = record387;
    vi.mocked(updateElements).mockClear();
    HTMLCanvasElement.prototype.getContext = vi.fn(() => ({
      clearRect: vi.fn(),
      drawImage: vi.fn(),
      imageSmoothingEnabled: false,
    })) as unknown as typeof HTMLCanvasElement.prototype.getContext;
  });



  it('computes identical measured transform dimensions for Workspace and Annotation padded stage rects', async () => {
    const workspaceMeasured = await measureWorkspaceTransform();
    const annotationMeasured = await measureAnnotationTransform();

    expect(workspaceMeasured).toEqual({ width: '375px', height: '605px' });
    expect(annotationMeasured).toEqual(workspaceMeasured);
  });

  it('keeps AnnotationPage image and SVG overlay in one measured wrapper for the 387 fixture', async () => {
    const { container } = renderAnnotationPage();
    const image = await screen.findByAltText('387_769v.jpg') as HTMLImageElement;
    const svg = container.querySelector('svg.absolute') as SVGSVGElement;
    const wrapper = svg.parentElement as HTMLElement;

    expect(imageDataUrl).toMatch(/^data:image\/jpeg;base64,/);
    expect(svg.getAttribute('viewBox')).toBe('0 0 750 1210');
    expect(svg.getAttribute('preserveAspectRatio')).toBe('none');
    expect(image).toHaveClass('object-fill');
    expect(wrapper.contains(image)).toBe(true);
    expect(wrapper.contains(svg)).toBe(true);
    expect(wrapper.style.transform).toBe('translate(0px, 0px) scale(1)');
  });

  it('keeps WorkspacePage image and overlay in one transformed wrapper for the 387 fixture', async () => {
    const { container } = renderWorkspacePage();
    const images = await screen.findAllByAltText('387_769v.jpg');
    const image = images.find((candidate) => candidate.className.includes('object-fill')) as HTMLImageElement;
    const wrapper = image.parentElement as HTMLElement;
    const viewport = wrapper.parentElement as HTMLElement;
    const overlay = container.querySelector('svg.absolute') as SVGSVGElement;

    image.getBoundingClientRect = vi.fn(() => FIXTURE_RECT);
    overlay.getBoundingClientRect = vi.fn(() => FIXTURE_RECT);
    viewport.setPointerCapture = vi.fn();
    viewport.releasePointerCapture = vi.fn();
    viewport.hasPointerCapture = vi.fn(() => true);

    expect(overlay.getAttribute('viewBox')).toBe('0 0 750 1210');
    expect(overlay.getAttribute('preserveAspectRatio')).toBe('none');
    expect(wrapper.contains(image)).toBe(true);
    expect(wrapper.contains(overlay)).toBe(true);
    expect(wrapper.style.transform).toBe('translate(0px, 0px) scale(1)');

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Zoom avant' }));
    });
    await act(async () => {
      dispatchPointer(viewport, 'pointerdown', { clientX: 100, clientY: 100, pointerId: 1, buttons: 1 });
    });
    await act(async () => {
      dispatchPointer(viewport, 'pointermove', { clientX: 130, clientY: 155, pointerId: 1, buttons: 1 });
    });
    await act(async () => {
      dispatchPointer(viewport, 'pointerup', { clientX: 130, clientY: 155, pointerId: 1 });
    });

    expect(wrapper.style.transform).toBe('translate(30px, 55px) scale(1.25)');
  });
});
