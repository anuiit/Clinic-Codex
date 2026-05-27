import { describe, expect, it } from 'vitest';
import {
  clampBBox,
  getBoxVisualState,
  hitTestBBoxes,
  hitTestHandle,
  hitTestHandles,
  isDragIntent,
  moveBBox,
  resizeBBox,
  type BBox,
} from './segmentationBoxes';

const imageSize = { width: 100, height: 80 };

describe('segmentationBoxes hit testing', () => {
  it('chooses the smallest containing bbox when boxes overlap', () => {
    const boxes: BBox[] = [
      [0, 0, 90, 70],
      [10, 10, 20, 20],
      [12, 12, 60, 40],
    ];

    expect(hitTestBBoxes({ x: 15, y: 15 }, boxes)).toBe(1);
  });

  it('returns null when no bbox contains the point', () => {
    expect(hitTestBBoxes({ x: 99, y: 79 }, [[0, 0, 10, 10]])).toBeNull();
  });

  it('treats bbox edges as inclusive and keeps first box for same-area ties', () => {
    const boxes: BBox[] = [
      [10, 20, 30, 40],
      [10, 20, 30, 40],
    ];

    expect(hitTestBBoxes({ x: 10, y: 20 }, boxes)).toBe(0);
    expect(hitTestBBoxes({ x: 40, y: 60 }, boxes)).toBe(0);
  });

  it('ignores bboxes with negative dimensions during hit testing', () => {
    const boxes: BBox[] = [
      [0, 0, -20, 30],
      [0, 0, 30, -20],
      [5, 5, 10, 10],
    ];

    expect(hitTestBBoxes({ x: 8, y: 8 }, boxes)).toBe(2);
  });

  it('detects the correct individual corner handle', () => {
    const bbox: BBox = [20, 30, 40, 25];

    expect(hitTestHandle({ x: 20, y: 30 }, bbox, 6)).toBe('tl');
    expect(hitTestHandle({ x: 60, y: 30 }, bbox, 6)).toBe('tr');
    expect(hitTestHandle({ x: 20, y: 55 }, bbox, 6)).toBe('bl');
    expect(hitTestHandle({ x: 60, y: 55 }, bbox, 6)).toBe('br');
  });

  it('chooses the smallest-area bbox when handle hits overlap', () => {
    const boxes: BBox[] = [
      [0, 0, 80, 80],
      [0, 0, 20, 20],
    ];

    expect(hitTestHandles({ x: 0, y: 0 }, boxes, 8)).toEqual({ idx: 1, handle: 'tl' });
  });

  it('returns null for handles outside the hit radius', () => {
    expect(hitTestHandle({ x: 27, y: 30 }, [20, 30, 40, 25], 6)).toBeNull();
    expect(hitTestHandles({ x: 27, y: 30 }, [[20, 30, 40, 25]], 6)).toBeNull();
  });

  it('detects every corner through hitTestHandles and supports exact zero-size handles', () => {
    const boxes: BBox[] = [[20, 30, 40, 25]];

    expect(hitTestHandles({ x: 20, y: 30 }, boxes, 0)).toEqual({ idx: 0, handle: 'tl' });
    expect(hitTestHandles({ x: 60, y: 30 }, boxes, 0)).toEqual({ idx: 0, handle: 'tr' });
    expect(hitTestHandles({ x: 20, y: 55 }, boxes, 0)).toEqual({ idx: 0, handle: 'bl' });
    expect(hitTestHandles({ x: 60, y: 55 }, boxes, 0)).toEqual({ idx: 0, handle: 'br' });
    expect(hitTestHandles({ x: 60.1, y: 55 }, boxes, 0)).toBeNull();
  });
});

describe('segmentationBoxes geometry mutation', () => {
  it('clamps bbox position and size inside image bounds', () => {
    expect(clampBBox([-5, -10, 120, 90], imageSize)).toEqual([0, 0, 100, 80]);
    expect(clampBBox([90, 70, 20, 20], imageSize)).toEqual([80, 60, 20, 20]);
  });

  it('clamps every coordinate to zero when the image has no drawable size', () => {
    expect(clampBBox([10, 20, 30, 40], { width: 0, height: 0 })).toEqual([0, 0, 0, 0]);
    expect(moveBBox([10, 20, 30, 40], { x: 5, y: -5 }, { width: 0, height: 0 })).toEqual([0, 0, 0, 0]);
  });

  it('moves bbox while preserving dimensions and clamping inside the image', () => {
    expect(moveBBox([80, 65, 20, 15], { x: 15, y: 20 }, imageSize)).toEqual([80, 65, 20, 15]);
    expect(moveBBox([10, 10, 20, 15], { x: -20, y: -30 }, imageSize)).toEqual([0, 0, 20, 15]);
  });

  it('preserves fractional deltas while moving inside bounds', () => {
    expect(moveBBox([10.25, 12.5, 20.5, 15.25], { x: 2.5, y: -1.25 }, imageSize)).toEqual([
      12.75,
      11.25,
      20.5,
      15.25,
    ]);
  });

  it('resizes top-left and clamps inside image bounds', () => {
    expect(resizeBBox([10, 10, 40, 30], 'tl', { x: -5, y: 5 }, imageSize)).toEqual([0, 5, 50, 35]);
  });

  it('resizes bottom-right and clamps inside image bounds', () => {
    expect(resizeBBox([70, 60, 20, 10], 'br', { x: 120, y: 120 }, imageSize)).toEqual([70, 60, 30, 20]);
  });

  it('resizes top-right and bottom-left corners with image-bound clamps', () => {
    expect(resizeBBox([70, 10, 20, 30], 'tr', { x: 120, y: -20 }, imageSize)).toEqual([70, 0, 30, 40]);
    expect(resizeBBox([10, 10, 40, 30], 'bl', { x: -5, y: 120 }, imageSize)).toEqual([0, 10, 50, 70]);
  });

  it('keeps resized boxes at least 1px in width and height', () => {
    expect(resizeBBox([10, 10, 40, 30], 'br', { x: 0, y: 0 }, imageSize)).toEqual([10, 10, 1, 1]);
    expect(resizeBBox([10, 10, 40, 30], 'tl', { x: 100, y: 100 }, imageSize)).toEqual([49, 39, 1, 1]);
  });

  it('clamps crossing top-right and bottom-left drags against the opposite edge', () => {
    expect(resizeBBox([10, 10, 40, 30], 'tr', { x: 5, y: 45 }, imageSize)).toEqual([10, 39, 1, 1]);
    expect(resizeBBox([10, 10, 40, 30], 'bl', { x: 60, y: 5 }, imageSize)).toEqual([49, 10, 1, 1]);
  });
});

describe('segmentationBoxes drag intent and visuals', () => {
  it('treats 0-4px client movement as click/select, then starts drag at >=5px', () => {
    const start = { x: 100, y: 100 };

    expect(isDragIntent(start, { x: 100, y: 100 })).toBe(false);
    expect(isDragIntent(start, { x: 104, y: 100 })).toBe(false);
    expect(isDragIntent(start, { x: 103, y: 104 })).toBe(true);
    expect(isDragIntent(start, { x: 105, y: 100 })).toBe(true);
  });

  it('prioritizes visual states deterministically', () => {
    expect(getBoxVisualState({ rejected: true, focused: true }).tone).toBe('rejected');
    expect(getBoxVisualState({ focused: true, hovered: true }).tone).toBe('focused');
    expect(getBoxVisualState({ hovered: true, listHovered: true }).tone).toBe('listHovered');
    expect(getBoxVisualState({ listHovered: true, submitted: true }).tone).toBe('listHovered');
    expect(getBoxVisualState({ submitted: true }).tone).toBe('submitted');
    expect(getBoxVisualState({}).tone).toBe('default');
  });
});
