export type BBox = [number, number, number, number];
export type BBoxHandle = 'tl' | 'tr' | 'bl' | 'br';

export interface Point {
  x: number;
  y: number;
}

export interface ImageSize {
  width: number;
  height: number;
}

export interface BoxVisualInput {
  focused?: boolean;
  hovered?: boolean;
  listHovered?: boolean;
  submitted?: boolean;
  rejected?: boolean;
}

export interface BoxVisualState {
  tone: 'default' | 'focused' | 'hovered' | 'listHovered' | 'submitted' | 'rejected';
  strokeColor: string;
  fillColor: string;
  labelColor: string;
  strokeWidth: number;
  strokeDasharray?: string;
}

const HANDLE_CORNERS: Array<{ handle: BBoxHandle; point: (bbox: BBox) => Point }> = [
  { handle: 'tl', point: ([x, y]) => ({ x, y }) },
  { handle: 'tr', point: ([x, y, w]) => ({ x: x + w, y }) },
  { handle: 'bl', point: ([x, y, , h]) => ({ x, y: y + h }) },
  { handle: 'br', point: ([x, y, w, h]) => ({ x: x + w, y: y + h }) },
];

function bboxArea([, , width, height]: BBox): number {
  return Math.max(0, width) * Math.max(0, height);
}

function containsPoint({ x, y }: Point, [boxX, boxY, width, height]: BBox): boolean {
  return width >= 0 && height >= 0 && x >= boxX && x <= boxX + width && y >= boxY && y <= boxY + height;
}

function clamp(value: number, min: number, max: number): number {
  if (max < min) return min;
  return Math.max(min, Math.min(max, value));
}

export function hitTestBBoxes(point: Point, boxes: BBox[]): number | null {
  let bestIdx: number | null = null;
  let bestArea = Infinity;

  boxes.forEach((bbox, idx) => {
    if (!containsPoint(point, bbox)) return;

    const area = bboxArea(bbox);
    if (area < bestArea) {
      bestArea = area;
      bestIdx = idx;
    }
  });

  return bestIdx;
}

export function hitTestHandle(point: Point, bbox: BBox, handleSize: number): BBoxHandle | null {
  let best: { handle: BBoxHandle; distanceSq: number } | null = null;
  const radius = Math.max(0, handleSize);

  for (const corner of HANDLE_CORNERS) {
    const handlePoint = corner.point(bbox);
    const dx = point.x - handlePoint.x;
    const dy = point.y - handlePoint.y;
    if (Math.abs(dx) > radius || Math.abs(dy) > radius) continue;

    const distanceSq = dx * dx + dy * dy;
    if (!best || distanceSq < best.distanceSq) {
      best = { handle: corner.handle, distanceSq };
    }
  }

  return best?.handle ?? null;
}

export function hitTestHandles(point: Point, boxes: BBox[], handleSize: number): { idx: number; handle: BBoxHandle } | null {
  let bestHit: { idx: number; handle: BBoxHandle; area: number } | null = null;

  for (let idx = 0; idx < boxes.length; idx += 1) {
    const bbox = boxes[idx];
    const handle = hitTestHandle(point, bbox, handleSize);
    if (!handle) continue;

    const area = bboxArea(bbox);
    if (bestHit === null || area < bestHit.area) {
      bestHit = { idx, handle, area };
    }
  }

  if (bestHit === null) return null;
  return { idx: bestHit.idx, handle: bestHit.handle };
}

export function clampBBox(bbox: BBox, imageSize: ImageSize): BBox {
  const imageWidth = Math.max(0, imageSize.width);
  const imageHeight = Math.max(0, imageSize.height);
  const width = clamp(bbox[2], 0, imageWidth);
  const height = clamp(bbox[3], 0, imageHeight);
  const x = clamp(bbox[0], 0, imageWidth - width);
  const y = clamp(bbox[1], 0, imageHeight - height);

  return [x, y, width, height];
}

export function moveBBox(origBBox: BBox, delta: Point, imageSize: ImageSize): BBox {
  return clampBBox([origBBox[0] + delta.x, origBBox[1] + delta.y, origBBox[2], origBBox[3]], imageSize);
}

export function resizeBBox(origBBox: BBox, handle: BBoxHandle, point: Point, imageSize: ImageSize): BBox {
  const [origX, origY, origW, origH] = origBBox;
  const maxX = origX + origW;
  const maxY = origY + origH;
  let nextX = origX;
  let nextY = origY;
  let nextW = origW;
  let nextH = origH;

  if (handle === 'tl' || handle === 'bl') {
    nextX = clamp(point.x, 0, maxX - 1);
    nextW = maxX - nextX;
  }

  if (handle === 'tr' || handle === 'br') {
    nextW = clamp(point.x - origX, 1, Math.max(1, imageSize.width - origX));
  }

  if (handle === 'tl' || handle === 'tr') {
    nextY = clamp(point.y, 0, maxY - 1);
    nextH = maxY - nextY;
  }

  if (handle === 'bl' || handle === 'br') {
    nextH = clamp(point.y - origY, 1, Math.max(1, imageSize.height - origY));
  }

  return clampBBox([nextX, nextY, nextW, nextH], imageSize);
}

export function isDragIntent(startClient: Point, currentClient: Point, thresholdPx = 5): boolean {
  const dx = currentClient.x - startClient.x;
  const dy = currentClient.y - startClient.y;
  return Math.hypot(dx, dy) >= thresholdPx;
}

export function getBoxVisualState({ focused, hovered, listHovered, submitted, rejected }: BoxVisualInput): BoxVisualState {
  if (rejected) {
    return {
      tone: 'rejected',
      strokeColor: '#fb7185',
      fillColor: 'rgba(239, 68, 68, 0.16)',
      labelColor: '#0c0a09',
      strokeWidth: 2,
    };
  }

  if (focused) {
    return {
      tone: 'focused',
      strokeColor: '#fbbf24',
      fillColor: 'rgba(245, 158, 11, 0.18)',
      labelColor: '#0c0a09',
      strokeWidth: 3,
    };
  }

  if (hovered) {
    return {
      tone: 'hovered',
      strokeColor: '#f59e0b',
      fillColor: 'rgba(245, 158, 11, 0.11)',
      labelColor: '#0c0a09',
      strokeWidth: 2,
    };
  }

  if (listHovered) {
    return {
      tone: 'listHovered',
      strokeColor: '#38bdf8',
      fillColor: 'rgba(56, 189, 248, 0.12)',
      labelColor: '#082f49',
      strokeWidth: 2,
    };
  }

  if (submitted) {
    return {
      tone: 'submitted',
      strokeColor: '#34d399',
      fillColor: 'rgba(16, 185, 129, 0.15)',
      labelColor: '#052e16',
      strokeWidth: 2,
    };
  }

  return {
    tone: 'default',
    strokeColor: '#a8a29e',
    fillColor: 'rgba(168, 162, 158, 0.08)',
    labelColor: '#0c0a09',
    strokeWidth: 2,
    strokeDasharray: '8 5',
  };
}
