export const IMAGE_STAGE_MIN_ZOOM = 0.25;
export const IMAGE_STAGE_MAX_ZOOM = 4;
export const IMAGE_STAGE_WHEEL_SENSITIVITY = 0.01;

export type ImageStageZoomBounds = {
  min?: number;
  max?: number;
};

export function clampZoom(
  zoom: number,
  {
    min = IMAGE_STAGE_MIN_ZOOM,
    max = IMAGE_STAGE_MAX_ZOOM,
  }: ImageStageZoomBounds = {},
): number {
  if (!Number.isFinite(zoom)) return min;
  return Math.max(min, Math.min(max, zoom));
}

export function zoomDeltaFromWheel(
  deltaY: number,
  sensitivity = IMAGE_STAGE_WHEEL_SENSITIVITY,
): number {
  if (!Number.isFinite(deltaY) || !Number.isFinite(sensitivity)) return 0;
  return -deltaY * sensitivity;
}

export function nextZoomFromWheel(
  currentZoom: number,
  deltaY: number,
  bounds?: ImageStageZoomBounds,
): number {
  return clampZoom(currentZoom + zoomDeltaFromWheel(deltaY), bounds);
}

export function shouldConsumeStageWheel(deltaY: number): boolean {
  return Number.isFinite(deltaY) && deltaY !== 0;
}
