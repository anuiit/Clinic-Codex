import { describe, expect, it } from "vitest";
import {
  clampZoom,
  IMAGE_STAGE_MAX_ZOOM,
  IMAGE_STAGE_MIN_ZOOM,
  nextZoomFromWheel,
  shouldConsumeStageWheel,
  zoomDeltaFromWheel,
} from "./imageStageZoom";

describe("image stage zoom primitives", () => {
  it("clamps zoom to the shared image-stage bounds", () => {
    expect(clampZoom(0)).toBe(IMAGE_STAGE_MIN_ZOOM);
    expect(clampZoom(10)).toBe(IMAGE_STAGE_MAX_ZOOM);
    expect(clampZoom(1.5)).toBe(1.5);
  });

  it("maps wheel delta to direct stage zoom and clamps min/max", () => {
    expect(zoomDeltaFromWheel(-25)).toBe(0.25);
    expect(nextZoomFromWheel(1, -25)).toBe(1.25);
    expect(nextZoomFromWheel(1, 1000)).toBe(IMAGE_STAGE_MIN_ZOOM);
    expect(nextZoomFromWheel(4, -1000)).toBe(IMAGE_STAGE_MAX_ZOOM);
  });

  it("consumes only finite non-zero wheel movement", () => {
    expect(shouldConsumeStageWheel(-1)).toBe(true);
    expect(shouldConsumeStageWheel(0)).toBe(false);
    expect(shouldConsumeStageWheel(Number.NaN)).toBe(false);
  });
});
