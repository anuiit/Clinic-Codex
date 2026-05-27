import { describe, expect, it, vi } from "vitest";
import type { ImageBBoxStageBox } from "./imageBBoxStage.types";
import {
  getBBoxOverlayHit,
  getBBoxOverlayRenderState,
} from "./useBBoxOverlayHitTest";

function svgWithRect(width = 800, height = 600) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
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
  return svg;
}

const boxes: ImageBBoxStageBox[] = [
  { id: "outer", bbox: [80, 80, 240, 220], label: "outer" },
  { id: "inner", bbox: [100, 120, 50, 40], label: "inner" },
];

describe("useBBoxOverlayHitTest shared helpers", () => {
  it("returns the smallest-area hit in all mode", () => {
    expect(
      getBBoxOverlayHit(svgWithRect(), 110, 130, {
        imageSize: [800, 600],
        boxes,
        overlayMode: "all",
      }),
    ).toBe("inner");
  });

  it("respects focused, hidden, and display-bbox modes", () => {
    const svg = svgWithRect();

    expect(
      getBBoxOverlayHit(svg, 110, 130, {
        imageSize: [800, 600],
        boxes,
        selectedId: "outer",
        overlayMode: "focused",
      }),
    ).toBe("outer");
    expect(
      getBBoxOverlayHit(svg, 110, 130, {
        imageSize: [800, 600],
        boxes,
        overlayMode: "hidden",
      }),
    ).toBeNull();
    expect(
      getBBoxOverlayHit(svg, 410, 410, {
        imageSize: [800, 600],
        boxes,
        displayBBoxById: { inner: [400, 400, 20, 20] },
      }),
    ).toBe("inner");
  });

  it("normalizes string/number ids while preserving render-state overrides", () => {
    expect(
      getBBoxOverlayRenderState(
        { id: 2, bbox: [0, 0, 10, 10], status: "validated" },
        "2",
        null,
        { 2: { listHovered: true } },
      ),
    ).toEqual(
      expect.objectContaining({
        focused: true,
        imageHovered: false,
        listHovered: true,
        submitted: true,
      }),
    );
  });
});
