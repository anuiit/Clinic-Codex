import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useImageStageViewport } from "./index";
import { useImageBBoxStageViewport } from "./useImageBBoxStageViewport";

const defaultRect = {
  x: 0,
  y: 0,
  left: 0,
  top: 0,
  right: 800,
  bottom: 600,
  width: 800,
  height: 600,
  toJSON: () => ({}),
} as DOMRect;

function Harness() {
  const {
    beginPan,
    containerRef,
    endPan,
    handleStageWheel,
    movePan,
    panOffset,
    resetView,
    transformSize,
    zoom,
    zoomIn,
    zoomOut,
  } = useImageBBoxStageViewport({
    imageSize: [800, 600],
    resetKey: "fixture",
  });

  return (
    <div
      ref={containerRef}
      data-testid="stage"
      onWheel={handleStageWheel}
      onPointerDown={beginPan}
      onPointerMove={movePan}
      onPointerUp={endPan}
    >
      <div
        data-testid="transform"
        style={{
          width: `${transformSize?.width ?? 0}px`,
          height: `${transformSize?.height ?? 0}px`,
          transform: `translate(${panOffset.x}px, ${panOffset.y}px) scale(${zoom})`,
        }}
      />
      <button type="button" onClick={zoomIn}>
        zoom in
      </button>
      <button type="button" onClick={zoomOut}>
        zoom out
      </button>
      <button type="button" onClick={resetView}>
        fit
      </button>
    </div>
  );
}

function PublicAliasHarness({ resetKey }: { resetKey: string }) {
  const {
    beginPan,
    containerRef,
    endPan,
    handleStageWheel,
    movePan,
    panOffset,
    resetView,
    transformSize,
    zoom,
    zoomIn,
    zoomOut,
  } = useImageStageViewport({
    imageSize: [800, 600],
    resetKey,
  });

  return (
    <div
      ref={containerRef}
      data-testid="alias-stage"
      onWheel={handleStageWheel}
      onPointerDown={beginPan}
      onPointerMove={movePan}
      onPointerUp={endPan}
    >
      <div
        data-testid="alias-transform"
        style={{
          width: `${transformSize?.width ?? 0}px`,
          height: `${transformSize?.height ?? 0}px`,
          transform: `translate(${panOffset.x}px, ${panOffset.y}px) scale(${zoom})`,
        }}
      />
      <button type="button" onClick={zoomIn}>
        alias zoom in
      </button>
      <button type="button" onClick={zoomOut}>
        alias zoom out
      </button>
      <button type="button" onClick={resetView}>
        alias fit
      </button>
    </div>
  );
}

function pointer(
  target: Element,
  type: "pointerdown" | "pointermove" | "pointerup",
  init: { clientX: number; clientY: number; pointerId?: number; buttons?: number },
) {
  const event = new MouseEvent(type, {
    bubbles: true,
    cancelable: true,
    clientX: init.clientX,
    clientY: init.clientY,
  });
  Object.defineProperty(event, "button", { value: 0 });
  Object.defineProperty(event, "buttons", { value: init.buttons ?? 0 });
  Object.defineProperty(event, "pointerId", { value: init.pointerId ?? 1 });
  fireEvent(target, event);
}

describe("useImageBBoxStageViewport", () => {
  beforeEach(() => {
    Element.prototype.getBoundingClientRect = vi.fn(() => defaultRect);
    HTMLElement.prototype.getBoundingClientRect = vi.fn(() => defaultRect);
  });

  it("shares wheel zoom, button zoom, fit-to-view, and pan behavior", async () => {
    render(<Harness />);
    const stage = screen.getByTestId("stage");
    const transform = screen.getByTestId("transform");
    stage.setPointerCapture = vi.fn();
    stage.releasePointerCapture = vi.fn();
    stage.hasPointerCapture = vi.fn(() => true);

    await waitFor(() => {
      expect(transform).toHaveStyle({ width: "800px", height: "600px" });
    });

    await act(async () => {
      fireEvent.wheel(stage, { deltaY: -100 });
    });
    expect(transform).toHaveStyle({
      transform: "translate(0px, 0px) scale(1.15)",
    });

    await act(async () => {
      pointer(stage, "pointerdown", { clientX: 100, clientY: 100, buttons: 1 });
      pointer(stage, "pointermove", { clientX: 140, clientY: 130, buttons: 1 });
      pointer(stage, "pointerup", { clientX: 140, clientY: 130 });
    });
    expect(transform).toHaveStyle({
      transform: "translate(40px, 30px) scale(1.15)",
    });

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "fit" }));
    });
    expect(transform).toHaveStyle({
      transform: "translate(0px, 0px) scale(1)",
    });

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "zoom in" }));
    });
    expect(transform).toHaveStyle({
      transform: "translate(0px, 0px) scale(1.25)",
    });
  });

  it("exposes the public useImageStageViewport alias and resets when resetKey changes", async () => {
    const { rerender } = render(<PublicAliasHarness resetKey="record-a" />);
    const transform = screen.getByTestId("alias-transform");

    await waitFor(() => {
      expect(transform).toHaveStyle({ width: "800px", height: "600px" });
    });

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "alias zoom in" }));
    });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "alias zoom in" }));
    });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "alias zoom out" }));
    });
    expect(transform).toHaveStyle({
      transform: "translate(0px, 0px) scale(1.25)",
    });

    rerender(<PublicAliasHarness resetKey="record-b" />);

    await waitFor(() => {
      expect(transform).toHaveStyle({
        transform: "translate(0px, 0px) scale(1)",
      });
    });
  });

  it("clamps pan inside a constrained container after zoom", async () => {
    const constrainedRect = {
      ...defaultRect,
      right: 200,
      bottom: 150,
      width: 200,
      height: 150,
    } as DOMRect;
    Element.prototype.getBoundingClientRect = vi.fn(() => constrainedRect);
    HTMLElement.prototype.getBoundingClientRect = vi.fn(() => constrainedRect);

    render(<PublicAliasHarness resetKey="constrained" />);
    const stage = screen.getByTestId("alias-stage");
    const transform = screen.getByTestId("alias-transform");
    stage.setPointerCapture = vi.fn();
    stage.releasePointerCapture = vi.fn();
    stage.hasPointerCapture = vi.fn(() => true);

    await waitFor(() => {
      expect(transform).toHaveStyle({ width: "200px", height: "150px" });
    });

    for (let i = 0; i < 4; i += 1) {
      await act(async () => {
        fireEvent.click(screen.getByRole("button", { name: "alias zoom in" }));
      });
    }
    await waitFor(() => {
      expect(transform).toHaveStyle({ transform: "translate(0px, 0px) scale(2)" });
    });

    await act(async () => {
      pointer(stage, "pointerdown", { clientX: 0, clientY: 0, buttons: 1 });
      pointer(stage, "pointermove", { clientX: 1000, clientY: 1000, buttons: 1 });
      pointer(stage, "pointerup", { clientX: 1000, clientY: 1000 });
    });

    expect(transform).toHaveStyle({
      transform: "translate(220px, 165px) scale(2)",
    });
  });
});
