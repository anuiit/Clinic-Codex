import { act, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  useImageBBoxStageSize,
  type UseImageBBoxStageSizeResult,
} from "./useImageBBoxStageSize";

const defaultRect = {
  x: 0,
  y: 0,
  left: 0,
  top: 0,
  right: 0,
  bottom: 0,
  width: 0,
  height: 0,
  toJSON: () => ({}),
};

let currentRect = { ...defaultRect };
let resizeObserverCallback: ResizeObserverCallback | null = null;

class MockResizeObserver implements ResizeObserver {
  readonly observe = vi.fn();
  readonly unobserve = vi.fn();
  readonly disconnect = vi.fn();

  constructor(callback: ResizeObserverCallback) {
    resizeObserverCallback = callback;
  }
}

type ProbeProps = {
  imageSize?: [number, number] | null;
  loading?: boolean;
  disabled?: boolean;
  onValue: (value: UseImageBBoxStageSizeResult) => void;
};

function Probe({
  imageSize = [1000, 500],
  loading = false,
  disabled = false,
  onValue,
}: ProbeProps) {
  const stageSize = useImageBBoxStageSize({ imageSize, loading, disabled });
  onValue(stageSize);
  const attachStage = (node: HTMLDivElement | null) => {
    stageSize.containerRef(node);
  };
  const updateMeasuredStage = () => {
    stageSize.updateStageSize();
  };

  return (
    <div
      ref={attachStage}
      data-testid="stage"
      style={{
        borderLeftWidth: "0px",
        borderRightWidth: "0px",
        borderTopWidth: "0px",
        borderBottomWidth: "0px",
        padding: "0px",
      }}
    >
      <button type="button" onClick={updateMeasuredStage}>
        update
      </button>
    </div>
  );
}

function setMeasuredRect(width: number, height: number) {
  currentRect = {
    ...defaultRect,
    right: width,
    bottom: height,
    width,
    height,
  };
}

function latest(values: UseImageBBoxStageSizeResult[]) {
  const value = values.at(-1);
  if (!value) throw new Error("No hook value captured");
  return value;
}

describe("useImageBBoxStageSize", () => {
  beforeEach(() => {
    setMeasuredRect(0, 0);
    resizeObserverCallback = null;
    vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(
      () => currentRect,
    );
    vi.stubGlobal("ResizeObserver", MockResizeObserver);
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("contains a wide image inside the measured content box and centers vertically", async () => {
    setMeasuredRect(500, 500);
    const values: UseImageBBoxStageSizeResult[] = [];

    render(<Probe imageSize={[1000, 500]} onValue={(value) => values.push(value)} />);

    await waitFor(() => {
      expect(latest(values).transformSize).toEqual({ width: 500, height: 250 });
    });
    expect(latest(values).stageWidth).toBe(500);
    expect(latest(values).stageHeight).toBe(250);
    expect(latest(values).displayRect).toEqual({
      left: 0,
      top: 125,
      width: 500,
      height: 250,
      scale: 0.5,
    });
  });

  it("contains a tall image inside the measured content box and centers horizontally", async () => {
    setMeasuredRect(500, 500);
    const values: UseImageBBoxStageSizeResult[] = [];

    render(<Probe imageSize={[500, 1000]} onValue={(value) => values.push(value)} />);

    await waitFor(() => {
      expect(latest(values).transformSize).toEqual({ width: 250, height: 500 });
    });
    expect(latest(values).displayRect).toEqual({
      left: 125,
      top: 0,
      width: 250,
      height: 500,
      scale: 0.5,
    });
  });

  it("subtracts padding and border before sizing, then reports border-box display offsets", async () => {
    setMeasuredRect(600, 420);
    const values: UseImageBBoxStageSizeResult[] = [];

    render(<Probe imageSize={[1000, 500]} onValue={(value) => values.push(value)} />);
    const stage = screen.getByTestId("stage");
    stage.style.paddingLeft = "20px";
    stage.style.paddingRight = "30px";
    stage.style.paddingTop = "10px";
    stage.style.paddingBottom = "30px";
    stage.style.borderLeftWidth = "2px";
    stage.style.borderRightWidth = "4px";
    stage.style.borderTopWidth = "3px";
    stage.style.borderBottomWidth = "7px";

    act(() => latest(values).updateStageSize());

    await waitFor(() => {
      expect(latest(values).transformSize).toEqual({ width: 544, height: 272 });
    });
    expect(latest(values).displayRect).toEqual({
      left: 22,
      top: 62,
      width: 544,
      height: 272,
      scale: 0.544,
    });
  });

  it("caps scale at 1 so small images are not upscaled", async () => {
    setMeasuredRect(500, 500);
    const values: UseImageBBoxStageSizeResult[] = [];

    render(<Probe imageSize={[200, 100]} onValue={(value) => values.push(value)} />);

    await waitFor(() => {
      expect(latest(values).transformSize).toEqual({ width: 200, height: 100 });
    });
    expect(latest(values).displayRect).toEqual({
      left: 150,
      top: 200,
      width: 200,
      height: 100,
      scale: 1,
    });
  });

  it("returns null transform state without an available image or while disabled/loading", async () => {
    const values: UseImageBBoxStageSizeResult[] = [];
    const { rerender } = render(
      <Probe imageSize={null} onValue={(value) => values.push(value)} />,
    );

    expect(latest(values).transformSize).toBeNull();
    expect(latest(values).displayRect).toBeNull();
    expect(latest(values).stageWidth).toBe(0);
    expect(latest(values).stageHeight).toBe(0);

    rerender(
      <Probe imageSize={[100, 50]} loading onValue={(value) => values.push(value)} />,
    );
    await waitFor(() => expect(latest(values).transformSize).toBeNull());

    rerender(
      <Probe imageSize={[100, 50]} disabled onValue={(value) => values.push(value)} />,
    );
    await waitFor(() => expect(latest(values).transformSize).toBeNull());
  });

  it("uses natural size before measurement and updates on ResizeObserver/window resize", async () => {
    const values: UseImageBBoxStageSizeResult[] = [];

    render(<Probe imageSize={[1000, 500]} onValue={(value) => values.push(value)} />);

    expect(latest(values).transformSize).toEqual({ width: 1000, height: 500 });
    expect(latest(values).displayRect).toEqual({
      left: 0,
      top: 0,
      width: 1000,
      height: 500,
      scale: 1,
    });

    setMeasuredRect(500, 500);
    act(() => {
      resizeObserverCallback?.([], {} as ResizeObserver);
    });

    await waitFor(() => {
      expect(latest(values).transformSize).toEqual({ width: 500, height: 250 });
    });

    setMeasuredRect(300, 300);
    act(() => window.dispatchEvent(new Event("resize")));

    await waitFor(() => {
      expect(latest(values).transformSize).toEqual({ width: 300, height: 150 });
    });
  });
});
