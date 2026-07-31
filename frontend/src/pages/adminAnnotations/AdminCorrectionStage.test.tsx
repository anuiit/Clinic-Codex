import { useState } from "react";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { AdminAnnotationAnalysis } from "../../types";
import {
  AdminCorrectionStage,
  type AdminCorrectionBBox,
} from "./AdminCorrectionStage";

const analysisFixture: AdminAnnotationAnalysis = {
  analysis_id: "analysis-001",
  image_path: "images/analysis-001.png",
  image_url: "images/analysis-001.png",
  image_exists: true,
  elements: [
    {
      key: "element-a",
      analysis_id: "analysis-001",
      index: 1,
      class_name: "catheter",
      bbox: [40, 50, 80, 60],
      crop_path: "crops/a.png",
      crop_url: "crops/a.png",
      crop_exists: true,
      review_status: "approved",
      trainable: true,
      dataset_split: "train",
      split_reason: "",
      source_fingerprint: "fingerprint-a",
      stale_decision: false,
    },
    {
      key: "element-b",
      analysis_id: "analysis-001",
      index: 2,
      class_name: "valve",
      bbox: [160, 110, 70, 50],
      crop_path: "crops/b.png",
      crop_url: "crops/b.png",
      crop_exists: true,
      review_status: "pending",
      trainable: false,
      dataset_split: "excluded",
      split_reason: "",
      source_fingerprint: "fingerprint-b",
      stale_decision: false,
    },
  ],
};

function Harness({ onChange = vi.fn() }: { onChange?: (bbox: AdminCorrectionBBox) => void }) {
  const [bbox, setBbox] = useState<AdminCorrectionBBox>([60, 70, 90, 80]);
  return (
    <AdminCorrectionStage
      analysis={analysisFixture}
      selectedElement={analysisFixture.elements[0]}
      bbox={bbox}
      onBboxChange={(next) => {
        onChange(next);
        setBbox(next);
      }}
      mutating={false}
    />
  );
}

function setNaturalSize(image: HTMLImageElement, width: number, height: number) {
  Object.defineProperty(image, "naturalWidth", { configurable: true, value: width });
  Object.defineProperty(image, "naturalHeight", { configurable: true, value: height });
}

describe("AdminCorrectionStage", () => {
  beforeEach(() => {
    Element.prototype.getBoundingClientRect = vi.fn(() => ({
      x: 0,
      y: 0,
      left: 0,
      top: 0,
      right: 400,
      bottom: 300,
      width: 400,
      height: 300,
      toJSON: () => ({}),
    }) as DOMRect);
    Object.defineProperty(SVGElement.prototype, "setPointerCapture", {
      configurable: true,
      value: vi.fn(),
    });
    Object.defineProperty(SVGElement.prototype, "releasePointerCapture", {
      configurable: true,
      value: vi.fn(),
    });
    Object.defineProperty(SVGElement.prototype, "hasPointerCapture", {
      configurable: true,
      value: vi.fn(() => true),
    });
  });

  it("uses the shared image shell and keeps every element visible in context", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    const preloader = screen.getByRole("img", {
      name: "Image à corriger analysis-001",
    }) as HTMLImageElement;
    setNaturalSize(preloader, 400, 300);
    await act(async () => fireEvent.load(preloader));

    const stage = await screen.findByTestId("admin-segmentation-stage");
    expect(stage).toHaveAttribute("data-natural-width", "400");
    expect(stage).toHaveAttribute("data-natural-height", "300");
    expect(screen.getAllByText("#1 catheter").length).toBeGreaterThan(0);
    expect(screen.getAllByText("#2 valve").length).toBeGreaterThan(0);

    const selected = screen.getByTestId("image-bbox-stage-box-element-a");
    expect(selected).toHaveAttribute("x", "60");
    expect(selected).toHaveAttribute("y", "70");
    expect(screen.getByTestId("admin-correction-original-bbox")).toHaveAttribute(
      "stroke-dasharray",
      "6 4",
    );

    await user.click(screen.getByRole("button", { name: "Zoomer" }));
    expect(screen.getAllByText("125 %").length).toBeGreaterThan(0);
    expect(screen.getByRole("button", { name: "Déplacer l’image" })).toBeEnabled();
  });

  it("maps a pointer drag back to natural image coordinates", async () => {
    const onChange = vi.fn();
    render(<Harness onChange={onChange} />);
    const preloader = screen.getByRole("img", {
      name: "Image à corriger analysis-001",
    }) as HTMLImageElement;
    setNaturalSize(preloader, 400, 300);
    await act(async () => fireEvent.load(preloader));

    const overlay = await screen.findByLabelText(
      "Redessiner la segmentation de l'élément 1",
    );
    const dispatchPointer = (type: string, clientX: number, clientY: number) => {
      const event = new Event(type, { bubbles: true, cancelable: true });
      Object.defineProperties(event, {
        clientX: { value: clientX },
        clientY: { value: clientY },
        pointerId: { value: 1 },
        button: { value: 0 },
      });
      fireEvent(overlay, event);
    };
    await act(async () => {
      dispatchPointer("pointerdown", 80, 90);
      dispatchPointer("pointermove", 110, 120);
      dispatchPointer("pointerup", 110, 120);
    });

    await waitFor(() => expect(onChange).toHaveBeenCalledWith([90, 100, 90, 80]));
  });
});
