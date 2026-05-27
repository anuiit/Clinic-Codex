import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { appText } from "../i18n/text";
import { AnnotationAnalyzerToolbar } from "./AnnotationAnalyzerToolbar";

const labels = appText.annotation;

function renderToolbar({
  canUndo,
  drawMode = false,
  showLabelNames = false,
  withZoom = false,
}: {
  canUndo: boolean;
  drawMode?: boolean;
  showLabelNames?: boolean;
  withZoom?: boolean;
}) {
  const onToggleDrawMode = vi.fn();
  const onUndo = vi.fn();
  const onToggleLabelNames = vi.fn();
  const onZoomOut = vi.fn();
  const onFitToView = vi.fn();
  const onZoomIn = vi.fn();

  render(
    <AnnotationAnalyzerToolbar
      drawMode={drawMode}
      canUndo={canUndo}
      showLabelNames={showLabelNames}
      labels={labels}
      onToggleDrawMode={onToggleDrawMode}
      onUndo={onUndo}
      onToggleLabelNames={onToggleLabelNames}
      onZoomOut={withZoom ? onZoomOut : undefined}
      onFitToView={withZoom ? onFitToView : undefined}
      onZoomIn={withZoom ? onZoomIn : undefined}
      zoomLabel={withZoom ? "125%" : undefined}
    />,
  );

  return {
    onToggleDrawMode,
    onUndo,
    onToggleLabelNames,
    onZoomOut,
    onFitToView,
    onZoomIn,
  };
}

describe("AnnotationAnalyzerToolbar", () => {
  it("keeps undo disabled and non-firing until bbox history exists", () => {
    const { onToggleDrawMode, onUndo, onToggleLabelNames } = renderToolbar({
      canUndo: false,
    });

    const undoButton = screen.getByRole("button", { name: labels.undoBbox });
    expect(undoButton).toBeDisabled();
    expect(undoButton).toHaveAttribute("title", labels.undoBboxTitle);

    fireEvent.click(undoButton);
    fireEvent.click(screen.getByRole("button", { name: labels.drawBbox }));
    fireEvent.click(screen.getByRole("button", { name: labels.labelsToggle }));

    expect(onUndo).not.toHaveBeenCalled();
    expect(onToggleDrawMode).toHaveBeenCalledTimes(1);
    expect(onToggleLabelNames).toHaveBeenCalledTimes(1);
  });

  it("fires undo once when enabled and preserves draw/label accessibility state", () => {
    const { onUndo } = renderToolbar({
      canUndo: true,
      drawMode: true,
      showLabelNames: true,
    });

    const drawButton = screen.getByRole("button", { name: labels.drawBbox });
    const undoButton = screen.getByRole("button", { name: labels.undoBbox });
    const labelsButton = screen.getByRole("button", { name: labels.labelsToggle });

    expect(drawButton).toHaveAttribute("aria-pressed", "true");
    expect(labelsButton).toHaveAttribute("aria-pressed", "true");
    expect(undoButton).toBeEnabled();

    fireEvent.click(undoButton);

    expect(onUndo).toHaveBeenCalledTimes(1);
  });

  it("can consolidate edit controls, zoom actions, and zoom percentage in one dock", () => {
    const { onZoomOut, onFitToView, onZoomIn } = renderToolbar({
      canUndo: true,
      withZoom: true,
    });

    const toolbar = screen.getByTestId("annotation-analyzer-toolbar");
    expect(toolbar).toHaveTextContent("125%");
    expect(within(toolbar).getByRole("button", { name: labels.zoomOut })).toBeInTheDocument();
    expect(within(toolbar).getByRole("button", { name: labels.fitToView })).toBeInTheDocument();
    expect(within(toolbar).getByRole("button", { name: labels.zoomIn })).toBeInTheDocument();

    fireEvent.click(within(toolbar).getByRole("button", { name: labels.zoomOut }));
    fireEvent.click(within(toolbar).getByRole("button", { name: labels.fitToView }));
    fireEvent.click(within(toolbar).getByRole("button", { name: labels.zoomIn }));

    expect(onZoomOut).toHaveBeenCalledTimes(1);
    expect(onFitToView).toHaveBeenCalledTimes(1);
    expect(onZoomIn).toHaveBeenCalledTimes(1);
  });
});
