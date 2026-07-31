import { readFileSync } from "node:fs";
import type { PointerEvent as ReactPointerEvent } from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ImageBBoxStage, ImageBBoxToolbar } from "./index";

function readSource(filePath: string) {
  return readFileSync(filePath, "utf8");
}

function sourceImports(filePath: string) {
  const source = readSource(filePath);
  return Array.from(
    source.matchAll(
      /import(?:\s+type)?(?:[\s\S]*?)from\s+["']([^"']+)["']|import\s+["']([^"']+)["']/g,
    ),
    (match) => match[1] ?? match[2],
  );
}

describe("ImageBBoxStage shared image/bbox boundary", () => {
  it("renders image and SVG overlay as siblings in the same transformed wrapper", () => {
    render(
      <ImageBBoxStage
        imageDataUrl="data:image/png;base64,abc"
        imageName="Fixture image"
        imageSize={[400, 300]}
        boxes={[{ id: 1, bbox: [10, 20, 80, 60], label: "pump" }]}
        selectedId={1}
        showLabelNames
        mode="inspect"
        viewport={{
          zoom: 1.5,
          panOffset: { x: 12, y: -8 },
          isPanning: false,
        }}
        toolbar={
          <ImageBBoxToolbar label="Stage toolbar">
            <button type="button">Toggle labels</button>
          </ImageBBoxToolbar>
        }
        badges={<span>Focused</span>}
        controls={<button type="button">Custom control</button>}
        testIds={{
          root: "bbox-root",
          stage: "bbox-stage",
          transform: "bbox-transform",
          overlay: "bbox-overlay",
          controls: "bbox-controls",
        }}
      />,
    );

    const root = screen.getByTestId("bbox-root");
    const stage = screen.getByTestId("bbox-stage");
    const transform = screen.getByTestId("bbox-transform");
    const image = screen.getByRole("img", { name: "Fixture image" });
    const overlay = screen.getByTestId("bbox-overlay");

    expect(root).toHaveClass("image-bbox-stage", "main-image-panel");
    expect(stage).toHaveClass("image-bbox-stage__stage");
    expect(transform).toHaveClass("image-bbox-stage__transform");
    expect(transform).toHaveStyle({
      transform: "translate(12px, -8px) scale(1.5)",
      transformOrigin: "center center",
      transition: "transform 0.1s ease",
    });
    expect(transform.children[0]).toBe(image);
    expect(transform.children[1]).toBe(overlay);
    expect(image).toHaveClass("image-bbox-stage__image");
    expect(image).toHaveAttribute("draggable", "false");
    expect(overlay).toHaveAttribute("viewBox", "0 0 400 300");
    expect(overlay).toHaveAttribute("preserveAspectRatio", "none");
    expect(screen.getByTestId("image-bbox-stage-box-1")).toHaveAttribute(
      "stroke",
      "#fbbf24",
    );
    expect(overlay).toHaveTextContent("#1 pump");
    expect(screen.getByRole("toolbar", { name: "Stage toolbar" })).toBeInTheDocument();
    expect(screen.getByTestId("bbox-controls")).toHaveTextContent(
      "Custom control",
    );
  });

  it("keeps image-stage browser selection prevention scoped to shared stage surfaces", () => {
    const panelCss = readSource("src/components/MainImagePanel.module.css");
    const stageCss = readSource("src/components/ImageBBoxStage/ImageBBoxStage.module.css");
    const css = `${panelCss}\n${stageCss}`;

    expect(css).toContain(".main-image-panel__stage");
    expect(css).toContain(".main-image-panel__transform");
    expect(css).toContain(".image-bbox-stage__image");
    expect(css).toContain("user-select: none");
    expect(css).toContain("-webkit-user-drag: none");
    expect(css).not.toMatch(/body[^{]*{[^}]*user-select:\s*none/s);
    expect(css).not.toMatch(/#root[^{]*{[^}]*user-select:\s*none/s);
  });

  it("supports focused overlay mode, display bbox overrides, custom labels, and edit extras", () => {
    render(
      <ImageBBoxStage
        imageDataUrl="data:image/png;base64,abc"
        imageName="Edit image"
        imageSize={[200, 100]}
        boxes={[
          { id: "a", bbox: [0, 0, 20, 20], label: "draft", status: "draft" },
          {
            id: "b",
            bbox: [30, 10, 40, 30],
            label: "validated",
            status: "validated",
          },
        ]}
        selectedId="b"
        hoveredId="a"
        mode="edit"
        overlayMode="focused"
        viewport={{
          zoom: 2,
          panOffset: { x: 0, y: 0 },
          isPanning: true,
        }}
        transformSize={{ width: 640, height: 320 }}
        imageFit="fill"
        displayBBoxById={{ b: [32, 12, 44, 34] }}
        boxStateById={{ b: { listHovered: true } }}
        renderLabel={(box) => `custom-${box.id}`}
        renderBoxExtras={(_, __, bbox) => (
          <circle
            data-testid="resize-handle"
            cx={bbox[0]}
            cy={bbox[1]}
            r={5}
          />
        )}
        overlayChildren={<rect data-testid="draft-box" x={1} y={2} width={3} height={4} />}
        testIds={{ transform: "edit-transform", overlay: "edit-overlay" }}
      />,
    );

    const transform = screen.getByTestId("edit-transform");
    const overlay = screen.getByTestId("edit-overlay");

    expect(transform).toHaveStyle({
      width: "640px",
      height: "320px",
      transform: "translate(0px, 0px) scale(2)",
      transition: "none",
    });
    expect(screen.getByRole("img", { name: "Edit image" })).toHaveClass(
      "h-full",
      "w-full",
      "object-fill",
    );
    expect(overlay).toHaveStyle({ width: "100%", height: "100%" });
    expect(screen.queryByTestId("image-bbox-stage-box-a")).not.toBeInTheDocument();
    expect(screen.getByTestId("image-bbox-stage-box-b")).toHaveAttribute("x", "32");
    expect(screen.getByTestId("image-bbox-stage-box-b")).toHaveAttribute("width", "44");
    expect(overlay).toHaveTextContent("custom-b");
    expect(screen.getByTestId("resize-handle")).toHaveAttribute("cx", "32");
    expect(screen.getByTestId("draft-box")).toBeInTheDocument();
  });

  it("passes optional SVG and image props without owning the page interaction state machine", () => {
    const onPointerDown = vi.fn();

    render(
      <ImageBBoxStage
        imageDataUrl="data:image/png;base64,abc"
        imageName="Interactive image"
        imageSize={[100, 80]}
        boxes={[]}
        mode="inspect"
        viewport={{ zoom: 1, panOffset: { x: 0, y: 0 } }}
        imageProps={{ "data-testid": "stage-image", onLoad: vi.fn() }}
        svgProps={{ "aria-label": "bbox overlay", onPointerDown }}
      />,
    );

    expect(screen.getByTestId("stage-image")).toHaveAttribute(
      "src",
      "data:image/png;base64,abc",
    );
    expect(screen.getByLabelText("bbox overlay")).toBeInTheDocument();
  });

  it("passes header slots and placement props through to MainImagePanel", () => {
    render(
      <ImageBBoxStage
        imageDataUrl="data:image/png;base64,abc"
        imageName="Slotted image"
        imageSize={[100, 80]}
        boxes={[]}
        mode="inspect"
        title={<h1>Slotted image</h1>}
        headerMeta={<span>100×80 · Classes 2</span>}
        headerActions={<button type="button">Annoter l’analyse</button>}
        toolbar={<ImageBBoxToolbar label="Bottom toolbox">tools</ImageBBoxToolbar>}
        toolbarPlacement="bottom-center"
        panelControls={[
          {
            id: "zoom-in",
            icon: <span aria-hidden="true">+</span>,
            label: "Zoom in",
            onClick: vi.fn(),
          },
        ]}
        controlsPlacement="bottom-center"
        zoomLabel="100%"
        viewport={{ zoom: 1, panOffset: { x: 0, y: 0 } }}
        testIds={{
          header: "stage-header",
          toolbar: "stage-toolbar",
          controls: "stage-controls",
        }}
      />,
    );

    expect(screen.getByTestId("stage-header")).toHaveTextContent("100×80");
    expect(screen.getByRole("button", { name: "Annoter l’analyse" })).toBeInTheDocument();
    expect(screen.getByTestId("stage-toolbar")).toHaveClass(
      "main-image-panel__toolbar--bottom-center",
    );
    expect(screen.getByTestId("stage-controls")).toHaveClass(
      "main-image-panel__controls--bottom-center",
    );
  });

  it("supports basic select and hover callbacks for composition consumers", () => {
    const onSelectBox = vi.fn();
    const onHoverBox = vi.fn();

    render(
      <ImageBBoxStage
        imageDataUrl="data:image/png;base64,abc"
        imageName="Callback image"
        imageSize={[100, 100]}
        boxes={[{ id: 1, bbox: [10, 20, 80, 60], label: "hit" }]}
        mode="inspect"
        viewport={{ zoom: 1, panOffset: { x: 0, y: 0 } }}
        onSelectBox={onSelectBox}
        onHoverBox={onHoverBox}
        testIds={{ overlay: "callback-overlay" }}
      />,
    );

    const overlay = screen.getByTestId("callback-overlay");
    vi.spyOn(overlay, "getBoundingClientRect").mockReturnValue({
      x: 0,
      y: 0,
      left: 0,
      top: 0,
      right: 100,
      bottom: 100,
      width: 100,
      height: 100,
      toJSON: () => ({}),
    });

    fireEvent(
      overlay,
      new MouseEvent("pointermove", {
        bubbles: true,
        cancelable: true,
        clientX: 50,
        clientY: 50,
      }),
    );
    fireEvent(
      overlay,
      new MouseEvent("pointerdown", {
        bubbles: true,
        button: 0,
        cancelable: true,
        clientX: 50,
        clientY: 50,
      }),
    );

    expect(onHoverBox).toHaveBeenNthCalledWith(1, 1);
    expect(onSelectBox).toHaveBeenCalledWith(1);

    fireEvent.pointerLeave(overlay);
    expect(onHoverBox).toHaveBeenLastCalledWith(null);
  });

  it("does not render or fire overlay callbacks when overlayMode is hidden", () => {
    const onSelectBox = vi.fn();
    const onHoverBox = vi.fn();

    render(
      <ImageBBoxStage
        imageDataUrl="data:image/png;base64,abc"
        imageName="Hidden overlay image"
        imageSize={[100, 100]}
        boxes={[{ id: 1, bbox: [10, 20, 80, 60], label: "hidden" }]}
        mode="inspect"
        overlayMode="hidden"
        viewport={{ zoom: 1, panOffset: { x: 0, y: 0 } }}
        onSelectBox={onSelectBox}
        onHoverBox={onHoverBox}
        testIds={{ stage: "hidden-stage", overlay: "hidden-overlay" }}
      />,
    );

    expect(screen.queryByTestId("hidden-overlay")).not.toBeInTheDocument();
    expect(screen.queryByTestId("image-bbox-stage-box-1")).not.toBeInTheDocument();

    fireEvent.pointerDown(screen.getByTestId("hidden-stage"), {
      button: 0,
      clientX: 50,
      clientY: 50,
    });
    fireEvent.pointerMove(screen.getByTestId("hidden-stage"), {
      clientX: 50,
      clientY: 50,
    });

    expect(onSelectBox).not.toHaveBeenCalled();
    expect(onHoverBox).not.toHaveBeenCalled();
  });

  it("ignores right-click selection and honors consumer preventDefault on pointerdown", () => {
    const onSelectBox = vi.fn();
    const onPointerDown = vi.fn((event: ReactPointerEvent<SVGSVGElement>) => {
      event.preventDefault();
    });

    const { rerender } = render(
      <ImageBBoxStage
        imageDataUrl="data:image/png;base64,abc"
        imageName="Guarded image"
        imageSize={[100, 100]}
        boxes={[{ id: 1, bbox: [10, 20, 80, 60], label: "hit" }]}
        mode="inspect"
        viewport={{ zoom: 1, panOffset: { x: 0, y: 0 } }}
        onSelectBox={onSelectBox}
        testIds={{ overlay: "guarded-overlay" }}
      />,
    );

    const overlay = screen.getByTestId("guarded-overlay");
    vi.spyOn(overlay, "getBoundingClientRect").mockReturnValue({
      x: 0,
      y: 0,
      left: 0,
      top: 0,
      right: 100,
      bottom: 100,
      width: 100,
      height: 100,
      toJSON: () => ({}),
    });

    fireEvent(
      overlay,
      new MouseEvent("pointerdown", {
        bubbles: true,
        button: 2,
        cancelable: true,
        clientX: 50,
        clientY: 50,
      }),
    );
    expect(onSelectBox).not.toHaveBeenCalled();

    rerender(
      <ImageBBoxStage
        imageDataUrl="data:image/png;base64,abc"
        imageName="Guarded image"
        imageSize={[100, 100]}
        boxes={[{ id: 1, bbox: [10, 20, 80, 60], label: "hit" }]}
        mode="inspect"
        viewport={{ zoom: 1, panOffset: { x: 0, y: 0 } }}
        onSelectBox={onSelectBox}
        svgProps={{ onPointerDown }}
        testIds={{ overlay: "guarded-overlay" }}
      />,
    );

    const preventedOverlay = screen.getByTestId("guarded-overlay");
    vi.spyOn(preventedOverlay, "getBoundingClientRect").mockReturnValue({
      x: 0,
      y: 0,
      left: 0,
      top: 0,
      right: 100,
      bottom: 100,
      width: 100,
      height: 100,
      toJSON: () => ({}),
    });

    fireEvent(
      preventedOverlay,
      new MouseEvent("pointerdown", {
        bubbles: true,
        button: 0,
        cancelable: true,
        clientX: 50,
        clientY: 50,
      }),
    );

    expect(onPointerDown).toHaveBeenCalledTimes(1);
    expect(onSelectBox).not.toHaveBeenCalled();
  });

  it("keeps API, storage, router, and page-specific modules out of ImageBBoxStage", () => {
    const files = [
      "src/components/ImageBBoxStage/ImageBBoxStage.tsx",
      "src/components/ImageBBoxStage/ImageBBoxOverlay.tsx",
      "src/components/ImageBBoxStage/ImageBBoxToolbar.tsx",
      "src/components/ImageBBoxStage/imageBBoxStage.types.ts",
      "src/components/ImageBBoxStage/index.ts",
      "src/components/ImageBBoxStage/useImageBBoxStageSize.ts",
      "src/components/ImageBBoxStage/useImageBBoxStageViewport.ts",
      "src/components/ImageBBoxStage/createImageBBoxZoomControls.tsx",
    ];

    const imports = files.flatMap(sourceImports);

    expect(imports).not.toEqual(
      expect.arrayContaining([
        expect.stringMatching(/react-router/),
        expect.stringMatching(/services\/(api|storage)/),
        expect.stringMatching(/pages\/(workspace|annotation)/),
      ]),
    );
  });
});
