import { readFileSync } from "node:fs";
import { render, screen } from "@testing-library/react";
import { createRef } from "react";
import { describe, expect, it, vi } from "vitest";
import MainImagePanel from "./MainImagePanel";

function sourceImports(filePath: string) {
  const source = readFileSync(filePath, "utf8");
  return Array.from(
    source.matchAll(/import(?:\s+type)?(?:[\s\S]*?)from\s+["']([^"']+)["']|import\s+["']([^"']+)["']/g),
    (match) => match[1] ?? match[2],
  );
}

describe("MainImagePanel shared boundary", () => {
  it("renders the shared viewport contract without owning page business behavior", () => {
    const stageRef = createRef<HTMLDivElement>();
    const transformRef = createRef<HTMLDivElement>();
    const onZoomIn = vi.fn();

    render(
      <MainImagePanel
        tone="annotation"
        title={<h1>Shared image</h1>}
        eyebrow="Clinic Codex"
        badges={<span>validated</span>}
        headerActions={<button type="button">Save</button>}
        toolbar={<button type="button">Draw</button>}
        image={<img alt="fixture" src="data:image/png;base64,abc" />}
        overlay={<svg aria-label="overlay" />}
        controls={[
          {
            id: "zoom-in",
            icon: <span aria-hidden="true">+</span>,
            label: "Zoom in",
            onClick: onZoomIn,
          },
        ]}
        zoomLabel="125%"
        footer={<span>footer</span>}
        className="custom-root"
        stageClassName="custom-stage"
        transformClassName="custom-transform"
        transformStyle={{ transform: "translate(4px, 5px) scale(1.25)" }}
        stageProps={{ "data-testid": "stage-from-props", role: "region" }}
        transformProps={{ "data-testid": "transform-from-props" }}
        stageRef={stageRef}
        transformRef={transformRef}
        testIds={{ root: "main-panel", stage: "shared-stage", transform: "shared-transform", controls: "shared-controls" }}
      />,
    );

    const root = screen.getByTestId("main-panel");
    const stage = screen.getByTestId("shared-stage");
    const transform = screen.getByTestId("shared-transform");

    expect(root).toHaveClass("main-image-panel", "main-image-panel--annotation", "custom-root", "overflow-hidden");
    expect(stage).toHaveClass(
      "main-image-panel__stage",
      "image-stage-frame",
      "image-stage-grid",
      "image-stage-scrollbar",
      "overflow-hidden",
      "custom-stage",
    );
    expect(stage).toHaveAttribute("role", "region");
    expect(transform).toHaveClass("main-image-panel__transform", "custom-transform");
    expect(transform).toHaveStyle({ transform: "translate(4px, 5px) scale(1.25)" });
    expect(stageRef.current).toBe(stage);
    expect(transformRef.current).toBe(transform);
    expect(screen.getByRole("button", { name: "Zoom in" })).toBeEnabled();
    expect(screen.getByTestId("shared-controls")).toHaveTextContent("125%");
  });

  it("keeps storage, API, router, and bbox-editing logic out of the shared component", () => {
    const imports = sourceImports("src/components/MainImagePanel.tsx");

    expect(imports).toEqual(["react"]);
    expect(imports).not.toEqual(
      expect.arrayContaining([
        expect.stringMatching(/react-router-dom/),
        expect.stringMatching(/services\/(api|storage)/),
        expect.stringMatching(/utils\/(imageCoords|segmentationBoxes|fuzzyClasses)/),
      ]),
    );
  });

  it("is the shared component imported by both image-workflow pages", () => {
    const workspaceImports = sourceImports("src/pages/WorkspacePage.tsx");
    const annotationImports = sourceImports("src/pages/AnnotationPage.tsx");

    expect(workspaceImports).toContain("../components/MainImagePanel");
    expect(annotationImports).toContain("../components/MainImagePanel");
  });
});
