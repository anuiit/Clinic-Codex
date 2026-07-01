import { readFileSync } from "node:fs";
import { render, screen } from "@testing-library/react";
import { createRef } from "react";
import { describe, expect, it, vi } from "vitest";
import MainImagePanel, {
  AnalyzerToolbar,
  AnalyzerToolbarButton,
} from "./MainImagePanel";

function readSource(filePath: string) {
  return readFileSync(filePath, "utf8");
}

function readJson<T>(filePath: string): T {
  return JSON.parse(readSource(filePath)) as T;
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
        toolbar={
          <AnalyzerToolbar aria-label="Analyzer tools">
            <AnalyzerToolbarButton active>Draw</AnalyzerToolbarButton>
          </AnalyzerToolbar>
        }
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
        testIds={{
          root: "main-panel",
          stage: "shared-stage",
          transform: "shared-transform",
          controls: "shared-controls",
        }}
      />,
    );

    const root = screen.getByTestId("main-panel");
    const stage = screen.getByTestId("shared-stage");
    const transform = screen.getByTestId("shared-transform");

    expect(root).toHaveClass(
      "main-image-panel",
      "main-image-panel--annotation",
      "custom-root",
      "overflow-hidden",
    );
    expect(stage).toHaveClass(
      "main-image-panel__stage",
      "image-stage-frame",
      "image-stage-grid",
      "image-stage-scrollbar",
      "overflow-hidden",
      "custom-stage",
    );
    expect(stage).toHaveAttribute("role", "region");
    expect(transform).toHaveClass(
      "main-image-panel__transform",
      "custom-transform",
    );
    expect(transform).toHaveStyle({
      transform: "translate(4px, 5px) scale(1.25)",
    });
    expect(stageRef.current).toBe(stage);
    expect(transformRef.current).toBe(transform);
    expect(stage.querySelector(".main-image-panel__toolbar")).toContainElement(
      screen.getByLabelText("Analyzer tools"),
    );
    expect(screen.getByRole("button", { name: "Draw" })).toHaveClass(
      "analyzer-toolbar__button",
      "analyzer-toolbar__button--active",
    );
    expect(screen.getByRole("button", { name: "Zoom in" })).toBeEnabled();
    expect(screen.getByTestId("shared-controls")).toHaveTextContent("125%");
    expect(
      screen
        .getByRole("button", { name: "Draw" })
        .closest(".main-image-panel__toolbar"),
    ).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "fixture" }).parentElement).toBe(
      screen.getByLabelText("overlay").parentElement,
    );
  });

  it("keeps image and overlay as siblings inside the shared transform wrapper", () => {
    render(
      <MainImagePanel
        image={<img alt="fixture" src="data:image/png;base64,abc" />}
        overlay={<svg aria-label="overlay" />}
        testIds={{ transform: "shared-transform" }}
      />,
    );

    const transform = screen.getByTestId("shared-transform");

    expect(transform.children).toHaveLength(2);
    expect(transform.children[0]).toBe(
      screen.getByRole("img", { name: "fixture" }),
    );
    expect(transform.children[1]).toBe(screen.getByLabelText("overlay"));
  });

  it("renders header metadata/actions and opt-in bottom-center dock placements", () => {
    render(
      <MainImagePanel
        title={<h1>alpha.png</h1>}
        headerMeta={<span>800×600 · Classes aleph 1</span>}
        headerActions={<button type="button">Annoter l’analyse</button>}
        toolbar={<AnalyzerToolbar aria-label="Combined toolbox">tools</AnalyzerToolbar>}
        toolbarPlacement="bottom-center"
        image={<img alt="fixture" src="data:image/png;base64,abc" />}
        controls={[
          {
            id: "fit",
            icon: <span aria-hidden="true">fit</span>,
            label: "Fit",
            onClick: vi.fn(),
          },
        ]}
        controlsPlacement="bottom-center"
        zoomLabel="100%"
        testIds={{
          header: "panel-header",
          toolbar: "panel-toolbar",
          controls: "panel-controls",
        }}
      />,
    );

    expect(screen.getByTestId("panel-header")).toHaveTextContent("800×600");
    expect(screen.getByRole("button", { name: "Annoter l’analyse" })).toBeInTheDocument();
    expect(screen.getByTestId("panel-toolbar")).toHaveClass(
      "main-image-panel__toolbar--bottom-center",
      "left-1/2",
    );
    expect(screen.getByTestId("panel-controls")).toHaveClass(
      "main-image-panel__controls--bottom-center",
      "left-1/2",
    );
  });

  it("documents the neutral analyzer chrome CSS contract", () => {
    const source = readFileSync("src/styles/app-theme.css", "utf8");

    expect(source).not.toMatch(
      new RegExp(
        String.raw`\.main-image-panel__stage\s*{[^}]*margin\s*:\s*1rem`,
        "i",
      ),
    );
    expect(source).not.toMatch(
      new RegExp(
        String.raw`\.main-image-panel--annotation\s*{[^}]*(border(?:-color)?\s*:|rgba\(245,\s*158,\s*11|amber)`,
        "i",
      ),
    );
    expect(source).not.toMatch(
      new RegExp(
        String.raw`\.annotation-stage-frame\s*,\s*\.image-stage-frame\s*{|\.image-stage-frame\s*,\s*\.annotation-stage-frame\s*{`,
      ),
    );
  });

  it("keeps storage, API, router, and bbox-editing logic out of the shared component", () => {
    const imports = sourceImports("src/components/MainImagePanel.tsx");

    expect(imports).toEqual(["react", "./MainImagePanel.module.css"]);
    expect(imports).not.toEqual(
      expect.arrayContaining([
        expect.stringMatching(/react-router-dom/),
        expect.stringMatching(/services\/(api|storage)/),
        expect.stringMatching(
          /utils\/(imageCoords|segmentationBoxes|fuzzyClasses)/,
        ),
      ]),
    );
  });

  it("is the shared component imported by both image-workflow pages", () => {
    const workspaceImports = sourceImports("src/pages/WorkspacePage.tsx");
    const annotationImports = sourceImports("src/pages/AnnotationPage.tsx");

    expect(workspaceImports).toContain("../components/ImageBBoxStage");
    expect(annotationImports).toContain("../components/ImageBBoxStage");
  });

  it("keeps the shared component prop surface presentation-only", () => {
    const source = readSource("src/components/MainImagePanel.tsx");

    expect(source).not.toMatch(
      /\b(record|analysis|bbox|save|submit|route|navigate|annotationStatus|api|storage)\b/,
    );
    expect(source).toMatch(/toolbar\??: ReactNode/);
    expect(source).toMatch(/image: ReactNode/);
    expect(source).toMatch(/overlay\??: ReactNode/);
  });

  it("guards the no-new-dependencies contract for the refactor", () => {
    const packageJson = readJson<{ dependencies: Record<string, string> }>(
      "package.json",
    );

    expect(Object.keys(packageJson.dependencies).sort()).toEqual([
      "axios",
      "lucide-react",
      "react",
      "react-dom",
      "react-router-dom",
    ]);
  });
});
