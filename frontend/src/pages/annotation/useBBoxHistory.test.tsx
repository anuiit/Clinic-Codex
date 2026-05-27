import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { describe, expect, it } from "vitest";
import type { AnnotationStatus, DetectedElement } from "../../types";
import { useBBoxHistory } from "./useBBoxHistory";

const FIRST: DetectedElement = {
  bbox: [100, 100, 50, 40],
  class_name: "atl",
  class_label: 1,
  confidence: 0.9,
  rejected: false,
  top_k: [],
};
const SECOND: DetectedElement = {
  bbox: [220, 180, 60, 50],
  class_name: "bet",
  class_label: 2,
  confidence: 0.75,
  rejected: false,
  top_k: [],
};

function HistoryProbe() {
  const [elements, setElements] = useState<DetectedElement[]>([FIRST, SECOND]);
  const [statuses, setStatuses] = useState<Record<number, AnnotationStatus>>({
    0: "validated",
    1: "draft",
  });
  const [focusedIdx, setFocusedIdx] = useState<number | null>(0);
  const history = useBBoxHistory({
    setElements,
    setAnnotationStatus: setStatuses,
    setFocusedIdx,
  });

  return (
    <div>
      <output data-testid="elements">{elements.map((element) => element.class_name).join(",")}</output>
      <output data-testid="statuses">{JSON.stringify(statuses)}</output>
      <output data-testid="focus">{focusedIdx ?? "none"}</output>
      <output data-testid="history-count">{history.bboxHistory.length}</output>
      <button
        type="button"
        onClick={() => {
          history.pushBboxHistory({
            type: "delete",
            idx: 0,
            element: FIRST,
            status: statuses[0],
            focusedIdx,
          });
          setElements([SECOND]);
          setStatuses({ 0: statuses[1] });
          setFocusedIdx(null);
        }}
      >
        delete-first
      </button>
      <button type="button" onClick={() => history.undoLastBboxChange()}>
        undo
      </button>
    </div>
  );
}

describe("useBBoxHistory", () => {
  it("restores a deleted bbox and its shifted annotation statuses", () => {
    render(<HistoryProbe />);

    fireEvent.click(screen.getByRole("button", { name: "delete-first" }));
    expect(screen.getByTestId("elements")).toHaveTextContent("bet");
    expect(screen.getByTestId("statuses")).toHaveTextContent('{"0":"draft"}');
    expect(screen.getByTestId("history-count")).toHaveTextContent("1");

    fireEvent.click(screen.getByRole("button", { name: "undo" }));
    expect(screen.getByTestId("elements")).toHaveTextContent("atl,bet");
    expect(screen.getByTestId("statuses")).toHaveTextContent('{"0":"validated","1":"draft"}');
    expect(screen.getByTestId("focus")).toHaveTextContent("0");
    expect(screen.getByTestId("history-count")).toHaveTextContent("0");
  });
});
