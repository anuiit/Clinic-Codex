import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ClassNameCombobox } from "./ClassNameCombobox";

const labels = {
  suggestions: "Suggestions",
  noSuggestion: "No suggestions",
  createElementName: "Create",
  renameElement: "Rename",
  nameElement: "Class name",
  elementNamePlaceholder: "Type a class",
};

describe("ClassNameCombobox", () => {
  it("renders suggestions in a portal and commits the highlighted suggestion with keyboard", async () => {
    const user = userEvent.setup();
    const onCommit = vi.fn();

    render(
      <ClassNameCombobox
        value="unknown"
        classNames={["atlas", "beta"]}
        customClassNames={["custom"]}
        topK={[{ class_name: "aleph", confidence: 0.8 }]}
        autoFocusToken={1}
        labels={labels}
        index={3}
        onCommit={onCommit}
      />,
    );

    const input = screen.getByRole("combobox", { name: "Class name 3" });
    const suggestionList = await screen.findByRole("listbox", { name: "Suggestions" });

    expect(document.body).toContainElement(suggestionList);
    expect(suggestionList).toHaveClass("ui-class-name-menu");
    expect(input).toHaveAttribute("aria-expanded", "true");
    expect(input).toHaveAttribute("aria-controls", suggestionList.id);
    expect(within(suggestionList).getByRole("option", { name: /aleph/i })).toHaveAttribute(
      "aria-selected",
      "true",
    );

    await user.keyboard("{Enter}");

    expect(onCommit).toHaveBeenCalledWith("aleph");
  });

  it("offers a create action for new normalized names and commits them", async () => {
    const user = userEvent.setup();
    const onCommit = vi.fn();

    render(
      <ClassNameCombobox
        value=""
        classNames={["atlas"]}
        customClassNames={[]}
        topK={[]}
        autoFocusToken={0}
        labels={labels}
        index={4}
        onCommit={onCommit}
      />,
    );

    const input = screen.getByRole("combobox", { name: "Class name 4" });
    await user.click(input);
    await user.type(input, "  new   label  ");

    const suggestionList = await screen.findByRole("listbox", { name: "Suggestions" });
    const createOption = within(suggestionList).getByRole("option", { name: /create/i });

    expect(createOption).toHaveTextContent("new label");

    await user.click(createOption);

    expect(onCommit).toHaveBeenCalledWith("new label");
  });

  it("suppresses create when the normalized value already exists", async () => {
    const user = userEvent.setup();

    render(
      <ClassNameCombobox
        value=""
        classNames={["Atlas"]}
        customClassNames={[]}
        topK={[]}
        autoFocusToken={0}
        labels={labels}
        index={5}
        onCommit={vi.fn()}
      />,
    );

    const input = screen.getByRole("combobox", { name: "Class name 5" });
    await user.click(input);
    await user.type(input, " atlas ");

    const suggestionList = await screen.findByRole("listbox", { name: "Suggestions" });
    expect(within(suggestionList).queryByRole("option", { name: /create/i })).not.toBeInTheDocument();
  });

  it("reaches the create action with the keyboard when fuzzy suggestions exist", async () => {
    const user = userEvent.setup();
    const onCommit = vi.fn();

    render(
      <ClassNameCombobox
        value=""
        classNames={["atlas"]}
        customClassNames={[]}
        topK={[]}
        autoFocusToken={0}
        labels={labels}
        index={6}
        onCommit={onCommit}
      />,
    );

    const input = screen.getByRole("combobox", { name: "Class name 6" });
    await user.click(input);
    await user.type(input, "atla");

    const suggestionList = await screen.findByRole("listbox", { name: "Suggestions" });
    const createOption = within(suggestionList).getByRole("option", { name: /create/i });
    await user.keyboard("{ArrowDown}");

    expect(createOption).toHaveAttribute("aria-selected", "true");
    expect(input).toHaveAttribute("aria-activedescendant", createOption.id);

    await user.keyboard("{Enter}");

    expect(onCommit).toHaveBeenCalledWith("atla");
  });
});
