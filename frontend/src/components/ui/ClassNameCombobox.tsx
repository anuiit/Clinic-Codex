import {
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent as ReactKeyboardEvent,
} from "react";
import { createPortal } from "react-dom";
import type { TopKItem } from "../../types";
import {
  getFuzzyClassSuggestions,
  hasExactClassName,
  isUnnamedClass,
  normalizeClassName,
} from "../../utils/fuzzyClasses";

function cx(...parts: Array<string | false | null | undefined>) {
  return parts.filter(Boolean).join(" ");
}

export interface ClassNameComboboxLabels {
  suggestions: string;
  noSuggestion: string;
  createElementName: string;
  renameElement: string;
  nameElement: string;
  elementNamePlaceholder: string;
}

export interface ClassNameComboboxProps {
  value: string;
  classNames: string[];
  customClassNames: string[];
  topK: TopKItem[];
  autoFocusToken: number;
  labels: ClassNameComboboxLabels;
  index: number;
  onCommit: (name: string) => void;
  onInputChange?: (name: string) => void;
  disabled?: boolean;
  inputId?: string;
  ariaLabel?: string;
  className?: string;
  menuClassName?: string;
}

export function ClassNameCombobox({
  value,
  classNames,
  customClassNames,
  topK,
  autoFocusToken,
  labels,
  index,
  onCommit,
  onInputChange,
  disabled = false,
  inputId,
  ariaLabel,
  className = "",
  menuClassName = "",
}: ClassNameComboboxProps) {
  const generatedId = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const displayValue = isUnnamedClass(value) ? "" : value;
  const [inputState, setInputState] = useState(() => ({
    sourceValue: value,
    inputValue: displayValue,
  }));
  const inputValue =
    inputState.sourceValue === value ? inputState.inputValue : displayValue;
  const setInputValue = (nextValue: string) => {
    setInputState({ sourceValue: value, inputValue: nextValue });
  };
  const [isOpen, setIsOpen] = useState(false);
  const [highlightedIdx, setHighlightedIdx] = useState(0);
  const [menuPosition, setMenuPosition] = useState<{
    left: number;
    top: number;
    width: number;
    maxHeight: number;
  } | null>(null);
  const suggestions = getFuzzyClassSuggestions(
    inputValue,
    classNames,
    topK,
    customClassNames,
  );
  const normalizedInput = normalizeClassName(inputValue);
  const allCandidateNames = [
    ...classNames,
    ...customClassNames,
    ...topK.map((item) => item.class_name),
  ];
  const canCreate =
    normalizedInput.length > 0 &&
    !hasExactClassName(normalizedInput, allCandidateNames);
  const optionCount = suggestions.length + (canCreate ? 1 : 0);
  const resolvedInputId = inputId ?? `element-name-${index}-${generatedId}`;
  const listboxId = `${resolvedInputId}-suggestions`;
  const createOptionId = `${listboxId}-create`;
  const activeOptionId =
    highlightedIdx < suggestions.length
      ? `${listboxId}-option-${highlightedIdx}`
      : canCreate && highlightedIdx === suggestions.length
        ? createOptionId
        : undefined;

  const updateMenuPosition = useCallback(() => {
    if (typeof window === "undefined") return;

    const input = inputRef.current;
    if (!input) return;

    const rect = input.getBoundingClientRect();
    const viewportPadding = 8;
    const viewportWidth = window.innerWidth || 1024;
    const viewportHeight = window.innerHeight || 768;
    const menuWidth = Math.min(
      Math.max(rect.width, 220),
      Math.max(220, viewportWidth - viewportPadding * 2),
    );
    const availableBelow = viewportHeight - rect.bottom - viewportPadding;
    const availableAbove = rect.top - viewportPadding;
    const openAbove = availableBelow < 160 && availableAbove > availableBelow;
    const availableHeight = openAbove ? availableAbove : availableBelow;
    const maxHeight = Math.max(120, Math.min(224, availableHeight - 4));
    const minLeft = viewportPadding;
    const maxLeft = Math.max(minLeft, viewportWidth - menuWidth - viewportPadding);
    const left = Math.max(minLeft, Math.min(rect.left, maxLeft));
    const top = openAbove
      ? Math.max(viewportPadding, rect.top - maxHeight - 4)
      : Math.min(rect.bottom + 4, viewportHeight - viewportPadding - maxHeight);

    setMenuPosition({ left, top, width: menuWidth, maxHeight });
  }, []);

  useEffect(() => {
    if (autoFocusToken <= 0 || disabled) return;
    inputRef.current?.focus();
    inputRef.current?.select();
    let active = true;
    queueMicrotask(() => {
      if (active) setIsOpen(true);
    });
    return () => {
      active = false;
    };
  }, [autoFocusToken, disabled]);

  useLayoutEffect(() => {
    if (!isOpen) return;

    updateMenuPosition();
    window.addEventListener("resize", updateMenuPosition);
    window.addEventListener("scroll", updateMenuPosition, true);

    return () => {
      window.removeEventListener("resize", updateMenuPosition);
      window.removeEventListener("scroll", updateMenuPosition, true);
    };
  }, [isOpen, inputValue, suggestions.length, canCreate, updateMenuPosition]);

  const commitName = (name: string) => {
    const normalizedName = normalizeClassName(name);
    if (!normalizedName) return;
    onCommit(normalizedName);
    setInputState({ sourceValue: normalizedName, inputValue: normalizedName });
    setIsOpen(false);
  };

  const handleKeyDown = (event: ReactKeyboardEvent<HTMLInputElement>) => {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setIsOpen(true);
      setHighlightedIdx((current) =>
        Math.min(current + 1, Math.max(optionCount - 1, 0)),
      );
      return;
    }
    if (event.key === "ArrowUp") {
      event.preventDefault();
      setHighlightedIdx((current) => Math.max(current - 1, 0));
      return;
    }
    if (event.key === "Escape") {
      setIsOpen(false);
      setInputValue(displayValue);
      return;
    }
    if (event.key === "Enter") {
      event.preventDefault();
      const highlightedSuggestion = suggestions[highlightedIdx];
      if (highlightedSuggestion) {
        commitName(highlightedSuggestion.name);
      } else if (canCreate) {
        commitName(inputValue);
      }
    }
  };

  const menuStyle: CSSProperties = {
    left: menuPosition?.left ?? 0,
    top: menuPosition?.top ?? 0,
    width: menuPosition?.width ?? 220,
    maxHeight: menuPosition?.maxHeight ?? 224,
  };

  const suggestionsMenu = (
    <div
      id={listboxId}
      role="listbox"
      aria-label={labels.suggestions}
      className={cx(
        "ui-panel ui-class-name-menu fixed overflow-y-auto rounded-none shadow-xl",
        menuClassName,
      )}
      data-testid="element-name-suggestions"
      style={menuStyle}
    >
      <div className="ui-divider ui-text-eyebrow border-b px-3 py-1.5">
        {labels.suggestions}
      </div>
      {suggestions.length === 0 && !canCreate && (
        <div className="ui-text-body-sm px-3 py-2">{labels.noSuggestion}</div>
      )}
      {suggestions.map((suggestion, suggestionIdx) => (
        <button
          id={`${listboxId}-option-${suggestionIdx}`}
          role="option"
          aria-selected={suggestionIdx === highlightedIdx}
          key={`${suggestion.source}-${suggestion.name}`}
          type="button"
          onMouseDown={(event) => event.preventDefault()}
          onClick={() => commitName(suggestion.name)}
          className={cx(
            "flex w-full items-center justify-between px-3 py-2 text-left text-sm transition-colors",
            suggestionIdx === highlightedIdx
              ? "ui-row--active text-[var(--text-main)]"
              : "text-[var(--text-main)] hover:bg-[var(--row-hover)]",
          )}
        >
          <span>{suggestion.name}</span>
          <span className="ui-text-meta uppercase tracking-[0.18em]">
            {suggestion.source}
          </span>
        </button>
      ))}
      {canCreate && (
        <button
          id={createOptionId}
          role="option"
          aria-selected={highlightedIdx === suggestions.length}
          type="button"
          onMouseDown={(event) => event.preventDefault()}
          onClick={() => commitName(normalizedInput)}
          className={cx(
            "ui-divider w-full border-t px-3 py-2 text-left text-sm font-medium text-[color:var(--status-ready-text)] transition-colors hover:bg-[color:var(--status-ready-soft)]",
            highlightedIdx === suggestions.length && "bg-[color:var(--status-ready-soft)]",
          )}
        >
          {labels.createElementName} « {normalizedInput} »
        </button>
      )}
    </div>
  );

  return (
    <div
      className={cx("relative", className)}
      onClick={(event) => event.stopPropagation()}
    >
      <label className="ui-text-eyebrow mb-1 block" htmlFor={resolvedInputId}>
        {labels.renameElement}
      </label>
      <input
        ref={inputRef}
        id={resolvedInputId}
        role="combobox"
        aria-autocomplete="list"
        aria-controls={isOpen ? listboxId : undefined}
        aria-expanded={isOpen}
        aria-activedescendant={isOpen ? activeOptionId : undefined}
        aria-label={ariaLabel ?? `${labels.nameElement} ${index}`}
        value={inputValue}
        disabled={disabled}
        onChange={(event) => {
          if (disabled) return;
          setInputValue(event.target.value);
          onInputChange?.(event.target.value);
          setHighlightedIdx(0);
          setIsOpen(true);
        }}
        onFocus={() => {
          if (!disabled) setIsOpen(true);
        }}
        onBlur={() => {
          if (disabled) return;
          if (normalizedInput) {
            commitName(inputValue);
          } else {
            setIsOpen(false);
          }
        }}
        onKeyDown={handleKeyDown}
        placeholder={labels.elementNamePlaceholder}
        className="ui-input w-full px-3 py-2"
      />
      {isOpen && typeof document !== "undefined"
        ? createPortal(suggestionsMenu, document.body)
        : null}
    </div>
  );
}
