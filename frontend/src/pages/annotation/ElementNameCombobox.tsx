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
import type { DetectedElement } from "../../types";
import { appText } from "../../i18n/text";
import {
  getFuzzyClassSuggestions,
  hasExactClassName,
  isUnnamedClass,
  normalizeClassName,
} from "../../utils/fuzzyClasses";
import annotationStyles from "./AnnotationChrome.module.css";

interface ElementNameComboboxProps {
  value: string;
  classNames: string[];
  customClassNames: string[];
  topK: DetectedElement["top_k"];
  autoFocusToken: number;
  labels: typeof appText.annotation;
  index: number;
  onCommit: (name: string) => void;
}

export function ElementNameCombobox({
  value,
  classNames,
  customClassNames,
  topK,
  autoFocusToken,
  labels,
  index,
  onCommit,
}: ElementNameComboboxProps) {
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
  const inputId = `element-name-${index}-${generatedId}`;
  const listboxId = `${inputId}-suggestions`;
  const activeOptionId = suggestions[highlightedIdx]
    ? `${listboxId}-option-${highlightedIdx}`
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
    if (autoFocusToken <= 0) return;
    inputRef.current?.focus();
    inputRef.current?.select();
    let active = true;
    queueMicrotask(() => {
      if (active) setIsOpen(true);
    });
    return () => {
      active = false;
    };
  }, [autoFocusToken]);

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
        Math.min(current + 1, Math.max(suggestions.length - 1, 0)),
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
      commitName(highlightedSuggestion?.name ?? inputValue);
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
      className="annotation-name-combobox__menu annotation-name-combobox__menu--portal ui-panel fixed overflow-y-auto rounded-none shadow-xl"
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
          className={`flex w-full items-center justify-between px-3 py-2 text-left text-sm transition-colors ${suggestionIdx === highlightedIdx ? "ui-row--active text-[var(--text-main)]" : "text-[var(--text-main)] hover:bg-[var(--row-hover)]"}`}
        >
          <span>{suggestion.name}</span>
          <span className="ui-text-meta uppercase tracking-[0.18em]">
            {suggestion.source}
          </span>
        </button>
      ))}
      {canCreate && (
        <button
          role="option"
          aria-selected={false}
          type="button"
          onMouseDown={(event) => event.preventDefault()}
          onClick={() => commitName(normalizedInput)}
          className="ui-divider w-full border-t px-3 py-2 text-left text-sm font-medium text-[color:var(--status-ready-text)] transition-colors hover:bg-[color:var(--status-ready-soft)]"
        >
          {labels.createElementName} « {normalizedInput} »
        </button>
      )}
    </div>
  );

  return (
    <div
      className={`${annotationStyles.owner} annotation-name-combobox relative`}
      onClick={(event) => event.stopPropagation()}
    >
      <label
        className="ui-text-eyebrow mb-1 block"
        htmlFor={inputId}
      >
        {labels.renameElement}
      </label>
      <input
        ref={inputRef}
        id={inputId}
        role="combobox"
        aria-autocomplete="list"
        aria-controls={isOpen ? listboxId : undefined}
        aria-expanded={isOpen}
        aria-activedescendant={isOpen ? activeOptionId : undefined}
        aria-label={`${labels.nameElement} ${index}`}
        value={inputValue}
        onChange={(event) => {
          setInputValue(event.target.value);
          setHighlightedIdx(0);
          setIsOpen(true);
        }}
        onFocus={() => setIsOpen(true)}
        onBlur={() => {
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
