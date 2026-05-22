import {
  useEffect,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
} from "react";
import type { DetectedElement } from "../../types";
import { appText } from "../../i18n/text";
import {
  getFuzzyClassSuggestions,
  hasExactClassName,
  isUnnamedClass,
  normalizeClassName,
} from "../../utils/fuzzyClasses";

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
  const inputRef = useRef<HTMLInputElement>(null);
  const displayValue = isUnnamedClass(value) ? "" : value;
  const [inputState, setInputState] = useState(() => ({
    sourceValue: value,
    inputValue: displayValue,
  }));
  const inputValue =
    inputState.sourceValue === value ? inputState.inputValue : displayValue;
  if (inputState.sourceValue !== value) {
    setInputState({ sourceValue: value, inputValue: displayValue });
  }
  const setInputValue = (nextValue: string) => {
    setInputState((current) => ({ ...current, inputValue: nextValue }));
  };
  const [isOpen, setIsOpen] = useState(false);
  const [highlightedIdx, setHighlightedIdx] = useState(0);
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

  return (
    <div className="relative" onClick={(event) => event.stopPropagation()}>
      <label
        className="mb-1 block text-xs font-medium uppercase tracking-[0.18em] text-stone-500"
        htmlFor={`element-name-${index}`}
      >
        {labels.renameElement}
      </label>
      <input
        ref={inputRef}
        id={`element-name-${index}`}
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
        className="w-full rounded-lg border border-stone-700 bg-stone-900 px-3 py-2 text-sm text-stone-100 outline-none transition-colors placeholder:text-stone-600 focus:border-amber-500"
      />
      {isOpen && (
        <div className="absolute z-20 mt-1 max-h-56 w-full overflow-y-auto rounded-lg border border-stone-700 bg-stone-950 shadow-xl">
          <div className="border-b border-stone-800 px-3 py-1.5 text-[11px] uppercase tracking-[0.2em] text-stone-500">
            {labels.suggestions}
          </div>
          {suggestions.length === 0 && !canCreate && (
            <div className="px-3 py-2 text-sm text-stone-500">
              {labels.noSuggestion}
            </div>
          )}
          {suggestions.map((suggestion, suggestionIdx) => (
            <button
              key={`${suggestion.source}-${suggestion.name}`}
              type="button"
              onMouseDown={(event) => event.preventDefault()}
              onClick={() => commitName(suggestion.name)}
              className={`flex w-full items-center justify-between px-3 py-2 text-left text-sm transition-colors ${suggestionIdx === highlightedIdx ? "bg-amber-500/15 text-amber-100" : "text-stone-100 hover:bg-stone-800"}`}
            >
              <span>{suggestion.name}</span>
              <span className="text-[10px] uppercase tracking-[0.18em] text-stone-500">
                {suggestion.source}
              </span>
            </button>
          ))}
          {canCreate && (
            <button
              type="button"
              onMouseDown={(event) => event.preventDefault()}
              onClick={() => commitName(normalizedInput)}
              className="w-full border-t border-stone-800 px-3 py-2 text-left text-sm font-medium text-emerald-300 transition-colors hover:bg-emerald-500/10"
            >
              {labels.createElementName} « {normalizedInput} »
            </button>
          )}
        </div>
      )}
    </div>
  );
}
