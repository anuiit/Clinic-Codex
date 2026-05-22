import { act, renderHook } from '@testing-library/react';
import { useRef, useState } from 'react';
import { describe, expect, it } from 'vitest';
import type { AnnotationStatus, DetectedElement } from '../../types';
import { useAnnotationElementModel } from './useAnnotationElementModel';

function element(className: string, bbox: [number, number, number, number], confidence = 0.8): DetectedElement {
  return {
    bbox,
    class_name: className,
    confidence,
    rejected: false,
    top_k: [{ class_name: className, confidence }],
  };
}

function useHarness(initialElements = [element('unknown', [1, 2, 3, 4]), element('atl', [5, 6, 7, 8])]) {
  const [elements, setElements] = useState<DetectedElement[]>(initialElements);
  const [annotationStatus, setAnnotationStatus] = useState<Record<number, AnnotationStatus>>({ 0: 'draft', 1: 'draft' });
  const [customClasses, setCustomClasses] = useState<string[]>([]);
  const cardRefs = useRef<Array<HTMLElement | null>>([]);
  const model = useAnnotationElementModel({
    elements,
    setElements,
    annotationStatus,
    setAnnotationStatus,
    classes: ['atl', 'calli'],
    setCustomClasses,
    focusedIdx: 1,
    cardRefs,
  });
  return { elements, annotationStatus, customClasses, cardRefs, model };
}

describe('useAnnotationElementModel Phase 5 state coverage', () => {
  it('loads elements and derives submitted/focused display state without mutating bbox metadata', () => {
    const { result } = renderHook(() => useHarness());

    expect(result.current.elements.map((item) => item.bbox)).toEqual([[1, 2, 3, 4], [5, 6, 7, 8]]);
    expect(result.current.model.focusedDisplayName).toBe('atl');
    expect(result.current.model.focusedConfidencePercent).toBe(80);
    expect(result.current.model.submittedCount).toBe(0);
  });

  it('renames an element, preserves bbox, resets validation to draft, and records custom classes', () => {
    const { result } = renderHook(() => useHarness());

    act(() => {
      result.current.model.setElementValidation(1, 'validated');
    });
    act(() => {
      result.current.model.commitElementName(1, 'New Glyph');
    });

    expect(result.current.elements[1]).toEqual(expect.objectContaining({ class_name: 'New Glyph', bbox: [5, 6, 7, 8] }));
    expect(result.current.annotationStatus[1]).toBe('draft');
    expect(result.current.customClasses).toEqual(['New Glyph']);
  });

  it('toggles validation/draft status and validates only named elements on submitNamedElements', () => {
    const { result } = renderHook(() => useHarness());

    act(() => {
      result.current.model.setElementValidation(1, 'validated');
    });
    expect(result.current.annotationStatus[1]).toBe('validated');
    expect(result.current.model.submittedCount).toBe(1);

    act(() => {
      result.current.model.setElementValidation(1, 'draft');
    });
    expect(result.current.annotationStatus[1]).toBe('draft');

    act(() => {
      result.current.model.submitNamedElements();
    });

    expect(result.current.annotationStatus).toEqual({ 0: 'draft', 1: 'validated' });
    expect(result.current.model.submittedCount).toBe(1);
  });

  it('filters and sorts displayed elements without changing original element order or bbox data', () => {
    const { result } = renderHook(() => useHarness([
      element('zeta', [9, 9, 2, 2], 0.2),
      element('alpha', [1, 1, 3, 3], 0.9),
    ]));

    act(() => {
      result.current.model.setSortMode('name');
    });
    expect(result.current.model.displayedElements.map(({ idx }) => idx)).toEqual([1, 0]);
    expect(result.current.elements.map((item) => item.bbox)).toEqual([[9, 9, 2, 2], [1, 1, 3, 3]]);

    act(() => {
      result.current.model.setListQuery('zet');
    });
    expect(result.current.model.displayedElements.map(({ idx }) => idx)).toEqual([0]);
  });
});
