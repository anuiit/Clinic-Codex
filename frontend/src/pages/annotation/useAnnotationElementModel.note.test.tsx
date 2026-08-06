import { act, renderHook } from '@testing-library/react';
import { useRef, useState } from 'react';
import { describe, expect, it } from 'vitest';
import type { AnnotationStatus, DetectedElement } from '../../types';
import { useAnnotationElementModel } from './useAnnotationElementModel';

function element(className: string, note?: string): DetectedElement {
  return {
    bbox: [1, 2, 3, 4],
    class_name: className,
    confidence: 0.8,
    rejected: false,
    top_k: [{ class_name: className, confidence: 0.8 }],
    ...(note !== undefined ? { note } : {}),
  };
}

function useHarness(initialElements: DetectedElement[]) {
  const [elements, setElements] = useState<DetectedElement[]>(initialElements);
  const [annotationStatus, setAnnotationStatus] = useState<Record<number, AnnotationStatus>>({ 0: 'draft' });
  const [customClasses, setCustomClasses] = useState<string[]>([]);
  const cardRefs = useRef<Array<HTMLElement | null>>([]);
  const model = useAnnotationElementModel({
    elements,
    setElements,
    annotationStatus,
    setAnnotationStatus,
    classes: ['atl'],
    setCustomClasses,
    focusedIdx: 0,
    cardRefs,
  });
  return { elements, model };
}

describe('useAnnotationElementModel element note', () => {
  it('sets a trimmed note on the element without touching other fields', () => {
    const { result } = renderHook(() => useHarness([element('atl')]));

    act(() => {
      result.current.model.commitElementNote(0, '  contour effacé  ');
    });

    expect(result.current.elements[0]).toEqual(
      expect.objectContaining({ class_name: 'atl', note: 'contour effacé', bbox: [1, 2, 3, 4] }),
    );
  });

  it('clears the note key entirely when committed empty', () => {
    const { result } = renderHook(() => useHarness([element('atl', 'brouillon')]));

    act(() => {
      result.current.model.commitElementNote(0, '   ');
    });

    expect('note' in result.current.elements[0]).toBe(false);
  });

  it('keeps object identity when the note is unchanged (no useless rerender)', () => {
    const initial = element('atl', 'stable');
    const { result } = renderHook(() => useHarness([initial]));

    act(() => {
      result.current.model.commitElementNote(0, 'stable');
    });

    expect(result.current.elements[0]).toBe(initial);
  });
});
