import { beforeEach, describe, expect, it } from 'vitest';
import type { AnalysisRecord } from '../types';
import { getHistory, saveAnalysis, updateElements } from './storage';

const RECORD: AnalysisRecord = {
  id: 'analysis-id',
  imageName: 'sample.png',
  imageDataUrl: 'data:image/png;base64,abc',
  timestamp: 1704067200000,
  result: {
    num_elements: 2,
    image_size: [800, 600],
    elements: [
      { bbox: [1, 2, 3, 4], class_name: 'atl', class_label: 1, confidence: 0.9, rejected: false, top_k: [] },
      { bbox: [5, 6, 7, 8], class_name: 'beta', class_label: 2, confidence: 0.8, rejected: false, top_k: [] },
    ],
  },
  annotations: {},
};

describe('storage annotation status compatibility', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it('reads old records without annotationStatus as draft-compatible records', () => {
    localStorage.setItem('codex_analyses', JSON.stringify([RECORD]));

    expect(getHistory()[0]).toEqual(expect.objectContaining({ annotationStatus: {} }));
  });

  it('persists and preserves annotation status when elements are updated', () => {
    saveAnalysis({ ...RECORD, annotationStatus: { 0: 'validated', 1: 'draft' } });

    const nextElements = [
      { ...RECORD.result.elements[0], class_name: 'renamed' },
    ];
    expect(updateElements('analysis-id', nextElements)).toBe(true);

    const [stored] = getHistory();
    expect(stored.result.elements).toHaveLength(1);
    expect(stored.annotationStatus).toEqual({ 0: 'validated' });
  });

  it('accepts explicit annotation status updates', () => {
    saveAnalysis(RECORD);

    expect(updateElements('analysis-id', RECORD.result.elements, { 0: 'validated' })).toBe(true);

    expect(getHistory()[0].annotationStatus).toEqual({ 0: 'validated' });
  });
});
