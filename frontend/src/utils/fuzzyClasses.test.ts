import { describe, expect, it } from 'vitest';
import { getFuzzyClassSuggestions, hasExactClassName, isUnnamedClass, normalizeClassName } from './fuzzyClasses';

describe('fuzzy class suggestions', () => {
  it('ranks prefix matches before contains and subsequence matches', () => {
    const suggestions = getFuzzyClassSuggestions('al', ['palma', 'aleph', 'dalet', 'lamed']);

    expect(suggestions.map((suggestion) => suggestion.name).slice(0, 3)).toEqual(['aleph', 'dalet', 'palma']);
  });

  it('mixes model classes, top-k and custom labels without duplicates', () => {
    const suggestions = getFuzzyClassSuggestions(
      'ta',
      ['beta', 'tav'],
      [{ class_name: 'tav', confidence: 0.92 }, { class_name: 'taw', confidence: 0.5 }],
      ['tag'],
    );

    expect(suggestions.map((suggestion) => suggestion.name)).toEqual(['tav', 'tag', 'taw', 'beta']);
  });

  it('normalizes created labels and detects exact/unnamed labels', () => {
    expect(normalizeClassName('  new   glyph  ')).toBe('new glyph');
    expect(hasExactClassName('ALEPH', ['aleph'])).toBe(true);
    expect(isUnnamedClass('unknown')).toBe(true);
    expect(isUnnamedClass('aleph')).toBe(false);
  });
});
