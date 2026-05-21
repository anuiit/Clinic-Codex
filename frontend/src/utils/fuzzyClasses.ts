import type { TopKItem } from '../types';

export interface ClassSuggestion {
  name: string;
  source: 'class' | 'top_k' | 'custom';
  score: number;
}

export function normalizeClassName(value: string): string {
  return value.trim().replace(/\s+/g, ' ');
}

function rankClassName(query: string, candidate: string, confidenceBonus = 0): number {
  const normalizedQuery = normalizeClassName(query).toLowerCase();
  const normalizedCandidate = normalizeClassName(candidate).toLowerCase();

  if (!normalizedCandidate) return 0;
  if (!normalizedQuery) return 10 + confidenceBonus;
  if (normalizedCandidate === normalizedQuery) return 1000 + confidenceBonus;
  if (normalizedCandidate.startsWith(normalizedQuery)) return 800 - normalizedCandidate.length + confidenceBonus;
  if (normalizedCandidate.includes(normalizedQuery)) return 500 - normalizedCandidate.indexOf(normalizedQuery) - normalizedCandidate.length + confidenceBonus;

  let cursor = 0;
  let gaps = 0;
  for (const char of normalizedQuery) {
    const foundAt = normalizedCandidate.indexOf(char, cursor);
    if (foundAt === -1) return 0;
    gaps += foundAt - cursor;
    cursor = foundAt + 1;
  }
  return 250 - gaps - normalizedCandidate.length + confidenceBonus;
}

export function getFuzzyClassSuggestions(
  query: string,
  classNames: string[],
  topK: TopKItem[] = [],
  customClassNames: string[] = [],
  limit = 6,
): ClassSuggestion[] {
  const byName = new Map<string, ClassSuggestion>();

  const addCandidate = (name: string, source: ClassSuggestion['source'], confidenceBonus = 0) => {
    const normalizedName = normalizeClassName(name);
    if (!normalizedName) return;
    const key = normalizedName.toLowerCase();
    const score = rankClassName(query, normalizedName, confidenceBonus);
    if (score <= 0) return;
    const current = byName.get(key);
    if (!current || score > current.score) {
      byName.set(key, { name: normalizedName, source, score });
    }
  };

  classNames.forEach((name) => addCandidate(name, 'class'));
  customClassNames.forEach((name) => addCandidate(name, 'custom', 75));
  topK.forEach((item) => addCandidate(item.class_name, 'top_k', Math.round(item.confidence * 100)));

  return Array.from(byName.values())
    .sort((a, b) => b.score - a.score || a.name.localeCompare(b.name))
    .slice(0, limit);
}

export function hasExactClassName(value: string, candidates: string[]): boolean {
  const normalizedValue = normalizeClassName(value).toLowerCase();
  return candidates.some((candidate) => normalizeClassName(candidate).toLowerCase() === normalizedValue);
}

export function isUnnamedClass(value: string): boolean {
  const normalizedValue = normalizeClassName(value).toLowerCase();
  return normalizedValue === '' || normalizedValue === 'unknown';
}
