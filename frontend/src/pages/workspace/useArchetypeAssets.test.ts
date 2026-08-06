import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('../../services/api', () => ({
  getSimilar: vi.fn(),
}));
vi.mock('./archetypeFlag', () => ({
  isArchetypeGalleryEnabled: vi.fn(),
}));

import { getSimilar } from '../../services/api';
import { isArchetypeGalleryEnabled } from './archetypeFlag';
import { useArchetypeAssets } from './useArchetypeAssets';

const mockedGetSimilar = vi.mocked(getSimilar);
const mockedFlag = vi.mocked(isArchetypeGalleryEnabled);

function similarResponse(results: Array<{ class_name: string; asset: string | null }>) {
  return {
    query: { bbox: [0, 0, 4, 4], mode: 'prototype' },
    best_match: { class_name: results[0]?.class_name ?? 'atl', similarity: 0.7, rejected: false },
    results: results.map((r, i) => ({
      rank: i + 1,
      match_type: 'class_prototype',
      class_name: r.class_name,
      class_label: null,
      similarity: 0.7 - i * 0.1,
      band: 'high' as const,
      asset: r.asset,
    })),
  };
}

describe('useArchetypeAssets', () => {
  beforeEach(() => {
    mockedFlag.mockReturnValue(true);
    mockedGetSimilar.mockReset();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('stays disabled when the flag is off', async () => {
    mockedFlag.mockReturnValue(false);
    const { result } = renderHook(() =>
      useArchetypeAssets('rec-1', 'data:image/png;base64,x', [0, 0, 4, 4], ['atl']),
    );
    expect(result.current.status).toBe('disabled');
    expect(mockedGetSimilar).not.toHaveBeenCalled();
  });

  it('maps per-class assets and reports coverage', async () => {
    mockedGetSimilar.mockResolvedValue(
      similarResponse([
        { class_name: 'atl', asset: '/samples/atl/a.png' },
        { class_name: 'calli', asset: null },
      ]) as never,
    );

    const { result } = renderHook(() =>
      useArchetypeAssets('rec-1', 'data:image/png;base64,x', [0, 0, 4, 4], ['atl', 'calli']),
    );

    await waitFor(() => expect(result.current.status).toBe('ready'));
    if (result.current.status !== 'ready') throw new Error('unreachable');
    expect(result.current.assets.get('atl')).toBe('/samples/atl/a.png');
    expect(result.current.assets.get('calli')).toBeNull();
    expect(result.current.covered).toBe(1);
    expect(result.current.total).toBe(2);
  });

  it('surfaces an error state when the similar call fails', async () => {
    mockedGetSimilar.mockRejectedValue(new Error('network') as never);

    const { result } = renderHook(() =>
      useArchetypeAssets('rec-1', 'data:image/png;base64,x', [0, 0, 4, 4], ['atl']),
    );

    await waitFor(() => expect(result.current.status).toBe('error'));
  });

  it('refetches when the focused element changes', async () => {
    mockedGetSimilar.mockResolvedValue(
      similarResponse([{ class_name: 'atl', asset: '/samples/atl/a.png' }]) as never,
    );

    const { result, rerender } = renderHook(
      ({ bbox }) => useArchetypeAssets('rec-1', 'data:image/png;base64,x', bbox, ['atl']),
      { initialProps: { bbox: [0, 0, 4, 4] as [number, number, number, number] } },
    );
    await waitFor(() => expect(result.current.status).toBe('ready'));
    expect(mockedGetSimilar).toHaveBeenCalledTimes(1);

    await act(async () => {
      rerender({ bbox: [4, 4, 8, 8] });
    });
    expect(mockedGetSimilar).toHaveBeenCalledTimes(2);
  });
});
