import { act, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi, type MockedFunction } from 'vitest';
import type { AnalysisRecord, TrustResult } from '../../types';
import { getTrust } from '../../services/api';
import { useWorkspaceViewport } from './useWorkspaceViewport';

vi.mock('../../services/api', () => ({
  getTrust: vi.fn(),
}));

const mockedGetTrust = getTrust as MockedFunction<typeof getTrust>;

type Deferred<T> = {
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (reason?: unknown) => void;
};

function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function trustResult(className: string): TrustResult {
  return {
    query: { bbox: [0, 0, 1, 1], predicted_class: className },
    trust: {
      predicted_class_rank: 1,
      predicted_class_similarity: 0.88,
      top1_class: className,
      top1_similarity: className === 'second' ? 0.96 : 0.84,
      margin_to_second: 0.2,
      above_rejection_threshold: true,
      rejection_threshold: 0.35,
      ambiguous: false,
      entropy: 0.12,
      top_k: [{ class_name: className, confidence: className === 'second' ? 0.96 : 0.84 }],
    },
  };
}

const RECORD: AnalysisRecord = {
  id: 'phase-4-record',
  imageName: 'phase4.png',
  imageDataUrl: 'data:image/png;base64,phase4',
  timestamp: 1770000000000,
  result: {
    num_elements: 2,
    image_size: [20, 20],
    elements: [
      { bbox: [1, 2, 3, 4], class_name: 'first', confidence: 0.4, rejected: false, top_k: [] },
      { bbox: [5, 6, 7, 8], class_name: 'second', confidence: 0.7, rejected: false, top_k: [] },
    ],
  },
  annotations: {},
};

function Harness({ record = RECORD }: { record?: AnalysisRecord | null }) {
  const viewport = useWorkspaceViewport(record);
  return (
    <div>
      <button type="button" onClick={() => viewport.setFocusedIdx(0)}>focus first</button>
      <button type="button" onClick={() => viewport.setFocusedIdx(1)}>focus second</button>
      <div data-testid="loading">{viewport.contextLoading ? 'loading' : 'idle'}</div>
      <div data-testid="trust">{viewport.trustData?.trust.top1_class ?? 'none'}</div>
    </div>
  );
}

describe('useWorkspaceViewport Phase 4 trust lifecycle', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('passes an abort signal to trust inspection requests', async () => {
    const first = deferred<TrustResult>();
    mockedGetTrust.mockReturnValueOnce(first.promise);
    const user = userEvent.setup();
    render(<Harness />);

    await user.click(screen.getByRole('button', { name: 'focus first' }));

    expect(mockedGetTrust).toHaveBeenCalledWith(
      'data:image/png;base64,phase4',
      [1, 2, 3, 4],
      'first',
      10,
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
    expect(screen.getByTestId('loading')).toHaveTextContent('loading');
  });

  it('aborts and ignores an older trust response when focus changes', async () => {
    const first = deferred<TrustResult>();
    const second = deferred<TrustResult>();
    mockedGetTrust.mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
    const user = userEvent.setup();
    render(<Harness />);

    await user.click(screen.getByRole('button', { name: 'focus first' }));
    const firstSignal = mockedGetTrust.mock.calls[0][4]?.signal;
    await user.click(screen.getByRole('button', { name: 'focus second' }));

    expect(firstSignal?.aborted).toBe(true);
    expect(screen.getByTestId('trust')).toHaveTextContent('none');
    expect(screen.getByTestId('loading')).toHaveTextContent('loading');
    expect(mockedGetTrust).toHaveBeenLastCalledWith(
      'data:image/png;base64,phase4',
      [5, 6, 7, 8],
      'second',
      10,
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );

    await act(async () => {
      second.resolve(trustResult('second'));
      await second.promise;
    });
    expect(screen.getByTestId('trust')).toHaveTextContent('second');
    expect(screen.getByTestId('loading')).toHaveTextContent('idle');

    await act(async () => {
      first.resolve(trustResult('first'));
      await first.promise;
    });
    expect(screen.getByTestId('trust')).toHaveTextContent('second');
    expect(screen.getByTestId('loading')).toHaveTextContent('idle');
  });

  it('ignores an older rejected trust request after a newer request is active', async () => {
    const first = deferred<TrustResult>();
    const second = deferred<TrustResult>();
    mockedGetTrust.mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
    const user = userEvent.setup();
    render(<Harness />);

    await user.click(screen.getByRole('button', { name: 'focus first' }));
    await user.click(screen.getByRole('button', { name: 'focus second' }));

    await act(async () => {
      second.resolve(trustResult('second'));
      await second.promise;
    });
    await act(async () => {
      first.reject(new DOMException('aborted', 'AbortError'));
      await first.promise.catch(() => undefined);
    });

    expect(screen.getByTestId('trust')).toHaveTextContent('second');
    expect(screen.getByTestId('loading')).toHaveTextContent('idle');
  });

  it('clears trust data and loading state for the active rejected request', async () => {
    const first = deferred<TrustResult>();
    mockedGetTrust.mockReturnValueOnce(first.promise);
    const user = userEvent.setup();
    render(<Harness />);

    await user.click(screen.getByRole('button', { name: 'focus first' }));
    await act(async () => {
      first.reject(new Error('network'));
      await first.promise.catch(() => undefined);
    });

    expect(screen.getByTestId('trust')).toHaveTextContent('none');
    expect(screen.getByTestId('loading')).toHaveTextContent('idle');
  });
});
