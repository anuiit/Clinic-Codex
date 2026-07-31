import { render, screen, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { RuntimeVersionBadge } from './RuntimeVersionBadge';
import { RuntimeVersionProvider } from './RuntimeVersionProvider';

const apiMock = vi.hoisted(() => ({
  getRuntimeVersion: vi.fn(),
}));

vi.mock('../services/api', () => apiMock);

describe('RuntimeVersionProvider', () => {
  it('hydrates the shared badge from the backend once', async () => {
    apiMock.getRuntimeVersion.mockResolvedValueOnce({
      app_name: 'Clinic Codex',
      app_version: '0.1.0',
      model_version: '1.0.0',
    });

    render(
      <RuntimeVersionProvider>
        <RuntimeVersionBadge />
      </RuntimeVersionProvider>,
    );

    await waitFor(() => {
      expect(screen.getByText('Clinic Codex v0.1.0')).toBeInTheDocument();
      expect(screen.getByText('Modèle v1.0.0')).toBeInTheDocument();
    });
    expect(apiMock.getRuntimeVersion).toHaveBeenCalledTimes(1);
    expect(apiMock.getRuntimeVersion).toHaveBeenCalledWith({
      signal: expect.any(AbortSignal),
    });
  });
});
