import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import {
  RuntimeVersionContext,
} from './RuntimeVersionContext';
import { RuntimeVersionBadge } from './RuntimeVersionBadge';

describe('RuntimeVersionBadge', () => {
  it('shows only the application and active-model versions', () => {
    render(
      <RuntimeVersionContext.Provider
        value={{
          app_name: 'Clinic Codex',
          app_version: '0.1.0',
          model_version: 'elements-v7',
        }}
      >
        <RuntimeVersionBadge />
      </RuntimeVersionContext.Provider>,
    );

    expect(screen.getByText('Clinic Codex v0.1.0')).toBeInTheDocument();
    expect(screen.getByText('Modèle elements-v7')).toBeInTheDocument();
    expect(screen.getByTestId('runtime-version')).toHaveAttribute(
      'aria-label',
      'Clinic Codex v0.1.0, Modèle elements-v7',
    );
  });

  it('uses a quiet fallback while model metadata is unavailable', () => {
    render(<RuntimeVersionBadge />);

    expect(screen.getByText('Clinic Codex')).toBeInTheDocument();
    expect(screen.getByText('Modèle —')).toBeInTheDocument();
  });
});
