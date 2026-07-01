import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { ThemeToggle } from './ThemeToggle';

describe('ThemeToggle', () => {
  it('uses shared action primitive semantics without legacy theme-toggle classes', () => {
    render(<ThemeToggle mode="dark" onToggle={vi.fn()} />);

    const button = screen.getByRole('button', { name: 'Activer le mode clair' });
    expect(button).toHaveAttribute('data-variant', 'ghost');
    expect(button).toHaveTextContent('Clair');
    expect(button.getAttribute('class')).not.toContain('theme-toggle');
  });

  it('switches the accessible and visible label in light mode', () => {
    render(<ThemeToggle mode="light" onToggle={vi.fn()} />);

    expect(screen.getByRole('button', { name: 'Activer le mode sombre' })).toHaveTextContent('Sombre');
  });
});
