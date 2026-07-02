import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { ActionButton, AdminSection, PillButton, StatusPill, type BadgeTone, type ButtonTone } from './AdminPrimitives';

function classOf(label: string) {
  return screen.getByText(label).getAttribute('class') ?? '';
}

describe('admin UI primitives', () => {
  it('renders all button variants with distinct neutral and ghost classes', () => {
    const tones: ButtonTone[] = ['primary', 'neutral', 'danger', 'ghost', 'ready'];
    render(
      <>
        {tones.map((tone) => (
          <ActionButton key={tone} tone={tone}>{tone}</ActionButton>
        ))}
      </>,
    );

    for (const tone of tones) {
      expect(screen.getByText(tone)).toHaveAttribute('data-variant', tone);
    }
    expect(classOf('neutral')).not.toEqual(classOf('ghost'));
    expect(classOf('neutral')).toContain('bg-[color:var(--surface-muted)]');
    expect(classOf('ghost')).toContain('bg-transparent');
    expect(classOf('ready')).toContain('bg-[color:var(--status-ready)]');
  });

  it('renders pill active and inactive states without page-scoped class hooks', () => {
    render(
      <>
        <PillButton active>Active</PillButton>
        <PillButton>Inactive</PillButton>
      </>,
    );

    const active = classOf('Active');
    const inactive = classOf('Inactive');
    expect(active).not.toEqual(inactive);
    expect(`${active} ${inactive}`).not.toContain('pill-reference');
  });

  it('renders closed badge tones without legacy chip classes', () => {
    const tones: BadgeTone[] = ['neutral', 'ready', 'warning', 'danger'];
    render(
      <>
        {tones.map((tone) => (
          <StatusPill key={tone} tone={tone}>{tone}</StatusPill>
        ))}
      </>,
    );

    for (const tone of tones) {
      const badge = screen.getByText(tone);
      expect(badge).toHaveAttribute('data-tone', tone);
      expect(badge.getAttribute('class')).not.toContain('ui-chip');
    }
  });

  it('renders flat sections through the reusable primitive while preserving admin hooks', () => {
    render(
      <>
        <AdminSection aria-label="Flat primitive">Flat content</AdminSection>
        <AdminSection as="div" variant="summary" aria-label="Summary primitive">
          Summary content
        </AdminSection>
      </>,
    );

    const flat = screen.getByLabelText('Flat primitive');
    const summary = screen.getByLabelText('Summary primitive');
    expect(flat).toHaveClass('admin-flat-section');
    expect(flat.getAttribute('class')).toContain('border-[color:var(--border-subtle)]');
    expect(summary).toHaveClass('admin-flat-section');
    expect(summary).toHaveClass('admin-page-summary');
  });
});
