import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { ReferenceDecisionPane, ReferenceThumb, ReferenceTileArt } from './ReferenceGlyphArt';

describe('admin reference glyph art', () => {
  it('does not render placeholder glyphs behind real media', () => {
    const { container } = render(
      <>
        <ReferenceThumb><img alt="real thumb" src="/crop.png" /></ReferenceThumb>
        <span className="relative block h-24 w-24">
          <ReferenceTileArt><img alt="real tile" src="/tile.png" /></ReferenceTileArt>
        </span>
        <ReferenceDecisionPane type="context"><img alt="real page" src="/page.png" /></ReferenceDecisionPane>
        <ReferenceDecisionPane type="crop"><img alt="real crop" src="/crop-large.png" /></ReferenceDecisionPane>
      </>,
    );

    expect(document.querySelector('[data-reference-art="thumb"]')).toBeTruthy();
    expect(document.querySelector('[data-reference-art="tile"]')).toBeTruthy();
    expect(document.querySelector('[data-reference-art="context"]')).toHaveClass('admin-decision-media__context');
    expect(document.querySelector('[data-reference-art="crop"]')).toHaveClass('admin-decision-media__crop');
    expect(screen.queryByText('Aperçu mock de glyphe')).toBeNull();
    expect(container.querySelector('svg')).toBeNull();
    for (const alt of ['real thumb', 'real tile', 'real page', 'real crop']) {
      const image = screen.getByAltText(alt);
      expect(image.closest('.opacity-0')).toBeNull();
      expect(image.closest('.z-10')).toBeTruthy();
    }
    for (const art of container.querySelectorAll('[data-reference-art]')) {
      expect(art).toHaveAttribute('data-real-media', 'true');
      expect(art.className).toContain('bg-transparent');
    }
  });
});
