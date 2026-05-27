import { describe, it, expect, vi } from 'vitest';
import { clientToImage, imageToClient } from './imageCoords';

function makeSvg(rect: { left: number; top: number; width: number; height: number }): SVGSVGElement {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg') as SVGSVGElement;
  svg.getBoundingClientRect = vi.fn().mockReturnValue({
    left: rect.left,
    top: rect.top,
    width: rect.width,
    height: rect.height,
    right: rect.left + rect.width,
    bottom: rect.top + rect.height,
    x: rect.left,
    y: rect.top,
    toJSON: () => {},
  });
  return svg;
}

describe('clientToImage', () => {
  it('maps center of rect to center of viewBox', () => {
    const svg = makeSvg({ left: 0, top: 0, width: 100, height: 100 });
    const vb = { width: 1000, height: 1000 };
    const result = clientToImage(svg, 50, 50, vb);
    expect(result.x).toBeCloseTo(500);
    expect(result.y).toBeCloseTo(500);
  });

  it('handles non-zero rect offset correctly', () => {
    const svg = makeSvg({ left: 200, top: 100, width: 400, height: 300 });
    const vb = { width: 800, height: 600 };
    // client point at rect origin → image (0,0)
    const origin = clientToImage(svg, 200, 100, vb);
    expect(origin.x).toBeCloseTo(0);
    expect(origin.y).toBeCloseTo(0);
    // client point at rect center → image center
    const center = clientToImage(svg, 400, 250, vb);
    expect(center.x).toBeCloseTo(400);
    expect(center.y).toBeCloseTo(300);
  });

  it('returns (0,0) for zero-width rect without NaN', () => {
    const svg = makeSvg({ left: 0, top: 0, width: 0, height: 0 });
    const vb = { width: 1000, height: 1000 };
    const result = clientToImage(svg, 50, 50, vb);
    expect(result.x).toBe(0);
    expect(result.y).toBe(0);
    expect(Number.isNaN(result.x)).toBe(false);
    expect(Number.isNaN(result.y)).toBe(false);
  });

  it('returns (0,0) for zero-width viewBox without NaN', () => {
    const svg = makeSvg({ left: 0, top: 0, width: 100, height: 100 });
    const vb = { width: 0, height: 0 };
    const result = clientToImage(svg, 50, 50, vb);
    expect(result.x).toBe(0);
    expect(result.y).toBe(0);
    expect(Number.isNaN(result.x)).toBe(false);
    expect(Number.isNaN(result.y)).toBe(false);
  });

  it('extrapolates points outside the displayed rect without clamping', () => {
    const svg = makeSvg({ left: 20, top: 10, width: 200, height: 100 });
    const vb = { width: 1000, height: 500 };

    const result = clientToImage(svg, 0, 130, vb);

    expect(result.x).toBeCloseTo(-100);
    expect(result.y).toBeCloseTo(600);
  });
});

describe('imageToClient / round-trip', () => {
  it('imageToClient(clientToImage(svg, x, y, vb), vb) round-trips within 1e-9', () => {
    const svg = makeSvg({ left: 50, top: 30, width: 640, height: 480 });
    const vb = { width: 1280, height: 960 };
    const clientX = 250;
    const clientY = 180;
    const imgPt = clientToImage(svg, clientX, clientY, vb);
    const back = imageToClient(svg, imgPt.x, imgPt.y, vb);
    expect(Math.abs(back.x - clientX)).toBeLessThan(1e-9);
    expect(Math.abs(back.y - clientY)).toBeLessThan(1e-9);
  });

  it('clientToImage(imageToClient(svg, x, y, vb), vb) round-trips image coordinates', () => {
    const svg = makeSvg({ left: 80, top: 45, width: 320, height: 240 });
    const vb = { width: 1280, height: 960 };
    const imgX = 987.5;
    const imgY = 123.25;

    const clientPt = imageToClient(svg, imgX, imgY, vb);
    const back = clientToImage(svg, clientPt.x, clientPt.y, vb);

    expect(Math.abs(back.x - imgX)).toBeLessThan(1e-9);
    expect(Math.abs(back.y - imgY)).toBeLessThan(1e-9);
  });

  it('returns (0,0) from imageToClient for zero rect or zero viewBox guards', () => {
    const zeroRectSvg = makeSvg({ left: 10, top: 20, width: 0, height: 100 });
    const regularSvg = makeSvg({ left: 10, top: 20, width: 200, height: 100 });

    expect(imageToClient(zeroRectSvg, 20, 30, { width: 100, height: 100 })).toEqual({
      x: 0,
      y: 0,
    });
    expect(imageToClient(regularSvg, 20, 30, { width: 0, height: 100 })).toEqual({
      x: 0,
      y: 0,
    });
  });
});
