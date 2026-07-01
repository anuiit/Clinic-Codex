import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const frontendRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');

const migratedThemeFiles = [
  'src/styles/app-theme.css',
  'src/components/AnalyzerToolbar.module.css',
  'src/components/AppChrome.module.css',
  'src/components/ImageBBoxStage/ImageBBoxStage.module.css',
  'src/components/LoadingSkeleton.module.css',
  'src/components/ui/ui-primitives.css',
  'src/components/MainImagePanel.module.css',
  'src/components/MainImagePanel.tsx',
  'src/components/SidebarChrome.module.css',
  'src/components/ThemeToggle.tsx',
  'src/components/WorkspaceHistoryPanel.tsx',
  'src/pages/WorkspacePage.tsx',
  'src/pages/AnnotationPage.tsx',
  'src/pages/workspace/WorkspaceDetectedPanel.tsx',
  'src/pages/workspace/WorkspaceHeader.tsx',
  'src/pages/workspace/WorkspaceChrome.module.css',
  'src/pages/workspace/WorkspaceEmptyState.tsx',
  'src/pages/workspace/WorkspaceUploadModal.tsx',
  'src/pages/AnnotationAnalyzerToolbar.tsx',
  'src/pages/AnnotationElementList.tsx',
  'src/pages/AnnotationPageChrome.tsx',
  'src/pages/annotation/AnnotationChrome.module.css',
  'src/pages/annotation/AnnotationSelectedInspector.tsx',
  'src/pages/annotation/ElementNameCombobox.tsx',
] as const;

const bannedThemeTokens = [
  /\btext-(amber|red|emerald|green|blue|purple|slate|gray|zinc|neutral)-/,
  /\bbg-(amber|red|emerald|green|blue|purple|slate|gray|zinc|neutral)-/,
  /\bborder-(amber|red|emerald|green|blue|purple|slate|gray|zinc|neutral)-/,
  /\bring-(amber|red|emerald|green|blue|purple|slate|gray|zinc|neutral)-/,
  /\boutline-(amber|red|emerald|green|blue|purple|slate|gray|zinc|neutral)-/,
  /\bshadow-(amber|red|emerald|green|blue|purple|slate|gray|zinc|neutral)-/,
  /\b(from|via|to)-(amber|red|emerald|green|blue|purple|slate|gray|zinc|neutral)-/,
  /\b(fill|stroke)=["']#[0-9a-f]/i,
  /\b(fill|stroke)=["']rgba\(/,
  /\b(fill|stroke)=["']var\(--/,
  /text-stone-/,
  /bg-stone-/,
  /border-stone-/,
  /text-white/,
  /bg-white/,
  /#ffffff/i,
  /#000000/i,
  /#0a0a0a/i,
  /#050505/i,
  /rgba\(255,\s*255,\s*255/i,
  /rgba\(0,\s*0,\s*0/i,
  /text-\[10px\]/,
  /text-\[11px\]/,
  /0\.625rem/,
] as const;

function findViolations(filePath: string) {
  const absolutePath = resolve(frontendRoot, filePath);
  return readFileSync(absolutePath, 'utf8')
    .split('\n')
    .flatMap((line, lineIdx) => {
      const matches = bannedThemeTokens
        .filter((pattern) => pattern.test(line))
        .map((pattern) => pattern.source);

      return matches.map((pattern) => `${filePath}:${lineIdx + 1}: ${pattern}: ${line.trim()}`);
    });
}

function themeBlock(theme: 'dark' | 'light') {
  const css = readFileSync(resolve(frontendRoot, 'src/styles/app-theme.css'), 'utf8');
  const pattern = theme === 'dark'
    ? /:root,\s*\n\[data-theme="dark"\]\s*{(?<body>[\s\S]*?)\n}/
    : /\[data-theme="light"\]\s*{(?<body>[\s\S]*?)\n}/;
  const match = css.match(pattern);
  if (!match?.groups?.body) throw new Error(`Missing ${theme} theme block`);
  return Object.fromEntries(
    [...match.groups.body.matchAll(/--([\w-]+):\s*(#[0-9a-f]{6})\s*;/gi)].map((token) => [
      token[1],
      token[2],
    ]),
  );
}

function relativeLuminance(hex: string) {
  const channel = (idx: number) => {
    const value = Number.parseInt(hex.slice(1 + idx * 2, 3 + idx * 2), 16) / 255;
    return value <= 0.03928 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * channel(0) + 0.7152 * channel(1) + 0.0722 * channel(2);
}

function contrastRatio(a: string, b: string) {
  const [lighter, darker] = [relativeLuminance(a), relativeLuminance(b)].sort((left, right) => right - left);
  return (lighter + 0.05) / (darker + 0.05);
}

describe('semantic theme token contract', () => {
  it('keeps migrated workspace and annotation chrome on semantic color tokens/classes', () => {
    const violations = migratedThemeFiles.flatMap(findViolations);

    expect(violations).toEqual([]);
  });

  it('keeps the annotation rename suggestion menu on a fully opaque theme surface', () => {
    const css = readFileSync(resolve(frontendRoot, 'src/pages/annotation/AnnotationChrome.module.css'), 'utf8');

    expect(css).toMatch(
      /:global\(\.annotation-name-combobox__menu\.ui-panel\)\s*{[^}]*opacity:\s*1;[^}]*background:\s*var\(--app-bg-soft\);/s,
    );
  });

  it('keeps solid semantic status foreground tokens readable in dark and light themes', () => {
    for (const theme of ['dark', 'light'] as const) {
      const tokens = themeBlock(theme);

      expect(contrastRatio(tokens['status-ready'], tokens['status-ready-on-solid'])).toBeGreaterThanOrEqual(4.5);
      expect(contrastRatio(tokens.danger, tokens['danger-on-solid'])).toBeGreaterThanOrEqual(4.5);
    }
  });
});
