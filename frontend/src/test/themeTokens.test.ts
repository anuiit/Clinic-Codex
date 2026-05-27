import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const frontendRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');

const migratedThemeFiles = [
  'src/index.css',
  'src/components/MainImagePanel.tsx',
  'src/components/WorkspaceHistoryPanel.tsx',
  'src/pages/WorkspacePage.tsx',
  'src/pages/AnnotationPage.tsx',
  'src/pages/workspace/WorkspaceDetectedPanel.tsx',
  'src/pages/workspace/WorkspaceHeader.tsx',
  'src/pages/workspace/WorkspaceEmptyState.tsx',
  'src/pages/workspace/WorkspaceUploadModal.tsx',
  'src/pages/AnnotationAnalyzerToolbar.tsx',
  'src/pages/AnnotationElementList.tsx',
  'src/pages/AnnotationPageChrome.tsx',
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

describe('semantic theme token contract', () => {
  it('keeps migrated workspace and annotation chrome on semantic color tokens/classes', () => {
    const violations = migratedThemeFiles.flatMap(findViolations);

    expect(violations).toEqual([]);
  });

  it('keeps the annotation rename suggestion menu on a fully opaque theme surface', () => {
    const css = readFileSync(resolve(frontendRoot, 'src/index.css'), 'utf8');

    expect(css).toMatch(
      /\.annotation-name-combobox__menu\.ui-panel\s*{[^}]*opacity:\s*1;[^}]*background:\s*var\(--app-bg-soft\);/s,
    );
  });
});
