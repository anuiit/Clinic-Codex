import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { basename, resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

const frontendRoot = resolve(__dirname, '..');

type OwnershipEntry = {
  selector: string;
  ownerModule: string;
  importOwner: string;
  requiresRule?: boolean;
};

type OwnershipContract = {
  routeShell: string[];
  ownedSelectors: OwnershipEntry[];
};

const adminModulePaths = [
  'pages/adminAnnotations/AdminHeader.module.css',
  'pages/adminAnnotations/AdminShared.module.css',
  'pages/adminAnnotations/DatasetTab.module.css',
  'pages/adminAnnotations/ReviewEditor.module.css',
  'pages/adminAnnotations/ReviewTab.module.css',
  'pages/adminAnnotations/TrainingTab.module.css',
];

const nonAdminModulePaths = [
  'components/AnalyzerToolbar.module.css',
  'components/AppChrome.module.css',
  'components/ImageBBoxStage/ImageBBoxStage.module.css',
  'components/LoadingSkeleton.module.css',
  'components/MainImagePanel.module.css',
  'components/SidebarChrome.module.css',
  'pages/workspace/WorkspaceChrome.module.css',
] as const;

const nonAdminOwnerConsumption = [
  {
    module: 'components/AnalyzerToolbar.module.css',
    consumer: 'components/AnalyzerToolbar.tsx',
    importPath: './AnalyzerToolbar.module.css',
  },
  {
    module: 'components/AppChrome.module.css',
    consumer: 'App.tsx',
    importPath: './components/AppChrome.module.css',
  },
  {
    module: 'components/ImageBBoxStage/ImageBBoxStage.module.css',
    consumer: 'components/ImageBBoxStage/ImageBBoxStage.tsx',
    importPath: './ImageBBoxStage.module.css',
  },
  {
    module: 'components/ImageBBoxStage/ImageBBoxStage.module.css',
    consumer: 'components/ImageBBoxStage/ImageBBoxToolbar.tsx',
    importPath: './ImageBBoxStage.module.css',
  },
  {
    module: 'components/LoadingSkeleton.module.css',
    consumer: 'pages/adminAnnotations/shared.tsx',
    importPath: '../../components/LoadingSkeleton.module.css',
  },
  {
    module: 'components/LoadingSkeleton.module.css',
    consumer: 'pages/workspace/WorkspaceDetectedPanel.tsx',
    importPath: '../../components/LoadingSkeleton.module.css',
  },
  {
    module: 'components/MainImagePanel.module.css',
    consumer: 'components/MainImagePanel.tsx',
    importPath: './MainImagePanel.module.css',
  },
  {
    module: 'components/SidebarChrome.module.css',
    consumer: 'components/WorkspaceHistoryPanel.tsx',
    importPath: './SidebarChrome.module.css',
  },
  {
    module: 'components/SidebarChrome.module.css',
    consumer: 'pages/workspace/WorkspaceDetectedPanel.tsx',
    importPath: '../../components/SidebarChrome.module.css',
  },
  {
    module: 'components/SidebarChrome.module.css',
    consumer: 'pages/annotation/AnnotationSelectedInspector.tsx',
    importPath: '../../components/SidebarChrome.module.css',
  },
  {
    module: 'pages/workspace/WorkspaceChrome.module.css',
    consumer: 'pages/WorkspacePage.tsx',
    importPath: './workspace/WorkspaceChrome.module.css',
  },
  {
    module: 'pages/workspace/WorkspaceChrome.module.css',
    consumer: 'pages/workspace/WorkspaceHeader.tsx',
    importPath: './WorkspaceChrome.module.css',
  },
  {
    module: 'pages/workspace/WorkspaceChrome.module.css',
    consumer: 'pages/workspace/WorkspaceDetectedPanel.tsx',
    importPath: './WorkspaceChrome.module.css',
  },
  {
    module: 'pages/workspace/WorkspaceChrome.module.css',
    consumer: 'components/WorkspaceHistoryPanel.tsx',
    importPath: '../pages/workspace/WorkspaceChrome.module.css',
  },
] as const;

const movedNonAdminSelectors = [
  '.main-image-panel',
  '.main-image-panel--workspace',
  '.main-image-panel--annotation',
  '.main-image-panel__header',
  '.main-image-panel__eyebrow',
  '.main-image-panel__title',
  '.main-image-panel__header-meta',
  '.main-image-panel__stage',
  '.main-image-panel__transform',
  '.main-image-panel__toolbar',
  '.main-image-panel__toolbar--bottom-center',
  '.main-image-panel__controls',
  '.main-image-panel__controls--bottom-center',
  '.main-image-panel__footer',
  '.image-stage-frame',
  '.image-stage-grid',
  '.image-stage-scrollbar',
  '.analyzer-toolbar',
  '.analyzer-toolbar__button--idle',
  '.analyzer-toolbar__button--active',
  '.analyzer-toolbar__icon-slot',
  '.analyzer-toolbar__label-slot',
  '.analyzer-toolbar__separator',
  '.workspace-analyzer-toolbar__zoom-label',
  '.annotation-analyzer-toolbar__zoom-label',
  '.image-bbox-stage',
  '.image-bbox-stage__stage',
  '.image-bbox-stage__transform',
  '.image-bbox-stage__image',
  '.image-bbox-toolbar',
  '.app-shell',
  '.app-sidebar',
  '.app-sidebar__header',
  '.sidebar-shell',
  '.sidebar-header',
  '.sidebar-body',
  '.workspace-trust-skeleton',
  '.workspace-page',
  '.app-header',
  '.app-header__icon',
  '.app-header__title',
  '.workspace-detected-grid',
  '.selection-card',
  '.selection-card--active',
] as const;

const deletedLegacySelectors = [
  '.theme-toggle',
  '.theme-toggle__icon',
  '.theme-toggle__label',
  '.annotation-action-button',
  '.annotation-action-button--ghost',
  '.annotation-action-button--primary',
  '.annotation-action-button--success',
  '.annotation-action-button--ready',
  '.annotation-action-button--danger',
  '.annotation-status-chip',
  '.annotation-status-chip--draft',
  '.annotation-status-chip--validated',
  '.annotation-status-chip--rejected',
] as const;


const uiPrimitiveSelectors = [
  '.ui-panel',
  '.ui-section',
  '.ui-divider',
  '.ui-row',
  '.ui-row--hover',
  '.ui-row--active',
  '.ui-icon-button',
  '.ui-action-ghost',
  '.ui-action-primary',
  '.ui-action-danger',
  '.ui-input',
  '.ui-select',
  '.ui-metric-strip',
  '.ui-metric-strip--bar',
  '.ui-metric-strip--cards',
  '.ui-metric',
  '.ui-metric__label',
  '.ui-metric__value',
  '.ui-metric__helper',
  '.ui-metric--ready',
  '.ui-metric--danger',
  '.ui-tabs-shell',
  '.ui-tabs-shell--pill',
  '.ui-tabs',
  '.ui-tabs--underline',
  '.ui-tabs--pill',
  '.ui-tab',
  '.ui-tab--active',
  '.ui-tab-label',
  '.ui-empty-state',
  '.ui-crop-shell',
  '.ui-progress-track',
  '.ui-progress-value',
  '.ui-progress-value--accent',
  '.ui-progress-value--danger',
  '.ui-progress-value--ready',
  '.ui-chip',
  '.ui-chip--accent',
  '.ui-chip--danger',
  '.ui-chip--ready',
  '.ui-text-eyebrow',
  '.ui-text-caption',
  '.ui-text-meta',
  '.ui-text-body-sm',
  '.ui-title-sm',
  '.ui-title-md',
  '.ui-alert',
  '.ui-alert--accent',
  '.ui-alert--danger',
  '.ui-alert--success',
  '.ui-status-dot--accent',
  '.ui-status-dot--danger',
  '.ui-card-selected',
  '.ui-scrollbar',
] as const;

const annotationChromeSelectors = [
  '.annotation-app',
  '.annotation-topbar',
  '.annotation-panel',
  '.annotation-stage-frame',
  '.annotation-stage',
  '.annotation-rail',
  '.annotation-inspector',
  '.annotation-selected-inspector',
  '.annotation-selected-inspector__body',
  '.annotation-name-combobox',
  '.annotation-name-combobox__menu',
  '.annotation-list-controls',
  '.annotation-crop',
  '.annotation-resize-handle',
  '.annotation-draw-preview',
  '.annotation-index-badge',
  '.annotation-index-badge--draft',
  '.annotation-index-badge--focused',
  '.annotation-index-badge--validated',
  '.annotation-card',
  '.annotation-card-selected',
  '.annotation-scrollbar',
  '.annotation-toast--ok',
  '.annotation-toast--error',
] as const;

const annotationOwnerConsumers = [
  {
    path: 'pages/AnnotationPage.tsx',
    importPath: './annotation/AnnotationChrome.module.css',
  },
  {
    path: 'pages/AnnotationPageChrome.tsx',
    importPath: './annotation/AnnotationChrome.module.css',
  },
  {
    path: 'pages/AnnotationElementList.tsx',
    importPath: './annotation/AnnotationChrome.module.css',
  },
  {
    path: 'pages/annotation/AnnotationSelectedInspector.tsx',
    importPath: './AnnotationChrome.module.css',
  },
  {
    path: 'pages/annotation/ElementNameCombobox.tsx',
    importPath: './AnnotationChrome.module.css',
  },
  {
    path: 'pages/annotation/AnnotationToast.tsx',
    importPath: './AnnotationChrome.module.css',
  },
] as const;

function readSource(relativePath: string): string {
  return readFileSync(resolve(frontendRoot, relativePath), 'utf8');
}

function productionSources(dir: string): string[] {
  const absolute = resolve(frontendRoot, dir);
  return readdirSync(absolute).flatMap((entry) => {
    const child = dir ? `${dir}/${entry}` : entry;
    const childAbsolute = resolve(frontendRoot, child);
    if (statSync(childAbsolute).isDirectory()) {
      return productionSources(child);
    }
    return /\.(tsx|ts|css)$/.test(entry) && !/(\.test\.|\/test\/)/.test(child) ? [child] : [];
  });
}

function countLines(relativePath: string): number {
  const source = readSource(relativePath);
  return source.split(/\r?\n/).length - (source.endsWith('\n') ? 1 : 0);
}

function adminCssUnion(): string {
  return [readSource('pages/AdminAnnotationsPage.css'), ...adminModulePaths.map(readSource)].join('\n');
}

function selectorBoundary(selector: string): RegExp {
  return new RegExp(selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '(?![\\w-])');
}

function selectorPreludes(css: string): string[] {
  const preludes: string[] = [];
  const withoutComments = css.replace(/\/\*[\s\S]*?\*\//g, '');
  const matcher = /(?:^|[{}])([^{}@][^{}]*)\{/g;
  let match: RegExpExecArray | null;
  while ((match = matcher.exec(withoutComments))) {
    const prelude = match[1].trim();
    if (prelude && prelude !== ':global' && prelude !== '.owner' && /\.[-_a-zA-Z]/.test(prelude)) {
      preludes.push(prelude);
    }
  }
  return preludes;
}

function selectorsInPrelude(prelude: string): string[] {
  return [...prelude.matchAll(/\.-?[_a-zA-Z]+[_a-zA-Z0-9-]*/g)].map((match) => match[0]);
}

function expectedImportFor(modulePath: string): string {
  return modulePath === 'adminAnnotations/AdminShared.module.css'
    ? './adminAnnotations/AdminShared.module.css'
    : `./${basename(modulePath)}`;
}

const ownership = JSON.parse(
  readSource('test/fixtures/slice-4-selector-ownership.json'),
) as OwnershipContract;

const ownerBySelector = new Map(
  ownership.ownedSelectors.map((entry) => [entry.selector, entry]),
);

function firstOwnedSelector(prelude: string): OwnershipEntry | 'route' | null {
  for (const selector of selectorsInPrelude(prelude)) {
    if (ownership.routeShell.includes(selector)) {
      return 'route';
    }
    const owner = ownerBySelector.get(selector);
    if (owner) {
      return owner;
    }
  }
  return null;
}

describe('CSS budget guardrails', () => {
  it('keeps legacy App.css deleted', () => {
    expect(existsSync(resolve(frontendRoot, 'App.css'))).toBe(false);
  });

  it('keeps global bootstrap CSS tiny', () => {
    expect(countLines('index.css')).toBeLessThanOrEqual(30);
  });

  it('keeps shared theme CSS token-focused after Slice 5 primitive extraction', () => {
    expect(countLines('styles/app-theme.css')).toBeLessThanOrEqual(230);
    expect(countLines('components/ui/ui-primitives.css')).toBeLessThanOrEqual(520);
    expect(countLines('styles/app-theme.css') + countLines('components/ui/ui-primitives.css')).toBeLessThanOrEqual(760);
  });

  it('keeps Slice 7 app-theme token/body focused and annotation CSS owned by annotation module', () => {
    const appTheme = readSource('styles/app-theme.css');
    const annotationModule = readSource('pages/annotation/AnnotationChrome.module.css');

    expect(countLines('pages/annotation/AnnotationChrome.module.css')).toBeLessThanOrEqual(260);
    expect(annotationModule).toMatch(/\.owner\s*{/);
    expect(selectorPreludes(appTheme)).toEqual(['.animate-slide-in-right']);

    for (const selector of annotationChromeSelectors) {
      expect(appTheme).not.toMatch(selectorBoundary(selector));
      expect(annotationModule).toMatch(selectorBoundary(selector));
    }

    for (const consumer of annotationOwnerConsumers) {
      const source = readSource(consumer.path);
      expect(source).toContain(consumer.importPath);
      expect(source).toContain('.owner');
    }

    const production = productionSources('')
      .filter((path) => !path.includes('test/fixtures'))
      .map(readSource)
      .join('\n');
    expect(production).not.toMatch(/text-app-muted|text-status-ready|bg-status-ready-soft|hover:bg-status-ready-soft|status-ready-(?:chip|badge)/);
  });

  it('keeps Slice 6 non-admin CSS owned by consumed modules', () => {
    const appTheme = readSource('styles/app-theme.css');

    for (const modulePath of nonAdminModulePaths) {
      expect(existsSync(resolve(frontendRoot, modulePath))).toBe(true);
      expect(countLines(modulePath)).toBeLessThanOrEqual(260);
      expect(readSource(modulePath)).toMatch(/\.owner\s*{/);
    }

    for (const selector of movedNonAdminSelectors) {
      expect(appTheme).not.toMatch(selectorBoundary(selector));
      const ownerMatches = nonAdminModulePaths.filter((modulePath) =>
        readSource(modulePath).match(selectorBoundary(selector)),
      );
      expect(ownerMatches.length, `${selector} owner count`).toBeGreaterThan(0);
    }

    for (const entry of nonAdminOwnerConsumption) {
      const consumer = readSource(entry.consumer);
      expect(consumer).toContain(entry.importPath);
      expect(consumer).toContain('.owner');
    }
  });

  it('keeps Slice 6 replaced legacy controls out of production JSX and CSS', () => {
    const appTheme = readSource('styles/app-theme.css');
    for (const selector of deletedLegacySelectors) {
      expect(appTheme).not.toMatch(selectorBoundary(selector));
    }

    const production = productionSources('')
      .filter((path) => !path.includes('test/fixtures'))
      .map(readSource)
      .join('\\n');
    expect(production).not.toMatch(/theme-toggle(?:__|--|\\b)/);
    expect(production).not.toContain('annotation-action-button');
    expect(production).not.toContain('annotation-status-chip');
  });

  it('keeps admin route CSS shell-sized and admin modules budgeted', () => {
    expect(countLines('pages/AdminAnnotationsPage.css')).toBeLessThanOrEqual(200);
    for (const modulePath of adminModulePaths) {
      expect(existsSync(resolve(frontendRoot, modulePath))).toBe(true);
      expect(countLines(modulePath)).toBeLessThanOrEqual(350);
    }
    const total = countLines('pages/AdminAnnotationsPage.css') + adminModulePaths.reduce(
      (sum, modulePath) => sum + countLines(modulePath),
      0,
    );
    expect(total).toBeLessThanOrEqual(1300);
  });

  it('keeps admin colors owned by shared tokens outside reference-art exemptions', () => {
    const productionChrome = adminCssUnion();
    const adminConsole = readSource('pages/AdminAnnotationsPage.css');

    expect(adminConsole).toMatch(/background:\s*var\(--app-bg\)/);
    expect(adminConsole).toMatch(/color:\s*var\(--text-main\)/);
    expect(adminConsole).not.toMatch(/#[0-9a-f]{3,8}\b/i);
    expect(adminConsole).not.toMatch(/rgba?\(/i);
    expect(adminConsole).not.toMatch(/hsla?\(/i);
    expect(adminConsole).not.toMatch(/--(?:app|bg|surface|border|text|accent|status|danger|warning|violet|orange)[\w-]*\s*:/i);
    expect(productionChrome).not.toMatch(/#[0-9a-f]{3,8}\b/i);
    expect(productionChrome).not.toMatch(/rgba?\(/i);
  });

  it('keeps admin chrome flat outside reference-art exemptions', () => {
    const productionChrome = adminCssUnion();
    expect(productionChrome).not.toContain('0 22px');
    expect(productionChrome).not.toContain('0 30px');
    expect(productionChrome).not.toContain('0 -18px');
    expect(productionChrome).not.toContain('var(--shadow-soft)');
  });

  it('keeps admin typography below heavy display weights', () => {
    const css = adminCssUnion();
    const routeSources = [
      'AdminHeader.tsx',
      'DatasetTab.tsx',
      'ReviewEditor.tsx',
      'ReviewInspector.tsx',
      'ReviewTab.tsx',
      'TrainingHelpers.tsx',
      'TrainingTab.tsx',
      'ReferenceGlyphArt.tsx',
      'shared.tsx',
    ].map((file) => readSource(`pages/adminAnnotations/${file}`)).join('\n');

    expect(css).not.toMatch(/font-weight:\s*(?:8|9)\d{2}/);
    expect(routeSources).not.toContain('font-bold');
  });

  it('keeps Button/Pill/Badge primitives and admin route free of legacy page hooks', () => {
    const sources = [
      readSource('components/ui/AdminPrimitives.tsx'),
      adminCssUnion(),
      ...[
        'AdminHeader.tsx',
        'DatasetTab.tsx',
        'ReviewEditor.tsx',
        'ReviewInspector.tsx',
        'ReviewTab.tsx',
        'TrainingHelpers.tsx',
        'TrainingTab.tsx',
        'ReferenceGlyphArt.tsx',
        'shared.tsx',
      ].map((file) => readSource(`pages/adminAnnotations/${file}`)),
    ].join('\n');

    expect(sources).not.toContain('pill-reference');
    expect(sources).not.toContain('ui-action-');
    expect(sources).not.toContain('ui-chip');
  });

  it('documents hidden admin controls before relying on display none', () => {
    const inventory = readSource('test/fixtures/slice-2-hidden-controls-inventory.md');
    expect(inventory).toContain('.admin-alert-stack:empty');
    expect(inventory).toContain('.admin-row-diagnostic');
    expect(inventory).toContain('.admin-workflow-steps');
    expect(inventory).toContain('.admin-decision-header');
    expect(inventory).toContain('.admin-inspector-flags');
    expect(inventory).toContain('.admin-training-job-panel');
    expect(adminCssUnion()).not.toMatch(/\.admin-command-actions\s*\{[^}]*display:\s*none/s);
  });

  it('keeps hard-coded fallback-art colors isolated to ReferenceGlyphArt', () => {
    const hardColor = /#[0-9a-f]{3,8}\b|rgba?\(|hsla?\(/i;
    const productionSources = [
      'components/ui/AdminPrimitives.tsx',
      'pages/adminAnnotations/AdminHeader.tsx',
      'pages/adminAnnotations/DatasetTab.tsx',
      'pages/adminAnnotations/ReviewEditor.tsx',
      'pages/adminAnnotations/ReviewInspector.tsx',
      'pages/adminAnnotations/ReviewTab.tsx',
      'pages/adminAnnotations/TrainingHelpers.tsx',
      'pages/adminAnnotations/TrainingTab.tsx',
      'pages/adminAnnotations/shared.tsx',
    ].map(readSource).join('\n');
    const referenceArt = readSource('pages/adminAnnotations/ReferenceGlyphArt.tsx');

    expect(productionSources).not.toMatch(hardColor);
    expect(referenceArt).toContain('reference-art-exempt');
    expect(referenceArt).toMatch(hardColor);
  });

  it('keeps migrated reference-art selectors out of route and module stylesheets', () => {
    const css = adminCssUnion();
    expect(css).not.toContain('.tile-img-reference');
    expect(css).not.toContain('.admin-row-thumb');
    expect(css).not.toContain('clip-path: polygon');
    expect(css).not.toContain('slice-2-exempt: reference-art');
  });

  it('keeps Slice 5 ui primitive CSS owned outside app-theme', () => {
    const appTheme = readSource('styles/app-theme.css');
    const primitives = readSource('components/ui/ui-primitives.css');

    for (const selector of uiPrimitiveSelectors) {
      expect(primitives).toMatch(selectorBoundary(selector));
    }

    const appThemeRulePreludes = selectorPreludes(appTheme);
    const remainingUiPreludes = appThemeRulePreludes.filter((prelude) =>
      selectorsInPrelude(prelude).some((selector) => selector.startsWith('.ui-')),
    );
    expect(remainingUiPreludes).toEqual([]);
    expect(appTheme).not.toMatch(/\.annotation-card-selected,\s*\n\.selection-card--active,\s*\n\.ui-card-selected\s*\{/);

    const appSource = readSource('App.tsx');
    expect(appSource.indexOf("./styles/app-theme.css")).toBeGreaterThanOrEqual(0);
    expect(appSource.indexOf("./components/ui/ui-primitives.css")).toBeGreaterThan(
      appSource.indexOf("./styles/app-theme.css"),
    );
  });

  it('keeps Slice 4 selector ownership machine-checkable', () => {
    const routeCss = readSource('pages/AdminAnnotationsPage.css');
    for (const routeSelector of ownership.routeShell) {
      expect(routeCss).toMatch(selectorBoundary(routeSelector));
    }

    for (const entry of ownership.ownedSelectors) {
      expect(routeCss).not.toMatch(selectorBoundary(entry.selector));
      const ownerSource = readSource(`pages/${entry.ownerModule}`);
      if (entry.requiresRule === false) {
        expect(ownerSource).not.toMatch(selectorBoundary(entry.selector));
        continue;
      }
      expect(ownerSource).toMatch(selectorBoundary(entry.selector));
    }

    expect(readSource('pages/AdminAnnotationsPage.css')).toContain("AdminAnnotationsPage.css");
    for (const entry of ownership.ownedSelectors) {
      const importOwner = readSource(entry.importOwner);
      expect(importOwner).toContain(expectedImportFor(entry.ownerModule));
    }
  });


  it('captures first global selectors inside CSS at-rules for ownership checks', () => {
    const sharedPreludes = selectorPreludes(readSource('pages/adminAnnotations/AdminShared.module.css'));
    expect(sharedPreludes).toContain(':global(.ui-metric-strip--bar)');
    expect(sharedPreludes).toContain(':global(.ui-metric-strip--bar .ui-metric)');
  });

  it('keeps CSS rule ownership aligned with the first owned selector in each module', () => {
    for (const modulePath of adminModulePaths) {
      const moduleKey = modulePath.replace('pages/', '');
      for (const prelude of selectorPreludes(readSource(modulePath))) {
        const owner = firstOwnedSelector(prelude);
        if (owner === null) {
          throw new Error(`Unowned selector prelude in ${modulePath}: ${prelude}`);
        }
        if (owner !== 'route') {
          expect(owner.ownerModule).toBe(moduleKey);
        }
      }
    }

    for (const prelude of selectorPreludes(readSource('pages/AdminAnnotationsPage.css'))) {
      expect(firstOwnedSelector(prelude)).toBe('route');
    }

    expect(adminCssUnion()).not.toContain('.admin-action-row');
  });
});
