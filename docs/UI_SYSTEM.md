# Clinic Codex UI system

This project uses a flat, workstation-style UI across Workspace, Annotation, and Admin. The goal is to keep every page on one continuous surface: no nested cards, no rounded main panels, no outer page padding, and no decorative separation that makes panes feel like disconnected widgets.

## Foundations

- Global tokens live in `frontend/src/styles/app-theme.css`.
- Reusable recipes live in `frontend/src/components/ui/ui-primitives.css`.
- React admin primitives live in `frontend/src/components/ui/AdminPrimitives.tsx`.
- Page-specific ownership belongs in CSS modules, with transitional global selectors scoped by an owner module.

## Surface rules

1. Page shells use the app background directly.
2. Panels and cards are flat: `border-radius: 0`, no broad box shadow, and no outer page margin/padding.
3. Separators are explicit 1px dividers only when they clarify resize/scroll ownership.
4. Empty states use `.ui-empty-state`: transparent, borderless, square corners.
5. Real media never sits on placeholder paper/tile art. Fallback glyph art is allowed only when no real image exists.

## Scroll ownership

- The page root should not fight inner panes for scroll.
- Long lists own their scroll: review queue, dataset class list, dataset gallery, training console/support details.
- Use the global scrollbar styling from `frontend/src/index.css`; do not create per-component scrollbar widgets.
- Prefer `min-height: 0` on grid/flex children that own scroll.

## Controls and typography

- Buttons and pills should use `ActionButton`, `PillButton`, or the `.ui-*` recipes instead of ad-hoc classes.
- Keep action weight moderate: primary actions can be visible without oversized text or heavy font weight.
- Text uses the app font stack from `app-theme.css`; avoid browser-default `system-ui` declarations in component CSS.
- Technical bbox/metadata belongs in compact captions or audit details, not dominant body UI.

## Route/tab contract

- Admin tabs are routable: `/admin/annotations/review`, `/dataset`, `/training`.
- Tab state must be reflected in URL for direct QA/playtest entry points.
- Workspace-to-Annotation handoff must preserve selected analysis and element query parameters.

## Validation contract

Run these before claiming UI-system changes are done:

```bash
cd frontend
npm test -- --run src/components/ui/AdminPrimitives.test.tsx src/test/cssBudgets.test.ts src/test/themeTokens.test.ts
npm test -- --run src/pages/AdminAnnotationsPage.test.tsx src/pages/adminAnnotations/model.test.ts src/pages/adminAnnotations/ReferenceGlyphArt.test.tsx
npm run lint
npm run build
npx playwright test --workers=1
cd ..
backend/.venv/bin/python -m pytest backend/tests/test_annotation_review.py backend/tests/test_admin_training_routes.py -q
```

The visual smoke screenshots for the latest admin review/dataset pass are kept under `frontend/.omx-artifacts/` and are not source-controlled.
