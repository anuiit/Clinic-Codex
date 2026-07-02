# Worker 3 admin UI/UX audit — 2026-06-27

Task: admin UI/UX audit and safe high-impact polish candidates.

Scope and coordination note:
- This worker repaired a malformed assignment by creating/claiming team task `4` for the explicit Worker 3 scope embedded in original task `2`.
- No production/external side effects were performed.
- No `.omx/ultragoal` state was mutated.
- I did **not** edit `frontend/src/pages/AdminAnnotationsPage.tsx` because the assignment explicitly says to coordinate before editing shared `AdminAnnotationsPage`.

## Evidence inspected

- `frontend/src/pages/AdminAnnotationsPage.tsx`
  - Header/tab framing: lines 71-118.
  - Review filter/list/inspector split: lines 600-677.
  - Dataset filter/list/inspector split: lines 885-963.
  - Training guard, pre-action summary, launch controls, artifacts/config panels: lines 1078-1233.
  - Page-level loading/error/tab panels: lines 1284-1374.
- `frontend/src/pages/AdminAnnotationsPage.test.tsx`
  - Current admin render/filter/navigation/training regression coverage includes lines 136-180 and 271-430.
- `frontend/src/index.css`
  - Shared panel/section/input/empty-state primitives: lines 711-835.
  - Admin-specific tab/action styling: lines 1139-1207.

## Ranked UI/UX polish candidates

### P1 — Add filter recovery controls and visible result counts in Review + Dataset

Current behavior:
- Review filters can produce an empty list at `AdminAnnotationsPage.tsx:649-662`, but the inspector still receives `selectedRow` derived from all rows at `604-667`. The UI can show “No review elements match” beside an inspector for an element outside the active filters.
- Dataset has the same empty-state pattern at `941-954`, with selection derived from filtered rows at `922`, but no one-click recovery.

Impact:
- High operator friction during triage: after filtering, the obvious next action is to recover/reset filters, but there is no direct affordance.
- Lower accessibility clarity: there is no “N results” summary for screen-reader or keyboard users.

Safe implementation slice:
- In Review: show “X of Y elements shown” near filters; when empty, render a `Clear filters` button that resets status/class/search and pass `null` to `ReviewElementInspector` when `filteredRows.length === 0`.
- In Dataset: show “X of Y crops shown” and a `Clear filters` button that resets dataset status/class.
- Add tests in `AdminAnnotationsPage.test.tsx` for empty filter recovery and inspector clearing.

Risk/coordination:
- Requires editing shared `frontend/src/pages/AdminAnnotationsPage.tsx` and its test; coordinate with worker(s) touching admin behavior first.

### P2 — Add explicit live status for queue mutations and training refresh

Current behavior:
- Review/modify actions set `mutatingKey` and disable buttons, but there is no persistent live “Saving…”/“Approving…” status region in the inspector or page (`AdminAnnotationsPage.tsx:1310-1334`).
- Training polling errors are surfaced, but successful refresh/polling has no live affordance (`1048-1060`).

Impact:
- Operators cannot easily distinguish a slow backend from a missed click, especially on local filesystem-heavy crop regeneration.
- Screen-reader users get error alerts but not in-progress state changes.

Safe implementation slice:
- Add a small `aria-live="polite"` status line near page alerts, keyed from `mutatingKey`, `editingKey`, and training `starting/jobRunning`.
- Add regression tests that the status appears during pending mocked promises.

Risk/coordination:
- Shared `AdminAnnotationsPage.tsx` edit; no backend/API contract change.

### P3 — Improve training full-run safety affordance without changing backend guard

Current behavior:
- The training copy correctly states the default-disabled guard and registry/promotion flow (`1078-1132`, `1165-1220`).
- When launch is allowed, switching from dry run to full training immediately enables the Start training button (`1176-1209`) without an explicit confirmation acknowledgment.

Impact:
- Full training is local-only and guarded, but it is still a heavier action than dry-run and could be accidentally selected.

Safe implementation slice:
- Keep backend guard unchanged.
- Require a local UI checkbox such as “I understand this writes a candidate registry package” before enabling `Start training` when `dryRun === false`.
- Preserve disabled-by-default training boundary.

Risk/coordination:
- Shared page/test edit; needs quick product signoff because it adds one interaction step.

### P4 — Add keyboard shortcuts/roving focus hints for large review queues

Current behavior:
- Review and Dataset rows use `role="option"` buttons inside `role="listbox"` (`649-658`, `941-952`), but there is no arrow-key navigation or visible keyboard hint.

Impact:
- Long queue review is click-heavy and slower for keyboard-only operators.

Safe implementation slice:
- Lower-risk first pass: add visible hint copy (“Use Tab to move through rows; Previous/Next controls move within active filters”) and ensure focus ring is obvious.
- Higher-risk later pass: implement arrow-key roving focus for listbox options.

Risk/coordination:
- Hint-only change is low risk but still in shared page/CSS; roving focus is more involved and needs extra tests.

## Recommended next action

After coordination on shared `AdminAnnotationsPage`, implement P1 first. It is the highest-impact, lowest-risk UI polish: local React-only, no dependencies, no backend contract changes, and it directly improves the admin review/dataset workflows.
