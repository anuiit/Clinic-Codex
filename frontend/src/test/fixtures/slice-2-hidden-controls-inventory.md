# Slice 2 hidden controls inventory

Purpose: every `display:none` in `AdminAnnotationsPage.css` is inventoried before Slice 2 implementation. This slice must not add hidden real controls/data without a documented alternate visible path.

| Current selector | Category | Justification / action for Slice 2 | Coverage |
|---|---|---|---|
| `.admin-alert-stack:empty` | empty container | Safe: hides only an empty alert wrapper. Keep. | DOM state; no user data. |
| `.admin-command-actions` in Visual-Ralph reference block | real controls, duplicate/conflict | Remove/override conflict by deleting old hide rule; final command actions must remain visible. | Admin tests and screenshots. |
| `.admin-row-diagnostic` | row diagnostics | Hidden for reference table density. Keep only if row status/class/size remain visible; note in visual notes. | Triage row tests and screenshots. |
| `.admin-tab-stack .admin-workflow-steps`, `.admin-tab-stack > .ui-metric-strip` | summary/metrics | Hidden to match training mock; existing summary data is still available in preflight/checks/console. Keep but document. | Training tests and screenshots. |
| `.admin-decision-header .ui-text-eyebrow`, `.ui-text-caption` | duplicated captions | Hidden because compact title/meta remain visible. Keep and document. | Triage inspector test. |
| `.admin-inspector-grid > ... nav/ui-alert/audit` | secondary controls/details | Risky but pre-existing for visual parity; keep only if primary actions/navigation remain visible/clickable. Document. | Navigation/action tests. |
| responsive `.admin-action-bar .admin-keys` | decorative keyboard hint | Safe on narrow layouts; not a real control. Keep. | Desktop screenshot unaffected. |
| `.admin-inspector-flags` | flags | Hidden for reference density; split/trainable flags are duplicated in dataset/status badges and meta. Keep and document. | Inspector screenshots. |
| training block hides `.ui-alert`, `.admin-class-summary`, `.admin-launch-summary`, `.admin-training-job-panel`, `.admin-audit-details` | summaries/job details | Later final block restores most via `display:block`. Slice 2 must avoid new conflicts and ensure job/preflight/console remain visible. | Training tests and screenshots. |

Implementation rule: if this slice touches a hidden selector, visual notes must state whether it is empty/decorative/duplicated or a real-control risk with alternate path.
