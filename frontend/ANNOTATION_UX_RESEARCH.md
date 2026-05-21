# Annotation manual UX research

Date: 2026-05-21
Scope: research only. No admin page, auth/roles, backend migration, or broad manual-edit redesign is implemented in this pass.

## Current pain points

- Some images contain 20+ segmentations, so a vertical card list makes selection and naming slower.
- Element creation/renaming now works, but the user has to open each card and scan manually.
- The current annotator-side status is now framed as “submitted”; it is not final admin approval.
- A later training pipeline should consume only admin-approved annotations, but that requires a separate backend/admin PRD.

## Reference patterns

### CVAT

- CVAT exposes an objects sidebar listing the objects in the current frame and includes a filter input for narrowing the object list: <https://docs.cvat.ai/docs/manual/basics/cvat-annotation-interface/objects-sidebar/>
- CVAT’s filter utility supports object filtering by properties such as label/type/shape, and recent filters can be reused: <https://docs.cvat.ai/docs/annotation/manual-annotation/utilities/filter/>

Takeaway: for dense images, the object list should become searchable/filterable instead of only scrollable.

### Label Studio

- Label Studio documents hotkeys as a productivity feature for faster labeling and allows custom editor keymaps through configuration: <https://labelstud.io/guide/hotkeys>

Takeaway: keyboard navigation and mode switching are high-value once manual segmentation is frequent.

### Roboflow Annotate

- Roboflow documents shortcuts for switching between drag/select and drawing tools, and its docs describe review mode as an interface where a reviewer can approve or reject annotations: <https://docs.roboflow.com/annotate/use-roboflow-annotate/keyboard-shortcuts>

Takeaway: separate “annotator submitted” from “reviewer approved”; do not overload one status for both.

## Recommended UX backlog

### 1. Short-term, low-risk

1. Add a compact object list mode with columns: index, name, status, confidence, bbox size.
2. Add list filtering by typed label/name and status: all, draft, submitted, unnamed.
3. Add next/previous element shortcuts and make selected object auto-scroll into view.
4. Keep the current fuzzy name combobox, but expose it prominently for the selected element.
5. Add visual affordances for unnamed elements and draft/submitted counts.

### 2. Medium-term editing improvements

1. Add a command-palette style “go to element / rename / create label” interaction.
2. Support multi-select actions: submit selected, draft selected, delete selected.
3. Add keyboard shortcuts for draw/select, submit selected, focus name input, delete, next/previous.
4. Add a small context preview/minimap for the selected element so users can keep orientation on dense images.

### 3. Admin/training gate, separate project

1. Introduce backend review state distinct from annotator submission: submitted, approved, rejected.
2. Add admin review page/queue with approve/reject and bulk approve.
3. Make export/retraining consume only admin-approved annotations.
4. Add auth/roles or a simpler local admin gate depending on deployment needs.

## Proposed first implementation after this pass

If the next scope is still frontend-only, start with a filterable compact object list plus next/previous shortcuts. It addresses the “20+ segmentations” pain without changing backend storage or training semantics.
