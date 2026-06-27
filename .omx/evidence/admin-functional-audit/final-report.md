# Admin Dashboard Functional Audit — worker-2

**Date:** 2026-06-27
**Lane:** worker-2 (functional audit) of the current feature/admin audit
**Goal context:** Leader-owned Ultragoal `G001-answer-what-product-features-remain`
**Worktree HEAD:** `0a88ba8` (Add admin review and training workflow)
**Scope:** Admin dashboard functional audit — separate intentionally training-disabled-by-default
behavior from real defects; propose/fix safe blockers. Read-only audit; no source files changed.

---

## Verdict

**The admin dashboard is functionally complete. No functional defects were found.**

The "admin is not fully functional" perception traces entirely to one **intentional safety
boundary**: the Training-tab launcher is **disabled by default** and requires a local operator to
start the backend with `ENABLE_ADMIN_TRAINING_JOBS=1` from a loopback session. This is by design,
working as intended — not a defect.

---

## What was audited (capability → status)

| Capability | Surface | Status | Evidence |
|---|---|---|---|
| Review queue load (counts, diagnostics, local-only warning) | `GET /admin/annotations` | ✅ Works | probe #2–5, admin route tests |
| Approve element → becomes trainable | `POST .../{i}/review` | ✅ Works | probe #6–7 |
| Reject element → excluded from training | `POST .../{i}/review` | ✅ Works | probe #8 |
| Modify element (class/bbox) → regen crop, return to pending, old crop cleaned | `POST .../{i}/modify` | ✅ Works | probe #9–13 |
| Save & approve (single modify intent) | `POST .../{i}/modify` | ✅ Works | probe #14 |
| Image / crop media serving | `GET .../image`, `.../crop` | ✅ Works | probe #15–16 |
| Input validation (bad status 400, missing element 404) | review/modify routes | ✅ Works | probe #17–18 |
| Dataset tab buckets/filters/distribution | frontend (pure derive from queue) | ✅ Works | frontend unit tests |
| Training summary (data counts, params, paths, artifacts, registry health) | `GET /admin/training/summary` | ✅ Works | probe #19, #22 |
| Training launch guard — disabled by default | `summary` + `POST /admin/training/jobs` | ✅ Intentional | probe #20–21, #23 |
| Training launch guard — flag flips the gate | `summary` with `ENABLE_ADMIN_TRAINING_JOBS=1` | ✅ Works | probe #24–25 |
| Job polling until terminal status | frontend + `GET .../jobs/latest` | ✅ Works | frontend unit tests |

## Wiring confirmed end-to-end
- Blueprints registered: `backend/app/routes/__init__.py` registers `admin_annotations_bp` and `admin_training_bp`.
- Service container wires both stores: `backend/app/services/container.py` (`AnnotationReviewStore`, `AdminTrainingService`).
- Frontend client covers every endpoint: `frontend/src/services/api.ts`.
- Route mounted in SPA: `frontend/src/App.tsx` → `/admin/annotations`.

---

## Intentional vs defect (the key separation requested)

**Intentional (NOT a defect) — training disabled by default**
`AdminTrainingService.launch_allowed()` (`backend/services/training_jobs.py`) returns
`launch_allowed_for_request: false` with reason `disabled_by_default: set ENABLE_ADMIN_TRAINING_JOBS=1 ...`
whenever the flag is off. Additional loopback-only guards (`non_loopback_remote_addr`, `nonlocal_host`,
`nonlocal_origin`) and a `missing_retrain_script` check further gate launches. The frontend renders the
disabled state with an explicit, accessible explanation and a CLI alternative
(`bash scripts/retrain.sh --dry-run`). Probe evidence:
- Flag OFF → launch blocked, `POST /admin/training/jobs` returns **403** (guard holds).
- Flag ON (tmp root) → `disabled_by_default` reason disappears; only the expected `missing_retrain_script`
  remains. The real repo ships `scripts/retrain.sh`, so a genuine local + loopback run would be launchable.

**Defects:** none found in the admin dashboard.

---

## Non-blocking polish candidates (recommendations only — NOT fixed here)

These are minor hardening/UX ideas, none of them blockers. **All three touch the shared file
`frontend/src/pages/AdminAnnotationsPage.tsx`**, which is coordinated with the worker-3 UI/UX lane
(task-3: "coordinate before editing shared AdminAnnotationsPage"). Per scope rules, worker-2 did not
edit the shared file. Surfacing for leader/worker-3 prioritization:

1. **Client-side `batch_size` validation.** Clearing the field yields `Number('') === 0`, and
   out-of-range values are only rejected server-side, surfacing a generic error. A small inline
   validation (mirroring the existing `ElementEditor` bbox checks) would improve feedback. *(P3)*
2. **No admin e2e spec.** Admin has strong unit coverage but no Playwright flow under
   `frontend/tests/e2e/`. A read/approve/modify smoke would lock in the HTTP contract. *(P3)*
3. **No manual "Refresh queue" control.** The queue reloads after each mutation but cannot be
   manually refreshed if another local operator changes state. *(P4)*

---

## Verification evidence (commands + results)

Python: `backend/.venv/bin/python` (leader venv). Run from worktree repo root with `PYTHONPATH=<worktree>`.

1. **End-to-end admin probe (no mocks, real Flask test client):** `admin_probe.py` →
   **26/26 checks PASS** (`admin-e2e-probe-20260627T213935Z.log`). Exit 0.
2. **Backend admin route tests:** `pytest backend/tests/test_admin_annotation_routes.py
   backend/tests/test_admin_training_routes.py` → **28 passed** (`admin-backend-tests-...log`). Exit 0.
3. **Frontend admin unit tests** (source byte-identical to leader HEAD, run via leader node_modules):
   `vitest run src/pages/AdminAnnotationsPage.test.tsx` → **14 passed**
   (`admin-frontend-tests-...log`). Exit 0.
4. **Full backend suite:** `pytest backend/tests` → **133 passed, 5 failed**
   (`full-backend-suite-...log`). The 5 failures are all in `test_route_characterization.py`
   (similar/trust/classify/classify_batch/segment) returning **503 SERVICE UNAVAILABLE** because the
   ML model assets (classifier weights / mobile_sam checkpoint) are not present in this environment.
   They are in the ML-inference lane, **not the admin lane**, and are unchanged by this audit
   (no source edits). PASS for admin scope; the 5 are pre-existing, asset-dependent, out of scope.

**PASS/FAIL summary (admin scope):** PASS — admin backend routes, admin frontend units, and the
end-to-end functional probe are all green. The only red in the broader suite is asset-dependent ML
inference, outside this lane.

---

## Reproduce

```bash
# from worktree repo root
PY=backend/.venv/bin/python   # or any env with flask+PIL+pyyaml
PYTHONPATH="$PWD" $PY .omx/evidence/admin-functional-audit/admin_probe.py
$PY -m pytest backend/tests/test_admin_annotation_routes.py backend/tests/test_admin_training_routes.py -q
# frontend (needs node_modules):
( cd frontend && node_modules/.bin/vitest run src/pages/AdminAnnotationsPage.test.tsx )
```
