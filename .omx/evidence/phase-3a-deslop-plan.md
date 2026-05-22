# Phase 3A Scoped Deslop Plan

## Scope
Ralph-owned Phase 3A files only:
- `docs/API_CONTRACT.md`
- `backend/tests/test_api_contracts.py`
- `frontend/src/services/api.ts`
- `frontend/src/services/api.test.ts`
- `frontend/src/types/index.ts`
- `.omx/evidence/phase-3a-*.txt` evidence artifacts

## Behavior lock already in place
- Backend contract + route tests: `.omx/evidence/phase-3a-backend-contract-tests.txt` (`39 passed`).
- Backend full tests: `.omx/evidence/phase-3a-final-backend-tests.txt` (`52 passed`).
- Frontend API tests: `.omx/evidence/phase-3a-frontend-api-tests.txt` (`11 passed`).
- Frontend full tests: `.omx/evidence/phase-3a-final-frontend-test.txt` (`99 passed`).
- Frontend lint/build: `.omx/evidence/phase-3a-final-frontend-lint.txt`, `.omx/evidence/phase-3a-final-frontend-build.txt`.

## Cleanup plan
1. Inventory fallback-like code in scoped files.
2. Inspect for dead code, duplication, needless abstraction, weak naming/error handling, and test brittleness.
3. Apply only high-signal, behavior-preserving edits if found.
4. Re-run backend contract/full tests, frontend API/full tests, lint, build, and hygiene checks.

## Expected constraints
- No backend route/storage changes.
- No package/CI changes.
- No unrelated worktree edits/staging.
