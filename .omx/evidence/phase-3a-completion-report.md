# Phase 3A API Contracts and Typed Frontend API Client Completion Report

## Scope completed
Executed Phase 3A using the approved artifacts:
- `.omx/plans/prd-phase-3a-api-contracts-client.md`
- `.omx/plans/test-spec-phase-3a-api-contracts-client.md`

Implemented the approved lightweight contract approach: `API_CONTRACT.md` plus backend/frontend contract tests. OpenAPI generation was not introduced.

## Phase 3A-owned files changed/added
- `docs/API_CONTRACT.md` — lightweight frontend/backend API contract source of truth.
- `backend/tests/test_api_contracts.py` — backend JSON contract tests for active endpoint success/error shapes.
- `frontend/src/services/api.ts` — typed API helper boundary, preserved exported functions, added optional `AbortSignal` support.
- `frontend/src/services/api.test.ts` — frontend API client characterization and abort forwarding tests.
- `frontend/src/types/index.ts` — compatibility-safe API types (`ClassLabel`, optional `saved_at`, typed save error code union).
- `.omx/evidence/phase-3a-*.txt` / `.md` — execution evidence.

## Preserved public API behavior
Existing exported frontend API function names remain:
- `segmentGlyph`
- `classifyElement`
- `getClasses`
- `getSimilar`
- `getTrust`
- `saveAnnotation`

Behavior preserved:
- Non-save functions still resolve backend data and reject/throw through axios on HTTP/network failures.
- `saveAnnotation` still returns a `SaveAnnotationResult` union and does not throw for expected save HTTP/storage/network failures.
- Optional `AbortSignal` support is trailing and does not force consumer call-site changes.
- Existing page mocks and call signatures still pass page regression tests.

## Contract decisions implemented
- `docs/API_CONTRACT.md` documents active endpoints `/health`, `/classes`, `/classify`, `/classify-batch`, `/segment`, `/similar`, `/trust`, `/save-annotation` plus legacy sample endpoints.
- Error taxonomy documents simple `{ error: string }`, nested `{ error: { code, message } }`, save validation, and save storage/internal error shapes.
- Frontend types now model observed compatibility:
  - `class_label` can be `string | number | null` and optional where backend/stubs permit.
  - `SaveAnnotationResponse.saved_at` is optional.
  - `SaveAnnotationErrorCode` includes `PERMISSION_DENIED`, `DISK_FULL`, `STORAGE_ERROR`, `INTERNAL_ERROR`, and client-side `NETWORK_ERROR`.

## Verification evidence
| Gate | Evidence | Result |
| --- | --- | --- |
| Baseline worktree | `.omx/evidence/phase-3a-baseline.txt` | Captured pre-existing package/CI/Phase 2 changes and hashes |
| Baseline backend route tests | `.omx/evidence/phase-3a-baseline-backend-tests.txt` | `30 passed` using `backend/.venv/bin/python` |
| Baseline frontend targeted tests | `.omx/evidence/phase-3a-baseline-frontend-tests.txt` | `2 passed / 31 tests passed` |
| Backend contract + route tests | `.omx/evidence/phase-3a-backend-contract-tests.txt` | `39 passed` |
| Backend full tests | `.omx/evidence/phase-3a-final-backend-tests.txt` | `52 passed` |
| Frontend API tests | `.omx/evidence/phase-3a-frontend-api-tests.txt` | `1 file / 11 tests passed` |
| Frontend affected page tests | `.omx/evidence/phase-3a-frontend-page-tests.txt` | `5 files / 52 tests passed` |
| Frontend full tests | `.omx/evidence/phase-3a-final-frontend-test.txt` | `13 files / 99 tests passed` |
| Frontend lint | `.omx/evidence/phase-3a-final-frontend-lint.txt` | passed |
| Frontend build/typecheck | `.omx/evidence/phase-3a-final-frontend-build.txt` | passed |
| Architect verification | Ralph architect verifier | APPROVE |
| Deslop pass | `.omx/evidence/phase-3a-deslop-report.md` | No code edits needed; grounded save fallback preserved |
| Post-deslop backend contract tests | `.omx/evidence/phase-3a-post-deslop-backend-contract-tests.txt` | `39 passed` |
| Post-deslop backend full tests | `.omx/evidence/phase-3a-post-deslop-backend-full-tests.txt` | `52 passed` |
| Post-deslop frontend API tests | `.omx/evidence/phase-3a-post-deslop-frontend-api-tests.txt` | `1 file / 11 tests passed` |
| Post-deslop frontend full tests | `.omx/evidence/phase-3a-post-deslop-frontend-full-test.txt` | `13 files / 99 tests passed` |
| Post-deslop frontend lint | `.omx/evidence/phase-3a-post-deslop-frontend-lint.txt` | passed |
| Post-deslop frontend build/typecheck | `.omx/evidence/phase-3a-post-deslop-frontend-build.txt` | passed |
| Final hygiene | `.omx/evidence/phase-3a-post-deslop-final-hygiene.txt` | `git diff --check` clean; no staged files |

## Notes on verification environment
- An initial baseline attempt with system `python`/`python3` failed because the system interpreter lacked Flask/Pillow. The successful backend evidence uses the project virtualenv: `backend/.venv/bin/python`.
- A backend full-test command accidentally run from `frontend/` failed due to wrong working directory; the successful full backend evidence is `.omx/evidence/phase-3a-final-backend-tests.txt` and post-deslop `.omx/evidence/phase-3a-post-deslop-backend-full-tests.txt`.

## Phase boundary confirmation
- No backend route implementation behavior was changed.
- No backend storage files or storage migration were changed.
- No package, lockfile, Node, Python, or CI files were changed by Phase 3A.
- No unrelated existing worktree changes were staged or overwritten.
- `git diff --cached --name-only` is empty in final hygiene evidence.
- Final hashes for `.github/workflows/smoke.yml`, `backend/requirements-dev.txt`, `frontend/package.json`, and `frontend/package-lock.json` match the Phase 3A baseline hashes.

## Rollback strategy
To roll back Phase 3A only, revert/remove these Phase 3A-owned files:
- `docs/API_CONTRACT.md`
- `backend/tests/test_api_contracts.py`
- `frontend/src/services/api.test.ts`
- edits to `frontend/src/services/api.ts`
- edits to `frontend/src/types/index.ts`
- optional evidence files `.omx/evidence/phase-3a-*`

Do not revert pre-existing package/CI or Phase 2 frontend page modularization work unless that is a separate approved task.

## Remaining risks / follow-ups
- Markdown + tests can drift in future phases; future endpoint changes must update `docs/API_CONTRACT.md` and contract tests together.
- OpenAPI/schema generation remains deferred to a later explicitly scoped phase.
