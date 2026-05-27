# Clinic Codex

Clinic Codex is a browser-based annotation and retraining tool for Nahuatl codex glyphs. It segments uploaded manuscript images, lets a human correct element boxes and labels, submits validated annotations for local admin review, and retrains the local classifier from admin-approved crops.

## Main workflow

1. **Upload/analyze** a glyph image on `/`.
2. **Inspect** segmentation results and model suggestions on the workspace.
3. **Annotate** on `/annotate/:id`: move/resize/draw boxes, select or type labels with fuzzy suggestions, create missing labels, and rename labels without deleting boxes.
4. **Validate** each element that is ready to submit for review. Draft elements stay visible but are excluded from submission exports.
5. **Send/export validated annotations** to `backend/annotations/<analysis_id>/`; submitted elements start as pending admin review.
6. **Review locally** at `/admin/annotations` (local/dev-only, not production-secured): approve/reject, correct class/bbox evidence, and inspect the Dataset tab.
7. **Retrain approved annotations only** with the guarded Training tab (`ENABLE_ADMIN_TRAINING_JOBS=1`) or `scripts/retrain.sh` / `scripts/retrain.ps1`. Retraining creates an immutable candidate under `backend/model_registry/versions/<version_id>/`; explicitly promote it with `scripts/promote_model.py`, then restart the backend to load new weights.

See [ANNOTATIONS.md](ANNOTATIONS.md) for the full annotation/export/retraining procedure.

## For researchers

- Upload codex images and review automatic segmentation.
- Use model confidence, top-k suggestions, similarity, and trust signals to decide whether a prediction is reliable.
- Correct boxes directly: pan/zoom the image, move boxes, resize handles, draw missing elements, or delete bad detections.
- Type the first letters of an element name to get suggestions. If the element is absent, create the new label during annotation.
- Mark only reviewed elements as **validated** before sending them for local admin review.

## Architecture

- **Backend**: Flask API serving segmentation, classification, similarity/trust helpers, classes, and annotation persistence. Default port: `7117`.
- **Frontend**: React + Vite + Tailwind application. Default port: `7118`.
- **Annotation storage**: backend-owned filesystem data under `backend/annotations/<analysis_id>/`, with element-level admin decisions in `backend/annotations/review-index.json`.
- **Training/versioning scripts**: `scripts/export_annotations.py`, `scripts/export_approved_annotations.py`, `scripts/retrain.sh`, `scripts/retrain.ps1`, and `scripts/promote_model.py`.

## Requirements

- Python `3.10` or `3.11`.
- Node.js `>=22.12.0` (matches `frontend/package.json`).
- Network access during setup for CPU PyTorch wheels, MobileSAM from GitHub, and model artifacts/downloads.

## Environment variables

Backend:

- `PORT=7117`
- `HOST=0.0.0.0`
- `CORS_ORIGINS=http://localhost:7118`
- `MODEL_DIR=/path/to/model/dir` (optional override)
- `ENABLE_ADMIN_TRAINING_JOBS=false` (set to `1` only for local loopback Training tab launches)

Frontend:

- `VITE_API_BASE_URL=http://localhost:7117`

## Development commands

```bash
# Start backend + frontend
bash scripts/run-dev.sh

# Start backend + frontend on alternate ports
BACKEND_PORT=7217 FRONTEND_PORT=7218 bash scripts/run-dev.sh

# Frontend only
cd frontend
npm install
npm run dev
npm run lint
npm run test
npm run build

# Backend/script tests
backend/.venv/bin/python -m pytest backend/tests scripts/test_export_annotations.py scripts/test_export_approved_annotations.py scripts/test_retrain_scripts.py scripts/test_promote_model.py
```

## Local model versioning

`scripts/retrain.*` never overwrites the runtime classifier in `backend/codex_model/`.
It writes checkpoints, prototypes, runtime-ready files, `manifest.json`,
`model-card.md`, and `checksums.sha256` into
`backend/model_registry/versions/<version_id>/`.

```bash
# Validate paths without training or changing runtime files
bash scripts/retrain.sh --dry-run

# After a real run, inspect and promote explicitly
backend/.venv/bin/python scripts/promote_model.py <version_id> --dry-run
backend/.venv/bin/python scripts/promote_model.py <version_id>

# Roll back to previous/original through the same checksum-verified path
backend/.venv/bin/python scripts/promote_model.py --rollback --dry-run
backend/.venv/bin/python scripts/promote_model.py --rollback
```

Promotion snapshots current runtime files under `backend/model_registry/snapshots/`
before atomically updating `backend/codex_model/weights/*.pt` and
`backend/codex_model/config.json`. Restart the backend after promotion or
rollback; candidate creation alone does not change predictions.

If `MODEL_DIR` is set, unset it before promotion unless you are deliberately
using `--runtime-dir` for a matching custom runtime package. The Training tab
launcher is Bash-only; use `scripts/retrain.ps1` directly on native Windows.

Script matrix:

- Ubuntu/macOS: `bash scripts/install.sh`, `bash scripts/run-dev.sh`.
- Native Windows PowerShell: `powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1`, `powershell -ExecutionPolicy Bypass -File .\scripts\run-dev.ps1`.

Do not treat `scripts/run-dev.ps1` as the Linux/macOS launcher under `pwsh`; use the bash launcher there. The dev launchers derive frontend API/CORS settings from `BACKEND_PORT` and `FRONTEND_PORT` unless you explicitly override `VITE_API_BASE_URL` or `CORS_ORIGINS`.

## Key documentation

- [frontend/README.md](frontend/README.md) — UI routes, annotation behavior, and frontend commands.
- [backend/README.md](backend/README.md) — Flask endpoints, payloads, storage, and backend limits.
- [ANNOTATIONS.md](ANNOTATIONS.md) — submission, local admin approval, and approved-only retraining workflow.
- [INSTALL.md](INSTALL.md) — installation notes.
