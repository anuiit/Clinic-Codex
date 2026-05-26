# Clinic Codex

Clinic Codex is a browser-based annotation and retraining tool for Nahuatl codex glyphs. It segments uploaded manuscript images, lets a human correct element boxes and labels, saves only validated annotations for training, and retrains the local classifier from those curated crops.

## Main workflow

1. **Upload/analyze** a glyph image on `/`.
2. **Inspect** segmentation results and model suggestions on the workspace.
3. **Annotate** on `/annotate/:id`: move/resize/draw boxes, select or type labels with fuzzy suggestions, create missing labels, and rename labels without deleting boxes.
4. **Validate** each element that is ready for training. Draft elements stay visible but are excluded from training exports.
5. **Send/export validated annotations** to `backend/annotations/<analysis_id>/`.
6. **Retrain** with `scripts/retrain.sh` or `scripts/retrain.ps1`, then restart the backend to load new weights.

See [ANNOTATIONS.md](ANNOTATIONS.md) for the full annotation/export/retraining procedure.

## For researchers

- Upload codex images and review automatic segmentation.
- Use model confidence, top-k suggestions, similarity, and trust signals to decide whether a prediction is reliable.
- Correct boxes directly: pan/zoom the image, move boxes, resize handles, draw missing elements, or delete bad detections.
- Type the first letters of an element name to get suggestions. If the element is absent, create the new label during annotation.
- Mark only reviewed elements as **validated** before sending them for training.

## Architecture

- **Backend**: Flask API serving segmentation, classification, similarity/trust helpers, classes, and annotation persistence. Default port: `7117`.
- **Frontend**: React + Vite + Tailwind application. Default port: `7118`.
- **Annotation storage**: backend-owned filesystem data under `backend/annotations/<analysis_id>/`.
- **Training scripts**: `scripts/export_annotations.py`, `scripts/retrain.sh`, and `scripts/retrain.ps1`.

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

# Backend tests
backend/.venv/bin/python -m pytest backend/tests scripts/test_export_annotations.py
```

Script matrix:

- Ubuntu/macOS: `bash scripts/install.sh`, `bash scripts/run-dev.sh`.
- Native Windows PowerShell: `powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1`, `powershell -ExecutionPolicy Bypass -File .\scripts\run-dev.ps1`.

Do not treat `scripts/run-dev.ps1` as the Linux/macOS launcher under `pwsh`; use the bash launcher there. The dev launchers derive frontend API/CORS settings from `BACKEND_PORT` and `FRONTEND_PORT` unless you explicitly override `VITE_API_BASE_URL` or `CORS_ORIGINS`.

## Key documentation

- [frontend/README.md](frontend/README.md) — UI routes, annotation behavior, and frontend commands.
- [backend/README.md](backend/README.md) — Flask endpoints, payloads, storage, and backend limits.
- [ANNOTATIONS.md](ANNOTATIONS.md) — validated-only annotation and retraining workflow.
- [INSTALL.md](INSTALL.md) — installation notes.
