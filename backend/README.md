# Clinic Codex Backend

Flask backend for Clinic Codex. It serves the local codex classifier, MobileSAM segmentation, similarity/trust helpers, class names, and validated annotation persistence.

Default runtime: `http://localhost:7117` (`HOST=0.0.0.0`, `PORT=7117`).

## Run

```bash
cd backend
pip install -r requirements.txt
python examples/flask_api.py
```

The repository-level dev helper starts backend and frontend together:

```bash
bash scripts/run-dev.sh
```

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `PORT` | `7117` | Flask port. |
| `HOST` | `0.0.0.0` | Flask bind host. |
| `CORS_ORIGINS` | `http://localhost:7118` | Comma-separated browser origins. |
| `MODEL_DIR` | auto-detected | Optional classifier weights/config directory. |

Requests are capped at **50 MB** via Flask `MAX_CONTENT_LENGTH`.

## API endpoints

| Endpoint | Method | Input | Output |
| --- | --- | --- | --- |
| `/segment` | `POST` | multipart form field `image` | `{ num_elements, image_size: [w,h], elements: [{ bbox, class_name, confidence, rejected, top_k, ... }] }` |
| `/classes` | `GET` | none | `{ num_classes, class_names }` from `backend/codex_model/config.json` |
| `/similar` | `POST` | JSON `{ image_base64, bbox, limit }` | prototype similarity ranking for the crop |
| `/trust` | `POST` | JSON `{ image_base64, bbox, predicted_class, top_k }` | trust signals: predicted rank/similarity, top1, margin, ambiguity, entropy |
| `/save-annotation` | `POST` | JSON payload below | saves validated annotation crops under `backend/annotations/<analysis_id>/` |

Legacy/demo endpoints `/classify`, `/classify-batch`, `/sample-image`, and `/similar-samples` may also exist for local experiments; the current frontend workflow depends on the endpoints listed above.

## `/save-annotation` payload

The frontend sends only validated, named elements. Bboxes are image-pixel coordinates in `[x, y, width, height]` format.

```json
{
  "analysis_id": "analysis_123",
  "image_name": "387_769v.jpg",
  "image_data_url": "data:image/jpeg;base64,...",
  "timestamp": 1779376522290,
  "annotations": [
    { "index": 0, "bbox": [120, 240, 80, 60], "class_name": "atl" }
  ]
}
```

Required fields: `analysis_id`, `image_data_url`, and non-empty `annotations`. `analysis_id` must contain only letters, digits, `_`, or `-`.

Successful response:

```json
{
  "status": "ok",
  "analysis_id": "analysis_123",
  "saved_count": 1,
  "classes": ["atl"],
  "saved_at": "2026-05-21T18:00:00+00:00"
}
```

Common errors:

- `400`: invalid JSON, missing field, empty annotations, invalid image data URL, invalid class name, invalid bbox.
- `409 PERMISSION_DENIED`: backend cannot write to `backend/annotations/`.
- `413`: request exceeds 50 MB.
- `507 DISK_FULL`: no disk space.
- `500 STORAGE_ERROR` or `INTERNAL`: unexpected storage/server failure.

## Annotation storage

`backend/services/annotation_storage.py` is the single source of truth for saved annotations.

For each save, the backend atomically replaces:

```text
backend/annotations/<analysis_id>/
├── image.png
├── metadata.json
└── elements/
    ├── 0.png
    └── 1.png
```

`metadata.json` contains the sanitized class names, clamped bboxes, crop paths, and timestamp. The storage service:

- sanitizes `class_name` by rejecting empty labels, `/`, `\\`, `..`, and null bytes;
- clamps bboxes to the image bounds and rejects zero/negative crop dimensions;
- writes through a temporary directory before replacing the target folder;
- no longer writes directly to `training_data/Elements` (the parameter remains for backward compatibility).

## Tests

```bash
backend/.venv/bin/python -m pytest backend/tests/test_annotation_storage.py backend/tests/test_save_endpoint.py
backend/.venv/bin/python -m pytest scripts/test_export_annotations.py
```

## Retraining

Validated crops saved under `backend/annotations/` are consumed by the retraining pipeline:

```bash
bash scripts/retrain.sh
# or on Windows / PowerShell
pwsh -NoProfile -File scripts/retrain.ps1
```

Both scripts run:

1. `backend/codex_pipeline/scripts/build_metadata.py`
2. `backend/codex_pipeline/scripts/precompute_embeddings.py`
3. `backend/codex_pipeline/scripts/train.py`
4. `backend/codex_pipeline/scripts/export_model.py`

A lockfile at `backend/.retrain.lock` prevents concurrent retrains. Restart the Flask backend after retraining so it loads the new weights.
