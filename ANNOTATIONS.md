# Annotations & Retraining Workflow

This document describes the supported Clinic Codex loop:

```text
browser annotation → validate elements → save/export validated data → retrain → restart backend
```

Only **validated** and **named** elements are eligible for training. Draft boxes remain useful for review, but they are not exported by default.

## TL;DR

1. Run the app with `bash scripts/run-dev.sh`.
2. Upload/analyze an image on `/`.
3. Open `/annotate/:id`.
4. Correct boxes and labels with the fuzzy label input.
5. Validate each element that should enter the dataset.
6. Click **Envoyer pour entraînement** or export from a localStorage dump with `scripts/export_annotations.py`.
7. Retrain with `bash scripts/retrain.sh` or `pwsh -NoProfile -File scripts/retrain.ps1`.
8. Restart the backend to use the new weights.

## 1. Annotate in the browser

From the workspace `/`, open an analysis and choose **Aller à l'annotation**.

In `/annotate/:id`:

- **Select** a box to inspect its class and confidence.
- **Move/resize** a box when the detected square is visually wrong.
- **Pan/zoom** the image; the image and SVG overlay move together.
- **Draw** a new missing element.
- **Type labels** in the combobox. Suggestions appear from the first letters; prefix matches are prioritized over contains matches.
- **Create labels** by entering a name that is not already returned by `/classes`.
- **Rename labels** directly in the combobox without deleting and redrawing the square.
- **Validate** reviewed elements. Any edit to label or geometry returns that element to draft until you validate it again.

A valid training candidate has:

- bbox `[x, y, width, height]` in image pixels;
- non-empty `class_name` that is not `unknown`;
- `annotationStatus[index] === "validated"`.

## 2. Save validated annotations through the backend

With the Flask backend running, click **Envoyer pour entraînement**. The frontend filters locally and posts only validated, named elements to `POST /save-annotation`.

Payload shape:

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

Success response:

```json
{
  "status": "ok",
  "analysis_id": "analysis_123",
  "saved_count": 1,
  "classes": ["atl"],
  "saved_at": "2026-05-21T18:00:00+00:00"
}
```

### Storage layout

The backend stores annotations as the canonical training review data:

```text
backend/annotations/<analysis_id>/
├── image.png
├── metadata.json
└── elements/
    └── <index>.png
```

`metadata.json` records `analysis_id`, upload time, class names, clamped bboxes, and crop paths. The backend sanitizes class names and clamps bboxes to image bounds before writing files.

## 3. Export from localStorage JSON

Use this option when the backend was not running during annotation or when exporting many browser records at once.

```bash
backend/.venv/bin/python scripts/export_annotations.py analyses.json
```

Default behavior is **validated-only**. Records with no validated annotations are skipped.

For legacy data that predates `annotationStatus`, use the explicit escape hatch:

```bash
backend/.venv/bin/python scripts/export_annotations.py analyses.json --include-unvalidated
```

Optional paths:

```bash
backend/.venv/bin/python scripts/export_annotations.py analyses.json \
  --annotations-dir backend/annotations \
  --output backend/training_data/Elements
```

`--output` is kept for compatibility with older workflows; current storage writes the canonical data under `backend/annotations/`.

## 4. Retrain

Once validated crops exist under `backend/annotations/`, run one retraining command from the repository root.

Linux/macOS:

```bash
bash scripts/retrain.sh
```

Windows/PowerShell:

```powershell
pwsh -NoProfile -File scripts/retrain.ps1
```

Dry-run the PowerShell step list without running the pipeline:

```powershell
pwsh -NoProfile -File scripts/retrain.ps1 -WhatIf
```

Both retraining scripts execute the same four steps:

| Step | Script | Purpose |
| --- | --- | --- |
| 1/4 | `build_metadata.py` | Scan saved annotations and build training metadata. |
| 2/4 | `precompute_embeddings.py` | Compute DINOv2 embeddings for crops. |
| 3/4 | `train.py` | Train prototype classifiers. |
| 4/4 | `export_model.py` | Export `codex_model/weights/{prototypes.pt,projection.pt}`. |

A lockfile at `backend/.retrain.lock` prevents concurrent runs. If a run crashed and no retrain process is active, remove the stale lockfile:

```bash
rm -f backend/.retrain.lock
```

Restart the Flask backend after retraining; running processes do not hot-load new weights.

## FAQ

### Why does my new class not appear in predictions immediately?

Creating a label in the frontend only creates annotation data. The model will not predict that class until enough validated examples are saved, the retraining pipeline exports new weights, and the backend is restarted.

### Why is an edited element draft again?

Geometry or label edits can invalidate a previous review decision. Validate it again once the corrected box and label are ready for training.

### Why did export skip my analysis?

The default export is validated-only. Validate at least one named element, or pass `--include-unvalidated` only for legacy data you intentionally want to migrate.

### What dataset size is required?

The scripts do not enforce a universal minimum, but retraining is only meaningful with enough validated examples per class to represent visual variation. If a class has too few examples, add and validate more crops before trusting predictions.

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `POST /save-annotation` returns 400 | Missing field, invalid JSON, empty annotations, invalid class name, invalid bbox, or bad image data URL | Check browser console/network response; validate at least one named element. |
| `POST /save-annotation` returns 413 | Payload exceeds 50 MB | Reduce image size before upload. |
| `POST /save-annotation` returns 409 | Backend cannot write to `backend/annotations/` | Fix directory permissions. |
| `POST /save-annotation` returns 507 | Disk full | Free disk space and retry. |
| Crops look wrong | Bbox format or image geometry mismatch | Bbox must be `[x, y, width, height]` in image pixels. Re-run geometry tests if code changed. |
| New weights not used after retraining | Backend still has old model in memory | Restart with `bash scripts/run-dev.sh` or `python backend/examples/flask_api.py`. |
