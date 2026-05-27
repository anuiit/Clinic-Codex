# Annotations & Retraining Workflow

This document describes the supported Clinic Codex loop:

```text
browser annotation → submit validated elements → Admin Review tab → Dataset tab → Training tab/script → promote version → restart backend
```

Only **admin-approved** submitted elements are eligible for retraining. Browser validation means "ready to submit for review"; it does not by itself make an element trainable.

## TL;DR

1. Run the app with `bash scripts/run-dev.sh`.
2. Upload/analyze an image on `/`.
3. Open `/annotate/:id`.
4. Correct boxes and labels with the fuzzy label input.
5. Validate each element that should enter the dataset.
6. Click **Envoyer pour entraînement** or export from a localStorage dump with `scripts/export_annotations.py`; the backend saves those elements as **pending admin review**.
7. Open `/admin/annotations` locally. Use **Review** to approve, reject, or correct class/bbox values; ordinary corrections return to pending unless you choose **Save & approve**.
8. Use **Dataset** to inspect trainable approved crops, approved-but-not-trainable diagnostics, rejected items, and pending items before training.
9. Use **Training** for a dry run/full local run when `ENABLE_ADMIN_TRAINING_JOBS=1` is set, or run `bash scripts/retrain.sh` / `pwsh -NoProfile -File scripts/retrain.ps1` from the repository root. Dry runs print and validate the approved-only command plan; full runs create a candidate version under `backend/model_registry/versions/<version_id>/`.
10. Promote the candidate with `backend/.venv/bin/python scripts/promote_model.py <version_id>`.
11. Restart the backend to use the promoted weights.

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

A valid submission candidate has:

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

The save endpoint preserves the submission as canonical evidence and starts each element as pending review. Pending and rejected elements are excluded from retraining.

### Storage layout

The backend stores submitted annotations as canonical review data:

```text
backend/annotations/<analysis_id>/
├── image.png
├── metadata.json
└── elements/
    └── <index>.png
```

`metadata.json` records `analysis_id`, upload time, class names, clamped bboxes, and crop paths. The backend sanitizes class names and clamps bboxes to image bounds before writing files.

## 3. Local admin review, dataset inspection, and guarded training

Open the local admin review page while backend and frontend are bound to your machine:

```text
http://localhost:7118/admin/annotations
```

The page calls backend local admin endpoints under `http://localhost:7117/admin/annotations` and, for the training tab, `http://localhost:7117/admin/training`.

Important boundaries:

- This route is **local/dev-only** and **not production-secured**. Do not expose it on a shared network as a real admin system.
- There is no auth, role, token, or production authorization. The model registry/rollback layer is local JSON/filesystem safety tooling, not access control.
- The Training tab is a guarded local wrapper around `scripts/retrain.sh`; it creates an immutable candidate version and is disabled by default. It must not be treated as production authorization.
- Decisions are **element-level**. One analysis can contain approved, rejected, and pending elements at the same time.
- The review manifest is stored at `backend/annotations/review-index.json`.
- If a crop, metadata row, annotation folder, or source fingerprint is stale/missing, that element is not trainable even if an old manifest entry says approved.

### Review tab

Use **Review** to compare the source image and element crop, then approve or reject each element. You can also correct the class name or bbox:

- **Save changes** rewrites metadata/crop evidence and resets the element to `pending`.
- **Save & approve** rewrites metadata/crop evidence and immediately records a fresh `approved` decision.
- Any modification regenerates the crop from `image.png` and creates a fresh source fingerprint so stale approvals cannot silently remain trainable.

Backend API:

```http
GET  /admin/annotations
POST /admin/annotations/<analysis_id>/<index>/review
POST /admin/annotations/<analysis_id>/<index>/modify
GET  /admin/annotations/<analysis_id>/image
GET  /admin/annotations/<analysis_id>/<index>/crop
```

Mutation payload:

```json
{ "status": "approved" }
```

Allowed statuses are `pending`, `approved`, and `rejected`; the UI exposes approve/reject actions.

Modify payload:

```json
{
  "class_name": "atl",
  "bbox": [120, 240, 80, 60],
  "approve_after_save": false
}
```

`approve_after_save: true` is equivalent to saving with status `approved`. Without it, modifications default to `pending`.

### Dataset tab

Use **Dataset** as a read-only preflight view before training. It derives its rows from the same review queue and separates:

- trainable approved crops;
- approved rows excluded by diagnostics such as missing/stale crop evidence;
- rejected rows;
- pending rows.

Filters and class distributions are UI-only helpers. The retraining bridge still uses the backend approved-only iterator, not client-side filtering.

### Training tab

The Training tab always shows approved-only counts, class distribution, resolved paths, artifact status, and the latest job/log tail. Starting a run is intentionally disabled unless all launch guards pass:

- set `ENABLE_ADMIN_TRAINING_JOBS=1` before starting the backend;
- access the backend from a loopback client (`localhost`, `127.0.0.1`, or `::1`);
- use a local `Host` header and local `Origin` header;
- keep `scripts/retrain.sh` present.

Launch payloads are limited to:

```json
{
  "dry_run": true,
  "device": "auto",
  "batch_size": 16,
  "notes": "optional short note"
}
```

Unknown fields, nonlocal requests, invalid devices, invalid batch sizes, concurrent launch attempts, and concurrent running jobs are rejected. The backend starts only the allowlisted command `bash scripts/retrain.sh` (plus `--dry-run` for dry runs) with a small allowlisted environment including `MODEL_VERSION_ID` and `MODEL_REGISTRY_DIR`. Job status/logs are written under `backend/training_runs/<run_id>/`, and the summary surfaces local registry aliases plus manifest/checksum health.
If a backend restart leaves behind a `running` dry-run without its in-memory process handle, or a full run whose lock PID and recorded process identity cannot still confirm the original retrain process, the next job read marks it failed so a stale local status file does not permanently block the launcher.

The browser Training tab launcher is Bash-only (`scripts/retrain.sh`). Native Windows users should use the PowerShell command-line path shown below unless they are running through WSL/Git Bash.

## 4. Export from localStorage JSON

Use this option when the backend was not running during annotation or when exporting many browser records at once.

```bash
backend/.venv/bin/python scripts/export_annotations.py analyses.json
```

Default behavior is **validated-only submission**. Records with no validated annotations are skipped, and exported submissions still require local admin approval before retraining.

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

## 5. Retrain

Once submitted crops have been approved under `/admin/annotations`, use the Training tab for a guarded local dry run/full run or run one retraining command from the repository root. Retraining now produces an immutable **candidate** package under `backend/model_registry/versions/<version_id>/`; it does **not** overwrite `backend/codex_model/`.

Training tab launch is off by default:

```bash
ENABLE_ADMIN_TRAINING_JOBS=1 bash scripts/run-dev.sh
```

If the tab reports `disabled_by_default: set ENABLE_ADMIN_TRAINING_JOBS=1 to allow local launches`,
use the command-line alternative below or restart the backend with the feature flag enabled. Keep this
as an explicit local opt-in rather than a committed dev-script default.

Linux/macOS:

```bash
bash scripts/retrain.sh
```

Windows/PowerShell:

```powershell
pwsh -NoProfile -File scripts/retrain.ps1
```

Dry-run the step list without running the pipeline:

```bash
bash scripts/retrain.sh --dry-run
```

```powershell
pwsh -NoProfile -File scripts/retrain.ps1 -DryRun
```

Both retraining scripts execute the same approved-only classifier stages with repo-root anchored explicit paths:

| Step | Script | Purpose |
| --- | --- | --- |
| 1/6 | `scripts/export_approved_annotations.py` | Materialize only admin-approved, non-stale crops into `backend/training_data/approved/Elements`. |
| 2/6 | `build_metadata.py` | Build `backend/training_data/approved/metadata.csv` from the generated Elements dataset. |
| 3/6 | `precompute_embeddings.py` | Compute DINOv2 embeddings to `backend/training_data/approved/precomputed/features.pt`. |
| 4/6 | `train.py` | Train projection/classifier checkpoints into `backend/model_registry/versions/<version_id>/checkpoints`. |
| 5/6 | `evaluate.py --export-prototypes` | Export `backend/model_registry/versions/<version_id>/prototypes/prototypes.pt`. |
| 6/6 | `export_model.py` | Export backend-loadable candidate files under `backend/model_registry/versions/<version_id>/runtime/`. Runtime writes are refused unless the bootstrap-only `--allow-runtime-write` flag is used outside retraining. |

No MobileSAM/segmentation retraining is run by these scripts.

Inspect the candidate before activation:

```bash
cat backend/model_registry/versions/<version_id>/export_model_manifest.json
cat backend/model_registry/versions/<version_id>/manifest.json
cat backend/model_registry/versions/<version_id>/model-card.md
```

Activate a candidate only with the promotion tool:

```bash
backend/.venv/bin/python scripts/promote_model.py <version_id> --dry-run
backend/.venv/bin/python scripts/promote_model.py <version_id>
```

Promotion verifies registered manifests/checksums, snapshots current runtime files under `backend/model_registry/snapshots/`, atomically replaces `backend/codex_model/weights/*.pt` and `backend/codex_model/config.json`, records previous/promoted pointers in `backend/model_registry/index.json`, and prints a restart reminder.

If `MODEL_DIR` is set, unset it before promotion. Promotion targets the local `backend/codex_model/` runtime package by default; an ambient `MODEL_DIR` override can cause the backend to load weights from a different location, so `scripts/promote_model.py` refuses to run in that ambiguous state unless an explicit `--runtime-dir` is supplied for a matching custom runtime package. If a promotion is interrupted, `backend/model_registry/promotion_in_progress.json` remains as a recovery marker and the Training summary reports it.

Rollback restores the previous/original version through the same checksum/snapshot path:

```bash
backend/.venv/bin/python scripts/promote_model.py --rollback --dry-run
backend/.venv/bin/python scripts/promote_model.py --rollback
```

A lockfile at `backend/.retrain.lock` prevents concurrent runs. If a run crashed and no retrain process is active, remove the stale lockfile:

```bash
rm -f backend/.retrain.lock
```

Restart the Flask backend after promotion or rollback; running processes do not hot-load new weights. Dry-runs and candidate creation alone do not change predictions.

## FAQ

### Why does my new class not appear in predictions immediately?

Creating a label in the frontend only creates submitted annotation data. The model will not predict that class until enough examples are submitted, admin-approved, retrained into a candidate, explicitly promoted, and the backend is restarted.

### Why is an edited element draft again?

Geometry or label edits can invalidate a previous review decision. Validate it again once the corrected box and label are ready for training.

### Why did export skip my analysis?

The default export is validated-only. Validate at least one named element, or pass `--include-unvalidated` only for legacy data you intentionally want to migrate.

### Why did retraining skip my submitted element?

The retraining bridge is approved-only. Check `/admin/annotations`: the element must be approved, its source metadata/crop must still exist, and any stale source fingerprint must be reapproved after replacement.

### What dataset size is required?

The scripts do not enforce a universal minimum, but retraining is only meaningful with enough validated examples per class to represent visual variation. If a class has too few examples, add and validate more crops before trusting predictions.

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `POST /save-annotation` returns 400 | Missing field, invalid JSON, empty annotations, invalid class name, invalid bbox, or bad image data URL | Check browser console/network response; validate at least one named element. |
| `POST /save-annotation` returns 413 | Payload exceeds 50 MB | Reduce image size before upload. |
| `POST /save-annotation` returns 409 | Backend cannot write to `backend/annotations/` | Fix directory permissions. |
| `POST /save-annotation` returns 507 | Disk full | Free disk space and retry. |
| Element stays pending in admin queue | Submitted but not approved | Open `/admin/annotations` locally and approve or reject the element. |
| Approved element is not trainable | Missing crop/source or stale decision | Re-submit or reapprove the current source annotation. |
| Crops look wrong | Bbox format or image geometry mismatch | Bbox must be `[x, y, width, height]` in image pixels. Re-run geometry tests if code changed. |
| New weights not used after retraining | Candidate was not promoted, or backend still has old model in memory | Run `scripts/promote_model.py <version_id>` and restart with `bash scripts/run-dev.sh` or `python backend/examples/flask_api.py`. |
