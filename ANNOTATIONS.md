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
9. Build a cumulative snapshot from the existing corpus and current approvals, configure `ADMIN_TRAINING_SNAPSHOT_DIR`, then use **Training** for a dry run/full local warm-start run when `ENABLE_ADMIN_TRAINING_JOBS=1` is set. Full runs create a candidate version under `backend/model_registry/versions/<version_id>/`.
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

Filters and class distributions are UI-only helpers. The snapshot builder reads the backend approved-only iterator, never client-side filtering.

### Training tab

The Training tab shows current approved counts, cumulative snapshot state, resolved paths, artifacts, and the latest job/log tail. Starting a run is intentionally disabled unless all launch guards pass:

- set `ENABLE_ADMIN_TRAINING_JOBS=1` before starting the backend;
- access the backend from a loopback client (`localhost`, `127.0.0.1`, or `::1`);
- use a local `Host` header and local `Origin` header;
- configure a valid `training-snapshot.v2` through `ADMIN_TRAINING_SNAPSHOT_DIR`;
- keep its review-index hash current and provide the pinned backbone plus current projection;
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

Unknown fields, nonlocal requests, invalid devices, invalid batch sizes, stale snapshots, concurrent launch attempts, and concurrent running jobs are rejected. The backend starts only the allowlisted `bash scripts/retrain.sh` command with the configured snapshot, backbone pin, warm-start projection, and optional `--dry-run`. Job status/logs are written under `backend/training_runs/<run_id>/`, and the summary surfaces the snapshot hash, local registry aliases, and manifest/checksum health.
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

The standard local mode combines the **shipped model base and all current approved annotations**. It needs no private corpus or manual snapshot. The installer downloads fixed MobileSAM and DINOv2 assets, enables local training, and permits the initial local administrator to review their own annotations.

1. Upload and analyze an image, open its annotation editor, correct boxes and labels, and mark the desired elements ready.
2. Send the annotations, then open **Admin → Review** and approve them.
3. Open **Training**, run the dry run, then select **Non, entraînement complet** and launch.
4. Inspect the candidate path and result in Training.

Each run captures current approved crops and review decisions automatically. Exact duplicate images count once; conflicting labels for identical images are rejected. Stale decisions and missing crops are excluded. Repeating the same approvals does not count their contribution twice.

The backbone and projection stay frozen. The update adapts prototypes for existing base-model classes; it does not train MobileSAM or introduce new classes. The original base provides the prior even when its training images are unavailable.

Candidates are stored under `backend/model_registry/versions/<version_id>/`, with provenance and checksums. They are **not activated**; promotion is blocked because this local mode has no independent holdout. Reported base/candidate scores measure training-image fit, not better generalization. The running model is unchanged and no restart is needed.

## FAQ

### Why does my new class not appear in predictions immediately?

Creating a label stores annotation data. Local retraining supports only the existing model taxonomy. Candidate creation does not alter predictions.

### Why is an edited element draft again?

Geometry or label edits can invalidate a previous review decision. Validate it again once the corrected box and label are ready for training.

### Why did export skip my analysis?

The default export is validated-only. Validate at least one named element, or pass `--include-unvalidated` only for legacy data you intentionally want to migrate.

### Why did retraining skip my submitted element?

Check `/admin/annotations`: the element must be approved, its source metadata/crop must still exist, and any stale source fingerprint must be reapproved after replacement. The next local run automatically captures current approvals. Only explicitly configured advanced snapshots need rebuilding.

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
| New weights not used after retraining | Expected: candidate-only workflow | Inspect the candidate in Training; it is not activated. |
