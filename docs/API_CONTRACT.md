# API Contract

This document captures the current Clinic Codex frontend/backend API contract for Phase 3A. It is intentionally lightweight and test-backed; Phase 3A does **not** introduce OpenAPI generation or generated clients.

## Base URL

The frontend API client reads `VITE_API_BASE_URL` and falls back to `http://localhost:7117`.

## Error Shape Taxonomy

The backend currently exposes several compatibility error shapes. Clients must model them explicitly instead of assuming a single global error envelope.

1. Simple route errors: `{ "error": string }`
   - Missing multipart image fields on `/classify`, `/classify-batch`, `/segment`.
   - Save validation/decode failures also include `status: "error"`.
2. Nested request errors: `{ "error": { "code": string, "message": string } }`
   - `/similar`, `/trust`, legacy `/similar-samples` validation errors, and corrupt upload bytes on `/classify`, `/classify-batch`, `/segment`.
3. Save storage/internal errors: `{ "status": "error", "error_code": string, "message": string, "hint"?: string | null, "trace_id"?: string }`
   - `/save-annotation` storage permission, disk, generic storage, and unknown internal failures.

## Active Endpoints

| Method | Path | Request | Success body | Error bodies | Frontend export |
| --- | --- | --- | --- | --- | --- |
| `GET` | `/health` | none | `{ status: "ok" }` | Flask/default errors only | none |
| `GET` | `/classes` | none | `{ num_classes: number, class_names: string[] }` | transport/HTTP failure | `getClasses()` |
| `POST` | `/classify` | multipart form field `image` | classifier payload | missing field simple error; corrupt bytes nested `INVALID_IMAGE` | `classifyElement(file)` |
| `POST` | `/classify-batch` | multipart form field(s) `images` | classifier payload array | missing field simple error; corrupt bytes nested `INVALID_IMAGE` | none |
| `POST` | `/segment` | multipart form field `image` | `{ num_elements, image_size, elements }` | missing field simple error; corrupt bytes nested `INVALID_IMAGE` | `segmentGlyph(file)` |
| `POST` | `/similar` | JSON `{ image_base64, bbox, limit?, mode? }` | `{ query, best_match, results }` | nested `INVALID_REQUEST`, `INVALID_IMAGE`, `INVALID_BBOX` | `getSimilar(...)` |
| `POST` | `/trust` | JSON `{ image_base64, bbox, predicted_class?, top_k? }` | `{ query, trust }` | nested `INVALID_REQUEST`, `INVALID_IMAGE`, `INVALID_BBOX` | `getTrust(...)` |
| `POST` | `/save-annotation` | JSON `{ analysis_id, image_data_url, annotations, image_name?, timestamp? }` | `{ status: "ok", analysis_id, saved_count, classes, saved_at? }` | save validation/simple or storage/internal flat errors | `saveAnnotation(payload)` |
| `GET` | `/admin/annotations` | none | local-only admin review queue with counts, grouped analyses, diagnostics | flat `{ status: "error", error }` for server failures | `getAdminAnnotationQueue()` |
| `POST` | `/admin/annotations/<analysis_id>/<index>/review` | JSON `{ status: "pending" \| "approved" \| "rejected" }` | `{ status: "ok", element, warning, local_only }` | flat `{ status: "error", error }` for validation/not-found | `setAdminAnnotationReviewStatus(...)` |
| `POST` | `/admin/annotations/<analysis_id>/<index>/modify` | JSON `{ class_name, bbox, approve_after_save?, status? }` | `{ status: "ok", element, counts, warning, local_only }` | flat `{ status: "error", error }` for validation/not-found | `modifyAdminAnnotationElement(...)` |
| `GET` | `/admin/annotations/<analysis_id>/image` | none | local original image bytes | flat `{ status: "error", error }` for validation/not-found | `adminAnnotationMediaUrl(...)` |
| `GET` | `/admin/annotations/<analysis_id>/<index>/crop` | none | local crop image bytes | flat `{ status: "error", error }` for validation/not-found | `adminAnnotationMediaUrl(...)` |
| `GET` | `/admin/training/summary` | none | local-only approved-data summary, parameters, paths, artifacts, latest job, launch guard reasons | flat `{ status: "error", error }` only for unexpected route failures | `getAdminTrainingSummary()` |
| `GET` | `/admin/training/jobs/latest` | none | `{ status: "ok", local_only: true, job }` where `job` may be `null` | flat `{ status: "error", error }` only for unexpected route failures | `getLatestAdminTrainingJob()` |
| `GET` | `/admin/training/jobs/<run_id>` | none | `{ status: "ok", local_only: true, job }` | flat `{ status: "error", error }` for invalid/not-found run id | none |
| `POST` | `/admin/training/jobs` | JSON `{ dry_run, device, batch_size, notes? }` | `202 { status: "ok", local_only: true, job }` | flat `{ status: "error", error }` for disabled/nonlocal/validation/conflict | `startAdminTrainingJob(...)` |

## Shared Classifier Payload

Classifier-like responses contain:

```json
{
  "class_name": "atl",
  "class_label": "Water",
  "confidence": 0.72,
  "rejected": false,
  "top_k": [
    { "class_name": "atl", "class_label": "Water", "confidence": 0.72 }
  ]
}
```

Compatibility notes:

- `class_label` may be a display string, a numeric label, `null`, or absent depending on classifier/stub path.
- `top_k[]` items may include `class_label` with the same compatibility shape.
- Frontend types must not force `class_label` to a single numeric type.

## Endpoint Details

### `GET /health`

Success:

```json
{ "status": "ok" }
```

### `GET /classes`

Success:

```json
{ "num_classes": 2, "class_names": ["atl", "tochtli"] }
```

### `POST /classify`

Request: multipart form with field `image`.

Missing image error:

```json
{ "error": "No 'image' file in request" }
```

Invalid uploaded image bytes:

```json
{ "error": { "code": "INVALID_IMAGE", "message": "uploaded file is not a valid image" } }
```

Success: shared classifier payload.

### `POST /classify-batch`

Request: multipart form with one or more `images` fields.

Missing images error:

```json
{ "error": "No 'images' files in request" }
```

Invalid uploaded image bytes in any submitted file:

```json
{ "error": { "code": "INVALID_IMAGE", "message": "uploaded file is not a valid image" } }
```

Success: array of shared classifier payloads.

### `POST /segment`

Request: multipart form with field `image`.

Invalid uploaded image bytes:

```json
{ "error": { "code": "INVALID_IMAGE", "message": "uploaded file is not a valid image" } }
```

Success:

```json
{
  "num_elements": 1,
  "image_size": [8, 6],
  "elements": [
    {
      "bbox": [1, 2, 3, 4],
      "class_name": "atl",
      "confidence": 0.72,
      "rejected": false,
      "top_k": []
    }
  ]
}
```

Compatibility notes:

- `image_size` order is `[width, height]`.
- `bbox` order is `[x, y, width, height]`.
- Backend currently classifies each accepted crop individually.

### `POST /similar`

Request:

```json
{
  "image_base64": "...",
  "bbox": [0, 0, 4, 4],
  "limit": 5,
  "mode": "prototype"
}
```

Success:

```json
{
  "query": { "bbox": [0, 0, 4, 4], "mode": "prototype" },
  "best_match": { "class_name": "atl", "similarity": 0.72, "rejected": false },
  "results": [
    {
      "rank": 1,
      "match_type": "class_prototype",
      "class_name": "atl",
      "class_label": "Water",
      "similarity": 0.72,
      "band": "high",
      "asset": null
    }
  ]
}
```

Validation errors use nested error objects:

```json
{ "error": { "code": "INVALID_REQUEST", "message": "image_base64 and bbox required" } }
```

Other known codes: `INVALID_IMAGE`, `INVALID_BBOX`.

### `POST /trust`

Request:

```json
{
  "image_base64": "...",
  "bbox": [0, 0, 4, 4],
  "predicted_class": "atl",
  "top_k": 10
}
```

Success:

```json
{
  "query": { "bbox": [0, 0, 4, 4], "predicted_class": "atl" },
  "trust": {
    "predicted_class_rank": 1,
    "predicted_class_similarity": 0.72,
    "top1_class": "atl",
    "top1_similarity": 0.72,
    "margin_to_second": 0.41,
    "above_rejection_threshold": true,
    "rejection_threshold": 0.35,
    "ambiguous": false,
    "entropy": 0.61,
    "top_k": []
  }
}
```

Validation errors match `/similar`.

### `POST /save-annotation`

Request:

```json
{
  "analysis_id": "analysis-001",
  "image_name": "glyph.png",
  "image_data_url": "data:image/png;base64,...",
  "timestamp": 1770000000000,
  "annotations": [
    { "index": 0, "bbox": [0, 0, 5, 5], "class_name": "atl" }
  ]
}
```

Compatibility notes:

- Backend currently requires `analysis_id`, `image_data_url`, and non-empty `annotations`.
- Frontend also sends `image_name` and `timestamp`; backend route logic currently ignores them.
- This phase must not migrate or alter backend storage.

Success:

```json
{
  "status": "ok",
  "analysis_id": "analysis-001",
  "saved_count": 1,
  "classes": ["atl"],
  "saved_at": "2026-05-22T00:00:00+00:00"
}
```

`saveAnnotation` frontend behavior:

- Resolves `{ ok: true, ...response }` on success.
- Resolves `{ ok: false, error_code, message, hint?, trace_id? }` for save HTTP/storage/network failures.
- Does not throw for expected save failures.

Saved elements enter local admin review as `pending`; they are not retraining-eligible until approved through `/admin/annotations`.

Validation errors:

```json
{ "status": "error", "error": "invalid JSON" }
{ "status": "error", "error": "missing field: analysis_id" }
{ "status": "error", "error": "annotations must be a non-empty list" }
```

Storage/internal errors:

```json
{
  "status": "error",
  "error_code": "PERMISSION_DENIED",
  "message": "Droits insuffisants sur le dossier annotations",
  "hint": "Vérifiez les permissions du dossier backend/annotations/ ou lancez l'application avec un compte ayant accès en écriture."
}
```

Known save error codes for the frontend result union:

- `PERMISSION_DENIED`
- `DISK_FULL`
- `STORAGE_ERROR`
- `INTERNAL_ERROR`
- `NETWORK_ERROR` (client-side fallback)

### Local admin annotation review

These endpoints are **local/dev-only** and **not production-secured**. They do not implement auth, roles, tokens, model registry, or rollback semantics. Training launch/status lives under `/admin/training/*` and is separately disabled by default plus loopback guarded.

`GET /admin/annotations` success:

```json
{
  "status": "ok",
  "schema_version": 1,
  "local_only": true,
  "warning": "Local/dev-only annotation review endpoint. It is not production-secured...",
  "counts": { "total": 2, "pending": 1, "approved": 1, "rejected": 0, "trainable": 1 },
  "analyses": [
    {
      "analysis_id": "analysis-001",
      "uploaded_at": "2026-05-26T13:00:00+00:00",
      "image_path": ".../backend/annotations/analysis-001/image.png",
      "image_url": "/admin/annotations/analysis-001/image",
      "image_exists": true,
      "elements": [
        {
          "key": "analysis-001:0",
          "analysis_id": "analysis-001",
          "index": 0,
          "class_name": "atl",
          "bbox": [0, 0, 5, 5],
          "crop_url": "/admin/annotations/analysis-001/0/crop",
          "review_status": "approved",
          "trainable": true,
          "source_fingerprint": "sha256:...",
          "stale_decision": false
        }
      ]
    }
  ],
  "diagnostics": []
}
```

`POST /admin/annotations/<analysis_id>/<index>/review` mutates one element-level decision:

```json
{ "status": "approved" }
```

Allowed statuses are `pending`, `approved`, and `rejected`. Retraining includes only canonical, non-stale elements whose status is exactly `approved`; orphan, stale, or missing-crop entries are diagnosed and excluded.

`POST /admin/annotations/<analysis_id>/<index>/modify` corrects one submitted element and regenerates canonical crop evidence from the stored source image:

```json
{
  "class_name": "edited-atl",
  "bbox": [1, 2, 5, 6],
  "approve_after_save": false
}
```

Compatibility notes:

- `class_name` is required, sanitized with the same class-name rules as save/export, and may not contain path separators or `..`.
- `bbox` is required in `[x, y, width, height]` image-pixel order, normalized with backend integer rounding, clamped to image bounds, and rejected if width/height become non-positive.
- Without `approve_after_save` or `status`, a modified element becomes `pending` and is not trainable.
- `approve_after_save: true` is equivalent to a fresh `approved` decision. If `status` is also provided it must be `"approved"`.
- The response shape matches the review mutation response and includes fresh queue `counts`.

Success:

```json
{
  "status": "ok",
  "local_only": true,
  "warning": "Local/dev-only annotation review endpoint...",
  "element": {
    "key": "analysis-001:0",
    "analysis_id": "analysis-001",
    "index": 0,
    "class_name": "edited-atl",
    "bbox": [1, 2, 5, 6],
    "review_status": "pending",
    "trainable": false,
    "crop_url": "/admin/annotations/analysis-001/0/crop"
  },
  "counts": { "total": 1, "pending": 1, "approved": 0, "rejected": 0, "trainable": 0 }
}
```

Validation/not-found errors use the flat admin shape, for example:

```json
{ "status": "error", "error": "missing field: class_name" }
{ "status": "error", "error": "width must be positive" }
{ "status": "error", "error": "annotation not found: analysis-001:99" }
```

### Local admin training summary and jobs

These endpoints are **local/dev-only** and **not production-secured**. Summary is visible to local admin UI callers, but launching a job is disabled unless `ENABLE_ADMIN_TRAINING_JOBS=1` and request locality checks pass. The job runner is not a shell executor; it starts only the allowlisted command `bash scripts/retrain.sh` with optional `--dry-run` and an allowlisted environment (`BATCH_SIZE`, `DEVICE`, `PYTHONUNBUFFERED`, plus process `PATH`/`HOME`).

`GET /admin/training/summary` success:

```json
{
  "status": "ok",
  "local_only": true,
  "warning": "Local/dev-only annotation review endpoint...",
  "training_jobs_enabled": false,
  "launch_allowed_for_request": false,
  "launch_disabled_reasons": [
    "disabled_by_default: set ENABLE_ADMIN_TRAINING_JOBS=1 to allow local launches"
  ],
  "data": {
    "total": 2,
    "pending": 1,
    "approved": 1,
    "rejected": 0,
    "trainable": 1,
    "classes": ["atl"],
    "per_class": { "atl": 1 },
    "diagnostics": []
  },
  "parameters": {
    "editable": {
      "dry_run": true,
      "device": ["auto", "cpu", "mps", "cuda"],
      "batch_size": { "default": 16, "min": 1, "max": 256 }
    },
    "script_env_defaults": { "BATCH_SIZE": "16", "DEVICE": "auto" },
    "config": {}
  },
  "paths": {
    "script": ".../scripts/retrain.sh",
    "approved_elements_dir": ".../backend/training_data/approved/Elements",
    "runs_dir": ".../backend/training_runs"
  },
  "artifacts": {
    "classifier_config": { "path": ".../backend/codex_model/config.json", "exists": true }
  },
  "latest_job": null
}
```

`POST /admin/training/jobs` request:

```json
{
  "dry_run": true,
  "device": "cpu",
  "batch_size": 8,
  "notes": "optional note, 200 chars max"
}
```

Defaults are `dry_run: false`, `device: "auto"`, and `batch_size: 16` when fields are omitted. Accepted devices are `auto`, `cpu`, `mps`, and `cuda`; `batch_size` must be an integer from 1 through 256.

Accepted response (`202`):

```json
{
  "status": "ok",
  "local_only": true,
  "job": {
    "run_id": "20260526T150000Z-ab12cd34",
    "status": "running",
    "dry_run": true,
    "device": "cpu",
    "batch_size": 8,
    "notes": "optional note",
    "started_at": "2026-05-26T15:00:00+00:00",
    "finished_at": null,
    "exit_code": null,
    "pid": 4242,
    "command": ["bash", ".../scripts/retrain.sh", "--dry-run"],
    "cwd": ".../clinic-codex",
    "env": { "BATCH_SIZE": "8", "DEVICE": "cpu", "PYTHONUNBUFFERED": "1" },
    "log_path": ".../backend/training_runs/20260526T150000Z-ab12cd34/train.log",
    "approved_export_manifest_hash": null,
    "log_tail": [],
    "artifacts": {}
  }
}
```

`GET /admin/training/jobs/latest` returns the latest hydrated job or `job: null`. `GET /admin/training/jobs/<run_id>` returns `404` with `{ "status": "error", "error": "training job not found: <run_id>" }` when absent/invalid. Hydrated jobs include `log_tail` and current artifact metadata; a running job may be marked `succeeded` or `failed` on read when its process handle exits in the current backend process.

Known `POST /admin/training/jobs` errors:

| HTTP | Error text includes | Meaning |
| --- | --- | --- |
| `400` | `invalid JSON` | Body is not a JSON object. |
| `400` | `unknown field(s)` | Payload included fields outside `dry_run`, `device`, `batch_size`, `notes`. |
| `400` | `device must be one of` | Device is not in the allowed device list. |
| `400` | `batch_size must be between` | Batch size is outside configured limits. |
| `403` | `disabled_by_default` | `ENABLE_ADMIN_TRAINING_JOBS` was not enabled for this backend process. |
| `403` | `non_loopback_remote_addr` | Request did not originate from loopback. |
| `403` | `nonlocal_host` | `Host` was not loopback/localhost. |
| `403` | `nonlocal_origin` | `Origin` was not local. |
| `409` | `training job already running` | A latest job is still `running`; concurrent starts are rejected. |

Retraining remains approved-only: the script first materializes `backend/training_data/approved/Elements` through `scripts/export_approved_annotations.py`, then runs classifier-only stages. No MobileSAM/segmentation retraining is exposed, and new weights are not hot-reloaded; restart the backend after a successful full run.

## Legacy Endpoints

When legacy endpoints are enabled, backend also registers:

- `GET /sample-image?path=...`
- `POST /similar-samples`

`/similar-samples` uses the same nested validation errors as `/similar` and `/trust`. These endpoints are compatibility endpoints and are not currently exported by `frontend/src/services/api.ts`.

## Test Backing

The contract is enforced by backend route tests and Phase 3A contract tests under `backend/tests/`, plus frontend API client characterization tests under `frontend/src/services/api.test.ts`.

Any future endpoint behavior change must update this document and the relevant contract tests in the same phase.
