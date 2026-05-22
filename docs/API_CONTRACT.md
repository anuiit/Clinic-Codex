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
   - `/similar`, `/trust`, and legacy `/similar-samples` validation errors.
3. Save storage/internal errors: `{ "status": "error", "error_code": string, "message": string, "hint"?: string | null, "trace_id"?: string }`
   - `/save-annotation` storage permission, disk, generic storage, and unknown internal failures.

## Active Endpoints

| Method | Path | Request | Success body | Error bodies | Frontend export |
| --- | --- | --- | --- | --- | --- |
| `GET` | `/health` | none | `{ status: "ok" }` | Flask/default errors only | none |
| `GET` | `/classes` | none | `{ num_classes: number, class_names: string[] }` | transport/HTTP failure | `getClasses()` |
| `POST` | `/classify` | multipart form field `image` | classifier payload | `{ error: "No 'image' file in request" }` | `classifyElement(file)` |
| `POST` | `/classify-batch` | multipart form field(s) `images` | classifier payload array | `{ error: "No 'images' files in request" }` | none |
| `POST` | `/segment` | multipart form field `image` | `{ num_elements, image_size, elements }` | `{ error: "No 'image' file in request" }` | `segmentGlyph(file)` |
| `POST` | `/similar` | JSON `{ image_base64, bbox, limit?, mode? }` | `{ query, best_match, results }` | nested `INVALID_REQUEST`, `INVALID_IMAGE`, `INVALID_BBOX` | `getSimilar(...)` |
| `POST` | `/trust` | JSON `{ image_base64, bbox, predicted_class?, top_k? }` | `{ query, trust }` | nested `INVALID_REQUEST`, `INVALID_IMAGE`, `INVALID_BBOX` | `getTrust(...)` |
| `POST` | `/save-annotation` | JSON `{ analysis_id, image_data_url, annotations, image_name?, timestamp? }` | `{ status: "ok", analysis_id, saved_count, classes, saved_at? }` | save validation/simple or storage/internal flat errors | `saveAnnotation(payload)` |

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

Success: shared classifier payload.

### `POST /classify-batch`

Request: multipart form with one or more `images` fields.

Missing images error:

```json
{ "error": "No 'images' files in request" }
```

Success: array of shared classifier payloads.

### `POST /segment`

Request: multipart form with field `image`.

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

## Legacy Endpoints

When legacy endpoints are enabled, backend also registers:

- `GET /sample-image?path=...`
- `POST /similar-samples`

`/similar-samples` uses the same nested validation errors as `/similar` and `/trust`. These endpoints are compatibility endpoints and are not currently exported by `frontend/src/services/api.ts`.

## Test Backing

The contract is enforced by backend route tests and Phase 3A contract tests under `backend/tests/`, plus frontend API client characterization tests under `frontend/src/services/api.test.ts`.

Any future endpoint behavior change must update this document and the relevant contract tests in the same phase.
