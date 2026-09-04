# Clinic Codex Backend

Flask backend for Clinic Codex. It serves the local codex classifier, MobileSAM segmentation, similarity/trust helpers, class names, submitted annotation persistence, and a local-only admin review gate for retraining eligibility.

Default runtime: `http://localhost:7117` (`HOST=127.0.0.1`, `PORT=7117`).

## Run

```bash
cd backend
pip install -r requirements.txt
python -m flask --app backend.wsgi run --host 127.0.0.1 --port 7117
```

The repository-level dev helper starts backend and frontend together:

```bash
bash scripts/run-dev.sh
```

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `PORT` | `7117` | Flask port. |
| `HOST` | `127.0.0.1` | Flask bind host for the Phase 1 local-only workflow. |
| `CORS_ORIGINS` | `http://localhost:7118` | Comma-separated browser origins. |
| `MAX_IMAGE_PIXELS` | `80000000` | Maximum decoded image pixels for upload and base64/data-url paths. |
| `MAX_IMAGE_DIMENSION` | `10000` | Maximum decoded image width or height in pixels. |
| `MODEL_DIR` | auto-detected | Optional classifier weights/config directory. |
| `ENABLE_LEGACY_ENDPOINTS` | `true` | Keep sample/demo endpoints `/sample-image` and `/similar-samples` available during Phase 1 compatibility. |
| `ENABLE_ADMIN_TRAINING_JOBS` | `false` | Enables the local-only `/admin/training/jobs` launcher when the request also passes loopback Host/Origin guards. |
| `ADMIN_TRAINING_SNAPSHOT_DIR` | unset | Optional advanced cumulative snapshot. Leave unset for base + current approvals. |
| `ADMIN_TRAINING_BACKBONE_MANIFEST` | local backbone pin | Optional absolute path to the DINOv2 pin manifest. |

Requests are capped at **50 MB** via Flask `MAX_CONTENT_LENGTH`; decoded images are also capped by `MAX_IMAGE_PIXELS` and `MAX_IMAGE_DIMENSION`.

## API endpoints

| Endpoint | Method | Input | Output |
| --- | --- | --- | --- |
| `/segment` | `POST` | multipart form field `image` | `{ num_elements, image_size: [w,h], elements: [{ bbox, class_name, confidence, rejected, top_k, ... }] }` |
| `/classes` | `GET` | none | `{ num_classes, class_names }` from `backend/codex_model/config.json` |
| `/similar` | `POST` | JSON `{ image_base64, bbox, limit }` | prototype similarity ranking for the crop |
| `/trust` | `POST` | JSON `{ image_base64, bbox, predicted_class, top_k }` | trust signals: predicted rank/similarity, top1, margin, ambiguity, entropy |
| `/save-annotation` | `POST` | JSON payload below | saves submitted annotation crops under `backend/annotations/<analysis_id>/` as pending admin review |
| `/admin/annotations` | `GET` | none | local/dev-only grouped admin review queue, counts, diagnostics, media URLs |
| `/admin/annotations/<analysis_id>/<index>/review` | `POST` | `{ "status": "approved" \| "rejected" \| "pending" }` | persists one element-level review decision |
| `/admin/annotations/<analysis_id>/<index>/modify` | `POST` | `{ "class_name": string, "bbox": [x,y,w,h], "approve_after_save"?: boolean, "status"?: "approved" \| "rejected" \| "pending" }` | rewrites metadata/crop evidence and records a fresh review decision |
| `/admin/annotations/<analysis_id>/image` | `GET` | none | local original image for visual review |
| `/admin/annotations/<analysis_id>/<index>/crop` | `GET` | none | local crop image for visual review |
| `/admin/training/summary` | `GET` | none | local-only base/approvals or advanced snapshot summary, script/config paths, artifact state, launch guard reasons, latest job |
| `/admin/training/jobs/latest` | `GET` | none | latest local training job, or `null` |
| `/admin/training/jobs/<run_id>` | `GET` | none | one local training job with log tail and artifact state |
| `/admin/training/jobs` | `POST` | `{ "dry_run": boolean, "device": "auto" \| "cpu" \| "mps" \| "cuda", "batch_size": number, "notes"?: string }` | starts the local Python job or explicitly configured snapshot script when enabled and local |

Compatibility endpoints `/classify` and `/classify-batch` remain available and tested during Phase 1. Sample/demo endpoints `/sample-image` and `/similar-samples` remain enabled by default (`ENABLE_LEGACY_ENDPOINTS=true`); set `ENABLE_LEGACY_ENDPOINTS=false` to hide only those sample/demo endpoints while retaining active frontend endpoints. `backend/examples/flask_api.py` is now a thin compatibility runner for the modular `backend.wsgi` app, not the primary route implementation.

The `/admin/annotations` and `/admin/training` endpoints are **Phase 1 local-only** and **not shared-production auth**. Current admin annotation routes, training summary, and training job status routes reject non-loopback/remote Host/Origin requests with `403 LOCAL_ONLY_FORBIDDEN`. `POST /admin/training/jobs` also keeps the disabled-by-default launch guard. Phase 1 intentionally does not add roles, sessions, tokens, reverse-proxy trust, or production authorization. Model registry and rollback semantics are local filesystem safety tooling only.

CORS is allowlist-only: unknown origins and requests without `Origin` do not receive `Access-Control-Allow-Origin`.

## `/save-annotation` payload

The frontend sends only browser-validated, named elements. Bboxes are image-pixel coordinates in `[x, y, width, height]` format. Browser validation means ready for local admin review; it does not make an element trainable until an admin approves it.

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

- `400 VALIDATION_ERROR`: invalid JSON, missing field, empty annotations, invalid image data URL, image above configured decode limits, invalid class name, invalid bbox. The response includes both `message` and legacy `error`.
- `409 PERMISSION_DENIED`: backend cannot write to `backend/annotations/`.
- `413`: request exceeds 50 MB.
- `507 DISK_FULL`: no disk space.
- `500 STORAGE_ERROR` or `INTERNAL_ERROR`: unexpected storage/server failure.

## Annotation storage

`backend/services/annotation_storage.py` is the canonical source of submitted annotation evidence.

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
- normalizes every bbox value with Python `round()` and `int()` before clamp/crop/save, so metadata and saved crop pixels use the same integer bbox;
- clamps bboxes to the image bounds and rejects zero/negative crop dimensions;
- writes through a temporary directory before replacing the target folder;
- no longer writes directly to `training_data/Elements` (the parameter remains for backward compatibility).

Admin review decisions are stored separately in `backend/annotations/review-index.json` by `backend/services/annotation_review.py`. Review decisions are element-level (`analysis_id:index`), while analyses are display groups. Existing annotations with no review entry appear as `pending`; only canonical, non-stale elements whose review status is exactly `approved` are yielded for retraining. Missing folders, metadata rows, crops, orphan manifest entries, and stale source fingerprints are diagnosed and excluded from training.

`POST /admin/annotations/<analysis_id>/<index>/modify` can correct a submitted element after review. It validates/sanitizes `class_name`, clamps and normalizes `[x, y, width, height]`, regenerates the crop from the stored `image.png`, rewrites metadata only after the crop exists, and records a fresh manifest decision. By default, a modification becomes `pending`; pass `approve_after_save: true` (or `status: "approved"`) only when the corrected evidence should be immediately trainable.

## Image upload errors

`/segment`, `/classify`, and `/classify-batch` validate uploaded image bytes before calling model services. A corrupt or non-image multipart file returns HTTP `400`:

```json
{
  "error": {
    "code": "INVALID_IMAGE",
    "message": "uploaded file is not a valid image"
  }
}
```

Missing multipart fields keep the legacy plain-string error responses documented by route tests.

## DINOv2 cache and offline operation

The classifier backbone is loaded through PyTorch Hub. On connected machines, pre-cache DINOv2 before launching an offline backend:

```bash
python - <<'PY'
import torch

print("torch hub cache:", torch.hub.get_dir())
torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
PY
```

PyTorch Hub uses `~/.cache/torch/hub` by default. Set `TORCH_HOME=/path/to/cache-root` to relocate the cache; the hub directory will be under that root. For offline operation, both the DINOv2 hub backbone and the local classifier assets (`prototypes.pt`, `projection.pt`, and config/model files) must already be present before launch.

If local classifier assets are absent, backend ML requests fail with HTTP `503` and error code `MODEL_ASSET_UNAVAILABLE`. If the DINOv2 hub backbone is absent from the cache while offline, PyTorch Hub/model initialization is expected to fail because the backend does not download or vendor the backbone in offline mode.

## Tests

```bash
backend/.venv/bin/python -m pytest backend/tests scripts/test_export_annotations.py scripts/test_export_approved_annotations.py scripts/test_retrain_scripts.py scripts/test_promote_model.py
```

## Retraining

The standard local mode combines the **shipped model base and all current approved annotations**. It needs no private corpus or manual snapshot. The installer downloads fixed MobileSAM and DINOv2 assets, enables local training, and permits the initial local administrator to review their own annotations.

1. Upload and analyze an image, open its annotation editor, correct boxes and labels, and mark the desired elements ready.
2. Send the annotations, then open **Admin → Review** and approve them.
3. Open **Training**, run the dry run, then select **Non, entraînement complet** and launch.
4. Inspect the candidate path and result in Training.

Each run captures current approved crops and review decisions automatically. Exact duplicate images count once; conflicting labels for identical images are rejected. Stale decisions and missing crops are excluded. Repeating the same approvals does not count their contribution twice.

The backbone and projection stay frozen. The update adapts prototypes for existing base-model classes; it does not train MobileSAM or introduce new classes. The original base provides the prior even when its training images are unavailable.

Candidates are stored under `backend/model_registry/versions/<version_id>/`, with provenance and checksums. They are **not activated**; promotion is blocked because this local mode has no independent holdout. Reported base/candidate scores measure training-image fit, not better generalization. The running model is unchanged and no restart is needed.

The normal Training tab uses `scripts/retrain_local.py` through the backend Python on Linux/macOS and native Windows. Jobs are guarded against concurrent launches. Status and logs are stored under `backend/training_runs/`. Unknown fields, invalid parameters, nonlocal requests and incompatible `MODEL_DIR` overrides are rejected.

See [the operator guide](../docs/admin-model-retraining-workflow.md) for permissions and advanced snapshot mode.
