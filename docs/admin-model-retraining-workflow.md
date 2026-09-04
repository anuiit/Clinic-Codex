# Local annotation and retraining workflow

Updated: 2026-09-04

The standard local mode combines the **shipped model base and all current approved annotations**. It needs no private corpus or manual snapshot. The installer downloads fixed MobileSAM and DINOv2 assets, enables local training, and permits the initial local administrator to review their own annotations.

1. Upload and analyze an image, open its annotation editor, correct boxes and labels, and mark the desired elements ready.
2. Send the annotations, then open **Admin → Review** and approve them.
3. Open **Training**, run the dry run, then select **Non, entraînement complet** and launch.
4. Inspect the candidate path and result in Training.

Each run captures current approved crops and review decisions automatically. Exact duplicate images count once; conflicting labels for identical images are rejected. Stale decisions and missing crops are excluded. Repeating the same approvals does not count their contribution twice.

The backbone and projection stay frozen. The update adapts prototypes for existing base-model classes; it does not train MobileSAM or introduce new classes. The original base provides the prior even when its training images are unavailable.

Candidates are stored under `backend/model_registry/versions/<version_id>/`, with provenance and checksums. They are **not activated**; promotion is blocked because this local mode has no independent holdout. Reported base/candidate scores measure training-image fit, not better generalization. The running model is unchanged and no restart is needed.

## Local permissions

The installer adds `ENABLE_ADMIN_TRAINING_JOBS=true` and `ALLOW_LOCAL_ADMIN_SELF_REVIEW=true` to the ignored `backend/.env`, preserving existing settings. Self-review requires the initial administrator, a loopback-bound backend and a local request. Session, role and CSRF checks still apply; the reviewer identity is recorded. Other accounts and configurations retain independent-review requirements.

## Existing annotations

Canonical folders under `backend/annotations/` appear in Review; their annotations need not be recreated. Missing or changed source files require repair and another review. Use model class-list labels for retraining. Custom labels can be saved, but this fixed-taxonomy local mode rejects unknown classes.

## Installation and recovery

Follow [INSTALL.md](../INSTALL.md). Re-running the installer preserves credentials and annotations. MobileSAM is downloaded through a temporary file, verified against a fixed SHA-256, then installed atomically. A failed download preserves the existing checkpoint. The startup smoke checks assets through `/ready`.

If a run fails, inspect its error/logs in Training, correct the issue, and retry. Current approvals are captured again.

## Advanced corpus mode

Setting `ADMIN_TRAINING_SNAPSHOT_DIR` explicitly selects the existing cumulative-corpus workflow. It requires a prepared snapshot and rejects stale evidence. Leave it unset for the standard workflow. A `MODEL_DIR` override blocks the admin launch to avoid incompatible model packages.

## Verification

Fast checks: backend pytest, frontend lint/test/build and Playwright.
`CLINIC_LOCAL_RETRAIN_E2E=1` enables real CPU inference and authenticated dry/full runs using bundled images.
`CLINIC_LIVE_E2E=1` enables the browser journey against a freshly installed backend. Run it in a disposable checkout: it creates an account, annotation and candidate.
