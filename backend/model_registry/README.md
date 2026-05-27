# Local model registry

This directory is the repo-local, dependency-light model registry for the
classifier.

- `index.json` is generated local state and records known versions and aliases.
- `versions/<version_id>/` contains immutable candidate/original packages.
- `snapshots/<timestamp>-pre-promotion/` contains runtime backups created before
  explicit promotion or rollback.
- `promotion_in_progress.json` is left behind only when promotion is interrupted
  before the registry can clear the recovery marker.

Generated registry state is ignored by git except for placeholder files and this
README. Training/export must create candidates here first; only
`scripts/promote_model.py` is allowed to update `backend/codex_model/` runtime
artifacts during the versioned retraining workflow.
