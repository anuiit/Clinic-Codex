# Admin model retraining and model-management workflow

Date: 2026-07-07

This note captures the current decisions and analysis about making Clinic Codex support a complete admin workflow for model retraining, model management, manual testing, comparison, promotion, and rollback.

## 1. Product goal

The goal is that admins can improve the model over time through the application:

1. users/admins create or correct annotations;
2. admins review and approve those annotations;
3. the approved annotations are accumulated into the training corpus;
4. admins launch retraining from the admin UI;
5. a candidate model is produced;
6. admins manually test and compare the candidate against the active model;
7. admins explicitly promote the candidate if acceptable;
8. admins can rollback if needed.

The intended workflow is:

```text
Review / Dataset
→ Training preflight
→ Dry-run
→ Full retrain
→ Candidate registry
→ Models tab
→ Manual test
→ Compare active vs candidate
→ Explicit promotion
→ Restart / smoke validation
→ Rollback if needed
```

## 2. Current behavior

The current training script does **not** fine-tune from the last promoted/best model by default.

It does this instead:

```text
DINOv2 pretrained frozen backbone
+ approved annotations exported for this run
→ recompute features
→ train a new projection head from scratch
→ recompute prototypes
→ export a candidate model package
```

Important details:

- `scripts/retrain.sh` calls `train.py` without `--resume`.
- `train.py` supports `--resume`, but the current retrain workflow does not use it.
- DINOv2 is used as a pretrained frozen feature extractor.
- The trainable part is mainly the small projection head, and prototypes are recalculated.
- The candidate is written under `backend/model_registry/versions/<version_id>/`.
- The runtime model under `backend/codex_model/` is not overwritten by training.
- Promotion is currently CLI-only through `scripts/promote_model.py`.

So the current behavior is best described as:

```text
retrain from scratch on the currently approved training data,
using a frozen pretrained DINOv2 backbone.
```

It is **not** currently:

```text
continue training from the last promoted model.
```

## 3. Does retraining need the original dataset?

Technically, the current script can run without the original dataset, because it uses approved annotations from the app.

However, for the product goal of improving the model over time, we need a **cumulative dataset**. If retraining uses only newly approved annotations, the model can forget old classes or degrade on old cases.

The correct product strategy is:

```text
training dataset for each retrain
= initial validated dataset
+ all approved user/admin annotations
+ all approved corrections
```

Therefore:

- the original dataset is not technically required by the script;
- but a validated initial/reference corpus is practically required if we want continuous improvement without forgetting;
- every retrain should use the full cumulative approved corpus, not just the new annotations.

## 4. What needs to exist in the repo

### 4.1 Cumulative training corpus

The repo needs a durable source of truth for training data.

Today, `backend/training_data/approved/` is mostly a generated output. The repo needs a clearer canonical corpus, for example:

```text
backend/training_corpus/
  approved/
  rejected/
  manifests/
  snapshots/
```

Or the existing `backend/annotations/` system must be formalized as the canonical source.

Required properties:

- supports initial dataset import;
- supports user/admin annotations;
- tracks review status;
- tracks source, e.g. `initial_dataset`, `user_annotation`, `admin_correction`;
- can produce a versioned dataset snapshot for a training run;
- records class counts and split counts;
- lets us answer: “which exact dataset trained this model?”

### 4.2 Initial dataset import

A script should import the original/reference dataset into the same review/training system:

```bash
scripts/import_initial_training_dataset.py
```

Expected behavior:

- convert the original dataset to the app's annotation/corpus format;
- mark examples as approved or reviewable;
- preserve class names and bounding boxes/crops;
- mark provenance as `source=initial_dataset`;
- generate a manifest/hash for reproducibility.

### 4.3 Dataset snapshots

Every full retrain should be tied to an immutable dataset snapshot:

```text
dataset_snapshot_id
manifest_hash
class_counts
split_counts
config_hash
```

This snapshot should be referenced by the model version manifest.

### 4.4 Training preflight

Before launching full training, the backend must validate whether the dataset can actually train with the current config.

The recent live full-training failure proved the current guard is too weak: it only checks that some trainable annotations exist, not whether the dataset satisfies few-shot requirements.

The preflight must check:

- number of images per class;
- `data.min_images_per_class`;
- train/val/test split viability;
- `training.k_shot`;
- `training.q_queries`;
- `evaluation.k_shot_values`;
- whether enough classes remain after filtering.

It should return structured reasons such as:

```text
class_comitl_val_insufficient: 1 sample < k_shot=3
not_enough_trainable_classes
class_filtered_by_min_images_per_class
```

### 4.5 Model registry UI/API

The backend already has a model registry foundation in `backend/services/model_registry.py`, but the admin UI needs a full model-management surface.

Recommended new admin tab:

```text
/admin/annotations/models
```

The existing Training tab should stay focused on launching training and viewing logs. The new Models tab should handle lifecycle management.

### 4.6 Admin Models tab

The Models tab should include:

1. active runtime card;
2. promoted version vs loaded version;
3. `restart_required` if the promoted model differs from the loaded one;
4. complete version table;
5. candidate inspection;
6. manual test bench;
7. active vs candidate comparison;
8. promotion controls;
9. rollback controls;
10. audit/history.

### 4.7 Backend APIs

All model-management endpoints should be local-only and guarded.

Recommended endpoints:

```text
GET  /admin/models
GET  /admin/models/versions
GET  /admin/models/versions/<version_id>
GET  /admin/models/runtime
POST /admin/models/versions/<version_id>/test
POST /admin/models/compare
POST /admin/models/versions/<version_id>/promote
POST /admin/models/rollback
GET  /admin/models/runtime/smoke
```

Recommended backend services:

```text
TrainingPreflightService
ModelManagementService
ModelEvaluationService
ModelSandboxInferenceService
PromotionService
```

## 5. Which models should be trained after accumulating annotations?

### 5.1 Primary model: glyph/element classifier

This is the main model to improve over time.

It consists of:

```text
DINOv2 frozen backbone
+ projection head
+ prototypes
```

For each retrain, we should train/update:

```text
projection head
+ prototypes
```

using:

```text
full cumulative approved dataset
```

not only new annotations.

### 5.2 Do not fine-tune DINOv2 initially

DINOv2 should remain frozen at first.

Reasons:

- requires less data;
- is more stable;
- is faster;
- reduces overfitting risk;
- current pipeline is designed around cached DINOv2 features.

Fine-tuning DINOv2 should be considered later only if the corpus becomes large enough and evaluation proves it is useful.

### 5.3 Calibration / thresholds

A second important output is calibration, not necessarily a heavy model.

Each model version should include or reference calibrated values such as:

```text
rejection_threshold
unknown threshold
top-k confidence behavior
possibly per-class thresholds
```

This could be stored as:

```text
calibration.json
```

The calibration should be computed from validation/test data and tied to the model version.

### 5.4 Segmentation/detection model

Currently `scripts/retrain.sh` explicitly does **not** retrain MobileSAM or segmentation.

For now:

```text
MobileSAM / segmentation: not trained
```

Short-term improvements should focus on:

- thresholds;
- post-processing;
- grouping;
- bbox/crop quality;
- proposal filtering.

A segmentation/detection model can be trained later if we accumulate enough corrected bounding boxes or masks.

## 6. Why the previous full training failed

The dry-runs inspected locally succeeded. The full training failed.

Successful dry-runs:

```text
20260702T115300Z-c764c596 → succeeded, dry_run=true
20260702T115350Z-9c8af08f → succeeded, dry_run=true
20260702T115400Z-7128f258 → succeeded, dry_run=true
```

Failed full run:

```text
20260702T115433Z-204e9587 → failed, dry_run=false
```

Failure reason:

```text
Exported 7 approved annotations across 4 classes
Images: 4 across 1 classes
Train: 3 vectors | Val: 1 vectors

ERROR: Not enough approved training data for episodic classifier training.
Split 'val' has no class with at least k_shot=3 examples.
Class counts in this split: comitl=1
```

Interpretation:

- 7 annotations were approved;
- 3 classes had only 1 image each;
- config filters classes with fewer than `min_images_per_class=2`;
- only `comitl` remained with 4 images;
- validation split contained only 1 example;
- config requires `k_shot=3`;
- validation could not form an episode.

The dry-run did not catch this because the current dry-run mostly prints intended commands instead of executing the real dataset feasibility checks.

## 7. Required fixes before claiming the workflow is complete

### P0 blockers

1. Add a real training preflight.
2. Import/preserve the initial dataset in the cumulative corpus.
3. Train on the cumulative approved corpus, not only new annotations.
4. Add a Models admin tab.
5. Add model listing/inspection APIs.
6. Add manual candidate testing without mutating runtime.
7. Add active-vs-candidate comparison.
8. Add promotion/rollback from admin UI with local-only protection.
9. Add runtime state detection: promoted version vs loaded version.
10. Ensure no tests mutate the real runtime model by default.

### P1 improvements

1. Version features/metadata per model version.
2. Store metrics in the model manifest.
3. Add calibration output.
4. Add log download endpoint.
5. Add candidate rejection/cleanup.
6. Add promotion history and actor/audit trail.
7. Add optional live-local e2e suite behind explicit flags.

## 8. Promotion behavior

Training should never auto-promote a new model.

Correct behavior:

```text
full retrain succeeds
→ candidate is created
→ admin tests candidate
→ admin compares candidate vs active
→ admin explicitly promotes if acceptable
→ backend restart or hot reload required
→ smoke check confirms loaded model
```

Promotion must be guarded by:

- manifest health;
- checksums;
- no training job running;
- no promotion already in progress;
- local-only request;
- explicit confirmation by version id;
- compare-and-swap expected current promoted version;
- refusal if `MODEL_DIR` override makes promotion ambiguous.

## 9. Test strategy

### Default CI tests

Must be safe and non-mutating:

- backend tests with temporary registry/runtime directories;
- frontend Playwright mocked endpoints;
- no writes to real `backend/codex_model`;
- no real model promotion against the workspace runtime.

### Optional live-local tests

Behind explicit env flags, for example:

```bash
RUN_LIVE_MODEL_E2E=1
```

Recommended live-local flow:

1. seed/import a tiny but feasible dataset;
2. preflight passes;
3. dry-run succeeds;
4. full retrain succeeds;
5. candidate appears in registry;
6. manual candidate test works;
7. compare active vs candidate works;
8. promotion dry-run works;
9. promotion to temporary runtime works;
10. rollback works;
11. runtime smoke confirms loaded version.

## 10. Advisor artifacts

External review artifacts generated on 2026-07-07:

- fable plan: `/home/sina/.advisor/sessions/20260707T163816-plan/normalized/fable.md`
- kimi review: `/home/sina/.advisor/sessions/20260707T163816-review/normalized/kimi.md`
- glm critique: `/home/sina/.advisor/sessions/20260707T163816-critique/normalized/glm.md`
- context file: `.omx/reviews/admin-model-workflow-context-20260707.md`

## 11. High-level implementation order

Recommended order when implementation starts:

1. Training preflight and dataset feasibility reasons.
2. Initial dataset import and cumulative corpus design.
3. Dataset snapshot manifest tied to each training run.
4. Registry read APIs.
5. New admin Models tab read-only.
6. Manual candidate test endpoint/UI.
7. Active vs candidate comparison.
8. Promotion/rollback API and UI.
9. Calibration and metrics persistence.
10. Live-local e2e smoke suite.

## 12. Final summary

The central decision is:

```text
Do not improve the model by training only on new annotations.
Improve it by retraining a candidate on the full cumulative approved corpus.
```

The model to retrain regularly is:

```text
classifier projection head + prototypes
```

The base DINOv2 feature extractor should remain frozen initially.

The admin workflow must make candidate testing, comparison, promotion, rollback, and runtime smoke validation explicit and safe.

## 2026-07-07 — External corpus audit gate

Before importing any historical data from `/mnt/f/CODEX`, run the read-only audit script instead of copying files directly:

```bash
python3 scripts/audit_external_corpus.py \
  --json-output .omx/reviews/data-discovery/external-corpus-audit.json \
  --markdown-output .omx/reviews/data-discovery/external-corpus-audit.md
```

The audit is intentionally non-mutating: it inventories candidate element-crop folders, maps source class names against `backend/codex_model/config.json`, detects duplicate image hashes, flags non-BMP/unknown/unmatched labels, and evaluates whether the resulting class counts can satisfy the current episodic settings (`k_shot`, `q_queries`, `n_way`).

Default audited sources:

- `/mnt/f/CODEX/AI_clinic_class/Main_Elements`
- `/mnt/f/CODEX/AI_clinic_class/MainElem_Original`
- `/mnt/f/CODEX/Clinic-Codex/data/Elements`

Current audit result from 2026-07-07:

- `Clinic-Codex/data/Elements` maps all 286 active classes, but also has 17 unmatched source folders.
- `Main_Elements` and `MainElem_Original` each map 26/31 folders to active classes.
- Aggregate active-class coverage reaches 286/286, but preflight still fails because 65 classes have too few images for the current few-shot episode settings.
- 49 duplicate hash groups were detected.
- Therefore historical data must still go through mapping review, deduplication, low-count-class policy, and image-format normalization before any import or model promotion.

Hard rule: a model candidate built from external historical data is not promotable until this audit is clean or every reported blocker has an explicit reviewed exception.
