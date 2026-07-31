# Reproducible pre-annotation baseline

This recipe rebuilds the classifier from the approved external Elements corpus
only. It does not read annotation crops and it never writes
`backend/codex_model`.

The runtime class order is an ABI: labels `0..285` must map to the exact order
in `backend/codex_model/config.json`. Do not alphabetize names or pass the raw
archive directly to training. Materialize an approved 286-class snapshot first.

Create a local DINO pin without copying weights into the repository:

```bash
backend/.venv/bin/python scripts/pin_dinov2.py \
  --repository-path ~/.cache/torch/hub/facebookresearch_dinov2_main \
  --weights-path ~/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth \
  --output /secure/local/dinov2-vits14-pin.json
```

Run the full recipe (GPU by default):

```bash
bash scripts/run_baseline_recipe.sh \
  --elements-dir backend/training_corpus/external/20260711-approved-external-corpus-v1/Elements \
  --approved-manifest backend/training_corpus/external/20260711-approved-external-corpus-v1/import_snapshot.json \
  --backbone-manifest /secure/local/dinov2-vits14-pin.json
```

Each immutable output directory records hashes for its inputs, features,
checkpoint, prototypes and candidate runtime artifacts. Sparse classes are kept
for prototype export; they are deliberately excluded from episodic head
training whenever they lack distinct support and query examples. Prototype
export uses the complete approved corpus, including training examples; this is
intentional for this baseline and is not a held-out accuracy measurement.

For a bounded CPU check of the head-training/export path, run:

```bash
bash scripts/smoke_baseline_recipe.sh
```

It runs the same cached-DINO snapshot twice and requires identical runtime
weights, prototypes and configuration. The full recipe additionally validates
metadata order and recomputes frozen-DINO features from the local pin.

## Freeze the historical Elements archive

Build the immutable legacy snapshot before training. It applies the historical
`min_images_per_class=2` rule, normalizes Unicode labels, and writes hashes
plus page-level provenance for every retained image.

```bash
uv run python scripts/freeze_legacy_elements.py Elements.zip \
  --output-dir /secure/local/elements-legacy-v1
```

This produces `/secure/local/elements-legacy-v1/Elements` and
`legacy_elements_manifest.json`. The raw archive and its extracted files are
never rewritten.

## Train and compare a replacement candidate

```bash
bash scripts/run_baseline_recipe.sh \
  --elements-dir /secure/local/elements-legacy-v1/Elements \
  --approved-manifest /secure/local/elements-legacy-v1/legacy_elements_manifest.json \
  --backbone-manifest /secure/local/dinov2-vits14-pin.json
```

`glyphs.zip` remains outside supervised element training. It needs a separate
glyph-level truth set before it can be a promotion metric.

Use a frozen train/test split manifest and compare the runtime against the
candidate through global 286-way inference:

```bash
uv run python scripts/benchmark_model_replacement.py \
  --features <features.pt> --split-manifest <split.json> \
  --runtime-dir backend/codex_model --candidate-dir <candidate/runtime> \
