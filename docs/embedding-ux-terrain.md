# Embedding UX Terrain Analysis

Analysis of short-term and medium-term features for the embedding-similarity
UX surface. Based on EMBEDDING-READINESS.md, the current codebase state, and
the dev lab prototype.

## Current State (committed on codex/embedding-ux-lab)

- **F1-v0**: Archetype thumbnails in workspace trust panel, gated behind
  `clinic.showArchetypes` localStorage flag. Backend serves indexed sample
  images through `/samples/<class>/<filename>` with path-traversal protection.
- **F2**: Free-text note field on annotation elements, persisted through
  save-annotation, surfaced in the inspector.
- **Dev Lab page** (`/dev`): Model info dashboard, sample coverage bar,
  feature flag toggles, embedding explorer (upload image -> nearest neighbors),
  text query placeholder.

## Short-Term Features (next 1-2 sprints)

### F1: Archetype Gallery (complete from v0)

**Status**: v0 shipped. Next steps:
- Remove the dev flag gate once validated with real sample data.
- Add hover tooltips showing class label and similarity score.
- Support click-to-navigate from a thumbnail to the full archetype detail.

### F3: Embedding Visualization (2D/3D projection)

**Goal**: Let users see the embedding space. Project the 128-dim vectors
to 2D (UMAP/t-SNE) or 3D and render an interactive scatter plot.

**Backend**: New endpoint `POST /embedding/project` that accepts a list of
class names or image crops, runs the projection head, and returns 2D/3D
coordinates. Pre-compute the full class prototype projection on startup
and cache it.

**Frontend**: Canvas or WebGL scatter plot. Hover shows class name + exemplar
thumbnail. Click selects a class and shows its nearest neighbors.

**Risk**: UMAP is non-deterministic without a fixed seed. Use PCA for
deterministic 2D or fix the UMAP random state.

### F4: Text-to-Glyph Search

**Goal**: Type a Nahuatl word or semantic description and find matching
glyphs via cross-modal embedding alignment.

**Approach**: Use a multilingual sentence transformer (e.g. LaBSE or
paraphrase-multilingual-MiniLM) to encode text queries into the same
128-dim space. This requires a projection layer trained to align text
embeddings with glyph embeddings.

**Simpler v0**: Use class name string matching + embedding similarity
as a hybrid. The text input searches class names with fuzzy matching,
then ranks by embedding distance to the query image (if provided) or
by prototype centrality.

**Backend**: `POST /search/text` accepting a query string, returning
ranked class matches with similarity scores and exemplar thumbnails.

### F5: Image-to-Image Translation (glyph style transfer)

**Goal**: Upload a modern drawing or photo and see it rendered as a
Nahuatl glyph in the codex style.

**Approach**: This is a medium-term research feature. Requires a
style-transfer model (CycleGAN, diffusion-based img2img with ControlNet)
trained on the glyph dataset. Not feasible without a curated training set
of paired modern/glyph images.

**Recommendation**: Defer to medium-term. The embedding explorer (F3)
and text search (F4) provide more immediate value.

## Medium-Term Features (next 3-6 months)

### F6: Instance-Level Embedding Search

**Goal**: Search individual historical glyph instances (not just class
prototypes). EMBEDDING-READINESS.md notes this as deferred work.

**Requirements**:
- Pre-compute and index embeddings for every sample image in the dataset.
- Add a vector database (e.g. FAISS, pgvector) for efficient nearest-neighbor
  search at scale.
- New endpoint `POST /search/instances` returning individual images ranked
  by similarity.

### F7: Trust Signal Dashboard

**Goal**: A dedicated view showing trust/confidence metrics across the
entire dataset. Which classes have high ambiguity? Where does the model
struggle?

**Backend**: Aggregate trust signals per class from historical analyses.
`GET /trust/dashboard` returning per-class margin, entropy, rejection
rate statistics.

**Frontend**: Bar charts or heatmaps showing per-class trust metrics.
Click a class to see individual ambiguous examples.

### F8: Active Learning Loop

**Goal**: Use embedding uncertainty to suggest which glyphs the annotator
should label next. High-entropy or low-margin predictions are surfaced
for priority annotation.

**Integration**: Ties into the admin annotation queue. Adds an "uncertainty"
sort order and highlights ambiguous elements.

## Implementation Order (Recommended)

1. **F3 (Embedding Viz)** — highest user-facing impact, builds on existing
   embedding infrastructure.
2. **F4 (Text Search v0)** — class name matching + embedding hybrid is
   low-risk and immediately useful.
3. **F1 polish** — remove flag, add interactions.
4. **F7 (Trust Dashboard)** — valuable for model evaluation.
5. **F6 (Instance Search)** — requires vector DB, higher engineering cost.
6. **F8 (Active Learning)** — depends on F6 + F7.
7. **F5 (Style Transfer)** — research project, defer until dataset is mature.

## Technical Notes

- The embedding pipeline uses DINOv2 ViT-S/14 backbone (384-dim CLS token)
  projected to 128-dim via a 2-layer MLP, then L2-normalized.
- 286 Nahuatl element classes. One prototype embedding per class (centroid).
- The `/similar` endpoint already returns `asset` URLs when sample images
  are indexed. The `/samples/<class>/<filename>` endpoint serves them with
  path-traversal protection.
- The dev lab page at `/dev` provides a sandbox for testing new embedding
  features without affecting the main annotation workflow.
