# Clinic Codex Frontend

React/Vite interface for uploading codex glyph images, inspecting segmentation results, correcting annotations, validating training labels, and sending curated annotations to the backend.

## Routes

| Route | Purpose |
| --- | --- |
| `/` | Workspace/dashboard: upload an image, run `/segment`, inspect prior analyses, view boxes over the image, pan/zoom, and open annotation. |
| `/annotate/:id` | Annotation editor for one browser-local analysis record stored in IndexedDB-backed storage. |

Analysis records are stored locally in the browser through an IndexedDB-backed storage boundary with legacy localStorage migration. The backend is only required for new segmentation, class lists, similarity/trust calls, and saving validated annotations.

## User workflow

1. Open `/` and upload an image.
2. The frontend calls `POST /segment`; the returned `image_size` is `[width, height]` and every bbox is `[x, y, width, height]` in image pixels.
3. Inspect the result on the workspace. Zoom and pan move the image and overlay together.
4. Open `/annotate/:id` to correct the result:
   - **Select/move/resize** existing boxes.
   - **Draw** a missing box.
   - **Type a label** in the fuzzy element combobox. Suggestions appear from the first letters; prefix matches rank ahead of contains matches.
   - **Create a label** by entering a name that is not in `/classes`.
   - **Rename** an element by editing the combobox value; the bbox is preserved.
   - **Validate** only elements ready for training. Edited, moved, resized, or newly drawn elements become draft until validated again.
5. Click **Enregistrer les modifications** to persist local changes.
6. Click **Envoyer pour entraînement** to send only validated, named elements to the backend.

Draft elements are never sent for training by the frontend. A label that is empty or `unknown` is treated as unnamed.

## Backend API used by the frontend

Set `VITE_API_BASE_URL` if the backend is not at `http://localhost:7117`.

| Frontend action | Endpoint |
| --- | --- |
| Upload/analyze | `POST /segment` multipart field `image` |
| Load label list | `GET /classes` |
| Similarity panel | `POST /similar` JSON `{ image_base64, bbox, limit }` |
| Trust panel | `POST /trust` JSON `{ image_base64, bbox, predicted_class, top_k }` |
| Send validated annotations | `POST /save-annotation` JSON payload documented in `backend/README.md` |

## Commands

```bash
cd frontend
npm install
npm run dev      # Vite dev server, usually http://localhost:7118
npm run lint     # ESLint
npm run test     # Vitest unit/component tests
npm run build    # TypeScript + production build
```

Targeted geometry/annotation checks include:

```bash
npm run test -- imageCoords AnnotationPage.pan WorkspacePage.pan Image387Alignment AnnotationPage.naming
npx playwright test tests/e2e/image-387-alignment.spec.ts
```

## Important implementation notes

- The image and SVG overlay must share the same rendered rectangle. The editor keeps bboxes in image-pixel coordinates and maps pointer events through the overlay rect.
- `preserveAspectRatio="none"` is intentional in the shared image/overlay SVG convention so both layers stretch identically inside the measured wrapper.
- `annotationStatus` is a per-element map with values `draft` or `validated`. Missing legacy statuses are read as draft-compatible data.
- Custom labels created in the combobox are local/session annotation labels. They do not become model predictions until validated data is exported, retrained, and the backend is restarted.
