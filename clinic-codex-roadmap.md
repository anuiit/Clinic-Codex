# Roadmap d’amélioration — Clinic Codex

_Date de revue : 22 mai 2026_  
_Dépôt analysé : `anuiit/clinic-codex`_  
_Nature de la revue : analyse statique du repository via les fichiers backend/frontend disponibles. Les tests et builds n’ont pas été exécutés dans cet environnement ; la roadmap inclut donc une première étape de vérification CI locale._

---

## 1. Résumé exécutif

Clinic Codex est une application d’annotation/retraining avec un backend Flask et un frontend React/Vite/Tailwind. Le produit est déjà fonctionnellement structuré autour d’un workflow clair : upload, segmentation, correction humaine des boxes, validation, sauvegarde des annotations validées et retraining. La documentation est plutôt bonne pour l’usage, l’installation et les endpoints.

Le principal problème n’est pas l’absence de logique métier, mais sa concentration dans quelques fichiers trop gros et trop couplés. Côté frontend, `AnnotationPage.tsx` concentre environ 1 180 lignes et mélange chargement de données, édition de bounding boxes, zoom/pan, historique undo, canvas preview, filtres, tri, sauvegarde locale, soumission backend et rendu. `WorkspacePage.tsx` est aussi très chargé, avec upload, historique local, overlay, zoom/pan, crops canvas, focus/hover et trust API. Côté backend, `backend/examples/flask_api.py` est en pratique le serveur applicatif complet : configuration, CORS, initialisation ML, endpoints actifs, endpoints legacy, validation de payloads, décodage image, logique de trust/similarité et persistance y sont mélangés.

La roadmap doit donc viser trois objectifs :

1. **Rendre le code maintenable** en séparant orchestration, UI, logique métier, validation, persistence et appels réseau.
2. **Améliorer la performance perçue et réelle** en batchant la classification, en évitant les redraws inutiles, en déplaçant les opérations lourdes hors du thread principal UI et en sécurisant le cycle de vie des requêtes.
3. **Réduire le bloat actif** en archivant ou sortant le legacy, en rendant la CI stricte, en introduisant des contrats API typés et en fixant des limites de taille par fichier/module.

Priorité recommandée : commencer par une phase de stabilisation courte, puis découper backend et frontend par vertical slices testables plutôt que par refactor massif en une seule PR.

---

## 2. Fichiers et zones inspectés

### Documentation et architecture

- `README.md`
- `frontend/README.md`
- `backend/README.md`
- `INSTALL.md`
- `_legacy/README.md`

### Frontend

- `frontend/package.json`
- `frontend/tsconfig.app.json`
- `frontend/eslint.config.js`
- `frontend/src/App.tsx`
- `frontend/src/pages/WorkspacePage.tsx`
- `frontend/src/pages/AnnotationPage.tsx`
- `frontend/src/pages/AnnotationElementList.tsx`
- `frontend/src/pages/annotation/AnnotationSelectedInspector.tsx`
- `frontend/src/pages/annotation/ElementNameCombobox.tsx`
- `frontend/src/components/MainImagePanel.tsx`
- `frontend/src/services/api.ts`
- `frontend/src/services/storage.ts`
- `frontend/src/types/index.ts`
- `frontend/src/utils/imageCoords.ts`
- `frontend/src/utils/imageStageZoom.ts`
- `frontend/src/utils/segmentationBoxes.ts`
- `frontend/src/utils/fuzzyClasses.ts`
- `frontend/src/index.css`

### Backend

- `backend/examples/flask_api.py`
- `backend/services/annotation_storage.py`
- `backend/codex_model/classifier.py`
- `backend/codex_pipeline/segmentation/mobilesam.py`
- `backend/requirements.txt`
- `backend/tests/test_annotation_storage.py`
- `backend/tests/test_save_endpoint.py`

### CI / scripts

- `.github/workflows/smoke.yml`
- `scripts/run-dev.sh`

---

## 3. Diagnostic détaillé

## 3.1 Points forts

Le projet a plusieurs bases saines :

- La documentation décrit clairement l’architecture : backend Flask sur `7117`, frontend React/Vite/Tailwind sur `7118`, stockage d’annotations sous `backend/annotations/<analysis_id>/` et scripts de retraining.
- Le frontend dispose déjà de composants séparés pour certaines zones : `MainImagePanel`, `WorkspaceHistoryPanel`, `WorkspaceDetectedPanel`, `AnnotationOverlay`, `AnnotationElementList`, `AnnotationSelectedInspector`, etc.
- Les utilitaires de géométrie sont déjà en partie extraits : `imageCoords.ts`, `imageStageZoom.ts`, `segmentationBoxes.ts`.
- Le backend a un service dédié pour le stockage : `backend/services/annotation_storage.py`.
- La sauvegarde d’annotations remplace atomiquement le dossier cible via un dossier temporaire, ce qui est une bonne base.
- Des tests existent pour le stockage d’annotation et l’endpoint `/save-annotation`.
- Le workflow utilisateur “draft vs validated” est documenté et pris en compte côté frontend/backend.

## 3.2 Dette principale

### Frontend : pages trop grosses et trop couplées

`AnnotationPage.tsx` doit être la priorité frontend. Le fichier contient :

- chargement de record depuis `localStorage` ;
- chargement des classes depuis le backend ;
- state complet des annotations ;
- logique de draw/move/resize des boxes ;
- undo history ;
- preview canvas ;
- zoom/pan ;
- filtres et tri de liste ;
- raccourcis clavier globaux ;
- sauvegarde locale ;
- soumission remote ;
- gestion de toast ;
- rendu principal.

Résultat : chaque changement UI risque d’impacter la logique métier. La page devient difficile à tester autrement que par tests d’intégration.

`WorkspacePage.tsx` présente le même problème à une échelle plus faible : historique, upload, preview, segmentation, zoom/pan, focus, overlays, crops canvas, trust calls et rendu sont dans le même module.

### Backend : serveur applicatif dans `examples/flask_api.py`

Le fichier `backend/examples/flask_api.py` est nommé comme un exemple, mais il sert de point d’entrée réel dans les scripts et la doc. Il mélange :

- config d’environnement ;
- création de l’app Flask ;
- CORS manuel ;
- initialisation globale du classifier ;
- lazy init du segmenter ;
- endpoints actifs ;
- endpoints legacy/demo ;
- validation des inputs ;
- décodage base64 ;
- découpe image/bbox ;
- logique trust/similarity ;
- gestion d’erreurs ;
- lancement serveur.

Cela rend l’app difficile à tester sans stubs, difficile à configurer proprement et difficile à faire évoluer vers une version prod/local robuste.

### Contrats API trop implicites

Le frontend définit des types TypeScript, le backend valide partiellement les payloads à la main, mais il n’y a pas de contrat partagé ni de validation structurée complète. Exemple : `/similar`, `/trust` et `/similar-samples` dupliquent la validation `image_base64 + bbox`. `/save-annotation` vérifie quelques champs avant d’appeler `save_annotation`, mais certaines erreurs de forme peuvent encore remonter sous forme d’exceptions internes.

Il y a aussi un mismatch de code d’erreur : le backend renvoie `INTERNAL`, alors que le frontend traite `INTERNAL_ERROR` dans `saveAnnotation` / `AnnotationPage`. L’utilisateur risque donc de recevoir un message générique au lieu du message prévu.

### Stockage frontend fragile

`frontend/src/services/storage.ts` stocke l’historique complet dans `localStorage`, y compris `imageDataUrl`. C’est simple, mais fragile :

- `localStorage` est limité ;
- les images base64 gonflent vite ;
- l’historique n’a pas de quota/cap ;
- pas de schema version explicite ;
- erreurs de parsing/écriture avalées silencieusement ;
- pas de séparation entre métadonnées et blobs image.

Pour une app d’annotation d’images, `IndexedDB` est plus adapté.

### CI pas assez stricte

Le workflow `.github/workflows/smoke.yml` lance un build frontend, mais le lint est en `continue-on-error: true`. Le backend n’exécute pas les tests ; il fait seulement un import check. Or le backend contient déjà des tests rapides sans modèle réel via stubs. Ces tests doivent tourner en CI.

Il y a aussi un risque d’alignement tooling : le workflow utilise Node 18 tandis que `frontend/package.json` référence des versions très récentes de Vite/React/TypeScript. La version Node supportée doit être explicitée via `.nvmrc`, `engines`, ou CI alignée.

### Bloat / legacy

`_legacy/` est conservé dans le repo actif. C’est acceptable temporairement, mais le dossier contient du code “référence seulement”. Il augmente le bruit dans les recherches, la taille du repo et le risque de confusion. Le README indique déjà que certains scripts SAM doivent être archivés séparément. Il faut finaliser cette séparation.

---

## 4. Architecture cible recommandée

## 4.1 Backend cible

Objectif : transformer `backend/examples/flask_api.py` en vraie application Flask modulaire.

Structure proposée :

```text
backend/
  app/
    __init__.py
    factory.py
    config.py
    extensions.py
    routes/
      health.py
      classes.py
      segment.py
      classify.py
      similarity.py
      annotations.py
    schemas/
      common.py
      annotation.py
      image.py
      similarity.py
    services/
      classifier_service.py
      segmentation_service.py
      crop_service.py
      trust_service.py
      annotation_storage.py
      sample_index.py
    errors.py
    response.py
  codex_model/
  codex_pipeline/
  tests/
  wsgi.py
```

Principes :

- `create_app(config)` retourne l’app Flask.
- Les routes appellent des services, elles ne contiennent pas de logique ML ou stockage lourde.
- Les services ML sont initialisés paresseusement mais protégés par lock si nécessaire.
- Les validations de payloads sont centralisées dans `schemas/`.
- Les erreurs retournent toutes un format cohérent :

```json
{
  "status": "error",
  "error_code": "INVALID_BBOX",
  "message": "bbox must be [x, y, w, h]",
  "hint": null,
  "trace_id": null
}
```

- Les endpoints demo/legacy sont désactivables par config : `ENABLE_LEGACY_ENDPOINTS=false`.

## 4.2 Frontend cible

Objectif : transformer les grosses pages en orchestrateurs courts.

Structure proposée :

```text
frontend/src/
  app/
    router.tsx
    providers.tsx
  shared/
    api/
      httpClient.ts
      errors.ts
    components/
      MainImagePanel.tsx
      Toast.tsx
    hooks/
      useAsyncTask.ts
      useDebouncedValue.ts
    utils/
      imageCoords.ts
      imageStageZoom.ts
      segmentationBoxes.ts
      fuzzyClasses.ts
  features/
    analyses/
      analysis.types.ts
      analysisRepository.ts
      useAnalysisHistory.ts
      useCurrentAnalysis.ts
    workspace/
      WorkspacePage.tsx
      components/
      hooks/
        useUploadAnalysis.ts
        useWorkspaceViewport.ts
        useWorkspaceSelection.ts
        useWorkspaceCrops.ts
        useTrustPanel.ts
    annotation/
      AnnotationPage.tsx
      components/
      hooks/
        useAnnotationRecord.ts
        useAnnotationEditor.ts
        useAnnotationViewport.ts
        useAnnotationKeyboardShortcuts.ts
        useAnnotationPreviewCanvas.ts
        useAnnotationSubmission.ts
      state/
        annotationReducer.ts
        annotationActions.ts
      utils/
        annotationFilters.ts
        annotationValidation.ts
  i18n/
```

Principes :

- Les pages doivent rester sous **150–220 lignes**.
- Un hook métier ne devrait pas dépasser **200 lignes**.
- Les utilitaires purs doivent être testés hors React.
- Les composants UI ne doivent pas connaître `localStorage`, `fetch`, `axios`, ni la logique de validation métier.
- L’état complexe d’annotation doit passer par `useReducer` plutôt qu’une accumulation de `useState`/`useRef`.

---

## 5. Roadmap priorisée

## Phase 0 — Stabilisation et garde-fous

Durée cible : 1 à 2 jours.

### Objectifs

Éviter que le refactor casse le workflow existant. Installer des garde-fous automatiques avant de découper.

### Actions

1. **Créer une branche dédiée**
   - `refactor/maintainability-roadmap`.
   - Pas de mélange avec nouvelles features.

2. **Lancer localement les commandes de référence**
   - Frontend : `npm ci`, `npm run build`, `npm run lint`, `npm run test`.
   - Backend : `python -m pytest backend/tests scripts/test_export_annotations.py`.
   - Documenter les erreurs actuelles avant correction.

3. **Rendre la CI stricte progressivement**
   - Supprimer `continue-on-error: true` sur le lint seulement après correction des erreurs existantes.
   - Ajouter `npm run test` au job frontend.
   - Ajouter `pytest backend/tests scripts/test_export_annotations.py` au job backend.
   - Garder un job ML-light avec stubs pour ne pas télécharger tous les modèles en CI.

4. **Fixer le mismatch d’erreur backend/frontend**
   - Backend : renvoyer `INTERNAL_ERROR` ou frontend : traiter `INTERNAL`.
   - Recommandation : standardiser `INTERNAL_ERROR` partout.

5. **Ajouter des fichiers de configuration d’environnement**
   - `.env.example` frontend avec `VITE_API_BASE_URL=http://localhost:7117`.
   - `.env.example` backend avec `PORT`, `HOST`, `CORS_ORIGINS`, `MODEL_DIR`, `ENABLE_LEGACY_ENDPOINTS`.
   - `.nvmrc` ou `package.json.engines` pour Node.

6. **Ajouter une règle de taille de fichier non bloquante au début**
   - Script `scripts/check-file-size.py` ou équivalent.
   - Warning si fichier frontend > 500 lignes.
   - Warning si route backend > 250 lignes.
   - La règle devient bloquante après refactor des pages principales.

### Definition of done

- CI exécute build + tests frontend + tests backend rapides.
- L’erreur interne `/save-annotation` est correctement affichée côté frontend.
- Les versions Node/Python attendues sont explicites.
- La roadmap peut être découpée en issues sans casser le produit.

---

## Phase 1 — Refactor backend en app Flask modulaire

Durée cible : 3 à 6 jours selon niveau de tests attendu.

### Objectifs

Réduire le fichier serveur monolithique, rendre les endpoints testables et préparer les optimisations ML.

### Actions prioritaires

#### 1. Créer `create_app`

Nouveau fichier : `backend/app/factory.py`.

```python
from flask import Flask

from .config import Settings
from .routes import register_routes
from .errors import register_error_handlers


def create_app(settings: Settings | None = None) -> Flask:
    app = Flask(__name__)
    settings = settings or Settings.from_env()
    app.config.from_mapping(settings.to_flask_config())
    register_routes(app, settings)
    register_error_handlers(app)
    return app
```

Puis `backend/wsgi.py` :

```python
from app.factory import create_app

app = create_app()
```

`scripts/run-dev.sh` doit lancer `backend/wsgi.py` ou `flask --app backend.wsgi run`.

#### 2. Découper les routes

Créer des blueprints :

- `routes/health.py` : `GET /health` et éventuellement `GET /ready`.
- `routes/classes.py` : `GET /classes`.
- `routes/segment.py` : `POST /segment`.
- `routes/similarity.py` : `POST /similar`, `POST /trust`, éventuellement `/similar-samples` si conservé.
- `routes/annotations.py` : `POST /save-annotation`.
- `routes/classify.py` : `/classify`, `/classify-batch` seulement si encore utile.

#### 3. Centraliser la validation image/bbox

Créer `services/crop_service.py` :

- `decode_base64_image(image_base64: str) -> Image.Image`
- `decode_data_url(data_url: str) -> Image.Image`
- `validate_bbox(bbox, image_size) -> tuple[int, int, int, int]`
- `crop_bbox(image, bbox) -> Image.Image`

Cela supprime la duplication actuelle dans `/similar`, `/trust`, `/similar-samples`.

#### 4. Introduire des schemas

Options :

- léger : dataclasses + fonctions de validation ;
- plus robuste : Pydantic v2 ou Marshmallow.

Exemple de contrats :

```python
class BBoxPayload(BaseModel):
    image_base64: str
    bbox: tuple[int, int, int, int]

class TrustPayload(BBoxPayload):
    predicted_class: str | None = None
    top_k: int = Field(default=10, ge=1, le=50)
```

But : les routes ne doivent plus manipuler directement des `dict` non validés.

#### 5. Standardiser les réponses d’erreur

Créer `backend/app/errors.py` :

- `ApiError(error_code, message, status_code, hint=None)`.
- Mapper `ValueError`, `AnnotationPermissionError`, `AnnotationDiskFullError`, `AnnotationStorageError`.
- Ajouter `trace_id` pour les erreurs internes.

Tous les endpoints doivent retourner le même format d’erreur.

#### 6. Optimiser `/segment`

Actuellement, chaque crop est classifié individuellement. Le classifier expose déjà `classify_batch`. Pour les images avec plusieurs propositions, remplacer :

```python
for proposal in proposals:
    result = clf.classify(proposal.crop)
```

par :

```python
valid_proposals = [p for p in proposals if p.crop is not None]
results = classifier.classify_batch([p.crop for p in valid_proposals])
```

Cela réduit les allers-retours dans le modèle et améliore la latence de segmentation.

#### 7. Encapsuler les modèles ML

Créer `classifier_service.py` et `segmentation_service.py` :

- lazy loading contrôlé ;
- lock d’initialisation ;
- méthode `warmup()` optionnelle ;
- logs structurés ;
- possibilité d’injecter des stubs dans les tests.

Important : `torch.hub.load` et le téléchargement MobileSAM ne doivent pas surprendre l’utilisateur au moment d’une requête sans signal clair. Prévoir un `/ready` qui indique si les poids sont présents.

#### 8. Séparer endpoints actifs et legacy

Endpoints actifs pour le frontend :

- `POST /segment`
- `GET /classes`
- `POST /similar`
- `POST /trust`
- `POST /save-annotation`

Endpoints à marquer legacy/configurable :

- `/classify`
- `/classify-batch`
- `/sample-image`
- `/similar-samples`

Ajouter `ENABLE_LEGACY_ENDPOINTS=false` par défaut si ces endpoints ne sont plus utilisés.

### Definition of done

- `backend/examples/flask_api.py` n’est plus le point d’entrée applicatif principal.
- Les tests backend ne nécessitent pas d’import lourd du vrai modèle.
- `/segment` utilise `classify_batch`.
- `/similar`, `/trust`, `/save-annotation` ont des erreurs homogènes.
- `run-dev.sh`, `backend/README.md` et `INSTALL.md` pointent vers le nouveau point d’entrée.

---

## Phase 2 — Refactor frontend : réduire les grosses pages

Durée cible : 5 à 10 jours.

## 2.1 `AnnotationPage.tsx`

### Objectif

Transformer `AnnotationPage.tsx` en orchestrateur lisible : données + rendu + wiring. Toute la logique complexe doit partir dans hooks, reducer et utilitaires testables.

### Découpage recommandé

```text
features/annotation/
  AnnotationPage.tsx
  state/
    annotationReducer.ts
    annotationHistory.ts
    annotationSelectors.ts
  hooks/
    useAnnotationRecord.ts
    useClassCatalog.ts
    useAnnotationEditor.ts
    useAnnotationViewport.ts
    useAnnotationKeyboardShortcuts.ts
    useAnnotationPreviewCanvas.ts
    useAnnotationSubmission.ts
    useDisplayedAnnotationElements.ts
  components/
    AnnotationTopBar.tsx
    AnnotationStage.tsx
    AnnotationInspectorRail.tsx
    AnnotationElementList.tsx
    AnnotationSelectedInspector.tsx
    ElementNameCombobox.tsx
    AnnotationToast.tsx
  utils/
    annotationFilters.ts
    annotationPayload.ts
```

### Extraction 1 : `useAnnotationRecord`

Responsabilités :

- lire `id` depuis route ;
- charger `AnalysisRecord` depuis repository ;
- charger les classes ;
- exposer `{ record, classes, loading, error }`.

Ne doit pas gérer les interactions de bbox.

### Extraction 2 : `annotationReducer`

Remplacer plusieurs `useState` liés par un reducer :

- `elements`
- `annotationStatus`
- `focusedIdx`
- `hoveredIdx`
- `listHoveredIdx`
- `drawMode`
- `bboxHistory`
- `tempBbox`
- `customClasses`

Actions :

```ts
type AnnotationAction =
  | { type: 'loadRecord'; elements: DetectedElement[]; status: Record<number, AnnotationStatus> }
  | { type: 'focusElement'; idx: number | null }
  | { type: 'commitName'; idx: number; name: string }
  | { type: 'validateElement'; idx: number; validated: boolean }
  | { type: 'createBBox'; bbox: BBox }
  | { type: 'moveBBox'; idx: number; bbox: BBox }
  | { type: 'resizeBBox'; idx: number; bbox: BBox }
  | { type: 'removeElement'; idx: number }
  | { type: 'undoBBoxChange' }
  | { type: 'submitNamedElements' };
```

Le reducer doit être testé sans React.

### Extraction 3 : `useAnnotationViewport`

Responsabilités :

- `zoom`, `panOffset`, `stageSize`, `isPanning` ;
- `applyZoom` ;
- `clampPan` ;
- `handleStageWheel` ;
- resize observer.

Cette logique est réutilisable avec Workspace.

### Extraction 4 : `useBboxInteraction` ou `useAnnotationEditor`

Responsabilités :

- pointer down/move/up sur SVG ;
- draw/move/resize ;
- drag intent ;
- hit test ;
- temp bbox throttlé via `requestAnimationFrame`.

Le hook doit recevoir des callbacks du reducer, pas modifier directement tous les states de page.

### Extraction 5 : `useAnnotationSubmission`

Responsabilités :

- construire le payload `SaveAnnotationPayload` ;
- bloquer les éléments unnamed ;
- appeler `saveAnnotation` ;
- mapper les erreurs API ;
- exposer `{ sending, send, toast }`.

### Extraction 6 : `useDisplayedAnnotationElements`

Responsabilités :

- filtrage par query ;
- statut ;
- tri ;
- `submittedCount`.

Test unitaire facile avec tableaux de fixtures.

### Problème à corriger dans `ElementNameCombobox`

Le composant appelle `setInputState` pendant le rendu si `inputState.sourceValue !== value`. Cela doit être remplacé par un `useEffect` ou par un état dérivé contrôlé, car mettre à jour un state pendant le render est fragile et peut provoquer des comportements inattendus.

Refactor recommandé :

- `inputValue` contrôlé localement ;
- `useEffect(() => setInputValue(displayValue), [displayValue])` ;
- gérer commit/blur sans mutation dans render.

### Definition of done pour `AnnotationPage`

- `AnnotationPage.tsx` < 220 lignes.
- Toute manipulation bbox testée en reducer/utilitaires.
- Le canvas preview est dans un hook ou composant dédié.
- Aucun state update pendant render.
- Les actions save/send sont testées.
- Les raccourcis clavier sont dans un hook isolé.

---

## 2.2 `WorkspacePage.tsx`

### Objectif

Même logique : la page doit orchestrer, pas gérer tout le workflow.

### Découpage recommandé

```text
features/workspace/
  WorkspacePage.tsx
  hooks/
    useAnalysisHistory.ts
    useUploadAnalysis.ts
    useWorkspaceViewport.ts
    useWorkspaceSelection.ts
    useWorkspaceCrops.ts
    useTrustPanel.ts
  components/
    WorkspaceHeader.tsx
    WorkspaceHistoryPanel.tsx
    WorkspaceStage.tsx
    WorkspaceDetectedPanel.tsx
    WorkspaceUploadModal.tsx
    WorkspaceEmptyState.tsx
```

### Extraction 1 : `useAnalysisHistory`

Responsabilités :

- lire l’historique ;
- sélectionner record courant ;
- supprimer record ;
- synchroniser l’URL `?analysis=` ;
- filtrer records.

### Extraction 2 : `useUploadAnalysis`

Responsabilités :

- drag/drop ;
- file input ;
- preview data URL ;
- appel `segmentGlyph` ;
- création `AnalysisRecord` ;
- erreurs upload/segment.

### Extraction 3 : `useWorkspaceViewport`

Responsabilités :

- zoom/pan ;
- wheel ;
- reset view ;
- interaction avec overlay.

À terme, mutualiser avec `useAnnotationViewport` via un hook shared `useImageViewport`.

### Extraction 4 : `useTrustPanel`

Responsabilités :

- déclencher `getTrust` lorsque `focusedIdx` change ;
- annuler/ignorer les réponses obsolètes ;
- debounce court si navigation clavier rapide ;
- gérer loading/error.

Actuellement, un changement rapide de focus peut lancer plusieurs appels sans annulation explicite.

### Extraction 5 : `useWorkspaceCrops`

Responsabilités :

- dessiner les canvas de crops ;
- redessiner seulement ce qui est nécessaire ;
- utiliser `requestAnimationFrame` si beaucoup d’éléments.

### Definition of done pour `WorkspacePage`

- `WorkspacePage.tsx` < 220 lignes.
- Les hooks sont testables ou au moins couverts par tests React ciblés.
- Les appels trust sont annulés/ignorés si obsolètes.
- Les crops ne sont pas redessinés inutilement.

---

## Phase 3 — API client, stockage et contrats frontend/backend

Durée cible : 3 à 5 jours.

## 3.1 Unifier le client HTTP

`frontend/src/services/api.ts` mélange `axios` et `fetch`. Choisir un seul outil.

Recommandation : garder `fetch` natif si les besoins restent simples, ou garder `axios` si on veut interceptors/timeouts standardisés. Mais éviter le mix.

Structure proposée :

```text
shared/api/
  httpClient.ts
  apiErrors.ts
features/analyses/api/
  segmentApi.ts
  annotationApi.ts
  classesApi.ts
  trustApi.ts
```

À ajouter :

- timeout ;
- abort signal ;
- parsing d’erreurs standard ;
- typed result ;
- helper `extractBase64FromDataUrl` robuste.

## 3.2 Migrer `localStorage` vers IndexedDB

### Problème

Le stockage local contient l’image en data URL. C’est risqué pour la taille et la performance.

### Cible

Créer un repository :

```text
features/analyses/analysisRepository.ts
```

Avec deux implémentations possibles :

- `localStorageAnalysisRepository` temporaire ;
- `indexedDbAnalysisRepository` cible.

Stockage cible :

```ts
type AnalysisMetadata = {
  id: string;
  imageName: string;
  timestamp: number;
  result: SegmentResult;
  annotations: Record<number, string>;
  annotationStatus: Record<number, AnnotationStatus>;
  schemaVersion: 2;
};
```

Images stockées séparément en `Blob` dans IndexedDB.

### Migration

- Au premier lancement, détecter records `localStorage` schema v1.
- Migrer vers IndexedDB.
- Garder une option de fallback si migration échoue.
- Ajouter limite d’historique configurable : par exemple 50 analyses ou quota utilisateur.

## 3.3 Contrats partagés

Options :

1. Générer OpenAPI côté backend et client TS côté frontend.
2. Plus léger : créer un document `API_CONTRACT.md` + tests contractuels JSON.

Minimum recommandé :

- documenter tous les payloads ;
- tester que `/save-annotation`, `/trust`, `/similar`, `/segment` renvoient les shapes attendues ;
- aligner `error_code` backend/frontend.

---

## Phase 4 — Performance

Durée cible : 3 à 7 jours selon profondeur.

## 4.1 Backend ML

### Batch classification dans `/segment`

Impact attendu : fort si beaucoup de crops.

Plan :

- collecter les crops valides ;
- appeler `clf.classify_batch` ;
- recomposer les éléments.

### Préchargement / readiness

Ajouter :

- `GET /health` : app vivante ;
- `GET /ready` : modèles et checkpoints présents ;
- logs au chargement modèle ;
- option `WARMUP_ON_START=true` si souhaité.

### Éviter les téléchargements surprises

`classifier.py` utilise `torch.hub.load` et MobileSAM télécharge un checkpoint si absent. Pour une app installée localement, il faut rendre ces artefacts explicites :

- script `scripts/download-weights.sh` obligatoire ou recommandé ;
- message clair si artefact absent ;
- mode offline possible avec chemin de modèle configuré.

### Segmentation longue

Si la segmentation prend 30–60 secondes, considérer :

- endpoint async : `POST /analyses` puis `GET /analyses/:id/status` ;
- ou un worker local simple avec queue ;
- progress UI côté frontend.

Ce n’est pas obligatoire pour une v1, mais utile si les utilisateurs traitent plusieurs images.

## 4.2 Frontend UI

### Annuler/debouncer les appels trust

Quand `focusedIdx` change vite, l’appel précédent doit être ignoré ou annulé.

Recommandation : `AbortController` et `requestId`.

### Virtualiser les listes si beaucoup d’éléments

Si le nombre d’éléments peut dépasser 100–200, virtualiser `AnnotationElementList` et les listes détectées.

### Canvas previews

- Ne redessiner que la crop focus ou les crops visibles.
- Utiliser `requestAnimationFrame` pour grouper les redraws.
- Éviter de redessiner tous les canvas au resize si non visible.

### Données image

- Remplacer data URL en mémoire persistée par Blob IndexedDB.
- Créer des object URLs temporaires côté UI.
- Révoquer les object URLs au cleanup.

### Bundle

- Lazy load `AnnotationPage` avec `React.lazy` car la page contient beaucoup de logique et n’est pas toujours nécessaire au démarrage.
- Supprimer exports et fonctions inutilisées (`classifyElement`, anciens endpoints si non utilisés).

---

## Phase 5 — Tests et qualité

Durée cible : continue, première passe 2 à 4 jours.

## 5.1 Tests frontend à ajouter

### Reducer annotation

Cas minimum :

- charger elements ;
- renommer élément ;
- validation/draft ;
- create bbox ;
- move bbox ;
- resize bbox ;
- delete ;
- undo create/delete/update ;
- `submitNamedElements` n’envoie pas unnamed.

### Utils

Déjà bons candidats :

- `segmentationBoxes.ts`
- `imageCoords.ts`
- `imageStageZoom.ts`
- `fuzzyClasses.ts`
- futurs `annotationFilters.ts`
- futurs `annotationPayload.ts`

### API client

Mocker `fetch` ou `axios` :

- succès ;
- erreur 400 structurée ;
- 409 permission ;
- 507 disk full ;
- erreur réseau ;
- mismatch d’erreur interne.

### E2E ciblé

Garder peu mais critique :

1. Upload mocké → segmentation mockée → workspace affiche boxes.
2. Ouvrir annotation → déplacer box → sauvegarder localement.
3. Renommer + valider → envoyer pour entraînement → payload correct.
4. Unnamed validé impossible à envoyer.

## 5.2 Tests backend à ajouter

### Routes

- `/health`, `/ready`.
- `/classes` sans relire inutilement le fichier config à chaque appel si remplacé.
- `/similar` : bbox invalide, image invalide, top_k invalide.
- `/trust` : predicted_class manquant, top_k borné.
- `/segment` avec segmenter/classifier mockés.
- `/save-annotation` avec annotations malformées : manque `index`, manque `class_name`, bbox non numérique, bbox vide, index dupliqué.

### Storage

- tmp dir unique même si deux saves simultanés même `analysis_id`.
- `crop_path` relatif plutôt qu’absolu, ou justification explicite.
- metadata versionné.
- base64 invalide avec `validate=True`.

## 5.3 CI cible

```yaml
jobs:
  frontend:
    steps:
      - npm ci
      - npm run lint
      - npm run test
      - npm run build

  backend-unit:
    steps:
      - pip install -r backend/requirements-ci.txt
      - pytest backend/tests scripts/test_export_annotations.py

  backend-import-heavy:
    if: manual or nightly
    steps:
      - pip install -r backend/requirements.txt
      - python -c "from codex_model import CodexClassifier; print('import ok')"
```

Recommandation : créer `backend/requirements-ci.txt` pour les tests rapides sans dépendances ML lourdes, ou améliorer les stubs.

---

## Phase 6 — Réduction du bloat et documentation technique

Durée cible : 1 à 3 jours.

### Actions

1. **Archiver `_legacy/` hors du repo actif**
   - Tag ou branche : `archive/legacy-2026-05`.
   - Supprimer du repo principal si non nécessaire.
   - Garder un lien dans `docs/legacy.md`.

2. **Nettoyer endpoints legacy**
   - Décider si `/classify`, `/classify-batch`, `/sample-image`, `/similar-samples` sont utiles.
   - S’ils restent : documenter comme legacy et désactiver par défaut.
   - Sinon : supprimer avec tests mis à jour.

3. **Ajouter une doc architecture**

Créer `docs/ARCHITECTURE.md` :

- diagramme backend/frontend ;
- stockage local vs backend ;
- cycle d’annotation ;
- contrats API ;
- règles de modules ;
- comment ajouter un endpoint ;
- comment ajouter une feature frontend.

4. **Ajouter `CONTRIBUTING.md`**

Inclure :

- commandes obligatoires avant PR ;
- convention de taille de fichier ;
- style de tests ;
- organisation frontend/backend ;
- règles de non-régression.

---

## 6. Backlog technique priorisé

| Priorité | Sujet | Action | Impact | Effort |
| --- | --- | --- | --- | --- |
| P0 | CI | Ajouter tests frontend/backend et rendre lint strict | Évite régressions | M |
| P0 | Error contract | Corriger mismatch `INTERNAL` vs `INTERNAL_ERROR` | UX + debug | S |
| P0 | Backend entrypoint | Remplacer `examples/flask_api.py` par `app/factory.py` + `wsgi.py` | Maintenabilité | M |
| P0 | AnnotationPage | Extraire reducer + hooks principaux | Maintenabilité forte | L |
| P1 | WorkspacePage | Extraire upload/history/viewport/trust/crops | Maintenabilité | M/L |
| P1 | API client | Unifier fetch/axios + erreurs typées + abort | Robustesse | M |
| P1 | `/segment` | Utiliser `classify_batch` | Performance | M |
| P1 | Storage frontend | Migrer images vers IndexedDB | Robustesse/perf | L |
| P1 | Validation backend | Schemas payloads | Sécurité/robustesse | M |
| P2 | Legacy | Sortir `_legacy/` et endpoints demo inutiles | Moins de bruit | S/M |
| P2 | UI perf | Virtualisation listes + canvas redraw visible only | Perf | M |
| P2 | Docs | `ARCHITECTURE.md`, `CONTRIBUTING.md` | Onboarding | S |

---

## 7. Ordre de PR recommandé

### PR 1 — Garde-fous CI et contrat d’erreur

Contenu :

- corriger `INTERNAL`/`INTERNAL_ERROR` ;
- ajouter tests autour de l’erreur ;
- ajouter `npm run test` en CI ;
- ajouter pytest backend en CI ;
- retirer `continue-on-error` si lint déjà propre, sinon corriger lint d’abord.

Pourquoi en premier : la suite du refactor aura besoin d’un filet de sécurité.

### PR 2 — Backend app factory sans changement fonctionnel

Contenu :

- créer `backend/app/factory.py`, `config.py`, `wsgi.py` ;
- déplacer le CORS et config ;
- garder les mêmes endpoints ;
- mettre à jour `run-dev.sh`.

Pourquoi : découpage structurel sans modifier la logique métier.

### PR 3 — Backend services et schemas

Contenu :

- extraire validation image/bbox ;
- extraire `trust_service`, `crop_service`, `classifier_service` ;
- standardiser erreurs ;
- tests routes.

### PR 4 — Optimisation `/segment`

Contenu :

- batch classification ;
- test avec classifier mock ;
- logs durée segmentation/classification.

### PR 5 — Frontend API client

Contenu :

- client HTTP unique ;
- erreurs typées ;
- abort support pour trust ;
- tests.

### PR 6 — Annotation reducer

Contenu :

- créer `annotationReducer` ;
- déplacer create/move/resize/delete/undo/validate/rename ;
- tests unitaires.

### PR 7 — Annotation hooks

Contenu :

- `useAnnotationRecord` ;
- `useAnnotationViewport` ;
- `useAnnotationEditor` ;
- `useAnnotationSubmission` ;
- réduire `AnnotationPage.tsx` sous 220 lignes.

### PR 8 — Workspace hooks

Contenu :

- `useAnalysisHistory` ;
- `useUploadAnalysis` ;
- `useWorkspaceViewport` ;
- `useTrustPanel` ;
- réduire `WorkspacePage.tsx` sous 220 lignes.

### PR 9 — Storage IndexedDB

Contenu :

- repository abstraction ;
- migration localStorage ;
- stockage Blob ;
- quota/history cap ;
- tests.

### PR 10 — Legacy + docs

Contenu :

- archiver `_legacy/` ;
- documenter architecture ;
- ajouter contributing ;
- activer check taille fichiers.

---

## 8. Règles de maintenabilité proposées

### Taille de fichiers

- Page React : max 220 lignes.
- Composant UI : max 250 lignes.
- Hook : max 200 lignes.
- Utilitaire pur : max 150 lignes.
- Route Flask : max 150 lignes.
- Service backend : max 250 lignes.

Au-delà : découper ou justifier dans la PR.

### Séparation des responsabilités

Un fichier ne doit pas contenir simultanément :

- rendu UI ;
- appel API ;
- persistence ;
- logique métier ;
- parsing d’erreur ;
- manipulation canvas ;
- validation payload.

Exception : page orchestratrice très courte qui assemble des hooks/composants.

### Tests minimaux par changement

- Toute modification de bbox : test reducer ou utilitaire.
- Tout endpoint backend : test route avec payload valide/invalide.
- Tout changement d’erreur API : test frontend + backend.
- Toute persistence : test migration et échec quota/parsing.

---

## 9. Risques et points d’attention

### Risque 1 : refactor trop large

Ne pas tenter de réécrire tout `AnnotationPage` en une PR. Le bon chemin est reducer d’abord, hooks ensuite, UI en dernier.

### Risque 2 : modèles ML difficiles à charger en CI

Garder des tests unitaires avec stubs. Créer un job nightly/manual pour les imports lourds ou la vraie inférence si nécessaire.

### Risque 3 : migration storage

Avant migration IndexedDB, ajouter un export/import JSON de secours pour éviter de perdre des annotations locales.

### Risque 4 : performance canvas

Les optimisations canvas doivent être validées par tests visuels/E2E sur les images problématiques documentées dans le repo, notamment les tests d’alignement image/overlay.

### Risque 5 : endpoints legacy

Supprimer trop vite les endpoints demo peut casser des scripts non documentés. Les désactiver par config avant suppression est plus sûr.

---

## 10. Critères de succès mesurables

À la fin de la roadmap :

- `AnnotationPage.tsx` < 220 lignes.
- `WorkspacePage.tsx` < 220 lignes.
- `backend/examples/flask_api.py` n’est plus le serveur principal.
- CI bloque sur lint/test/build frontend.
- CI bloque sur tests backend rapides.
- `/segment` utilise batch classification.
- Les erreurs API ont un format unique.
- Les images ne sont plus persistées en data URL dans `localStorage`.
- `_legacy/` n’est plus dans le chemin actif du repo principal ou est clairement exclu.
- Le projet contient `docs/ARCHITECTURE.md` et `CONTRIBUTING.md`.
- Tout nouveau module critique a des tests unitaires ou route tests.

---

## 11. Synthèse courte des priorités

1. **Sécuriser la CI et les contrats d’erreur.**
2. **Transformer le backend en app Flask modulaire.**
3. **Découper `AnnotationPage.tsx` via reducer + hooks.**
4. **Découper `WorkspacePage.tsx` via hooks de workflow.**
5. **Unifier le client API et annuler les requêtes obsolètes.**
6. **Batcher la classification dans `/segment`.**
7. **Migrer l’historique image vers IndexedDB.**
8. **Sortir le legacy et documenter l’architecture cible.**

Cette roadmap garde le produit fonctionnel à chaque étape tout en attaquant les causes principales de dette : fichiers trop gros, responsabilités mélangées, contrats implicites, stockage local fragile et CI insuffisamment stricte.
