# Réentraînement avec les images existantes — validation du 4 septembre 2026

> Rapport historique du mode avancé avec snapshot local. Pour le parcours standard
> sans corpus privé, voir [le guide actuel](admin-model-retraining-workflow.md)
> et [la validation sur clone neuf](stable-release-new-user-validation-20260904.md).
> Les chiffres ci-dessous ne mesurent pas le mode automatique `local_prior`.

## Verdict

Le parcours de création d'un candidat fonctionne réellement depuis l'API HTTP jusqu'au GPU, à l'export et à l'inférence sur images. Il utilise les annotations actuellement validées, sans ajout d'images et sans écraser le modèle actif. Le gain mesuré reste faible : ce travail fiabilise la fonctionnalité, il ne démontre pas une amélioration générale du modèle.

Implémentation dans le worktree `/home/sina/omxpro/clinic-codex-retrainable-release`, branche `codex/retrainable-release`. Aucun push ni promotion effectué. Les changements préexistants du dépôt principal sont préservés.

## Données et apprentissage

- 7 annotations actuellement approuvées, issues d'une seule image, couvrent calli, comitl, coztic et tlapalli. Les validations historiques périmées, les annotations en attente et rejetées ne sont pas automatiquement approuvées.
- Snapshot enfant `snapshot-bfa3dbf807d8f622b89f` : 9 266 lignes, dont 9 135 train, 56 dev et 75 locked_test. Les 7 annotations approuvées sont toutes dans train. 731 doublons sont consolidés et 22 annotations en conflit sont exclues.
- Dans le snapshot parent, les 7 annotations étaient réservées au test : elles n'auraient pas entraîné le modèle. Le nouveau snapshot retire leur composante source/doublons du holdout, conserve les autres affectations et ne modifie pas le parent.
- DINOv2 et projection existants gelés ; seuls les 4 prototypes des classes annotées sont recalculés à partir de tout leur train cumulé. Les 282 autres prototypes restent identiques bit à bit. Cela ne garantit pas des prédictions inchangées : les classes restent en concurrence.
- Conservation des 286 noms et identifiants numériques, y compris les identifiants non contigus. Une incompatibilité bloque l'export ou la promotion.

## Défauts corrigés

1. Annotations approuvées comptées comme disponibles mais exclues de l'apprentissage par leur affectation au holdout.
2. Recalcul de toutes les classes alors que seules quatre reçoivent des annotations nouvelles.
3. Risque de renumérotation des classes lors de l'export.
4. Comparaison de prototypes recalculés plutôt que des vrais poids exportés.
5. Décalage du prétraitement : ancien cache en bilinéaire, inférence en Lanczos. Le cache utilise désormais exactement la fonction runtime, avec un marqueur versionné et refus des anciens caches.
6. Comptage UI et garde API fondés sur les lignes réellement utilisées en train. Un snapshot périmé ou contenant des annotations réservées au test est refusé.

## Mesures finales, prétraitement aligné

Les scores ci-dessous évaluent les prototypes réellement exportés, sans les réentraîner dans l'évaluateur.

| Jeu existant | Modèle actif | Candidat |
| --- | ---: | ---: |
| Dev, 56 crops | 51/56 — 91,07 % | 51/56 — 91,07 % |
| Test réservé, 75 crops | 66/75 — 88,00 % | 66/75 — 88,00 % |
| Test métier, 303 crops | 146/303 — 48,18 % | 147/303 — 48,51 % |
| Annotations utilisées pour apprendre, 7 crops | 4/7 | 4/7 |

Le jeu métier comporte 87 classes et 10 pages sources, sans recouvrement RGB exact détecté avec le train. Cette vérification ne prouve pas l'absence de quasi-doublons ni une indépendance complète des lignées sources. Une prédiction devient correcte (calli), aucune correcte ne devient incorrecte ; top-3 inchangé à 193/303 (63,70 %). Delta top-1 : +0,33 point. L'intervalle bootstrap à 95 % par page contient zéro : pas de gain statistiquement établi.

Les 7 annotations de train servent seulement à mesurer l'ajustement sur les exemples appris, pas la généralisation. Trois restent mal classées. Le faible nombre de sources et les holdouts incomplets empêchent toute garantie sur les 286 classes. Les scores antérieurs calculés avec le cache bilinéaire ne sont pas des résultats runtime valides.

## Vérifications

- 109 tests backend ciblés : construction et intégrité des snapshots, API admin, cache/runtime, mise à jour sélective, export, benchmark, scripts et garde de promotion. Inclut le refus de publier le candidat si une évaluation échoue.
- 19 tests frontend ciblés et build TypeScript/Vite réussis.
- Test opt-in `backend/tests/test_retraining_live_e2e.py` : vrai serveur HTTP local, dry-run puis job GPU, chargement du candidat et classification des 7 images réelles. Même DINO local épinglé par hashes, aucune fausse feature. Prédictions et scores du cache vérifiés contre ceux de l'inférence réelle.
- Projection, prototypes non concernés, noms/labels, poids/config actifs et index de validation vérifiés inchangés. SHA-256 du manifeste parent inchangé : `e2c06404c6cd523e96de7fffa274c4384f56f6c9e128751a6f60465dbd72fa40`.
- Revue indépendante en lecture seule. Le mode explicite de benchmark `refit` reste un diagnostic et ne doit pas servir à décider une promotion.
- Pas de parcours cliqué dans un vrai navigateur : la couverture E2E réelle concerne HTTP → calcul → export → inférence ; l'interface est couverte par tests React et build.

Version finale testée : `20260904T160638Z-8751e89d-c0eec395`.

- [Rapport du parcours HTTP/GPU et inférence](../backend/model_registry/versions/20260904T160638Z-8751e89d-c0eec395/evaluation/http-e2e.json)
- [Comparaison dev](../backend/model_registry/versions/20260904T160638Z-8751e89d-c0eec395/evaluation/dev.json) et [test réservé](../backend/model_registry/versions/20260904T160638Z-8751e89d-c0eec395/evaluation/locked_test.json)
- [Comparaison métier](../backend/model_registry/versions/20260904T160638Z-8751e89d-c0eec395/evaluation/business-final.json)

Dev et locked_test sont calculés avant l'inscription au registre, puis inventoriés et hachés avec les poids. Les rapports HTTP E2E et métier sont des preuves complémentaires ajoutées après coup, non scellées par le manifeste du candidat. L'évaluation métier réutilise l'outil déjà présent à `/home/sina/omxpro/clinic-codex/scripts/evaluate_business_test_set.py`, avec le nouveau cache `backend/training_corpus/evaluation/business-final-runtime-lanczos-20260904/precomputed/features.pt`. Les images et poids générés restent locaux et hors Git.

## Utilisation opérateur

Depuis le checkout contenant les corrections, préparer un nouveau snapshot cumulatif. Les chemins suivants pointent vers les données locales déjà présentes ; adapter uniquement le checkout de sortie lors du déploiement.

```bash
export PYTHON=/home/sina/omxpro/clinic-codex/backend/.venv/bin/python
corpus=/home/sina/omxpro/clinic-codex/backend/training_corpus
"$PYTHON" scripts/build_training_snapshot.py \
  --runtime-config /home/sina/omxpro/clinic-codex/backend/codex_model/config.json \
  --legacy-manifest "$corpus/frozen/legacy-elements-v1/legacy_elements_manifest.json" \
  --external-snapshot "$corpus/external/20260711-approved-external-corpus-v1/import_snapshot.json" \
  --annotations-dir /home/sina/omxpro/clinic-codex/backend/annotations \
  --parent-manifest "$corpus/snapshots/snapshot-58de624e411fc613e64a/snapshot_manifest.json" \
  --output-root backend/training_corpus/snapshots \
  --train-live-annotations --exclude-conflicts --allow-underfilled-holdouts --json
```

Le snapshot testé existe déjà : ne pas tenter de le remplacer. Après de nouvelles validations, générer un nouvel enfant et renseigner son chemin. `--exclude-conflicts` exclut, il ne résout pas les conflits. `--allow-underfilled-holdouts` permet d'apprendre avec les seules images existantes, mais conserve le statut recherche et le blocage de promotion.

Configurer le backend avec :

```bash
export ENABLE_ADMIN_TRAINING_JOBS=true
export ADMIN_TRAINING_SNAPSHOT_DIR=/home/sina/omxpro/clinic-codex-retrainable-release/backend/training_corpus/snapshots/snapshot-bfa3dbf807d8f622b89f
export ADMIN_TRAINING_BACKBONE_MANIFEST=/home/sina/omxpro/clinic-codex/backend/training_corpus/backbone-pins/dinov2-vits14-local.json
```

Le backend déployé doit disposer de sa configuration et de ses poids actifs. Lancer ensuite le dry-run puis le réentraînement dans l'onglet Training. Aucun epoch n'est nécessaire en mode sélectif : la mise à jour des prototypes est un calcul supervisé direct.

Commande équivalente, pour reproduire le test dans le worktree isolé :

```bash
DEVICE=cuda bash scripts/retrain.sh \
  --elements-dir "$ADMIN_TRAINING_SNAPSHOT_DIR/Elements" \
  --approved-manifest "$ADMIN_TRAINING_SNAPSHOT_DIR/snapshot_manifest.json" \
  --metadata-csv "$ADMIN_TRAINING_SNAPSHOT_DIR/metadata.csv" \
  --backbone-manifest "$ADMIN_TRAINING_BACKBONE_MANIFEST" \
  --config backend/codex_pipeline/config/snapshot-warmstart.yaml \
  --init-projection /home/sina/omxpro/clinic-codex/backend/codex_model/weights/projection.pt \
  --update-annotated-prototypes
```

Ajouter `--dry-run` pour inspecter les commandes. Les sorties restent dans une nouvelle version de `backend/model_registry/versions`, avec rapports `evaluation/dev.json` et `evaluation/locked_test.json`. Un succès de job n'est pas une autorisation de promotion et aucun critère de non-régression n'est automatiquement assoupli.

Pour reproduire le parcours HTTP/GPU complet :

```bash
CLINIC_RETRAIN_E2E_SOURCE_BACKEND=/home/sina/omxpro/clinic-codex/backend \
  "$PYTHON" -m pytest -q -s -p no:cacheprovider backend/tests/test_retraining_live_e2e.py
```

Le test est ignoré sans les variables explicites ; il ne lance pas de GPU dans une suite ordinaire.

## Décision de livraison

La fonctionnalité de fabrication et de comparaison d'un candidat est validée sur les données disponibles. Le contrat de promotion P4 demeure non validé et bloque l'activation de ce candidat. Le modèle actif reste le choix de production : la faible amélioration et la couverture actuelle ne justifient pas de contourner ces gardes. Pas de nouvelles images imposées ; revoir les annotations historiques déjà présentes est une possibilité ultérieure, pas une validation automatique ni un prérequis ajouté à cette livraison.
