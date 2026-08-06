# Plan de livraison du classifieur R2

Profil Argos : `medium`. Session conseil : `adv_20260805T130141_c71019d4`.

## Objectif et règle de décision

Livrer un candidat R2 full-data pour les 286 classes, sans modifier le modèle principal. Le paquet reste shadow-only jusqu’à ce qu’une évaluation end-to-end indépendante, liée à la spec gelée, valide simultanément qualité, calibration, latence et robustesse. Le canary vient ensuite ; la promotion atomique est la dernière action et n’est pas automatique.

## Phase 0 — contrats et preuves préalables

Livrables : spec R2 et seuils de promotion versionnés, diagnostic des caches de folds, plan de rollback. Le cache hybride a été rejeté après diagnostic : quatre lignes dépendent du contexte de batch malgré des cosinus supérieurs à `0.999996`.

Acceptation : spec hashée avant l’extraction full-data ; runtime principal inchangé ; aucun alias promu modifié.

## Phase 1 — double extraction full-data

Créer deux caches indépendants des 9 990 lignes avec : tri lexicographique `row_id`, DINOv2-B/14 épinglé, image 224, batch de quatre, huit vues, seed `20260803`, déterminisme strict et stockage float16.

Gates : 9 990 lignes, 286 classes, 300 composants, taxonomie identique au runtime, metadata et tenseurs exacts entre les deux extractions. Comparer chaque ligne aux caches R2 de folds : cosinus minimum `0.9999` pour bases et vues, médiane au moins `0.999999`. Pinner corpus, pin B/14, ordre, environnement CUDA/GPU et hashes sémantiques.

Rollback : supprimer uniquement les nouveaux caches versionnés ; aucun fichier runtime n’est touché.

## Phase 2 — double refit et paquet candidat

Entraîner deux répliques sur le cache validé avec la recette R2 exacte : projection `768→768→128`, VICReg 30 epochs/batch 256/LR `3e-4`/WD `1e-4`, puis supervision épisodique 20-way/3-shot/5-query, 30×100 épisodes, LR `1e-3`, WD `1e-4`, température `0.1`, warmup 5.

Gates : états, prototypes, labels et diagnostics exactement égaux par hashes sémantiques. L’identité des fichiers sérialisés est observée, non utilisée comme substitut à l’égalité tensorielle. Calculer exactement 286 prototypes.

Exporter un paquet autonome sous `backend/model_registry/versions/<version>/` avec config, poids, provenance, manifest, checksums et model-card. Statut `candidate`, `promotion_eligible=false`, seuil de rejet `0.35` marqué hérité/non validé. Les hashes de `backend/codex_model` restent inchangés.

## Phase 3 — chargement autonome et registre

Faire accepter au chargeur deux contrats explicites :

- legacy : répertoire contenant directement les poids, config globale inchangée ;
- paquet : racine `runtime/`, config à la racine et poids sous `weights/`.

Valider avant le backbone : schéma, checksums, backbone allowlist, dimensions de projection/prototypes, labels, taxonomie et ordre des 286 classes, image size et seuil de rejet. Ajouter la résolution sûre `version|alias → runtime/` au registre.

Gates : snapshots de prédictions legacy inchangés, paquet B/14 factice chargé, paquets incohérents refusés.

## Phase 4 — gate end-to-end avant canary

Ajouter un harness annoté qui compare principal et candidat sur le même jeu indépendant. Le rapport contient les hashes du paquet et de la spec, la disjonction pixel et cas/spécimen quand disponible, les métriques et un verdict calculé. La promotion refuse tout rapport absent, modifié ou non concluant.

## Phase 5 — feature flag, shadow et canary bloqué

Modes : `off|shadow|canary`, défaut `off`. Le candidat n’est ni importé ni chargé en `off`.

Le shadow s’exécute dans un processus séparé, CPU par défaut, avec file et batch bornés ; toute panne, timeout ou OOM conserve la réponse principale et dégrade explicitement readiness/télémétrie. Seules des statistiques agrégées sont gardées.

Le canary est stable par hash du contenu complet de la page/batch, utilise une seule variante par page, retombe sur le principal sur erreur, et refuse de démarrer sans rapport offline `pass` lié à la spec. Le pourcentage par défaut est zéro.

## Phase 6 — activation et promotion, hors exécution automatique

Ordre : shadow sain → rapport indépendant `pass` → canary limité et sain → répétition du rollback → promotion atomique par l’outil existant. En l’absence de données indépendantes ou de trafic, la livraison s’arrête légitimement au candidat désactivé.

## Vérification et arrêt

Chaque phase reçoit ses tests ciblés puis une revue Argos. Tout échec de gate arrête la phase sans assouplissement. Le conseil est consulté après le paquet candidat et avant toute activation/promotion.
