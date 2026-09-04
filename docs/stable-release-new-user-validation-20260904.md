# Validation de livraison — nouvel utilisateur, 4 septembre 2026

## Périmètre et verdict

Le parcours local CPU Linux/WSL et Windows natif est opérationnel : installer le dépôt, obtenir le modèle
de base, créer le premier compte administrateur, analyser une image existante,
annoter, valider, lancer un essai à blanc et créer un candidat sans activation.
Cette livraison ne promet pas une amélioration de précision.

Code préparé dans le worktree `clinic-codex-retrainable-release`, branche
`codex/retrainable-release` (base Git `8751e89` + modifications de livraison).
Aucun commit, push ou remplacement du modèle actif n'a été effectué.

## Essai sur clone isolé

Un vrai `git clone --no-local --branch codex/retrainable-release` a été créé
dans `/tmp/clinic-stable-new-user-dRAIr4`, puis le diff et les nouveaux fichiers
de livraison y ont été appliqués. Il s'agit donc du contenu à publier,
pas d'une validation du dépôt distant encore inchangé.

L'installation initiale partait sans environnement Python, sans configuration
locale, sans compte ni annotations, avec des caches de modèles isolés.
`bash scripts/install.sh` a installé les dépendances CPU, exporté les 286 classes
du modèle fourni et téléchargé les poids publics MobileSAM/DINOv2 vérifiés.
`bash scripts/run-dev.sh` a obtenu HTTP 200 sur `/ready` et le frontend.

Une seconde exécution complète de l'installeur a réussi en conservant exactement
les hashes de `backend/.env`, de la configuration, des poids actifs et de l'index
des validations. Le serveur temporaire a ensuite été laissé arrêté.

Le test Chromium `frontend/tests/e2e/local-retraining-live.spec.ts` a exécuté :

1. Création du premier administrateur depuis le formulaire.
2. Chargement de l'image existante `0015-cacahuatl/03_04_22-27.bmp`.
3. Analyse réelle MobileSAM/DINOv2, puis annotation et envoi pour revue.
4. Validation de l'annotation depuis Admin → Trier.
5. Essai à blanc CPU, puis réentraînement CPU depuis Admin → Entraîner.
6. Affichage du candidat, rechargement et vérification que son résultat persiste.
7. Vérification que le panneau est accessible à l'écran et que les hashes du
   modèle actif n'ont pas changé.

Dernier passage : **1 test réussi en 32,9 secondes**.
Les données des essais intermédiaires ont été conservées uniquement dans le
clone temporaire. Aucun corpus privé n'a servi à ce parcours.

## Vérifications finales

- Backend et scripts Linux/WSL : **688 réussites, 73 skips**, commande
  `python -m pytest backend/tests scripts`.
- Les skips Linux sont explicitement motivés : 65 relectures d'archives privées absentes,
  2 tests CUDA, 2 E2E opt-in et 4 tests Windows natifs. Les tests de recherche autonomes restent actifs.
  Aucun module entier ni contrôle applicatif n'a été désactivé.
- E2E CPU opt-in : **11 réussites** avec
  `CLINIC_LOCAL_RETRAIN_E2E=1 python -m pytest backend/tests/test_local_retraining.py`.
  Un essai à blanc et deux vrais candidats successifs ont été exécutés ;
  mêmes prototypes à validations identiques, base et index de revue inchangés.
- Frontend : **288 tests**, lint et build réussis ; audit npm : **0 vulnérabilité**.
- Playwright standard : **12 réussites**, le test réel opt-in étant exécuté
  séparément. Couvre analyse, annotations, sauvegarde, zoom/pan et onglets admin.
- Revue indépendante : pas de blocage restant dans le périmètre auth,
  création/intégrité du candidat, concurrence et téléchargement.
- `git diff --check` réussi. Les verrous, poids générés et données locales
  restent exclus du push.

## Validation Windows native

Un second vrai clone Git a été créé dans `D:\CodexTests\Clinic Windows 20260904`,
avec application du diff de livraison et des fichiers nouveaux. Le chemin contient
volontairement des espaces. Aucun environnement WSL n'exécute les tests Windows.
Caches de modèles et données utilisateur sont isolés de l'installation habituelle.

Environnement : Windows 10 x64 (build 19045), Python 3.11.4, Node 24.19.0,
Windows PowerShell 5.1.19041 et PowerShell 7.6.5, PyTorch 2.5.1 CPU.

- Installation complète PowerShell 5.1 : réussie après correction des erreurs
  détectées sur la première installation. MobileSAM et DINOv2 téléchargés et vérifiés.
- Démarrage `run-dev.ps1 -Smoke` : réussi sous PowerShell 5.1 et 7 ;
  HTTP 200 pour le backend prêt et le frontend, puis arrêt des processus.
- Réinstallation et diagnostic `check-windows-runtime.ps1` sous PowerShell 5.1 et 7 :
  réussis ; contrôle sous PowerShell 7 des hashes de `backend/.env`, configuration, projection, prototypes
  actifs et index des validations strictement conservés.
- Chromium réel : **1 réussite en 40,7 secondes**. Premier compte, upload du même
  BMP existant, vraie segmentation, annotation, validation, essai à blanc et vrai
  candidat CPU depuis le frontend. Résultat visible après rechargement ; runtime inchangé.
- E2E CPU et tests du mode local : **11 réussites en 28,61 secondes** ;
  deux candidats réels identiques à validations constantes.
- Suite backend/scripts native : **680 réussites, 81 skips en 116,05 secondes**.
  Les tests PowerShell exercent les deux versions, les chemins avec espaces,
  les lancements concurrents CLI/app et la reprise du verrou après arrêt forcé.
  Les scripts Bash, deux audits historiques POSIX/ext4, les archives privées
  absentes et les tests opt-in ne sont pas présentés comme validés sur Windows.
- Frontend natif : **288 réussites**, lint/build réussis et audit npm à
  **0 vulnérabilité**. Playwright standard : **12 réussites, 1 E2E opt-in ignoré**
  (ce dernier a été exécuté séparément avec les vrais modèles).
- Panne simulée du backend : le lanceur se termine en erreur et arrête aussi le
  frontend ; aucun port de test ne reste ouvert.

Les corrections Windows portent sur les arguments Python/PowerShell, la mise à
jour de pip, UTF-8, le traitement des avertissements, le verrou OS partagé,
l'environnement du processus candidat (`USERNAME`), le lancement PowerShell du
mode snapshot, la réparation atomique du cache DINOv2 et les fins de ligne Git.
Les assertions de chemins des tests sont portables. Le mode snapshot avancé est
couvert par ses arguments/dry-runs, pas par un nouvel entraînement sur corpus privé.

La CI couvre désormais backend et frontend sur Windows et Linux ; ces jobs
modifiés n'ont pas encore été exécutés sur GitHub, puisque rien n'a été poussé.
Une revue indépendante finale n'a identifié aucun nouveau blocage dans ce périmètre.
Les corrections sont ciblées, sans nouvelle dépendance applicative.

Journaux locaux de cette validation : `D:\CodexTests\windows-*.log`.
Capture conservée : `D:\CodexTests\windows-retraining-candidate.png`.
Ces preuves et les données générées ne font pas partie du dépôt à publier.

## Résultats du modèle : ne pas confondre fonctionnement et gain

Le réentraînement standard adapte les prototypes à partir du modèle fourni et
des annotations actuellement validées. DINOv2 et la projection restent figés.
Les classes sans nouvelles annotations sont conservées ; les doublons exacts
de crops sont comptés une seule fois et les labels contradictoires sont refusés.

Sur le crop créé par le test navigateur : base **0/1**, candidat **0/1**.
Sur l'image complète `zo` du test CPU : base **1/1**, candidat **1/1**.
Ces mesures portent sur les images apprises : elles ne constituent ni un
benchmark représentatif ni une preuve de généralisation. Les performances
du mode avancé décrites dans l'ancien rapport ne sont pas celles de ce mode.

Le candidat reste local et non activé. Le rapport enregistre
`generalization_validated=false` et la promotion reste bloquée sans évaluation
indépendante. Aucune amélioration automatique n'est annoncée à l'utilisateur.

## Corrections livrées et limites

Corrections ciblées : installation MobileSAM vérifiée et bornée, readiness réelle,
exception d'auto-validation réservée au premier administrateur local,
réentraînement sans corpus privé, reprise après arrêt brutal du processus,
cycle d'import Python, boucle de requêtes frontend, conservation des notes,
navigation admin et panneau de résultats autrefois masqué par CSS.
Les tests et guides d'installation ont été synchronisés.

La suite utilise le Pillow figé dans `backend/uv.lock` pour les anciennes preuves
hachées ; les tests de l'installation utilisateur ont également exercé Pillow 12.3.
Windows natif CPU et Linux/WSL sont validés dans les environnements ci-dessus.
macOS, Windows ARM et le réentraînement GPU Windows n'ont pas été exécutés.
Les lanceurs natifs refusent les chemins UNC/WSL ; un clone sur disque local
est requis. Le verrou a été exercé sur les systèmes de fichiers locaux, pas
sur des partages réseau.

Guide opérateur : [réentraînement local](admin-model-retraining-workflow.md).
