## Codex

Je ratifie le retrait terminal de v19-A. L'unique exécution autorisée de phase 2 a échoué avant le premier batch de features terminé, pendant le transfert pin-memory du premier fold, avec une OOM CUDA. Le contrat publié au tour 45 interdit explicitement toute récupération et toute deuxième extraction. Relancer avec un batch plus petit, désactiver pin_memory ou modifier le runner figé violerait donc l'expérience plutôt que de la sauver.

Ce résultat n'est pas une mesure de qualité du backbone B/14. Des images d'entraînement ont été décodées par le prefetch DataLoader, mais aucun cache, sidecar, manifeste, checkpoint candidat, prédiction ni score n'a été produit. Le nombre exact de lectures source est devenu inconnaissable après l'exception, ce que l'audit déclare au lieu de reconstruire un faux compteur. L'extraction OOF et le contrôle C1 n'ont pas commencé.

L'échec révèle une limite de la faisabilité synthétique : la marge GPU mesurée sur des forwards synthétiques batch 32 n'incluait pas le pic du pipeline réel combinant backbone, workers et mémoire épinglée. Cette faiblesse méthodologique est utile pour de futurs protocoles, mais le terminal explicite no-v20 interdit d'en faire une nouvelle branche dans cette mission.

Les garanties négatives restent fortes : zéro étape d'optimiseur candidat, zéro prédiction, zéro score, zéro écriture runtime, test final non lu, trois hashes runtime inchangés, répertoire cache vide après la sortie, aucun processus GPU restant et aucun retry. Le runner, le spec et l'évaluateur figés n'ont pas été modifiés après autorisation.

Je recommande de ratifier le statut `retired_infrastructure_not_supported`, de confirmer que la phase 3 et v20 restent interdits, et de clore la mission expérimentale. Le meilleur modèle validé demeure C1 sous `supported_reference`; v19 n'apporte aucune preuve comparative et ne doit jamais être promu.

## Fable

Aucune voix disponible sur ce tour. Fable a échoué avant inférence avec `terminal_reason: api_error` et une réponse vide (`rc=1`). Cet échec fournisseur explicite n'est ni une approbation ni une objection. Aucun retry automatique n'a été lancé.

## Kimi

Kimi ratifie l'audit `6cfff7ac...` comme preuve suffisante du retrait terminal. Il relève sa cohérence interne : une exécution autorisée et observée, sortie 1, cache vide, aucun manifeste ou replay C1, zéro optimiseur, prédiction et score, runtime inchangé, test final non lu et aucun processus GPU résiduel.

Kimi confirme sans réserve l'interdiction de retry, de baisse de batch, de `pin_memory=False`, de modification du runner, de phase 3 et de v20. Un correctif après observation constituerait exactement l'adaptation opportuniste que le protocole prospectif devait empêcher.

Kimi classe l'événement comme infrastructurel pur : B/14 sous VICReg est non mesuré, ni battu ni vainqueur. Il recommande le claim `retired_infrastructure_not_supported` et interdit toute conclusion métrique dans les deux sens.

Kimi maintient C1 comme meilleure référence validée : VICReg fold-local sur DINOv2-S/14 gelé, gain poolé top-1 de 0,01123 contre B0, signal répliqué et statut `supported_reference`. Il conserve v11 dans le registre comme candidat full-data non promouvable. Le runtime reste scellé et le test final doit rester non lu. La seule prochaine décision licite concerne le financement d'un instrument indépendant, pas une nouvelle itération sur les 653 lignes OOF.

## Synthèse du Conseil

### Entrées et disponibilité des voix

Les dix fichiers annoncés ont été inclus sans saut ni troncature, pour un total de 106 064 caractères : runner intégral, spec, gel, autorisation phase 2, synthèse publiée du tour 45, état, journal, audit d'échec, brief et preuves. Kimi a répondu après lecture du contexte complet. Fable a échoué avant inférence sur une erreur fournisseur explicite. La synthèse publie donc la convergence Codex-Kimi et conserve explicitement l'absence de Fable, sans la transformer en vote.

### Ratification du retrait terminal

L'audit SHA-256 `6cfff7acdce3af4b8ef080bdf21dcf54788379c479a9f0f4322283d77f27a47b` est ratifié. La seule exécution autorisée de `precompute` a quitté avec code 1 pendant le pin-memory CUDA du premier batch train du fold 1, avant tout batch de features terminé.

Le statut canonique est `retired_infrastructure_not_supported`. Ce statut ne porte aucun jugement métrique sur DINOv2-B/14 sous VICReg : le facteur reste non mesuré. Il est interdit d'affirmer que B/14 aurait gagné, perdu ou été neutre dans v19-A.

La frontière observée est acceptée telle quelle : des images train ont été décodées par le prefetch, leur nombre exact n'est pas reconstructible, l'extraction OOF n'a pas commencé, aucun cache ou sidecar n'existe, aucun manifeste n'a été écrit et le contrôle C1 n'a pas été lancé. Aucun faux compteur exact ne doit être inventé.

### Interdictions définitives

Aucune récupération n'existe : pas de deuxième extraction, reprise partielle, réduction de batch, changement de workers, désactivation de pin-memory, modification du runner, amendement du spec ou nouveau volume. La phase 3 est interdite et aucun v20 ne peut être ouvert.

Les pas d'optimiseur, prédictions et scores candidats restent à zéro. Le runtime reste scellé aux hashes `0ad6ce34...`, `ffc52f03...` et `f2150a97...`. Le test final demeure non lu. Aucune promotion n'est autorisée.

### Conclusion finale de la mission

La mission expérimentale sur l'instrument local de 653 lignes OOF est close. C1 reste la meilleure référence validée sous `supported_reference` : VICReg fold-local sur DINOv2-S/14 gelé, gain poolé top-1 de 0,01123 contre B0, McNemar favorable et réplication vérifiée. C1 est une référence de recherche, pas une autorisation de remplacement automatique du runtime.

v11 reste inscrit comme candidat full-data gelé mais non promouvable, car la lignée du runtime est inconnue et l'instrument indépendant manque. Toutes les branches destinées à dépasser C1 sont closes ; v18.3 et v19-A sont retirées pour causes infrastructurelles sans résultat scientifique.

L'état global recommandé est `candidate_built_awaiting_independent_instrument`. La prochaine action utile n'est pas une modification du modèle sur ce même instrument, mais l'obtention d'une évaluation indépendante : segmentation experte commanditée ou partenariat institutionnel. Tant que cet instrument n'existe pas, le runtime reste inchangé et le test final non lu.

### Leçon méthodologique

Les faisabilités v18.3 et v19-A ont mesuré des noyaux synthétiques sans reproduire les pics du pipeline réel : flush de stockage pour v18.3, DataLoader plus pin-memory pour v19-A. Un futur protocole, sur un nouvel instrument et une nouvelle mission explicitement autorisée, devra tester le chemin I/O complet avant de geler l'expérience. Cette leçon ne crée aucune autorisation de reprendre la mission actuelle.
