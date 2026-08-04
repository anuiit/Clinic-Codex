# Position indépendante Codex - tour 46

Je ratifie le retrait terminal de v19-A. L'unique exécution autorisée de phase 2 a échoué avant le premier batch de features terminé, pendant le transfert pin-memory du premier fold, avec une OOM CUDA. Le contrat publié au tour 45 interdit explicitement toute récupération et toute deuxième extraction. Relancer avec un batch plus petit, désactiver pin_memory ou modifier le runner figé violerait donc l'expérience plutôt que de la sauver.

Ce résultat n'est pas une mesure de qualité du backbone B/14. Des images d'entraînement ont été décodées par le prefetch DataLoader, mais aucun cache, sidecar, manifeste, checkpoint candidat, prédiction ni score n'a été produit. Le nombre exact de lectures source est devenu inconnaissable après l'exception, ce que l'audit déclare au lieu de reconstruire un faux compteur. L'extraction OOF et le contrôle C1 n'ont pas commencé.

L'échec révèle une limite de la faisabilité synthétique : la marge GPU mesurée sur des forwards synthétiques batch 32 n'incluait pas le pic du pipeline réel combinant backbone, workers et mémoire épinglée. Cette faiblesse méthodologique est utile pour de futurs protocoles, mais le terminal explicite no-v20 interdit d'en faire une nouvelle branche dans cette mission.

Les garanties négatives restent fortes : zéro étape d'optimiseur candidat, zéro prédiction, zéro score, zéro écriture runtime, test final non lu, trois hashes runtime inchangés, répertoire cache vide après la sortie, aucun processus GPU restant et aucun retry. Le runner, le spec et l'évaluateur figés n'ont pas été modifiés après autorisation.

Je recommande au Council de ratifier le statut `retired_terminal_phase_2_infrastructure_failure`, de confirmer que la phase 3 et v20 restent interdites, et de clore la mission expérimentale. Le meilleur modèle validé demeure C1 sous `supported_reference`; v19 n'apporte aucune preuve comparative et ne doit jamais être promu. La conclusion honnête n'est pas qu'un meilleur modèle a été trouvé, mais que la dernière hypothèse autorisée n'a pas pu être évaluée sans violer le protocole.
