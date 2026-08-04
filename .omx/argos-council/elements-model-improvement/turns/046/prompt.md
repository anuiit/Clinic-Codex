# Brief de décision - retrait terminal v19-A

L'unique extraction B/14 autorisée au tour 45 a échoué. Le Council doit statuer sur la clôture terminale, pas proposer une récupération.

Questions obligatoires :

1. Ratifierez-vous l'audit d'échec SHA-256 `6cfff7ac...` comme preuve suffisante du retrait terminal de v19-A ?
2. Confirmez-vous qu'aucun retry, changement de batch, désactivation de pin_memory, modification du runner, phase 3 ou v20 n'est permis par la synthèse publiée au tour 45 ?
3. Confirmez-vous que cet échec est infrastructurel et ne constitue ni une victoire ni une défaite métrique de B/14 ?
4. Quelle conclusion finale doit être conservée pour la mission et quel modèle reste la meilleure référence validée ?

Faits clés : une seule commande `precompute` a été exécutée. Elle a quitté avec code 1 lors du pin-memory CUDA du premier batch du fold 1. Des images d'entraînement ont été décodées par les workers, mais aucun batch de features n'a été terminé ; le nombre exact de lectures est inconnu à cause du prefetch. L'extraction OOF n'a pas commencé. Le dossier cache est vide, aucun manifeste n'existe et le contrôle C1 n'a pas été lancé.

Les compteurs candidat restent à zéro pour optimiseur, prédiction et score. Le runtime est inchangé, le test final n'a pas été lu, aucun retry n'a eu lieu. La condition explicite no-v20 demeure inscrite dans l'état. C1 reste la meilleure référence répliquée sous `supported_reference`.
