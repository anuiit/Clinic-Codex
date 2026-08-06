# Diagnostic des caches B/14 pour le refit R2

## Résultat

Les cinq caches de folds contiennent 39 960 occurrences d’entraînement : 9 990 lignes uniques, chacune présente dans exactement quatre caches et absente de son fold de validation déclaré. La gate de split passe sans exception.

Sur les 9 990 lignes, 9 986 ont quatre copies base et vues byte-identiques. Quatre divergent :

| Ligne | Classe | Groupes byte-identiques | Réextraction ciblée |
| --- | --- | --- | --- |
| `legacy:99` | `maitl_2` | 3–1 | groupe de 3 |
| `legacy:997` | `cimatl` | 2–2 | groupe folds 3/5 |
| `legacy:998` | `cimatl` | 2–2 | groupe folds 3/5 |
| `legacy:999` | `cimatl` | 2–1–1 | copie fold 5 |

Les trois dernières lignes sont consécutives, de même classe et de même composant. Les cosinus entre variantes restent supérieurs à `0.999996`. Les maxima absolus sont `0.013671875` pour les bases et `0.021484375` pour les vues ; toutes les sources sont en float16.

Une réextraction GPU ciblée des quatre lignes avec le pin officiel DINOv2-B/14, batch de quatre, huit vues, seed `20260803` et déterminisme strict a été répétée deux fois. Les deux répétitions sont byte-identiques pour les bases et les vues, et retrouvent exactement les variantes listées dans la dernière colonne.

## Décision

Le refit final n’utilisera pas un assemblage hybride des caches de folds. Il effectuera une nouvelle extraction full-data des 9 990 lignes dans un ordre gelé. Cette option coûte davantage de GPU mais supprime toute politique arbitraire de sélection par ligne et donne au candidat final un cache unique, reproductible et auditable.
