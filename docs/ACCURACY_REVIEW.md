# Revue de précision de DealVoice — mise à jour du 17 novembre 2026

Cette revue porte sur le parcours ingestion → indexation → recherche → calcul →
génération → validation → API/interface, ainsi que sur les scripts d'évaluation et
la configuration Docker. C'est une revue de code et de contrats logiciels, pas une
certification de précision sur des documents clients. Aucun score d'accuracy réel
n'est déduit des tests unitaires. Le modèle de la VM n'a pas été interrogé dans
cette revue et la VM n'a pas été modifiée.

Validation locale : **203 tests ciblés réussis** (188 pour génération, calculs,
recherche et API de recherche ; 15 pour l'API web). Ruff, le contrôle de format
Python et `node --check` réussissent sur les fichiers modifiés. Les appels LLM
sont simulés et aucun modèle n'est téléchargé. Ces résultats vérifient les
contrats du code, pas la capacité de Qwen3 à toujours choisir le bon plan.

Ce total correspond à la première série de corrections du 16 novembre. Pour la
mise à jour du 17 novembre, **176 cas distincts ont réussi** dans les séries
ciblées de génération, évaluation, recherche API et compatibilité web. Ruff,
le contrôle de format, `git diff --check` et la syntaxe JavaScript passent.
Les appels modèles sont simulés. Les contrôles ajoutés sont les suivants :

- Un refus ou une clarification est vérifié avec **tous les passages inclus
  dans le contexte du planificateur**, même si le plan n'en cite aucun. Si cette
  vérification dépasse le budget, elle échoue explicitement.
- Une limite de contexte sans réponse est exposée comme `unverified`, pas comme
  une réponse partielle utilisable.
- Une association entité/valeur non vérifiée ne peut plus faire réussir
  l'évaluation. Les valeurs acceptées par Python et les citations disponibles
  sont accessibles dans la réponse API, sans exposer le plan du modèle.
- `scripts/evaluate_client.py` teste le vrai parcours API séquentiellement et
  enregistre les résultats. Voir les [commandes et critères](EVALUATION.md#client-api-acceptance-recommended-before-deployment).

## Corrections limitées à des défauts identifiés

- **Doublons de réponse** : `GenerationPlan` élimine une partie répétée seulement
  si son libellé normalisé ET son contenu structuré sont identiques. Deux calculs
  ou deux périmètres différents restent distincts, même avec le même libellé.
- **Place disponible pour les preuves** : les instructions sont raccourcies sans
  augmenter la fenêtre du modèle. Le catalogue indique explicitement que Python
  possède les lignes des tableaux ; l'absence de cellules dans le prompt ne doit
  pas être interprétée comme une absence de données.
- **Classements et ambiguïtés** : consigne générique `sort` + direction + limite
  pour les N premiers/derniers. Plusieurs mesures plausibles sans critère explicite
  nécessitent une clarification. Aucune règle métier ni réponse sur Horizon n'est
  inscrite dans le code. Cette compréhension reste une tâche du LLM, à évaluer.
- **Reranking** : il s'applique aussi quand le nombre de candidats est inférieur
  ou égal à `top_k`. L'ordre est important car la génération peut omettre les
  derniers passages faute de place.
- **Portée de la réponse** : un calcul accepté sur un tableau dont la couverture
  source n'est pas vérifiée est présenté comme `partial` par l'API. Le calcul reste
  affiché et sourcé. La note de l'interface distingue bloc extrait et document
  exhaustif. Seuls les tableaux effectivement utilisés déterminent cette limite.

## Ce qui reste important pour un client

| Priorité | Couche et constat dans le code | Conséquence et minimum à vérifier |
| --- | --- | --- |
| Haute | `ingestion/document.py` exporte Docling en Markdown. `indexer.py::_split_pdf_tables` vérifie la structure du bloc, pas toutes les cellules du PDF ; `table_source_complete=False`. Les pages et coordonnées Docling ne sont pas conservées. | Une ligne absente, un tableau réparti sur plusieurs pages ou une mauvaise colonne peut changer un classement. Comparer les tableaux clients avec l'original ; conserver la provenance Docling avant de revendiquer une extraction exhaustive. |
| Haute | `ingestion/audio.py` utilise les mots de WhisperX et fusionne les segments consécutifs d'un même locuteur. La diarisation ne certifie pas son identité ; les boucles répétées sont nettoyées automatiquement. | Vérifier noms, chiffres et négations sur l'audio. Une citation horodatée désigne un passage, pas une preuve que chaque mot est exact. Un long tour de parole peut dépasser la taille cible de chunk. |
| Haute | `retrieval/search.py` fait une recherche dense, puis un reranking. La récupération supplémentaire des en-têtes utilise des mots exacts et examine au plus 200 points. | Le bon fichier ne garantit pas le bon tableau. Tester les synonymes, plusieurs tableaux semblables, les filtres de documents et les données contradictoires. L'exhaustivité au-delà des candidats bornés n'est pas garantie. |
| Haute | `retrieval/corrective.py::CoverageEvaluator` mesure surtout la structure et la correspondance d'en-têtes. Sans correspondance, il renvoie `complete` en laissant la validation sémantique au générateur. La requête corrective ajoute un suffixe générique. | `evidence_coverage=complete` n'est pas une certification de réponse exhaustive. Une recherche corrective peut retrouver les mêmes passages. Le badge final de calcul est désormais prudent, mais cette évaluation de couverture reste limitée. |
| Haute | `generation/generator.py` utilise un budget conservateur en octets, pas un tokenizer du modèle. Il réserve la réponse et une marge fixe ; le schéma natif n'est pas mesuré précisément. | Des preuves peuvent être écartées alors que la fenêtre réelle pourrait les contenir ; une réserve fixe ne garantit pas non plus le coût exact du protocole. Mesurer sur la VM avant de modifier les tailles de contexte. |
| Haute | La proposition et la validation utilisent le même LLM. Les refus/clarifications voient désormais tous les passages inclus ; pour un plan entièrement répondu, `_verify_plan` examine les preuves citées. Les résultats calculés sont montrés au vérificateur sur cinq lignes au maximum. | Le second appel peut répéter une erreur du premier. Il ne prouve ni l'absence de contradictions ailleurs ni la justesse du choix de colonne. Conserver des tests humains de refus, d'ambiguïté et de mauvais périmètre. |
| Haute | `evidence/tabular.py` exécute des opérations déterministes sur un seul bloc ; il n'offre pas de jointure, prévision ou optimisation de commande. | Un calcul peut être exact sur une mauvaise mesure choisie par le LLM. Une recommandation nécessite un critère, des hypothèses et les données métier nécessaires ; refuser une quantité optimale non justifiée. |
| Haute | `scripts/evaluate_generation.py` et `validate_pilot.py` appellent directement `search()` puis `generate()`, sans le gestionnaire correctif de l'API. Les contextes enregistrés figent une ancienne récupération. Les métriques de faits sont lexicales et fondées sur des variantes déclarées. | Les anciens rapports ne valident pas exactement le parcours déployé ni l'absence d'hallucinations. Refaire une recette via `/api/v1/search` et examiner les réponses, les cellules et les sources ; recréer les contextes après réindexation. |
| Haute si plusieurs utilisateurs | `api/main.py::get_current_user_role` lit `X-User-Role`. Caddy fournit une protection globale par mot de passe, pas une isolation par client. | L'en-tête seul ne constitue pas une authentification. Garder le périmètre de pilote à un espace documentaire partagé de confiance ; identité vérifiée côté serveur et isolation obligatoires pour des clients séparés. |
| Moyenne | `qdrant_schema.py` vérifie la dimension des vecteurs, pas l'identité/version du modèle d'embedding. | Un modèle de même dimension peut être incompatible avec l'index. Réindexer après changement d'embedding ; enregistrer ensuite son identité avec la collection pour bloquer les mélanges. |
| Moyenne | `indexer.py::list_indexed_documents` ne lit que la première page de 10 000 points. Les écritures Qdrant, les parents et les fichiers ne forment pas une transaction unique. `api/documents.py` conserve les jobs en mémoire. | À grande taille, des documents peuvent manquer de la liste ; une panne peut laisser un import incomplet. Tester sauvegarde/restauration des volumes ensemble et reprise après interruption avant un usage durable. |
| Moyenne | `web/app.js` envoie seulement la question courante et les documents sélectionnés. | L'affichage ressemble à une conversation, mais « prix détail » ne complète pas la demande précédente. Pour ce pilote, reformuler la question complète après une clarification ; une vraie mémoire de conversation serait une fonctionnalité distincte. |
| Moyenne | `/health` vérifie que l'API répond ; `/api/v1/status` vérifie la présence des objets. Le verrou d'opération appartient à un processus. | Un état `healthy` ne prouve pas qu'Ollama répond, qu'un upload réussit ou que la réponse est juste. Conserver un seul worker API et tester le parcours réel après chaque déploiement. |

## Éléments déjà solides à conserver

- Calculs numériques en Python avec `Decimal`, contrôle des colonnes et des
  unités, refus des opérations globales sur un bloc explicitement incomplet.
- Validation Pydantic du plan et des références ; citations construites depuis
  les métadonnées, sans exécution de Python ou SQL généré par le LLM.
- Reconstruction des parents et recontrôle des filtres ; une seule recherche
  corrective, sans boucle autonome.
- Processus d'ingestion séparés avec délais, limites d'upload et nettoyage sur
  erreur ; réutilisation des embeddings déjà chargés.
- API liée à localhost dans Compose, Qdrant/Ollama non publiés sur l'hôte,
  volumes persistants et application conteneurisée sans root.

## Recette sur la VM avant accès client

Effectuer les essais séquentiellement dans le portail, avec le PDF sélectionné.
Reformuler la question entière à chaque essai puisque l'historique n'est pas envoyé.

1. « Donne-moi les trois produits les plus chers selon le prix au détail. »
   Sur le référentiel Horizon fourni : Lessive 2 L (5,90 €), Café 250 g (4,90 €),
   Shampooing 250 ml (3,50 €), en ordre décroissant, avec le bon tableau cité.
2. « Donne-moi les trois produits les moins chers selon le prix au détail. »
   Eau 1,5 L (0,60 €), Yaourt nature (0,70 €), Coca-Cola 33 cl (1,00 €).
3. « Donne-moi trois produits les plus chers. » Si plusieurs prix sont
   disponibles sans convention préalable, demander la colonne ; pas deux refus
   identiques. Ne pas utiliser implicitement la marge.
4. « Donne le prix maximal et le stock restant minimal. » Garder deux demandes
   indépendantes et citer leurs tableaux respectifs.
5. « Selon le stock restant le plus faible, quel produit réapprovisionner ? »
   Exécuter le minimum du stock restant et en préciser la portée ; ne pas inventer
   de quantité optimale à commander.
6. Demander une information absente, puis une information dont deux sources se
   contredisent. Vérifier refus explicite ou désaccord signalé.
7. Sélectionner uniquement l'audio et reposer une question sur les prix du PDF :
   aucune donnée du PDF non sélectionné ne doit servir de preuve.
8. Reposer les questions sur la réunion et vérifier les noms à l'écoute ; ne pas
   accepter une identité inventée même si le reste du résumé est juste.

Recommencer avec des documents inconnus du développement. Consigner pour chaque
question : réponse, source et passage, bonne colonne, ordre/valeurs, refus ou
clarification attendu, limites annoncées. Des tests logiciels verts ne remplacent
pas cette recette. Aucune de ces étapes ne justifie une promesse de 100 % sur des
questions ou documents arbitraires.

## Mise en service de ces changements

Ces modifications sont locales jusqu'au commit et au push effectués par
l'opérateur. Sur la VM, après publication du commit et contrôle d'un arbre Git
propre :

```bash
cd ~/deal-voice
git pull --ff-only
docker compose --env-file .env.docker -f docker-compose.yml -f docker-compose.gpu.yml up -d --build app
```

Ce changement ne requiert pas de réindexation : les données et volumes existants
restent utilisés. Exécuter ensuite la recette ci-dessus ; ne pas publier un
nouveau score d'accuracy avant d'avoir examiné ces réponses réelles.
