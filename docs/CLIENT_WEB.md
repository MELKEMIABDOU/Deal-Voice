# Interface web client

Le portail utilise HTML, CSS et JavaScript, servis par FastAPI à la même adresse
que l'API. Aucun serveur Streamlit, service frontend, npm ou build JavaScript
n'est nécessaire. Les modèles et le pipeline RAG restent ceux de `.env`.

## Lancer dans l'environnement existant

Arrêter Streamlit et toute autre instance qui utilise `local_qdrant_db`.
Depuis la racine du projet :

```powershell
uv run --no-sync python -m uvicorn deal_voice.api.main:app --app-dir src --host 127.0.0.1 --port 8000 --workers 1
```

Ouvrir **http://127.0.0.1:8000**. Le premier démarrage charge les modèles
d'embedding/reranking et peut prendre du temps. Ollama doit déjà être démarré
avec le modèle configuré. Le portail pilote suppose `RBAC_ENABLED=false` :
les documents sont partagés dans cette installation mono-utilisateur.

Pour une nouvelle installation avec `uv` :

```powershell
uv sync --locked --no-dev --extra pdf --extra audio --extra local-ai
```

Installer aussi FFmpeg, configurer `.env` et mettre les modèles en cache avant
le test client. `HF_TOKEN` est nécessaire pour la diarisation audio et reste
côté serveur. Aucun secret n'est placé dans les fichiers HTML ou JavaScript.
Installer le modèle configuré dans Ollama séparément.

## Utilisation

- Importer un ou plusieurs PDF/audios ; le navigateur les envoie successivement.
- Chaque fichier est transféré sur disque avec une limite de taille vérifiée
  par le serveur, puis traité en arrière-plan. La page consulte l'état de
  l'import ; elle ne maintient pas une longue requête de transcription ouverte.
- Sélectionner les documents puis poser une question autonome. Les échanges
  précédents restent visibles dans la page mais ne sont pas transmis au modèle.
- Déplier les passages et vérifier les sources, pages et intervalles audio.
  « Bloc extrait entier » ne garantit pas que tout le PDF a été correctement lu.
- La suppression demande confirmation et retire l'index et les fichiers.

Une seule opération lourde est admise à la fois, pour limiter le cumul mémoire
et les accès à Qdrant local. Une autre demande reçoit HTTP 409. L'interface
reste disponible pour consulter l'état du serveur et de l'import.

## Déployer toi-même

Le même lancement fonctionne sur une machine dédiée après installation des
dépendances. Pour un conteneur ou un réseau privé, adapter l'adresse d'écoute
à `--host 0.0.0.0`. Le port reste configurable avec `--port`.

Conserver **un seul worker** et une seule instance avec Qdrant local. Ne pas
lancer Streamlit en parallèle sur la même base. Pour plusieurs processus,
préparer une architecture de stockage adaptée avant d'augmenter les workers.

Sur un hébergement distant :

1. Configurer un accès privé/authentifié et HTTPS au niveau de l'hébergement
   ou du reverse proxy. Ce portail n'ajoute pas d'identification client : son
   URL ne doit pas donner accès publiquement aux documents et à leur suppression.
2. Conserver `local_qdrant_db`, `data/raw` et `data/processed` sur un disque
   persistant et les sauvegarder ensemble, application arrêtée. Ne pas copier
   ces données dans l'image ou un dépôt public.
3. Transmettre les secrets par l'environnement serveur ; ne pas publier `.env`.
   Ollama et les workers d'extraction restent sur le réseau privé.
4. Adapter les délais du proxy pour les requêtes de recherche/génération,
   qui restent synchrones. Les imports utilisent des requêtes courtes suivies
   d'un polling. Le contrôle `/health` vérifie le serveur HTTP ; `/api/v1/status`
   indique l'initialisation et l'occupation, pas une preuve que le LLM répond.
5. Vérifier la mémoire et les délais avec les modèles et documents retenus.
   Un hébergement statique gratuit peut servir les assets, mais pas exécuter
   WhisperX, Docling, les embeddings et Ollama. Le frontal léger ne supprime
   pas le coût mémoire ni le temps d'inférence du backend.

Les 100 derniers états d'import sont conservés en mémoire, sans file durable.
Une coupure/redémarrage perd ce suivi : actualiser les documents et vérifier
ce qui a été indexé avant de relancer. Cette version convient à un pilote
avec une instance, pas à une promesse de reprise automatique des traitements.

## Limite fonctionnelle conservée

Le changement d'interface n'améliore pas automatiquement la précision du RAG.
Les erreurs d'attribution d'entités observées dans le diagnostic du
1 novembre 2026 restent connues. La nouvelle interface affiche les sources et
les contextes partiels sans présenter le verdict du modèle comme une garantie.
