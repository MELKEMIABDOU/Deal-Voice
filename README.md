# DealVoice

Local multimodal RAG for commercial calls, meetings and PDF documents.
A French web portal lets you import files, select documents, ask questions and
inspect the passages used to answer.

**Status:** a single-instance pilot. Authentication, durable import recovery
and client production acceptance still need deployment work. Model-generated
answers can be wrong, even when they include valid citations.

## What it does

- **Audio:** WhisperX transcription, alignment and speaker diarization, with
  saved Markdown/JSON transcripts and timestamped passages.
- **PDF:** Docling extraction, section-aware text chunks and extracted tables.
- **Search:** BGE-M3 embeddings, local Qdrant parent/child retrieval and an
  optional BGE reranker.
- **Answers:** Ollama proposes structured claims or table queries. Python
  renders citations and performs supported table calculations.
- **Interface:** HTML/CSS/JavaScript served by FastAPI; no frontend build needed.

See [architecture and limits](docs/ARCHITECTURE.md) for the layer contracts.

## Requirements

- Python in the project's supported range (`>=3.10,<3.14`) and `uv`.
- Ollama and the configured model.
- FFmpeg available on `PATH`.
- A Hugging Face token with access to the diarization models used by WhisperX.
- Enough RAM for embeddings/reranking plus extraction or generation.
  The upload pipeline currently uses WhisperX `medium`, French, CPU `int8`,
  batch size 1; installing a GPU alone does not change that configuration.

Model downloads require network access on first use. Cache them before a client
session. Test memory and latency on the actual deployment machine.

## Install and run

Commands below use PowerShell, from the repository root.

```powershell
uv sync --locked --no-dev --extra pdf --extra audio --extra local-ai
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
ollama pull qwen3:8b
```

Edit `.env` and set `HF_TOKEN` for audio diarization. Keep Ollama running
(start `ollama serve` in another terminal only if it is not already running).

```powershell
uv run --no-sync python -m uvicorn deal_voice.api.main:app --app-dir src --host 127.0.0.1 --port 8000 --workers 1
```

Open **http://127.0.0.1:8000**. Import a PDF or audio file, select it and ask a
question. Questions are independent: the displayed conversation is not sent
as chat history.

Keep one API process with local Qdrant. Stop another API/Streamlit instance
before opening the same database.

## Configuration and data

Copy supported configuration from [.env.example](.env.example); the complete
definitions are in [config.py](src/deal_voice/config.py).

| Setting | Default |
| --- | --- |
| LLM | `qwen3:8b` through `OLLAMA_MODEL_NAME` |
| Ollama server | `http://127.0.0.1:11434` |
| Context / output | 4096 / 512 tokens |
| Embeddings | `BAAI/bge-m3`, 1024 dimensions |
| Reranker | `BAAI/bge-reranker-v2-m3` |
| Qdrant directory | `./local_qdrant_db` |
| Upload limits | 100 MB/file, 5 files per browser batch, 120 minutes/audio |

Uploads live in `data/raw/`; transcript artifacts in `data/processed/`.
Qdrant vectors and parent records are kept in its configured directory.
These directories contain application data and must be backed up together.

Changing only the LLM requires downloading it and restarting the API.
Changing the embedding model requires a compatible new index and reindexing;
equal vector dimensions do not make two embedding models compatible.

## Tests and evaluation

Install development dependencies, including the optional interfaces used by
existing tests:

```powershell
uv sync --locked --extra all --group dev
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
uv run --no-sync ruff check src scripts tests streamlit_app.py
uv run --no-sync pytest -q
```

Default discovery includes unit tests and the deterministic document-lifecycle
integration test. It excludes real PDF/audio/retrieval model tests and DeepEval.
Some Python ML libraries are still imported by unit tests; no real model
weights or Ollama calls should be needed.

See [the testing and evaluation guide](docs/EVALUATION.md) for explicit model
tests, the 50-question dataset, memory-conscious runs and the historical
transcription baseline. Generated reports stay local under `reports/`.

## Optional development tools

- **Streamlit:** retained for local debugging. Install the `ui` extra and run
  `uv run --no-sync streamlit run streamlit_app.py`. Do not run it alongside
  another owner of the same Qdrant directory.
- **Wav2Vec2 benchmark:** retained as an experiment, not an ingestion backend.
- **DeepEval:** available through `--group evaluation`; excluded from normal
  test discovery.

## Client deployment

Read [the client deployment guide](docs/CLIENT_WEB.md). The current portal has
no built-in login; protect all remote access with authentication and HTTPS.
The optional role header is a development filter, not authenticated identity.

One expensive operation runs at a time. Other operations can receive HTTP 409.
Import state is held in memory and is lost on restart. `/health` checks the
HTTP process, not LLM readiness. These constraints must be addressed or explicitly
accepted before client use.

## Repository layout

```text
src/deal_voice/
  api/          HTTP endpoints and document operations
  ingestion/    Audio/PDF extraction, workers and indexing
  retrieval/    Parent/child search and reranking
  generation/   Prompts, schemas and response generation
  evidence/     Citation validation and table calculations
  web/          Client portal
  evaluation/   Evaluation metrics and resource observations
scripts/        Evaluation and pilot commands
tests/          Unit, integration and model evaluations
data/evaluation/ Reference texts and question sets
docs/           Architecture, evaluation and client operation guides
```

Secrets, model caches, runtime databases and generated reports are not source
code. Older committed reports remain in Git history; they are not current
accuracy guarantees.
