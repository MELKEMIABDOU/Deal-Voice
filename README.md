# Deal-Voice

AI-powered multimodal RAG system for analysing commercial calls, meetings, PDFs,
and business documents.

## Client web interface

A lightweight French client portal is served directly by FastAPI. It includes
PDF/audio import with background status, document selection, questions and
source excerpts. It needs no frontend build or Streamlit process.

```powershell
uv run --no-sync python -m uvicorn deal_voice.api.main:app --app-dir src --host 127.0.0.1 --port 8000 --workers 1
```

Open **http://127.0.0.1:8000**. Stop Streamlit first if it owns the same Qdrant
folder. See [the client launch and deployment guide](docs/CLIENT_WEB.md).
Deployment remains operator-managed; the frontend does not change model
accuracy or provide client authentication.

Local Ollama generation is memory-bounded by default: a 4096-token context,
512 generated tokens, four CPU threads, reasoning disabled, and model unloading
after each response. These values can be overridden with `OLLAMA_NUM_CTX`,
`OLLAMA_NUM_PREDICT`, `OLLAMA_NUM_THREAD`, `OLLAMA_REASONING`, and
`OLLAMA_KEEP_ALIVE`. `OLLAMA_BASE_URL` can point to a separate Ollama server.

## Pipelines

### PDF ingestion

```text
PDF -> Docling -> Markdown document -> parent chunks -> child chunks -> Qdrant
```

Docling preserves the document structure as Markdown. The parent-child index
uses the configured chunk sizes from `src/deal_voice/config.py` and retrieves
parent context after searching child embeddings.

### Audio ingestion

```text
MP3/WAV -> WhisperX -> alignment -> speaker diarisation
     -> CallTurn -> Markdown -> parent/child chunks -> Qdrant
```

Speaker diarisation requires a Hugging Face token in `.env`:

```dotenv
HF_TOKEN=hf_...
```

Never commit `.env` or expose the token in logs.

### Evidence and table-query contracts

The RAG pipeline uses the same contracts for different document domains:

| Layer | Contract |
|---|---|
| Ingestion | Keep PDF explanations within their section and associate tables through section IDs. Reject empty extraction; preserve malformed grids as text with validation errors. |
| Retrieval | Search child embeddings, reconstruct and deduplicate parent passages, apply source/document/role filters, then optionally rerank a bounded candidate set. Text, audio and tables remain eligible together. |
| Generation | Ollama uses a narrative, table-only or mixed planning schema according to the available evidence. Plans are validated before answering or executing. References use evidence IDs; the application renders citations from their metadata. |
| Table operations | Python executes `select`, `min`, `max`, `sum`, `mean`, `count` and numeric `sort` on validated columns and filters. There are no product-specific column aliases, model-generated code or model-computed arithmetic. |
| Evaluation | Score facts, source citations, locator details and values separately. Check all citations for a file, including multiple sections; use structured entity/field/value traces when available. |

Table schemas are sent to the model while the application retains the rows for
execution. Minimum and maximum operations retain tied rows unless an explicit
result limit is requested. Numeric operations reject malformed values,
ambiguous decimal formats and incompatible units. Aggregations require a
complete extracted block; selections from partial blocks remain partial.
The model-facing plans require a null decimal separator. The executor alone
uses the source's declared format, or the strict cell parser when none is
declared; the model cannot override it.
Missing columns and unsupported computations require a clarification or an
explicit failure rather than an invented result. `count` counts rows; distinct
counts and arbitrary joins are not supported.

`table_complete` means the entire **extracted Markdown block** is available.
`table_completeness_scope="extracted_block"` and `table_source_complete=false`
make clear that this does not verify OCR accuracy, every row in a multipage
PDF, or coverage of every relevant table in the corpus. Review extraction
quality before relying on a global conclusion.

Generation uses Ollama's native
[structured outputs](https://docs.ollama.com/capabilities/structured-outputs)
with a JSON schema. The prompt also explains the required fields;
constrained decoding alone does not explain their meaning. Text/audio-only
contexts show a compact, valid narrative JSON example while native decoding
still enforces the full narrative schema. Table and mixed contexts show their
compact JSON schema. The context budget counts the rendered prompt for the
passages actually included, preserving whole passages and the configured
context/output limits without loading another tokenizer or model.
Table-only contexts use a flat `TablePlan` (operation, evidence ID, column and
filters), without free-form claims. Mixed contexts retain both capabilities.
The executor reads the rows and calculates results. Table plans may explicitly
decline or request clarification instead of executing. Narrative/mixed decoding
branches and application validation enforce status/payload consistency, so an
insufficient-information result cannot include a clarification or factual payload.
Language instructions apply to claim text, not machine enums or source labels.
Narrative claims undergo one additional, sequential check with the same Ollama
model before display. It receives the question, claims and their cited original
passages, including source locators, and checks entity/role attribution, negation
and time. Its request is limited to 64 output tokens, with a separate input
budget, while preserving the configured context and thread limits.
An unsupported, contradicted, invalid or oversized check cannot release
the draft answer. Table-only answers retain their deterministic execution path.
This adds a model call, not a second resident model or a rewriting loop. The
verifier can still make mistakes; measure attribution errors and false refusals
on representative French examples. An insufficiency response refers to the
consulted passages, not a proven absence from the entire corpus.
Prompt definitions live in `src/deal_voice/generation/prompts.py`; generation
and rendering remain in `generator.py`, and output contracts in `schemas.py`.
Requests have a configurable
`OLLAMA_REQUEST_TIMEOUT_SECONDS` (default: 120 seconds).
Schema validation checks structure, references and the
allowed operation; it cannot prove that the model selected the intended
column, filter or supporting passage. The retrieval engine is semantic with
optional reranking, not a hybrid lexical/vector search. Context and candidate
budgets are bounded; they do not guarantee that every relevant passage is found.

Fact recall remains a lexical check with explicitly declared variants. A
structured value trace adds relation checks when present; unavailable relation
checks are reported separately. A passing regression test is not evidence of
universal accuracy or production readiness. Keep a separate, unseen validation
set after tuning questions or prompts.

Existing index entries remain readable, but must be reindexed to acquire the
new section and extraction metadata. Recreate prepared evaluation contexts
after changing the index or retrieval implementation. Keep historical reports
as historical results; rerun them before comparing the new generation pipeline.

## Tests

Install and run the project with `uv`:

```powershell
uv sync --locked --extra all --group dev
uv run pytest tests\unit -q
uv run pytest tests\integration\test_retrieval.py -q
```

The real PDF test runs Docling and verifies the resulting parent and child
chunk counts:

```powershell
uv run pytest tests\integration\test_real_ingestion.py::test_real_pdf_to_parent_child_chunks -q -s
```

The real audio test is intentionally opt-in because it downloads models and
can take several minutes:

```powershell
$env:RUN_REAL_AUDIO_TESTS="1"
uv run pytest tests\integration\test_real_ingestion.py::test_real_audio_to_parent_child_chunks -q -s
```

To save the result of a transcription without rerunning WhisperX later, pass
an output directory to `transcribe_to_document`:

```python
document = transcriber.transcribe_to_document(
    audio_path,
    hf_token=settings.hf_token,
    output_dir="data/processed",
)
```

This creates a readable `*_transcript.md` file and a structured
`*_transcript.json` file containing speakers, text, and timestamps.

Search responses expose optional audio citation fields:

```json
{
  "source": "call.mp3",
  "speakers": ["SPEAKER_00", "SPEAKER_01"],
  "start_time": 0.0,
  "end_time": 128.4
}
```

## Ingestion process isolation

`WhisperXTranscriber.transcribe_and_diarize` and `transcribe_to_document` run
WhisperX, alignment and diarization in a fresh Python process per file. That
process exits before the caller receives the transcript. The parent formats
and saves the transcript, then indexes it with its existing embedding/Qdrant
instances. It does not decode the audio or import WhisperX. The duration limit
is checked on the worker's 16 kHz waveform before ASR model loading.

PDF extraction uses the same process transport. Docling is imported and runs
inside its child, returns Markdown, and exits before the parent indexes it.
The existing table extraction options and document metadata are preserved.

The worker uses the same Python environment and receives its request, including
the Hugging Face token, over stdin. Transcript results use a validated JSON
file; library/native output goes to a temporary log rather than a RAM buffer.
Failures and timeouts propagate without retrying the models inside Streamlit.
Temporary exchange files are removed after the worker exits.

On Windows the parent owns a
[Job Object](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects)
containing the audio worker and its descendants. Closing the parent or reaching
the timeout terminates that job, including FFmpeg. Cleanup checks that no active
job processes remain before admitting another worker; if it cannot confirm this,
further audio uploads fail until the application restarts. The worker relinquishes
its own job handle immediately so it cannot keep itself alive after parent death.
On other platforms the direct child is killed on timeout and watches for parent
exit; descendant containment should also be provided by the deployment supervisor.

The application settings `AUDIO_WORKER_TIMEOUT_SECONDS` (default 3600) and
`AUDIO_WORKER_CPU_THREADS` (default 2) control each run. The existing
`MAX_AUDIO_DURATION_MINUTES` setting still applies to uploads. A semaphore
permits one audio or PDF worker at a time **per parent process**; waiting for a slot
has the same timeout. This is a synchronous local worker, without a durable
task queue or a global limit across multiple API/UI processes.
PDF runs use `PDF_WORKER_TIMEOUT_SECONDS` (default 1800) and
`PDF_WORKER_CPU_THREADS` (default 2).

Process isolation reclaims worker allocations at exit; it is not a RAM/VRAM
quota and cannot guarantee that the peak during ingestion fits the host.

## Controlled application pilot

The [latest short single-user pilot](reports/pilot/essential-fixes-20261101.md)
passed the audio and PDF upload/index/search/generation paths. Three additional
generation checks covered minimum, filtering and missing information. This
does not establish general accuracy or concurrent-user capacity. The
[earlier audio-isolation report](reports/pilot/audio-isolation-20261101.md) and
[initial report](reports/pilot/pilot-20261101.md) remain historical baselines.

Use an idle local Ollama instance and cached models to exercise a short PDF
and audio through the actual upload/index/search/generation functions:

```powershell
uv run python scripts\validate_pilot.py `
  --pdf data\evaluation\Rapport_gestion_produits_Horizon_structure.pdf `
  --pdf-page 3 `
  --audio "data\evaluation\B2B Appel.mp3" `
  --audio-seconds 45 `
  --single-user-only
```

The default questions/expected facts describe these evaluation fixtures. For
other documents supply `--pdf-question`, repeatable `--pdf-expected`,
`--audio-question` and repeatable `--audio-expected`.
Use `--audio-first` to exercise the audio lifecycle before the PDF case; the
same functional and memory checks still apply to both cases.
`--single-user-only` checks those cases and skips concurrent load tests.

The worker owns an isolated database and copied uploads under `data/raw/pilot-*`;
the normal knowledge base is untouched. Reports, phase checkpoints and memory
samples are retained under `reports/pilot/`. After successful single-user checks,
omitting `--single-user-only` enables bursts of 2 then 5 in-process ASGI clients
querying the real search endpoint using
the same models/client. This tests short repeated queries, not browser/network
behavior, simultaneous uploads, or sustained capacity.

A separate supervisor stops its worker if available Windows physical RAM or
commit headroom falls below `--min-free-gib` (default 4), or the overall timeout
expires. Admission also reserves `--generation-reserve-gib` (default 8) for model
loading plus additional concurrent-request headroom. These are precautionary
budgets, not hard memory caps: termination does not necessarily stop Ollama's
separate runner immediately. A skipped load stage does not establish capacity.
Exit codes: 0 all stages passed; 1 failed; 2 partial with unexecuted load stages.
The supervisor finalizes interrupted checkpoints after its job is stopped.
`supervisor.json` records the termination reason; older reports may retain a
worker checkpoint marked `running` and must be read with this supervision log.

Upload indexing reuses the search engine's embedding instance. Within the audio
worker, stages release their own ASR/alignment/diarization references and collect
cycles before the next stage, following the phase separation used by the
[WhisperX transcription implementation](https://github.com/m-bain/whisperX/blob/main/whisperx/transcribe.py).
Each extraction worker exits before indexing. The pilot measures remaining system
headroom rather than assuming that separation is sufficient on every host.

## Local backup and restore

Stop Streamlit, the API and all other Qdrant owners before copying data.
Back up the database (including `parent_docstore`), original files and transcripts
together. For the default paths, from the project directory:

```powershell
$backupDir = Join-Path $env:USERPROFILE ("DealVoice-backups/" + (Get-Date -Format "yyyyMMdd-HHmmss"))
New-Item -ItemType Directory -Path "$backupDir/data" -ErrorAction Stop
Copy-Item -LiteralPath "local_qdrant_db" -Destination "$backupDir/local_qdrant_db" -Recurse
Copy-Item -LiteralPath "data/raw", "data/processed" -Destination "$backupDir/data" -Recurse
```

Adjust paths if configured differently. To restore, place these three folders
in a stopped installation using the same model configuration, then verify search
and access to the original files before reopening it. First verify the restore
in a separate copy of the installation. The integration test
`test_stopped_database_and_artifacts_can_be_backed_up_and_restored` exercises
this copy/restore procedure on a real temporary Qdrant database with deterministic
embeddings, without touching the application's knowledge base.

## Transcription evaluation

The recommended ASR benchmark uses the project's own representative audio
files with a manually validated `.txt` file next to each audio file:

```text
data/evaluation/
├── B2B Appel.mp3
├── B2B Appel.txt
├── Reunion dequipe.mp3
└── Reunion dequipe.txt
```

Run the same French WhisperX models on every audio/reference pair:

```powershell
uv run python scripts\evaluate_transcription_dataset.py `
  --evaluation-dir data\evaluation `
  --models base small medium large-v3 `
  --language fr `
  --output-dir reports\transcription\evaluation-dataset
```

The command reports audio duration, transcription time, real-time factor,
WER (Word Error Rate), and CER (Character Error Rate). Lower WER/CER is
better. It writes JSON, CSV, and Markdown reports, including per-file results
and weighted summaries. Speaker labels such as `[Speaker 1]` in the reference
files are ignored during metric calculation.

Results on the current two-file evaluation set (CPU, `int8`):

| Model | Weighted WER | Weighted CER | Total transcription time |
|---|---:|---:|---:|
| `base` | 17.92% | 8.09% | 20.6 s |
| `small` | 9.83% | 5.39% | 49.9 s |
| `medium` | 7.37% | 3.75% | 131.5 s |
| `large-v3` | **4.91%** | **3.39%** | 207.5 s |

These scores describe this local evaluation set only. Add more representative
audio/reference pairs before making a production model decision. For a fast
development default use `base` or `small`; for the best measured accuracy on
this set use `large-v3`.

For PDF results, these fields are `null`. The current values describe the
indexed audio document range; turn-level citations can be added later with a
turn-aware splitter.

Unit tests mock or isolate external services where possible. The RAG quality
evaluation additionally requires a running Ollama server and model.

## Access control

The default mode is single-user: searches do not apply a role filter, and the
Streamlit interface hides role selection. This is intended for a local
prototype operated by one person. It does not isolate data between users.

Optional role filtering can be enabled with `RBAC_ENABLED=true`. In that mode,
the API accepts `X-User-Role` and filters documents by `allowed_roles`; the
Streamlit interface also exposes role selection. This header is only a
development mechanism: enabling the filter does not authenticate users or
provide secure team sharing. A multi-user deployment must bind permissions to
an authenticated identity and enforce document ownership or team access on the
server.

## Streamlit interface

Launch the interactive RAG interface:

```powershell
uv run streamlit run streamlit_app.py
```

Then open the displayed local URL, normally
`http://localhost:8501`.

The interface lets you:

- upload and index one or several PDF/MP3/WAV/M4A/FLAC files;
- choose the Ollama model and number of sources;
- choose one/several files uploaded in the current Streamlit session;
- search with or without the optional RBAC role filter;
- inspect the parent documents returned by Qdrant;
- view audio speakers, timestamps, and reranking scores;
- read the final answer generated by the LLM.

Each indexed upload receives a unique document ID, so files with the same name
remain distinct. The interface lists indexed documents by ID and lets you
delete selected documents, including their Qdrant vectors, parent records,
original uploads, and audio transcript artifacts. Upload limits can be
configured with `MAX_UPLOAD_SIZE_MB` (default 100), `MAX_UPLOAD_FILES`
(default 5), and `MAX_AUDIO_DURATION_MINUTES` (default 120). These checks run
in the Streamlit application; deployments exposed through another upload API
must enforce equivalent limits at that entry point too.

Uploads that fail during parsing, transcription, or indexing have their newly
created raw files and transcript artifacts cleaned up; partially indexed
vectors and parent records are rolled back when possible. Existing index data
from before document IDs were introduced can be migrated with **Migrer les
documents historiques sans ID** in Streamlit. Migration assigns a stable ID
grouped by legacy source filename and preserves indexed content. Older
same-name uploads cannot be distinguished when their source metadata is
identical; back up the local Qdrant directory before migrating legacy data.

Before launching it, start Ollama:

```powershell
ollama serve
ollama pull qwen3:8b
```

Use **Ajouter un document** in the Streamlit interface to create the index.
There is no seed database or fictitious document loader: only files uploaded
through the interface are added to the session's selectable source list. The
uploaded file is saved under `data/raw/`, processed, and indexed into the
configured Qdrant collection. Audio uploads also write their Markdown and JSON
transcription artifacts under `data/processed/` and require `HF_TOKEN`.

`Nombre maximum de résultats` controls `top_k`: it is the maximum number of
parent documents available to generation after vector retrieval and reranking.
It is not the number of child chunks used internally by Qdrant. Audio parents
repeat complete trailing turns within the configured overlap budget; child
chunks also use the configured overlap. PDF text parents respect section
boundaries, and full extracted table blocks are stored as table parents.
After a successful upload, an exact binary-content duplicate replaces its old
index entry; same-name files with different content remain distinct.
Audio parent chunks are grouped from diarized speaker turns and keep the
speaker, start time, and end time metadata instead of splitting the transcript
at arbitrary character boundaries.
The application uses WhisperX `medium` for French transcription on CPU with
`int8` compute and a batch size of 1. Transcript metadata records the model,
language, device, and compute type used for each new upload.

### Transcription benchmark

For the project's model selection, use the multi-file reference evaluation
below. It runs on `data/evaluation/` and is the primary benchmark because it
covers both a B2B call and a team meeting. The single-file command shown here
remains useful for quick experiments on another recording.

Compare transcription models on the same audio file without running speaker
diarisation:

```powershell
uv run python scripts\evaluate_transcription.py `
  --audio data\raw\appel.mp3 `
  --models base small `
  --language fr
```

The benchmark measures audio duration, transcription time, real-time factor,
segment and word counts, and adjacent repeated phrases. It writes JSON, CSV,
Markdown, and one transcript per model under `reports/transcription/`.
With a manually validated reference transcript, it also calculates WER
(Word Error Rate) and CER (Character Error Rate). Lower scores are better:

```powershell
uv run python scripts\evaluate_transcription.py `
  --audio "data\raw\centre d'appels prise de rendez vous mutuelle sante assurance.mp3" `
  --models base small medium large-v3 `
  --language fr `
  --reference data\raw\reference.txt `
  --output-dir reports\transcription\centre-appels-all-models
```

Benchmark results on a 403.6-second French call, using CPU `int8` inference:

| Model | Transcription (s) | RTF | WER | CER |
|---|---:|---:|---:|---:|
| `base` | 35.6 | 0.09 | 46.27% | 30.96% |
| `small` | 84.7 | 0.21 | 41.58% | 28.56% |
| `medium` | 209.4 | 0.52 | 36.50% | 25.59% |
| `large-v3` | 354.3 | 0.88 | **31.49%** | **21.20%** |

`large-v3` achieved the best transcription accuracy in this single-file
evaluation. Use the multi-file results below when choosing the production
model; `base` and `small` remain useful for fast development.
The benchmark is specific to this audio and reference; it should be repeated
on representative files before making a production model decision.

### Alternative French ASR model

Wav2Vec2 can be benchmarked separately from WhisperX:

```powershell
uv run python scripts\evaluate_wav2vec2_french.py `
  --audio "data\raw\appel.mp3"
```

The default model is
`jonatasgrosman/wav2vec2-large-xlsr-53-french`. It is a French CTC model and
does not provide WhisperX's alignment or speaker diarisation by itself. Its
report is written under `reports/transcription/wav2vec2-french/`. Compare it
with WhisperX using the same manually validated reference before changing the
production pipeline.

On the same 403.6-second call, Wav2Vec2 French scored **81.81% WER** and
**49.74% CER** in 220.8 seconds (RTF `0.55`). It therefore remains an
experimental benchmark and is not used by the production ingestion pipeline.

### Multi-file reference evaluation

This is the recommended evaluation for Deal-Voice model selection.

The evaluation dataset can contain one audio file and one `.txt` reference
with the same filename stem. The current dataset contains:

```text
data/evaluation/
├── B2B Appel.mp3
├── B2B Appel.txt
├── Reunion dequipe.mp3
└── Reunion dequipe.txt
```

Run all WhisperX models on every pair:

```powershell
uv run python scripts\evaluate_transcription_dataset.py `
  --evaluation-dir data\evaluation `
  --models base small medium large-v3 `
  --language fr `
  --output-dir reports\transcription\evaluation-dataset
```

The report contains per-file results and weighted WER/CER summaries in
`benchmark.json`, `benchmark.csv`, and `benchmark.md`. Speaker labels such as
`[Speaker 1]` in the references are ignored when calculating the error rates.

### RAG retrieval evaluation

#### 50-question French evaluation set

`data/evaluation/questions.jsonl` contains 50 reference questions spanning
transcript facts, PDF text, tables, cross-section questions, and information
that is absent from the source documents. The evaluator accepts this JSONL
schema and the legacy JSON-array question files. It maps transcript reference
names such as `B2B-Appel.txt` to indexed files with the same name stem, such as
`B2B Appel.mp3`.

Index the two audio files and the Horizon PDF in Streamlit before evaluating.
The generation evaluator searches the full index without filtering by the gold
source; it reports separately whether that source was actually retrieved and
whether the answer cites it. Use `--category` and `--limit` for a small pilot:

```powershell
uv run python scripts\evaluate_generation.py `
  --questions data\evaluation\questions.jsonl `
  --category pdf_text `
  --model qwen3:8b `
  --top-k 3 `
  --limit 1 `
  --no-reranker `
  --output reports\rag\generation-pilot.json
```

After the pilot is stable, run the full set:

```powershell
uv run python scripts\evaluate_rag.py `
  --questions data\evaluation\questions.jsonl `
  --top-k 5 `
  --no-reranker `
  --output reports\rag\retrieval-50-questions.json

uv run python scripts\evaluate_generation.py `
  --questions data\evaluation\questions.jsonl `
  --model qwen3:8b `
  --top-k 5 `
  --no-reranker `
  --output reports\rag\generation-50-questions.json
```

For a laptop with limited memory, prepare the retrieved passages in a separate
process first. Close Streamlit to release the local Qdrant lock. Stop any
loaded Ollama model before preparation; keep WhisperX and generation jobs idle.

```powershell
$env:PYTHONPATH="src"
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
$env:OMP_NUM_THREADS="2"
$env:MKL_NUM_THREADS="2"

uv run python scripts\evaluate_generation.py `
  --questions data\evaluation\questions.jsonl `
  --top-k 3 `
  --no-reranker `
  --prepare-contexts reports\rag\medium-50.contexts.json
```

Wait for that command to exit: process exit releases BGE-M3 and Qdrant.
Then evaluate just one question using the saved passages:

```powershell
uv run python scripts\evaluate_generation.py `
  --questions data\evaluation\questions.jsonl `
  --from-contexts reports\rag\medium-50.contexts.json `
  --model qwen3:8b `
  --limit 1 `
  --output reports\rag\generation-isolated-1.json
```

`--from-contexts` does not load the retrieval engine, embeddings, or reranker,
and does not open Qdrant. It reuses the saved passages, metadata, retrieval
settings and measured retrieval latency. The question-file hash and selected
question text are validated before generation; prepare again after changing
the dataset, and whenever you want to evaluate a new index. Reference answers
and expected source files are used only for scoring, never for retrieval.
For subsequent small batches, use `--offset 1 --limit 4`, then `--offset 5
--limit 5`, with a different output filename each time.

Every run writes `<output stem>.memory.jsonl` (or use `--memory-log PATH`).
Preparation uses `<contexts stem>.memory.jsonl`. On Windows this records
system available RAM, commit limit and availability, and the Python process's
working set/private/peak working set every second and at phase changes. Logs
are appended and flushed so the last available samples can be inspected after
an interruption. It does not measure VRAM, cap memory, or guarantee that Windows
will remain stable. This separation reduces simultaneous model residency;
it does not establish the cause of a previous shutdown. No inference is needed
to run `--help` or the unit tests for these controls.

Both scripts accept `--category <name>` (repeatable) and `--limit N`.
The generation evaluator also accepts `--offset N`, so memory-constrained local
runs can process non-overlapping batches. They write
an atomic `.checkpoint.json` file after every completed question, so answers
from finished questions can be inspected after an interruption. The checkpoint
does not resume the run; restarting evaluates the questions again. The generation report
separates expected-source availability, source hits in retrieved results, and
source citations in the answer. It also reports fact/value recall, refusals,
explicitly listed forbidden-phrase matches, latency, and category results.
Forbidden-phrase compliance is not a semantic faithfulness score. Citation page
details are scored only when the indexed context includes page metadata; the
current PDF parser does not preserve page numbers. These 12 audio questions use
transcript text as their reference and do not measure ASR or diarization.
The French-language rate is a lexical heuristic, not an automatic language
quality judgment. When a question explicitly requires French, this heuristic
also contributes to pass/fail; short or mixed-language answers need manual
review. Answer questions
must include at least one source and one required fact or expected value so
incomplete references cannot silently score as pass. The retrieval rate counts
a multi-source question as retrieved only when every expected source is found.
Citation sections are checked against the citation for their own source file;
an extra section attributed to another source fails the deterministic check.

The reference questions in `data/evaluation/questions.json` evaluate whether
the retriever returns the expected audio source and enough expected facts.
Index the evaluation audio files through Streamlit first, then run:

```powershell
uv run python scripts\evaluate_rag.py `
  --questions data\evaluation\questions.json `
  --top-k 5
```

The report is written to `reports/rag/retrieval-evaluation.json` and contains
Source Recall@k, expected-fact recall, retrieved sources, and missing facts.
This evaluates retrieval independently from the LLM, so a generation failure
does not hide a retrieval regression.

Expected facts are human-defined business concepts, not simply every word
spoken in an audio file. Literal dialogue terms can be stored as
`literal_evidence` for secondary inspection, but they do not determine the
main RAG quality score.

Historical 10-question retrieval baseline (not comparable to the corrected
50-question evaluation):

| Metric | Result |
|---|---:|
| Questions | 10 |
| Source Recall@5 | **100%** |
| Mean fact recall | **96.67%** |

Nine questions reached 100% fact recall. The remaining question retrieved the
source and most of the expected facts but missed the phrase `fontaines à eau`
for the broad commercial-objective query; the same concept was retrieved by a
more specific product question. This baseline is stored in
`reports/rag/evaluation-10-questions.json` and should be rerun after retrieval
or chunking changes. The next evaluation layer is the final LLM response:
French-language compliance, factuality, citation correctness, and
hallucination rate.

### LLM generation evaluation

After the evaluation audio files have been indexed and Ollama is running,
evaluate the final answers on the same 10 questions:

```powershell
uv run python scripts\evaluate_generation.py `
  --questions data\evaluation\questions.json `
  --model qwen3:8b `
  --top-k 5 `
  --output reports\rag\generation-evaluation.json
```

This produces deterministic baseline checks for source citation, expected-fact
recall, non-empty answers, and French-language compliance. It is a regression
signal, not a replacement for human review or an LLM-as-judge evaluation:
the report keeps exact lexical recall separate from controlled concept-variant
recall, because paraphrases can be semantically correct without containing the
exact expected words. Retrieval is performed over the full index; the expected
source is used only for scoring.

The following 10-question generation results are historical. The earlier
evaluator filtered retrieval by the reference source, which made retrieval
results optimistic, and named answer citation coverage `source_accuracy`. The
corrected report separates retrieved-source hits from citations in the answer;
rerun the evaluation before using these old numbers as a current baseline.

The first controlled generation smoke test used `qwen3:8b` on one question:

| Model | Questions | Source accuracy | Concept fact recall | French compliance | Pass rate |
|---|---:|---:|---:|---:|---:|
| `qwen3:8b` | 1 | **100%** | **100%** | **100%** | **100%** |

This is a smoke-test baseline, not a general model-quality claim. The report is
stored in `reports/rag/generation-qwen3-variants-test.json`. Run the full
10-question evaluation only after confirming local hardware stability.

The first complete 10-question generation run with `qwen3:8b` produced:

| Metric | Result |
|---|---:|
| Questions | 10 |
| Source accuracy | **100%** |
| Mean concept fact recall | **83.33%** |
| French compliance | **100%** |
| Pass rate | **70%** |

The report is stored in `reports/rag/generation-qwen3-10-questions.json`.
Three answers did not reach the deterministic 80% fact threshold: the product
question omitted the expected commercial-proposal concept, and two meeting
questions did not mention the expected responsibility or compensation concepts.
These failures require human review because the current checks remain
concept- and phrase-based; they are not proof of hallucination or poor French.

After aligning the expected concepts and adding controlled French paraphrases,
the v3 report reached the following result without changing the model, prompt,
or retrieval configuration:

| Metric | Result |
|---|---:|
| Questions | 10 |
| Source accuracy | **100%** |
| Concept fact recall | **100%** |
| French compliance | **100%** |
| Pass rate | **100%** |

The v3 report is stored in `reports/rag/generation-qwen3-10-questions-v3.json`.
This is a positive-set regression baseline. The next evaluation phase should
add out-of-scope questions and verify that the system refuses unsupported
claims instead of generating plausible answers.

Out-of-scope questions are stored separately in
`data/evaluation/questions-negative.json`. They use
`"expected_behavior": "refusal"` and are scored with `refusal_accuracy`, rather
than source or fact recall. Run them with the same evaluator:

```powershell
uv run python scripts\evaluate_generation.py `
  --questions data\evaluation\questions-negative.json `
  --model qwen3:8b `
  --top-k 3 `
  --output reports\rag\generation-qwen3-negative.json
```

This negative set checks that unsupported questions receive an explicit
insufficient-information response in French. It is intentionally separate from
the positive 10-question baseline.

The first negative-set run with `qwen3:8b` reached:

| Metric | Result |
|---|---:|
| Questions | 4 |
| Refusal accuracy | **100%** |
| French compliance | **100%** |
| Pass rate | **100%** |

The report is stored in `reports/rag/generation-qwen3-negative-v2.json`.
`answer_citation_file_recall` is not applicable to refusal questions: a correct
refusal should not cite a source as if it contained the requested fact.

### Historical 10-question evaluation baseline

The earlier local baseline separates retrieval, supported-answer generation,
and unsupported-question refusal:

| Layer | Dataset | Result |
|---|---|---:|
| Retrieval | 10 positive questions | Source Recall@5 **100%**, fact recall **96.67%** |
| Generation | 10 positive questions | Source accuracy **100%**, concept fact recall **100%**, pass rate **100%** |
| Refusal | 4 negative questions | Refusal accuracy **100%**, pass rate **100%** |

These results are regression baselines on a small, manually curated dataset.
They do not represent production accuracy. Before presenting the project as
production-ready, expand the dataset, add human review, measure latency, and
test access-control and cross-source isolation cases.

The evaluation scripts also record per-question latency. Retrieval reports
contain `retrieval_latency_ms` and aggregate mean/p50/p95/max values. Generation
reports separate retrieval, generation, and total latency, with mean/p50/p95
values. These timings are machine-specific and should be compared only when
the model, hardware, warm-up state, `top_k`, and dataset are held constant.

### Benchmark versus evaluation

This project uses both terms deliberately:

- A **benchmark** measures a system or model under a fixed workload. It answers
  questions such as: how long does retrieval take, which Whisper model has the
  lowest WER/CER, and how much time does Qwen3 need to generate an answer?
- An **evaluation** checks whether the output meets a defined quality
  requirement. It answers questions such as: was the correct source retrieved,
  were the expected business concepts covered, was the answer faithful, and
  did the system refuse an unsupported question?

The benchmark is the measurement procedure; the evaluation is the quality
judgement applied to its outputs. Both must use a frozen dataset, fixed
configuration, versioned reports, and explicit acceptance criteria.

The reproducible evaluation matrix is:

| Layer | Dataset or command | Main signals |
|---|---|---|
| Transcription benchmark | `scripts/evaluate_transcription_dataset.py` | WER, CER, duration, RTF |
| Retrieval baseline | `data/evaluation/questions.json` | Recall@5, fact recall, MRR |
| Retrieval robustness | `data/evaluation/questions-robustness.json` | paraphrase recall, latency |
| Supported generation | `data/evaluation/questions.json` | citation, fact recall, French compliance |
| Refusal and safety | `data/evaluation/questions-negative.json` | refusal accuracy, fidelity |
| Multi-hop generation | `data/evaluation/questions-multihop.json` | completeness, source, faithfulness |

The current validated local baselines are:

| Test | Result |
|---|---:|
| Unit tests | **34 passed** |
| Positive retrieval Source Recall@5 | **100%** |
| Positive retrieval mean fact recall | **96.67%** |
| Robust retrieval Source Recall@5 | **100%** |
| Robust retrieval mean fact recall | **100%** |
| Qwen3 positive generation pass rate | **100%** |
| Qwen3 refusal accuracy | **100%** |
| Qwen3 multi-hop pass rate | **100%** |
| Robust retrieval latency p95 | **182.27 ms** |
| Multi-hop Qwen3 generation latency | **24.41 s** for one smoke-test question |

These are local baselines on a small, manually curated dataset. They are
regression signals, not production guarantees. The reports are kept under
`reports/` and should not be overwritten when a new model, prompt, chunking
strategy, or embedding model is tested. Use a versioned filename such as
`generation-qwen3-10-questions-v4.json`.

### Release gate

Before merging a change that affects ingestion, chunking, retrieval, prompts,
or models, run the offline gate:

```powershell
uv run pytest tests\unit -q
uv run python -m json.tool data\evaluation\questions.json > $null
uv run python -m json.tool data\evaluation\questions-negative.json > $null
uv run python -m json.tool data\evaluation\questions-robustness.json > $null
uv run python -m json.tool data\evaluation\questions-multihop.json > $null
```

Then run only the relevant benchmark layer and save a new report. Compare the
new report with the previous baseline before accepting the change. A quality
regression should be investigated rather than hidden by changing the threshold
or expected facts without human review.

### Next roadmap

The next work should be performed in this order:

1. Expand the golden dataset from 10 positive and 4 negative questions to
   approximately 50--100 questions, with human-reviewed facts and relevant
   chunks.
2. Replace the current source-level precision and binary nDCG approximations
   with graded chunk relevance labels (`0` to `3`) and true nDCG@k.
3. Add human review for faithfulness, relevance, completeness, citation
   correctness, and French quality; keep these dimensions separate.
4. Add security tests for prompt injection in transcripts/PDFs, unauthorized
   source access, and cross-tenant or cross-session leakage.
5. Add CI for offline tests and JSON/schema validation. Keep GPU/Ollama
   benchmarks manual or scheduled because they depend on local hardware.
6. Compare Qwen3, Mistral, and other models using the same dataset, prompt,
   `top_k`, temperature, warm-up protocol, and latency measurements.

### Professional evaluation extensions

The current evaluation is deliberately split into independent layers. Source
selection remains a user-controlled feature in the application; it is not
silently imposed as a system security boundary.

#### Retrieval quality

`evaluate_rag.py` now reports:

- Source Recall@k;
- mean fact recall;
- precision@k for the expected source;
- MRR (mean reciprocal rank);
- binary source-rank score (a temporary approximation, not true nDCG@k).

Run the same evaluation on paraphrased questions with:

```powershell
uv run python scripts\evaluate_rag.py `
  --questions data\evaluation\questions-robustness.json `
  --top-k 5 `
  --output reports\rag\retrieval-robustness.json
```

The robustness dataset changes wording while preserving the expected source
and business facts. This is a deterministic retrieval regression set, not a
claim of general language robustness.

#### Question robustness

The paraphrase set covers short questions, synonyms, colloquial wording, and
different question structures across both audios. In a production program,
expand it with spelling errors, multilingual inputs, ambiguous questions, and
domain-specific abbreviations, then freeze the dataset by version.

#### Hallucination and faithfulness

Generation questions may define `forbidden_claims`. The evaluator reports
`forbidden_claims_found` and `forbidden_claim_compliance_rate` for those listed
phrases. This deterministic lexical check does not detect every unsupported
claim and is not a semantic faithfulness score. Use human review or an
LLM-as-judge to assess subtle unsupported inferences.

#### Multi-hop reasoning

`data/evaluation/questions-multihop.json` contains questions that require
combining several facts from one conversation rather than answering from a
single passage. The evaluation still scores each expected concept separately,
so a partially complete answer is visible instead of being reduced to a
single subjective score.

Run one smoke test first:

```powershell
uv run python scripts\evaluate_generation.py `
  --questions data\evaluation\questions-multihop.json `
  --model qwen3:8b `
  --top-k 3 `
  --limit 1 `
  --output reports\rag\generation-qwen3-multihop-smoke.json
```

Only compare a thinking-enabled configuration after the normal configuration
has been measured on the same questions. Keep retrieval, prompt, model
temperature, and `top_k` constant so the comparison isolates generation
behavior.

Controlled variants include `rapidement` and `urgence` for the expected fact
`vite`. This prevents a semantically correct paraphrase from being counted as
missing while keeping the accepted vocabulary explicit and reviewable.

### Multilingual retrieval models

The retrieval stack uses multilingual models:

```text
Embeddings: BAAI/bge-m3
Vector size: 1024
Reranker: BAAI/bge-reranker-v2-m3
```

These models are better suited to French and mixed-language conversations than
the previous English-oriented models. They are downloaded automatically by
Sentence Transformers on first use.

Because the embedding dimension changed from 384 to 1024, the existing local
Qdrant collection must be recreated and all files must be uploaded/indexed
again. Stop Streamlit first, then remove only the local database directory:

```powershell
Remove-Item -Recurse -Force .\local_qdrant_db
uv run streamlit run streamlit_app.py
```

Do not use the old vectors after this change: vectors created with the previous
embedding model are not compatible with `BAAI/bge-m3`.

After changing embedding configuration, restart the Streamlit process
completely. A hot reload can keep an old Qdrant vector store in memory.

The source selector starts empty. Existing files from a previous run are not
displayed as selectable chat sources. This avoids asking questions against
documents that were not uploaded in the current session.
When the Streamlit process stops, cached model and Qdrant resources are cleared;
the indexed data on disk remains available for a future ingestion or cleanup.

### Changing the LLM

The default model is configured in
`src/deal_voice/config.py`:

```python
ollama_model_name: str = "llama3"
```

Prefer overriding it in `.env` without editing Python:

```dotenv
OLLAMA_MODEL_NAME=mistral
```

Download the model once, then restart Streamlit:

```powershell
ollama pull mistral
uv run streamlit run streamlit_app.py
```

You can also change the model for the current Streamlit session in the
**Modèle Ollama** field. The selected model must already exist in Ollama.
