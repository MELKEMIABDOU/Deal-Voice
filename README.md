# Deal-Voice

AI-powered multimodal RAG system for analysing commercial calls, meetings, PDFs,
and business documents.

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

## Tests

Install and run the project with `uv`:

```powershell
uv sync --group dev
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
ollama pull llama3
```

Use **Ajouter un document** in the Streamlit interface to create the index.
There is no seed database or fictitious document loader: only files uploaded
through the interface are added to the session's selectable source list. The
uploaded file is saved under `data/raw/`, processed, and indexed into the
configured Qdrant collection. Audio uploads also write their Markdown and JSON
transcription artifacts under `data/processed/` and require `HF_TOKEN`.

`Nombre maximum de résultats` controls `top_k`: it is the maximum number of
parent documents returned to the LLM after vector retrieval and reranking. It is not
the number of child chunks used internally by Qdrant. For audio documents,
parent and child overlaps are disabled to avoid repeating the same dialogue
across adjacent chunks; PDF chunking keeps its configured overlaps. Re-uploading
the same filename replaces its existing vectors instead of duplicating them.
Audio parent chunks are grouped from diarized speaker turns and keep the
speaker, start time, and end time metadata instead of splitting the transcript
at arbitrary character boundaries.
The WhisperX selector lets you choose `base` or `small` for better transcription
quality at the cost of CPU time and memory.

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

The current 10-question retrieval baseline, using the two isolated evaluation
audio files, is:

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
exact expected words. Each question is evaluated against its expected source to
avoid mixing unrelated conversations.

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
`source_accuracy` is not applicable to refusal questions: a correct refusal
should not cite a source as if it contained the requested fact.

### Current evaluation baseline

The current local baseline separates retrieval, supported-answer generation,
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
`unsupported_claims_found` and `fidelity_accuracy` in addition to fact recall.
This deterministic lexical check catches explicitly prohibited claims, but it
does not replace human review or an LLM-as-judge for subtle unsupported
inferences. Keep forbidden claims grounded in the reference context; do not
use them as arbitrary style constraints.

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
