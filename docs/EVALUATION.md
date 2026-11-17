# Tests and evaluation

Run commands from the repository root. Software tests and real-model evaluation
are separate: passing tests does not demonstrate answer accuracy.

See the [accuracy review and client acceptance cases](ACCURACY_REVIEW.md) for the
current limitations. The CLI generation evaluations call search and generation
directly; they do not exercise the API's corrective retrieval path. Test the
deployed `/api/v1/search` endpoint as well before accepting a client release.

## Client API acceptance (recommended before deployment)

`scripts/evaluate_client.py` exercises the same `/api/v1/search` route as the web
portal, including corrective retrieval, generation and final verification. It
does not load embeddings, Whisper or an LLM in the evaluation process. The API
must already be running and the documents indexed. The server still performs
real inference, so run one batch at a time on suitable hardware.

Before starting, verify `http://127.0.0.1:8000/api/v1/status`. The script performs
this check itself and exits clearly if the API is stopped or not ready; in that
case no RAG answer has been evaluated.

```powershell
uv run --no-sync python scripts/evaluate_client.py --questions data/evaluation/questions.jsonl --offset 0 --limit 5 --output reports/rag/client-01-05.json
```

Next batch: `--offset 5 --limit 5 --output reports/rag/client-06-10.json`, then
offsets 10, 15, etc. through 45. On the VM, with the application container already
running, use its Python environment. The runtime image does not include scripts
or evaluation data, so copy just the required files to its temporary directory:

```bash
dc() { docker compose --env-file .env.docker -f docker-compose.yml -f docker-compose.gpu.yml "$@"; }
dc cp scripts/evaluate_client.py app:/tmp/evaluate_client.py
dc cp data/evaluation/questions.jsonl app:/tmp/questions.jsonl
dc exec app python /tmp/evaluate_client.py --questions /tmp/questions.jsonl --offset 0 --limit 5 --output /tmp/client-01-05.json
mkdir -p reports/rag
dc cp app:/tmp/client-01-05.json reports/rag/client-01-05.json
```

The script defaults to `http://127.0.0.1:8000`. Use `--api-url` for another local
port or an SSH tunnel. It does not bypass the public proxy's authentication;
the container command accesses the internal API. `--document-id` (repeatable)
explicitly reproduces the portal's document selection; expected source names
are **never** used as hidden search filters. `--role` is only needed if the
existing API has RBAC enabled.

Each completed response is saved atomically with its retrieved passages,
statuses, verified numeric values, citations, latency and failed checks. A
timeout or HTTP error stops the run without retrying: the API may still be
processing the request. Check the server before resuming. Existing report names
are refused to preserve earlier evidence. Copy the report out of the container
even if the test reports failed checks (exit code 1); `/tmp/` is not persistent.

`checks_passed` is the existing lexical/reference evaluation. `accepted` also
requires the expected API status: `complete` for answer questions or
`insufficient` for refusal questions. A correct but explicitly limited table
result can therefore have `checks_passed=true` and `accepted=false`. An unknown
entity/value relationship, missing citation detail, validation error or context
capacity failure cannot count as acceptance. Requirements are not removed just
because retrieval failed to provide them. Interrupted runs have no overall rate.

These automated checks do **not** prove semantic accuracy, correct ranking or
absence of hallucinations. Review each answer against the source; test ambiguous
requests, paraphrases, contradictions and new documents manually too. Record the
commit, model/configuration and indexed document versions with the report.

## Default tests (no real model inference)

```powershell
uv sync --locked --extra all --group dev
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
uv run --no-sync ruff check src scripts tests streamlit_app.py
uv run --no-sync pytest -q
```

Default discovery is limited to `tests/unit` and
`tests/integration/test_document_lifecycle.py`. The latter uses deterministic
embeddings and a temporary database, including a backup/restore check.
Some ML libraries are imported, but no real models should be downloaded or run.
Streamlit remains installed in this development setup because lifecycle tests
still exercise its upload adapter.

## Explicit real-model tests

These commands can load large models. Run them separately on suitable hardware,
with no competing ingestion or generation workload.

```powershell
uv run --no-sync pytest tests/integration/test_retrieval.py -q
uv run --no-sync pytest tests/integration/test_real_ingestion.py::test_real_pdf_to_parent_child_chunks -q
```

The real ingestion tests currently read fixtures from `data/raw/`: the Horizon
PDF and the call named in `tests/integration/test_real_ingestion.py`. Those files
are local, not guaranteed by a fresh clone. Provision them before these tests.
The audio test also requires `RUN_REAL_AUDIO_TESTS=1` and `HF_TOKEN`.

DeepEval is optional:

```powershell
uv sync --locked --extra all --group dev --group evaluation
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
uv run --no-sync pytest tests/evaluation/test_rag_quality.py -q
```

This historical judge test uses `llama3` and its own small dataset; it is not
the current Qwen3/50-question acceptance run. Install its model before running.
Judge scores are fallible and require human review.

## Reference dataset

`data/evaluation/questions.jsonl` contains 50 French questions. See the dataset
[notice](../data/evaluation/README.txt) and
[verification notes](../data/evaluation/verification.json).

Canonical transcript filenames match their audio stems:

```text
data/evaluation/
  B2B Appel.mp3          # local input, ignored by Git
  B2B Appel.txt          # reference
  Reunion dequipe.mp3    # local input, ignored by Git
  Reunion dequipe.txt    # reference
  questions.jsonl
```

The older `questions.json`, negative, robustness and multihop JSON sets are
retained as historical fixtures. They are not interchangeable with the 50-question
set. Audio questions in that set check transcript facts, not ASR accuracy.

## Retrieval and generation with limited memory

Index the required documents through the portal first. Stop the API/Streamlit
before running another owner of the local Qdrant directory. Unload competing
Ollama models before preparing contexts.

Prepare only five questions, then let that process exit:

```powershell
uv run --no-sync python scripts/evaluate_generation.py --questions data/evaluation/questions.jsonl --top-k 3 --offset 0 --limit 5 --no-reranker --prepare-contexts reports/rag/contexts-01-05.json
```

Generate in a separate process with Ollama running:

```powershell
uv run --no-sync python scripts/evaluate_generation.py --questions data/evaluation/questions.jsonl --from-contexts reports/rag/contexts-01-05.json --model qwen3:8b --offset 0 --limit 5 --output reports/rag/generation-01-05.json
```

For the next batch, use offset 5 in both commands and filenames `06-10`, then
10, 15, through 45. Use smaller batches if needed. `--no-reranker` changes the
retrieval configuration; keep it identical when comparing runs and validate
the intended deployed configuration separately.

Recreate contexts after changing the dataset, index or retrieval configuration.
The transcript filename cleanup changes the dataset fingerprint: contexts
prepared against the previous JSONL must be regenerated.

For retrieval alone:

```powershell
uv run --no-sync python scripts/evaluate_rag.py --questions data/evaluation/questions.jsonl --top-k 3 --limit 5 --no-reranker --output reports/rag/retrieval-5.json
```

Fact recall is lexical presence with explicit variants, not semantic correctness.
Known ASR spelling variants for names must be declared per question with
`expected_fact_variants`; the evaluator does not apply global fuzzy matching that
could confuse two different people.
Citation checks validate filenames/locators, not whether they support each claim.
Unavailable checks are reported separately. Review answers manually, including
entity attribution, negation, missing evidence and false refusals.

### Multipart questions and recommendations

The generator now plans separate requests and records `part_statuses`, the
proposed plan, computed results and the semantic review in its trace. A partial
answer is not a complete pass. Add checks for the requested metric and coverage
of **every** request when reviewing the JSON reports.

The deterministic tests in `tests/unit/test_reasoning.py` use fake model outputs.
They cover two table operations, a known fact alongside an unresolved request,
rejection of a wrong metric, and recommendations grounded in computed results.
They do not measure Qwen3 accuracy. Test the actual configured model separately,
one question at a time, using the current index or prepared contexts.

Useful manual checks on the Horizon PDF include:

- Ask for the highest retail price **and** products below their stock thresholds.
  Check that both requests are covered and that retail price is not replaced by margin.
- Ask which products should be bought more. Expect an explicit criterion or a
  clarification, not an unqualified optimal order quantity.
- Request the difference between stock threshold and observed stock. Verify the
  arithmetic and whether the table describes current or historical inventory.

On the meeting audio, ask for both the topic and participant identities. Unknown
names must remain unknown; speaker labels alone are not verified identities.
Run the same checks with unrelated schemas and different values. The existing
50-question dataset is useful but does not by itself validate business advice.

## Transcription benchmark

Use manually validated `.txt` references with the same stems as the audios:

```powershell
uv run --no-sync python scripts/evaluate_transcription_dataset.py --evaluation-dir data/evaluation --models medium --language fr --device cpu --compute-type int8 --batch-size 1 --output-dir reports/transcription/evaluation-dataset
```

This benchmark measures ASR without speaker diarization. It is not the total
upload/indexing time. Model-loading time is separate from timed transcription.
WER/CER are normalized edit distances; lower is better. Review names, dates,
amounts and negation separately. Wav2Vec2 remains available via
`scripts/evaluate_wav2vec2_french.py` as an optional experiment.

### Historical two-file reference

Copied from the previously tracked `reports/transcription/evaluation-dataset/benchmark.md`
and its JSON settings (French, CPU, int8). Not rerun during repository cleanup.

| Model | Files | Total transcription time (s) | Weighted WER | Weighted CER |
| --- | ---: | ---: | ---: | ---: |
| base | 2 | 20.1 | 17.92% | 8.09% |
| small | 2 | 49.1 | 9.83% | 5.39% |
| medium | 2 | 129.3 | 7.37% | 3.75% |
| large-v3 | 2 | 212.8 | 4.91% | 3.39% |

The old report does not record all environment/model revisions, so these values
are indicative historical observations, not a current production guarantee.
Generated transcripts, checkpoints and resource logs remain local in `reports/`.
Older committed reports can still be recovered from Git history.
