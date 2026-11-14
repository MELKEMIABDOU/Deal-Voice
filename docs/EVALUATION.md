# Tests and evaluation

Run commands from the repository root. Software tests and real-model evaluation
are separate: passing tests does not demonstrate answer accuracy.

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
Citation checks validate filenames/locators, not whether they support each claim.
Unavailable checks are reported separately. Review answers manually, including
entity attribution, negation, missing evidence and false refusals.

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
