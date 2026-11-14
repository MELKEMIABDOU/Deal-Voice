# Architecture and boundaries

DealVoice is currently a single-instance application. Its layers are Python
modules, not independently deployed microservices.

## Ingestion

PDF: Docling extracts Markdown in a child process. The indexer preserves
section boundaries and validated table blocks, then embeds child chunks while
storing their parent passages. PDF page-level provenance is not preserved by
the current Markdown-only export.

Audio: WhisperX transcribes, aligns and diarizes in a child process. The
application saves Markdown/JSON artifacts, then indexes parent passages made
from speaker turns, with overlap. The upload path uses medium/fr/cpu/int8 and
batch size 1. Benchmark CLI options do not change these upload defaults.

The shared worker transport limits extraction concurrency and execution time.
Heavy model allocations belong to the child. On Windows a Job Object contains
workers and descendants. On other platforms the direct child is stopped on
timeout; deployment supervision must also contain descendant processes.

The audio duration check runs on the decoded 16 kHz waveform before loading
the ASR model. This does not bound the memory needed for decoding itself.

Uploads receive UUIDs and content hashes. An exact binary duplicate replaces
an older indexed copy after new indexing succeeds. Different files with the
same name remain distinct. Exception-time rollback is best effort, not a
durable transaction across the filesystem and Qdrant.

## Retrieval and storage

Search uses BGE-M3 child vectors, loads parent passages, checks requested
filters again, deduplicates and optionally reranks a bounded candidate set.
It is semantic retrieval, not hybrid lexical/vector search.

The application shares one embedded Qdrant client and embedding instance.
Parent records are in `parent_docstore` under the configured database path.
Back up vectors, parents and uploaded/transcribed files together with the
application stopped. For multiple processes, redesign shared storage before
adding API workers. Changing embeddings requires reindexing even if the new
model has the same vector dimension.

## Generation and evidence

Ollama produces a narrative, table-only or mixed plan. Pydantic validates the
plan and evidence IDs. The application renders citations from source metadata;
the model does not execute generated code or SQL.

Python performs table `select`, `min`, `max`, `sum`, `mean`, `count` and numeric
`sort`. Numeric parsing rejects ambiguous values and incompatible units.
`count` counts rows; arbitrary joins and distinct counts are unsupported.
`table_complete` refers only to the extracted Markdown block. It does not prove
that OCR recovered the entire PDF or that the retrieved table covers the query.

Prompt definitions are in `generation/prompts.py`, contracts in `schemas.py`,
and orchestration/rendering in `generator.py`. Whole evidence blocks are packed
using a conservative UTF-8 byte budget. Excluded blocks are disclosed in the
response. A context limit can therefore cause an incomplete answer or refusal.

Narrative claims receive a second sequential call to the same model with a
64-token verdict budget. This check can reject a draft but can also approve
an incorrect claim. Known entity-attribution errors remain; citations and
schema validation are not semantic correctness guarantees.

## Operational boundaries

- The API has no built-in user authentication or tenant isolation.
- Optional role filtering trusts a development header, not verified identity.
- One global operation lock serializes import, search and deletion; competing
  operations receive HTTP 409.
- Import jobs are held in memory, with no durable queue or automatic recovery.
- The health endpoint confirms HTTP availability, not dependency readiness.
- Memory observations in the evaluation tools are Windows-specific and do not
  measure VRAM or enforce a memory quota.

See [client operation](CLIENT_WEB.md) and [evaluation](EVALUATION.md).
