# Architecture and boundaries

DealVoice is currently a single-instance application. Its layers are Python
modules, not independently deployed microservices.

## Ingestion

PDF: Docling extracts Markdown in a child process. The indexer preserves
section boundaries and validated table blocks, then embeds child chunks while
storing their parent passages. Each table receives a dedicated title-and-header
child in addition to row-group children, so column-focused questions can recover
the complete table without row values diluting the embedding. PDF page-level
provenance is not preserved by the current Markdown-only export.

Audio: WhisperX transcribes, aligns and diarizes in a child process. The
application saves Markdown/JSON artifacts, then indexes parent passages made
from speaker turns, with overlap. Upload transcription defaults to
medium/fr/cpu/int8 and batch size 1, and can be configured through the
`WHISPER_*` environment settings. Benchmark CLI options do not change these
upload settings.

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
It combines semantic retrieval and Cross-Encoder ranking with a narrow exact
table-header promotion. It is not a general BM25/vector hybrid search.

The application shares one embedded Qdrant client and embedding instance.
Parent records are in `parent_docstore` under the configured database path.
Back up vectors, parents and uploaded/transcribed files together with the
application stopped. For multiple processes, redesign shared storage before
adding API workers. Changing embeddings requires reindexing even if the new
model has the same vector dimension.

## Generation and evidence

Ollama produces one bounded plan with up to six request parts. Each part answers,
asks for clarification or reports insufficient evidence. Parts may contain
several table operations and sourced claims. Pydantic validates the plan and
evidence IDs. The application renders citations from source metadata; the model
does not execute generated code or SQL.

Python performs table `select`, `min`, `max`, `sum`, `mean`, `count` and numeric
`sort`, numeric comparisons between columns, and row-wise differences between
two original columns. Calculated columns may be selected, filtered or aggregated.
Numeric parsing rejects ambiguous values and incompatible units.
`count` counts rows; arbitrary joins and distinct counts are unsupported.
`table_complete` refers only to the extracted Markdown block. It does not prove
that OCR recovered the entire PDF or that the retrieved table covers the query.

Prompt definitions are in `generation/prompts.py`, contracts in `schemas.py`,
and orchestration/rendering in `generator.py`. Whole evidence blocks are packed
using a conservative UTF-8 byte budget. Excluded blocks are disclosed in the
response. A context limit can therefore cause an incomplete answer or refusal.

The schema sent to Ollama keeps field types and status-specific structures, but
omits maximum string lengths: expanding those limits into a native grammar can
exceed llama.cpp's rule limit. Python still validates every original length and
business constraint before executing a plan; `num_predict` bounds model output.

All plans receive a second sequential call to the same model with a 192-token
verdict budget. It checks coverage of the original question, entity attribution,
the meaning of chosen columns and operations, and the support for advice. Only
supported parts are displayed as answers; unresolved or rejected parts are
explicit. A coverage failure labels the response as partial. There is no retry
loop or additional model in memory.

Recommendations are labelled as proposals and must specify a criterion,
references to claims or query results, and limits; assumptions are displayed
when supplied. The reviewer receives the referenced computed results. An exact
gap to a threshold is not an optimal order quantity or a forecast. Unsupported
formulas and joins require clarification or insufficient-evidence responses.

The semantic review remains fallible, especially when using the same model as
the planner. Mocked tests verify the contracts and rejection paths, not Qwen3's
actual ability to choose the right metric or cover every request. Real-model
evaluation is required; citations and schema validation are not semantic
correctness guarantees. Retrieval remains bounded and may miss relevant data.

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

## Recherche corrective bornée

La recherche applicative exécute une recherche initiale, puis évalue la
couverture des preuves avant la génération. Un tableau correspondant à la
mesure demandée doit être marqué comme extrait entièrement. Si ce contrôle
échoue, une seule recherche complémentaire est autorisée avec les mêmes
filtres de rôle et de documents. Les deux résultats sont ensuite fusionnés,
dédupliqués et limités au `top_k` demandé.

Il n’existe aucune boucle autonome. Les calculs restent exécutés en Python et
la validation finale du plan de réponse reste obligatoire. L’API expose
séparément `answer_status`, `retrieval_status`, `evidence_coverage` et
`corrective_search_used` afin de distinguer une réponse partielle d’une réponse
non vérifiée.
