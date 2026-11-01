r"""Evaluate generated RAG answers for the reference question set.

The evaluation documents must already be indexed. Ollama is needed only for
generation. On memory-constrained machines, use --prepare-contexts in one
process, then --from-contexts in a second process to avoid keeping embeddings
and the generation model loaded together.

Example:
    uv run python scripts\evaluate_generation.py `
      --questions data\evaluation\questions.jsonl `
      --model qwen3:8b
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from langchain_core.documents import Document

from deal_voice.evaluation.generation import evaluate_generation
from deal_voice.evaluation.question_set import (
    load_question_set,
    normalize_text,
    resolve_indexed_sources,
    source_matches,
)
from deal_voice.evaluation.resources import MemoryMonitor


def percentile(values: list[float], percentile_value: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = round((percentile_value / 100) * (len(ordered) - 1))
    return ordered[index]


def citation_details_available(
    citations: list[dict[str, object]], documents: list[object]
) -> tuple[list[dict[str, object]], list[str]]:
    """Exclude citation details that the retrieved context cannot support."""
    evaluable: list[dict[str, object]] = []
    not_evaluable: list[str] = []
    for citation in citations:
        source = citation.get("file")
        citation_documents = [
            document
            for document in documents
            if not source
            or source_matches(
                str(source),
                str((getattr(document, "metadata", {}) or {}).get("source", "")),
            )
        ]
        context_parts: list[str] = []
        has_page_metadata = False
        for document in citation_documents:
            metadata = getattr(document, "metadata", {}) or {}
            context_parts.append(str(getattr(document, "page_content", "")))
            context_parts.extend(
                str(metadata[key])
                for key in ("section", "table_title")
                if metadata.get(key)
            )
            has_page_metadata = has_page_metadata or any(
                metadata.get(key) is not None
                for key in ("page", "page_number", "page_no")
            )
        normalized_context = normalize_text(" ".join(context_parts))
        retained = dict(citation)
        for key in ("page", "section", "table"):
            value = citation.get(key)
            if value is None:
                continue
            normalized_value = normalize_text(str(value))
            detail_available = (
                has_page_metadata
                if key == "page"
                else bool(normalized_value)
                and f" {normalized_value} " in f" {normalized_context} "
            )
            if not detail_available:
                label = f"page {value}" if key == "page" else str(value)
                not_evaluable.append(label)
                retained.pop(key, None)
        evaluable.append(retained)
    return evaluable, not_evaluable


def write_checkpoint(
    output_path: Path,
    model: str,
    top_k: int,
    expected_count: int,
    question_set: str,
    reranker_enabled: bool,
    evaluations: list[dict[str, object]],
) -> Path:
    """Atomically persist completed question rows during long evaluations."""
    checkpoint_path = output_path.with_name(
        f"{output_path.stem}.checkpoint{output_path.suffix}"
    )
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    payload = {
        "model": model,
        "top_k": top_k,
        "question_set": question_set,
        "reranker_enabled": reranker_enabled,
        "question_count": expected_count,
        "completed_question_count": len(evaluations),
        "checkpoint": True,
        "results": evaluations,
    }
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(checkpoint_path)
    return checkpoint_path


def prepare_contexts(
    questions: list[dict], args: argparse.Namespace, monitor: MemoryMonitor
) -> None:
    """Save actual retrieval results without creating a generation model."""
    from deal_voice.retrieval.search import LocalSearchEngine

    monitor.mark("loading_retrieval")
    search_engine = LocalSearchEngine(use_reranker=not args.no_reranker)
    try:
        indexed_sources = [
            str(document["source"])
            for document in search_engine.list_documents()
            if document.get("source")
        ]
        results = []
        for item in questions:
            monitor.mark("retrieval", item["id"])
            started = time.perf_counter()
            documents = search_engine.search(
                query=item["question"], source=None, top_k=args.top_k
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            results.append(
                {
                    "question_id": item["id"],
                    "question": item["question"],
                    "documents": [
                        {
                            "page_content": document.page_content,
                            "metadata": document.metadata,
                        }
                        for document in documents
                    ],
                    "retrieval_latency_ms": round(elapsed_ms, 2),
                }
            )
            print(
                f"[{len(results)}/{len(questions)}] {item['id']} "
                f"prepared={len(documents)}",
                flush=True,
            )
    finally:
        search_engine.client.close()

    payload = {
        "schema_version": 1,
        "question_set": str(args.questions),
        "question_set_sha256": hashlib.sha256(args.questions.read_bytes()).hexdigest(),
        "top_k": args.top_k,
        "reranker_enabled": not args.no_reranker,
        "indexed_sources": indexed_sources,
        "results": results,
    }
    output = args.prepare_contexts
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(output)
    monitor.mark("retrieval_finished")
    print(f"Contexts written to {output}. This process can now exit.")


def load_prepared_contexts(
    path: Path, question_set: Path, questions: list[dict]
) -> dict:
    """Reject stale or incomplete retrieval snapshots before loading Ollama."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Unsupported prepared-context format.")
    expected_hash = hashlib.sha256(question_set.read_bytes()).hexdigest()
    if payload.get("question_set_sha256") != expected_hash:
        raise ValueError("The question set changed; prepare the contexts again.")
    if (
        type(payload.get("top_k")) is not int
        or payload["top_k"] < 1
        or not isinstance(payload.get("reranker_enabled"), bool)
        or not isinstance(payload.get("indexed_sources"), list)
        or any(not isinstance(source, str) for source in payload["indexed_sources"])
    ):
        raise ValueError("Invalid prepared-context retrieval settings.")
    entries = payload.get("results")
    if not isinstance(entries, list) or not entries:
        raise ValueError("The prepared-context file contains no results.")
    by_id = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("question_id"), str):
            raise ValueError("Invalid prepared question entry.")
        question_id = entry["question_id"]
        if question_id in by_id:
            raise ValueError(f"Duplicate prepared question: {question_id}")
        documents = entry.get("documents")
        latency = entry.get("retrieval_latency_ms")
        if (
            not isinstance(documents, list)
            or any(
                not isinstance(document, dict)
                or not isinstance(document.get("page_content"), str)
                or not isinstance(document.get("metadata"), dict)
                for document in documents
            )
            or not isinstance(latency, (float, int))
            or isinstance(latency, bool)
            or latency < 0
        ):
            raise ValueError(f"Invalid prepared documents or latency: {question_id}")
        by_id[question_id] = entry
    for item in questions:
        entry = by_id.get(item["id"])
        if entry is None or entry.get("question") != item["question"]:
            raise ValueError(f"Missing or changed prepared question: {item['id']}")
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate final LLM answers on reference questions."
    )
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--top-k", type=int, default=None)
    context_mode = parser.add_mutually_exclusive_group()
    context_mode.add_argument(
        "--prepare-contexts",
        type=Path,
        help="Retrieve and save contexts, then exit without calling Ollama.",
    )
    context_mode.add_argument(
        "--from-contexts",
        type=Path,
        help="Generate using saved contexts without loading embeddings or Qdrant.",
    )
    parser.add_argument(
        "--memory-log",
        type=Path,
        help="JSONL resource log; defaults to <output stem>.memory.jsonl.",
    )
    parser.add_argument(
        "--category",
        action="append",
        help="Evaluate only this category; repeat to select multiple categories.",
    )
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="Disable the cross-encoder reranker to reduce CPU/RAM usage.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip the first N selected questions before applying --limit.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N questions.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/rag/generation-evaluation.json"),
    )
    args = parser.parse_args(argv)
    log_base = args.prepare_contexts or args.output
    memory_path = args.memory_log or log_base.with_suffix(".memory.jsonl")
    with MemoryMonitor(memory_path) as monitor:
        run_evaluation(args, parser, monitor)


def run_evaluation(
    args: argparse.Namespace, parser: argparse.ArgumentParser, monitor: MemoryMonitor
) -> None:

    questions = load_question_set(args.questions)
    if args.category:
        selected_categories = set(args.category)
        questions = [
            item for item in questions if item.get("category") in selected_categories
        ]
    if args.offset < 0:
        parser.error("--offset must be at least 0")
    questions = questions[args.offset :]
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be at least 1")
        questions = questions[: args.limit]
    if not questions:
        parser.error("No questions match the selected category/offset/limit.")
    if args.top_k is not None and args.top_k < 1:
        parser.error("--top-k must be at least 1")

    prepared = None
    if args.from_contexts:
        try:
            prepared = load_prepared_contexts(
                args.from_contexts, args.questions, questions
            )
        except (OSError, ValueError) as error:
            parser.error(str(error))
        if args.top_k is not None and args.top_k != prepared["top_k"]:
            parser.error("--top-k differs from the saved contexts; prepare them again.")
        if args.no_reranker and prepared["reranker_enabled"]:
            parser.error("The saved contexts used a reranker; prepare them again.")
        args.top_k = prepared["top_k"]
        args.no_reranker = not prepared["reranker_enabled"]
    elif args.top_k is None:
        args.top_k = 5
    if args.prepare_contexts:
        prepare_contexts(questions, args, monitor)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    search_engine = None
    evaluations = []
    try:
        if prepared is not None:
            prepared_by_id = {
                entry["question_id"]: entry for entry in prepared["results"]
            }
            indexed_sources = prepared["indexed_sources"]
        else:
            from deal_voice.retrieval.search import LocalSearchEngine

            monitor.mark("loading_retrieval")
            search_engine = LocalSearchEngine(use_reranker=not args.no_reranker)
            indexed_sources = [
                str(document["source"])
                for document in search_engine.list_documents()
                if document.get("source")
            ]
        from deal_voice.generation.generator import RAGGenerator

        monitor.mark("initializing_generator")
        generator = RAGGenerator(model_name=args.model)
        expected_sources_in_set = {
            str(citation["file"])
            for item in questions
            for citation in item["expected_citations"]
            if citation.get("file")
        } or {source for item in questions for source in item["sources"]}
        _, missing_sources = resolve_indexed_sources(
            sorted(expected_sources_in_set), indexed_sources
        )
        if missing_sources:
            print(
                "Warning: expected source files are not indexed: "
                + ", ".join(missing_sources),
                flush=True,
            )

        for item in questions:
            expected_citation_sources = [
                str(citation["file"])
                for citation in item["expected_citations"]
                if citation.get("file")
            ]
            reference_sources = expected_citation_sources or item["sources"]
            resolved_sources, unresolved_sources = resolve_indexed_sources(
                reference_sources, indexed_sources
            )
            monitor.mark("retrieval", item["id"])
            retrieval_started = time.perf_counter()
            if prepared is not None:
                entry = prepared_by_id[item["id"]]
                documents = [Document(**document) for document in entry["documents"]]
                retrieval_latency_ms = entry["retrieval_latency_ms"]
            else:
                documents = search_engine.search(
                    query=item["question"],
                    # Never filter by the reference answer's source.
                    source=None,
                    top_k=args.top_k,
                )
                retrieval_latency_ms = (time.perf_counter() - retrieval_started) * 1000
            retrieved_sources = sorted(
                {
                    str(document.metadata["source"])
                    for document in documents
                    if document.metadata.get("source")
                }
            )
            retrieval_source_hit = (
                all(
                    any(
                        source_matches(expected, actual) for actual in retrieved_sources
                    )
                    for expected in reference_sources
                )
                if reference_sources
                else None
            )
            monitor.mark("generation", item["id"])
            generation_started = time.perf_counter()
            answer = generator.generate(item["question"], documents)
            generation_trace = getattr(generator, "last_trace", {}) or {}
            generation_latency_ms = (time.perf_counter() - generation_started) * 1000
            monitor.mark("generation_finished", item["id"])
            total_latency_ms = retrieval_latency_ms + generation_latency_ms
            evaluable_citations, citation_details_not_evaluable = (
                citation_details_available(item["expected_citations"], documents)
            )
            evaluation = evaluate_generation(
                item["id"],
                item["question"],
                expected_citation_sources or item["sources"],
                item["expected_facts"],
                answer,
                item.get("expected_fact_variants"),
                item.get("expected_behavior", "answer"),
                item.get("forbidden_claims"),
                item["expected_values"],
                evaluable_citations,
                expected_language=item.get("language"),
                actual_values=generation_trace.get("actual_values"),
                available_citations=generation_trace.get("structured_citations"),
            )
            normalized_behavior = "refusal" if evaluation.refusal_expected else "answer"
            expected_source_index_recall = (
                len(resolved_sources) / len(reference_sources)
                if reference_sources
                else None
            )
            forbidden_claim_checkable = bool(item.get("forbidden_claims"))
            evaluations.append(
                {
                    "question_id": evaluation.question_id,
                    "category": item.get("category", "unspecified"),
                    "question": item["question"],
                    "answer": evaluation.answer,
                    "generation_status": generation_trace.get("status", "legacy_text"),
                    "generation_trace": generation_trace,
                    "computations": generation_trace.get("computations", []),
                    "expected_source": evaluation.source_expected,
                    "expected_sources": reference_sources,
                    "resolved_sources": resolved_sources,
                    "unresolved_sources": unresolved_sources,
                    "expected_source_index_recall": expected_source_index_recall,
                    "expected_sources_available": (
                        not unresolved_sources if reference_sources else None
                    ),
                    "retrieved_document_count": len(documents),
                    "retrieved_sources": retrieved_sources,
                    "all_expected_sources_retrieved": retrieval_source_hit,
                    "citation_source_present": evaluation.source_present,
                    "facts_found": evaluation.facts_found,
                    "facts_missing": evaluation.facts_missing,
                    "fact_recall": evaluation.fact_recall,
                    "expected_values_found": evaluation.expected_values_found,
                    "expected_values_missing": evaluation.expected_values_missing,
                    "value_recall": evaluation.value_recall,
                    "value_relations_found": evaluation.value_relations_found,
                    "value_relations_missing": evaluation.value_relations_missing,
                    "value_relations_not_evaluable": evaluation.value_relations_not_evaluable,
                    "value_relation_recall": evaluation.value_relation_recall,
                    "citation_files_found": evaluation.citation_files_found,
                    "citation_files_missing": evaluation.citation_files_missing,
                    "citation_file_recall": evaluation.citation_file_recall,
                    "citation_details_found": evaluation.citation_details_found,
                    "citation_details_missing": evaluation.citation_details_missing,
                    "invalid_citation_details": evaluation.invalid_citation_details,
                    "citation_format_warnings": evaluation.citation_format_warnings,
                    "citation_details_not_evaluable": citation_details_not_evaluable,
                    "citation_detail_recall": evaluation.citation_detail_recall,
                    "lexical_facts_found": evaluation.lexical_facts_found,
                    "lexical_facts_missing": evaluation.lexical_facts_missing,
                    "lexical_fact_recall": evaluation.lexical_fact_recall,
                    "answer_non_empty": evaluation.answer_non_empty,
                    "heuristic_french": evaluation.likely_french,
                    "language_compliant": evaluation.language_compliant,
                    "expected_behavior": normalized_behavior,
                    "refusal_detected": evaluation.refusal_detected,
                    "forbidden_claims": item.get("forbidden_claims", []),
                    "forbidden_claims_found": evaluation.forbidden_claims_found,
                    "forbidden_claim_checkable": forbidden_claim_checkable,
                    "forbidden_claim_compliant": (
                        not evaluation.forbidden_claims_found
                        if forbidden_claim_checkable
                        else None
                    ),
                    "retrieval_latency_ms": round(retrieval_latency_ms, 2),
                    "generation_latency_ms": round(generation_latency_ms, 2),
                    "total_latency_ms": round(total_latency_ms, 2),
                    "passed": evaluation.passed,
                }
            )
            checkpoint_path = write_checkpoint(
                args.output,
                args.model,
                args.top_k,
                len(questions),
                str(args.questions),
                not args.no_reranker,
                evaluations,
            )
            print(
                f"[{len(evaluations)}/{len(questions)}] {item['id']} "
                f"retrieved={len(documents)} all_sources_hit={retrieval_source_hit} "
                f"generation={generation_latency_ms / 1000:.1f}s; "
                f"checkpoint={checkpoint_path}",
                flush=True,
            )
    finally:
        if search_engine is not None:
            search_engine.client.close()

    answer_evaluations = [
        item for item in evaluations if item["expected_behavior"] == "answer"
    ]
    expected_source_evaluations = [
        item for item in answer_evaluations if item["expected_sources"]
    ]
    citation_detail_evaluations = [
        item
        for item in evaluations
        if item["citation_details_found"] or item["citation_details_missing"]
    ]
    forbidden_claim_evaluations = [
        item for item in evaluations if item["forbidden_claim_checkable"]
    ]
    value_relation_evaluations = [
        item for item in evaluations if item["value_relation_recall"] is not None
    ]
    report = {
        "evaluation_schema_version": 2,
        "model": args.model,
        "question_set": str(args.questions),
        "reranker_enabled": not args.no_reranker,
        "top_k": args.top_k,
        "offset": args.offset,
        "from_contexts": str(args.from_contexts) if args.from_contexts else None,
        "memory_log": str(monitor.path),
        "question_count": len(evaluations),
        "completed_question_count": len(evaluations),
        "partial": False,
        "answer_question_count": len(answer_evaluations),
        "expected_source_index_coverage": (
            sum(
                item["expected_source_index_recall"]
                for item in expected_source_evaluations
            )
            / len(expected_source_evaluations)
            if expected_source_evaluations
            else None
        ),
        "all_expected_sources_retrieval_rate": (
            sum(
                bool(item["all_expected_sources_retrieved"])
                for item in expected_source_evaluations
            )
            / len(expected_source_evaluations)
            if expected_source_evaluations
            else None
        ),
        "answer_citation_file_recall": (
            sum(item["citation_file_recall"] for item in answer_evaluations)
            / len(answer_evaluations)
            if answer_evaluations
            else None
        ),
        "mean_fact_recall": (
            sum(item["fact_recall"] for item in answer_evaluations)
            / len(answer_evaluations)
            if answer_evaluations
            else None
        ),
        "heuristic_french_rate": sum(item["heuristic_french"] for item in evaluations)
        / len(evaluations),
        "language_compliance_rate": sum(
            item["language_compliant"] for item in evaluations
        )
        / len(evaluations),
        "refusal_accuracy": (
            sum(
                item["passed"]
                for item in evaluations
                if item["expected_behavior"] == "refusal"
            )
            / sum(item["expected_behavior"] == "refusal" for item in evaluations)
            if any(item["expected_behavior"] == "refusal" for item in evaluations)
            else None
        ),
        "forbidden_claim_compliance_rate": (
            sum(
                bool(item["forbidden_claim_compliant"])
                for item in forbidden_claim_evaluations
            )
            / len(forbidden_claim_evaluations)
            if forbidden_claim_evaluations
            else None
        ),
        "mean_value_recall": (
            sum(
                item["value_recall"]
                for item in evaluations
                if item["expected_values_found"] or item["expected_values_missing"]
            )
            / sum(
                bool(item["expected_values_found"] or item["expected_values_missing"])
                for item in evaluations
            )
            if any(
                item["expected_values_found"] or item["expected_values_missing"]
                for item in evaluations
            )
            else None
        ),
        "mean_value_relation_recall": (
            sum(item["value_relation_recall"] for item in value_relation_evaluations)
            / len(value_relation_evaluations)
            if value_relation_evaluations
            else None
        ),
        "value_relation_evaluated_question_count": len(value_relation_evaluations),
        "value_relations_not_evaluable_count": sum(
            len(item["value_relations_not_evaluable"]) for item in evaluations
        ),
        "generation_status_counts": {
            status: sum(item["generation_status"] == status for item in evaluations)
            for status in sorted({item["generation_status"] for item in evaluations})
        },
        "mean_citation_detail_recall": (
            sum(item["citation_detail_recall"] for item in citation_detail_evaluations)
            / len(citation_detail_evaluations)
            if citation_detail_evaluations
            else None
        ),
        "citation_details_not_evaluable_count": sum(
            len(item["citation_details_not_evaluable"]) for item in evaluations
        ),
        "invalid_citation_detail_count": sum(
            len(item["invalid_citation_details"]) for item in evaluations
        ),
        "citation_format_warning_count": sum(
            len(item["citation_format_warnings"]) for item in evaluations
        ),
        "pass_rate": sum(item["passed"] for item in evaluations) / len(evaluations),
        "metric_notes": {
            "pass_rate": "Share of questions meeting deterministic criteria; it does not judge subtle unsupported claims semantically.",
            "mean_fact_recall": "Lexical presence after normalization and explicit expected_fact_variants only. This is not a semantic correctness or faithfulness score.",
            "mean_value_recall": "Lexical presence of expected value strings; it does not verify which entity or field a value belongs to.",
            "mean_value_relation_recall": "Entity/field/value triples compared only when the generator exposes structured actual_values. Missing structured output is not counted as a verified relation.",
            "citation_format_warning_count": "Correct evidence titles labeled as section instead of table (or vice versa) are reported separately from unsupported reference details.",
            "heuristic_french_rate": "Simple lexical estimate; when French is required it also contributes to pass/fail. Short or mixed-language answers need human review.",
            "latency_ms": "With saved contexts, retrieval latency comes from preparation; total is retrieval plus generation, not the replay's wall-clock time.",
            "all_expected_sources_retrieval_rate": "Share of answerable questions where every expected source appears in retrieved results; search is not filtered by reference sources.",
            "answer_citation_file_recall": "Expected source filename is explicitly cited in the generated answer.",
            "forbidden_claim_compliance_rate": "Checks only forbidden phrases explicitly listed in the question set; this is not a semantic faithfulness score.",
            "mean_citation_detail_recall": "Only page/section/table details present in retrieved metadata or text are scored; unavailable details are reported separately.",
        },
        "latency_ms": {
            "retrieval": {
                "mean": round(
                    sum(item["retrieval_latency_ms"] for item in evaluations)
                    / len(evaluations),
                    2,
                ),
                "p50": round(
                    percentile(
                        [item["retrieval_latency_ms"] for item in evaluations],
                        50,
                    ),
                    2,
                ),
                "p95": round(
                    percentile(
                        [item["retrieval_latency_ms"] for item in evaluations],
                        95,
                    ),
                    2,
                ),
            },
            "generation": {
                "mean": round(
                    sum(item["generation_latency_ms"] for item in evaluations)
                    / len(evaluations),
                    2,
                ),
                "p50": round(
                    percentile(
                        [item["generation_latency_ms"] for item in evaluations],
                        50,
                    ),
                    2,
                ),
                "p95": round(
                    percentile(
                        [item["generation_latency_ms"] for item in evaluations],
                        95,
                    ),
                    2,
                ),
            },
            "total": {
                "mean": round(
                    sum(item["total_latency_ms"] for item in evaluations)
                    / len(evaluations),
                    2,
                ),
                "p50": round(
                    percentile(
                        [item["total_latency_ms"] for item in evaluations],
                        50,
                    ),
                    2,
                ),
                "p95": round(
                    percentile(
                        [item["total_latency_ms"] for item in evaluations],
                        95,
                    ),
                    2,
                ),
            },
        },
        "results": evaluations,
    }
    report["by_category"] = {
        category: {
            "question_count": sum(item["category"] == category for item in evaluations),
            "pass_rate": (
                sum(
                    item["passed"]
                    for item in evaluations
                    if item["category"] == category
                )
                / sum(item["category"] == category for item in evaluations)
            ),
            "mean_fact_recall": (
                sum(
                    item["fact_recall"]
                    for item in evaluations
                    if item["category"] == category
                    and item["expected_behavior"] == "answer"
                )
                / sum(
                    item["category"] == category
                    and item["expected_behavior"] == "answer"
                    for item in evaluations
                )
                if any(
                    item["category"] == category
                    and item["expected_behavior"] == "answer"
                    for item in evaluations
                )
                else None
            ),
        }
        for category in sorted({item["category"] for item in evaluations})
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Expected source index coverage: {report['expected_source_index_coverage']}")
    print(
        f"All expected sources retrieved: {report['all_expected_sources_retrieval_rate']}"
    )
    print(f"Answer citation file recall: {report['answer_citation_file_recall']}")
    mean_fact_recall = report["mean_fact_recall"]
    if mean_fact_recall is None:
        print("Mean fact recall: N/A (no answerable questions)")
    else:
        print(f"Mean fact recall: {mean_fact_recall:.2%}")
    print(f"Heuristic French rate: {report['heuristic_french_rate']:.2%}")
    print(f"Pass rate: {report['pass_rate']:.2%}")
    print(f"Mean value recall: {report['mean_value_recall']}")
    print(
        f"Mean structured value relation recall: {report['mean_value_relation_recall']}"
    )
    print(
        "Value relations not evaluable: "
        f"{report['value_relations_not_evaluable_count']}"
    )
    print(f"Mean citation detail recall: {report['mean_citation_detail_recall']}")
    print(
        "Forbidden-claim compliance (listed phrases only): "
        f"{report['forbidden_claim_compliance_rate']}"
    )
    print(f"Invalid citation details: {report['invalid_citation_detail_count']}")
    print(f"Citation format warnings: {report['citation_format_warning_count']}")
    print("Pass rate by category:")
    for category, metrics in report["by_category"].items():
        print(
            f"  {category}: {metrics['pass_rate']:.2%} ({metrics['question_count']} questions)"
        )
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
