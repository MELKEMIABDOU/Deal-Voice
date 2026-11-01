r"""Evaluate retrieval against the reference question set.

The indexed evaluation documents must already exist in Qdrant.

Example:
    uv run python scripts\evaluate_rag.py `
      --questions data\evaluation\questions.jsonl `
      --top-k 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from deal_voice.evaluation.question_set import load_question_set
from deal_voice.evaluation.rag import evaluate_retrieval
from deal_voice.retrieval.search import LocalSearchEngine


def percentile(values: list[float], percentile_value: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = round((percentile_value / 100) * (len(ordered) - 1))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Deal-Voice retrieval.")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--category",
        action="append",
        help="Evaluate only this category; repeat to select multiple categories.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N questions after category filtering.",
    )
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="Disable the cross-encoder reranker to reduce CPU/RAM usage.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/rag/retrieval-evaluation.json"),
    )
    args = parser.parse_args()
    questions = load_question_set(args.questions)
    if args.category:
        selected_categories = set(args.category)
        questions = [
            item for item in questions if item.get("category") in selected_categories
        ]
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be at least 1")
        questions = questions[: args.limit]
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if not questions:
        parser.error("No questions match the selected category/limit.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    search_engine = LocalSearchEngine(use_reranker=not args.no_reranker)
    evaluations = []
    try:
        for item in questions:
            started = time.perf_counter()
            documents = search_engine.search(
                query=item["question"],
                source=None,
                top_k=args.top_k,
            )
            retrieval_latency_ms = (time.perf_counter() - started) * 1000
            evaluation = evaluate_retrieval(
                item["id"],
                item["sources"],
                item["expected_facts"],
                documents,
                item.get("expected_fact_variants"),
            )
            evaluations.append(
                {
                    "question_id": evaluation.question_id,
                    "expected_behavior": item["expected_behavior"],
                    "retrieved_sources": evaluation.retrieved_sources,
                    "source_hit": evaluation.source_hit,
                    "expected_sources_found": evaluation.expected_sources_found,
                    "expected_sources_missing": evaluation.expected_sources_missing,
                    "source_recall": evaluation.source_recall,
                    "expected_facts_found": evaluation.expected_facts_found,
                    "expected_facts_missing": evaluation.expected_facts_missing,
                    "fact_recall": evaluation.fact_recall,
                    "expected_source_rank": evaluation.expected_source_rank,
                    "reciprocal_rank": evaluation.reciprocal_rank,
                    "binary_rank_score": evaluation.binary_rank_score,
                    "precision_at_k": evaluation.precision_at_k,
                    "retrieval_latency_ms": round(retrieval_latency_ms, 2),
                }
            )
            checkpoint_path = args.output.with_name(
                f"{args.output.stem}.checkpoint{args.output.suffix}"
            )
            temporary_path = checkpoint_path.with_suffix(
                checkpoint_path.suffix + ".tmp"
            )
            temporary_path.write_text(
                json.dumps(
                    {
                        "top_k": args.top_k,
                        "question_set": str(args.questions),
                        "reranker_enabled": not args.no_reranker,
                        "question_count": len(questions),
                        "completed_question_count": len(evaluations),
                        "checkpoint": True,
                        "results": evaluations,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            temporary_path.replace(checkpoint_path)
            print(
                f"[{len(evaluations)}/{len(questions)}] {item['id']} "
                f"retrieved_sources={evaluation.retrieved_sources}; "
                f"checkpoint={checkpoint_path}",
                flush=True,
            )
    finally:
        search_engine.client.close()

    answer_evaluations = [
        item for item in evaluations if item["expected_behavior"] == "answer"
    ]
    scored_evaluations = answer_evaluations or evaluations
    source_recall = sum(item["source_recall"] for item in scored_evaluations) / len(
        scored_evaluations
    )
    source_hit_rate = sum(item["source_hit"] for item in scored_evaluations) / len(
        scored_evaluations
    )
    fact_recall = sum(item["fact_recall"] for item in scored_evaluations) / len(
        scored_evaluations
    )
    mrr = sum(item["reciprocal_rank"] for item in scored_evaluations) / len(
        scored_evaluations
    )
    binary_rank_score = sum(
        item["binary_rank_score"] for item in scored_evaluations
    ) / len(scored_evaluations)
    precision = sum(item["precision_at_k"] for item in scored_evaluations) / len(
        scored_evaluations
    )
    latencies = [item["retrieval_latency_ms"] for item in evaluations]
    report = {
        "question_set": str(args.questions),
        "top_k": args.top_k,
        "reranker_enabled": not args.no_reranker,
        "question_count": len(evaluations),
        "completed_question_count": len(evaluations),
        "partial": False,
        "answerable_question_count": len(answer_evaluations),
        "source_recall_at_k": source_recall,
        "source_hit_rate_at_k": source_hit_rate,
        "mean_fact_recall": fact_recall,
        "mean_precision_at_k": precision,
        "mrr": mrr,
        "mean_binary_rank_score": binary_rank_score,
        "latency_ms": {
            "mean": round(sum(latencies) / len(latencies), 2),
            "p50": round(percentile(latencies, 50), 2),
            "p95": round(percentile(latencies, 95), 2),
            "max": round(max(latencies), 2),
        },
        "results": evaluations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Source Recall@{args.top_k}: {source_recall:.2%}")
    print(f"Source hit rate@{args.top_k}: {source_hit_rate:.2%}")
    print(f"Mean fact recall: {fact_recall:.2%}")
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
