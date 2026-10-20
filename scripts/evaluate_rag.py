r"""Evaluate retrieval against the reference question set.

The indexed evaluation documents must already exist in Qdrant.

Example:
    uv run python scripts\evaluate_rag.py `
      --questions data\evaluation\questions.json `
      --top-k 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

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
        "--output",
        type=Path,
        default=Path("reports/rag/retrieval-evaluation.json"),
    )
    args = parser.parse_args()
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    search_engine = LocalSearchEngine()
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
                item["source"],
                item["expected_facts"],
                documents,
                item.get("expected_fact_variants"),
            )
            evaluations.append(
                {
                    "question_id": evaluation.question_id,
                    "retrieved_sources": evaluation.retrieved_sources,
                    "source_hit": evaluation.source_hit,
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
    finally:
        search_engine.client.close()

    source_recall = sum(item["source_hit"] for item in evaluations) / len(evaluations)
    fact_recall = sum(item["fact_recall"] for item in evaluations) / len(evaluations)
    mrr = sum(item["reciprocal_rank"] for item in evaluations) / len(evaluations)
    binary_rank_score = sum(
        item["binary_rank_score"] for item in evaluations
    ) / len(evaluations)
    precision = sum(item["precision_at_k"] for item in evaluations) / len(evaluations)
    latencies = [item["retrieval_latency_ms"] for item in evaluations]
    report = {
        "top_k": args.top_k,
        "question_count": len(evaluations),
        "source_recall_at_k": source_recall,
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
    print(f"Mean fact recall: {fact_recall:.2%}")
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
