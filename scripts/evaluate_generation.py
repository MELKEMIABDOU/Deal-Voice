r"""Evaluate generated RAG answers for the reference question set.

The evaluation documents must already be indexed and Ollama must be running.

Example:
    uv run python scripts\evaluate_generation.py `
      --questions data\evaluation\questions.json `
      --model qwen3:8b
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from deal_voice.evaluation.generation import evaluate_generation
from deal_voice.generation.generator import RAGGenerator
from deal_voice.retrieval.search import LocalSearchEngine


def percentile(values: list[float], percentile_value: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = round((percentile_value / 100) * (len(ordered) - 1))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate final LLM answers on reference questions."
    )
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--top-k", type=int, default=5)
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
    args = parser.parse_args()

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be at least 1")
        questions = questions[: args.limit]
    if not questions:
        parser.error("The questions file must contain at least one question")

    search_engine = LocalSearchEngine()
    generator = RAGGenerator(model_name=args.model)
    evaluations = []
    try:
        for item in questions:
            started = time.perf_counter()
            retrieval_started = time.perf_counter()
            documents = search_engine.search(
                query=item["question"],
                source=item["source"],
                top_k=args.top_k,
            )
            retrieval_latency_ms = (time.perf_counter() - retrieval_started) * 1000
            generation_started = time.perf_counter()
            answer = generator.generate(item["question"], documents)
            generation_latency_ms = (time.perf_counter() - generation_started) * 1000
            total_latency_ms = (time.perf_counter() - started) * 1000
            evaluation = evaluate_generation(
                item["id"],
                item["question"],
                item["source"],
                item["expected_facts"],
                answer,
                item.get("expected_fact_variants"),
                item.get("expected_behavior", "answer"),
                item.get("forbidden_claims"),
            )
            evaluations.append(
                {
                    "question_id": evaluation.question_id,
                    "question": item["question"],
                    "answer": evaluation.answer,
                    "expected_source": evaluation.source_expected,
                    "source_present": evaluation.source_present,
                    "facts_found": evaluation.facts_found,
                    "facts_missing": evaluation.facts_missing,
                    "fact_recall": evaluation.fact_recall,
                    "lexical_facts_found": evaluation.lexical_facts_found,
                    "lexical_facts_missing": evaluation.lexical_facts_missing,
                    "lexical_fact_recall": evaluation.lexical_fact_recall,
                    "answer_non_empty": evaluation.answer_non_empty,
                    "likely_french": evaluation.likely_french,
                    "expected_behavior": item.get("expected_behavior", "answer"),
                    "refusal_detected": evaluation.refusal_detected,
                    "forbidden_claims": item.get("forbidden_claims", []),
                    "unsupported_claims_found": evaluation.unsupported_claims_found,
                    "fidelity_pass": not evaluation.unsupported_claims_found,
                    "retrieval_latency_ms": round(retrieval_latency_ms, 2),
                    "generation_latency_ms": round(generation_latency_ms, 2),
                    "total_latency_ms": round(total_latency_ms, 2),
                    "passed": evaluation.passed,
                }
            )
    finally:
        search_engine.client.close()

    report = {
        "model": args.model,
        "top_k": args.top_k,
        "question_count": len(evaluations),
        "source_accuracy": sum(
            item["source_present"] for item in evaluations
        ) / len(evaluations),
        "mean_fact_recall": sum(
            item["fact_recall"] for item in evaluations
        ) / len(evaluations),
        "language_compliance": sum(
            item["likely_french"] for item in evaluations
        ) / len(evaluations),
        "refusal_accuracy": (
            sum(
                item["passed"]
                for item in evaluations
                if item["expected_behavior"] == "refusal"
            )
            / sum(
                item["expected_behavior"] == "refusal"
                for item in evaluations
            )
            if any(
                item["expected_behavior"] == "refusal" for item in evaluations
            )
            else None
        ),
        "fidelity_accuracy": sum(
            item["fidelity_pass"] for item in evaluations
        ) / len(evaluations),
        "pass_rate": sum(item["passed"] for item in evaluations) / len(evaluations),
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Source accuracy: {report['source_accuracy']:.2%}")
    print(f"Mean fact recall: {report['mean_fact_recall']:.2%}")
    print(f"Language compliance: {report['language_compliance']:.2%}")
    print(f"Pass rate: {report['pass_rate']:.2%}")
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
