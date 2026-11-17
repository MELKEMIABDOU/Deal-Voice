"""Evaluate the real client API sequentially, without loading local AI models."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import requests

from deal_voice.evaluation.generation import evaluate_generation
from deal_voice.evaluation.question_set import load_question_set


def assess_response(question: dict, response: dict) -> dict:
    """Keep reference requirements intact; partial/unverified is not acceptance."""
    if not isinstance(response, dict) or not isinstance(response.get("answer"), str):
        raise ValueError("The API response must contain an answer string.")
    evaluation = evaluate_generation(
        question_id=question["id"],
        question=question["question"],
        expected_source=question["sources"],
        expected_facts=question["expected_facts"],
        answer=response["answer"],
        expected_fact_variants=question.get("expected_fact_variants"),
        expected_behavior=question["expected_behavior"],
        forbidden_claims=question.get("forbidden_claims"),
        expected_values=question["expected_values"],
        expected_citations=question["expected_citations"],
        expected_language=question.get("language", "fr"),
        actual_values=response.get("actual_values"),
        available_citations=response.get("available_citations", []),
    )
    expected_status = (
        "insufficient" if question["expected_behavior"] == "refusal" else "complete"
    )
    reasons = [
        name
        for name in (
            "facts_missing",
            "expected_values_missing",
            "value_relations_missing",
            "value_relations_not_evaluable",
            "citation_files_missing",
            "citation_details_missing",
            "invalid_citation_details",
            "forbidden_claims_found",
        )
        if getattr(evaluation, name)
    ]
    if not evaluation.language_compliant:
        reasons.append("language_mismatch")
    if not evaluation.answer_non_empty:
        reasons.append("empty_answer")
    if evaluation.refusal_expected and not evaluation.refusal_detected:
        reasons.append("refusal_missing")
    if response.get("answer_status") != expected_status:
        reasons.append("unexpected_answer_status")
    return {
        "checks_passed": evaluation.passed,
        "accepted": evaluation.passed and not reasons,
        "failure_reasons": reasons,
        "fact_recall": evaluation.fact_recall,
        "checks": asdict(evaluation),
    }


def check_api(session: requests.Session, api_url: str, timeout: float = 10) -> dict:
    """Fail before evaluation when the target API is unreachable or not ready."""
    endpoint = api_url.rstrip("/") + "/api/v1/status"
    try:
        response = session.get(endpoint, timeout=(5, timeout), allow_redirects=False)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError(
            f"DealVoice API is unavailable at {endpoint}: {exc}"
        ) from exc
    if not isinstance(payload, dict) or payload.get("ready") is not True:
        raise RuntimeError(
            f"DealVoice API answered at {endpoint}, but it is not ready."
        )
    return payload


def run_evaluation(
    session: requests.Session,
    *,
    api_url: str,
    questions: list[dict],
    output: Path,
    top_k: int = 5,
    timeout: float = 600,
    document_ids: list[str] | None = None,
    dataset_sha256: str = "",
) -> dict:
    """One HTTP request at a time; checkpoint even on a timeout, then stop."""
    if not questions:
        raise ValueError("At least one question is required.")
    report = {
        "evaluation_mode": "client_api",
        "dataset_sha256": dataset_sha256,
        "top_k": top_k,
        "document_ids": document_ids,
        "question_count": len(questions),
        "completed_question_count": 0,
        "completed": False,
        "automated_acceptance_rate": None,
        "requires_human_review": True,
        "results": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    for index, question in enumerate(questions, start=1):
        payload = {"query": question["question"], "top_k": top_k}
        if document_ids is not None:
            payload["document_ids"] = document_ids
        row = {"id": question["id"], "question": question["question"]}
        started = time.perf_counter()
        try:
            result = session.post(
                api_url.rstrip("/") + "/api/v1/search",
                json=payload,
                timeout=(10, timeout),
                allow_redirects=False,
            )
            result.raise_for_status()
            if result.status_code != 200:
                raise ValueError("Expected a JSON HTTP 200 response.")
            response = result.json()
            row.update(assess_response(question, response))
            row["response"] = response
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            # No automatic retry: the server may still be processing a timeout.
            row.update(
                accepted=False,
                failure_reasons=["request_or_response_error"],
                error_type=type(exc).__name__,
                error_message=str(exc)[:500],
            )
            if isinstance(exc, requests.HTTPError) and exc.response is not None:
                row["http_status"] = exc.response.status_code
        row["latency_seconds"] = round(time.perf_counter() - started, 3)
        report["results"].append(row)
        completed = sum("response" in item for item in report["results"])
        report["completed_question_count"] = completed
        report["completed"] = completed == len(questions)
        # Interrupted runs never report a rate for the entire requested sample.
        if report["completed"]:
            report["automated_acceptance_rate"] = sum(
                item["accepted"] for item in report["results"]
            ) / len(questions)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(output)
        print(f"[{index}/{len(questions)}] {question['id']} accepted={row['accepted']}")
        if "error_type" in row:
            break
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--document-id", action="append", dest="document_ids")
    parser.add_argument("--role", help="Only for an API configured to require a role.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.offset < 0
        or args.limit < 1
        or not 1 <= args.top_k <= 50
        or args.timeout <= 0
    ):
        parser.error("Require offset >= 0, limit >= 1, top-k in 1..50, timeout > 0.")
    questions = load_question_set(args.questions)[
        args.offset : args.offset + args.limit
    ]
    if not questions:
        parser.error("No questions in the selected range.")
    if args.output.exists():
        parser.error(
            "Output already exists; use a new report name to preserve results."
        )
    with requests.Session() as session:
        if args.role:
            session.headers["X-User-Role"] = args.role
        status = check_api(session, args.api_url)
        print(
            "API ready: "
            f"model={status.get('model', 'unknown')} "
            f"busy={status.get('busy', 'unknown')}"
        )
        report = run_evaluation(
            session,
            api_url=args.api_url,
            questions=questions,
            output=args.output,
            top_k=args.top_k,
            timeout=args.timeout,
            document_ids=args.document_ids,
            dataset_sha256=hashlib.sha256(args.questions.read_bytes()).hexdigest(),
        )
    print(f"Report written to {args.output}")
    rate = report["automated_acceptance_rate"]
    print(
        f"Automated acceptance: {rate:.2%}" if rate is not None else "Run incomplete."
    )
    return 0 if rate == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
