"""Acceptance uses actual API statuses and evidence; no model or network calls."""

import json
from types import SimpleNamespace

import pytest
import requests

from scripts.evaluate_client import assess_response, check_api, run_evaluation


@pytest.fixture
def question():
    return {
        "id": "price-1",
        "question": "Quel est le prix de l'article Alpha ?",
        "sources": ["rapport.pdf"],
        "expected_facts": ["Alpha"],
        "expected_behavior": "answer",
        "expected_values": [{"entity": "Alpha", "field": "Prix", "value": "10 €"}],
        "expected_citations": [{"file": "rapport.pdf", "table": "Tarifs"}],
    }


@pytest.fixture
def response():
    return {
        "answer": "Le prix de Alpha est 10 €.\nSource : rapport.pdf — table « Tarifs »",
        "answer_status": "complete",
        "actual_values": [{"entity": "Alpha", "field": "Prix", "value": "10 €"}],
        "available_citations": [{"file": "rapport.pdf", "table": "Tarifs"}],
    }


def test_complete_response_with_verified_values_passes(question, response):
    assert assess_response(question, response)["accepted"]


@pytest.mark.parametrize("response", [None, [], {"answer": None}, {}])
def test_invalid_response_is_rejected(question, response):
    with pytest.raises(ValueError, match="answer string"):
        assess_response(question, response)


@pytest.mark.parametrize("status", ["partial", "unverified", "insufficient", "unknown"])
def test_correct_facts_do_not_hide_an_unaccepted_status(question, response, status):
    response["answer_status"] = status
    result = assess_response(question, response)
    assert result["checks_passed"]
    assert not result["accepted"]
    assert result["failure_reasons"] == ["unexpected_answer_status"]


def test_legacy_api_without_numeric_evidence_cannot_pass(question, response):
    response.pop("actual_values")
    result = assess_response(question, response)
    assert "value_relations_not_evaluable" in result["failure_reasons"]
    assert not result["accepted"]


def test_missing_page_requirement_is_not_silently_removed(question, response):
    question["expected_citations"][0]["page"] = 2
    result = assess_response(question, response)
    assert "citation_details_missing" in result["failure_reasons"]
    assert not result["accepted"]


@pytest.mark.parametrize(
    "status,accepted", [("insufficient", True), ("unverified", False)]
)
def test_refusal_is_not_confused_with_validation_failure(question, status, accepted):
    question.update(
        expected_behavior="refusal",
        sources=[],
        expected_citations=[],
        expected_values=[],
        expected_facts=[],
    )
    result = assess_response(
        question,
        {
            "answer": "Je ne dispose pas de suffisamment d'informations pour répondre.",
            "answer_status": status,
        },
    )
    assert result["accepted"] is accepted


def test_timeout_preserves_previous_results_and_never_retries(
    question, response, tmp_path
):
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        if len(calls) == 2:
            raise requests.Timeout("server may still be working")
        return SimpleNamespace(
            status_code=200, raise_for_status=lambda: None, json=lambda: response
        )

    output = tmp_path / "results.json"
    report = run_evaluation(
        SimpleNamespace(post=post),
        api_url="http://localhost:8000",
        questions=[dict(question, id=f"q-{i}") for i in range(3)],
        output=output,
        document_ids=["allowed-doc"],
    )
    assert len(calls) == 2
    assert calls[0][1]["json"] == {
        "query": question["question"],
        "top_k": 5,
        "document_ids": ["allowed-doc"],
    }
    assert report["completed_question_count"] == 1
    assert report["completed"] is False
    assert report["automated_acceptance_rate"] is None
    assert report["results"][0]["accepted"]
    assert report["results"][1]["failure_reasons"] == ["request_or_response_error"]
    assert "server may still be working" in report["results"][1]["error_message"]
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_api_preflight_requires_a_ready_server():
    response = SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {"ready": False},
    )
    session = SimpleNamespace(get=lambda *args, **kwargs: response)
    with pytest.raises(RuntimeError, match="not ready"):
        check_api(session, "http://localhost:8000")


def test_api_preflight_reports_connection_error():
    def unavailable(*args, **kwargs):
        raise requests.ConnectionError("connection refused")

    with pytest.raises(RuntimeError, match="connection refused"):
        check_api(SimpleNamespace(get=unavailable), "http://localhost:8000")


def test_runner_exercises_api_corrective_path(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from langchain_core.documents import Document

    from deal_voice.api.main import app
    from deal_voice.config import settings
    from tests.unit.test_api_search import CorrectedGenerator

    calls = []

    def search(**kwargs):
        calls.append(kwargs)
        return [Document(page_content="Fait", metadata={"source": "rapport.pdf"})]

    monkeypatch.setattr(settings, "rbac_enabled", False)
    monkeypatch.setattr(
        app.state, "search_engine", SimpleNamespace(search=search), raising=False
    )
    generator = CorrectedGenerator()
    monkeypatch.setattr(app.state, "generator", generator, raising=False)
    client = TestClient(app)
    session = SimpleNamespace(post=lambda url, **kw: client.post(url, json=kw["json"]))
    question = {
        "id": "corrective",
        "question": "Qui a signé ?",
        "sources": ["rapport.pdf"],
        "expected_facts": ["Fait"],
        "expected_values": [],
        "expected_citations": [],
        "expected_behavior": "answer",
    }
    try:
        report = run_evaluation(
            session,
            api_url="http://testserver",
            questions=[question],
            output=tmp_path / "api.json",
        )
    finally:
        client.close()
    assert len(calls) == 2
    assert len(generator.calls) == 2
    assert report["completed"]
    assert report["results"][0]["response"]["corrective_search_used"]
    assert report["results"][0]["response"]["answer"] == "Corrected answer"
    assert not report["results"][0]["accepted"]
