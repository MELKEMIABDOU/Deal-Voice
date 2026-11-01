import hashlib
import json
import sys
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from deal_voice.evaluation.question_set import load_question_set
from scripts import evaluate_generation as evaluator


@pytest.fixture
def question_file(tmp_path):
    path = tmp_path / "questions.jsonl"
    question = {
        "id": "q1",
        "language": "fr",
        "category": "audio_fact",
        "question": "Quel produit propose le commercial ?",
        "sources": ["call.mp3"],
        "required_facts": ["fontaines"],
        "expected_citations": [{"file": "call.mp3", "page": 2}],
    }
    path.write_text(json.dumps(question, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def quiet_monitor(monkeypatch):
    class QuietMonitor:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def mark(self, phase, question_id=None):
            pass

    monkeypatch.setattr(evaluator, "MemoryMonitor", QuietMonitor)


def write_snapshot(path, question_file):
    question = load_question_set(question_file)[0]
    document = {
        "page_content": "Le commercial propose des fontaines.",
        "metadata": {
            "source": "call.mp3",
            "page": 2,
            "speakers": ["SPEAKER_00"],
            "start_time": 1.5,
            "end_time": 4.0,
        },
    }
    payload = {
        "schema_version": 1,
        "question_set": str(question_file),
        "question_set_sha256": hashlib.sha256(question_file.read_bytes()).hexdigest(),
        "top_k": 3,
        "reranker_enabled": False,
        "indexed_sources": ["call.mp3", "unrelated.pdf"],
        "results": [
            {
                "question_id": question["id"],
                "question": question["question"],
                "documents": [document],
                "retrieval_latency_ms": 12.5,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_preparation_saves_unfiltered_contexts_without_creating_a_generator(
    monkeypatch, tmp_path, question_file
):
    calls = {"search": [], "closed": False}
    document = Document(
        page_content="Le commercial propose des fontaines.",
        metadata={"source": "call.mp3", "page": 2},
    )

    class FakeSearchEngine:
        def __init__(self, use_reranker):
            assert use_reranker is False
            self.client = SimpleNamespace(close=self.close)

        def list_documents(self):
            return [{"source": "call.mp3"}, {"source": "unrelated.pdf"}]

        def search(self, **kwargs):
            calls["search"].append(kwargs)
            return [document]

        def close(self):
            calls["closed"] = True

    class ForbiddenGenerator:
        def __init__(self, **kwargs):
            pytest.fail("Preparation must not create a generation model.")

    monkeypatch.setitem(
        sys.modules,
        "deal_voice.retrieval.search",
        SimpleNamespace(LocalSearchEngine=FakeSearchEngine),
    )
    monkeypatch.setitem(
        sys.modules,
        "deal_voice.generation.generator",
        SimpleNamespace(RAGGenerator=ForbiddenGenerator),
    )
    output = tmp_path / "contexts.json"

    evaluator.main(
        [
            "--questions",
            str(question_file),
            "--top-k",
            "3",
            "--no-reranker",
            "--prepare-contexts",
            str(output),
        ]
    )

    snapshot = json.loads(output.read_text(encoding="utf-8"))
    assert calls["closed"] is True
    assert calls["search"] == [
        {
            "query": load_question_set(question_file)[0]["question"],
            "source": None,
            "top_k": 3,
        }
    ]
    assert snapshot["indexed_sources"] == ["call.mp3", "unrelated.pdf"]
    assert snapshot["results"][0]["documents"] == [
        {"page_content": document.page_content, "metadata": document.metadata}
    ]
    assert "required_facts" not in snapshot["results"][0]
    assert "expected_answer" not in snapshot["results"][0]
    assert not output.with_suffix(".json.tmp").exists()


@pytest.mark.parametrize("trace_mode", [None, "correct", "wrong_relation"])
def test_replay_preserves_context_and_scoring_without_loading_retrieval(
    monkeypatch, tmp_path, question_file, trace_mode
):
    if trace_mode is not None:
        question = json.loads(question_file.read_text(encoding="utf-8"))
        question["expected_values"] = [
            {"entity": "F001", "field": "Produit", "value": "fontaines"}
        ]
        question_file.write_text(json.dumps(question), encoding="utf-8")
    contexts = tmp_path / "contexts.json"
    snapshot = write_snapshot(contexts, question_file)
    generated = []

    class ForbiddenSearchEngine:
        def __init__(self, **kwargs):
            pytest.fail("Generation from contexts must not load retrieval.")

    class FakeGenerator:
        def __init__(self, model_name):
            assert model_name == "test-model"
            if trace_mode is not None:
                self.last_trace = {
                    "status": "answer",
                    "actual_values": [
                        {
                            "entity": "F001" if trace_mode == "correct" else "F002",
                            "field": "Produit",
                            "value": "fontaines",
                        }
                    ],
                    "structured_citations": [{"file": "call.mp3", "page": 2}],
                    "computations": [{"operation": "select", "evidence_id": "E1"}],
                }

        def generate(self, question, documents):
            generated.append((question, documents))
            return "Le produit concerne les fontaines.\nSource : call.mp3 — page 2"

    monkeypatch.setitem(
        sys.modules,
        "deal_voice.retrieval.search",
        SimpleNamespace(LocalSearchEngine=ForbiddenSearchEngine),
    )
    monkeypatch.setitem(
        sys.modules,
        "deal_voice.generation.generator",
        SimpleNamespace(RAGGenerator=FakeGenerator),
    )
    output = tmp_path / "generation.json"

    evaluator.main(
        [
            "--questions",
            str(question_file),
            "--from-contexts",
            str(contexts),
            "--model",
            "test-model",
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    checkpoint = json.loads(
        output.with_name("generation.checkpoint.json").read_text(encoding="utf-8")
    )
    assert generated[0][0] == snapshot["results"][0]["question"]
    assert (
        generated[0][1][0].page_content
        == (snapshot["results"][0]["documents"][0]["page_content"])
    )
    assert (
        generated[0][1][0].metadata
        == (snapshot["results"][0]["documents"][0]["metadata"])
    )
    assert report["top_k"] == 3
    assert report["reranker_enabled"] is False
    assert report["pass_rate"] == (0.0 if trace_mode == "wrong_relation" else 1.0)
    assert report["mean_fact_recall"] == 1.0
    assert report["answer_citation_file_recall"] == 1.0
    assert report["results"][0]["retrieval_latency_ms"] == 12.5
    assert checkpoint["results"] == report["results"]
    assert report["evaluation_schema_version"] == 2
    if trace_mode is not None:
        assert report["results"][0]["generation_status"] == "answer"
        assert report["results"][0]["computations"][0]["operation"] == "select"
        assert report["mean_value_recall"] == 1.0
        assert report["mean_value_relation_recall"] == (
            0.0 if trace_mode == "wrong_relation" else 1.0
        )
    else:
        assert report["results"][0]["generation_status"] == "legacy_text"
        assert report["mean_value_relation_recall"] is None


def test_changed_question_file_is_rejected_before_generation(
    monkeypatch, tmp_path, question_file, capsys
):
    contexts = tmp_path / "contexts.json"
    write_snapshot(contexts, question_file)
    question_file.write_text(
        question_file.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )

    class ForbiddenGenerator:
        def __init__(self, **kwargs):
            pytest.fail("Stale contexts must be rejected before loading a model.")

    monkeypatch.setitem(
        sys.modules,
        "deal_voice.generation.generator",
        SimpleNamespace(RAGGenerator=ForbiddenGenerator),
    )

    with pytest.raises(SystemExit) as error:
        evaluator.main(
            [
                "--questions",
                str(question_file),
                "--from-contexts",
                str(contexts),
                "--output",
                str(tmp_path / "generation.json"),
            ]
        )

    assert error.value.code == 2
    assert "question set changed" in capsys.readouterr().err


@pytest.mark.parametrize("invalid_kind", ["duplicate", "missing", "question_changed"])
def test_invalid_prepared_questions_are_rejected(tmp_path, question_file, invalid_kind):
    contexts = tmp_path / "contexts.json"
    snapshot = write_snapshot(contexts, question_file)
    if invalid_kind == "duplicate":
        snapshot["results"].append(dict(snapshot["results"][0]))
        expected = "Duplicate prepared question"
    elif invalid_kind == "missing":
        snapshot["results"][0]["question_id"] = "another-question"
        expected = "Missing or changed prepared question"
    else:
        snapshot["results"][0]["question"] = "Une autre question ?"
        expected = "Missing or changed prepared question"
    contexts.write_text(json.dumps(snapshot), encoding="utf-8")

    with pytest.raises(ValueError, match=expected):
        evaluator.load_prepared_contexts(
            contexts, question_file, load_question_set(question_file)
        )


def test_replay_rejects_a_different_top_k(tmp_path, question_file, capsys):
    contexts = tmp_path / "contexts.json"
    write_snapshot(contexts, question_file)

    with pytest.raises(SystemExit) as error:
        evaluator.main(
            [
                "--questions",
                str(question_file),
                "--from-contexts",
                str(contexts),
                "--top-k",
                "5",
                "--output",
                str(tmp_path / "generation.json"),
            ]
        )

    assert error.value.code == 2
    assert "--top-k differs" in capsys.readouterr().err
