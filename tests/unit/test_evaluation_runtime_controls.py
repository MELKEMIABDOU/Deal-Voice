import json
from pathlib import Path
from types import SimpleNamespace

from langchain_core.documents import Document

from deal_voice.generation.generator import RAGGenerator
from deal_voice.retrieval.search import LocalSearchEngine
from scripts.evaluate_generation import citation_details_available, write_checkpoint


class FakeRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.search_kwargs = {}

    def invoke(self, query):
        return self.documents


def test_no_reranker_still_honors_top_k():
    documents = [Document(page_content=f"Result {index}") for index in range(8)]
    engine = LocalSearchEngine.__new__(LocalSearchEngine)
    engine.vector_store = SimpleNamespace(
        similarity_search=lambda query, **kwargs: documents
    )
    engine.docstore = SimpleNamespace(mget=lambda keys: [])
    engine.candidate_pool_size = 30
    engine.rerank_batch_size = 1
    engine.reranker = None

    results = engine.search("Explain this document", top_k=3)

    assert results == documents[:3]


def test_empty_retrieval_returns_a_french_refusal_for_french_question(offline_ollama):
    generator = RAGGenerator()

    answer = generator.generate("Quel est le sujet du document ?", [])

    assert "Je ne dispose pas" in answer


def test_page_citation_is_not_scored_when_retrieved_metadata_has_no_page():
    citation = {
        "file": "rapport.pdf",
        "page": 3,
        "section": "1. Objet et périmètre du rapport",
    }
    documents = [
        Document(
            page_content="1. Objet et périmètre du rapport",
            metadata={"source": "rapport.pdf"},
        )
    ]

    evaluable, not_evaluable = citation_details_available([citation], documents)

    assert "page" not in evaluable[0]
    assert evaluable[0]["section"] == citation["section"]
    assert not_evaluable == ["page 3"]


def test_checkpoint_contains_completed_rows_and_is_marked_partial(monkeypatch):
    writes = {}

    def fake_write_text(path, text, encoding=None):
        writes["temporary_path"] = path
        writes["text"] = text

    def fake_replace(path, target):
        writes["target"] = target
        return target

    monkeypatch.setattr(Path, "write_text", fake_write_text)
    monkeypatch.setattr(Path, "replace", fake_replace)
    output_path = Path("generation.json")

    checkpoint = write_checkpoint(
        output_path,
        model="qwen3:8b",
        top_k=3,
        expected_count=5,
        question_set="questions.jsonl",
        reranker_enabled=False,
        evaluations=[{"question_id": "pdf-001"}],
    )

    payload = json.loads(writes["text"])
    assert checkpoint == Path("generation.checkpoint.json")
    assert writes["target"] == checkpoint
    assert payload["checkpoint"] is True
    assert payload["completed_question_count"] == 1
    assert payload["results"] == [{"question_id": "pdf-001"}]
