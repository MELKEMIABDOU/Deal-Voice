from langchain_core.documents import Document

from deal_voice.retrieval.corrective import (
    CorrectiveRetrievalManager,
    CoverageEvaluator,
    EvidenceMerger,
)


def table(*, complete: bool, value: int = 12) -> Document:
    return Document(
        page_content=(
            f"| Produit | Stock restant |\n| --- | --- |\n| Beurre | {value} |"
        ),
        metadata={
            "source": "stock.pdf",
            "content_kind": "table",
            "table_complete": complete,
            "table_id": f"stock-{complete}-{value}",
        },
    )


class SearchEngine:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return self.results.pop(0)


def test_complete_matching_table_does_not_trigger_corrective_search():
    engine = SearchEngine([[table(complete=True)]])
    result = CorrectiveRetrievalManager(engine).retrieve(
        query="Quel produit a le stock restant le plus faible ?",
        user_role="sales",
        document_ids=["doc-1"],
        top_k=5,
    )

    assert result.coverage.status == "complete"
    assert not result.corrective_search_used
    assert result.retrieval_status == "initial"
    assert len(engine.calls) == 1


def test_incomplete_matching_table_gets_one_filtered_corrective_search():
    engine = SearchEngine([[table(complete=False)], [table(complete=True)]])
    result = CorrectiveRetrievalManager(engine).retrieve(
        query="Quel produit a le stock restant le plus faible ?",
        user_role="sales",
        document_ids=["doc-1"],
        top_k=3,
    )

    assert result.coverage.status == "complete"
    assert result.corrective_search_used
    assert result.retrieval_status == "corrected"
    assert len(engine.calls) == 2
    assert "tableau complet" in engine.calls[1]["query"]
    assert engine.calls[1]["user_role"] == "sales"
    assert engine.calls[1]["document_ids"] == ["doc-1"]
    assert engine.calls[1]["top_k"] == 3


def test_failed_correction_stops_after_one_attempt():
    engine = SearchEngine([[], []])
    result = CorrectiveRetrievalManager(engine).retrieve(
        query="Information absente", top_k=5
    )

    assert result.coverage.status == "insufficient"
    assert result.corrective_search_used
    assert result.retrieval_status == "exhausted"
    assert len(engine.calls) == 2


def test_narrative_results_keep_existing_semantic_validation_path():
    evidence = Document(
        page_content="Le contrat commence lundi.", metadata={"source": "contrat.txt"}
    )
    assessment = CoverageEvaluator().evaluate("Quand commence le contrat ?", [evidence])

    assert assessment.status == "complete"
    assert not assessment.corrective_search_required


def test_merger_deduplicates_and_respects_the_result_limit():
    duplicate = table(complete=False)
    other = table(complete=True, value=16)
    merged = EvidenceMerger.merge(
        [duplicate], [duplicate, other], question="stock restant", limit=2
    )

    assert len(merged) == 2
    assert {document.page_content for document in merged} == {
        duplicate.page_content,
        other.page_content,
    }


def test_semantic_correction_is_one_search_with_the_same_filters():
    initial = Document(page_content="Contexte initial", metadata={"source": "a.txt"})
    extra = Document(page_content="Preuve complémentaire", metadata={"source": "b.txt"})
    engine = SearchEngine([[extra]])

    result = CorrectiveRetrievalManager(engine).correct(
        query="Qui a signé ?",
        initial=[initial],
        user_role="sales",
        document_ids=["doc-1"],
        top_k=5,
    )

    assert result.corrective_search_used
    assert len(result.documents) == 2
    assert len(engine.calls) == 1
    assert engine.calls[0]["user_role"] == "sales"
    assert engine.calls[0]["document_ids"] == ["doc-1"]
