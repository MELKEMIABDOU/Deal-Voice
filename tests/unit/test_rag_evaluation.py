from langchain_core.documents import Document

from deal_voice.evaluation.rag import evaluate_retrieval


def test_retrieval_evaluation_scores_source_and_facts():
    result = evaluate_retrieval(
        "q1",
        "call.mp3",
        ["fontaines à eau", "bouteilles", "rendez-vous"],
        [
            Document(
                page_content="Proposition de fontaines à eau au lieu des bouteilles.",
                metadata={"source": "call.mp3"},
            )
        ],
    )

    assert result.source_hit is True
    assert result.expected_facts_found == ["fontaines à eau", "bouteilles"]
    assert result.fact_recall == 2 / 3
    assert result.expected_source_rank == 1
    assert result.reciprocal_rank == 1.0
    assert result.binary_rank_score == 1.0
    assert result.precision_at_k == 1.0


def test_retrieval_evaluation_accepts_controlled_fact_variants():
    result = evaluate_retrieval(
        "q2",
        "call.mp3",
        ["fontaines à eau"],
        [
            Document(
                page_content="Une proposition de fontaines est présentée.",
                metadata={"source": "call.mp3"},
            )
        ],
        {"fontaines à eau": ["fontaines", "système de fontaines"]},
    )

    assert result.expected_facts_found == ["fontaines à eau"]
    assert result.expected_facts_missing == []
    assert result.fact_recall == 1.0
