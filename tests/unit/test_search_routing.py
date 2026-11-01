"""Retrieval contracts independent of question wording or business columns."""

from copy import deepcopy

import pytest
from langchain_core.documents import Document

from deal_voice.retrieval.search import LocalSearchEngine


class FakeVectorStore:
    def __init__(self, children):
        self.children = children
        self.calls = []

    def similarity_search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return self.children[: kwargs["k"]]


class FakeDocstore:
    def __init__(self, parents):
        self.parents = parents
        self.calls = []

    def mget(self, keys):
        self.calls.append(keys)
        return [self.parents.get(key) for key in keys]


def make_engine(children, parents=None, reranker=None):
    engine = LocalSearchEngine.__new__(LocalSearchEngine)
    engine.vector_store = FakeVectorStore(children)
    engine.docstore = FakeDocstore(parents or {})
    engine.reranker = reranker
    engine.candidate_pool_size = 30
    engine.rerank_batch_size = 1
    return engine


@pytest.mark.parametrize(
    "query",
    [
        "Comment calcule-t-on la marge totale ?",
        "Quels stocks faut-il comparer lors de l'inventaire ?",
        "Quel montant est le plus élevé dans tous les tableaux ?",
        "Explain allocation policy and compare its alternatives.",
    ],
)
def test_question_wording_cannot_discard_retrieved_explanations_or_tables(query):
    explanation = Document(
        page_content="Calculation rule", metadata={"source": "a.pdf"}
    )
    first_table = Document(
        page_content="Table A", metadata={"source": "a.pdf", "table_id": "a"}
    )
    second_table = Document(
        page_content="Table B", metadata={"source": "a.pdf", "table_id": "b"}
    )
    engine = make_engine([explanation, first_table, second_table])
    assert engine.search(query, top_k=3) == [explanation, first_table, second_table]


def test_children_reconstruct_complete_parents_and_duplicate_parent_hits_collapse():
    parent = Document(
        page_content="Complete table with every row", metadata={"source": "a.pdf"}
    )
    children = [
        Document(page_content=str(index), metadata={"doc_id": "p"})
        for index in range(3)
    ]
    engine = make_engine(children, {"p": parent})
    results = engine.search("What does it contain?", top_k=3)
    assert len(results) == 1
    assert results[0].page_content == parent.page_content
    assert results[0].metadata["parent_id"] == "p"
    assert engine.docstore.calls == [["p"]]
    assert "parent_id" not in parent.metadata


def test_filters_are_per_request_and_returned_parent_filters_are_checked():
    allowed = Document(
        page_content="Allowed",
        metadata={"source": "a.pdf", "document_id": "a", "allowed_roles": ["sales"]},
    )
    denied = Document(
        page_content="Denied",
        metadata={
            "source": "private.pdf",
            "document_id": "b",
            "allowed_roles": ["admin"],
        },
    )
    engine = make_engine([allowed, denied])
    assert engine.search(
        "question", user_role="sales", source="a.pdf", document_ids="a"
    ) == [allowed]
    must = engine.vector_store.calls[0][1]["filter"].must
    assert {condition.key for condition in must} == {
        "metadata.allowed_roles",
        "metadata.source",
        "metadata.document_id",
    }
    assert engine.search("question") == [allowed, denied]
    assert "filter" not in engine.vector_store.calls[1][1]


def test_explicit_empty_filter_returns_nothing():
    engine = make_engine([Document(page_content="Knowledge")])
    assert engine.search("question", source=[]) == []
    assert engine.search("question", document_ids=[]) == []
    assert engine.vector_store.calls == []


def test_missing_parent_is_dropped_instead_of_returning_a_partial_table():
    engine = make_engine(
        [Document(page_content="Incomplete", metadata={"doc_id": "missing"})]
    )
    assert engine.search("question") == []


def test_parent_with_inconsistent_authorization_is_not_returned():
    child = Document(
        page_content="Allowed child",
        metadata={"doc_id": "p", "allowed_roles": ["sales"]},
    )
    parent = Document(
        page_content="Private parent", metadata={"allowed_roles": ["admin"]}
    )
    engine = make_engine([child], {"p": parent})

    assert engine.search("question", user_role="sales") == []


def test_reranking_is_bounded_uses_small_batches_and_does_not_mutate_stored_docs():
    documents = [Document(page_content=f"Evidence {index}") for index in range(4)]
    original = deepcopy(documents)

    class Reranker:
        def predict(self, pairs, **kwargs):
            assert len(pairs) == 4
            assert kwargs == {"batch_size": 1, "show_progress_bar": False}
            return [1, 4, 3, 2]

    engine = make_engine(documents, reranker=Reranker())
    result = engine.search("question", top_k=2)
    assert [document.page_content for document in result] == [
        "Evidence 1",
        "Evidence 2",
    ]
    assert documents == original


def test_repeated_content_in_distinct_documents_is_not_deduplicated():
    documents = [
        Document(page_content="same", metadata={"document_id": key})
        for key in ("a", "b")
    ]
    engine = make_engine(documents)
    assert len(engine.search("question")) == 2


def test_larger_top_k_expands_the_bounded_candidate_pool():
    engine = make_engine([])
    engine.search("question", top_k=50)
    assert engine.vector_store.calls[0][1]["k"] == 50


def test_invalid_reranker_scores_are_rejected():
    class Reranker:
        def predict(self, pairs, **kwargs):
            return [float("nan")] * len(pairs)

    engine = make_engine(
        [Document(page_content="a"), Document(page_content="b")], reranker=Reranker()
    )
    with pytest.raises(ValueError, match="non-finite"):
        engine.search("question", top_k=1)


@pytest.mark.parametrize("top_k", [0, -1, True, 201])
def test_unbounded_or_invalid_requests_are_rejected(top_k):
    with pytest.raises(ValueError, match="top_k"):
        make_engine([]).search("question", top_k=top_k)


def test_empty_query_is_rejected():
    with pytest.raises(ValueError, match="query"):
        make_engine([]).search("  ")
