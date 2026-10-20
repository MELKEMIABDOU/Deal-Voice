import pytest
from deal_voice.ingestion.indexer import EnterpriseIndexer
from deal_voice.retrieval.search import LocalSearchEngine
from langchain_core.documents import Document


@pytest.fixture
def temp_qdrant_db(tmp_path):
    """Creates a temporary, isolated Qdrant database for testing."""
    db_dir = str(tmp_path / "qdrant_test_db")
    indexer = EnterpriseIndexer(db_path=db_dir, collection_name="test_collection")
    
    # Ingest two distinct test documents with different access levels
    test_docs = [
        Document(
            page_content="Enterprise Q3 discount pricing is 15% off standard seat licenses.",
            metadata={"source": "pricing_guide.pdf", "allowed_roles": ["sales", "admin"]},
        ),
        Document(
            page_content="Executive board memo: CEO annual compensation target set to $2M.",
            metadata={"source": "exec_memo.pdf", "allowed_roles": ["admin"]},
        ),
    ]
    indexer.index_documents(test_docs)
    return db_dir


def test_rbac_security_isolation(temp_qdrant_db):
    search_engine = LocalSearchEngine(
        db_path=temp_qdrant_db, collection_name="test_collection"
    )
    query = "What is the Q3 pricing discount and the CEO compensation?"

    # 1. Test as 'sales' role -> Should retrieve pricing, but NOT CEO compensation
    sales_results = search_engine.search(query=query, user_role="sales", top_k=5)
    sales_contents = [doc.page_content for doc in sales_results]

    assert len(sales_results) == 1
    assert "Enterprise Q3 discount pricing" in sales_contents[0]
    assert not any("CEO annual compensation" in text for text in sales_contents)

    # 2. Test as 'admin' role -> Should retrieve both documents
    admin_results = search_engine.search(query=query, user_role="admin", top_k=5)
    assert len(admin_results) == 2

    # 3. Test as 'intern' role -> Should retrieve zero documents
    intern_results = search_engine.search(query=query, user_role="intern", top_k=5)
    assert len(intern_results) == 0


def test_search_without_role_returns_knowledge_base_results(temp_qdrant_db):
    search_engine = LocalSearchEngine(
        db_path=temp_qdrant_db, collection_name="test_collection"
    )

    results = search_engine.search(
        query="What is the Q3 pricing discount and the CEO compensation?",
        top_k=5,
    )

    assert len(results) == 2


def test_search_can_filter_by_source_and_top_k(temp_qdrant_db):
    search_engine = LocalSearchEngine(
        db_path=temp_qdrant_db, collection_name="test_collection"
    )

    results = search_engine.search(
        query="What is the Q3 pricing discount?",
        source=["pricing_guide.pdf"],
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].metadata["source"] == "pricing_guide.pdf"