import shutil

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from deal_voice.config import settings
from deal_voice.ingestion import indexer as indexer_module
from deal_voice.ingestion.indexer import EnterpriseIndexer
from deal_voice.retrieval import search as search_module
from deal_voice.retrieval.search import LocalSearchEngine


class FixedEmbeddings(Embeddings):
    """Deterministic vectors keep this integration test independent of model downloads."""

    def __init__(self, model_name):
        pass

    def _vector(self):
        return [1.0] + [0.0] * (settings.embedding_vector_size - 1)

    def embed_documents(self, texts):
        return [self._vector() for _ in texts]

    def embed_query(self, text):
        return self._vector()


class UnusedReranker:
    def __init__(self, model_name):
        pass


def test_stopped_database_and_artifacts_can_be_backed_up_and_restored(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(indexer_module, "HuggingFaceEmbeddings", FixedEmbeddings)
    monkeypatch.setattr(search_module, "HuggingFaceEmbeddings", FixedEmbeddings)
    live = tmp_path / "live"
    raw = live / "data" / "raw" / "upload"
    processed = live / "data" / "processed" / "upload"
    raw.mkdir(parents=True)
    processed.mkdir(parents=True)
    (raw / "call.wav").write_bytes(b"original audio bytes")
    (processed / "call_transcript.md").write_text("A: Bonjour.", encoding="utf-8")
    document = Document(
        page_content="A: Bonjour.",
        metadata={
            "source": "call.wav",
            "original_path": "data/raw/upload/call.wav",
            "transcript_markdown": "data/processed/upload/call_transcript.md",
        },
    )
    indexer = EnterpriseIndexer(
        db_path=str(live / "local_qdrant_db"), collection_name="backup_test"
    )
    try:
        indexer.index_documents([document])
        expected_ids = [row["document_id"] for row in indexer.list_documents()]
    finally:
        indexer.client.close()  # Copy only after the owner has stopped.

    backup = tmp_path / "backup"
    restored = tmp_path / "restored"
    for relative in ("local_qdrant_db", "data/raw", "data/processed"):
        shutil.copytree(live / relative, backup / relative)
        shutil.copytree(backup / relative, restored / relative)
    monkeypatch.chdir(restored)
    search = LocalSearchEngine(
        db_path=str(restored / "local_qdrant_db"),
        collection_name="backup_test",
        use_reranker=False,
    )
    try:
        assert [row["document_id"] for row in search.list_documents()] == expected_ids
        results = search.search("Bonjour", top_k=1)
        assert results[0].page_content == "A: Bonjour."
        assert (
            restored / results[0].metadata["original_path"]
        ).read_bytes() == b"original audio bytes"
        assert (restored / results[0].metadata["transcript_markdown"]).read_text(
            encoding="utf-8"
        ) == "A: Bonjour."
    finally:
        search.client.close()


def test_qdrant_same_named_documents_can_be_deleted_independently(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(indexer_module, "HuggingFaceEmbeddings", FixedEmbeddings)
    monkeypatch.setattr(search_module, "HuggingFaceEmbeddings", FixedEmbeddings)
    monkeypatch.setattr(search_module, "CrossEncoder", UnusedReranker)

    db_path = str(tmp_path / "qdrant")
    collection = "document_lifecycle"
    documents = [
        Document(
            page_content="The first homonymous document contains the first unique record.",
            metadata={"source": "same.pdf"},
        ),
        Document(
            page_content="The second homonymous document contains the second unique record.",
            metadata={"source": "same.pdf"},
        ),
    ]

    indexer = EnterpriseIndexer(db_path=db_path, collection_name=collection)
    try:
        indexer.index_documents(documents)
        first_id, second_id = [doc.metadata["document_id"] for doc in documents]
        assert first_id != second_id
        assert len(indexer.list_documents()) == 2
    finally:
        indexer.client.close()

    search = LocalSearchEngine(db_path=db_path, collection_name=collection)
    try:
        assert len(search.search("record", document_ids=[first_id], top_k=2)) == 1
        assert len(search.search("record", document_ids=[second_id], top_k=2)) == 1

        search.delete_documents([first_id])

        assert search.search("record", document_ids=[first_id], top_k=2) == []
        remaining = search.search("record", document_ids=[second_id], top_k=2)
        assert len(remaining) == 1
        assert remaining[0].metadata["source"] == "same.pdf"
        assert remaining[0].metadata["document_id"] == second_id
        assert [doc["document_id"] for doc in search.list_documents()] == [second_id]
    finally:
        search.client.close()


def test_qdrant_legacy_document_migration_preserves_searchability(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(indexer_module, "HuggingFaceEmbeddings", FixedEmbeddings)
    monkeypatch.setattr(search_module, "HuggingFaceEmbeddings", FixedEmbeddings)
    monkeypatch.setattr(search_module, "CrossEncoder", UnusedReranker)
    db_path = str(tmp_path / "legacy_qdrant")
    collection = "legacy_migration"
    indexer = EnterpriseIndexer(db_path=db_path, collection_name=collection)
    try:
        indexer.index_documents(
            [
                Document(
                    page_content="Legacy pricing record is $75 per seat.",
                    metadata={"source": "legacy.pdf"},
                )
            ]
        )
        points, _ = indexer.client.scroll(
            collection_name=collection, limit=100, with_payload=True
        )
        for point in points:
            payload = dict(point.payload)
            metadata = dict(payload["metadata"])
            metadata.pop("document_id")
            payload["metadata"] = metadata
            indexer.client.set_payload(
                collection_name=collection, payload=payload, points=[point.id]
            )
        for key in indexer.docstore.yield_keys():
            stored = indexer.docstore.mget([key])[0]
            stored.metadata.pop("document_id", None)
            indexer.docstore.mset([(key, stored)])
    finally:
        indexer.client.close()

    search = LocalSearchEngine(db_path=db_path, collection_name=collection)
    try:
        assert len(search.list_documents()) == 0
        assert search.migrate_legacy_documents() == 1
        migrated = search.list_documents()
        assert len(migrated) == 1
        document_id = migrated[0]["document_id"]
        results = search.search("pricing", document_ids=[document_id], top_k=1)
        assert len(results) == 1
        assert results[0].page_content == "Legacy pricing record is $75 per seat."
    finally:
        search.client.close()
