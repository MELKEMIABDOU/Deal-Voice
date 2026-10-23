from types import SimpleNamespace

from langchain_core.documents import Document
from deal_voice.ingestion.indexer import (
    EnterpriseIndexer,
    delete_indexed_documents,
    migrate_legacy_documents,
)
from deal_voice.retrieval.search import LocalSearchEngine


class CapturingRetriever:
    def __init__(self):
        self.documents = []

    def add_documents(self, documents, ids=None):
        self.documents.extend(documents)


def test_indexing_assigns_unique_ids_to_same_named_documents():
    indexer = EnterpriseIndexer.__new__(EnterpriseIndexer)
    indexer.retriever = CapturingRetriever()
    docs = [
        Document(page_content="alpha", metadata={"source": "same.pdf"}),
        Document(page_content="beta", metadata={"source": "same.pdf"}),
    ]

    indexer.index_documents(docs)

    ids = [doc.metadata["document_id"] for doc in docs]
    assert len(set(ids)) == 2
    assert all(ids)


def test_search_filters_by_document_id():
    engine = LocalSearchEngine.__new__(LocalSearchEngine)
    returned = Document(page_content="found", metadata={"document_id": "doc-a"})

    class Retriever:
        search_kwargs = {}

        def invoke(self, query):
            return [returned]

    engine.retriever = Retriever()

    assert engine.search("question", document_ids=["doc-a"], top_k=1) == [returned]
    condition = engine.retriever.search_kwargs["filter"].must[0]
    assert condition.key == "metadata.document_id"
    assert condition.match.any == ["doc-a"]


class FakeClient:
    def __init__(self, points):
        self.points = points
        self.deleted = False

    def scroll(self, **kwargs):
        return self.points, None

    def delete(self, **kwargs):
        self.deleted = True


class FakeDocstore:
    def __init__(self, documents):
        self.documents = documents
        self.removed = []

    def yield_keys(self):
        return iter(self.documents)

    def mget(self, keys):
        return [self.documents.get(key) for key in keys]

    def mdelete(self, keys):
        self.removed.extend(keys)

    def mset(self, pairs):
        self.documents.update(pairs)


def test_delete_removes_vectors_parent_records_and_file_artifacts(tmp_path):
    original = tmp_path / "raw" / "upload.pdf"
    transcript = tmp_path / "processed" / "transcript.md"
    original.parent.mkdir()
    transcript.parent.mkdir()
    original.write_text("pdf", encoding="utf-8")
    transcript.write_text("transcript", encoding="utf-8")
    metadata = {
        "document_id": "doc-a",
        "doc_id": "parent-a",
        "original_path": str(original),
        "transcript_markdown": str(transcript),
    }
    point = SimpleNamespace(payload={"metadata": metadata})
    client = FakeClient([point])
    docstore = FakeDocstore(
        {"parent-a": Document(page_content="parent", metadata=metadata)}
    )

    delete_indexed_documents(client, docstore, "collection", ["doc-a"])

    assert client.deleted
    assert "parent-a" in docstore.removed
    assert not original.exists()
    assert not transcript.exists()
    assert not original.parent.exists()
    assert not transcript.parent.exists()


def test_legacy_migration_assigns_one_stable_id_per_source():
    class LegacyClient(FakeClient):
        def set_payload(self, collection_name, payload, points):
            point = next(point for point in self.points if point.id == points[0])
            point.payload = payload

    points = [
        SimpleNamespace(
            id=f"child-{index}",
            payload={"metadata": {"source": "old.pdf", "doc_id": f"parent-{index}"}},
        )
        for index in range(2)
    ]
    client = LegacyClient(points)
    docstore = FakeDocstore(
        {
            f"parent-{index}": Document(
                page_content=f"parent {index}", metadata={"source": "old.pdf"}
            )
            for index in range(2)
        }
    )

    assert migrate_legacy_documents(client, docstore, "collection") == 1
    migrated_ids = [point.payload["metadata"]["document_id"] for point in points]
    assert len(set(migrated_ids)) == 1
    assert all(doc.metadata["document_id"] == migrated_ids[0] for doc in docstore.documents.values())
    assert migrate_legacy_documents(client, docstore, "collection") == 0


def test_upload_count_and_size_limits_reject_before_writing(monkeypatch):
    import streamlit_app

    monkeypatch.setattr(streamlit_app.settings, "max_upload_files", 1)
    monkeypatch.setattr(streamlit_app.settings, "max_upload_size_mb", 1)

    class Upload:
        name = "sample.pdf"
        size = 1

    import pytest

    with pytest.raises(ValueError, match="fichiers par import"):
        streamlit_app.index_uploaded_files([Upload(), Upload()], "base")

    monkeypatch.setattr(Upload, "size", 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="limite de 1 Mo"):
        streamlit_app.index_uploaded_files([Upload()], "base")


def test_audio_duration_limit_rejects_and_removes_temporary_upload(
    monkeypatch, tmp_path
):
    import sys
    import streamlit_app

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(streamlit_app.settings, "max_upload_files", 1)
    monkeypatch.setattr(streamlit_app.settings, "max_upload_size_mb", 1)
    monkeypatch.setattr(streamlit_app.settings, "max_audio_duration_minutes", 1)
    monkeypatch.setattr(streamlit_app.settings, "hf_token", "test-token")
    monkeypatch.setitem(
        sys.modules,
        "torchaudio",
        SimpleNamespace(info=lambda path: SimpleNamespace(num_frames=7200, sample_rate=60)),
    )

    class Upload:
        name = "call.wav"
        size = 4

        @staticmethod
        def getbuffer():
            return b"fake"

    import pytest

    with pytest.raises(ValueError, match="durée maximale"):
        streamlit_app.index_uploaded_files([Upload()], "base")

    assert not list((tmp_path / "data" / "raw").glob("*"))


def test_failed_indexing_rolls_back_upload_files_and_partial_index(
    monkeypatch, tmp_path
):
    import streamlit_app
    import pytest

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(streamlit_app.settings, "max_upload_files", 1)
    monkeypatch.setattr(streamlit_app.settings, "max_upload_size_mb", 1)
    monkeypatch.setattr(
        streamlit_app.EnterpriseDocumentParser,
        "parse_pdf",
        lambda self, file_path, allowed_roles=None: [
            Document(page_content="parsed", metadata={"source": "file.pdf"})
        ],
    )
    cleanup_calls = []

    class FailingIndexer:
        class Client:
            @staticmethod
            def close():
                pass

        client = Client()

        def index_documents(self, documents):
            raise RuntimeError("simulated Qdrant failure")

        def delete_documents(self, document_ids):
            cleanup_calls.extend(document_ids)

    monkeypatch.setattr(streamlit_app, "EnterpriseIndexer", FailingIndexer)

    class Upload:
        name = "file.pdf"
        size = 4

        @staticmethod
        def getbuffer():
            return b"pdf"

    with pytest.raises(RuntimeError, match="simulated Qdrant failure"):
        streamlit_app.index_uploaded_files([Upload()], "base")

    assert len(cleanup_calls) == 1
    assert not list((tmp_path / "data" / "raw").glob("*"))
