from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient

from deal_voice.config import settings
from deal_voice.ingestion import indexer as module


class FixedEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[1.0] + [0.0] * (settings.embedding_vector_size - 1) for _ in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def test_injected_embeddings_do_not_reload_a_second_model(monkeypatch, tmp_path):
    def forbidden_model_load(**kwargs):
        raise AssertionError("An already loaded embedding model must be reused.")

    monkeypatch.setattr(module, "HuggingFaceEmbeddings", forbidden_model_load)
    embeddings = FixedEmbeddings()
    client = QdrantClient(":memory:")
    try:
        indexer = module.EnterpriseIndexer(
            db_path=str(tmp_path), client=client, embeddings=embeddings
        )
        indexer.index_documents(
            [
                Document(
                    page_content="Shared model record.",
                    metadata={"source": "sample.txt"},
                )
            ]
        )
        assert indexer.embeddings is embeddings
        assert not indexer._owns_client
        assert client.count(settings.collection_name, exact=True).count == 1
    finally:
        client.close()
