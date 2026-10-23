import json
from typing import List
from sentence_transformers import CrossEncoder
from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore, EncoderBackedStore
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from deal_voice.config import settings
from deal_voice.ingestion.qdrant_schema import ensure_collection_schema
from deal_voice.ingestion.indexer import (
    delete_indexed_documents,
    list_indexed_documents,
    migrate_legacy_documents as migrate_legacy_index_entries,
)


class LocalSearchEngine:

    def __init__(
        self,
        db_path: str = settings.qdrant_storage_path,
        collection_name: str = settings.collection_name,
    ):
        self.collection_name = collection_name
        self.client = QdrantClient(path=db_path)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model_name
        )
        self.reranker = CrossEncoder(settings.reranker_model_name)
        ensure_collection_schema(
            self.client,
            self.collection_name,
            settings.embedding_vector_size,
            db_path,
        )

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

        fs = LocalFileStore(f"{db_path}/parent_docstore")
        self.docstore = EncoderBackedStore(
            store=fs,
            key_encoder=lambda x: x,
            value_serializer=lambda doc: json.dumps(
                doc.model_dump() if hasattr(doc, "model_dump") else doc.dict()
            ).encode("utf-8"),
            value_deserializer=lambda bytes_data: Document(
                **json.loads(bytes_data.decode("utf-8"))
            ),
        )

        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vector_store,
            docstore=self.docstore,
            child_splitter=RecursiveCharacterTextSplitter(
                chunk_size=settings.child_chunk_size,
                chunk_overlap=settings.child_chunk_overlap,
            ),
        )

    def search(
        self,
        query: str,
        user_role: str | None = None,
        source: str | list[str] | None = None,
        document_ids: str | list[str] | None = None,
        top_k: int = 5,
    ) -> List[Document]:
        search_kwargs: dict[str, object] = {"k": 30}
        conditions = []
        if user_role:
            conditions.append(
                models.FieldCondition(
                    key="metadata.allowed_roles",
                    match=models.MatchAny(any=[user_role]),
                )
            )
        if source:
            source_values = [source] if isinstance(source, str) else source
            conditions.append(
                models.FieldCondition(
                    key="metadata.source",
                    match=models.MatchAny(any=source_values),
                )
            )
        if document_ids:
            ids = [document_ids] if isinstance(document_ids, str) else document_ids
            conditions.append(
                models.FieldCondition(
                    key="metadata.document_id",
                    match=models.MatchAny(any=ids),
                )
            )
        if conditions:
            search_kwargs["filter"] = models.Filter(
                must=conditions
            )

        self.retriever.search_kwargs = search_kwargs
        initial_parent_results = self.retriever.invoke(query)

        if len(initial_parent_results) <= top_k:
            return initial_parent_results

        sentence_pairs = [
            [query, doc.page_content] for doc in initial_parent_results
        ]
        relevance_scores = self.reranker.predict(sentence_pairs)

        for score, doc in zip(relevance_scores, initial_parent_results):
            doc.metadata["rerank_score"] = float(score)

        reranked_results = sorted(
            initial_parent_results,
            key=lambda x: x.metadata["rerank_score"],
            reverse=True,
        )
        return reranked_results[:top_k]

    def list_documents(self) -> list[dict[str, str]]:
        return list_indexed_documents(self.client, self.collection_name)

    def delete_documents(self, document_ids: list[str]) -> None:
        delete_indexed_documents(
            self.client, self.docstore, self.collection_name, document_ids
        )

    def migrate_legacy_documents(self) -> int:
        return migrate_legacy_index_entries(
            self.client, self.docstore, self.collection_name
        )
