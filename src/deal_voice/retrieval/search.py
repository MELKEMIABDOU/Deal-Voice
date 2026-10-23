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

        # Table questions often require a global scan (all rows, maximum,
        # minimum, ranking). Semantic retrieval can otherwise select a
        # smaller table that happens to contain the queried words. When a
        # complete table is retrieved, include the other complete tables from
        # the same document so the generator can compare the full evidence.
        if self._is_global_table_query(query):
            existing_ids = {doc.metadata.get("table_id") for doc in initial_parent_results}
            for key in self.docstore.yield_keys():
                stored = self.docstore.mget([key])[0]
                if not stored or stored.metadata.get("content_kind") != "table":
                    continue
                if not stored.metadata.get("table_complete"):
                    continue
                if user_role and user_role not in stored.metadata.get("allowed_roles", []):
                    continue
                if source and stored.metadata.get("source") not in (
                    [source] if isinstance(source, str) else source
                ):
                    continue
                if document_ids and stored.metadata.get("document_id") not in (
                    [document_ids] if isinstance(document_ids, str) else document_ids
                ):
                    continue
                table_id = stored.metadata.get("table_id")
                if table_id not in existing_ids:
                    initial_parent_results.append(stored)
                    existing_ids.add(table_id)

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
        if self._is_global_table_query(query):
            table_results = [
                doc for doc in reranked_results
                if doc.metadata.get("content_kind") == "table"
                and doc.metadata.get("table_complete")
            ]
            other_results = [
                doc for doc in reranked_results
                if doc.metadata.get("content_kind") != "table"
            ]
            return (table_results + other_results)[: max(top_k, len(table_results))]
        return reranked_results[:top_k]

    @staticmethod
    def _is_global_table_query(query: str) -> bool:
        terms = (
            "tous", "toutes", "chaque", "maximum", "minimum", "plus cher",
            "moins cher", "classe", "classer", "comparer", "total", "liste",
            "all", "every", "maximum", "minimum", "highest", "lowest", "rank",
        )
        normalized = query.lower()
        return any(term in normalized for term in terms)

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
