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