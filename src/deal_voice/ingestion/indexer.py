import json
from typing import Any
from typing import List
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import EncoderBackedStore, LocalFileStore
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from deal_voice.config import settings
from deal_voice.ingestion.qdrant_schema import ensure_collection_schema


class EnterpriseIndexer:

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

        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.parent_chunk_size,
            chunk_overlap=settings.parent_chunk_overlap,
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.child_chunk_size,
            chunk_overlap=settings.child_chunk_overlap,
        )

        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vector_store,
            docstore=self.docstore,
            child_splitter=self.child_splitter,
            parent_splitter=self.parent_splitter,
        )

    def index_documents(self, documents: List[Document]):
        audio_documents = []
        for document in documents:
            if document.metadata.get("file_type") == "audio":
                audio_documents.extend(self._audio_parent_documents(document))
        other_documents = [
            document
            for document in documents
            if document.metadata.get("file_type") != "audio"
        ]

        if other_documents:
            self.retriever.add_documents(other_documents, ids=None)

        if audio_documents:
            audio_retriever = ParentDocumentRetriever(
                vectorstore=self.vector_store,
                docstore=self.docstore,
                child_splitter=RecursiveCharacterTextSplitter(
                    chunk_size=settings.child_chunk_size,
                    chunk_overlap=0,
                ),
                parent_splitter=RecursiveCharacterTextSplitter(
                    chunk_size=settings.parent_chunk_size,
                    chunk_overlap=0,
                ),
            )
            audio_retriever.add_documents(audio_documents, ids=None)

    @staticmethod
    def _audio_parent_documents(document: Document) -> List[Document]:
        """Create parent documents without splitting across speaker turns."""
        turns = document.metadata.get("turns")
        if not isinstance(turns, list) or not turns:
            return [document]

        parents: List[Document] = []
        current: list[dict[str, Any]] = []
        current_length = 0
        max_length = settings.parent_chunk_size

        def add_parent(turn_group: list[dict[str, Any]]) -> None:
            if not turn_group:
                return
            content = "\n\n".join(
                f"**{turn.get('speaker', 'Unknown')}**: {turn.get('text', '')}"
                for turn in turn_group
            )
            metadata = dict(document.metadata)
            metadata.pop("turns", None)
            metadata.update(
                {
                    "speakers": sorted(
                        {str(turn.get("speaker", "Unknown")) for turn in turn_group}
                    ),
                    "start_time": turn_group[0].get("start_time"),
                    "end_time": turn_group[-1].get("end_time"),
                }
            )
            parents.append(Document(page_content=content, metadata=metadata))

        for turn in turns:
            text = str(turn.get("text", "")).strip()
            if not text:
                continue
            turn_length = len(text)
            if current and current_length + turn_length > max_length:
                add_parent(current)
                current = []
                current_length = 0
            current.append(turn)
            current_length += turn_length
        add_parent(current)
        return parents

    def delete_sources(self, sources: List[str]) -> None:
        """Delete indexed vectors belonging to the given source names."""
        if not sources:
            return
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.source",
                            match=models.MatchAny(any=sources),
                        )
                    ]
                )
            ),
        )