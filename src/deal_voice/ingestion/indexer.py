import json
import uuid
from pathlib import Path
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


def list_indexed_documents(client: QdrantClient, collection_name: str) -> list[dict[str, str]]:
    points, _ = client.scroll(
        collection_name=collection_name, limit=10000,
        with_payload=True, with_vectors=False,
    )
    documents: dict[str, dict[str, str]] = {}
    for point in points:
        metadata = (point.payload or {}).get("metadata", {})
        document_id = metadata.get("document_id")
        if document_id:
            documents[document_id] = {
                "document_id": document_id,
                "source": str(metadata.get("source", "inconnu")),
                "file_type": str(metadata.get("file_type", "document")),
            }
    return list(documents.values())


def migrate_legacy_documents(client: QdrantClient, docstore: Any, collection_name: str) -> int:
    """Assign stable IDs to pre-ID Qdrant points and matching parent records.

    Legacy chunks are grouped by source filename because older index versions
    did not persist an upload ID. Files with the same legacy source therefore
    cannot be distinguished retroactively.
    """
    grouped_points: dict[str, list[Any]] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            offset=offset,
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            metadata = payload.get("metadata", {})
            if metadata.get("document_id"):
                continue
            source = str(metadata.get("source") or "unknown")
            grouped_points.setdefault(source, []).append(point)
        if offset is None:
            break

    for source, points in grouped_points.items():
        document_id = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"deal-voice-legacy:{collection_name}:{source}")
        )
        parent_ids: set[str] = set()
        for point in points:
            payload = dict(point.payload or {})
            metadata = dict(payload.get("metadata", {}))
            metadata["document_id"] = document_id
            payload["metadata"] = metadata
            client.set_payload(
                collection_name=collection_name,
                payload=payload,
                points=[point.id],
            )
            if metadata.get("doc_id"):
                parent_ids.add(str(metadata["doc_id"]))
        for parent_id in parent_ids:
            stored = docstore.mget([parent_id])
            if stored and stored[0]:
                stored[0].metadata["document_id"] = document_id
                docstore.mset([(parent_id, stored[0])])
    return len(grouped_points)


def delete_indexed_documents(
    client: QdrantClient, docstore: Any, collection_name: str,
    document_ids: List[str],
) -> None:
    if not document_ids:
        return
    condition = models.FieldCondition(
        key="metadata.document_id", match=models.MatchAny(any=document_ids),
    )
    points, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=models.Filter(must=[condition]),
        limit=10000, with_payload=True, with_vectors=False,
    )
    parent_keys: set[str] = set()
    paths: set[Path] = set()
    for point in points:
        metadata = (point.payload or {}).get("metadata", {})
        if metadata.get("doc_id"):
            parent_keys.add(str(metadata["doc_id"]))
        for field in ("original_path", "transcript_markdown", "transcript_json"):
            if metadata.get(field):
                paths.add(Path(metadata[field]))
    for key in docstore.yield_keys():
        stored = docstore.mget([key])
        if stored and stored[0] and stored[0].metadata.get("document_id") in document_ids:
            parent_keys.add(key)
            for field in ("original_path", "transcript_markdown", "transcript_json"):
                if stored[0].metadata.get(field):
                    paths.add(Path(stored[0].metadata[field]))
    client.delete(
        collection_name=collection_name,
        points_selector=models.FilterSelector(filter=models.Filter(must=[condition])),
    )
    if parent_keys:
        docstore.mdelete(list(parent_keys))
    for path in paths:
        try:
            path.unlink(missing_ok=True)
            path.parent.rmdir()
        except OSError:
            pass


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
        self.parent_store = fs

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
        for document in documents:
            document.metadata.setdefault("document_id", str(uuid.uuid4()))
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

    def list_documents(self) -> list[dict[str, str]]:
        """Return unique indexed documents for the local management UI."""
        return list_indexed_documents(self.client, self.collection_name)

    def migrate_legacy_documents(self) -> int:
        """Migrate legacy entries without deleting indexed content."""
        return migrate_legacy_documents(
            self.client, self.docstore, self.collection_name
        )

    def delete_documents(self, document_ids: List[str]) -> None:
        """Remove vectors, parent records, originals, and transcript artifacts."""
        delete_indexed_documents(
            self.client, self.docstore, self.collection_name, document_ids
        )

    def delete_sources(self, sources: List[str]) -> None:
        """Compatibility helper; source names may match more than one document."""
        if not sources:
            return
        matched = [doc["document_id"] for doc in self.list_documents() if doc["source"] in sources]
        self.delete_documents(matched)
