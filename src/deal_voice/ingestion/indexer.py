import json
import re
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
        client: QdrantClient | None = None,
    ):
        self.collection_name = collection_name
        self.client = client or QdrantClient(path=db_path)
        self._owns_client = client is None
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
        table_documents = []
        text_documents = []
        for document in documents:
            if document.metadata.get("file_type") == "audio":
                audio_documents.extend(self._audio_parent_documents(document))
            elif document.metadata.get("file_type") == "pdf":
                pdf_text, pdf_tables = self._split_pdf_tables(document)
                text_documents.extend(pdf_text)
                table_documents.extend(pdf_tables)
            else:
                text_documents.append(document)

        if text_documents:
            self.retriever.add_documents(text_documents, ids=None)

        if table_documents:
            self._index_table_documents(table_documents)

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
    def _is_markdown_table_separator(line: str) -> bool:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        return len(cells) >= 2 and all(
            re.fullmatch(r":?-{3,}:?", cell) is not None for cell in cells
        )

    @classmethod
    def _split_pdf_tables(
        cls, document: Document
    ) -> tuple[List[Document], List[Document]]:
        """Keep each exported Markdown table as a complete parent document."""
        lines = document.page_content.splitlines()
        regular_lines: list[str] = []
        tables: list[Document] = []
        current_heading = ""
        index = 0
        table_index = 0

        while index < len(lines):
            is_table_start = (
                index + 1 < len(lines)
                and "|" in lines[index]
                and cls._is_markdown_table_separator(lines[index + 1])
            )
            if not is_table_start:
                line = lines[index]
                regular_lines.append(line)
                if line.lstrip().startswith("#"):
                    current_heading = line.strip()
                index += 1
                continue

            table_start = index
            index += 2  # Header and Markdown separator.
            while index < len(lines) and "|" in lines[index]:
                index += 1

            table_lines = lines[table_start:index]
            table_index += 1
            metadata = dict(document.metadata)
            document_id = metadata.setdefault("document_id", str(uuid.uuid4()))
            metadata.update(
                {
                    "content_kind": "table",
                    "table_id": (
                        f"{document_id}::table::{table_index:04d}"
                    ),
                    "table_title": current_heading,
                    "table_row_count": max(0, len(table_lines) - 2),
                    "table_complete": True,
                }
            )
            content = "\n".join(
                part for part in (current_heading, *table_lines) if part
            )
            tables.append(Document(page_content=content, metadata=metadata))

        regular_content = "\n".join(regular_lines).strip()
        text_documents = []
        if regular_content:
            text_metadata = dict(document.metadata)
            text_metadata["content_kind"] = "text"
            text_documents.append(
                Document(page_content=regular_content, metadata=text_metadata)
            )
        if not tables:
            return [document], []
        return text_documents, tables

    def _index_table_documents(self, table_documents: List[Document]) -> None:
        """Index row-group children while storing each full table as its parent."""
        parents_to_store = []
        children_to_index = []
        for table in table_documents:
            parent_id = str(uuid.uuid4())
            parents_to_store.append((parent_id, table))
            lines = table.page_content.splitlines()
            separator_index = next(
                (
                    index
                    for index, line in enumerate(lines[1:], start=1)
                    if self._is_markdown_table_separator(line)
                ),
                None,
            )
            if separator_index is None or separator_index == 0:
                continue

            prefix = lines[: separator_index - 1]
            header = lines[separator_index - 1 : separator_index + 1]
            rows = lines[separator_index + 1 :]
            row_groups: list[tuple[int, list[str]]] = []
            group_start = 1
            current_rows: list[str] = []
            for row_number, row in enumerate(rows, start=1):
                candidate = prefix + header + current_rows + [row]
                if (
                    current_rows
                    and len("\n".join(candidate)) > settings.child_chunk_size
                ):
                    row_groups.append((group_start, current_rows))
                    group_start = row_number
                    current_rows = []
                current_rows.append(row)
            if current_rows:
                row_groups.append((group_start, current_rows))

            for group_index, (first_row, row_group) in enumerate(row_groups, start=1):
                child_metadata = dict(table.metadata)
                child_metadata.update(
                    {
                        "doc_id": parent_id,
                        "table_complete": False,
                        "table_chunk_index": group_index,
                        "table_row_start": first_row,
                        "table_row_end": first_row + len(row_group) - 1,
                    }
                )
                children_to_index.append(
                    Document(
                        page_content="\n".join(prefix + header + row_group),
                        metadata=child_metadata,
                    )
                )

        if parents_to_store:
            self.docstore.mset(parents_to_store)
        if children_to_index:
            self.vector_store.add_documents(children_to_index)

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
