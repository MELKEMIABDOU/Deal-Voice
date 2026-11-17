"""Bounded semantic retrieval over child vectors and their complete parents."""

import hashlib
import json
import math

from langchain_classic.storage import EncoderBackedStore, LocalFileStore
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import models
from sentence_transformers import CrossEncoder

from deal_voice.config import settings
from deal_voice.evidence.tabular import normalized_label, parse_markdown_table
from deal_voice.ingestion.indexer import (
    delete_indexed_documents,
    list_indexed_documents,
)
from deal_voice.ingestion.indexer import (
    migrate_legacy_documents as migrate_legacy_index_entries,
)
from deal_voice.ingestion.qdrant_schema import (
    create_qdrant_client,
    ensure_collection_schema,
)


class LocalSearchEngine:
    MAX_CANDIDATES = 200
    RERANK_DOMINANCE_RATIO = 10.0

    def __init__(
        self,
        db_path: str = settings.qdrant_storage_path,
        collection_name: str = settings.collection_name,
        use_reranker: bool = True,
        candidate_pool_size: int = 30,
        rerank_batch_size: int = 1,
        qdrant_url: str | None = settings.qdrant_url,
        qdrant_api_key: str | None = settings.qdrant_api_key,
        parent_docstore_path: str | None = None,
    ):
        if (
            type(candidate_pool_size) is not int
            or not 1 <= candidate_pool_size <= self.MAX_CANDIDATES
        ):
            raise ValueError("candidate_pool_size must be between 1 and 200")
        if type(rerank_batch_size) is not int or not 1 <= rerank_batch_size <= 32:
            raise ValueError("rerank_batch_size must be between 1 and 32")
        self.collection_name = collection_name
        self.candidate_pool_size = candidate_pool_size
        self.rerank_batch_size = rerank_batch_size
        self.client = create_qdrant_client(
            path=db_path, url=qdrant_url, api_key=qdrant_api_key
        )
        parent_docstore_path = parent_docstore_path or (
            settings.parent_docstore_path
            if db_path == settings.qdrant_storage_path
            else f"{db_path}/parent_docstore"
        )
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=settings.embedding_model_name
            )
            self.reranker = (
                CrossEncoder(settings.reranker_model_name) if use_reranker else None
            )
            ensure_collection_schema(
                self.client,
                self.collection_name,
                settings.embedding_vector_size,
                parent_docstore_path,
            )
            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
            )
            self.docstore = EncoderBackedStore(
                store=LocalFileStore(parent_docstore_path),
                key_encoder=lambda key: key,
                value_serializer=lambda doc: json.dumps(doc.model_dump()).encode(
                    "utf-8"
                ),
                value_deserializer=lambda data: Document(
                    **json.loads(data.decode("utf-8"))
                ),
            )
        except Exception:
            self.client.close()
            raise

    def search(
        self,
        query: str,
        user_role: str | None = None,
        source: str | list[str] | None = None,
        document_ids: str | list[str] | None = None,
        top_k: int = 5,
    ) -> list[Document]:
        if type(top_k) is not int or not 1 <= top_k <= self.MAX_CANDIDATES:
            raise ValueError("top_k must be between 1 and 200")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        sources = [source] if isinstance(source, str) else source
        identifiers = [document_ids] if isinstance(document_ids, str) else document_ids
        conditions = []
        for key, values in (
            ("allowed_roles", [user_role] if user_role else None),
            ("source", sources),
            ("document_id", identifiers),
        ):
            if values is not None:
                if not values:
                    return []
                conditions.append(
                    models.FieldCondition(
                        key=f"metadata.{key}", match=models.MatchAny(any=values)
                    )
                )
        # All request state stays local: simultaneous searches cannot overwrite
        # another request's document or role filter.
        kwargs = {"k": max(top_k, self.candidate_pool_size)}
        if conditions:
            kwargs["filter"] = models.Filter(must=conditions)
        children = self.vector_store.similarity_search(query, **kwargs)
        semantic_parent_keys = list(
            dict.fromkeys(
                str(child.metadata["doc_id"])
                for child in children
                if child.metadata.get("doc_id")
            )
        )
        header_parent_keys = self._matching_table_parent_keys(query, conditions)
        parent_keys = list(dict.fromkeys([*semantic_parent_keys, *header_parent_keys]))
        parents = dict(zip(parent_keys, self.docstore.mget(parent_keys), strict=True))
        candidates = []
        seen = set()

        def append_candidate(document: Document | None, parent_key=None) -> None:
            if document is None:
                return
            metadata = dict(document.metadata)
            # Recheck the returned parent as well as the child-vector filter.
            if user_role and user_role not in metadata.get("allowed_roles", []):
                return
            if sources is not None and metadata.get("source") not in sources:
                return
            if (
                identifiers is not None
                and metadata.get("document_id") not in identifiers
            ):
                return
            if parent_key:
                metadata["parent_id"] = str(parent_key)
            identity = self._evidence_identity(document, metadata)
            if identity in seen:
                return
            seen.add(identity)
            candidates.append(
                Document(page_content=document.page_content, metadata=metadata)
            )

        for child in children:
            parent_key = child.metadata.get("doc_id")
            document = parents.get(str(parent_key)) if parent_key else child
            append_candidate(document, parent_key)
        for parent_key in header_parent_keys:
            append_candidate(parents.get(parent_key), parent_key)

        # Ordering still matters when every candidate fits top_k: generation
        # packs them in rank order and may have room for only the first ones.
        if self.reranker is not None and len(candidates) > 1:
            scores = self.reranker.predict(
                [[query, document.page_content] for document in candidates],
                batch_size=self.rerank_batch_size,
                show_progress_bar=False,
            )
            for document, score in zip(candidates, scores, strict=True):
                score = float(score)
                if not math.isfinite(score):
                    raise ValueError("The reranker returned a non-finite score")
                document.metadata["rerank_score"] = score
            candidates.sort(
                key=lambda document: document.metadata["rerank_score"], reverse=True
            )
        self._promote_exact_table_header(query, candidates)
        self._promote_overlapping_audio_context(candidates)
        candidates = self._filter_dominated_candidates(candidates)
        return candidates[:top_k]

    @staticmethod
    def _promote_overlapping_audio_context(candidates: list[Document]) -> None:
        """Keep overlapping turns from one recording next to each other.

        Cross-encoders can rank the passage naming a person above the overlapping
        passage containing that person's statement. Keeping the pair adjacent
        prevents unrelated documents from consuming the generation budget between
        them. Exact table-header promotion remains intact because the existing
        first candidate is never displaced.
        """
        ordered: list[Document] = []
        remaining = list(candidates)
        while remaining:
            anchor = remaining.pop(0)
            ordered.append(anchor)
            metadata = anchor.metadata
            start = metadata.get("start_time")
            end = metadata.get("end_time")
            source = metadata.get("source")
            if (
                not source
                or not isinstance(start, (int, float))
                or not isinstance(end, (int, float))
            ):
                continue
            neighbours = []
            others = []
            for document in remaining:
                candidate = document.metadata
                candidate_start = candidate.get("start_time")
                candidate_end = candidate.get("end_time")
                overlaps = (
                    candidate.get("source") == source
                    and isinstance(candidate_start, (int, float))
                    and isinstance(candidate_end, (int, float))
                    and candidate_start <= end
                    and candidate_end >= start
                )
                (neighbours if overlaps else others).append(document)
            ordered.extend(neighbours)
            remaining = others
        candidates[:] = ordered

    @classmethod
    def _filter_dominated_candidates(cls, candidates: list[Document]) -> list[Document]:
        """Remove obvious reranker noise only after a clear score separation.

        Comparable scores are left untouched. Exact table-header matches survive
        because they carry deterministic query evidence independent of the model
        score. This avoids a fixed absolute threshold across reranker versions.
        """
        scored = [
            (index, float(document.metadata["rerank_score"]))
            for index, document in enumerate(candidates)
            if document.metadata.get("rerank_score") is not None
        ]
        if len(scored) != len(candidates) or len(scored) < 2:
            return candidates
        ordered_scores = sorted((score for _, score in scored), reverse=True)
        best, second = ordered_scores[:2]
        if best <= 0 or second > best / cls.RERANK_DOMINANCE_RATIO:
            return candidates
        cutoff = best / cls.RERANK_DOMINANCE_RATIO
        return [
            document
            for document in candidates
            if float(document.metadata["rerank_score"]) >= cutoff
            or document.metadata.get("header_match_score") is not None
        ]

    def _matching_table_parent_keys(
        self, query: str, conditions: list[models.FieldCondition]
    ) -> list[str]:
        """Recover exact table-column matches missed by dense retrieval."""
        client = getattr(self, "client", None)
        if client is None:
            return []
        header_condition = models.FieldCondition(
            key="metadata.table_chunk_kind",
            match=models.MatchValue(value="header"),
        )
        points, _ = client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(must=[*conditions, header_condition]),
            limit=self.MAX_CANDIDATES,
            with_payload=True,
            with_vectors=False,
        )
        matches = []
        for point in points:
            metadata = (point.payload or {}).get("metadata", {})
            parent_key = metadata.get("doc_id")
            columns = metadata.get("table_columns")
            if not parent_key or not isinstance(columns, list):
                continue
            score = self._header_score(query, [str(column) for column in columns])
            if score[0] or score[1] >= 2:
                matches.append((score, str(parent_key)))
        matches.sort(reverse=True)
        return list(dict.fromkeys(parent_key for _, parent_key in matches))

    @staticmethod
    def _header_score(
        query: str, headers: list[str] | tuple[str, ...]
    ) -> tuple[int, int]:
        query_tokens = normalized_label(query).split()
        query_terms = set(query_tokens)
        query_pairs = set(zip(query_tokens, query_tokens[1:], strict=False))
        header_terms: set[str] = set()
        header_pairs: set[tuple[str, str]] = set()
        for header in headers:
            tokens = normalized_label(header).split()
            header_terms.update(tokens)
            header_pairs.update(zip(tokens, tokens[1:], strict=False))
        return len(query_pairs & header_pairs), len(query_terms & header_terms)

    @staticmethod
    def _promote_exact_table_header(query: str, candidates: list[Document]) -> None:
        """Put the strongest exact column match first so it reaches generation.

        Semantic rerankers can prefer related prose over the table containing
        the exact requested measure. Promotion requires either an exact header
        phrase of at least two words or at least two matching header terms.
        """
        best_index, best_score = None, (0, 0)
        for index, document in enumerate(candidates):
            if document.metadata.get("content_kind") != "table":
                continue
            try:
                headers = parse_markdown_table(document.page_content).headers
            except ValueError:
                continue
            score = LocalSearchEngine._header_score(query, headers)
            if score > best_score:
                best_index, best_score = index, score
        if best_index is None or not (best_score[0] or best_score[1] >= 2):
            return
        document = candidates.pop(best_index)
        document.metadata["header_match_score"] = {
            "phrases": best_score[0],
            "terms": best_score[1],
        }
        candidates.insert(0, document)

    @staticmethod
    def _evidence_identity(document: Document, metadata: dict) -> str:
        """Deduplicate passages, not all tables/sections sharing a filename."""
        locator = {
            key: metadata.get(key)
            for key in (
                "document_id",
                "source",
                "parent_id",
                "table_id",
                "_id",
                "page",
                "page_number",
                "section",
                "start_time",
                "end_time",
            )
        }
        identity = (
            json.dumps(locator, sort_keys=True, default=str) + document.page_content
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

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
