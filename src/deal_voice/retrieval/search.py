import json
import re
import unicodedata
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

        # Global table questions need one complete, column-matching table per
        # selected source. Sending every table from a PDF confuses the LLM
        # when different sections contain purchase, retail, wholesale, and
        # sales values for the same products.
        if self._is_global_table_query(query):
            table_candidates = []
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
                table_candidates.append(stored)
            selected_tables = self._select_relevant_tables(query, table_candidates)
            if selected_tables:
                initial_parent_results = selected_tables

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

    @staticmethod
    def _normalize_text(value: str) -> str:
        decomposed = unicodedata.normalize("NFKD", value.casefold())
        without_accents = "".join(
            char for char in decomposed if not unicodedata.combining(char)
        )
        return " ".join(re.findall(r"[a-z0-9]+", without_accents))

    @classmethod
    def _table_headers(cls, document: Document) -> list[str]:
        lines = document.page_content.splitlines()
        for index, line in enumerate(lines):
            if (
                index > 0
                and "|" in line
                and cls._is_table_separator(line)
            ):
                return [
                    cls._normalize_text(cell)
                    for cell in lines[index - 1].strip().strip("|").split("|")
                ]
        return []

    @staticmethod
    def _is_table_separator(line: str) -> bool:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        return len(cells) >= 2 and all(
            re.fullmatch(r":?-{3,}:?", cell) is not None for cell in cells
        )

    @classmethod
    def _query_table_fields(cls, query: str) -> set[str]:
        normalized = cls._normalize_text(query)
        field_aliases = {
            "retail_price": (
                "prix detail", "prix de detail", "prix au detail",
                "prix de vente au detail", "detail", "plus cher", "moins cher",
            ),
            "wholesale_price": (
                "prix gros", "prix de gros", "prix de vente en gros", "gros",
            ),
            "purchase_price": (
                "prix achat", "prix d achat", "cout achat", "cout d achat",
            ),
            "retail_margin": ("marge detail", "marge de detail"),
            "wholesale_margin": ("marge gros", "marge de gros"),
            "initial_stock": ("stock initial",),
            "remaining_stock": ("stock restant", "stock final"),
            "sold_quantity": ("quantite vendue", "quantites vendues"),
            "purchased_quantity": ("quantite achetee", "quantites achetees"),
            "supplier": ("fournisseur", "fournisseurs"),
            "category": ("categorie", "categories"),
            "product": ("produit", "produits"),
        }
        return {
            field
            for field, aliases in field_aliases.items()
            if any(alias in normalized for alias in aliases)
        }

    @classmethod
    def _select_relevant_tables(
        cls, query: str, candidates: list[Document]
    ) -> list[Document]:
        if not candidates:
            return []

        fields = cls._query_table_fields(query)
        query_tokens = set(cls._normalize_text(query).split())
        ignored_tokens = {
            "a", "au", "aux", "avec", "de", "des", "du", "en", "et",
            "la", "le", "les", "leur", "leurs", "quel", "quelle", "quels",
            "quelles", "que", "qui", "dans", "est", "sont", "sur", "pour",
            "donne", "liste", "tous", "toutes", "chaque", "maximum", "minimum",
            "plus", "moins", "cher", "chere", "all", "every", "the", "with",
            "from", "what", "which", "show", "list", "products", "product",
            "produit", "produits", "prix", "price", "detail", "detaill", "cout",
        }
        query_tokens.difference_update(ignored_tokens)

        def score(document: Document) -> tuple[int, int, int]:
            headers = cls._table_headers(document)
            header_text = " ".join(headers)
            title = cls._normalize_text(str(document.metadata.get("table_title", "")))
            matched_fields = sum(
                1
                for field in fields
                if cls._field_matches_header(field, headers)
            )
            matched_terms = len(
                query_tokens & set((header_text + " " + title).split())
            )
            row_count = int(document.metadata.get("table_row_count", 0) or 0)
            # For global questions, prefer the broadest table when its
            # requested columns are equally relevant.
            return matched_fields, matched_terms, row_count

        best_by_document: dict[str, tuple[tuple[int, int, int], Document]] = {}
        for document in candidates:
            identity = str(document.metadata.get("document_id", document.metadata.get("source", "")))
            candidate_score = score(document)
            if not any(candidate_score[:2]):
                continue
            current = best_by_document.get(identity)
            if current is None or candidate_score > current[0]:
                best_by_document[identity] = (candidate_score, document)
        return [item[1] for item in best_by_document.values()]

    @classmethod
    def _field_matches_header(cls, field: str, headers: list[str]) -> bool:
        normalized_headers = set(headers)
        header_text = " ".join(headers)
        aliases = {
            "retail_price": ("prix detail", "detail"),
            "wholesale_price": ("prix gros", "gros"),
            "purchase_price": ("prix achat", "achat"),
            "retail_margin": ("marge detail",),
            "wholesale_margin": ("marge gros",),
            "initial_stock": ("stock initial",),
            "remaining_stock": ("stock restant", "stock final"),
            "sold_quantity": ("quantite vendue", "ventes"),
            "purchased_quantity": ("quantite achetee", "achats"),
            "supplier": ("fournisseur",),
            "category": ("categorie",),
            "product": ("produit",),
        }
        return any(
            alias in normalized_headers or alias in header_text
            for alias in aliases.get(field, ())
        )

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
