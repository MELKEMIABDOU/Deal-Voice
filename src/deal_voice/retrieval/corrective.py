"""One bounded corrective retrieval pass with deterministic coverage checks."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from deal_voice.evidence.tabular import parse_markdown_table
from deal_voice.retrieval.search import LocalSearchEngine


@dataclass(frozen=True)
class CoverageAssessment:
    """Evidence coverage before answer generation."""

    status: str
    missing_information: tuple[str, ...] = ()
    corrective_query: str | None = None

    @property
    def corrective_search_required(self) -> bool:
        return self.corrective_query is not None


@dataclass(frozen=True)
class CorrectiveRetrievalResult:
    documents: list[Document]
    coverage: CoverageAssessment
    corrective_search_used: bool

    @property
    def retrieval_status(self) -> str:
        if not self.corrective_search_used:
            return "initial"
        return "corrected" if self.coverage.status == "complete" else "exhausted"


class CoverageEvaluator:
    """Prefer verifiable metadata checks over an additional LLM judgment."""

    def evaluate(self, question: str, documents: list[Document]) -> CoverageAssessment:
        if not documents:
            return CoverageAssessment(
                status="insufficient",
                missing_information=("retrievable_evidence",),
                corrective_query=self.corrective_query(question),
            )

        matching_tables: list[Document] = []
        matching_columns: list[str] = []
        for document in documents:
            if document.metadata.get("content_kind") != "table":
                continue
            try:
                headers = parse_markdown_table(document.page_content).headers
            except ValueError:
                continue
            phrases, terms = LocalSearchEngine._header_score(question, headers)
            if phrases or terms >= 2:
                matching_tables.append(document)
                matching_columns.extend(headers)

        if not matching_tables:
            # Narrative evidence and tables without an explicit column match are
            # evaluated semantically by the existing answer validator.
            return CoverageAssessment(status="complete")
        if any(
            document.metadata.get("table_complete") is True
            for document in matching_tables
        ):
            return CoverageAssessment(status="complete")

        columns = tuple(dict.fromkeys(matching_columns))
        detail = ", ".join(columns[:6])
        return CoverageAssessment(
            status="partial",
            missing_information=("complete_table",),
            corrective_query=self.corrective_query(question, detail),
        )

    @staticmethod
    def corrective_query(question: str, columns: str | None = None) -> str:
        suffix = "tableau complet et source complète"
        if columns:
            suffix += f" avec les colonnes {columns}"
        # QueryRequest is bounded to 2,000 characters; keep the internally
        # generated query under the same public limit.
        available = 2000 - len(suffix) - 3
        return f"{question.strip()[:available]} — {suffix}"


class EvidenceMerger:
    """Merge two ranked result sets without duplicating evidence blocks."""

    @staticmethod
    def merge(
        initial: list[Document],
        corrective: list[Document],
        *,
        question: str,
        limit: int,
    ) -> list[Document]:
        unique: dict[str, Document] = {}
        for document in [*initial, *corrective]:
            identity = LocalSearchEngine._evidence_identity(
                document, dict(document.metadata)
            )
            unique.setdefault(identity, document)
        documents = list(unique.values())

        def score(document: Document) -> float:
            value = document.metadata.get("rerank_score")
            return float(value) if value is not None else float("-inf")

        documents.sort(key=score, reverse=True)
        LocalSearchEngine._promote_exact_table_header(question, documents)
        return documents[:limit]


class CorrectiveRetrievalManager:
    """Run the initial search and at most one controlled corrective search."""

    def __init__(
        self,
        search_engine: LocalSearchEngine,
        evaluator: CoverageEvaluator | None = None,
    ):
        self.search_engine = search_engine
        self.evaluator = evaluator or CoverageEvaluator()

    def retrieve(self, *, query: str, **search_options) -> CorrectiveRetrievalResult:
        initial = self.search_engine.search(query=query, **search_options)
        assessment = self.evaluator.evaluate(query, initial)
        if not assessment.corrective_search_required:
            return CorrectiveRetrievalResult(initial, assessment, False)

        corrective = self.search_engine.search(
            query=assessment.corrective_query,
            **search_options,
        )
        merged = EvidenceMerger.merge(
            initial,
            corrective,
            question=query,
            limit=search_options.get("top_k", 5),
        )
        final_assessment = self.evaluator.evaluate(query, merged)
        return CorrectiveRetrievalResult(merged, final_assessment, True)

    def correct(
        self,
        *,
        query: str,
        initial: list[Document],
        **search_options,
    ) -> CorrectiveRetrievalResult:
        """Perform the one semantic correction requested by answer validation."""
        corrective_query = self.evaluator.corrective_query(query)
        corrective = self.search_engine.search(
            query=corrective_query,
            **search_options,
        )
        merged = EvidenceMerger.merge(
            initial,
            corrective,
            question=query,
            limit=search_options.get("top_k", 5),
        )
        assessment = self.evaluator.evaluate(query, merged)
        return CorrectiveRetrievalResult(merged, assessment, True)
