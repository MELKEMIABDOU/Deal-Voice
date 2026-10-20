"""Retrieval evaluation helpers for the reference question set."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from langchain_core.documents import Document


@dataclass
class RetrievalEvaluation:
    """Evaluation result for one reference question."""

    question_id: str
    expected_source: str
    retrieved_sources: list[str]
    source_hit: bool
    expected_facts_found: list[str]
    expected_facts_missing: list[str]
    expected_source_rank: int | None

    @property
    def fact_recall(self) -> float:
        total = len(self.expected_facts_found) + len(self.expected_facts_missing)
        return len(self.expected_facts_found) / total if total else 1.0

    @property
    def reciprocal_rank(self) -> float:
        return 1 / self.expected_source_rank if self.expected_source_rank else 0.0

    @property
    def binary_rank_score(self) -> float:
        """Return a source-rank score; this is not graded nDCG."""
        return self.reciprocal_rank

    @property
    def precision_at_k(self) -> float:
        if not self.retrieved_sources:
            return 0.0
        return sum(
            source == self.expected_source for source in self.retrieved_sources
        ) / len(self.retrieved_sources)


def evaluate_retrieval(
    question_id: str,
    expected_source: str,
    expected_facts: Iterable[str],
    documents: list[Document],
    expected_fact_variants: dict[str, list[str]] | None = None,
) -> RetrievalEvaluation:
    """Score source retrieval and controlled-variant fact coverage."""
    contents = "\n".join(document.page_content.lower() for document in documents)
    facts = list(expected_facts)
    variants = expected_fact_variants or {}
    found = [
        fact
        for fact in facts
        if any(
            variant.lower() in contents
            for variant in [fact, *variants.get(fact, [])]
        )
    ]
    missing = [fact for fact in facts if fact not in found]
    sources = [
        str(document.metadata.get("source", ""))
        for document in documents
    ]
    return RetrievalEvaluation(
        question_id=question_id,
        expected_source=expected_source,
        retrieved_sources=sources,
        source_hit=expected_source in sources,
        expected_facts_found=found,
        expected_facts_missing=missing,
        expected_source_rank=(
            next(
                (
                    index
                    for index, source in enumerate(sources, start=1)
                    if source == expected_source
                ),
                None,
            )
        ),
    )
