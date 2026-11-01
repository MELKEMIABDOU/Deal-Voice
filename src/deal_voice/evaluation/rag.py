"""Retrieval evaluation helpers for the reference question set."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from langchain_core.documents import Document

from deal_voice.evaluation.question_set import normalize_text, source_matches


@dataclass
class RetrievalEvaluation:
    """Evaluation result for one reference question."""

    question_id: str
    expected_source: str
    expected_sources_found: list[str]
    expected_sources_missing: list[str]
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
    def source_recall(self) -> float:
        total = len(self.expected_sources_found) + len(self.expected_sources_missing)
        return len(self.expected_sources_found) / total if total else 1.0

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
        expected_sources = self.expected_sources_found + self.expected_sources_missing
        return sum(
            any(source_matches(expected, source) for expected in expected_sources)
            for source in self.retrieved_sources
        ) / len(self.retrieved_sources)


def evaluate_retrieval(
    question_id: str,
    expected_source: str | Iterable[str],
    expected_facts: Iterable[str],
    documents: list[Document],
    expected_fact_variants: dict[str, list[str]] | None = None,
) -> RetrievalEvaluation:
    """Score source retrieval and controlled-variant fact coverage."""
    contents = normalize_text(
        "\n".join(document.page_content for document in documents)
    )
    facts = list(expected_facts)
    variants = expected_fact_variants or {}
    found = [
        fact
        for fact in facts
        if any(
            bool(normalize_text(variant))
            and f" {normalize_text(variant)} " in f" {contents} "
            for variant in [fact, *variants.get(fact, [])]
        )
    ]
    missing = [fact for fact in facts if fact not in found]
    sources = [str(document.metadata.get("source", "")) for document in documents]
    expected_sources = (
        [expected_source] if isinstance(expected_source, str) else list(expected_source)
    )
    sources_found = [
        expected
        for expected in expected_sources
        if any(source_matches(expected, actual) for actual in sources)
    ]
    sources_missing = [
        expected for expected in expected_sources if expected not in sources_found
    ]
    rank = next(
        (
            index
            for index, actual in enumerate(sources, start=1)
            if any(source_matches(expected, actual) for expected in expected_sources)
        ),
        None,
    )
    return RetrievalEvaluation(
        question_id=question_id,
        expected_source=", ".join(expected_sources),
        expected_sources_found=sources_found,
        expected_sources_missing=sources_missing,
        retrieved_sources=sources,
        source_hit=bool(sources_found),
        expected_facts_found=found,
        expected_facts_missing=missing,
        expected_source_rank=rank,
    )
