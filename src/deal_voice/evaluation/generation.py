"""Lightweight checks for generated RAG answers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GenerationEvaluation:
    """Deterministic checks for one generated answer."""

    question_id: str
    answer: str
    source_expected: str
    source_present: bool
    facts_found: list[str]
    facts_missing: list[str]
    lexical_facts_found: list[str]
    lexical_facts_missing: list[str]
    answer_non_empty: bool
    likely_french: bool
    refusal_expected: bool
    refusal_detected: bool
    unsupported_claims_found: list[str]

    @property
    def fact_recall(self) -> float:
        total = len(self.facts_found) + len(self.facts_missing)
        return len(self.facts_found) / total if total else 1.0

    @property
    def lexical_fact_recall(self) -> float:
        total = len(self.lexical_facts_found) + len(self.lexical_facts_missing)
        return len(self.lexical_facts_found) / total if total else 1.0

    @property
    def passed(self) -> bool:
        if self.refusal_expected:
            return (
                self.answer_non_empty
                and self.likely_french
                and self.refusal_detected
                and not self.unsupported_claims_found
            )
        return (
            self.answer_non_empty
            and self.source_present
            and self.fact_recall >= 0.8
            and self.likely_french
            and not self.unsupported_claims_found
        )


def evaluate_generation(
    question_id: str,
    question: str,
    expected_source: str,
    expected_facts: list[str],
    answer: str,
    expected_fact_variants: dict[str, list[str]] | None = None,
    expected_behavior: str = "answer",
    forbidden_claims: list[str] | None = None,
) -> GenerationEvaluation:
    """Apply lexical and controlled-variant checks to a generated answer."""
    normalized_answer = answer.casefold()
    lexical_facts_found = [
        fact for fact in expected_facts if fact.casefold() in normalized_answer
    ]
    lexical_facts_missing = [
        fact for fact in expected_facts if fact.casefold() not in normalized_answer
    ]
    variants = expected_fact_variants or {}
    facts_found = [
        fact
        for fact in expected_facts
        if any(
            variant.casefold() in normalized_answer
            for variant in [fact, *variants.get(fact, [])]
        )
    ]
    facts_missing = [fact for fact in expected_facts if fact not in facts_found]
    french_markers = {
        " le ", " la ", " les ", " des ", " une ", " dans ",
        " est ", " pour ", " avec ", " et ",
    }
    padded_answer = f" {normalized_answer} "
    likely_french = any(marker in padded_answer for marker in french_markers)
    refusal_markers = (
        "ne dispose pas de suffisamment",
        "informations autorisées",
        "informations suffisantes",
        "je ne peux pas répondre",
        "je ne peux pas déterminer",
    )
    refusal_detected = any(marker in normalized_answer for marker in refusal_markers)
    unsupported_claims_found = [
        claim for claim in (forbidden_claims or [])
        if claim.casefold() in normalized_answer
    ]
    return GenerationEvaluation(
        question_id=question_id,
        answer=answer,
        source_expected=expected_source,
        source_present=expected_source.casefold() in normalized_answer,
        facts_found=facts_found,
        facts_missing=facts_missing,
        lexical_facts_found=lexical_facts_found,
        lexical_facts_missing=lexical_facts_missing,
        answer_non_empty=bool(answer.strip()),
        likely_french=likely_french,
        refusal_expected=expected_behavior == "refusal",
        refusal_detected=refusal_detected,
        unsupported_claims_found=unsupported_claims_found,
    )
