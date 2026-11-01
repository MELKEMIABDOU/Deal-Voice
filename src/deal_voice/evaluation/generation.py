"""Lightweight checks for generated RAG answers."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from deal_voice.evaluation.question_set import normalize_text, source_matches


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
    language_compliant: bool
    refusal_expected: bool
    refusal_detected: bool
    forbidden_claims_found: list[str]
    expected_values_found: list[str]
    expected_values_missing: list[str]
    citation_files_found: list[str]
    citation_files_missing: list[str]
    citation_details_found: list[str]
    citation_details_missing: list[str]
    invalid_citation_details: list[str]
    citation_format_warnings: list[str] = field(default_factory=list)
    value_relations_found: list[str] = field(default_factory=list)
    value_relations_missing: list[str] = field(default_factory=list)
    value_relations_not_evaluable: list[str] = field(default_factory=list)

    @property
    def fact_recall(self) -> float:
        total = len(self.facts_found) + len(self.facts_missing)
        return len(self.facts_found) / total if total else 1.0

    @property
    def lexical_fact_recall(self) -> float:
        total = len(self.lexical_facts_found) + len(self.lexical_facts_missing)
        return len(self.lexical_facts_found) / total if total else 1.0

    @property
    def value_recall(self) -> float:
        total = len(self.expected_values_found) + len(self.expected_values_missing)
        return len(self.expected_values_found) / total if total else 1.0

    @property
    def value_relation_recall(self) -> float | None:
        total = len(self.value_relations_found) + len(self.value_relations_missing)
        return len(self.value_relations_found) / total if total else None

    @property
    def citation_file_recall(self) -> float:
        total = len(self.citation_files_found) + len(self.citation_files_missing)
        return len(self.citation_files_found) / total if total else 1.0

    @property
    def citation_detail_recall(self) -> float:
        total = len(self.citation_details_found) + len(self.citation_details_missing)
        return len(self.citation_details_found) / total if total else 1.0

    @property
    def passed(self) -> bool:
        if self.refusal_expected:
            return (
                self.answer_non_empty
                and self.language_compliant
                and self.refusal_detected
                and not self.forbidden_claims_found
                and not self.invalid_citation_details
            )
        return (
            self.answer_non_empty
            and self.language_compliant
            and self.source_present
            and self.fact_recall == 1.0
            and self.value_recall == 1.0
            and self.citation_detail_recall == 1.0
            and not self.forbidden_claims_found
            and not self.invalid_citation_details
            and not self.value_relations_missing
        )


def _citation_file_present(answer: str, expected_file: str) -> bool:
    return _citation_line_for_file(answer, expected_file) is not None


def _contains_normalized_phrase(text: str, phrase: str) -> bool:
    normalized_phrase = normalize_text(phrase)
    if not normalized_phrase:
        return False
    return f" {normalized_phrase} " in f" {text} "


def _citation_lines_for_file(answer: str, filename: str) -> list[str]:
    """Return every citation to a file; a PDF can have several cited passages."""
    lines: list[str] = []
    for line in answer.splitlines():
        match = re.match(
            r"^\s*(?:\*\*|__)?source(?:\*\*|__)?\s*:\s*(.*?)\s*$",
            line,
            re.IGNORECASE,
        )
        if not match:
            continue
        if source_matches(filename, match.group(1).strip()):
            lines.append(line)
            continue
        cited_filename = re.split(
            r"\s+[—–-]\s+(?:(?:section|page|table|tableau|time|timestamp|horodatage)\b"
            r"|\d+(?:[.,]\d+)?\s*[—–-]\s*\d+(?:[.,]\d+)?\s*s(?=\s*(?:[—–-]|$)))",
            match.group(1),
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip()
        if source_matches(filename, cited_filename):
            lines.append(line)
    return lines


def _citation_line_for_file(answer: str, filename: str) -> str | None:
    """Compatibility helper for callers that only need citation existence."""
    return next(iter(_citation_lines_for_file(answer, filename)), None)


def _normalized_locator(value: str) -> str:
    """Ignore Markdown and heading numbering, but preserve the title itself."""
    title = value.strip().lstrip("#").strip()
    title = re.sub(r"^(?:\d+(?:\.\d+)*[.)]\s+)+", "", title)
    return normalize_text(title)


def _named_locators(line: str) -> list[tuple[str, str]]:
    matches = re.finditer(
        r"\b(section|table|tableau)\s*[«\"“](.+?)[»\"”]",
        line,
        re.IGNORECASE,
    )
    return [
        (
            "table"
            if match.group(1).casefold() == "tableau"
            else match.group(1).casefold(),
            match.group(2).strip(),
        )
        for match in matches
    ]


def _citation_detail_present(
    answer: str, citation: dict[str, object], key: str
) -> bool:
    value = citation.get(key)
    if value is None:
        return True
    filename = citation.get("file")
    lines = _citation_lines_for_file(answer, str(filename)) if filename else []
    if not lines:
        return False
    if key == "page":
        page = re.escape(str(value))
        return any(
            re.search(rf"\b(?:page|p)\s*{page}(?:\D|$)", normalize_text(line))
            is not None
            for line in lines
        )
    return any(
        _normalized_locator(actual_title) == _normalized_locator(str(value))
        for line in lines
        for _, actual_title in _named_locators(line)
    )


def _value_relation_label(value: dict[str, object]) -> str:
    return " / ".join(str(value.get(key, "")) for key in ("entity", "field", "value"))


def _value_relation_present(
    expected: dict[str, object], actual_values: list[dict[str, object]]
) -> bool:
    """Compare explicit triples, rather than infer a relation from nearby words."""
    return any(
        all(
            normalize_text(str(expected.get(key, "")))
            == normalize_text(str(actual.get(key, "")))
            for key in ("entity", "field")
        )
        and " ".join(str(expected.get("value", "")).casefold().split())
        == " ".join(str(actual.get("value", "")).casefold().split())
        for actual in actual_values
    )


def evaluate_generation(
    question_id: str,
    question: str,
    expected_source: str | Iterable[str],
    expected_facts: list[str],
    answer: str,
    expected_fact_variants: dict[str, list[str]] | None = None,
    expected_behavior: str = "answer",
    forbidden_claims: list[str] | None = None,
    expected_values: list[dict[str, object]] | None = None,
    expected_citations: list[dict[str, object]] | None = None,
    expected_language: str | None = None,
    actual_values: list[dict[str, object]] | None = None,
    available_citations: list[dict[str, object]] | None = None,
) -> GenerationEvaluation:
    """Check lexical facts, explicit variants, citations and supplied value triples.

    Fact recall is a reproducible lexical metric, not a semantic judge. A
    paraphrase is accepted only when the question set explicitly declares it.
    Value relations are verified only when structured actual_values are supplied;
    text-only value occurrence is reported separately as a lexical metric.
    """
    normalized_answer = answer.casefold()
    normalized_text_answer = normalize_text(answer)
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
            _contains_normalized_phrase(normalized_text_answer, variant)
            for variant in [fact, *variants.get(fact, [])]
        )
    ]
    facts_missing = [fact for fact in expected_facts if fact not in facts_found]
    french_markers = {
        " le ",
        " la ",
        " les ",
        " des ",
        " une ",
        " dans ",
        " est ",
        " pour ",
        " avec ",
        " et ",
    }
    # Source names and section titles are not evidence of the answer's language.
    answer_without_citations = re.sub(
        r"(?m)^\s*(?:\*\*|__)?source(?:\*\*|__)?\s*:.*$",
        "",
        normalized_answer,
    )
    padded_answer = f" {' '.join(answer_without_citations.split())} "
    likely_french = any(marker in padded_answer for marker in french_markers) or (
        re.search(r"\b[ld]['’]\w", answer_without_citations) is not None
    )
    normalized_expected_language = (expected_language or "").casefold()
    language_compliant = (
        likely_french
        if normalized_expected_language in {"fr", "fra", "french", "français"}
        else True
    )
    refusal_markers = (
        "ne dispose pas de suffisamment",
        "informations autorisées",
        "informations suffisantes",
        "je ne peux pas répondre",
        "je ne peux pas déterminer",
    )
    extra_refusal_markers = (
        "je ne dispose pas de suffisamment",
        "aucune information",
        "information n est pas indiquee",
        "information n est pas fournie",
        "je ne peux pas repondre",
        "je ne peux pas determiner",
    )
    refusal_detected = any(
        normalize_text(marker) in normalized_text_answer
        for marker in (*refusal_markers, *extra_refusal_markers)
    )
    forbidden_claims_found = [
        claim
        for claim in (forbidden_claims or [])
        if _contains_normalized_phrase(normalized_text_answer, claim)
    ]
    if isinstance(expected_source, str):
        expected_sources = [expected_source] if expected_source else []
    else:
        expected_sources = list(expected_source)
    citations = expected_citations or []
    citation_files = [
        str(citation["file"]) for citation in citations if citation.get("file")
    ]
    if not citation_files and expected_behavior.casefold() not in {"refusal", "refuse"}:
        citation_files = expected_sources
    citation_files_found = [
        source for source in citation_files if _citation_file_present(answer, source)
    ]
    citation_files_missing = [
        source for source in citation_files if source not in citation_files_found
    ]
    citation_details_found: list[str] = []
    citation_details_missing: list[str] = []
    invalid_citation_details: list[str] = []
    citation_format_warnings: list[str] = []
    # Validate against the union of available passages in each file. Checking one
    # expected section against another valid section in the same PDF is wrong.
    allowed_citations = (
        available_citations if available_citations is not None else citations
    )
    for filename in dict.fromkeys(
        str(item["file"]) for item in citations if item.get("file")
    ):
        file_citations = [
            item
            for item in allowed_citations
            if source_matches(str(item.get("file", "")), filename)
        ]
        allowed_titles = {
            _normalized_locator(str(item[key]))
            for item in file_citations
            for key in ("section", "table")
            if item.get(key) is not None
        }
        allowed_pages = {
            str(item["page"]) for item in file_citations if item.get("page") is not None
        }
        for line in _citation_lines_for_file(answer, filename):
            for kind, title in _named_locators(line):
                normalized_title = _normalized_locator(title)
                if normalized_title not in allowed_titles and (
                    allowed_titles or available_citations is not None
                ):
                    invalid_citation_details.append(
                        f"{filename}: unexpected {kind} '{title}'"
                    )
                elif allowed_titles and not any(
                    item.get(kind) is not None
                    and _normalized_locator(str(item[kind])) == normalized_title
                    for item in file_citations
                ):
                    citation_format_warnings.append(
                        f"{filename}: title '{title}' cited as {kind}"
                    )
            if allowed_pages or available_citations is not None:
                for match in re.finditer(
                    r"\b(?:page|p)\.?\s*(\d+)\b", line, re.IGNORECASE
                ):
                    if match.group(1) not in allowed_pages:
                        invalid_citation_details.append(
                            f"{filename}: unexpected page '{match.group(1)}'"
                        )
    for citation in citations:
        filename = citation.get("file")
        if not filename:
            continue
        for key in ("page", "section", "table"):
            if citation.get(key) is None:
                continue
            value = str(citation[key])
            label = f"page {value}" if key == "page" else value
            if _citation_detail_present(answer, citation, key):
                citation_details_found.append(label)
            else:
                citation_details_missing.append(label)
    values = expected_values or []
    expected_values_found = [
        str(value["value"])
        for value in values
        if value.get("value")
        and _contains_normalized_phrase(normalized_text_answer, str(value["value"]))
    ]
    expected_values_missing = [
        str(value["value"])
        for value in values
        if value.get("value")
        and not _contains_normalized_phrase(normalized_text_answer, str(value["value"]))
    ]
    relation_values = [
        value
        for value in values
        if value.get("entity") is not None and value.get("field") is not None
    ]
    value_relations_found = []
    value_relations_missing = []
    value_relations_not_evaluable = []
    for value in relation_values:
        label = _value_relation_label(value)
        if actual_values is None:
            value_relations_not_evaluable.append(label)
        elif _value_relation_present(value, actual_values):
            value_relations_found.append(label)
        else:
            value_relations_missing.append(label)
    return GenerationEvaluation(
        question_id=question_id,
        answer=answer,
        source_expected=", ".join(expected_sources),
        source_present=not citation_files_missing,
        facts_found=facts_found,
        facts_missing=facts_missing,
        lexical_facts_found=lexical_facts_found,
        lexical_facts_missing=lexical_facts_missing,
        answer_non_empty=bool(answer.strip()),
        likely_french=likely_french,
        language_compliant=language_compliant,
        refusal_expected=expected_behavior.casefold() in {"refusal", "refuse"},
        refusal_detected=refusal_detected,
        forbidden_claims_found=forbidden_claims_found,
        expected_values_found=expected_values_found,
        expected_values_missing=expected_values_missing,
        citation_files_found=citation_files_found,
        citation_files_missing=citation_files_missing,
        citation_details_found=citation_details_found,
        citation_details_missing=citation_details_missing,
        invalid_citation_details=invalid_citation_details,
        citation_format_warnings=citation_format_warnings,
        value_relations_found=value_relations_found,
        value_relations_missing=value_relations_missing,
        value_relations_not_evaluable=value_relations_not_evaluable,
    )
