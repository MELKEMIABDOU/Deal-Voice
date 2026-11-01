"""Load and normalize evaluation question sets."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any


def load_question_set(path: Path) -> list[dict[str, Any]]:
    """Load either a legacy JSON array or JSONL and expose a common schema."""
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError(f"Question file is empty: {path}")

    if text.startswith("["):
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]

    if not isinstance(rows, list) or not rows:
        raise ValueError("The question file must contain a non-empty list.")

    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, raw in enumerate(rows, start=1):
        if not isinstance(raw, dict):
            raise TypeError(f"Question {line_number} must be a JSON object.")
        item = dict(raw)
        item_id = str(item.get("id", "")).strip()
        if not item_id:
            raise ValueError(f"Question {line_number} has no id.")
        if item_id in seen_ids:
            raise ValueError(f"Duplicate question id: {item_id}")
        seen_ids.add(item_id)
        if not str(item.get("question", "")).strip():
            raise ValueError(f"Question {item_id} has no question text.")

        citations = item.get("expected_citations", [])
        if not isinstance(citations, list) or any(
            not isinstance(citation, dict) for citation in citations
        ):
            raise ValueError(f"Question {item_id} has invalid expected_citations.")
        citation_sources = [
            citation["file"]
            for citation in citations
            if isinstance(citation.get("file"), str) and citation["file"].strip()
        ]

        sources = item.get("sources")
        if sources is None:
            legacy_source = item.get("source")
            sources = [legacy_source] if legacy_source else []
        if not sources and citation_sources:
            sources = citation_sources
        if not isinstance(sources, list) or any(
            not isinstance(source, str) or not source.strip() for source in sources
        ):
            raise ValueError(
                f"Question {item_id} has invalid sources; expected a list of strings."
            )

        behavior = str(item.get("expected_behavior", "answer")).strip().casefold()
        behavior = {"refuse": "refusal", "unanswerable": "refusal"}.get(
            behavior, behavior
        )
        if behavior not in {"answer", "refusal"}:
            raise ValueError(
                f"Question {item_id} has unsupported expected_behavior: {behavior}"
            )

        facts = item.get("required_facts", item.get("expected_facts", []))
        if not isinstance(facts, list) or any(
            not isinstance(fact, str) or not fact.strip() for fact in facts
        ):
            raise ValueError(f"Question {item_id} has invalid required_facts.")
        variants = item.get("expected_fact_variants", {})
        if not isinstance(variants, dict) or any(
            not isinstance(fact, str)
            or not isinstance(options, list)
            or any(
                not isinstance(option, str) or not option.strip() for option in options
            )
            for fact, options in variants.items()
        ):
            raise ValueError(f"Question {item_id} has invalid expected_fact_variants.")
        values = item.get("expected_values", [])
        if not isinstance(values, list) or any(
            not isinstance(value, dict)
            or value.get("value") is None
            or not str(value.get("value", "")).strip()
            for value in values
        ):
            raise ValueError(f"Question {item_id} has invalid expected_values.")
        if any(
            ("file" in citation and not isinstance(citation["file"], str))
            or (
                any(key in citation for key in ("page", "section", "table"))
                and not str(citation.get("file", "")).strip()
            )
            for citation in citations
        ):
            raise ValueError(f"Question {item_id} has invalid citation file names.")
        if behavior == "answer":
            if not sources:
                raise ValueError(
                    f"Answer question {item_id} must define at least one source."
                )
            if not facts and not values:
                raise ValueError(
                    f"Answer question {item_id} must define required_facts or expected_values."
                )

        item["sources"] = sources
        item["expected_facts"] = facts
        item["expected_values"] = values
        item["expected_citations"] = citations
        item["expected_behavior"] = behavior
        normalized.append(item)

    return normalized


def normalize_text(value: str) -> str:
    """Normalize accents, case, punctuation, and common French number words."""
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    without_accents = "".join(
        char for char in decomposed if not unicodedata.combining(char)
    )
    text = " ".join(re.findall(r"[a-z0-9]+", without_accents))
    number_words = {
        "zero": "0",
        "un": "1",
        "une": "1",
        "deux": "2",
        "trois": "3",
        "quatre": "4",
        "cinq": "5",
        "six": "6",
        "sept": "7",
        "huit": "8",
        "neuf": "9",
        "dix": "10",
        "onze": "11",
        "douze": "12",
        "treize": "13",
        "quatorze": "14",
        "quinze": "15",
        "seize": "16",
        "vingt": "20",
        "trente": "30",
        "quarante": "40",
        "cinquante": "50",
        "soixante": "60",
        "cent": "100",
    }
    return " ".join(number_words.get(token, token) for token in text.split())


def source_identity(source: str) -> str:
    """Return a filename identity that ignores accents, spaces, and extension."""
    stem = Path(source.replace("\\", "/")).stem
    return re.sub(r"[^a-z0-9]", "", normalize_text(stem))


def source_matches(expected: str, actual: str) -> bool:
    """Match source references with equivalent stems (for .txt/.mp3 pairs)."""
    return expected.casefold() == actual.casefold() or (
        bool(source_identity(expected))
        and source_identity(expected) == source_identity(actual)
    )


def resolve_indexed_sources(
    expected_sources: list[str], indexed_sources: list[str]
) -> tuple[list[str], list[str]]:
    """Map reference filenames to their actual indexed filename when possible."""
    resolved: list[str] = []
    missing: list[str] = []
    for expected in expected_sources:
        match = next(
            (
                source
                for source in indexed_sources
                if source.casefold() == expected.casefold()
            ),
            None,
        )
        if match is None:
            match = next(
                (
                    source
                    for source in indexed_sources
                    if source_matches(expected, source)
                ),
                None,
            )
        if match is None:
            missing.append(expected)
        elif match not in resolved:
            resolved.append(match)
    return resolved, missing
