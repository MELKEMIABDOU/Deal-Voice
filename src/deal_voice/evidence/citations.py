"""Preserve passage identity and render citations from source metadata."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from langchain_core.documents import Document

from deal_voice.evidence.tabular import normalized_label


def clean_title(value: str) -> str:
    if "\n" in value or "\r" in value:
        raise ValueError("Citation titles must be single-line strings.")
    return re.sub(r"^\s*#{1,6}\s*", "", value).strip()


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    document: Document
    locators: tuple[tuple[str, str], ...]

    @property
    def source(self) -> str:
        return str(self.document.metadata["source"]).strip()

    def citation(self, locator: str | None = None) -> str:
        selected = None
        if locator:
            selected = next(
                (
                    item
                    for item in self.locators
                    if normalized_label(item[1])
                    == normalized_label(clean_title(locator))
                ),
                None,
            )
            if selected is None:
                raise ValueError("The citation locator is absent from this evidence.")
        elif len(self.locators) == 1:
            selected = self.locators[0]
        text = f"Source : {self.source}"
        if selected:
            kind, title = selected
            text += f" — {kind} « {title} »"
        page = self.document.metadata.get(
            "page_number", self.document.metadata.get("page")
        )
        if page is not None:
            text += f" — page {page}"
        start, end = (
            self.document.metadata.get(
                "start_time", self.document.metadata.get("start")
            ),
            self.document.metadata.get("end_time", self.document.metadata.get("end")),
        )
        if start is not None and end is not None:
            text += f" — {start}–{end} s"
        return text

    def available_citations(self) -> list[dict[str, object]]:
        base: dict[str, object] = {"file": self.source}
        page = self.document.metadata.get(
            "page_number", self.document.metadata.get("page")
        )
        if page is not None:
            base["page"] = page
        return [dict(base, **{kind: title}) for kind, title in self.locators] or [base]


def evidence_from_documents(documents: list[Document]) -> list[Evidence]:
    evidence = {}
    for document in documents:
        metadata = document.metadata
        raw_source = metadata.get("source", "")
        if not isinstance(raw_source, str):
            raise ValueError("Every evidence block requires a source filename string.")
        source = raw_source.strip()
        if not source or "\n" in raw_source or "\r" in raw_source:
            raise ValueError("Every evidence block requires a valid source filename.")
        page = metadata.get("page_number", metadata.get("page"))
        if page is not None and (
            isinstance(page, bool) or not str(page).isdigit() or int(page) < 1
        ):
            raise ValueError("Citation pages must be positive one-based integers.")
        identity = json.dumps(
            [
                source,
                metadata.get("document_id"),
                metadata.get("parent_id"),
                metadata.get("table_id"),
                metadata.get("page_number", metadata.get("page")),
                metadata.get("section"),
                metadata.get("table_title"),
                metadata.get("start_time", metadata.get("start")),
                metadata.get("end_time", metadata.get("end")),
                document.page_content,
            ],
            ensure_ascii=False,
            sort_keys=True,
        )
        evidence_id = "E_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
        locators = []
        if metadata.get("content_kind") == "table":
            if metadata.get("table_title"):
                locators.append(("table", clean_title(str(metadata["table_title"]))))
        elif metadata.get("section"):
            locators.append(("section", clean_title(str(metadata["section"]))))
        else:
            # Legacy chunks retain real Markdown headings; prose mentions do not count.
            locators.extend(
                ("section", clean_title(line))
                for line in document.page_content.splitlines()
                if re.match(r"^\s*#{1,6}\s+\S", line)
            )
        evidence[evidence_id] = Evidence(
            evidence_id, document, tuple(dict.fromkeys(locators))
        )
    return list(evidence.values())
