"""Generate evidence-backed prose or execute validated table operations locally."""

from __future__ import annotations

import json
import logging
import re
from contextvars import ContextVar

from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, ValidationError

from deal_voice.config import settings
from deal_voice.evidence.citations import Evidence, evidence_from_documents
from deal_voice.evidence.tabular import (
    TableQuery,
    execute_table_query,
    normalized_label,
    parse_markdown_table,
)
from deal_voice.generation.prompts import (
    SYSTEM_PROMPT,
    VERIFICATION_PROMPT,
    answer_language,
    prompt_values,
)
from deal_voice.generation.schemas import (
    AnswerPart,
    GenerationPlan,
    GroundingCheck,
    decoding_schema,
)

logger = logging.getLogger(__name__)
VERIFICATION_OUTPUT_TOKENS = 192


def estimated_prompt_tokens(text: str) -> int:
    """Conservatively estimate tokens without loading a second model tokenizer.

    Ollama context limits are token counts, not byte counts. Two UTF-8 bytes per
    token remains deliberately cautious for French prose and JSON while avoiding
    rejection of ordinary transcripts that fit comfortably in Qwen's context.
    """
    return (len(text.encode("utf-8")) + 1) // 2


class RAGGenerator:
    """The model proposes; code validates references and performs table calculations."""

    def __init__(self, model_name: str = settings.ollama_model_name):
        prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", "{question}")]
        )
        logger.info("Connecting to local Ollama model: %s", model_name)
        self.llm = ChatOllama(
            model=model_name,
            base_url=settings.ollama_base_url,
            temperature=0.0,
            reasoning=settings.ollama_reasoning,
            num_ctx=settings.ollama_num_ctx,
            num_predict=settings.ollama_num_predict,
            num_thread=settings.ollama_num_thread,
            keep_alive=settings.ollama_keep_alive,
            client_kwargs={"timeout": settings.ollama_request_timeout_seconds},
        )
        self.chain = prompt | self.llm.with_structured_output(
            decoding_schema(), method="json_schema"
        )
        self.verifier = self.llm.with_structured_output(
            GroundingCheck, method="json_schema"
        ).bind(
            options={
                "num_ctx": settings.ollama_num_ctx,
                "num_predict": VERIFICATION_OUTPUT_TOKENS,
                "num_thread": settings.ollama_num_thread,
                "temperature": 0.0,
            }
        )
        self._trace: ContextVar[dict | None] = ContextVar(
            "dealvoice_generation", default=None
        )

    @property
    def last_trace(self) -> dict:
        return self._trace.get() or {}

    def _record(self, trace: dict) -> None:
        self._trace.set(trace)

    @staticmethod
    def _unavailable(question: str, reason: str = "insufficient") -> str:
        if answer_language(question) == "French":
            if reason == "context_limit":
                return "Le contexte dépasse la capacité configurée. Précisez le document ou la section à analyser."
            if reason == "invalid_plan":
                return "Je ne peux pas établir une réponse vérifiable avec les preuves et l’opération disponibles. Précisez votre demande."
            return "Je ne dispose pas de suffisamment d'informations dans les passages consultés pour répondre à cette question."
        if reason == "context_limit":
            return "The context exceeds the configured capacity. Specify a document or section."
        if reason == "invalid_plan":
            return "I cannot establish a verifiable answer with the available evidence and operation. Please clarify your request."
        return "I do not have enough information in the consulted passages to answer this question."

    def _catalog(
        self, question: str, documents: list[Document]
    ) -> tuple[list[Evidence], str, int]:
        # Keep the conservative byte upper bound without loading a tokenizer/model.
        # Reserve the output and chat-template tokens, then count the rendered prompt.
        budget = settings.ollama_num_ctx - settings.ollama_num_predict - 256
        included, entries = [], []
        evidence = evidence_from_documents(documents)
        for item in evidence:
            metadata = item.document.metadata
            entry: dict[str, object] = {
                "evidence_id": item.evidence_id,
                "source": item.source,
                "locators": [
                    {key: value for key, value in citation.items() if key != "file"}
                    for citation in item.available_citations()
                ],
            }
            if metadata.get("content_kind") == "table":
                try:
                    table = parse_markdown_table(item.document.page_content)
                except ValueError as exc:
                    raise ValueError(
                        "A retrieved table has an invalid structure."
                    ) from exc
                entry.update(
                    {
                        "kind": "table",
                        "columns": table.headers,
                        "row_count": len(table.rows),
                        "complete_extracted_block": metadata.get("table_complete")
                        is True,
                        "decimal_separator": metadata.get("decimal_separator"),
                    }
                )
            else:
                entry.update(
                    {
                        "kind": metadata.get("file_type", "text"),
                        "content": item.document.page_content,
                    }
                )
            encoded = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
            candidates = [*included, item]
            context = (
                f"Omitted evidence blocks: {len(evidence) - len(candidates)}\n"
                + "\n".join([*entries, encoded])
            )
            has_tables = any(
                entry.document.metadata.get("content_kind") == "table"
                for entry in candidates
            )
            values = prompt_values(question, context, has_tables)
            rendered = SYSTEM_PROMPT.format(**values) + question
            if estimated_prompt_tokens(rendered) > budget:
                continue
            entries.append(encoded)
            included.append(item)
        omitted = len(evidence) - len(included)
        return (
            included,
            f"Omitted evidence blocks: {omitted}\n" + "\n".join(entries),
            omitted,
        )

    @staticmethod
    def _clarification(question: str, kind: str) -> str:
        choices = {
            "source": (
                "Quel document souhaitez-vous interroger ?",
                "Which document should be used?",
            ),
            "column": (
                "Quelle colonne souhaitez-vous utiliser ?",
                "Which column should be used?",
            ),
            "operation": (
                "Quel calcul souhaitez-vous effectuer ?",
                "Which calculation should be performed?",
            ),
            "scope": (
                "Pouvez-vous préciser le périmètre de votre question ?",
                "Please clarify the scope of your question.",
            ),
            "criterion": (
                "Quel critère souhaitez-vous utiliser pour cette recommandation ?",
                "Which criterion should guide this recommendation?",
            ),
        }
        return choices[kind][0 if answer_language(question) == "French" else 1]

    def generate(self, question: str, retrieved_docs: list[Document]) -> str:
        self._record({"status": "initializing", "actual_values": None})
        if not retrieved_docs:
            self._record({"status": "insufficient", "actual_values": None})
            return self._unavailable(question)
        try:
            included, context, omitted = self._catalog(question, retrieved_docs)
        except ValueError as exc:
            self._record(
                {"status": "invalid_evidence", "error": str(exc), "actual_values": None}
            )
            return self._unavailable(question, "invalid_plan")
        trace = {
            "status": "planning",
            "actual_values": None,
            "evidence_ids": [item.evidence_id for item in included],
            "omitted_evidence_count": omitted,
            "structured_citations": [
                citation for item in included for citation in item.available_citations()
            ],
        }
        self._record(trace)
        if not included:
            self._record(dict(trace, status="context_limit"))
            return self._unavailable(question, "context_limit")
        try:
            raw = self.chain.invoke(
                prompt_values(
                    question,
                    context,
                    any(
                        item.document.metadata.get("content_kind") == "table"
                        for item in included
                    ),
                )
            )
            plan = GenerationPlan.model_validate(
                raw.model_dump() if isinstance(raw, BaseModel) else raw
            )
            # A single answer part represents the complete user request. Use the
            # original text as its label: local models commonly correct spelling
            # while planning, which must not invalidate an otherwise valid plan.
            if len(plan.parts) == 1:
                plan.parts[0].request = question
            else:
                labels: dict[str, int] = {}
                for index, part in enumerate(plan.parts, start=1):
                    label = part.request.strip() or (
                        f"Partie {index}"
                        if answer_language(question) == "French"
                        else f"Part {index}"
                    )
                    key = normalized_label(label).strip(" ?.!;:")
                    labels[key] = labels.get(key, 0) + 1
                    if labels[key] > 1:
                        label = f"{label} ({labels[key]})"
                    part.request = label
            trace["proposed_plan"] = plan.model_dump()
            prepared = []
            for part in plan.parts:
                if part.status != "answer":
                    text = (
                        self._clarification(question, part.clarification)
                        if part.status == "clarify"
                        else self._unavailable(question)
                    )
                    prepared.append(
                        {
                            "status": part.status,
                            "text": text,
                            "values": None,
                            "computations": [],
                        }
                    )
                    continue
                try:
                    text, values, computations = self._render_part(
                        part, included, question
                    )
                    prepared.append(
                        {
                            "status": "answer",
                            "text": text,
                            "values": values,
                            "computations": computations,
                        }
                    )
                except ValueError as exc:
                    prepared.append(
                        {
                            "status": "invalid_plan",
                            "text": self._unavailable(question, "invalid_plan"),
                            "values": None,
                            "computations": [],
                            "error": str(exc),
                        }
                    )
            review = self._verify_plan(question, plan, included, prepared)
            if review is None:
                self._record(
                    dict(trace, status="context_limit", verification="context_limit")
                )
                return self._unavailable(question, "context_limit")
            trace["verification"] = review.model_dump()
        except (ValidationError, OutputParserException, ValueError) as exc:
            logger.warning("Rejected unverifiable generation plan: %s", exc)
            self._record(dict(trace, status="invalid_plan", error=str(exc)))
            return self._unavailable(question, "invalid_plan")

        paragraphs, values, computations, statuses = [], [], [], []
        for part, item, verdict in zip(
            plan.parts, prepared, review.verdicts, strict=True
        ):
            status, text = item["status"], item["text"]
            if verdict != "supported":
                status, text = "unverified", self._unavailable(question)
            elif status == "answer":
                values.extend(item["values"] or [])
                computations.extend(item["computations"])
            statuses.append(status)
            if len(plan.parts) > 1:
                text = f"**{part.request.strip()}**\n\n{text}"
            paragraphs.append(text)
        french = answer_language(question) == "French"
        if review.coverage != "complete":
            paragraphs.append(
                "Réponse partielle : toutes les demandes de votre question n'ont pas été couvertes. Posez séparément les demandes restantes."
                if french
                else "Partial answer: not all requests were covered. Please ask the remaining requests separately."
            )
        if omitted and (
            review.coverage != "complete"
            or any(part_status != "answer" for part_status in statuses)
        ):
            paragraphs.append(
                f"Contexte partiel : {omitted} passage(s) récupéré(s) n’ont pas été inclus dans la capacité disponible."
                if french
                else f"Partial context: {omitted} retrieved passage(s) did not fit the available capacity."
            )
        status = statuses[0] if len(set(statuses)) == 1 else "partial"
        if review.coverage != "complete":
            status = "partial"
        self._record(
            dict(
                trace,
                status=status,
                part_statuses=statuses,
                actual_values=values or None,
                computations=computations,
                table_scope_limited=any(
                    item.document.metadata.get("table_source_complete") is not True
                    for item in included
                    if any(
                        computation["query"]["evidence_id"] == item.evidence_id
                        for computation in computations
                    )
                ),
            )
        )
        return "\n\n".join(paragraphs)

    def _verify_plan(
        self,
        question: str,
        plan: GenerationPlan,
        evidence: list[Evidence],
        prepared: list[dict],
    ) -> GroundingCheck | None:
        """One bounded review of coverage, meaning, claims and advice; no retry loop."""
        used = {
            ref.evidence_id
            for part in plan.parts
            for claim in part.claims
            for ref in claim.references
        } | {query.evidence_id for part in plan.parts for query in part.table_queries}
        # A refusal or clarification cannot be checked against an empty catalog.
        # The reviewer needs all evidence the planner saw to judge missing data.
        unresolved = any(part.status != "answer" for part in plan.parts)
        catalog = {}
        for item in evidence:
            if item.evidence_id not in used and not unresolved:
                continue
            entry = {"citation": item.citation()}
            if item.document.metadata.get("content_kind") == "table":
                entry["columns"] = parse_markdown_table(
                    item.document.page_content
                ).headers
                entry["complete_extracted_block"] = (
                    item.document.metadata.get("table_complete") is True
                )
            else:
                entry["content"] = item.document.page_content
            catalog[item.evidence_id] = entry
        parts = []
        for part, item in zip(plan.parts, prepared, strict=True):
            entry = part.model_dump(exclude_none=True, exclude_defaults=True)
            if item["status"] == "invalid_plan":
                entry["execution_error"] = item["error"]
            elif item["computations"]:
                entry["computed_results"] = [
                    self._compact_computation(index, computation)
                    for index, computation in enumerate(item["computations"])
                ]
            parts.append(entry)
        payload = json.dumps(
            {"question": question, "parts": parts, "evidence": catalog},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        budget = settings.ollama_num_ctx - VERIFICATION_OUTPUT_TOKENS - 256
        if estimated_prompt_tokens(VERIFICATION_PROMPT + payload) > budget:
            return None
        raw = self.verifier.invoke(
            [("system", VERIFICATION_PROMPT), ("human", payload)]
        )
        review = GroundingCheck.model_validate(
            raw.model_dump() if isinstance(raw, BaseModel) else raw
        )
        if len(review.verdicts) != len(plan.parts):
            review = self._align_review_verdicts(review, len(plan.parts))
        return review

    @staticmethod
    def _align_review_verdicts(
        review: GroundingCheck, part_count: int
    ) -> GroundingCheck:
        """Conservatively combine claim-level verdicts for one answer part."""
        if part_count != 1 or not review.verdicts:
            raise ValueError("The reviewer must assess every part exactly once.")
        if "contradicted" in review.verdicts:
            verdict = "contradicted"
        elif "insufficient" in review.verdicts:
            verdict = "insufficient"
        else:
            verdict = "supported"
        return review.model_copy(update={"verdicts": [verdict]})

    @staticmethod
    def _compact_computation(index: int, computation: dict) -> dict:
        """Give the reviewer bounded Python results, including winning rows."""
        result = computation["result"]
        rows = list(result["rows"])
        shown = rows[:5]
        return {
            "query_index": index,
            "operation": computation["query"]["operation"],
            "column": computation["query"].get("column"),
            "headers": result["headers"],
            "rows": shown,
            "aggregate": result["aggregate"],
            "matched_row_count": result["matched_row_count"],
            "review_rows_omitted": max(0, len(rows) - len(shown)),
        }

    def _render_part(
        self, plan: AnswerPart, evidence: list[Evidence], question: str
    ) -> tuple[str, list | None, list]:
        catalog = {item.evidence_id: item for item in evidence}
        paragraphs, values, computations = [], [], []
        has_aggregate = False
        for claim in plan.claims:
            citations = []
            if re.search(r"(?i)\bsource(?:\*\*)?\s*:", claim.text):
                raise ValueError("Claims cannot supply their own citations.")
            for reference in claim.references:
                item = catalog.get(reference.evidence_id)
                if item is None:
                    raise ValueError("Unknown evidence reference.")
                if (
                    item.document.metadata.get("content_kind") == "table"
                    or item.document.metadata.get("table_structure_valid") is False
                ):
                    raise ValueError("Table facts require an executed table query.")
                citations.append(item.citation(reference.locator))
            paragraphs.append(
                claim.text.strip() + "\n" + "\n".join(dict.fromkeys(citations))
            )
        for query in plan.table_queries:
            item = catalog.get(query.evidence_id)
            if item is None or item.document.metadata.get("content_kind") != "table":
                raise ValueError("The requested table is not in the evidence catalog.")
            separator = item.document.metadata.get("decimal_separator")
            if separator not in {None, ",", "."}:
                raise ValueError("The source declares an unsupported numeric locale.")
            query = TableQuery.model_validate(
                {**query.model_dump(), "decimal_separator": separator}
            )
            table = parse_markdown_table(item.document.page_content)
            result = execute_table_query(
                table,
                query,
                complete=item.document.metadata.get("table_complete") is True,
            )
            french = answer_language(question) == "French"
            labels = {
                "select": "Sélection",
                "min": "Minimum",
                "max": "Maximum",
                "sum": "Somme",
                "mean": "Moyenne",
                "count": "Nombre de lignes",
                "sort": "Classement",
            }
            label = labels[query.operation] if french else query.operation.capitalize()
            label += f" — {query.column}" if query.column else ""
            if query.calculations:
                label += "\n" + "\n".join(
                    f"{item.name} = {item.left} − {item.right}"
                    for item in query.calculations
                )
            if result.aggregate is not None:
                has_aggregate = True
                text = f"{label} : {result.aggregate}."
            elif not result.table.rows:
                text = "Aucune ligne correspondante." if french else "No matching rows."
            else:
                text = (
                    label
                    + "\n"
                    + self._render_table(result.table.headers, result.table.rows)
                )
            if result.truncated:
                text += f"\n{len(result.table.rows)} / {result.matched_row_count} " + (
                    "lignes affichées." if french else "rows shown."
                )
            text += "\n" + (
                "Le résultat est limité aux lignes du tableau fourni."
                if french
                else "The result is limited to the rows of the supplied table."
            )
            if item.document.metadata.get("table_complete") is not True:
                text += "\n" + (
                    "Tableau incomplet : sélection limitée aux lignes disponibles."
                    if french
                    else "Incomplete table: selection is limited to the available rows."
                )
            paragraphs.append(text + "\n" + item.citation())
            computations.append(
                {
                    "query": query.model_dump(),
                    "aggregate": result.aggregate,
                    "matched_row_count": result.matched_row_count,
                    "input_row_count": len(table.rows),
                    "result": {
                        "headers": result.table.headers,
                        "rows": result.table.rows,
                        "aggregate": result.aggregate,
                        "truncated": result.truncated,
                        "matched_row_count": result.matched_row_count,
                    },
                }
            )
            if result.aggregate is None:
                identities = query.identity_columns or [table.headers[0]]
                canonical_headers = {
                    normalized_label(header): header for header in result.table.headers
                }
                identities = [
                    canonical_headers[normalized_label(name)] for name in identities
                ]
                for row in result.table.rows:
                    record = dict(zip(result.table.headers, row, strict=True))
                    for identity in identities:
                        if identity in record:
                            values.extend(
                                {
                                    "entity": record[identity],
                                    "field": field,
                                    "value": value,
                                }
                                for field, value in record.items()
                            )
        if plan.recommendation:
            advice = plan.recommendation
            texts = [
                advice.action,
                advice.criterion,
                *advice.assumptions,
                *advice.limits,
            ]
            if any(re.search(r"(?i)\bsource(?:\*\*)?\s*:", text) for text in texts):
                raise ValueError("Advice cannot supply its own citations.")
            french = answer_language(question) == "French"
            paragraphs.append(
                ("Proposition : " if french else "Proposal: ")
                + advice.action
                + "\n"
                + ("Critère : " if french else "Criterion: ")
                + advice.criterion
                + (
                    "\n"
                    + ("Hypothèses : " if french else "Assumptions: ")
                    + "; ".join(advice.assumptions)
                    if advice.assumptions
                    else ""
                )
                + "\n"
                + ("Limites : " if french else "Limits: ")
                + "; ".join(advice.limits)
            )
        return (
            "\n\n".join(paragraphs),
            (values if plan.table_queries and not has_aggregate else None),
            computations,
        )

    @staticmethod
    def _render_table(
        headers: tuple[str, ...], rows: tuple[tuple[str, ...], ...]
    ) -> str:
        def escaped(cell: str) -> str:
            return cell.replace("|", r"\|").replace("\n", " ")

        lines = [
            "| " + " | ".join(escaped(cell) for cell in headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        lines.extend(
            "| " + " | ".join(escaped(cell) for cell in row) + " |" for row in rows
        )
        return "\n".join(lines)
