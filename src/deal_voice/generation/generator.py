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
    GenerationPlan,
    GroundingCheck,
    NarrativePlan,
    TablePlan,
    decoding_schema,
)

logger = logging.getLogger(__name__)
VERIFICATION_OUTPUT_TOKENS = 64


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
        self.chains = {
            plan_type: prompt
            | self.llm.with_structured_output(
                TablePlan if plan_type is TablePlan else decoding_schema(plan_type),
                method="json_schema",
            )
            for plan_type in (NarrativePlan, TablePlan, GenerationPlan)
        }
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
            # An excluded table must not consume the larger mixed-plan schema.
            plan_type = self._plan_type([entry.document for entry in candidates])
            values = prompt_values(question, context, plan_type)
            rendered = SYSTEM_PROMPT.format(**values) + question
            if len(rendered.encode("utf-8")) > budget:
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
    def _plan_type(documents: list[Document]) -> type[BaseModel]:
        if documents and all(
            doc.metadata.get("content_kind") == "table" for doc in documents
        ):
            return TablePlan
        return (
            GenerationPlan
            if any(doc.metadata.get("content_kind") == "table" for doc in documents)
            else NarrativePlan
        )

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
            plan_type = self._plan_type([item.document for item in included])
            raw = self.chains[plan_type].invoke(
                prompt_values(question, context, plan_type)
            )
            parsed = plan_type.model_validate(
                raw.model_dump() if isinstance(raw, BaseModel) else raw
            )
            plan = (
                parsed.as_generation_plan()
                if isinstance(parsed, TablePlan)
                else GenerationPlan.model_validate(parsed.model_dump())
            )
            trace = dict(trace, proposed_plan=plan.model_dump())
            if plan.status == "insufficient":
                self._record(dict(trace, status="insufficient"))
                return self._unavailable(question)
            if plan.status == "clarify":
                self._record(dict(trace, status="clarify"))
                french = answer_language(question) == "French"
                clarifications = {
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
                }
                return clarifications[plan.clarification][0 if french else 1]
            answer, values, computations = self._render_plan(plan, included, question)
            if plan.claims:
                verdict = self._verify_claims(question, plan, included)
                trace["verification"] = verdict
                if verdict != "supported":
                    status = (
                        "context_limit" if verdict == "context_limit" else "unverified"
                    )
                    self._record(dict(trace, status=status))
                    return self._unavailable(question, status)
        except (ValidationError, OutputParserException, ValueError) as exc:
            logger.warning("Rejected unverifiable generation plan: %s", exc)
            self._record(dict(trace, status="invalid_plan", error=str(exc)))
            return self._unavailable(question, "invalid_plan")
        self._record(
            dict(
                trace, status="answer", actual_values=values, computations=computations
            )
        )
        if omitted:
            answer += "\n\n" + (
                f"Contexte partiel : {omitted} passage(s) récupéré(s) n’ont pas été inclus dans la capacité disponible."
                if answer_language(question) == "French"
                else f"Partial context: {omitted} retrieved passage(s) did not fit the available capacity."
            )
        return answer

    def _verify_claims(
        self, question: str, plan: GenerationPlan, evidence: list[Evidence]
    ) -> str:
        """One bounded, sequential check; no rewriting or automatic retries."""
        cited = {ref.evidence_id for claim in plan.claims for ref in claim.references}
        payload = json.dumps(
            {
                "question": question,
                "claims": [
                    claim.model_dump(exclude_none=True) for claim in plan.claims
                ],
                "evidence": {
                    item.evidence_id: {
                        "citation": item.citation(),
                        "content": item.document.page_content,
                    }
                    for item in evidence
                    if item.evidence_id in cited
                },
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        budget = settings.ollama_num_ctx - VERIFICATION_OUTPUT_TOKENS - 256
        if len((VERIFICATION_PROMPT + payload).encode("utf-8")) > budget:
            return "context_limit"
        raw = self.verifier.invoke(
            [("system", VERIFICATION_PROMPT), ("human", payload)]
        )
        return GroundingCheck.model_validate(
            raw.model_dump() if isinstance(raw, BaseModel) else raw
        ).verdict

    def _render_plan(
        self, plan: GenerationPlan, evidence: list[Evidence], question: str
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
                }
            )
            if result.aggregate is None:
                identities = query.identity_columns or [table.headers[0]]
                canonical_headers = {
                    normalized_label(header): header for header in table.headers
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
