"""Bounded answer parts, sourced recommendations and a separate semantic review."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from deal_voice.evidence.tabular import TableQuery, normalized_label


class EvidenceReference(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    evidence_id: str
    locator: str | None = None


class AnswerClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: str = Field(min_length=1, max_length=1200)
    references: list[EvidenceReference] = Field(min_length=1, max_length=8)


class PlannedTableQuery(TableQuery):
    """Only source metadata may determine numeric locale."""

    decimal_separator: None = None


class Recommendation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    action: str = Field(min_length=1, max_length=500)
    criterion: str = Field(min_length=1, max_length=300)
    claim_indices: list[int] = Field(default_factory=list, max_length=6)
    query_indices: list[int] = Field(default_factory=list, max_length=4)
    assumptions: list[str] = Field(default_factory=list, max_length=4)
    limits: list[str] = Field(min_length=1, max_length=4)


class AnswerPart(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    request: str = Field(min_length=1, max_length=2000)
    status: Literal["answer", "insufficient", "clarify"]
    claims: list[AnswerClaim] = Field(default_factory=list, max_length=6)
    table_queries: list[PlannedTableQuery] = Field(default_factory=list, max_length=4)
    recommendation: Recommendation | None = None
    clarification: (
        Literal["source", "column", "operation", "scope", "criterion"] | None
    ) = None

    @model_validator(mode="after")
    def consistent_status_and_basis(self):
        if self.status == "answer":
            if not (self.claims or self.table_queries) or self.clarification:
                raise ValueError(
                    "An answer requires evidence and cannot be a clarification."
                )
        elif self.claims or self.table_queries or self.recommendation:
            raise ValueError(
                "An unresolved part cannot contain factual answers or advice."
            )
        if (self.status == "clarify") != (self.clarification is not None):
            raise ValueError("Only a clarification requires a clarification type.")
        if self.recommendation:
            advice = self.recommendation
            if not (advice.claim_indices or advice.query_indices):
                raise ValueError(
                    "A recommendation requires an explicit evidence basis."
                )
            for indices, items in (
                (advice.claim_indices, self.claims),
                (advice.query_indices, self.table_queries),
            ):
                if len(set(indices)) != len(indices) or any(
                    i < 0 or i >= len(items) for i in indices
                ):
                    raise ValueError(
                        "Recommendation references an unknown or duplicate result."
                    )
            if any(not text.strip() for text in advice.assumptions + advice.limits):
                raise ValueError("Assumptions and limits must not be empty.")
        return self


class GenerationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    parts: list[AnswerPart] = Field(min_length=1, max_length=6)

    @model_validator(mode="after")
    def deduplicate_parts(self):
        """Remove repeated requests only when their entire plans also agree."""
        unique, seen = [], []
        for part in self.parts:
            signature = part.model_dump()
            signature["request"] = normalized_label(part.request).strip(" ?.!;:")
            if signature not in seen:
                seen.append(signature)
                unique.append(part)
        self.parts = unique
        return self


class GroundingCheck(BaseModel):
    """Verdicts correspond to plan parts in order; coverage includes unresolved parts."""

    model_config = ConfigDict(extra="forbid", strict=True)
    coverage: Literal["complete", "incomplete"]
    verdicts: list[Literal["supported", "contradicted", "insufficient"]] = Field(
        min_length=1, max_length=6
    )


def decoding_schema() -> dict:
    """Constrain structure in Ollama; validate text lengths in Python afterwards."""
    schema = GenerationPlan.model_json_schema()
    fields = schema["$defs"]["AnswerPart"]["properties"]

    def branch(status: str, properties: dict, required: list[str]) -> dict:
        return {
            "type": "object",
            "properties": {
                "request": fields["request"],
                "status": {"type": "string", "const": status},
                **properties,
            },
            "required": ["request", "status", *required],
            "additionalProperties": False,
        }

    answer_fields = {
        key: fields[key] for key in ("claims", "table_queries", "recommendation")
    }
    branches = [
        branch("answer", {**answer_fields, key: {**fields[key], "minItems": 1}}, [key])
        for key in ("claims", "table_queries")
    ]
    clarification = next(
        item
        for item in fields["clarification"]["anyOf"]
        if item.get("type") == "string"
    )
    branches.extend(
        [
            branch("insufficient", {}, []),
            branch("clarify", {"clarification": clarification}, ["clarification"]),
        ]
    )
    schema["$defs"]["AnswerPart"] = {"anyOf": branches}

    def omit_string_max_lengths(node):
        # Finite string repetitions expand the native grammar and can exceed
        # llama.cpp's rule limit. GenerationPlan retains every length check;
        # num_predict still bounds generation, including malformed responses.
        if isinstance(node, dict):
            if node.get("type") == "string":
                node.pop("maxLength", None)
            for value in node.values():
                omit_string_max_lengths(value)
        elif isinstance(node, list):
            for value in node:
                omit_string_max_lengths(value)

    omit_string_max_lengths(schema)
    return schema
