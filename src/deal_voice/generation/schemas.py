"""The model may propose claims and table operations, never executable code."""

import json
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from deal_voice.evidence.tabular import TableQuery


class EvidenceReference(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    evidence_id: str
    locator: str | None = None


class AnswerClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: str = Field(min_length=1)
    references: list[EvidenceReference] = Field(min_length=1, max_length=8)


class GroundingCheck(BaseModel):
    """A separate check of all narrative claims against question and evidence."""

    model_config = ConfigDict(extra="forbid", strict=True)
    verdict: Literal["supported", "contradicted", "insufficient"]


class _PlanBase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    status: Literal["answer", "insufficient", "clarify"] = Field(
        description="answer delegates table calculations to code or states supported prose; insufficient means missing evidence; clarify requires a user choice."
    )
    claims: list[AnswerClaim] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def consistent_status(self):
        queries = getattr(self, "table_queries", [])
        clarification = getattr(self, "clarification", None)
        if self.status == "answer" and not (self.claims or queries):
            raise ValueError(
                "An answer requires evidence-backed claims or table queries."
            )
        if self.status != "answer" and (self.claims or queries):
            raise ValueError(
                "Refusals and clarifications cannot contain factual answers."
            )
        if self.status == "clarify" and not clarification:
            raise ValueError("A clarification requires a question.")
        if self.status != "clarify" and clarification is not None:
            raise ValueError("Only a clarification may specify a clarification type.")
        return self


class NarrativePlan(_PlanBase):
    clarification: Literal["source", "column", "operation", "scope"] | None = None


class PlannedTableQuery(TableQuery):
    """Formatting belongs to source metadata/the parser, never to the model."""

    decimal_separator: None = None


class GenerationPlan(_PlanBase):
    table_queries: list[PlannedTableQuery] = Field(
        default_factory=list,
        max_length=8,
        description="Executable requests over catalog tables. Rows are available to the executor even though only schemas are shown.",
    )
    clarification: Literal["source", "column", "operation", "scope"] | None = None


class TablePlan(PlannedTableQuery):
    """A table-only planner chooses an operation, not a prose answer status."""

    operation: Literal[
        "select",
        "min",
        "max",
        "sum",
        "mean",
        "count",
        "sort",
        "insufficient",
        "clarify",
    ]

    def as_generation_plan(self) -> GenerationPlan:
        if self.operation == "insufficient":
            return GenerationPlan(status="insufficient")
        if self.operation == "clarify":
            return GenerationPlan(status="clarify", clarification="operation")
        query = PlannedTableQuery.model_validate(self.model_dump())
        return GenerationPlan(status="answer", table_queries=[query])


@lru_cache(maxsize=3)
def schema_instructions(model: type[BaseModel]) -> str:
    """Show the model the actual shape; native decoding alone does not explain it."""

    def compact(value):
        if isinstance(value, dict):
            return {
                key: compact(item)
                for key, item in value.items()
                if key not in {"title", "description", "default"}
            }
        return [compact(item) for item in value] if isinstance(value, list) else value

    schema = compact(model.model_json_schema())
    return json.dumps(schema, ensure_ascii=False, separators=(",", ":"))


def decoding_schema(model: type[BaseModel]) -> dict:
    """Encode status/payload consistency in the grammar as well as validation.

    Provider JSON decoders cannot infer Pydantic's after-validators. Each branch
    is a complete object; no unsupported JSON-Schema if/then rules are needed.
    The compact schema shown to the model still describes the same field types.
    """
    schema = json.loads(schema_instructions(model))
    fields = schema["properties"]

    def branch(status: str, properties: dict, required: list[str]) -> dict:
        return {
            "type": "object",
            "properties": {"status": {"type": "string", "const": status}, **properties},
            "required": ["status", *required],
            "additionalProperties": False,
        }

    branches = [
        branch(
            "answer",
            {
                "claims": {**fields["claims"], "minItems": 1},
                "clarification": {"type": "null"},
            },
            ["claims"],
        )
    ]
    if "table_queries" in fields:
        branches.append(
            branch(
                "answer",
                {
                    "claims": fields["claims"],
                    "table_queries": {**fields["table_queries"], "minItems": 1},
                    "clarification": {"type": "null"},
                },
                ["table_queries"],
            )
        )
    branches.append(branch("insufficient", {}, []))
    clarification = next(
        item
        for item in fields["clarification"]["anyOf"]
        if item.get("type") == "string"
    )
    branches.append(
        branch("clarify", {"clarification": clarification}, ["clarification"])
    )
    return {"$defs": schema.get("$defs", {}), "anyOf": branches}
