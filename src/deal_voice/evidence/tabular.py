"""Parse extracted tables and execute a small, validated query language.

No document-specific column names, generated code, SQL, or model arithmetic.
Completeness refers to the extracted block, not verified coverage of the PDF.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from fractions import Fraction
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


def normalized_label(value: str) -> str:
    text = unicodedata.normalize("NFKD", value.casefold())
    return " ".join(
        "".join(char for char in text if not unicodedata.combining(char)).split()
    )


@dataclass(frozen=True)
class MarkdownTable:
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


def _cells(line: str) -> tuple[str, ...]:
    text = line.strip()
    if text.startswith("|"):
        text = text[1:]
    if text.endswith("|"):
        backslashes = len(text[:-1]) - len(text[:-1].rstrip("\\"))
        if backslashes % 2 == 0:
            text = text[:-1]
    cells, current = [], []
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\\" and index + 1 < len(text) and text[index + 1] in {"\\", "|"}:
            current.append(text[index + 1])
            index += 2
            continue
        if char == "|":
            cells.append("".join(current).strip())
            current.clear()
        else:
            current.append(char)
        index += 1
    cells.append("".join(current).strip())
    return tuple(cells)


def parse_markdown_table(content: str) -> MarkdownTable:
    lines = content.splitlines()
    starts = [
        index - 1
        for index, line in enumerate(lines)
        if index
        and "|" in lines[index - 1]
        and len(_cells(line)) >= 2
        and all(re.fullmatch(r":?-{3,}:?", cell) for cell in _cells(line))
    ]
    if len(starts) != 1:
        raise ValueError("Expected exactly one Markdown table.")
    start = starts[0]
    headers = _cells(lines[start])
    labels = [normalized_label(header) for header in headers]
    if not all(labels) or len(set(labels)) != len(labels):
        raise ValueError("Table headers must be nonempty and unique.")
    if len(_cells(lines[start + 1])) != len(headers):
        raise ValueError("Table separator and header widths differ.")
    rows = []
    for line in lines[start + 2 :]:
        if not line.strip() or "|" not in line:
            break
        row = _cells(line)
        if len(row) != len(headers):
            raise ValueError("Table row and header widths differ.")
        rows.append(row)
    if not rows:
        raise ValueError("Table contains no data rows.")
    return MarkdownTable(headers, tuple(rows))


class TableFilter(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    column: str
    operator: Literal["eq", "ne", "contains", "lt", "le", "gt", "ge"]
    value: str


class TableQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    evidence_id: str
    operation: Literal["select", "min", "max", "sum", "mean", "count", "sort"]
    column: str | None = Field(
        default=None,
        description="Exact catalog column for min/max/sum/mean/sort; omit for row selection/count.",
    )
    filters: list[TableFilter] = Field(default_factory=list, max_length=16)
    select_columns: list[str] = Field(default_factory=list)
    identity_columns: list[str] = Field(default_factory=list)
    descending: bool = False
    limit: int | None = Field(default=None, ge=1, le=1000)
    decimal_separator: Literal[",", "."] | None = None


@dataclass(frozen=True)
class NumericCell:
    amount: Decimal
    unit: str


def numeric_cell(value: str, decimal_separator: str | None = None) -> NumericCell:
    text = value.strip().replace("−", "-")
    # Keep units explicit: no implicit conversion between currencies or percent.
    match = re.fullmatch(
        r"([€$£]|[A-Za-z]{3})?\s*([+-]?[\d\s\u00a0\u202f.,]+)\s*([€$£%]|[A-Za-z]{1,8})?",
        text,
    )
    if not match or (match[1] and match[3]):
        raise ValueError(f"Not an unambiguous numeric cell: {value!r}")
    unit = match[1] or match[3] or ""
    unit = {"€": "EUR"}.get(unit, unit.upper())
    number = match[2].strip()
    if re.search(r"\d\s+\d", number):
        if not re.fullmatch(
            r"[+-]?\d{1,3}(?:[\s\u00a0\u202f]+\d{3})+(?:[.,]\d+)?", number
        ):
            raise ValueError(f"Invalid digit grouping: {value!r}")
    number = re.sub(r"\s", "", number)
    if decimal_separator:
        grouping = "." if decimal_separator == "," else ","
        integer = number.split(decimal_separator)[0]
        if grouping in number:
            if not re.fullmatch(
                rf"[+-]?\d{{1,3}}(?:{re.escape(grouping)}\d{{3}})+", integer
            ):
                raise ValueError(f"Invalid digit grouping: {value!r}")
            number = number.replace(grouping, "")
        number = number.replace(decimal_separator, ".")
    elif "," in number and "." in number:
        separator = "," if number.rfind(",") > number.rfind(".") else "."
        return numeric_cell(value, separator)
    elif "," in number or "." in number:
        separator = "," if "," in number else "."
        parts = number.split(separator)
        if len(parts) != 2 or (len(parts[1]) == 3 and parts[0].lstrip("+-") != "0"):
            raise ValueError(f"Specify the decimal separator for {value!r}.")
        number = number.replace(separator, ".")
    try:
        amount = Decimal(number)
    except InvalidOperation as exc:
        raise ValueError(f"Invalid number: {value!r}") from exc
    if not amount.is_finite():
        raise ValueError("Non-finite numbers are unsupported.")
    return NumericCell(amount, unit)


@dataclass(frozen=True)
class TableResult:
    table: MarkdownTable
    operation: str
    column: str | None
    aggregate: str | None = None
    truncated: bool = False
    matched_row_count: int = 0


def execute_table_query(
    table: MarkdownTable, query: TableQuery, *, complete: bool
) -> TableResult:
    lookup = {
        normalized_label(header): index for index, header in enumerate(table.headers)
    }

    def column_index(name: str) -> int:
        try:
            return lookup[normalized_label(name)]
        except KeyError as exc:
            raise ValueError(f"Unknown column: {name!r}") from exc

    if query.operation != "select" and not complete:
        raise ValueError("This operation requires a complete extracted table.")
    if query.operation in {"count", "sum", "mean"} and query.limit is not None:
        raise ValueError("A row limit cannot be applied to an aggregate result.")
    if query.operation in {"min", "max", "sum", "mean", "sort"} and not query.column:
        raise ValueError("This operation requires a column.")
    target = column_index(query.column) if query.column else None
    for name in query.select_columns + query.identity_columns:
        column_index(name)
    for rule in query.filters:
        column_index(rule.column)
        if rule.operator in {"lt", "le", "gt", "ge"}:
            numeric_cell(rule.value, query.decimal_separator)

    def matches(row: tuple[str, ...], rule: TableFilter) -> bool:
        cell = row[column_index(rule.column)]
        if rule.operator in {"eq", "ne", "contains"}:
            left, right = normalized_label(cell), normalized_label(rule.value)
            return {
                "eq": left == right,
                "ne": left != right,
                "contains": right in left,
            }[rule.operator]
        left = numeric_cell(cell, query.decimal_separator)
        right = numeric_cell(rule.value, query.decimal_separator)
        if left.unit != right.unit:
            raise ValueError("Filter units differ from the column's units.")
        return {
            "lt": left.amount < right.amount,
            "le": left.amount <= right.amount,
            "gt": left.amount > right.amount,
            "ge": left.amount >= right.amount,
        }[rule.operator]

    rows = [
        row for row in table.rows if all(matches(row, rule) for rule in query.filters)
    ]
    count = len(rows)
    aggregate = None
    if query.operation == "count":
        aggregate = str(count)
    elif query.operation in {"min", "max", "sum", "mean", "sort"}:
        if not rows:
            raise ValueError("No rows match; no numeric result is available.")
        numbers = [numeric_cell(row[target], query.decimal_separator) for row in rows]
        if len({number.unit for number in numbers}) != 1:
            raise ValueError("The column contains incompatible units.")
        if query.operation in {"min", "max"}:
            best = (min if query.operation == "min" else max)(
                number.amount for number in numbers
            )
            rows = [
                row
                for row, number in zip(rows, numbers, strict=True)
                if number.amount == best
            ]
        elif query.operation == "sort":
            rows = [
                row
                for _, row in sorted(
                    zip([n.amount for n in numbers], rows, strict=True),
                    key=lambda item: item[0],
                    reverse=query.descending,
                )
            ]
        else:
            with localcontext() as context:
                # Exact sums even when cells exceed Decimal's default precision.
                context.prec = max(
                    28,
                    sum(
                        len(number.amount.as_tuple().digits)
                        + abs(number.amount.as_tuple().exponent)
                        for number in numbers
                    )
                    + 10,
                )
                total = sum((number.amount for number in numbers), Decimal(0))
                result_number = format(total, "f")
                if query.operation == "mean":
                    fraction = Fraction(total) / count
                    denominator = fraction.denominator
                    for prime in (2, 5):
                        while denominator % prime == 0:
                            denominator //= prime
                    result_number = (
                        format(total / count, "f")
                        if denominator == 1
                        else f"{fraction.numerator}/{fraction.denominator}"
                    )
            aggregate = f"{result_number} {numbers[0].unit}".strip()

    # Full winning rows preserve identifiers and make extrema independently auditable.
    columns = list(table.headers)
    if query.operation == "select" and query.select_columns:
        identities = query.identity_columns or [table.headers[0]]
        columns = list(
            dict.fromkeys(
                table.headers[column_index(name)]
                for name in identities + query.select_columns
            )
        )
    indices = [column_index(name) for name in columns]
    matched_result_count = len(rows)
    truncated = query.limit is not None and matched_result_count > query.limit
    if query.limit is not None:
        rows = rows[: query.limit]
    result_table = MarkdownTable(
        tuple(table.headers[i] for i in indices),
        tuple(tuple(row[i] for i in indices) for row in rows),
    )
    return TableResult(
        result_table,
        query.operation,
        query.column,
        aggregate,
        truncated,
        matched_result_count,
    )
