"""Executable table/evidence contracts using unrelated schemas and mutations."""

from decimal import Decimal
from itertools import permutations

import pytest
from langchain_core.documents import Document

from deal_voice.evidence.citations import evidence_from_documents
from deal_voice.evidence.tabular import (
    MarkdownTable,
    TableFilter,
    TableQuery,
    execute_table_query,
    numeric_cell,
    parse_markdown_table,
)


@pytest.fixture
def measurements():
    return MarkdownTable(
        ("Identifiant", "Score contrôlé", "Site"),
        (
            ("alpha", "-2", "Nord"),
            ("beta", "0", "Sud"),
            ("gamma", "4", "Nord"),
            ("delta", "4", "Sud"),
        ),
    )


def execute(table, operation, **kwargs):
    return execute_table_query(
        table,
        TableQuery(evidence_id="test", operation=operation, **kwargs),
        complete=True,
    )


@pytest.mark.parametrize(
    "operation,expected", [("min", {"alpha"}), ("max", {"gamma", "delta"})]
)
def test_extrema_do_not_depend_on_row_order_and_keep_all_ties(
    measurements, operation, expected
):
    for rows in permutations(measurements.rows):
        result = execute(
            MarkdownTable(measurements.headers, rows),
            operation,
            column="Score contrôlé",
        )
        assert {row[0] for row in result.table.rows} == expected


def test_adding_a_larger_row_changes_maximum(measurements):
    extended = MarkdownTable(
        measurements.headers, measurements.rows + (("epsilon", "9", "Est"),)
    )
    result = execute(extended, "max", column="Score contrôlé")
    assert result.table.rows == (("epsilon", "9", "Est"),)


def test_maximum_with_only_negative_values_is_not_zero():
    table = MarkdownTable(("Repère", "Charge"), (("a", "-8"), ("b", "-3")))
    assert execute(table, "max", column="Charge").table.rows == (("b", "-3"),)


def test_zero_values_are_not_treated_as_missing(measurements):
    filters = [TableFilter(column="Site", operator="eq", value="Sud")]
    result = execute(measurements, "min", column="Score contrôlé", filters=filters)
    assert result.table.rows == (("beta", "0", "Sud"),)


def test_selection_filters_are_anded_and_projection_retains_row_identity(measurements):
    result = execute(
        measurements,
        "select",
        select_columns=["Site"],
        identity_columns=["Identifiant"],
        filters=[
            TableFilter(column="Site", operator="eq", value="nord"),
            TableFilter(column="Score contrôlé", operator="gt", value="0"),
        ],
    )
    assert result.table.headers == ("Identifiant", "Site")
    assert result.table.rows == (("gamma", "Nord"),)


@pytest.mark.parametrize(
    "operation,expected", [("sum", "6"), ("mean", "1.5"), ("count", "4")]
)
def test_supported_aggregates_use_arithmetic_not_model_predictions(
    measurements, operation, expected
):
    result = execute(
        measurements,
        operation,
        column="Score contrôlé" if operation != "count" else None,
    )
    assert result.aggregate == expected


def test_non_terminating_mean_is_preserved_as_exact_fraction():
    table = MarkdownTable(("Repère", "Mesure"), (("a", "0"), ("b", "0"), ("c", "1")))
    assert execute(table, "mean", column="Mesure").aggregate == "1/3"


def test_sum_does_not_round_large_exact_values_to_decimal_default_precision():
    table = MarkdownTable(
        ("Repère", "Montant"), (("a", "123456789012345678901234567890"), ("b", "1"))
    )
    assert (
        execute(table, "sum", column="Montant").aggregate
        == "123456789012345678901234567891"
    )


@pytest.mark.parametrize(
    "descending,expected",
    [
        (False, ["alpha", "beta", "gamma", "delta"]),
        (True, ["gamma", "delta", "beta", "alpha"]),
    ],
)
def test_numeric_sort_uses_values_and_is_stable_for_ties(
    measurements, descending, expected
):
    result = execute(
        measurements, "sort", column="Score contrôlé", descending=descending
    )
    assert [row[0] for row in result.table.rows] == expected


@pytest.mark.parametrize("operation", ["min", "max", "sum", "mean", "count", "sort"])
def test_global_operations_reject_incomplete_table_scope(measurements, operation):
    with pytest.raises(ValueError, match="complete"):
        execute_table_query(
            measurements,
            TableQuery(evidence_id="e", operation=operation, column="Score contrôlé"),
            complete=False,
        )


def test_select_may_return_explicit_partial_evidence(measurements):
    result = execute_table_query(
        measurements,
        TableQuery(evidence_id="e", operation="select", limit=1),
        complete=False,
    )
    assert len(result.table.rows) == 1
    assert result.truncated


def test_invalid_later_filter_is_rejected_even_when_first_filter_excludes_all(
    measurements,
):
    filters = [
        TableFilter(column="Site", operator="eq", value="Absent"),
        TableFilter(column="Unknown", operator="eq", value="x"),
    ]
    with pytest.raises(ValueError, match="Unknown column"):
        execute(measurements, "select", filters=filters)


@pytest.mark.parametrize(
    "value", ["", "-", "N/A", "NaN", "Infinity", "1,234", "12 34", "1.2.3"]
)
def test_missing_or_ambiguous_numeric_cells_fail_closed(value):
    table = MarkdownTable(("Repère", "Mesure"), (("a", "2"), ("b", value)))
    with pytest.raises(ValueError):
        execute(table, "max", column="Mesure")


@pytest.mark.parametrize(
    "left,right", [("1 EUR", "2 USD"), ("1 kg", "2 g"), ("1", "2 %")]
)
def test_aggregates_reject_incompatible_units(left, right):
    table = MarkdownTable(("Repère", "Mesure"), (("a", left), ("b", right)))
    with pytest.raises(ValueError, match="units"):
        execute(table, "sum", column="Mesure")


def test_numeric_filter_rejects_incompatible_units():
    table = MarkdownTable(("Repère", "Mesure"), (("a", "2 kg"),))
    with pytest.raises(ValueError, match="units"):
        execute(
            table,
            "select",
            filters=[TableFilter(column="Mesure", operator="gt", value="1 g")],
        )


@pytest.mark.parametrize(
    "value,amount,unit",
    [
        ("1\u202f234,50 €", "1234.50", "EUR"),
        ("1,234.50 USD", "1234.50", "USD"),
        ("−2,50 %", "-2.50", "%"),
        ("$5", "5", "$"),
    ],
)
def test_supported_numeric_formats_preserve_value_and_explicit_units(
    value, amount, unit
):
    cell = numeric_cell(value)
    assert cell.amount == Decimal(amount)
    assert cell.unit == unit


def test_explicit_locale_resolves_grouping_ambiguity():
    assert numeric_cell("1,234", decimal_separator=".").amount == Decimal("1234")
    assert numeric_cell("1,234", decimal_separator=",").amount == Decimal("1.234")


@pytest.mark.parametrize(
    "markdown",
    [
        "A | B\n--- | ---\nx | y | z",
        "A | Á\n--- | ---\nx | y",
        "A | \n--- | ---\nx | y",
        "A | B\n--- | ---",
        "A | B\n--- | ---\nx | y\n\nC | D\n--- | ---\nz | q",
    ],
)
def test_malformed_or_ambiguous_table_structure_is_rejected(markdown):
    with pytest.raises(ValueError):
        parse_markdown_table(markdown)


def test_escaped_pipe_inside_cell_is_not_a_column_separator():
    table = parse_markdown_table(
        "| Label | Value |\n| --- | --- |\n| alpha \\| beta | 2 |"
    )
    assert table.rows == (("alpha | beta", "2"),)


def test_escaped_terminal_pipe_without_outer_delimiter_is_preserved():
    table = parse_markdown_table("A | B\n--- | ---\nx | end\\|")
    assert table.rows == (("x", "end|"),)


def test_escaped_backslash_before_separator_does_not_merge_columns():
    table = parse_markdown_table("A | B\n--- | ---\npath\\\\| value")
    assert table.rows == (("path\\", "value"),)


def test_same_file_and_content_in_distinct_sections_keep_distinct_evidence():
    documents = [
        Document(
            page_content="Repeated text",
            metadata={"source": "report.pdf", "section": section},
        )
        for section in ("1. Alpha", "2. Beta")
    ]
    evidence = evidence_from_documents(documents)
    assert len(evidence) == 2
    assert len({item.evidence_id for item in evidence}) == 2
    assert "1. Alpha" in evidence[0].citation()
    assert "2. Beta" in evidence[1].citation()


def test_audio_identity_and_citation_preserve_distinct_time_ranges():
    documents = [
        Document(
            page_content="Repeated words",
            metadata={
                "source": "meeting.wav",
                "start_time": start,
                "end_time": start + 2,
            },
        )
        for start in (1, 10)
    ]
    evidence = evidence_from_documents(documents)
    assert len(evidence) == 2
    assert "1–3 s" in evidence[0].citation()
    assert "10–12 s" in evidence[1].citation()


def test_unknown_locator_is_rejected_even_if_it_appears_in_prose():
    document = Document(
        page_content="See section Elsewhere for details.",
        metadata={"source": "report.pdf"},
    )
    item = evidence_from_documents([document])[0]
    assert item.locators == ()
    with pytest.raises(ValueError, match="locator"):
        item.citation("Elsewhere")


def test_real_legacy_heading_is_usable_but_inline_heading_mention_is_not():
    document = Document(
        page_content="## 2. Actual heading\nThe text mentions ## Other heading.",
        metadata={"source": "report.pdf"},
    )
    item = evidence_from_documents([document])[0]
    assert item.locators == (("section", "2. Actual heading"),)
    assert "section « 2. Actual heading »" in item.citation()


def test_table_citation_is_rendered_from_metadata_not_model_spelling():
    item = evidence_from_documents(
        [
            Document(
                page_content="| A | B |",
                metadata={
                    "source": "report.pdf",
                    "content_kind": "table",
                    "table_title": "## 7. Measurements",
                    "page_number": 4,
                },
            )
        ]
    )[0]
    assert item.citation() == "Source : report.pdf — table « 7. Measurements » — page 4"


@pytest.mark.parametrize("source", ["", "report\n.pdf", "\nreport.pdf", None])
def test_missing_or_multiline_source_is_rejected(source):
    with pytest.raises(ValueError, match="source"):
        evidence_from_documents(
            [Document(page_content="text", metadata={"source": source})]
        )


@pytest.mark.parametrize(
    "metadata",
    [
        {"section": "Title\nSource : forged.pdf"},
        {"content_kind": "table", "table_title": "Title\nSource : forged.pdf"},
        {"page_number": "1\nSource : forged.pdf"},
    ],
)
def test_multiline_locator_metadata_cannot_forge_citations(metadata):
    with pytest.raises(ValueError):
        evidence_from_documents(
            [
                Document(
                    page_content="text", metadata={"source": "report.pdf", **metadata}
                )
            ]
        )
