import pytest
from langchain_core.documents import Document

from deal_voice.ingestion.indexer import EnterpriseIndexer


def pdf_document(content):
    return Document(
        page_content=content,
        metadata={"source": "example.pdf", "document_id": "pdf-1", "file_type": "pdf"},
    )


def test_pdf_keeps_explanations_and_tables_linked_to_their_section():
    document = pdf_document(
        "# Operations\n"
        "## Capacity\n"
        "Capacity is measured in litres.\n"
        "| Site | Capacity |\n"
        "| --- | --- |\n"
        "| A | 12 |\n"
        "| B | 9 |\n"
        "Figures exclude the reserve tank.\n"
        "## Method\n"
        "Add the measured capacity and the reserve."
    )

    text, tables = EnterpriseIndexer._split_pdf_tables(document)

    assert len(tables) == 1
    assert len(text) == 2
    assert text[0].metadata["section"] == "Capacity"
    assert text[0].metadata["section_path"] == ["Operations", "Capacity"]
    assert text[0].metadata["section_id"] == tables[0].metadata["section_id"]
    assert "exclude the reserve" in text[0].page_content
    assert "Add the measured" not in text[0].page_content
    assert text[1].metadata["section"] == "Method"
    assert tables[0].metadata["table_columns"] == ["Site", "Capacity"]
    assert tables[0].metadata["table_row_count"] == 2
    assert tables[0].metadata["table_complete"] is True
    assert tables[0].metadata["table_structure_valid"] is True
    assert tables[0].metadata["table_completeness_scope"] == "extracted_block"
    assert tables[0].metadata["table_source_complete"] is False
    assert "page" not in tables[0].metadata


@pytest.mark.parametrize(
    "table",
    [
        "| Site | Capacity |\n| --- | --- |\n| A | 12 | extra |",
        "| Site | Site |\n| --- | --- |\n| A | B |",
        "| Site |  |\n| --- | --- |\n| A | 12 |",
    ],
)
def test_invalid_table_is_preserved_as_text_without_claiming_valid_grid(table):
    text, tables = EnterpriseIndexer._split_pdf_tables(
        pdf_document("## Data\n" + table)
    )

    assert tables == []
    assert len(text) == 1
    assert text[0].metadata["content_kind"] == "text"
    assert text[0].metadata["table_structure_valid"] is False
    assert text[0].metadata["table_validation_error"]
    assert table in text[0].page_content
    assert "table_complete" not in text[0].metadata


def test_two_tables_have_distinct_ids_but_keep_their_shared_section():
    text, tables = EnterpriseIndexer._split_pdf_tables(
        pdf_document(
            "## Samples\n"
            "| Sample | Mass |\n| --- | --- |\n| First | 2 |\n\n"
            "Both samples use the same balance.\n\n"
            "| Sample | Mass |\n| --- | --- |\n| Second | 4 |"
        )
    )

    assert len(tables) == 2
    assert tables[0].metadata["table_id"] != tables[1].metadata["table_id"]
    assert tables[0].metadata["section_id"] == tables[1].metadata["section_id"]
    assert "same balance" in text[0].page_content


def test_sections_are_preserved_even_without_a_table():
    text, tables = EnterpriseIndexer._split_pdf_tables(
        pdf_document(
            "## Terms\nPayment in 30 days.\n## Contact\nAsk the account manager."
        )
    )

    assert tables == []
    assert [document.metadata["section"] for document in text] == ["Terms", "Contact"]
    assert "Payment" not in text[1].page_content
