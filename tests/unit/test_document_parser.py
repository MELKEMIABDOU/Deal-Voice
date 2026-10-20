import pytest
from pathlib import Path
from deal_voice.ingestion.document import EnterpriseDocumentParser


def test_pdf_extraction_and_rbac_metadata():
    parser = EnterpriseDocumentParser(default_roles=["sales", "admin"])
    test_pdf_path = Path(__file__).parents[1] / "data" / "sample.pdf"

    if not test_pdf_path.exists():
        pytest.skip(f"No sample PDF found at {test_pdf_path}")

    docs = parser.parse_pdf(test_pdf_path)

    assert len(docs) > 0, "Parser failed to extract any pages."

    first_page_metadata = docs[0].metadata
    assert first_page_metadata["file_type"] == "pdf"
    assert "sales" in first_page_metadata["allowed_roles"]
    assert "## Page 1" in docs[0].page_content