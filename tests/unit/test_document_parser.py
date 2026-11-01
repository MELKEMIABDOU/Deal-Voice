import builtins
import os

import pytest

from deal_voice.ingestion import document as document_module
from deal_voice.ingestion.document import EnterpriseDocumentParser


def test_pdf_parent_does_not_import_docling(monkeypatch, tmp_path):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "docling" or name.startswith("docling."):
            pytest.fail("Docling must only be imported in the extraction worker.")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    parser = parser_returning("A valid paragraph.", monkeypatch)
    path = tmp_path / "file.pdf"
    path.touch()
    assert parser.parse_pdf(path)[0].page_content == "A valid paragraph."


@pytest.mark.parametrize("code,result", [(1, {}), (0, None), (0, {"markdown": 7})])
def test_pdf_failure_never_produces_a_document(monkeypatch, tmp_path, code, result):
    monkeypatch.setattr(
        document_module, "run_worker", lambda *args, **kwargs: (code, result)
    )
    path = tmp_path / "file.pdf"
    path.touch()
    with pytest.raises(RuntimeError):
        EnterpriseDocumentParser().parse_pdf(path)


def test_pdf_round_trip_uses_a_real_child_without_real_models(monkeypatch, tmp_path):
    # Inject only a tiny Docling stand-in into the CHILD Python startup.
    # The production subprocess/Job Object/file protocol still runs unchanged.
    (tmp_path / "sitecustomize.py").write_text(
        "import sys\nfrom types import SimpleNamespace as S\n"
        "sys.modules['docling'] = S()\n"
        "sys.modules['docling.datamodel'] = S()\n"
        "sys.modules['docling.datamodel.base_models'] = S(InputFormat=S(PDF='pdf'))\n"
        "sys.modules['docling.datamodel.pipeline_options'] = S(\n"
        "    PdfPipelineOptions=lambda: S(accelerator_options=S(num_threads=1)))\n"
        "def converter(**kwargs):\n"
        "    assert kwargs['format_options']['pdf'].pipeline_options.do_table_structure\n"
        "    return S(convert=lambda path: S(document=S(\n"
        "        export_to_markdown=lambda: '## Terms\\nPayment in 30 days.')))\n"
        "sys.modules['docling.document_converter'] = S(\n"
        "    DocumentConverter=converter, PdfFormatOption=lambda **kwargs: S(**kwargs))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join([str(tmp_path), os.environ.get("PYTHONPATH", "")])
    )
    path = tmp_path / "file.pdf"
    path.write_bytes(b"not decoded by the fake model")
    result = EnterpriseDocumentParser(default_roles=[]).parse_pdf(path)
    assert result[0].page_content == "## Terms\nPayment in 30 days."
    assert result[0].metadata["source"] == "file.pdf"
    assert result[0].metadata["allowed_roles"] == []


def parser_returning(markdown, monkeypatch):
    """Exercise parser contracts without loading Docling inference models."""
    monkeypatch.setattr(
        document_module,
        "run_worker",
        lambda *args, **kwargs: (0, {"markdown": markdown}),
    )
    return EnterpriseDocumentParser(default_roles=["sales", "admin"])


def test_pdf_extraction_preserves_metadata_without_inventing_pages(
    tmp_path, monkeypatch
):
    pdf_path = tmp_path / "example.pdf"
    pdf_path.write_bytes(b"%PDF-test")
    parser = parser_returning("## Terms\nPayment in 30 days.", monkeypatch)

    documents = parser.parse_pdf(pdf_path)

    assert len(documents) == 1
    assert documents[0].page_content == "## Terms\nPayment in 30 days."
    assert documents[0].metadata["file_type"] == "pdf"
    assert documents[0].metadata["allowed_roles"] == ["sales", "admin"]
    assert documents[0].metadata["extraction_engine"] == "docling"
    assert documents[0].metadata["source_extraction_verified"] is False
    assert "page" not in documents[0].metadata


def test_empty_pdf_extraction_fails_explicitly(tmp_path, monkeypatch):
    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_bytes(b"%PDF-test")

    with pytest.raises(ValueError, match="No text or table"):
        parser_returning("  \n", monkeypatch).parse_pdf(pdf_path)


def test_explicit_empty_permissions_are_not_replaced_with_defaults(
    tmp_path, monkeypatch
):
    pdf_path = tmp_path / "restricted.pdf"
    pdf_path.write_bytes(b"%PDF-test")

    documents = parser_returning("Restricted content", monkeypatch).parse_pdf(
        pdf_path, allowed_roles=[]
    )

    assert documents[0].metadata["allowed_roles"] == []
