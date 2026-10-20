from pathlib import Path
from typing import Any, Dict, List
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain_core.documents import Document
from deal_voice.config import settings


class EnterpriseDocumentParser:
    """Parseur PDF haute fidélité basé sur IBM Docling."""

    def __init__(self, default_roles: List[str] | None = None):
        self.default_roles = default_roles or settings.default_roles

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_table_structure = True

        self.converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            },
        )

    def parse_pdf(
        self, file_path: str | Path, allowed_roles: List[str] | None = None
    ) -> List[Document]:
        pdf_path = Path(file_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF non trouvé : {pdf_path}")

        roles = allowed_roles or self.default_roles
        conversion_result = self.converter.convert(pdf_path)
        structured_markdown = conversion_result.document.export_to_markdown()

        metadata: Dict[str, Any] = {
            "source": pdf_path.name,
            "file_type": "pdf",
            "allowed_roles": roles,
        }

        return [Document(page_content=structured_markdown, metadata=metadata)]

