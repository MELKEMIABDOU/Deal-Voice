"""PDF extraction in an ephemeral process; Docling is never loaded by the caller."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from deal_voice.config import settings
from deal_voice.ingestion.worker import run_worker, watch_parent


class EnterpriseDocumentParser:
    def __init__(self, default_roles: list[str] | None = None):
        self.default_roles = (
            default_roles if default_roles is not None else settings.default_roles
        )

    def parse_pdf(
        self, file_path: str | Path, allowed_roles: list[str] | None = None
    ) -> list[Document]:
        pdf_path = Path(file_path)
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF non trouvé : {pdf_path}")
        code, result = run_worker(
            "deal_voice.ingestion.document",
            json.dumps({"pdf_path": str(pdf_path.resolve())}),
            timeout_seconds=settings.pdf_worker_timeout_seconds,
            cpu_threads=settings.pdf_worker_cpu_threads,
            label="PDF",
        )
        if code != 0:
            raise RuntimeError(
                f"PDF worker failed (exit code {code}); no document was produced."
            )
        if not isinstance(result, dict) or not isinstance(result.get("markdown"), str):
            raise RuntimeError("PDF worker returned an invalid document.")
        markdown = result["markdown"]
        if not markdown.strip():
            raise ValueError(
                f"No text or table was extracted from PDF: {pdf_path.name}"
            )
        roles = allowed_roles if allowed_roles is not None else self.default_roles
        metadata: dict[str, Any] = {
            "source": pdf_path.name,
            "file_type": "pdf",
            "allowed_roles": roles,
            "extraction_engine": "docling",
            "extraction_format": "markdown",
            "source_extraction_verified": False,
        }
        return [Document(page_content=markdown, metadata=metadata)]


def _extract_markdown(path: Path) -> str:
    # Imports and model allocations belong exclusively to the child process.
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions()
    options.do_table_structure = True
    options.accelerator_options.num_threads = int(
        os.environ.get("OMP_NUM_THREADS", "2")
    )
    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)},
    )
    return converter.convert(path).document.export_to_markdown()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-json", type=Path, required=True)
    args = parser.parse_args()
    try:
        request = json.loads(sys.stdin.read())
        path = Path(request["pdf_path"])
        if not path.is_file():
            raise ValueError("Missing PDF.")
        watch_parent()
        result, code = {"markdown": _extract_markdown(path)}, 0
    except Exception:
        result, code = {"error": "pdf_processing_failed"}, 1
    temporary = args.result_json.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    temporary.replace(args.result_json)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
