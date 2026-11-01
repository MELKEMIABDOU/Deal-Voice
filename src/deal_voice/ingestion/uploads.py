"""Shared upload ingestion for the web API and the optional Streamlit client."""

import logging
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.documents import Document

from deal_voice.config import settings
from deal_voice.ingestion.audio import WhisperXTranscriber
from deal_voice.ingestion.document import EnterpriseDocumentParser
from deal_voice.ingestion.indexer import EnterpriseIndexer

logger = logging.getLogger(__name__)
SUPPORTED_SUFFIXES = frozenset({".pdf", ".mp3", ".wav", ".m4a", ".flac"})


def _upload_directory(document_id: str, kind: str) -> Path:
    if str(UUID(document_id)) != document_id:
        raise ValueError("Invalid upload identifier.")
    root = (Path("data") / kind).resolve()
    directory = (root / document_id).resolve()
    if directory.parent != root:
        raise ValueError("The upload directory must remain inside its data folder.")
    return directory


def cleanup_upload(document_id: str) -> None:
    """Remove only the two validated directories of a failed, newly created upload."""
    directories = [
        _upload_directory(document_id, kind) for kind in ("raw", "processed")
    ]
    for directory in directories:
        shutil.rmtree(directory, ignore_errors=True)


def _parse_upload(
    file_path: Path, document_id: str, content_sha256: str, whisper_model: str
) -> list[Document]:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        documents = EnterpriseDocumentParser(default_roles=[]).parse_pdf(
            file_path, allowed_roles=None
        )
    elif suffix in SUPPORTED_SUFFIXES:
        if not settings.hf_token:
            raise ValueError(
                "HF_TOKEN est nécessaire pour la diarisation audio. "
                "Ajoutez-le dans .env avant l'upload."
            )
        documents = [
            WhisperXTranscriber(
                model_name=whisper_model,
                device="cpu",
                compute_type="int8",
                language="fr",
                batch_size=1,
                max_audio_duration_minutes=settings.max_audio_duration_minutes,
                worker_timeout_seconds=settings.audio_worker_timeout_seconds,
                worker_cpu_threads=settings.audio_worker_cpu_threads,
            ).transcribe_to_document(
                file_path,
                hf_token=settings.hf_token,
                output_dir=Path("data") / "processed" / document_id,
            )
        ]
    else:
        raise ValueError("Format non supporté. Utilisez PDF, MP3, WAV, M4A ou FLAC.")
    for document in documents:
        document.metadata.update(
            {
                "document_id": document_id,
                "original_path": str(file_path),
                "content_sha256": content_sha256,
            }
        )
    return documents


def index_uploaded_batch(
    uploads: list[tuple[Path, str, str]],
    get_search_engine: Callable[[], Any],
    whisper_model: str = "medium",
) -> None:
    """Parse the batch before committing its index, and roll back new uploads on error."""
    if not uploads:
        return
    for path, document_id, _ in uploads:
        if Path(path).parent.resolve() != _upload_directory(document_id, "raw"):
            raise ValueError(
                "The uploaded file must remain inside its upload directory."
            )
        _upload_directory(document_id, "processed")
    indexer = None
    document_ids = [document_id for _, document_id, _ in uploads]
    try:
        documents = [
            document
            for path, document_id, fingerprint in uploads
            for document in _parse_upload(
                Path(path), document_id, fingerprint, whisper_model
            )
        ]
        engine = get_search_engine()
        indexer = EnterpriseIndexer(
            db_path=settings.qdrant_storage_path,
            client=engine.client,
            embeddings=engine.embeddings,
        )
        indexer.index_documents(documents)
    except Exception:
        if indexer is not None:
            try:
                indexer.delete_documents(document_ids)
            except Exception:
                logger.exception("Failed to roll back the upload index")
        for document_id in document_ids:
            cleanup_upload(document_id)
        raise


def index_uploaded_file(
    file_path: Path,
    document_id: str,
    content_sha256: str,
    search_engine: Any,
    whisper_model: str = "medium",
) -> None:
    """Index one already-persisted file using the application's shared resources."""
    index_uploaded_batch(
        [(file_path, document_id, content_sha256)], lambda: search_engine, whisper_model
    )
