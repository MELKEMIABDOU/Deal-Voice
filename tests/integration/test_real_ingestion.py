import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from deal_voice.config import settings
from deal_voice.ingestion.audio import WhisperXTranscriber
from deal_voice.ingestion.document import EnterpriseDocumentParser
from deal_voice.ingestion.indexer import EnterpriseIndexer


PROJECT_ROOT = Path(__file__).parents[2]
REAL_PDF = PROJECT_ROOT / "data" / "raw" / "Rapport_gestion_produits_Horizon_structure.pdf"
REAL_AUDIO = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Centre D appel Conversation entre le téléconseiller et le client Opérateur Télé.mp3"
)


def test_real_pdf_to_parent_child_chunks():
    parser = EnterpriseDocumentParser(default_roles=["sales", "admin"])
    documents = parser.parse_pdf(REAL_PDF)

    with TemporaryDirectory() as db_path:
        indexer = EnterpriseIndexer(
            db_path=db_path,
            collection_name="real_pdf_test",
        )
        indexer.index_documents(documents)
        child_count = indexer.client.count(
            collection_name="real_pdf_test",
            exact=True,
        ).count
        parent_count = len(list(indexer.docstore.yield_keys()))
        indexer.client.close()

    print(
        f"PDF: source_documents={len(documents)}, "
        f"parent_chunks={parent_count}, child_chunks={child_count}"
    )
    assert documents
    assert child_count > 1
    assert parent_count > 1


@pytest.mark.skipif(
    os.getenv("RUN_REAL_AUDIO_TESTS") != "1",
    reason="Set RUN_REAL_AUDIO_TESTS=1 to run the real WhisperX test.",
)
def test_real_audio_to_parent_child_chunks():
    if not settings.hf_token:
        pytest.skip("Set HF_TOKEN in .env to run diarization.")

    transcriber = WhisperXTranscriber(
        model_name=os.getenv("WHISPERX_MODEL", "base"),
        device=os.getenv("WHISPERX_DEVICE", "cpu"),
        compute_type=os.getenv("WHISPERX_COMPUTE_TYPE", "int8"),
        language="fr",
    )
    document = transcriber.transcribe_to_document(
        REAL_AUDIO,
        hf_token=settings.hf_token,
    )

    with TemporaryDirectory() as db_path:
        indexer = EnterpriseIndexer(
            db_path=db_path,
            collection_name="real_audio_test",
        )
        indexer.index_documents([document])
        child_count = indexer.client.count(
            collection_name="real_audio_test",
            exact=True,
        ).count
        parent_count = len(list(indexer.docstore.yield_keys()))
        indexer.client.close()

    print(
        f"Audio: speaker_count={document.metadata['speaker_count']}, "
        f"parent_chunks={parent_count}, child_chunks={child_count}"
    )
    assert document.metadata["speaker_count"] >= 1
    assert "**SPEAKER_" in document.page_content
    assert "allowed_roles" not in document.metadata
    assert child_count > 0
    assert parent_count > 0
