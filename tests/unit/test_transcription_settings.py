from pathlib import Path

import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from deal_voice.config import Settings, settings
from deal_voice.ingestion import uploads


def test_cuda_whisper_profile_is_valid():
    configured = Settings(
        _env_file=None,
        whisper_model_name="large-v3",
        whisper_device="cuda",
        whisper_compute_type="float16",
        whisper_language="fr",
        whisper_batch_size=1,
    )

    assert configured.whisper_model_name == "large-v3"
    assert configured.whisper_device == "cuda"
    assert configured.whisper_compute_type == "float16"


def test_cpu_rejects_half_precision():
    with pytest.raises(ValidationError, match="CPU transcription"):
        Settings(
            _env_file=None,
            whisper_device="cpu",
            whisper_compute_type="float16",
        )


def test_audio_upload_uses_runtime_whisper_settings(monkeypatch, tmp_path: Path):
    captured = {}

    class Transcriber:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def transcribe_to_document(self, *args, **kwargs):
            return Document(page_content="transcript", metadata={"source": "call.mp3"})

    monkeypatch.setattr(uploads, "WhisperXTranscriber", Transcriber)
    monkeypatch.setattr(settings, "hf_token", "token")
    monkeypatch.setattr(settings, "whisper_model_name", "large-v3")
    monkeypatch.setattr(settings, "whisper_device", "cuda")
    monkeypatch.setattr(settings, "whisper_compute_type", "float16")
    monkeypatch.setattr(settings, "whisper_language", "fr")
    monkeypatch.setattr(settings, "whisper_batch_size", 1)
    path = tmp_path / "call.mp3"
    path.write_bytes(b"audio")

    documents = uploads._parse_upload(
        path,
        "45d8b558-e83e-42ca-a238-b9fcb038f477",
        "digest",
        None,
    )

    assert captured["model_name"] == "large-v3"
    assert captured["device"] == "cuda"
    assert captured["compute_type"] == "float16"
    assert captured["language"] == "fr"
    assert captured["batch_size"] == 1
    assert documents[0].metadata["content_sha256"] == "digest"
