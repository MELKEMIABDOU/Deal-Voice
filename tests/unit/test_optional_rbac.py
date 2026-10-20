from deal_voice.ingestion.audio import WhisperXTranscriber


def test_audio_document_does_not_require_roles(monkeypatch, tmp_path):
    audio_path = tmp_path / "call.mp3"
    audio_path.write_bytes(b"audio")

    monkeypatch.setattr(
        WhisperXTranscriber,
        "transcribe_and_diarize",
        lambda self, audio_path, hf_token: [
            type(
                "Turn",
                (),
                {
                    "speaker": "SPEAKER_00",
                    "text": "Bonjour.",
                    "start_time": 0.0,
                    "end_time": 1.0,
                },
            )()
        ],
    )

    document = WhisperXTranscriber().transcribe_to_document(
        audio_path,
        hf_token="test-token",
    )

    assert document.metadata["file_type"] == "audio"
    assert "allowed_roles" not in document.metadata
    assert "**SPEAKER_00**" in document.page_content
