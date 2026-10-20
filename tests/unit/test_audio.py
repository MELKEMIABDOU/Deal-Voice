from deal_voice.ingestion.audio import CallTurn, WhisperXTranscriber


def test_whisper_segments_are_merged_into_speaker_turns():
    segments = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0, "text": "Bonjour."},
        {"speaker": "SPEAKER_00", "start": 1.1, "end": 2.0, "text": "Comment allez-vous ?"},
        {"speaker": "SPEAKER_01", "start": 2.1, "end": 3.0, "text": "Très bien."},
    ]

    turns = WhisperXTranscriber._segments_to_turns(segments)

    assert len(turns) == 2
    assert turns[0].speaker == "SPEAKER_00"
    assert turns[0].text == "Bonjour. Comment allez-vous ?"
    assert turns[0].start_time == 0.0
    assert turns[0].end_time == 2.0
    assert turns[1].speaker == "SPEAKER_01"


def test_whisper_repeated_phrases_are_collapsed():
    turns = WhisperXTranscriber._segments_to_turns(
        [
            {
                "speaker": "SPEAKER_00",
                "start": 0.0,
                "end": 5.0,
                "text": "C'est très gentil. C'est très gentil. C'est très gentil.",
            },
            {
                "speaker": "SPEAKER_01",
                "start": 5.0,
                "end": 10.0,
                "text": "Si vous voulez, si vous voulez, si vous voulez.",
            },
        ]
    )

    assert turns[0].text == "C'est très gentil."
    assert turns[1].text == "Si vous voulez,"


def test_transcription_artifacts_are_persisted(tmp_path, monkeypatch):
    def fake_transcribe(self, audio_path, hf_token):
        return [
            CallTurn(
                speaker="SPEAKER_00",
                text="Bonjour.",
                start_time=0.0,
                end_time=1.5,
            )
        ]

    monkeypatch.setattr(
        WhisperXTranscriber,
        "transcribe_and_diarize",
        fake_transcribe,
    )
    audio_path = tmp_path / "call.mp3"
    audio_path.write_bytes(b"not decoded in this unit test")

    document = WhisperXTranscriber().transcribe_to_document(
        audio_path,
        hf_token="test-token",
        output_dir=tmp_path / "processed",
    )

    markdown_path = tmp_path / "processed" / "call_transcript.md"
    json_path = tmp_path / "processed" / "call_transcript.json"
    assert markdown_path.read_text(encoding="utf-8") == "**SPEAKER_00** `[00:00]`: Bonjour."
    assert '"speaker_count": 1' in json_path.read_text(encoding="utf-8")
    assert document.metadata["transcript_markdown"] == str(markdown_path)
    assert document.metadata["speakers"] == ["SPEAKER_00"]
    assert document.metadata["start_time"] == 0.0
    assert document.metadata["end_time"] == 1.5
