from deal_voice.ingestion.audio import CallTurn, WhisperXTranscriber


def test_public_transcription_uses_the_worker_and_preserves_configuration(
    monkeypatch, tmp_path
):
    from deal_voice.ingestion import audio_worker

    def worker(path, **kwargs):
        assert path == tmp_path / "call.wav"
        assert kwargs == {
            "hf_token": "test-token",
            "model_name": "medium",
            "device": "cpu",
            "compute_type": "int8",
            "language": "fr",
            "batch_size": 1,
            "max_duration_seconds": 120,
            "timeout_seconds": 19,
            "cpu_threads": 3,
        }
        return [{"speaker": "A", "text": "Bonjour.", "start_time": 0, "end_time": 1}]

    monkeypatch.setattr(audio_worker, "run_audio_worker", worker)
    transcriber = WhisperXTranscriber(
        language="fr",
        max_audio_duration_minutes=2,
        worker_timeout_seconds=19,
        worker_cpu_threads=3,
    )
    turns = transcriber.transcribe_and_diarize(tmp_path / "call.wav", "test-token")

    assert len(turns) == 1
    assert turns[0].to_dict() == {
        "speaker": "A",
        "text": "Bonjour.",
        "start_time": 0,
        "end_time": 1,
    }


def test_duration_is_checked_in_worker_before_loading_models(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace
    from unittest.mock import Mock

    import pytest

    from deal_voice.ingestion.audio import AudioDurationLimitError

    class LongAudio:
        def __len__(self):
            return 960_001  # Just over one minute at 16 kHz.

    load_model = Mock(side_effect=AssertionError("The ASR model must not be loaded."))
    monkeypatch.setitem(
        sys.modules,
        "whisperx",
        SimpleNamespace(load_audio=lambda path: LongAudio(), load_model=load_model),
    )
    path = tmp_path / "long.wav"
    path.touch()
    with pytest.raises(AudioDurationLimitError):
        WhisperXTranscriber()._transcribe_and_diarize_in_process(
            path, "test-token", max_duration_seconds=60
        )
    load_model.assert_not_called()


def test_stage_models_are_released_before_the_next_stage(monkeypatch, tmp_path):
    import sys
    import weakref
    from types import SimpleNamespace

    references = {}
    segments = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "Bonjour."}]

    class Model:
        def __init__(self):
            self.cycle = self  # Explicit collection is needed for cyclic objects.

        def transcribe(self, audio, **kwargs):
            return {"language": "fr", "segments": segments}

    def load_asr(*args, **kwargs):
        model = Model()
        references["asr"] = weakref.ref(model)
        return model

    def load_alignment(**kwargs):
        assert references["asr"]() is None
        model = Model()
        references["alignment"] = weakref.ref(model)
        return model, {}

    class Diarizer(Model):
        def __init__(self, **kwargs):
            assert references["alignment"]() is None
            super().__init__()
            references["diarizer"] = weakref.ref(self)

        def __call__(self, audio):
            return []

    monkeypatch.setitem(
        sys.modules,
        "whisperx",
        SimpleNamespace(
            load_audio=lambda path: [],
            load_model=load_asr,
            load_align_model=load_alignment,
            align=lambda *args, **kwargs: {"segments": segments},
            assign_word_speakers=lambda diarization, result: result,
        ),
    )
    monkeypatch.setitem(
        sys.modules, "whisperx.diarize", SimpleNamespace(DiarizationPipeline=Diarizer)
    )
    path = tmp_path / "fixture.wav"
    path.write_bytes(b"mock audio; no decoding in this lifecycle test")
    turns = WhisperXTranscriber(language="fr")._transcribe_and_diarize_in_process(
        path, "token"
    )

    assert turns[0].text == "Bonjour."
    assert all(reference() is None for reference in references.values())


def test_medium_cpu_is_the_default_transcription_configuration():
    transcriber = WhisperXTranscriber()

    assert transcriber.model_name == "medium"
    assert transcriber.device == "cpu"
    assert transcriber.compute_type == "int8"
    assert transcriber.batch_size == 1


def test_whisper_segments_are_merged_into_speaker_turns():
    segments = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0, "text": "Bonjour."},
        {
            "speaker": "SPEAKER_00",
            "start": 1.1,
            "end": 2.0,
            "text": "Comment allez-vous ?",
        },
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
    assert (
        markdown_path.read_text(encoding="utf-8")
        == "**SPEAKER_00** `[00:00]`: Bonjour."
    )
    assert '"speaker_count": 1' in json_path.read_text(encoding="utf-8")
    assert '"transcription_model": "medium"' in json_path.read_text(encoding="utf-8")
    assert document.metadata["transcription_model"] == "medium"
    assert document.metadata["transcription_device"] == "cpu"
    assert document.metadata["transcription_compute_type"] == "int8"
    assert document.metadata["transcript_markdown"] == str(markdown_path)
    assert document.metadata["speakers"] == ["SPEAKER_00"]
    assert document.metadata["start_time"] == 0.0
    assert document.metadata["end_time"] == 1.5
