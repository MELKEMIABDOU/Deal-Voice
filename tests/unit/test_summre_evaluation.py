import wave

import numpy as np

from scripts.evaluate_summre import result_row, weighted_summary, write_audio_sample
from deal_voice.evaluation.transcription import TranscriptionBenchmarkResult


def test_write_audio_sample_materializes_waveform(tmp_path):
    path = write_audio_sample(
        {"array": np.array([0.0, 0.5, -0.5]), "sampling_rate": 16_000},
        tmp_path / "track.wav",
    )

    with wave.open(str(path), "rb") as audio:
        assert audio.getframerate() == 16_000
        assert audio.getnchannels() == 1
        assert audio.getnframes() == 3


def test_result_rows_exclude_transcript_text_and_summary_is_weighted():
    benchmark = TranscriptionBenchmarkResult(
        audio="temporary.wav",
        model="small",
        language="fr",
        device="cpu",
        compute_type="int8",
        duration_seconds=60,
        transcription_seconds=30,
        real_time_factor=0.5,
        segment_count=4,
        word_count=10,
        character_count=50,
        repeated_phrase_count=0,
        text="hypothesis should not be written to the report",
        wer=0.5,
        cer=0.25,
    )
    row = result_row(
        benchmark,
        {
            "audio_id": "meeting-speaker",
            "meeting_id": "meeting",
            "speaker_id": "speaker",
            "transcript": "un deux trois quatre",
        },
    )

    assert "text" not in row
    assert row["audio"] == "meeting-speaker"
    assert row["reference_word_count"] == 4
    summary = weighted_summary([row])[0]
    assert summary["weighted_wer"] == 0.5
    assert summary["weighted_cer"] == 0.25
    assert summary["real_time_factor"] == 0.5
