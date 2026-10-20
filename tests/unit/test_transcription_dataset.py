from pathlib import Path

from deal_voice.evaluation.transcription import error_rate
from scripts.evaluate_transcription_dataset import find_pairs


def test_find_pairs_matches_audio_and_reference(tmp_path: Path):
    (tmp_path / "call.mp3").write_bytes(b"audio")
    (tmp_path / "call.txt").write_text("[Speaker 1]\nBonjour.", encoding="utf-8")

    assert find_pairs(tmp_path) == [(tmp_path / "call.mp3", tmp_path / "call.txt")]


def test_error_rate_ignores_speaker_labels():
    assert error_rate("[Speaker 1]\nBonjour le monde.", "Bonjour le monde.") == 0.0
