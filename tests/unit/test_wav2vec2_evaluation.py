from pathlib import Path

import pytest

from deal_voice.evaluation.wav2vec2 import benchmark_wav2vec2


def test_wav2vec2_requires_existing_audio(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        benchmark_wav2vec2(tmp_path / "missing.mp3")
