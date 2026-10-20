"""Reproducible WhisperX transcription benchmarks."""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class TranscriptionBenchmarkResult:
    """Metrics collected for one model and one audio file."""

    audio: str
    model: str
    language: str | None
    device: str
    compute_type: str
    duration_seconds: float
    transcription_seconds: float
    real_time_factor: float
    segment_count: int
    word_count: int
    character_count: int
    repeated_phrase_count: int
    text: str
    wer: float | None = None
    cer: float | None = None
    reference_word_count: int | None = None
    reference_character_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_text(text: str) -> str:
    without_speakers = re.sub(
        r"(?im)^\s*\[speaker[^\]]+\]\s*",
        "",
        text,
    )
    return re.sub(r"[^\wÀ-ÿ']+", " ", without_speakers.lower()).strip()


def _edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_item in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_item in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def error_rate(reference: str, hypothesis: str, *, character: bool = False) -> float:
    """Calculate WER or CER after conservative text normalization."""
    normalized_reference = _normalize_text(reference)
    normalized_hypothesis = _normalize_text(hypothesis)
    if character:
        reference_units = list(normalized_reference.replace(" ", ""))
        hypothesis_units = list(normalized_hypothesis.replace(" ", ""))
    else:
        reference_units = normalized_reference.split()
        hypothesis_units = normalized_hypothesis.split()
    if not reference_units:
        raise ValueError("The reference transcript must contain text.")
    return round(
        _edit_distance(reference_units, hypothesis_units) / len(reference_units),
        4,
    )


def _count_repeated_phrases(text: str) -> int:
    """Count adjacent repeated word sequences without external dependencies."""
    normalized = re.sub(r"[^\wÀ-ÿ']+", " ", text.lower()).split()
    count = 0
    for size in range(1, min(8, len(normalized) // 2 + 1)):
        for index in range(len(normalized) - 2 * size + 1):
            left = normalized[index : index + size]
            right = normalized[index + size : index + 2 * size]
            if left == right:
                count += 1
    return count


def benchmark_model(
    audio_path: str | Path,
    model_name: str,
    *,
    language: str | None = "fr",
    device: str = "cpu",
    compute_type: str = "int8",
    batch_size: int = 4,
    reference_text: str | None = None,
) -> TranscriptionBenchmarkResult:
    """Transcribe one file with WhisperX and collect comparable metrics."""
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    import whisperx

    audio = whisperx.load_audio(str(path))
    duration_seconds = len(audio) / 16_000
    model = whisperx.load_model(
        model_name,
        device=device,
        compute_type=compute_type,
        language=language,
    )
    started = time.perf_counter()
    result = model.transcribe(
        audio,
        batch_size=batch_size,
        language=language,
    )
    transcription_seconds = time.perf_counter() - started
    segments = result.get("segments", [])
    text = " ".join(
        str(segment.get("text", "")).strip()
        for segment in segments
        if str(segment.get("text", "")).strip()
    )
    words = text.split()

    return TranscriptionBenchmarkResult(
        audio=str(path),
        model=model_name,
        language=result.get("language") or language,
        device=device,
        compute_type=compute_type,
        duration_seconds=round(duration_seconds, 3),
        transcription_seconds=round(transcription_seconds, 3),
        real_time_factor=round(
            transcription_seconds / duration_seconds, 3
        ) if duration_seconds else 0.0,
        segment_count=len(segments),
        word_count=len(words),
        character_count=len(text),
        repeated_phrase_count=_count_repeated_phrases(text),
        wer=error_rate(reference_text, text) if reference_text else None,
        cer=error_rate(reference_text, text, character=True)
        if reference_text
        else None,
        text=text,
    )


def result_to_markdown(results: list[TranscriptionBenchmarkResult]) -> str:
    """Render a compact human-readable comparison report."""
    lines = [
        "# Transcription benchmark",
        "",
        "| Model | Duration (s) | Transcription (s) | RTF | WER | CER | Segments | Words | Repeated phrases |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.model} | {result.duration_seconds:.1f} | "
            f"{result.transcription_seconds:.1f} | {result.real_time_factor:.2f} | "
            f"{getattr(result, 'wer', None) if getattr(result, 'wer', None) is not None else 'n/a'} | "
            f"{getattr(result, 'cer', None) if getattr(result, 'cer', None) is not None else 'n/a'} | "
            f"{result.segment_count} | {result.word_count} | "
            f"{result.repeated_phrase_count} |"
        )
    return "\n".join(lines) + "\n"
