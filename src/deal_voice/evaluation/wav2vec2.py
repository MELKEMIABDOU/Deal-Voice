"""French Wav2Vec2 transcription benchmark."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .transcription import (
    TranscriptionBenchmarkResult,
    _count_repeated_phrases,
    error_rate,
)


def benchmark_wav2vec2(
    audio_path: str | Path,
    *,
    model_name: str = "jonatasgrosman/wav2vec2-large-xlsr-53-french",
    device: str = "cpu",
    reference_text: str | None = None,
) -> TranscriptionBenchmarkResult:
    """Transcribe a French audio file with a Wav2Vec2 CTC model."""
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    import torch
    import whisperx
    from transformers import (
        AutoFeatureExtractor,
        AutoModelForCTC,
        AutoTokenizer,
        Wav2Vec2Processor,
    )

    audio = whisperx.load_audio(str(path))
    duration_seconds = len(audio) / 16_000
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )
    model = AutoModelForCTC.from_pretrained(model_name).to(device)
    model.eval()

    started = time.perf_counter()
    inputs = processor(
        audio,
        sampling_rate=16_000,
        return_tensors="pt",
        padding=True,
    )
    input_values = inputs.input_values.to(device)
    attention_mask = getattr(inputs, "attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.inference_mode():
        logits = model(
            input_values=input_values,
            attention_mask=attention_mask,
        ).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    text = processor.batch_decode(predicted_ids)[0].strip()
    transcription_seconds = time.perf_counter() - started
    words = text.split()

    return TranscriptionBenchmarkResult(
        audio=str(path),
        model=model_name,
        language="fr",
        device=device,
        compute_type="torch",
        duration_seconds=round(duration_seconds, 3),
        transcription_seconds=round(transcription_seconds, 3),
        real_time_factor=round(
            transcription_seconds / duration_seconds, 3
        ) if duration_seconds else 0.0,
        segment_count=1 if text else 0,
        word_count=len(words),
        character_count=len(text),
        repeated_phrase_count=_count_repeated_phrases(text),
        wer=error_rate(reference_text, text) if reference_text else None,
        cer=error_rate(reference_text, text, character=True)
        if reference_text
        else None,
        text=text,
    )
