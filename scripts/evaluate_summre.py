"""Evaluate French ASR on manually transcribed SUMM-RE dev/test tracks.

The public SUMM-RE Hugging Face extract contains manually corrected ASR
references for dev/test, but no meeting-summary labels. This script therefore
reports transcription WER/CER and runtime metrics, not RAG summary quality.
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from deal_voice.evaluation.transcription import (
    TranscriptionBenchmarkResult,
    _normalize_text,
    benchmark_model,
)


DATASET_ID = "linagora/SUMM-RE"
DATASET_CARD = "https://huggingface.co/datasets/linagora/SUMM-RE"
DATASET_CITATION = (
    "Hunter et al. (2024), SUMM-RE: A corpus of French meeting-style conversations, "
    "TALN 2024. https://aclanthology.org/2024.jeptalnrecital-taln.35/"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark French WhisperX ASR against SUMM-RE manual references."
    )
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of distinct meetings to sample (one speaker track each; default: 3).",
    )
    parser.add_argument("--models", nargs="+", default=["small"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--max-duration-seconds",
        type=float,
        default=None,
        help="Evaluate only the beginning of each track up to this duration.",
    )
    parser.add_argument(
        "--include-transcripts",
        action="store_true",
        help="Write reference and hypothesis text to transcripts.md for manual review.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/transcription/summ-re"),
    )
    return parser.parse_args()


def write_audio_sample(sample_audio: dict[str, Any], destination: Path) -> Path:
    """Materialize a Hugging Face Audio value as a temporary WAV file."""
    source_path = sample_audio.get("path")
    if source_path and Path(source_path).is_file():
        return Path(source_path)

    raw_bytes = sample_audio.get("bytes")
    if raw_bytes:
        extension = Path(source_path or "audio.wav").suffix or ".wav"
        destination = destination.with_suffix(extension)
        destination.write_bytes(raw_bytes)
        return destination

    samples = sample_audio.get("array")
    sample_rate = int(sample_audio.get("sampling_rate") or 0)
    if samples is None or sample_rate <= 0:
        raise ValueError("SUMM-RE row has no decodable audio waveform or sample rate.")

    waveform = np.asarray(samples)
    if waveform.ndim == 1:
        waveform = waveform[:, None]
    elif waveform.ndim == 2 and waveform.shape[0] <= 8 < waveform.shape[1]:
        waveform = waveform.T
    if waveform.ndim != 2:
        raise ValueError(f"Unexpected SUMM-RE waveform shape: {waveform.shape}")
    if np.issubdtype(waveform.dtype, np.floating):
        waveform = np.clip(waveform, -1.0, 1.0) * 32767.0
    pcm = np.asarray(waveform, dtype="<i2")

    destination = destination.with_suffix(".wav")
    with wave.open(str(destination), "wb") as output:
        output.setnchannels(pcm.shape[1])
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())
    return destination


def result_row(
    result: TranscriptionBenchmarkResult,
    sample: dict[str, Any],
    reference_text: str | None = None,
    *,
    include_transcripts: bool = False,
) -> dict[str, Any]:
    row = result.to_dict()
    # Keep transcript text out of structured reports unless explicitly requested.
    row.pop("text", None)
    reference_text = reference_text or str(sample.get("transcript", ""))
    row["audio"] = str(sample.get("audio_id") or sample.get("meeting_id", "track"))
    row["meeting_id"] = sample.get("meeting_id")
    row["speaker_id"] = sample.get("speaker_id")
    # Use the same reference excerpt as WER/CER when the audio was truncated.
    row["reference_word_count"] = len(_normalize_text(reference_text).split())
    row["reference_character_count"] = len(
        _normalize_text(reference_text).replace(" ", "")
    )
    if include_transcripts:
        row["reference_text"] = reference_text
        row["hypothesis_text"] = result.text
    return row


def truncate_wav(path: Path, max_duration_seconds: float) -> float:
    """Truncate a WAV file in place and return its resulting duration."""
    temporary_path = path.with_suffix(".trimmed.wav")
    with wave.open(str(path), "rb") as source:
        params = source.getparams()
        frame_limit = int(max_duration_seconds * params.framerate)
        frames = source.readframes(frame_limit)
        frame_count = len(frames) // (params.sampwidth * params.nchannels)
    with wave.open(str(temporary_path), "wb") as destination:
        destination.setparams(params)
        destination.writeframes(frames)
    temporary_path.replace(path)
    return frame_count / params.framerate


def weighted_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for model in sorted({str(row["model"]) for row in rows}):
        group = [row for row in rows if row["model"] == model]
        words = sum(int(row["reference_word_count"]) for row in group)
        characters = sum(int(row["reference_character_count"]) for row in group)
        wer = sum(float(row["wer"] or 0) * row["reference_word_count"] for row in group) / words
        cer = sum(float(row["cer"] or 0) * row["reference_character_count"] for row in group) / characters
        seconds = sum(float(row["duration_seconds"]) for row in group)
        inference_seconds = sum(float(row["transcription_seconds"]) for row in group)
        summaries.append(
            {
                "model": model,
                "track_count": len(group),
                "reference_hours": round(seconds / 3600, 3),
                "transcription_seconds": round(inference_seconds, 2),
                "real_time_factor": round(inference_seconds / seconds, 3) if seconds else 0,
                "weighted_wer": round(wer, 4),
                "weighted_cer": round(cer, 4),
            }
        )
    return summaries


def write_reports(output_dir: Path, metadata: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = weighted_summary(rows)
    report = {**metadata, "tracks": rows, "summary": summaries}
    (output_dir / "benchmark.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "benchmark.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# SUMM-RE transcription evaluation",
        "",
        f"Split: `{metadata['split']}`; tracks: {metadata['evaluated_tracks']}",
        "",
        "| Model | Tracks | Audio hours | Time (s) | RTF | Weighted WER | Weighted CER |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary['model']} | {summary['track_count']} | "
            f"{summary['reference_hours']:.3f} | {summary['transcription_seconds']:.1f} | "
            f"{summary['real_time_factor']:.3f} | {summary['weighted_wer']:.2%} | "
            f"{summary['weighted_cer']:.2%} |"
        )
    lines.extend(["", f"Dataset: {DATASET_CARD}", f"Citation: {DATASET_CITATION}", ""])
    (output_dir / "benchmark.md").write_text("\n".join(lines), encoding="utf-8")
    if metadata.get("include_transcripts"):
        transcript_lines = ["# Références et transcriptions WhisperX", ""]
        for row in rows:
            transcript_lines.extend(
                [
                    f"## {row['audio']} — {row['model']}",
                    "",
                    f"WER : {row['wer']:.2%} · CER : {row['cer']:.2%}",
                    "",
                    "### Référence SUMM-RE",
                    "",
                    row["reference_text"],
                    "",
                    "### Transcription du modèle",
                    "",
                    row["hypothesis_text"],
                    "",
                ]
            )
        (output_dir / "transcripts.md").write_text(
            "\n".join(transcript_lines), encoding="utf-8"
        )


def main() -> None:
    args = parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be at least 1; use a small sample to control downloads.")
    if args.max_duration_seconds is not None and args.max_duration_seconds <= 0:
        raise SystemExit("--max-duration-seconds must be positive.")

    from datasets import Audio, load_dataset

    print(
        f"Streaming {DATASET_ID} split={args.split} language=fr; evaluating {args.limit} distinct "
        "meetings (one manual-reference speaker track per meeting)."
    )
    dataset = load_dataset(DATASET_ID, split=args.split, streaming=True)
    # Keep audio encoded in the Parquet stream; WhisperX/FFmpeg reads the
    # temporary file and this avoids adding librosa/soundfile dependencies.
    dataset = dataset.cast_column("audio", Audio(decode=False))
    rows: list[dict[str, Any]] = []
    sampled_meetings: set[str] = set()
    with tempfile.TemporaryDirectory(prefix="summre_eval_") as scratch:
        scratch_dir = Path(scratch)
        for sample in dataset:
            reference = str(sample.get("transcript", "")).strip()
            if not reference:
                continue
            meeting_id = str(sample.get("meeting_id") or "unknown")
            if meeting_id in sampled_meetings:
                continue
            sampled_meetings.add(meeting_id)
            audio_id = str(sample.get("audio_id") or f"track-{len(sampled_meetings):04d}")
            audio_path = write_audio_sample(
                sample["audio"], scratch_dir / f"{audio_id}.wav"
            )
            reference = str(sample.get("transcript", "")).strip()
            if args.max_duration_seconds is not None:
                if not sample.get("segments"):
                    raise ValueError(
                        f"SUMM-RE track {audio_id} has no segments for duration-aligned reference text."
                    )
                audio_duration = truncate_wav(audio_path, args.max_duration_seconds)
                reference = " ".join(
                    str(segment.get("transcript", "")).strip()
                    for segment in sample["segments"]
                    if float(segment.get("end", 0)) <= audio_duration
                    and str(segment.get("transcript", "")).strip()
                )
                if not reference:
                    raise ValueError(f"No reference segments fit in the audio excerpt {audio_id}.")
            for model_name in args.models:
                print(f"{model_name}: {audio_id}")
                result = benchmark_model(
                    audio_path,
                    model_name,
                    language="fr",
                    device=args.device,
                    compute_type=args.compute_type,
                    batch_size=args.batch_size,
                    reference_text=reference,
                )
                rows.append(
                    result_row(
                        result,
                        sample,
                        reference,
                        include_transcripts=args.include_transcripts,
                    )
                )
            if len(sampled_meetings) >= args.limit:
                break

    if not rows:
        raise RuntimeError("No SUMM-RE rows with reference transcripts were evaluated.")
    metadata = {
        "dataset": DATASET_ID,
        "dataset_url": DATASET_CARD,
        "license": "CC-BY-SA-4.0",
        "citation": DATASET_CITATION,
        "split": args.split,
        "language": "fr",
        "max_duration_seconds": args.max_duration_seconds,
        "include_transcripts": args.include_transcripts,
        "evaluated_tracks": len(rows) // len(args.models),
        "sampled_meetings": len(sampled_meetings),
        "models": args.models,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sampling": "first non-empty row per unique meeting from Hugging Face streaming split",
        "scope_note": (
            "This public extract supports manual-reference ASR evaluation. It does not "
            "contain meeting-summary references for evaluating RAG summarization."
        ),
    }
    write_reports(args.output_dir, metadata, rows)
    print(f"Reports written to {args.output_dir}")


if __name__ == "__main__":
    main()
