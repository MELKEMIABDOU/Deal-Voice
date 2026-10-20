r"""Benchmark WhisperX models on audio/reference pairs in one directory.

Expected files:
    data\evaluation\call.mp3
    data\evaluation\call.txt

Example:
    uv run python scripts\evaluate_transcription_dataset.py `
      --evaluation-dir data\evaluation `
      --models base small medium large-v3
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from deal_voice.evaluation.transcription import (
    TranscriptionBenchmarkResult,
    _normalize_text,
    benchmark_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark WhisperX models on an evaluation dataset."
    )
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["base", "small", "medium", "large-v3"],
    )
    parser.add_argument("--language", default="fr")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/transcription/evaluation-dataset"),
    )
    return parser.parse_args()


def find_pairs(evaluation_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for audio_path in sorted(evaluation_dir.iterdir()):
        if audio_path.suffix.lower() not in {".mp3", ".wav", ".m4a", ".flac"}:
            continue
        reference_path = audio_path.with_suffix(".txt")
        if not reference_path.is_file():
            raise FileNotFoundError(
                f"Missing reference transcript for {audio_path.name}: "
                f"{reference_path.name}"
            )
        pairs.append((audio_path, reference_path))
    if not pairs:
        raise FileNotFoundError(f"No audio/reference pairs found in {evaluation_dir}")
    return pairs


def weighted_summary(
    results: list[TranscriptionBenchmarkResult],
) -> list[dict[str, object]]:
    grouped: dict[str, list[TranscriptionBenchmarkResult]] = defaultdict(list)
    for result in results:
        grouped[result.model].append(result)

    summaries = []
    for model, model_results in grouped.items():
        reference_words = sum(
            result.reference_word_count or 0
            for result in model_results
        )
        weighted_wer = (
            sum(
                (result.wer or 0.0) * (result.reference_word_count or 0)
                for result in model_results
            )
            / reference_words
            if reference_words
            else None
        )
        reference_characters = sum(
            result.reference_character_count or 0 for result in model_results
        )
        weighted_cer = (
            sum(
                (result.cer or 0.0) * (result.reference_character_count or 0)
                for result in model_results
            )
            / reference_characters
            if reference_characters
            else None
        )
        summaries.append(
            {
                "model": model,
                "file_count": len(model_results),
                "transcription_seconds": round(
                    sum(result.transcription_seconds for result in model_results), 3
                ),
                "weighted_wer": round(weighted_wer, 4)
                if weighted_wer is not None
                else None,
                "weighted_cer": round(weighted_cer, 4)
                if weighted_cer is not None
                else None,
            }
        )
    return summaries


def main() -> None:
    args = parse_args()
    pairs = find_pairs(args.evaluation_dir)
    results: list[TranscriptionBenchmarkResult] = []

    for audio_path, reference_path in pairs:
        reference_text = reference_path.read_text(encoding="utf-8")
        for model_name in args.models:
            print(f"Benchmarking {model_name} on {audio_path.name}...")
            result = benchmark_model(
                audio_path,
                model_name,
                language=args.language,
                device=args.device,
                compute_type=args.compute_type,
                batch_size=args.batch_size,
                reference_text=reference_text,
            )
            normalized_reference = _normalize_text(reference_text)
            result.reference_word_count = len(normalized_reference.split())
            result.reference_character_count = len(
                normalized_reference.replace(" ", "")
            )
            results.append(result)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = weighted_summary(results)
    (args.output_dir / "benchmark.json").write_text(
        json.dumps(
            {"files": [result.to_dict() for result in results], "summary": summaries},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with (args.output_dir / "benchmark.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].to_dict().keys()))
        writer.writeheader()
        writer.writerows(result.to_dict() for result in results)
    lines = [
        "# Transcription evaluation dataset",
        "",
        "| Model | Files | Total time (s) | Weighted WER | Weighted CER |",
        "|---|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary['model']} | {summary['file_count']} | "
            f"{summary['transcription_seconds']:.1f} | "
            f"{summary['weighted_wer']:.2%} | {summary['weighted_cer']:.2%} |"
        )
    (args.output_dir / "benchmark.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(f"Reports written to {args.output_dir}")


if __name__ == "__main__":
    main()
