r"""Compare WhisperX models on one audio file.

Example:
    uv run python scripts\evaluate_transcription.py --audio data\raw\call.mp3
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from deal_voice.evaluation.transcription import (
    TranscriptionBenchmarkResult,
    benchmark_model,
    result_to_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark WhisperX transcription models."
    )
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["base", "small"],
        help="WhisperX models to compare.",
    )
    parser.add_argument("--language", default="fr")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--reference",
        type=Path,
        help="Human reference transcript used for WER/CER.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/transcription"))
    return parser.parse_args()


def write_reports(
    results: list[TranscriptionBenchmarkResult], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark.json").write_text(
        json.dumps([result.to_dict() for result in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "benchmark.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].to_dict().keys()))
        writer.writeheader()
        writer.writerows(result.to_dict() for result in results)
    (output_dir / "benchmark.md").write_text(
        result_to_markdown(results),
        encoding="utf-8",
    )
    for result in results:
        (output_dir / f"{Path(result.audio).stem}.{result.model}.txt").write_text(
            result.text,
            encoding="utf-8",
        )


def main() -> None:
    args = parse_args()
    reference_text = (
        args.reference.read_text(encoding="utf-8")
        if args.reference
        else None
    )
    results = []
    for model_name in args.models:
        print(f"Benchmarking {model_name}...")
        results.append(
            benchmark_model(
                args.audio,
                model_name,
                language=args.language,
                device=args.device,
                compute_type=args.compute_type,
                batch_size=args.batch_size,
                reference_text=reference_text,
            )
        )
    write_reports(results, args.output_dir)
    print(f"Reports written to {args.output_dir}")


if __name__ == "__main__":
    main()
