r"""Benchmark a French Wav2Vec2 model on an audio file.

Example:
    uv run python scripts\evaluate_wav2vec2_french.py --audio data\raw\call.mp3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from deal_voice.evaluation.wav2vec2 import benchmark_wav2vec2


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark a French Wav2Vec2 CTC model."
    )
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument(
        "--model",
        default="jonatasgrosman/wav2vec2-large-xlsr-53-french",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--reference",
        type=Path,
        help="Human reference transcript used for WER/CER.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/transcription/wav2vec2-french"),
    )
    args = parser.parse_args()
    reference_text = (
        args.reference.read_text(encoding="utf-8")
        if args.reference
        else None
    )

    print(f"Benchmarking {args.model}...")
    result = benchmark_wav2vec2(
        args.audio,
        model_name=args.model,
        device=args.device,
        reference_text=reference_text,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.audio.stem
    (args.output_dir / "benchmark.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / f"{stem}.txt").write_text(
        result.text,
        encoding="utf-8",
    )
    print(f"Reports written to {args.output_dir}")


if __name__ == "__main__":
    main()
