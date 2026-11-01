"""Short-lived audio process. Only validated transcript turns cross the boundary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from deal_voice.ingestion.worker import run_worker, watch_parent


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    audio_path: str
    model_name: str = "medium"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str | None = None
    batch_size: int = Field(default=1, ge=1)
    hf_token: str = Field(repr=False, min_length=1)
    max_duration_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    cpu_threads: int = Field(default=2, ge=1)


class _Turn(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    speaker: str = Field(min_length=1)
    text: str = Field(min_length=1)
    start_time: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    end_time: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def ordered_timestamps(self) -> _Turn:
        if (
            self.start_time is not None
            and self.end_time is not None
            and self.end_time < self.start_time
        ):
            raise ValueError("The end of a turn precedes its start.")
        return self


class _Result(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    turns: list[_Turn]


def run_audio_worker(
    audio_path: Path,
    *,
    hf_token: str,
    model_name: str = "medium",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str | None = None,
    batch_size: int = 1,
    max_duration_seconds: float | None = None,
    timeout_seconds: float = 3600,
    cpu_threads: int = 2,
) -> list[dict]:
    """Validate the audio protocol, then delegate to the shared process transport."""
    path = Path(audio_path).resolve(strict=True)
    request = _Request(
        audio_path=str(path),
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        language=language,
        batch_size=batch_size,
        hf_token=hf_token,
        max_duration_seconds=max_duration_seconds,
        cpu_threads=cpu_threads,
    )
    code, result = run_worker(
        "deal_voice.ingestion.audio_worker",
        request.model_dump_json(),
        timeout_seconds=timeout_seconds,
        cpu_threads=cpu_threads,
        label="Audio",
    )
    if (
        code == 2
        and isinstance(result, dict)
        and result.get("error") == "audio_too_long"
    ):
        raise ValueError(
            "Le fichier dépasse la durée maximale autorisée "
            f"({max_duration_seconds} secondes)."
        )
    if code != 0:
        raise RuntimeError(
            f"Audio worker failed (exit code {code}); no document was produced."
        )
    try:
        return _Result.model_validate(result).model_dump()["turns"]
    except ValidationError:
        raise RuntimeError("Audio worker returned an invalid transcript.") from None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-json", type=Path, required=True)
    args = parser.parse_args()
    try:
        request = _Request.model_validate_json(sys.stdin.read())
        if not Path(request.audio_path).is_file() or not request.hf_token.strip():
            raise ValueError("Invalid audio request.")
    except (ValueError, OSError):
        args.result_json.write_text('{"error":"invalid_request"}', encoding="utf-8")
        return 2

    from deal_voice.ingestion.audio import AudioDurationLimitError, WhisperXTranscriber

    code = 0
    try:
        watch_parent()
        transcriber = WhisperXTranscriber(
            model_name=request.model_name,
            device=request.device,
            compute_type=request.compute_type,
            language=request.language,
            batch_size=request.batch_size,
            worker_cpu_threads=request.cpu_threads,
        )
        turns = transcriber._transcribe_and_diarize_in_process(
            request.audio_path,
            request.hf_token,
            max_duration_seconds=request.max_duration_seconds,
        )
        result = _Result.model_validate({"turns": [turn.to_dict() for turn in turns]})
        output = result.model_dump_json()
    except AudioDurationLimitError:
        output, code = '{"error":"audio_too_long"}', 2
    except Exception:
        # Do not serialize library exception messages, credentials or transcripts.
        output, code = '{"error":"processing_failed"}', 1
    temporary = args.result_json.with_suffix(".tmp")
    temporary.write_text(output, encoding="utf-8")
    temporary.replace(args.result_json)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
