"""Audio transcription, diarization, and chunk-ready document formatting."""

import gc
import json
import logging
import re
from pathlib import Path
from typing import Any, ClassVar

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class CallTurn:
    """A single speaker turn in a commercial call."""

    def __init__(
        self,
        speaker: str,
        text: str,
        start_time: float | None = None,
        end_time: float | None = None,
    ):
        self.speaker = speaker
        self.text = text
        self.start_time = start_time
        self.end_time = end_time

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker": self.speaker,
            "text": self.text,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


class AudioTranscriptParser:
    """Converts raw diarized call data into structured Markdown."""

    SPEAKER_MAP: ClassVar[dict[int, str]] = {0: "Agent", 1: "Customer"}

    # ── Deepgram JSON ────────────────────────────────────────────────
    @staticmethod
    def parse_deepgram_json(json_data: dict[str, Any]) -> list[CallTurn]:
        """
        Parses Deepgram's diarization output:
        results.channels[0].alternatives[0].paragraphs.paragraphs
        """
        turns: list[CallTurn] = []
        try:
            paragraphs = (
                json_data.get("results", {})
                .get("channels", [{}])[0]
                .get("alternatives", [{}])[0]
                .get("paragraphs", {})
                .get("paragraphs", [])
            )
            for p in paragraphs:
                speaker_id = p.get("speaker", 0)
                speaker = AudioTranscriptParser.SPEAKER_MAP.get(
                    speaker_id, f"Speaker {speaker_id}"
                )
                sentences = [s.get("text", "") for s in p.get("sentences", [])]
                text = " ".join(sentences).strip()
                if text:
                    turns.append(
                        CallTurn(
                            speaker=speaker,
                            text=text,
                            start_time=p.get("start"),
                            end_time=p.get("end"),
                        )
                    )
        except (IndexError, KeyError) as exc:
            logger.warning("Error parsing Deepgram JSON: %s", exc)
        return turns

    # ── Plain text transcript ────────────────────────────────────────
    @staticmethod
    def parse_text_transcript(raw_text: str) -> list[CallTurn]:
        """
        Parses simple text transcripts formatted as:
            Agent: Hello, how can I help?
            Customer: I need pricing info.
        """
        turns: list[CallTurn] = []
        prefixes = {
            "agent:": "Agent",
            "rep:": "Agent",
            "sales:": "Agent",
            "customer:": "Customer",
            "client:": "Customer",
        }

        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue

            matched = False
            for prefix, role in prefixes.items():
                if line.lower().startswith(prefix):
                    turns.append(
                        CallTurn(speaker=role, text=line[len(prefix) :].strip())
                    )
                    matched = True
                    break

            if not matched:
                # Continuation of previous turn
                if turns:
                    turns[-1].text += f" {line}"
                else:
                    turns.append(CallTurn(speaker="Unknown", text=line))

        return turns

    # ── Markdown formatter ───────────────────────────────────────────
    @staticmethod
    def turns_to_markdown(
        turns: list[CallTurn], include_timestamps: bool = True
    ) -> str:
        """Formats call turns into clean Markdown dialogue blocks."""
        lines: list[str] = []
        for turn in turns:
            ts = ""
            if include_timestamps and turn.start_time is not None:
                mins, secs = divmod(int(turn.start_time), 60)
                ts = f" `[{mins:02d}:{secs:02d}]`"
            lines.append(f"**{turn.speaker}**{ts}: {turn.text}")
        return "\n\n".join(lines)


class AudioDurationLimitError(ValueError):
    """Decoded audio exceeds the configured limit before model loading."""


class WhisperXTranscriber:
    """Transcribes audio, assigns speakers, and returns chunk-ready documents."""

    def __init__(
        self,
        model_name: str = "medium",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = None,
        batch_size: int = 1,
        max_audio_duration_minutes: float | None = None,
        worker_timeout_seconds: float = 3600,
        worker_cpu_threads: int = 2,
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.batch_size = batch_size
        self.max_audio_duration_minutes = max_audio_duration_minutes
        self.worker_timeout_seconds = worker_timeout_seconds
        self.worker_cpu_threads = worker_cpu_threads

    @staticmethod
    def _serialize_turn(turn: CallTurn) -> dict[str, Any]:
        """Serialize CallTurn objects and compatible diarization test doubles."""
        to_dict = getattr(turn, "to_dict", None)
        if callable(to_dict):
            return to_dict()
        return {
            "speaker": turn.speaker,
            "text": turn.text,
            "start_time": turn.start_time,
            "end_time": turn.end_time,
        }

    def transcribe_and_diarize(
        self,
        audio_path: str | Path,
        hf_token: str,
    ) -> list[CallTurn]:
        """Run audio models in a separate process and wait for its termination."""
        from deal_voice.ingestion.audio_worker import run_audio_worker

        if not hf_token.strip():
            raise ValueError("A Hugging Face token is required for diarization.")
        turns = run_audio_worker(
            Path(audio_path),
            hf_token=hf_token,
            model_name=self.model_name,
            device=self.device,
            compute_type=self.compute_type,
            language=self.language,
            batch_size=self.batch_size,
            max_duration_seconds=self.max_audio_duration_minutes * 60
            if self.max_audio_duration_minutes is not None
            else None,
            timeout_seconds=self.worker_timeout_seconds,
            cpu_threads=self.worker_cpu_threads,
        )
        return [CallTurn(**turn) for turn in turns]

    def _transcribe_and_diarize_in_process(
        self,
        audio_path: str | Path,
        hf_token: str,
        *,
        max_duration_seconds: float | None = None,
    ) -> list[CallTurn]:
        """Worker implementation; application callers use the isolated method."""
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")
        if not hf_token.strip():
            raise ValueError("A Hugging Face token is required for diarization.")

        import whisperx

        audio = whisperx.load_audio(str(path))
        if (
            max_duration_seconds is not None
            and len(audio) / 16_000 > max_duration_seconds
        ):
            raise AudioDurationLimitError(
                "Audio exceeds its configured duration limit."
            )
        model = whisperx.load_model(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
            language=self.language,
            threads=self.worker_cpu_threads,
        )
        try:
            result = model.transcribe(
                audio, batch_size=self.batch_size, language=self.language
            )
        finally:
            del model
            self._release_model_memory()

        language = result.get("language") or self.language
        if not language:
            raise RuntimeError("WhisperX did not detect an audio language.")

        align_model, align_metadata = whisperx.load_align_model(
            language_code=language,
            device=self.device,
        )
        try:
            result = whisperx.align(
                result["segments"],
                align_model,
                align_metadata,
                audio,
                self.device,
                return_char_alignments=False,
            )
        finally:
            del align_model
            del align_metadata
            self._release_model_memory()

        from whisperx.diarize import DiarizationPipeline

        diarize_model = DiarizationPipeline(
            token=hf_token,
            device=self.device,
        )
        try:
            diarize_segments = diarize_model(audio)
        finally:
            del diarize_model
            self._release_model_memory()
        result = whisperx.assign_word_speakers(diarize_segments, result)

        return self._segments_to_turns(result.get("segments", []))

    def _release_model_memory(self) -> None:
        """Collect stage-local model cycles before loading the next model."""
        gc.collect()
        if self.device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()

    @staticmethod
    def _segments_to_turns(segments: list[dict[str, Any]]) -> list[CallTurn]:
        """Convert WhisperX segments into merged consecutive speaker turns."""
        turns: list[CallTurn] = []
        for segment in segments:
            text = str(segment.get("text", "")).strip()
            speaker = str(segment.get("speaker", "Unknown")).strip()
            text = WhisperXTranscriber._collapse_repetitions(text)
            if not text:
                continue

            start = segment.get("start")
            end = segment.get("end")
            if turns and turns[-1].speaker == speaker:
                turns[-1].text = f"{turns[-1].text} {text}".strip()
                turns[-1].end_time = end
                continue

            turns.append(
                CallTurn(
                    speaker=speaker,
                    text=text,
                    start_time=start,
                    end_time=end,
                )
            )
        return turns

    @staticmethod
    def _collapse_repetitions(text: str) -> str:
        """Remove obvious Whisper loops without rewriting normal speech."""
        previous = None
        while text != previous:
            previous = text
            text = re.sub(
                r"(?P<sentence>[^.!?]+[.!?])(?:\s+(?P=sentence)){2,}",
                r"\g<sentence>",
                text,
                flags=re.IGNORECASE,
            )
            text = re.sub(
                r"\b((?:[\wÀ-ÿ']+\s+){1,5}[\wÀ-ÿ']+),?"
                r"(?:\s+\1,?){2,}[.!?]?",
                r"\1,",
                text,
                flags=re.IGNORECASE,
            )
            text = re.sub(
                r"\b((?:[\wÀ-ÿ']+\s+){0,7}[\wÀ-ÿ']+[,.!?]?)(?:\s+\1){2,}",
                r"\1",
                text,
                flags=re.IGNORECASE,
            )
        return re.sub(r"\s+", " ", text).strip()

    def transcribe_to_document(
        self,
        audio_path: str | Path,
        hf_token: str,
        allowed_roles: list[str] | None = None,
        output_dir: str | Path | None = None,
    ) -> Document:
        """Return a document and optionally persist transcript artifacts."""
        path = Path(audio_path)
        turns = self.transcribe_and_diarize(path, hf_token)
        if not turns:
            raise RuntimeError(f"No speech segments were detected in {path}.")

        markdown = AudioTranscriptParser.turns_to_markdown(turns)
        metadata: dict[str, Any] = {
            "source": path.name,
            "file_type": "audio",
            "transcription_model": self.model_name,
            "transcription_language": self.language,
            "transcription_device": self.device,
            "transcription_compute_type": self.compute_type,
            "speaker_count": len({turn.speaker for turn in turns}),
            "speakers": sorted({turn.speaker for turn in turns}),
            "start_time": min(
                (turn.start_time for turn in turns if turn.start_time is not None),
                default=None,
            ),
            "end_time": max(
                (turn.end_time for turn in turns if turn.end_time is not None),
                default=None,
            ),
        }
        if allowed_roles:
            metadata["allowed_roles"] = allowed_roles

        if output_dir is not None:
            artifact_dir = Path(output_dir)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            artifact_stem = path.stem
            markdown_path = artifact_dir / f"{artifact_stem}_transcript.md"
            json_path = artifact_dir / f"{artifact_stem}_transcript.json"
            markdown_path.write_text(markdown, encoding="utf-8")
            json_path.write_text(
                json.dumps(
                    {
                        "source": path.name,
                        "file_type": "audio",
                        "transcription_model": metadata["transcription_model"],
                        "transcription_language": metadata["transcription_language"],
                        "transcription_device": metadata["transcription_device"],
                        "transcription_compute_type": metadata[
                            "transcription_compute_type"
                        ],
                        "speaker_count": metadata["speaker_count"],
                        "speakers": metadata["speakers"],
                        "start_time": metadata["start_time"],
                        "end_time": metadata["end_time"],
                        "turns": [self._serialize_turn(turn) for turn in turns],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            metadata["transcript_markdown"] = str(markdown_path)
            metadata["transcript_json"] = str(json_path)

        metadata["turns"] = [self._serialize_turn(turn) for turn in turns]
        return Document(
            page_content=markdown,
            metadata=metadata,
        )
