"""Audio transcription, diarization, and chunk-ready document formatting."""

import logging
import json
import re
from pathlib import Path
from typing import Any, Dict, List

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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "speaker": self.speaker,
            "text": self.text,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


class AudioTranscriptParser:
    """Converts raw diarized call data into structured Markdown."""

    SPEAKER_MAP = {0: "Agent", 1: "Customer"}

    # ── Deepgram JSON ────────────────────────────────────────────────
    @staticmethod
    def parse_deepgram_json(json_data: Dict[str, Any]) -> List[CallTurn]:
        """
        Parses Deepgram's diarization output:
        results.channels[0].alternatives[0].paragraphs.paragraphs
        """
        turns: List[CallTurn] = []
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
            print(f"[AudioParser] Error parsing Deepgram JSON: {exc}")
        return turns

    # ── Plain text transcript ────────────────────────────────────────
    @staticmethod
    def parse_text_transcript(raw_text: str) -> List[CallTurn]:
        """
        Parses simple text transcripts formatted as:
            Agent: Hello, how can I help?
            Customer: I need pricing info.
        """
        turns: List[CallTurn] = []
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
                    turns.append(CallTurn(speaker=role, text=line[len(prefix):].strip()))
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
        turns: List[CallTurn], include_timestamps: bool = True
    ) -> str:
        """Formats call turns into clean Markdown dialogue blocks."""
        lines: List[str] = []
        for turn in turns:
            ts = ""
            if include_timestamps and turn.start_time is not None:
                mins, secs = divmod(int(turn.start_time), 60)
                ts = f" `[{mins:02d}:{secs:02d}]`"
            lines.append(f"**{turn.speaker}**{ts}: {turn.text}")
        return "\n\n".join(lines)


class WhisperXTranscriber:
    """Transcribes audio, assigns speakers, and returns chunk-ready documents."""

    def __init__(
        self,
        model_name: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = None,
        batch_size: int = 4,
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.batch_size = batch_size

    @staticmethod
    def _serialize_turn(turn: CallTurn) -> Dict[str, Any]:
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
    ) -> List[CallTurn]:
        """Run WhisperX transcription and pyannote speaker diarization."""
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")
        if not hf_token.strip():
            raise ValueError("A Hugging Face token is required for diarization.")

        import whisperx
        from whisperx.diarize import DiarizationPipeline

        audio = whisperx.load_audio(str(path))
        model = whisperx.load_model(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
            language=self.language,
        )
        result = model.transcribe(
            audio,
            batch_size=self.batch_size,
            language=self.language,
        )

        language = result.get("language") or self.language
        if not language:
            raise RuntimeError("WhisperX did not detect an audio language.")

        align_model, align_metadata = whisperx.load_align_model(
            language_code=language,
            device=self.device,
        )
        result = whisperx.align(
            result["segments"],
            align_model,
            align_metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )

        diarize_model = DiarizationPipeline(
            token=hf_token,
            device=self.device,
        )
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)

        return self._segments_to_turns(result.get("segments", []))

    @staticmethod
    def _segments_to_turns(segments: List[Dict[str, Any]]) -> List[CallTurn]:
        """Convert WhisperX segments into merged consecutive speaker turns."""
        turns: List[CallTurn] = []
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
        allowed_roles: List[str] | None = None,
        output_dir: str | Path | None = None,
    ) -> Document:
        """Return a document and optionally persist transcript artifacts."""
        path = Path(audio_path)
        turns = self.transcribe_and_diarize(path, hf_token)
        if not turns:
            raise RuntimeError(f"No speech segments were detected in {path}.")

        markdown = AudioTranscriptParser.turns_to_markdown(turns)
        metadata: Dict[str, Any] = {
            "source": path.name,
            "file_type": "audio",
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
