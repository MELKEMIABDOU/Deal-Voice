"""Run an isolated, memory-supervised pilot with real local model calls.

Exercises Streamlit's upload function, then the shared RAG pipeline and ASGI
search endpoint. This is not a browser, network, or concurrent-upload test.
HF downloads are disabled; French alignment weights are checked locally.
Other engines use their installed caches. Test files/database stay in data/raw.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests

from deal_voice.evaluation.resources import MemoryMonitor, memory_snapshot
from deal_voice.ingestion.audio_process import (
    audio_process_job,
    join_audio_process_job,
)

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3


def save_report(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def finalize_interrupted_report(
    path: Path, returncode: int, reason: str | None
) -> None:
    """A killed worker cannot write its final checkpoint; record the supervisor outcome."""
    result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if reason or returncode != 0:
        result["status"] = "failed" if returncode != 2 or reason else "partial"
        result["supervisor_returncode"] = returncode
        if reason:
            result["stopped_reason"] = reason
        save_report(path, result)


def memory_headroom(snapshot: dict, minimum_gib: float) -> bool:
    counters = [
        snapshot.get("system_available_bytes"),
        snapshot.get("commit_available_bytes"),
    ]
    return all(value is not None and value >= minimum_gib * GIB for value in counters)


def generation_budget(users: int, reserve_gib: float, minimum_gib: float) -> float:
    # Reserve covers model loading + first context; allow an extra GiB per
    # concurrent request for KV/CPU activations. This is an admission estimate,
    # not a measured capacity limit or a guarantee against hardware failures.
    return minimum_gib + reserve_gib + max(0, users - 1)


def overall_status(report: dict) -> str:
    if not report.get("single_user") or not all(
        row["passed"] for row in report["single_user"]
    ):
        return "failed"
    loads = report.get("concurrency", [])
    if any(row["status"] == "failed" for row in loads):
        return "failed"
    if report.get("single_user_only"):
        return "passed"
    return (
        "passed"
        if {row["users"] for row in loads if row["status"] == "passed"} == {2, 5}
        else "partial"
    )


def require_french_alignment_cache() -> None:
    # WhisperX fr uses torchaudio's Torch Hub bundle rather than HF Hub.
    # Torch Hub does not honor HF_HUB_OFFLINE; prevent an implicit download.
    from torch.hub import get_dir

    checkpoint = (
        Path(get_dir()) / "checkpoints" / "wav2vec2_voxpopuli_base_10k_asr_fr.pt"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(
            "French alignment checkpoint is not cached; audio test not started."
        )


def validated_workspace(workspace: str, report_dir: str) -> tuple[Path, Path]:
    work = Path(workspace).resolve()
    report = Path(report_dir).resolve()
    if (
        work.parent != (ROOT / "data" / "raw").resolve()
        or not work.name.startswith("pilot-")
        or not work.is_relative_to(ROOT)
    ):
        raise ValueError(
            "Pilot workspace must be a dedicated project data/raw/pilot-* directory."
        )
    if report.parent != (
        ROOT / "reports" / "pilot"
    ).resolve() or not report.is_relative_to(ROOT):
        raise ValueError(
            "Pilot report must be a dedicated project reports/pilot directory."
        )
    return work, report


@dataclass
class Upload:
    name: str
    content: bytes

    @property
    def size(self) -> int:
        return len(self.content)

    def getbuffer(self) -> memoryview:
        return memoryview(self.content)


def create_fixtures(args, workspace: Path) -> list[tuple[str, Path, str, list[str]]]:
    import pymupdf

    fixtures = workspace / "fixtures"
    fixtures.mkdir()
    pdf_path = fixtures / "pilot-report.pdf"
    with pymupdf.open(args.pdf) as source, pymupdf.open() as selected:
        if not 1 <= args.pdf_page <= len(source):
            raise ValueError("The requested one-based PDF page does not exist.")
        selected.insert_pdf(
            source, from_page=args.pdf_page - 1, to_page=args.pdf_page - 1
        )
        selected.save(pdf_path)
    cases = [("pdf", pdf_path, args.pdf_question, args.pdf_expected)]
    if args.audio:
        executable = shutil.which("ffmpeg")
        if not executable:
            raise RuntimeError("FFmpeg is required for the short audio fixture.")
        audio_path = fixtures / "pilot-call.wav"
        subprocess.run(  # noqa: S603 -- resolved executable, argv without shell, isolated output.
            [
                executable,
                "-nostdin",
                "-v",
                "error",
                "-i",
                str(args.audio),
                "-t",
                str(args.audio_seconds),
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(audio_path),
            ],
            check=True,
            timeout=60,
        )
        cases.append(("audio", audio_path, args.audio_question, args.audio_expected))
        if args.audio_first:
            cases.reverse()
    return cases


def worker(args) -> int:
    if os.name == "nt":
        join_audio_process_job(os.environ["DEAL_VOICE_PILOT_JOB"])
    workspace, report_dir = validated_workspace(args.workspace, args.report_dir)
    os.environ["QDRANT_STORAGE_PATH"] = str(workspace / "qdrant")
    os.environ["COLLECTION_NAME"] = "pilot_validation"
    # Import before chdir so the project's .env is loaded, while the isolated
    # database path is already set when constructor defaults are evaluated.
    import streamlit_app
    from deal_voice.config import settings
    from deal_voice.evaluation.generation import evaluate_generation

    os.chdir(workspace)
    report_path = report_dir / "result.json"
    report = {
        "status": "running",
        "phase": "initializing",
        "single_user": [],
        "uploads": [],
        "generation_admissions": [],
        "concurrency": [],
        "single_user_only": args.single_user_only,
        "workspace": str(workspace),
        "scope": "real upload logic + local models + in-process ASGI clients; no browser/network or concurrent uploads",
        "database": settings.qdrant_storage_path,
        "models": {
            "embedding": settings.embedding_model_name,
            "reranker": settings.reranker_model_name,
            "llm": settings.ollama_model_name,
            "transcription": "medium/fr/cpu/int8",
        },
        "admission_policy": {
            "minimum_free_gib": args.min_free_gib,
            "generation_reserve_gib": args.generation_reserve_gib,
        },
    }
    engine = None
    with MemoryMonitor(report_dir / "worker.memory.jsonl") as monitor:

        def phase(name: str) -> None:
            report["phase"] = name
            monitor.mark(name)
            save_report(report_path, report)

        try:
            phase("creating_short_fixtures")
            cases = create_fixtures(args, workspace)
            generator = None
            for kind, path, question, expected in cases:
                if kind == "audio":
                    phase("preflight_audio_alignment_cache")
                    require_french_alignment_cache()
                phase(f"upload_and_index_{kind}")
                started = time.perf_counter()
                identifiers = streamlit_app.index_uploaded_files(
                    [Upload(path.name, path.read_bytes())], "medium"
                )
                upload_seconds = time.perf_counter() - started
                engine = streamlit_app.load_search_engine()
                report["uploads"].append(
                    {
                        "kind": kind,
                        "source": path.name,
                        "document_ids": identifiers,
                        "seconds": round(upload_seconds, 2),
                    }
                )
                phase(f"memory_admission_{kind}")
                needed = generation_budget(
                    1, args.generation_reserve_gib, args.min_free_gib
                )
                snapshot = memory_snapshot()
                report["generation_admissions"].append(
                    {
                        "kind": kind,
                        "minimum_free_gib_required": needed,
                        "memory": snapshot,
                    }
                )
                save_report(report_path, report)
                if not memory_headroom(snapshot, needed):
                    raise MemoryError(
                        f"Generation not started: need {needed:.1f} GiB free including precautionary reserve."
                    )
                phase(f"retrieve_and_generate_{kind}")
                generator = streamlit_app.load_generator(settings.ollama_model_name)
                started = time.perf_counter()
                documents = engine.search(question, document_ids=identifiers, top_k=3)
                answer = generator.generate(question, documents)
                trace = generator.last_trace
                evaluation = evaluate_generation(
                    kind,
                    question,
                    path.name,
                    expected,
                    answer,
                    expected_language="fr",
                    available_citations=trace.get("structured_citations"),
                )
                passed = trace.get("status") == "answer" and evaluation.passed
                report["single_user"].append(
                    {
                        "kind": kind,
                        "passed": passed,
                        "upload_seconds": round(upload_seconds, 2),
                        "query_seconds": round(time.perf_counter() - started, 2),
                        "question": question,
                        "answer": answer,
                        "generation_trace": trace,
                        "indexed_document_ids": identifiers,
                        "retrieved_count": len(documents),
                        "reranker_applied": any(
                            "rerank_score" in doc.metadata for doc in documents
                        ),
                    }
                )
                save_report(report_path, report)
                if not passed:
                    raise RuntimeError(
                        f"The {kind} single-user functional check failed; load tests skipped."
                    )

            # One shared engine/client matches single-process deployment. Each
            # simulated user reaches the actual API endpoint and its worker pool.
            from fastapi.testclient import TestClient

            from deal_voice.api.main import app

            app.state.search_engine, app.state.generator = engine, generator

            def request_once(user: int) -> dict:
                started = time.perf_counter()
                with_client = TestClient(
                    app
                )  # No second lifespan/model initialization.
                try:
                    response = with_client.post(
                        "/api/v1/search", json={"query": args.pdf_question, "top_k": 3}
                    )
                    payload = response.json()
                    result = evaluate_generation(
                        str(user),
                        args.pdf_question,
                        "pilot-report.pdf",
                        args.pdf_expected,
                        payload.get("answer", ""),
                        expected_language="fr",
                    )
                    return {
                        "user": user,
                        "http_status": response.status_code,
                        "passed": response.status_code == 200 and result.passed,
                        "seconds": round(time.perf_counter() - started, 2),
                        "answer": payload.get("answer", ""),
                    }
                finally:
                    with_client.close()

            for users in () if args.single_user_only else (2, 5):
                needed = generation_budget(
                    users, args.generation_reserve_gib, args.min_free_gib
                )
                snapshot = memory_snapshot()
                if not memory_headroom(snapshot, needed):
                    report["concurrency"].append(
                        {
                            "users": users,
                            "status": "not_run_memory_budget",
                            "minimum_free_gib_required": needed,
                            "memory": snapshot,
                        }
                    )
                    save_report(report_path, report)
                    continue
                phase(f"concurrent_api_{users}_users")
                started = time.perf_counter()
                with ThreadPoolExecutor(max_workers=users) as pool:
                    results = list(pool.map(request_once, range(1, users + 1)))
                passed = all(result["passed"] for result in results)
                report["concurrency"].append(
                    {
                        "users": users,
                        "status": "passed" if passed else "failed",
                        "wall_seconds": round(time.perf_counter() - started, 2),
                        "requests": results,
                    }
                )
                save_report(report_path, report)
                if not passed:
                    break
            report["status"] = overall_status(report)
            phase("completed")
            return {"passed": 0, "failed": 1, "partial": 2}[report["status"]]
        except Exception as exc:
            report.update(
                status="failed", error_type=type(exc).__name__, error=str(exc)
            )
            save_report(report_path, report)
            return 1
        finally:
            if engine is not None:
                engine.client.close()
            streamlit_app.clear_streamlit_resources()


def supervise(args) -> int:
    stamp = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    report_dir = ROOT / "reports" / "pilot" / stamp
    workspace = ROOT / "data" / "raw" / f"pilot-{stamp}"
    report_dir.mkdir(parents=True)
    workspace.mkdir(parents=True)
    args.report_dir, args.workspace = str(report_dir), str(workspace)
    from deal_voice.config import settings

    response = requests.get(settings.ollama_base_url.rstrip("/") + "/api/ps", timeout=5)
    response.raise_for_status()
    if response.json().get("models"):
        raise RuntimeError(
            "Ollama already has active models; leave it idle before the isolated pilot."
        )
    if not memory_headroom(
        memory_snapshot(), args.min_free_gib + args.generation_reserve_gib
    ):
        raise MemoryError(
            "Insufficient initial memory headroom; no models were loaded."
        )
    child_args = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--workspace",
        args.workspace,
        "--report-dir",
        args.report_dir,
        "--pdf",
        str(args.pdf),
        "--pdf-page",
        str(args.pdf_page),
        "--pdf-question",
        args.pdf_question,
        "--audio-question",
        args.audio_question,
        "--audio-seconds",
        str(args.audio_seconds),
        "--min-free-gib",
        str(args.min_free_gib),
        "--generation-reserve-gib",
        str(args.generation_reserve_gib),
    ]
    if args.audio:
        child_args.extend(["--audio", str(args.audio)])
    if args.audio_first:
        child_args.append("--audio-first")
    if args.single_user_only:
        child_args.append("--single-user-only")
    for fact in args.pdf_expected:
        child_args.extend(["--pdf-expected", fact])
    for fact in args.audio_expected:
        child_args.extend(["--audio-expected", fact])
    environment = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join([str(ROOT / "src"), str(ROOT)]),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        PYTHONIOENCODING="utf-8",
        PYTHONUNBUFFERED="1",
        OLLAMA_NUM_THREAD="2",
        OLLAMA_KEEP_ALIVE="0s",
    )
    stopped = None
    started = time.monotonic()
    print(f"Pilot report: {report_dir}", flush=True)
    with (
        audio_process_job() as job_name,
        (report_dir / "worker.log").open("w", encoding="utf-8") as log,
    ):
        if job_name is not None:
            environment["DEAL_VOICE_PILOT_JOB"] = job_name
        process = subprocess.Popen(  # noqa: S603 -- own script/executable; explicit argv; no shell.
            child_args,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        last_phase = None
        with MemoryMonitor(report_dir / "supervisor.memory.jsonl"):
            while process.poll() is None:
                if not memory_headroom(memory_snapshot(), args.min_free_gib):
                    stopped = "minimum RAM/commit headroom reached"
                elif time.monotonic() - started > args.max_seconds:
                    stopped = "maximum pilot duration reached"
                if stopped:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
                    break
                result_path = report_dir / "result.json"
                if result_path.exists():
                    current = json.loads(result_path.read_text(encoding="utf-8")).get(
                        "phase"
                    )
                    if current != last_phase:
                        print(f"Phase: {current}", flush=True)
                        last_phase = current
                time.sleep(1)
    finalize_interrupted_report(report_dir / "result.json", process.returncode, stopped)
    save_report(
        report_dir / "supervisor.json",
        {
            "child_pid": process.pid,
            "returncode": process.returncode,
            "stopped_reason": stopped,
            "seconds": round(time.monotonic() - started, 2),
            "workspace": str(workspace),
        },
    )
    print(
        f"Pilot finished: returncode={process.returncode}, stopped={stopped}",
        flush=True,
    )
    return process.returncode or (1 if stopped else 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--pdf-page", type=int, default=3)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--audio-seconds", type=int, default=45)
    parser.add_argument(
        "--audio-first", action="store_true", help="Validate audio before the PDF case."
    )
    parser.add_argument(
        "--single-user-only",
        action="store_true",
        help="Skip concurrent API load stages.",
    )
    parser.add_argument(
        "--pdf-question",
        default="Quel produit a le prix de détail le plus élevé dans le référentiel ?",
    )
    parser.add_argument(
        "--audio-question",
        default="Dans quel équipement la société qui appelle est-elle spécialisée ?",
    )
    parser.add_argument("--pdf-expected", action="append", default=None)
    parser.add_argument("--audio-expected", action="append", default=None)
    parser.add_argument("--min-free-gib", type=float, default=4)
    parser.add_argument("--generation-reserve-gib", type=float, default=8)
    parser.add_argument("--max-seconds", type=int, default=900)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--workspace", help=argparse.SUPPRESS)
    parser.add_argument("--report-dir", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.pdf = args.pdf.resolve(strict=True)
    args.audio = args.audio.resolve(strict=True) if args.audio else None
    args.pdf_expected = args.pdf_expected or ["Lessive", "5,90"]
    args.audio_expected = args.audio_expected or ["fontaines"]
    if (
        min(
            args.min_free_gib,
            args.generation_reserve_gib,
            args.max_seconds,
            args.audio_seconds,
        )
        <= 0
    ):
        parser.error("Memory limits, audio duration and timeout must be positive.")
    return worker(args) if args.worker else supervise(args)


if __name__ == "__main__":
    raise SystemExit(main())
