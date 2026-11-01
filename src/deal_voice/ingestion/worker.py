"""Shared, bounded process transport for local audio and PDF extraction."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from deal_voice.ingestion.audio_process import (
    AudioProcessCleanupError,
    audio_process_job,
    join_audio_process_job,
)

_WORKER_SLOT = threading.BoundedSemaphore(1)
_WORKER_UNHEALTHY = False


def run_worker(
    module: str,
    request_json: str,
    *,
    timeout_seconds: float,
    cpu_threads: int,
    label: str,
) -> tuple[int, object]:
    """Only one extraction child per parent; never retry models inside the caller."""
    global _WORKER_UNHEALTHY

    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("The worker timeout must be positive and finite.")
    if type(cpu_threads) is not int or cpu_threads < 1:
        raise ValueError("The worker CPU thread count must be a positive integer.")
    environment = os.environ.copy()
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        environment.pop(name, None)
    source = str(Path(__file__).resolve().parents[2])
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (source, environment.get("PYTHONPATH")) if part
    )
    environment.update(
        OMP_NUM_THREADS=str(cpu_threads),
        MKL_NUM_THREADS=str(cpu_threads),
        OPENBLAS_NUM_THREADS=str(cpu_threads),
        TOKENIZERS_PARALLELISM="false",
        PYTHONIOENCODING="utf-8",
        DEAL_VOICE_WORKER_PARENT_PID=str(os.getpid()),
    )
    if not _WORKER_SLOT.acquire(timeout=timeout_seconds):
        raise TimeoutError("Timed out waiting for the ingestion worker slot.")
    try:
        if _WORKER_UNHEALTHY:
            raise RuntimeError(
                "Worker cleanup was not confirmed; restart the application before another upload."
            )
        with tempfile.TemporaryDirectory(prefix="dealvoice-ingestion-") as temporary:
            result_path = Path(temporary) / "result.json"
            with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as log:
                try:
                    with audio_process_job() as job_name:
                        environment.pop("DEAL_VOICE_WORKER_JOB", None)
                        if job_name is not None:
                            environment["DEAL_VOICE_WORKER_JOB"] = job_name
                        process = subprocess.run(  # noqa: S603 -- own module, fixed interpreter, argv without shell.
                            [
                                sys.executable,
                                "-m",
                                module,
                                "--result-json",
                                str(result_path),
                            ],
                            input=request_json,
                            encoding="utf-8",
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            env=environment,
                            timeout=timeout_seconds,
                            creationflags=subprocess.CREATE_NO_WINDOW
                            if os.name == "nt"
                            else 0,
                            check=False,
                        )
                except AudioProcessCleanupError:
                    _WORKER_UNHEALTHY = True
                    raise RuntimeError(
                        "Process termination could not be confirmed; restart the application."
                    ) from None
                except subprocess.TimeoutExpired:
                    raise TimeoutError(
                        f"{label} processing exceeded its time limit."
                    ) from None
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                raise RuntimeError(
                    f"{label} worker returned no valid result."
                ) from None
            return process.returncode, result
    finally:
        _WORKER_SLOT.release()


def watch_parent() -> None:
    """Join Windows containment, or watch the direct parent on other platforms."""
    if os.name == "nt":
        join_audio_process_job(os.environ["DEAL_VOICE_WORKER_JOB"])
        return
    parent_pid = os.getppid()

    def wait() -> None:
        while os.getppid() == parent_pid:
            time.sleep(0.5)
        os._exit(1)

    threading.Thread(target=wait, daemon=True, name="ingestion-parent-watch").start()
