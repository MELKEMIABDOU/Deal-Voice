"""Lightweight Windows memory observations without importing ML libraries."""

from __future__ import annotations

import ctypes
import json
import logging
import os
import threading
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_uint32),
        ("load", ctypes.c_uint32),
        *[
            (name, ctypes.c_uint64)
            for name in (
                "total_physical",
                "available_physical",
                "total_pagefile",
                "available_pagefile",
                "total_virtual",
                "available_virtual",
                "available_extended_virtual",
            )
        ],
    ]


class _ProcessMemory(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_uint32),
        ("page_faults", ctypes.c_uint32),
        *[
            (name, ctypes.c_size_t)
            for name in (
                "peak_working_set",
                "working_set",
                "peak_paged_pool",
                "paged_pool",
                "peak_nonpaged_pool",
                "nonpaged_pool",
                "pagefile",
                "peak_pagefile",
                "private",
            )
        ],
    ]


@lru_cache(maxsize=1)
def _windows_libraries():
    """Configure Win32 calls once rather than loading DLLs for every sample."""
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MemoryStatus)]
    kernel.GlobalMemoryStatusEx.restype = wintypes.BOOL
    kernel.GetCurrentProcess.argtypes = []
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_ProcessMemory),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    return kernel, psapi


def memory_snapshot() -> dict[str, int | None]:
    """Return byte counts; commit means system commit, not free disk space."""
    snapshot: dict[str, int | None] = dict.fromkeys(
        (
            "system_total_bytes",
            "system_available_bytes",
            "commit_limit_bytes",
            "commit_available_bytes",
            "process_working_set_bytes",
            "process_private_bytes",
            "process_peak_working_set_bytes",
        )
    )
    if os.name != "nt":
        return snapshot
    kernel, psapi = _windows_libraries()
    status = _MemoryStatus(length=ctypes.sizeof(_MemoryStatus))
    process = _ProcessMemory(length=ctypes.sizeof(_ProcessMemory))
    if not kernel.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError(ctypes.get_last_error())
    if not psapi.GetProcessMemoryInfo(
        kernel.GetCurrentProcess(), ctypes.byref(process), ctypes.sizeof(process)
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    snapshot.update(
        {
            "system_total_bytes": status.total_physical,
            "system_available_bytes": status.available_physical,
            "commit_limit_bytes": status.total_pagefile,
            "commit_available_bytes": status.available_pagefile,
            "process_working_set_bytes": process.working_set,
            "process_private_bytes": process.private,
            "process_peak_working_set_bytes": process.peak_working_set,
        }
    )
    return snapshot


class MemoryMonitor:
    """Flush memory samples to JSONL; this observes usage, not a memory cap."""

    def __init__(self, path: Path, interval_seconds: float = 1.0):
        if interval_seconds <= 0:
            raise ValueError("Memory sampling interval must be positive.")
        self.path = Path(path)
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._phase = "start"
        self._question_id: str | None = None

    def __enter__(self) -> MemoryMonitor:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")
        self._stop.clear()
        self.mark("start")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def mark(self, phase: str, question_id: str | None = None) -> None:
        with self._lock:
            self._phase = phase
            self._question_id = question_id
        self._sample()

    def _sample(self) -> None:
        with self._lock:
            row = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "pid": os.getpid(),
                "phase": self._phase,
                "question_id": self._question_id,
                "platform": os.name,
            }
            try:
                row.update(memory_snapshot())
            except Exception as error:
                logger.warning("Could not sample memory: %s", error)
                row["sampling_error"] = str(error)
            self._file.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._file.flush()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._stop.set()
        self._thread.join()
        try:
            try:
                self.mark("error" if exc_type is not None else "finished")
            except OSError as error:
                if exc_type is None:
                    raise
                logger.warning("Could not finalize memory log: %s", error)
        finally:
            try:
                self._file.close()
            except OSError as error:
                if exc_type is None:
                    raise
                logger.warning("Could not close memory log: %s", error)
