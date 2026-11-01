"""Contain Windows audio workers and their descendants in a parent-owned job."""

from __future__ import annotations

import ctypes
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from uuid import uuid4

_JOB_OBJECT_ASSIGN_PROCESS = 0x0001
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION = 1
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_ERROR_ALREADY_EXISTS = 183
_DRAIN_TIMEOUT_SECONDS = 5.0


class AudioProcessCleanupError(RuntimeError):
    """Worker termination could not be confirmed; do not admit another worker."""


class _BasicLimitInformation(ctypes.Structure):
    # DWORD stays 32 bits and SIZE_T/ULONG_PTR follow the interpreter's pointer size.
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_uint64)
        for name in (
            "ReadOperationCount",
            "WriteOperationCount",
            "OtherOperationCount",
            "ReadTransferCount",
            "WriteTransferCount",
            "OtherTransferCount",
        )
    ]


class _ExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _BasicAccountingInformation(ctypes.Structure):
    _fields_ = [
        ("TotalUserTime", ctypes.c_int64),
        ("TotalKernelTime", ctypes.c_int64),
        ("ThisPeriodTotalUserTime", ctypes.c_int64),
        ("ThisPeriodTotalKernelTime", ctypes.c_int64),
        ("TotalPageFaultCount", ctypes.c_uint32),
        ("TotalProcesses", ctypes.c_uint32),
        ("ActiveProcesses", ctypes.c_uint32),
        ("TotalTerminatedProcesses", ctypes.c_uint32),
    ]


@lru_cache(maxsize=1)
def _kernel32():
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.OpenJobObjectW.argtypes = [
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.LPCWSTR,
    ]
    kernel.OpenJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel.SetInformationJobObject.restype = wintypes.BOOL
    kernel.QueryInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel.QueryInformationJobObject.restype = wintypes.BOOL
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.TerminateJobObject.restype = wintypes.BOOL
    kernel.GetCurrentProcess.argtypes = []
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    return kernel


def _terminate_and_wait(kernel, handle) -> None:
    """Confirm the job is empty before the caller can release its worker slot."""
    if not kernel.TerminateJobObject(handle, 1):
        raise AudioProcessCleanupError(
            "Could not terminate the audio process job."
        ) from ctypes.WinError(ctypes.get_last_error())
    deadline = time.monotonic() + _DRAIN_TIMEOUT_SECONDS
    while True:
        accounting = _BasicAccountingInformation()
        if not kernel.QueryInformationJobObject(
            handle,
            _JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION,
            ctypes.byref(accounting),
            ctypes.sizeof(accounting),
            None,
        ):
            raise AudioProcessCleanupError(
                "Could not confirm audio process termination."
            ) from ctypes.WinError(ctypes.get_last_error())
        if accounting.ActiveProcesses == 0:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AudioProcessCleanupError(
                "Audio processes remained active after termination."
            )
        time.sleep(min(0.025, remaining))


@contextmanager
def audio_process_job() -> Iterator[str | None]:
    """Own a job until all audio processes terminate; yield None off Windows."""
    if os.name != "nt":
        yield None
        return

    kernel = _kernel32()
    name = f"Local\\DealVoiceAudio-{uuid4().hex}"
    # NULL security attributes use the creator's default DACL and a handle that
    # cannot be inherited. Only the application keeps an owning handle open.
    ctypes.set_last_error(0)
    handle = kernel.CreateJobObjectW(None, name)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    creation_error = ctypes.get_last_error()
    try:
        if creation_error == _ERROR_ALREADY_EXISTS:
            raise FileExistsError("The audio process job name already exists.")
        limits = _ExtendedLimitInformation()
        limits.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel.SetInformationJobObject(
            handle,
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            yield name
        finally:
            # Termination is enforced by the kernel even when native code holds
            # Python's GIL. Closing the owner on application death also kills
            # workers and their descendants, including FFmpeg.
            _terminate_and_wait(kernel, handle)
    finally:
        kernel.CloseHandle(handle)


def join_audio_process_job(name: str) -> None:
    """Join before loading models, then immediately relinquish the job handle."""
    if os.name != "nt":
        return

    kernel = _kernel32()
    handle = kernel.OpenJobObjectW(_JOB_OBJECT_ASSIGN_PROCESS, False, name)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        # Windows 8+ supports nested jobs, including a containing launcher job.
        # If containment cannot be established, fail before model loading;
        # callers must never fall back to processing audio inline.
        if not kernel.AssignProcessToJobObject(handle, kernel.GetCurrentProcess()):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        # Retaining this child handle would defeat KILL_ON_JOB_CLOSE on owner
        # death. Descendants inherit job membership without inheriting handles.
        kernel.CloseHandle(handle)
