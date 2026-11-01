import builtins
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import pytest

from deal_voice.ingestion import audio_worker
from deal_voice.ingestion import worker as worker_transport


@pytest.fixture
def audio_path(tmp_path):
    path = tmp_path / "call.wav"
    path.write_bytes(b"not decoded by the parent process")
    return path


@pytest.fixture
def turns():
    return [
        {
            "speaker": "SPEAKER_00",
            "text": "Bonjour.",
            "start_time": 0.0,
            "end_time": 1.5,
        }
    ]


def mock_worker(monkeypatch, *, stdout, returncode=0, output=""):
    def run(command, **kwargs):
        if stdout is not None:
            result_index = command.index("--result-json") + 1
            Path(command[result_index]).write_text(stdout, encoding="utf-8")
        if output:
            kwargs["stdout"].write(output)
            kwargs["stdout"].flush()
        return subprocess.CompletedProcess(command, returncode)

    run_mock = Mock(side_effect=run)
    monkeypatch.setattr(worker_transport.subprocess, "run", run_mock)
    return run_mock


def test_worker_transports_settings_and_token_only_over_stdin(
    monkeypatch, audio_path, turns
):
    secret = f"hf_{uuid4().hex}"
    monkeypatch.setenv("DEAL_VOICE_WORKER_TEST", "inherited")
    monkeypatch.setenv("PYTHONPATH", "existing-package-path")
    token_names = ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"]
    for name in token_names:
        monkeypatch.setenv(name, "inherited-token")
    run = mock_worker(monkeypatch, stdout=json.dumps({"turns": turns}))

    result = audio_worker.run_audio_worker(
        audio_path,
        model_name="small",
        device="cpu",
        compute_type="int8",
        language="fr",
        batch_size=2,
        hf_token=secret,
        max_duration_seconds=12.5,
        timeout_seconds=17,
        cpu_threads=3,
    )

    assert result == turns
    run.assert_called_once()
    command = run.call_args.args[0]
    options = run.call_args.kwargs
    assert command[:3] == [sys.executable, "-m", "deal_voice.ingestion.audio_worker"]
    assert command[3] == "--result-json"
    assert Path(command[4]).is_absolute()
    assert secret not in repr(command)
    assert json.loads(options["input"]) == {
        "audio_path": str(audio_path.resolve()),
        "model_name": "small",
        "device": "cpu",
        "compute_type": "int8",
        "language": "fr",
        "batch_size": 2,
        "hf_token": secret,
        "max_duration_seconds": 12.5,
        "cpu_threads": 3,
    }
    assert options["timeout"] == 17
    assert options["stdout"] != subprocess.PIPE
    assert hasattr(options["stdout"], "write")
    assert options["stderr"] == subprocess.STDOUT
    assert not options.get("capture_output", False)
    assert options.get("text") or options.get("encoding")
    environment = options["env"]
    assert environment["DEAL_VOICE_WORKER_TEST"] == "inherited"
    assert secret not in repr(environment)
    assert not any(name in environment for name in token_names)
    paths = environment["PYTHONPATH"].split(os.pathsep)
    package_source = Path(audio_worker.__file__).resolve().parents[2]
    assert str(package_source) in paths
    assert "existing-package-path" in paths
    assert environment["OMP_NUM_THREADS"] == "3"
    assert environment["MKL_NUM_THREADS"] == "3"
    assert environment["DEAL_VOICE_WORKER_PARENT_PID"] == str(os.getpid())


def test_worker_preserves_default_transcription_settings(
    monkeypatch, audio_path, turns
):
    run = mock_worker(monkeypatch, stdout=json.dumps({"turns": turns}))

    audio_worker.run_audio_worker(audio_path, hf_token="test-token")

    options = run.call_args.kwargs
    request = json.loads(options["input"])
    assert request["model_name"] == "medium"
    assert request["device"] == "cpu"
    assert request["compute_type"] == "int8"
    assert request["batch_size"] == 1
    assert request["language"] is None
    assert request["max_duration_seconds"] is None
    assert request["cpu_threads"] == 2
    assert options["timeout"] == 3600


def test_parent_does_not_import_whisperx(monkeypatch, audio_path, turns):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "whisperx" or name.startswith("whisperx."):
            pytest.fail("WhisperX must only be imported inside the child process")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    mock_worker(monkeypatch, stdout=json.dumps({"turns": turns}))

    assert audio_worker.run_audio_worker(audio_path, hf_token="test-token") == turns


def test_worker_failure_is_generic_and_never_falls_back_inline(
    monkeypatch, audio_path, caplog
):
    secret = f"hf_{uuid4().hex}"
    private_output = "private transcript and worker internals"
    run = mock_worker(
        monkeypatch,
        returncode=1,
        stdout=f"{secret}: {private_output}",
        output=f"{secret}: {private_output}",
    )
    from deal_voice.ingestion.audio import WhisperXTranscriber

    inline = Mock(side_effect=AssertionError("inline transcription is forbidden"))
    monkeypatch.setattr(WhisperXTranscriber, "transcribe_and_diarize", inline)
    monkeypatch.setattr(
        WhisperXTranscriber, "_transcribe_and_diarize_in_process", inline
    )

    with pytest.raises(RuntimeError) as error:
        audio_worker.run_audio_worker(audio_path, hf_token=secret)

    run.assert_called_once()
    inline.assert_not_called()
    details = "".join(traceback.format_exception(error.value)) + caplog.text
    assert secret not in details
    assert private_output not in details


def test_worker_timeout_is_clean_and_never_retries(monkeypatch, audio_path, caplog):
    secret = f"hf_{uuid4().hex}"
    run = Mock(
        side_effect=subprocess.TimeoutExpired(
            [sys.executable, "-m", "deal_voice.ingestion.audio_worker"],
            timeout=0.1,
            output=secret,
            stderr=secret,
        )
    )
    monkeypatch.setattr(worker_transport.subprocess, "run", run)

    with pytest.raises(TimeoutError) as error:
        audio_worker.run_audio_worker(audio_path, hf_token=secret, timeout_seconds=0.1)

    run.assert_called_once()
    assert secret not in "".join(traceback.format_exception(error.value))
    assert secret not in caplog.text


def test_worker_duration_error_is_actionable_without_leaking_output(
    monkeypatch, audio_path
):
    mock_worker(
        monkeypatch,
        stdout=json.dumps({"error": "audio_too_long"}),
        returncode=2,
    )

    with pytest.raises(ValueError):
        audio_worker.run_audio_worker(
            audio_path, hf_token="test-token", max_duration_seconds=30
        )


@pytest.mark.parametrize("failure", ["nonzero", "invalid-result", "timeout"])
def test_worker_slot_is_released_after_failure(monkeypatch, audio_path, turns, failure):
    run = mock_worker(
        monkeypatch,
        stdout="{}",
        returncode=1 if failure == "nonzero" else 0,
    )
    if failure == "timeout":
        run.side_effect = subprocess.TimeoutExpired("worker", timeout=0.1)
    error_type = TimeoutError if failure == "timeout" else RuntimeError
    with pytest.raises(error_type):
        audio_worker.run_audio_worker(
            audio_path, hf_token="test-token", timeout_seconds=0.1
        )

    mock_worker(monkeypatch, stdout=json.dumps({"turns": turns}))

    assert (
        audio_worker.run_audio_worker(
            audio_path, hf_token="test-token", timeout_seconds=0.1
        )
        == turns
    )


def test_unconfirmed_cleanup_blocks_further_workers(monkeypatch, audio_path, turns):
    @contextmanager
    def failed_cleanup():
        yield None
        raise worker_transport.AudioProcessCleanupError("termination not confirmed")

    run = mock_worker(monkeypatch, stdout=json.dumps({"turns": turns}))
    with monkeypatch.context() as scope:
        scope.setattr(worker_transport, "_WORKER_UNHEALTHY", False)
        scope.setattr(worker_transport, "audio_process_job", failed_cleanup)

        with pytest.raises(RuntimeError, match="restart"):
            audio_worker.run_audio_worker(audio_path, hf_token="test-token")
        run.assert_called_once()

        with pytest.raises(RuntimeError, match="restart"):
            audio_worker.run_audio_worker(
                audio_path, hf_token="test-token", timeout_seconds=0.1
            )
        run.assert_called_once()

    assert audio_worker.run_audio_worker(audio_path, hf_token="test-token") == turns
    assert run.call_count == 2


@pytest.mark.parametrize(
    "stdout",
    [
        None,
        "",
        "not JSON",
        'model progress\n{"turns": []}',
        "null",
        "{}",
        '{"turns": {}}',
        '{"turns": [{}]}',
        '{"turns": [{"speaker": "A", "text": 7, "start_time": 0, "end_time": 1}]}',
        '{"turns": [{"speaker": 7, "text": "Hello", "start_time": 0, "end_time": 1}]}',
        '{"turns": [{"speaker": "A", "text": "Hello", "start_time": "0", "end_time": 1}]}',
        '{"turns": [{"speaker": "A", "text": "Hello", "start_time": true, "end_time": 1}]}',
        '{"turns": [{"speaker": "A", "text": "Hello", "start_time": -1, "end_time": 1}]}',
        '{"turns": [{"speaker": "A", "text": "Hello", "start_time": NaN, "end_time": 1}]}',
        '{"turns": [{"speaker": "A", "text": "Hello", "start_time": 2, "end_time": 1}]}',
        '{"error": "processing_failed"}',
    ],
    ids=[
        "missing-result-file",
        "no-output",
        "malformed-json",
        "stdout-contaminated-by-logs",
        "null-result",
        "missing-turns",
        "turns-not-list",
        "missing-turn-fields",
        "non-string-text",
        "non-string-speaker",
        "non-numeric-timestamp",
        "boolean-timestamp",
        "negative-timestamp",
        "non-finite-timestamp",
        "reversed-timestamps",
        "error-envelope",
    ],
)
def test_invalid_worker_output_fails_closed(monkeypatch, audio_path, stdout):
    run = mock_worker(monkeypatch, stdout=stdout)

    with pytest.raises(RuntimeError):
        audio_worker.run_audio_worker(audio_path, hf_token="test-token")

    run.assert_called_once()


def test_real_worker_rejects_missing_audio_before_loading_whisperx(tmp_path):
    marker = tmp_path / "whisperx-was-imported"
    (tmp_path / "sitecustomize.py").write_text(
        "import builtins\n"
        "from pathlib import Path\n"
        "original_import = builtins.__import__\n"
        "def guarded_import(name, *args, **kwargs):\n"
        "    if name == 'whisperx' or name.startswith('whisperx.'):\n"
        f"        Path({str(marker)!r}).write_text('attempted')\n"
        "        raise AssertionError('model import is forbidden')\n"
        "    return original_import(name, *args, **kwargs)\n"
        "builtins.__import__ = guarded_import\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    source = Path(audio_worker.__file__).resolve().parents[2]
    environment["PYTHONPATH"] = os.pathsep.join([str(tmp_path), str(source)])
    secret = f"hf_{uuid4().hex}"
    request = {
        "audio_path": str(tmp_path / "does-not-exist.wav"),
        "model_name": "medium",
        "device": "cpu",
        "compute_type": "int8",
        "language": None,
        "batch_size": 1,
        "hf_token": secret,
        "max_duration_seconds": None,
        "cpu_threads": 2,
    }

    result_path = tmp_path / "result.json"
    result = subprocess.run(  # noqa: S603 - fixed module and temporary test paths.
        [
            sys.executable,
            "-m",
            "deal_voice.ingestion.audio_worker",
            "--result-json",
            str(result_path),
        ],
        input=json.dumps(request),
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
        timeout=20,
        check=False,
    )

    assert result.returncode != 0
    assert not marker.exists(), result.stderr
    assert "No module named" not in result.stderr
    assert secret not in result.stdout + result.stderr
    assert json.loads(result_path.read_text(encoding="utf-8")) == {
        "error": "invalid_request"
    }


@pytest.fixture
def sleeping_worker(monkeypatch, tmp_path):
    package = tmp_path / "whisperx"
    package.mkdir()
    (package / "__init__.py").write_text(
        "import os\n"
        "import time\n"
        "from pathlib import Path\n"
        "def load_audio(path):\n"
        "    Path(os.environ['DEAL_VOICE_WORKER_PID']).write_text(str(os.getpid()))\n"
        "    time.sleep(60)\n",
        encoding="utf-8",
    )
    (package / "diarize.py").write_text(
        "class DiarizationPipeline:\n    pass\n", encoding="utf-8"
    )
    (tmp_path / "torch.py").write_text(
        "def set_num_threads(count):\n    pass\n"
        "def set_num_interop_threads(count):\n    pass\n",
        encoding="utf-8",
    )
    pid_path = tmp_path / "worker.pid"
    source = Path(audio_worker.__file__).resolve().parents[2]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), str(source)]))
    monkeypatch.setenv("DEAL_VOICE_WORKER_PID", str(pid_path))
    return pid_path


def process_is_running(pid):
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x00100000, False, pid)
        if not handle:
            return False
        try:
            return kernel.WaitForSingleObject(handle, 0) == 258
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    status = Path(f"/proc/{pid}/stat")
    if status.exists() and status.read_text().split()[2] == "Z":
        return False
    return True


def kill_owned_worker(pid):
    if not process_is_running(pid):
        return
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(1, False, pid)
        if handle:
            try:
                kernel.TerminateProcess(handle, 1)
            finally:
                kernel.CloseHandle(handle)
    else:
        os.kill(pid, signal.SIGKILL)


def wait_until(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def test_real_timeout_stops_the_worker(audio_path, sleeping_worker):
    try:
        with pytest.raises(TimeoutError):
            audio_worker.run_audio_worker(
                audio_path, hf_token="test-token", timeout_seconds=5
            )

        assert sleeping_worker.exists(), "fake WhisperX did not reach audio loading"
        pid = int(sleeping_worker.read_text())
        assert wait_until(lambda: not process_is_running(pid))
    finally:
        if sleeping_worker.exists():
            kill_owned_worker(int(sleeping_worker.read_text()))


@pytest.mark.skipif(sys.platform != "win32", reason="Windows job containment")
def test_real_timeout_stops_worker_and_descendant(
    audio_path, sleeping_worker, tmp_path
):
    descendant_pid_path = tmp_path / "descendant.pid"
    descendant_script = (
        "import os, time\n"
        "from pathlib import Path\n"
        f"Path({str(descendant_pid_path)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(60)\n"
    )
    (tmp_path / "whisperx" / "__init__.py").write_text(
        "import os\n"
        "import subprocess\n"
        "import sys\n"
        "import time\n"
        "from pathlib import Path\n"
        "def load_audio(path):\n"
        "    Path(os.environ['DEAL_VOICE_WORKER_PID']).write_text(str(os.getpid()))\n"
        f"    subprocess.Popen([sys.executable, '-c', {descendant_script!r}], "
        "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        "    time.sleep(60)\n",
        encoding="utf-8",
    )
    try:
        with pytest.raises(TimeoutError):
            audio_worker.run_audio_worker(
                audio_path, hf_token="test-token", timeout_seconds=5
            )

        assert sleeping_worker.exists(), "fake WhisperX did not reach audio loading"
        assert descendant_pid_path.exists(), "worker descendant did not start"
        worker_pid = int(sleeping_worker.read_text())
        descendant_pid = int(descendant_pid_path.read_text())
        assert wait_until(
            lambda: (
                not process_is_running(worker_pid)
                and not process_is_running(descendant_pid)
            )
        )
    finally:
        for marker in (sleeping_worker, descendant_pid_path):
            if marker.exists():
                kill_owned_worker(int(marker.read_text()))


def test_worker_exits_when_its_parent_dies(audio_path, sleeping_worker, tmp_path):
    launcher_pid_path = tmp_path / "launcher.pid"
    script = (
        "from deal_voice.ingestion.audio_worker import run_audio_worker\n"
        "from pathlib import Path\n"
        "import os\n"
        f"Path({str(launcher_pid_path)!r}).write_text(str(os.getpid()))\n"
        f"run_audio_worker(Path({str(audio_path)!r}), "
        "hf_token='test-token', timeout_seconds=60)\n"
    )
    with (tmp_path / "launcher.log").open("w+", encoding="utf-8") as log:
        parent = subprocess.Popen(  # noqa: S603 - fixed Python launcher and owned paths.
            [sys.executable, "-c", script],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
        try:
            assert wait_until(sleeping_worker.exists, timeout=20), (
                "fake WhisperX did not start; launcher exit=" + str(parent.poll())
            )
            pid = int(sleeping_worker.read_text())
            assert process_is_running(pid)

            # A Windows venv launcher can wrap the actual parent interpreter.
            launcher_pid = int(launcher_pid_path.read_text())
            kill_owned_worker(launcher_pid)
            parent.wait(timeout=5)

            assert wait_until(lambda: not process_is_running(pid)), (
                "audio worker survived its parent process"
            )
        finally:
            if launcher_pid_path.exists():
                kill_owned_worker(int(launcher_pid_path.read_text()))
            if parent.poll() is None:
                parent.kill()
            parent.wait(timeout=5)
            if sleeping_worker.exists():
                kill_owned_worker(int(sleeping_worker.read_text()))
