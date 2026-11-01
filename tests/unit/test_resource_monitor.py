import json
import threading

import pytest

from deal_voice.evaluation import resources


def test_monitor_flushes_periodic_samples_with_question_and_phase(
    tmp_path, monkeypatch
):
    sampled = threading.Event()
    snapshots = []

    def fake_snapshot():
        snapshots.append(True)
        if len(snapshots) >= 3:
            sampled.set()
        return {"system_available_bytes": 12345}

    monkeypatch.setattr(resources, "memory_snapshot", fake_snapshot)
    path = tmp_path / "memory.jsonl"
    with resources.MemoryMonitor(path, interval_seconds=0.01) as monitor:
        monitor.mark("generation", "audio-001")
        assert sampled.wait(timeout=2)
        with monitor._lock:
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
        assert rows[0]["phase"] == "start"
        assert any(
            row["phase"] == "generation" and row["question_id"] == "audio-001"
            for row in rows
        )
        assert all(row["system_available_bytes"] == 12345 for row in rows)

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["phase"] == "finished"
    assert len(rows) >= 4


def test_monitor_logs_sampling_error_without_breaking_evaluation(
    tmp_path, monkeypatch, caplog
):
    def failing_snapshot():
        raise OSError("memory counters unavailable")

    monkeypatch.setattr(resources, "memory_snapshot", failing_snapshot)
    path = tmp_path / "memory.jsonl"
    with resources.MemoryMonitor(path):
        pass

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["sampling_error"] == "memory counters unavailable"
    assert "Could not sample memory" in caplog.text


def test_monitor_records_interrupted_run_and_preserves_exception(tmp_path, monkeypatch):
    monkeypatch.setattr(resources, "memory_snapshot", lambda: {})
    path = tmp_path / "memory.jsonl"
    with pytest.raises(RuntimeError, match="generation failed"):
        with resources.MemoryMonitor(path):
            raise RuntimeError("generation failed")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["phase"] == "error"


def test_monitor_rejects_nonpositive_sampling_interval(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        resources.MemoryMonitor(tmp_path / "memory.jsonl", interval_seconds=0)


def test_final_logging_failure_preserves_generation_exception(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setattr(resources, "memory_snapshot", lambda: {})

    class BrokenLog:
        def write(self, text):
            raise OSError("disk write failed")

        def close(self):
            raise OSError("disk close failed")

    with pytest.raises(RuntimeError, match="generation failed"):
        with resources.MemoryMonitor(tmp_path / "memory.jsonl") as monitor:
            monitor._file.close()
            monitor._file = BrokenLog()
            raise RuntimeError("generation failed")

    assert "Could not finalize memory log: disk write failed" in caplog.text
    assert "Could not close memory log: disk close failed" in caplog.text
