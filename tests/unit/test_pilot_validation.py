import json
from pathlib import Path

import pytest

from scripts import validate_pilot as pilot


def test_only_explicit_single_user_mode_can_pass_without_load_tests():
    report = {"single_user": [{"passed": True}], "concurrency": []}
    assert pilot.overall_status(report) == "partial"
    assert pilot.overall_status({**report, "single_user_only": True}) == "passed"
    report["single_user"][0]["passed"] = False
    assert pilot.overall_status({**report, "single_user_only": True}) == "failed"


def test_supervisor_finalizes_a_killed_worker_without_erasing_completed_cases(tmp_path):
    path = tmp_path / "result.json"
    pilot.save_report(
        path,
        {
            "status": "running",
            "phase": "upload_and_index_pdf",
            "single_user": [{"kind": "audio", "passed": True}],
        },
    )
    pilot.finalize_interrupted_report(path, 1, "minimum RAM/commit headroom reached")
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert result["phase"] == "upload_and_index_pdf"
    assert result["single_user"] == [{"kind": "audio", "passed": True}]
    assert result["stopped_reason"] == "minimum RAM/commit headroom reached"


@pytest.mark.parametrize(
    "available,commit,expected",
    [
        (5, 5, True),
        (5, 3, False),
        (3, 5, False),
        (None, 5, False),
        (5, None, False),
    ],
)
def test_memory_guard_requires_both_ram_and_commit_headroom(
    available, commit, expected
):
    snapshot = {
        "system_available_bytes": available * pilot.GIB
        if available is not None
        else None,
        "commit_available_bytes": commit * pilot.GIB if commit is not None else None,
    }
    assert pilot.memory_headroom(snapshot, 4) is expected


def test_larger_load_requires_additional_admission_headroom():
    assert pilot.generation_budget(1, 8, 4) < pilot.generation_budget(2, 8, 4)
    assert pilot.generation_budget(2, 8, 4) < pilot.generation_budget(5, 8, 4)


@pytest.mark.parametrize(
    "loads,expected",
    [
        (
            [{"users": 2, "status": "passed"}, {"users": 5, "status": "passed"}],
            "passed",
        ),
        ([{"users": 2, "status": "failed"}], "failed"),
        (
            [
                {"users": 2, "status": "passed"},
                {"users": 5, "status": "not_run_memory_budget"},
            ],
            "partial",
        ),
        ([], "partial"),
    ],
)
def test_skipped_or_failed_load_stage_cannot_be_reported_as_success(loads, expected):
    assert (
        pilot.overall_status({"single_user": [{"passed": True}], "concurrency": loads})
        == expected
    )


def test_workspace_guard_rejects_broad_or_outside_paths(monkeypatch):
    root = Path.cwd().resolve()
    monkeypatch.setattr(pilot, "ROOT", root)
    report = root / "reports" / "pilot" / "test"
    for invalid in (root, root.parent, root / "data" / "raw", root / "local_qdrant_db"):
        with pytest.raises(ValueError):
            pilot.validated_workspace(str(invalid), str(report))
    work, output = pilot.validated_workspace(
        str(root / "data" / "raw" / "pilot-test"), str(report)
    )
    assert work.name == "pilot-test"
    assert output == report
