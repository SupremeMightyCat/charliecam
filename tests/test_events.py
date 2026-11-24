import json
from pathlib import Path

import pytest

from charliecam.events import EventRecorder
from conftest import BLUE, RED, make_frame


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def recorder(tmp_path, clock):
    return EventRecorder(tmp_path / "events.jsonl", tmp_path / "snaps", cooldown_s=10, clock=clock)


def read_log(recorder):
    return [json.loads(line) for line in recorder.log_path.read_text().splitlines()]


def test_unknown_face_logged_with_snapshot(pipeline, recorder):
    frame = make_frame((20, 20, BLUE))
    events = recorder.record(frame, pipeline.process(frame))
    assert len(events) == 1
    ev = read_log(recorder)[0]
    assert ev["label"] == "Unknown #1"
    assert ev["known"] is False
    assert Path(ev["snapshot"]).exists()


def test_known_face_logged_without_snapshot(pipeline, recorder):
    pipeline.database.add("alice", RED)
    frame = make_frame((20, 20, RED))
    recorder.record(frame, pipeline.process(frame))
    ev = read_log(recorder)[0]
    assert ev["label"] == "alice" and ev["known"] is True
    assert ev["snapshot"] is None
    assert list(recorder.snapshot_dir.iterdir()) == []


def test_cooldown_suppresses_repeat_alerts(pipeline, recorder, clock):
    frame = make_frame((20, 20, BLUE))
    recorder.record(frame, pipeline.process(frame))
    clock.t += 5
    assert recorder.record(frame, pipeline.process(frame)) == []
    clock.t += 6  # 11 s after first sighting -> report again
    assert len(recorder.record(frame, pipeline.process(frame))) == 1
    assert len(read_log(recorder)) == 2


def test_cooldown_is_per_identity(pipeline, recorder):
    pipeline.database.add("alice", RED)
    recorder.record(make_frame(), pipeline.process(make_frame((20, 20, RED))))
    frame = make_frame((20, 20, RED), (200, 100, BLUE))
    events = recorder.record(frame, pipeline.process(frame))
    assert [e["label"] for e in events] == ["Unknown #1"]


def test_snapshot_known_option(pipeline, tmp_path):
    rec = EventRecorder(tmp_path / "e.jsonl", tmp_path / "s", snapshot_known=True)
    pipeline.database.add("alice", RED)
    frame = make_frame((20, 20, RED))
    assert rec.record(frame, pipeline.process(frame))[0]["snapshot"] is not None
