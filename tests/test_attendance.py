"""Attendance report built from the event log."""
import json
from datetime import date, datetime, timedelta, timezone

from charliecam import cli
from charliecam.attendance import read_events, report, to_csv
from charliecam.database import FaceDatabase


def local(day, hour, minute=0):
    return datetime(day.year, day.month, day.day, hour, minute).astimezone()


def write_log(path, events):
    lines = [json.dumps({"timestamp": ts.astimezone(timezone.utc).isoformat(), "label": label,
                         "known": known, "similarity": 0.7}) for ts, label, known in events]
    path.write_text("\n".join(lines + ["{broken"]) + "\n", encoding="utf-8")


DAY = date(2026, 11, 23)
PREV = DAY - timedelta(days=7)


def sample_log(tmp_path):
    log = tmp_path / "events.jsonl"
    write_log(log, [
        (local(PREV, 18), "Ann", True),
        (local(DAY, 18, 5), "Ben", True),
        (local(DAY, 18, 1), "Ann", True),
        (local(DAY, 18, 30), "Ann", True),
        (local(DAY, 18, 10), "Unknown #2", False),
    ])
    return log


def test_report_present_absent_and_days(tmp_path):
    rep = report(sample_log(tmp_path), ["Ann", "Ben", "cara"], DAY)
    assert rep["date"] == "2026-11-23"
    assert [a["name"] for a in rep["present"]] == ["Ann", "Ben"]  # order of arrival
    ann = rep["present"][0]
    assert ann["sightings"] == 2
    assert datetime.fromisoformat(ann["first_seen"]).hour == 18
    assert datetime.fromisoformat(ann["last_seen"]).minute == 30
    assert rep["absent"] == ["cara"]
    assert rep["days"] == ["2026-11-23", "2026-11-16"]


def test_unknown_faces_never_count(tmp_path):
    rep = report(sample_log(tmp_path), [], DAY)
    assert all(not a["name"].startswith("Unknown") for a in rep["present"])


def test_missing_log_and_bad_lines(tmp_path):
    assert list(read_events(tmp_path / "none.jsonl")) == []
    assert len(list(read_events(sample_log(tmp_path)))) == 5  # "{broken" skipped
    assert report(tmp_path / "none.jsonl", ["Ann"], DAY)["absent"] == ["Ann"]


def test_csv(tmp_path):
    rows = to_csv(report(sample_log(tmp_path), ["Ann", "Ben", "cara"], DAY)).splitlines()
    assert rows[0] == "date,name,status,first_seen,last_seen,sightings"
    assert rows[1] == "2026-11-23,Ann,present,18:01:00,18:30:00,2"
    assert rows[-1] == "2026-11-23,cara,absent,,,0"


def test_cli_attendance(tmp_path, capsys):
    log = sample_log(tmp_path)
    log.rename(tmp_path / "events.jsonl")
    db = FaceDatabase(0.5)
    db.add("Ann", [1, 0])
    db.add("Dee", [0, 1])
    db.save(tmp_path / "faces.npz")
    out_csv = tmp_path / "out.csv"
    code = cli.main(["--data-dir", str(tmp_path), "attendance", "--date", "2026-11-23",
                     "--csv", str(out_csv)])
    text = capsys.readouterr().out
    assert code == 0
    assert "2 present, 1 absent" in text and "absent   Dee" in text
    assert "Ann,present" in out_csv.read_text()
