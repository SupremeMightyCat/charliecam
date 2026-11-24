"""CLI commands and the live loop, end to end, using fakes and a video file."""
import json

import cv2
import pytest

from charliecam import cli
from charliecam.camera import run_live
from charliecam.database import FaceDatabase
from charliecam.events import EventRecorder
from charliecam.pipeline import RecognitionPipeline
from conftest import BLUE, GREEN, FakeDetector, FakeEmbedder, make_frame


@pytest.fixture
def fake_models(monkeypatch):
    monkeypatch.setattr(
        cli, "_build_pipeline",
        lambda cfg, db: RecognitionPipeline(FakeDetector(), FakeEmbedder(), db),
    )


def run(tmp_path, *args):
    return cli.main(["--data-dir", str(tmp_path), "--threshold", "0.9", *args])


def test_list_empty(tmp_path, capsys):
    assert run(tmp_path, "list") == 0
    assert "No one enrolled" in capsys.readouterr().out


def test_enroll_from_folder_list_and_remove(tmp_path, capsys, fake_models):
    photos = tmp_path / "photos"
    photos.mkdir()
    cv2.imwrite(str(photos / "a.png"), make_frame((20, 20, GREEN)))
    cv2.imwrite(str(photos / "b.png"), make_frame((90, 40, GREEN)))
    cv2.imwrite(str(photos / "empty.png"), make_frame())  # no face -> skipped

    assert run(tmp_path, "enroll", "bob", str(photos)) == 0
    assert "Added 2 sample(s)" in capsys.readouterr().out

    run(tmp_path, "list")
    assert "bob: 2 sample(s)" in capsys.readouterr().out

    assert run(tmp_path, "remove", "bob") == 0
    assert run(tmp_path, "remove", "bob") == 1


def test_enroll_without_images_errors(tmp_path, fake_models):
    assert run(tmp_path, "enroll", "bob") == 2


def test_enroll_unreadable_file(tmp_path, fake_models):
    bad = tmp_path / "bad.jpg"
    bad.write_text("not an image")
    assert run(tmp_path, "enroll", "bob", str(bad)) == 1


def test_missing_models_give_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download-models"):
        cli._build_pipeline(cli.Config(models_dir=tmp_path), FaceDatabase(0.5))


def _write_video(path, frames):
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (w, h))
    if not writer.isOpened():
        pytest.skip("No video encoder available")
    for f in frames:
        writer.write(f)
    writer.release()


def test_live_loop_on_video_file(tmp_path):
    """Known person, then a stranger: one event each thanks to the cooldown."""
    video = tmp_path / "clip.avi"
    frames = [make_frame((40, 40, GREEN))] * 5 + [make_frame((40, 40, GREEN), (200, 120, BLUE))] * 5
    _write_video(video, frames)

    db = FaceDatabase(threshold=0.9)
    db.add("bob", GREEN)
    pipeline = RecognitionPipeline(FakeDetector(), FakeEmbedder(), db)
    rec = EventRecorder(tmp_path / "events.jsonl", tmp_path / "snaps", cooldown_s=60)

    processed = run_live(pipeline, rec, str(video), show_window=False)

    assert processed == 10
    events = [json.loads(line) for line in rec.log_path.read_text().splitlines()]
    assert [e["label"] for e in events] == ["bob", "Unknown #1"]
    assert len(list(rec.snapshot_dir.glob("*.jpg"))) == 1


def test_bad_source_raises(tmp_path):
    pipeline = RecognitionPipeline(FakeDetector(), FakeEmbedder(), FaceDatabase(0.9))
    rec = EventRecorder(tmp_path / "e.jsonl", tmp_path / "s")
    with pytest.raises(RuntimeError, match="Could not open"):
        run_live(pipeline, rec, str(tmp_path / "missing.avi"), show_window=False)
