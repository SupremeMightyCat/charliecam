import pytest

from charliecam.enrollment import EnrollmentSession, delete_face_photos, face_folder, validate_name
from conftest import BLUE, GREEN, make_frame


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


@pytest.fixture
def clock():
    return Clock()


def session(db, tmp_path, clock, samples=3, name="Maria"):
    return EnrollmentSession(name, db, tmp_path / "faces", samples=samples, interval_s=0.5, clock=clock)


def feed_n(s, pipeline, frame, clock, n):
    for _ in range(n):
        clock.t += 1.0
        progress = s.feed(frame, pipeline.process(frame))
    return progress


@pytest.mark.parametrize("raw,clean", [("  Maria  ", "Maria"), ("Ana   Lucia", "Ana Lucia")])
def test_validate_name_cleans(raw, clean):
    assert validate_name(raw) == clean


@pytest.mark.parametrize("bad", ["", "   ", "Unknown", "unknown #3", "x" * 41])
def test_validate_name_rejects(bad):
    with pytest.raises(ValueError):
        validate_name(bad)


def test_face_folder_is_filesystem_safe(tmp_path):
    assert face_folder(tmp_path, "Ana Lucia").name == "Ana_Lucia"
    assert face_folder(tmp_path, "../../etc").name == "etc"
    assert face_folder(tmp_path, "***").name == "person"


def test_full_enrollment_commits_samples_and_photos(pipeline, tmp_path, clock):
    s = session(pipeline.database, tmp_path, clock)
    progress = feed_n(s, pipeline, make_frame((40, 40, GREEN)), clock, 3)
    assert progress.done and progress.captured == 3
    assert pipeline.database.people() == {"Maria": 3}
    assert len(list((tmp_path / "faces" / "Maria").glob("*.jpg"))) == 3


def test_nothing_saved_until_complete(pipeline, tmp_path, clock):
    s = session(pipeline.database, tmp_path, clock)
    progress = feed_n(s, pipeline, make_frame((40, 40, GREEN)), clock, 2)
    assert not progress.done and progress.captured == 2
    assert len(pipeline.database) == 0
    assert not (tmp_path / "faces").exists()


def test_waits_between_samples(pipeline, tmp_path, clock):
    s = session(pipeline.database, tmp_path, clock)
    frame = make_frame((40, 40, GREEN))
    s.feed(frame, pipeline.process(frame))
    clock.t += 0.1  # too soon for the next sample
    assert s.feed(frame, pipeline.process(frame)).captured == 1


def test_requires_exactly_one_face(pipeline, tmp_path, clock):
    s = session(pipeline.database, tmp_path, clock)
    empty = s.feed(make_frame(), [])
    assert empty.captured == 0 and "No face" in empty.message
    two = make_frame((10, 10, GREEN), (200, 100, BLUE))
    progress = s.feed(two, pipeline.process(two))
    assert progress.captured == 0 and "one person" in progress.message


def test_feed_after_done_is_noop(pipeline, tmp_path, clock):
    s = session(pipeline.database, tmp_path, clock, samples=1)
    frame = make_frame((40, 40, GREEN))
    feed_n(s, pipeline, frame, clock, 3)
    assert pipeline.database.people() == {"Maria": 1}


def test_delete_face_photos(pipeline, tmp_path, clock):
    s = session(pipeline.database, tmp_path, clock, samples=1)
    feed_n(s, pipeline, make_frame((40, 40, GREEN)), clock, 1)
    delete_face_photos(tmp_path / "faces", "Maria")
    assert not (tmp_path / "faces" / "Maria").exists()
    delete_face_photos(tmp_path / "faces", "Nobody")  # no error
