import pytest

from conftest import BLUE, GREEN, RED, make_frame


def labels(faces):
    return sorted(f.track_label for f in faces)


def test_no_faces(pipeline):
    assert pipeline.process(make_frame()) == []


def test_known_face_recognised(pipeline):
    pipeline.database.add("alice", RED)
    faces = pipeline.process(make_frame((20, 20, RED)))
    assert len(faces) == 1
    assert faces[0].match.name == "alice"
    assert faces[0].track_label == "alice"


def test_unknown_face_flagged(pipeline):
    pipeline.database.add("alice", RED)
    faces = pipeline.process(make_frame((20, 20, BLUE)))
    assert faces[0].match.is_known is False
    assert faces[0].track_label == "Unknown #1"


def test_known_and_unknown_in_same_frame(pipeline):
    pipeline.database.add("alice", RED)
    faces = pipeline.process(make_frame((20, 20, RED), (200, 100, GREEN)))
    assert labels(faces) == ["Unknown #1", "alice"]


def test_same_stranger_keeps_stable_id(pipeline):
    first = pipeline.process(make_frame((20, 20, BLUE)))
    moved = pipeline.process(make_frame((150, 90, BLUE)))
    assert first[0].track_label == moved[0].track_label == "Unknown #1"


def test_different_strangers_get_different_ids(pipeline):
    pipeline.process(make_frame((20, 20, BLUE)))
    faces = pipeline.process(make_frame((20, 20, GREEN)))
    assert faces[0].track_label == "Unknown #2"


def test_enroll_then_recognise(pipeline):
    pipeline.enroll_from_image("bob", make_frame((20, 20, GREEN)))
    faces = pipeline.process(make_frame((100, 100, GREEN)))
    assert faces[0].track_label == "bob"


def test_enroll_uses_largest_face(pipeline):
    img = make_frame((150, 100, BLUE))  # 60x60 face
    img[10:30, 10:30] = RED  # 20x20 face
    pipeline.enroll_from_image("carol", img)
    assert pipeline.process(make_frame((20, 20, RED)))[0].track_label != "carol"
    assert pipeline.process(make_frame((20, 20, BLUE)))[0].track_label == "carol"


def test_enroll_without_face_raises(pipeline):
    with pytest.raises(ValueError, match="No face"):
        pipeline.enroll_from_image("dave", make_frame())
