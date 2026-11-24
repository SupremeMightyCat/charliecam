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


# ------------------------------------------------- moving people (IdentityTracker)
class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


@pytest.fixture
def moving(db):
    from charliecam.pipeline import RecognitionPipeline
    from conftest import FakeDetector, FakeEmbedder

    clock = Clock()
    db.add("alice", RED)
    return RecognitionPipeline(FakeDetector(), FakeEmbedder(), db, clock=clock), clock


def test_blurry_frame_keeps_name_while_walking(moving):
    """A frame too blurred to match, overlapping alice's last box, stays alice."""
    pipe, clock = moving
    assert pipe.process(make_frame((20, 20, RED)))[0].track_label == "alice"
    clock.t = 0.1
    blurred = pipe.process(make_frame((30, 20, BLUE)))[0]  # moved 10 px, colour "blurred"
    assert blurred.track_label == "alice" and blurred.match.held
    clock.t = 0.2
    assert pipe.process(make_frame((40, 20, RED)))[0].match.held is False


def test_hold_expires_and_is_not_extended_by_held_frames(moving):
    pipe, clock = moving
    pipe.process(make_frame((20, 20, RED)))
    for t in (0.5, 0.9):
        clock.t = t
        assert pipe.process(make_frame((20, 20, BLUE)))[0].track_label == "alice"
    clock.t = 1.2  # 1.2 s after the last real match
    assert pipe.process(make_frame((20, 20, BLUE)))[0].track_label == "Unknown #1"


def test_stranger_elsewhere_in_frame_is_not_held(moving):
    pipe, clock = moving
    pipe.process(make_frame((20, 20, RED)))
    clock.t = 0.1
    labels_now = sorted(f.track_label for f in pipe.process(make_frame((20, 20, RED), (200, 120, BLUE))))
    assert labels_now == ["Unknown #1", "alice"]


def test_iou():
    from charliecam.pipeline import iou
    from charliecam.types import BoundingBox

    a = BoundingBox(0, 0, 10, 10)
    assert iou(a, a) == 1.0
    assert iou(a, BoundingBox(20, 20, 10, 10)) == 0.0
    assert iou(a, BoundingBox(5, 0, 10, 10)) == pytest.approx(50 / 150)
