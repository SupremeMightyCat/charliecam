import numpy as np

from conftest import BLUE, RED, make_frame
from flockcam.config import Config, parse_source
from flockcam.overlay import GREEN as BOX_GREEN, RED as BOX_RED, draw_faces
from flockcam.types import BoundingBox


def test_known_face_gets_green_box_unknown_red(pipeline):
    pipeline.database.add("alice", RED)
    frame = make_frame((20, 100, RED), (200, 100, BLUE))
    faces = pipeline.process(frame)
    out = draw_faces(np.zeros_like(frame), faces)
    by_label = {f.track_label: f.detection.box for f in faces}
    a, u = by_label["alice"], by_label["Unknown #1"]
    assert tuple(out[a.y + a.h, a.x + a.w // 2]) == BOX_GREEN  # bottom edge
    assert tuple(out[u.y + u.h, u.x + u.w // 2]) == BOX_RED


def test_parse_source():
    assert parse_source("0") == 0
    assert parse_source("2") == 2
    assert parse_source("rtsp://cam/stream") == "rtsp://cam/stream"
    assert parse_source("clip.mp4") == "clip.mp4"


def test_config_paths(tmp_path):
    cfg = Config(data_dir=tmp_path)
    assert cfg.database_path.parent == tmp_path
    assert cfg.snapshots_dir.parent == tmp_path


def test_bounding_box_clip():
    assert BoundingBox(-5, -5, 20, 20).clip(100, 100) == BoundingBox(0, 0, 15, 15)
    assert BoundingBox(90, 90, 20, 20).clip(100, 100) == BoundingBox(90, 90, 10, 10)
