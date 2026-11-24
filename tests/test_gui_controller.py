"""Drives the GUI exactly as a user would (typing, clicking) with no real window."""
import pytest

from conftest import BLUE, GREEN, make_frame
from flockcam.database import FaceDatabase
from flockcam.events import EventRecorder
from flockcam.gui.controller import GuiController
from flockcam.gui.theme import HEIGHT, WIDTH


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def app(pipeline, tmp_path, clock):
    rec = EventRecorder(tmp_path / "events.jsonl", tmp_path / "snaps", clock=clock)
    c = GuiController(pipeline.database, rec, tmp_path / "faces.npz", tmp_path / "faces",
                      samples=3, clock=clock)
    c.set_pipeline(pipeline)
    c.process_frame(None)  # first render establishes clickable regions
    return c


def type_text(app, text):
    for ch in text:
        app.on_key(ord(ch))


def click(app, region):
    x0, y0, x1, y1 = app.renderer.regions[region]
    app.on_click((x0 + x1) // 2, (y0 + y1) // 2)


def run_frames(app, frame, clock, n):
    for _ in range(n):
        clock.t += 1.0
        out = app.process_frame(frame)
    return out


def test_render_size(app):
    assert app.process_frame(make_frame()).shape == (HEIGHT, WIDTH, 3)


def test_capture_disabled_until_name_typed(app):
    assert "capture" not in app.renderer.regions
    type_text(app, "Maria")
    app.process_frame(None)
    assert "capture" in app.renderer.regions


def test_capture_disabled_while_models_load(pipeline, tmp_path, clock):
    c = GuiController(pipeline.database, EventRecorder(tmp_path / "e", tmp_path / "s"),
                      tmp_path / "db.npz", tmp_path / "f")
    type_text(c, "Maria")
    c.process_frame(None)
    assert "capture" not in c.renderer.regions


def test_typing_and_backspace(app):
    type_text(app, "Mariax")
    app.on_key(8)
    assert app.name_text == "Maria"
    app.on_key(-1)  # waitKey timeout
    assert app.name_text == "Maria"


def test_enroll_by_clicking_button(app, clock, tmp_path):
    type_text(app, "Maria")
    app.process_frame(None)
    click(app, "capture")
    assert app.session is not None
    run_frames(app, make_frame((40, 40, GREEN)), clock, 3)
    assert app.session is None
    assert "Saved 3 photos of Maria" in app.toast
    assert app.name_text == ""
    assert FaceDatabase.load(tmp_path / "faces.npz", 0.9).people() == {"Maria": 3}


def test_enter_key_starts_enrollment(app):
    type_text(app, "Maria")
    app.on_key(13)
    assert app.session is not None and app.session.name == "Maria"


def test_recognises_after_enrolling(app, clock):
    type_text(app, "Maria")
    app.on_key(13)
    run_frames(app, make_frame((40, 40, GREEN)), clock, 3)
    run_frames(app, make_frame((100, 80, GREEN)), clock, 1)
    assert app.last_faces[0].track_label == "Maria"
    assert app.activity[0].label == "Maria" and app.activity[0].known


def test_escape_cancels_without_saving(app, clock, tmp_path):
    type_text(app, "Maria")
    app.on_key(13)
    run_frames(app, make_frame((40, 40, GREEN)), clock, 2)
    app.on_key(27)
    assert app.session is None
    assert len(app.database) == 0
    assert not (tmp_path / "faces.npz").exists()


def test_invalid_name_shows_error(app):
    type_text(app, "Unknown")
    app.on_key(13)
    assert app.session is None
    assert "reserved" in app.name_error


def test_typing_locked_during_capture(app):
    type_text(app, "Maria")
    app.on_key(13)
    type_text(app, "zzz")
    assert app.name_text == "Maria"


def test_no_alerts_logged_while_enrolling(app, clock):
    type_text(app, "Maria")
    app.on_key(13)
    run_frames(app, make_frame((40, 40, GREEN)), clock, 1)
    assert list(app.activity) == []


def test_unknown_face_appears_in_activity(app, clock):
    run_frames(app, make_frame((40, 40, BLUE)), clock, 1)
    assert app.activity[0].label == "Unknown #1" and not app.activity[0].known


def test_remove_needs_two_clicks(app, clock, tmp_path):
    type_text(app, "Maria")
    app.on_key(13)
    run_frames(app, make_frame((40, 40, GREEN)), clock, 3)
    assert (tmp_path / "faces" / "Maria").exists()
    click(app, "remove:Maria")
    assert "Maria" in app.database  # first click only asks to confirm
    app.process_frame(None)
    click(app, "remove:Maria")
    assert "Maria" not in app.database
    assert not (tmp_path / "faces" / "Maria").exists()
    assert FaceDatabase.load(tmp_path / "faces.npz", 0.9).people() == {}


def test_remove_confirmation_expires(app, clock):
    app.database.add("Bob", GREEN)
    app.process_frame(None)
    click(app, "remove:Bob")
    clock.t += 5
    app.process_frame(None)
    assert app.confirm_remove is None
    click(app, "remove:Bob")
    assert "Bob" in app.database


def test_hover_tracks_buttons(app):
    type_text(app, "Maria")
    app.process_frame(None)
    x0, y0, x1, y1 = app.renderer.regions["capture"]
    app.on_mouse_move(x0 + 5, y0 + 5)
    assert app.hover == "capture"
    app.on_mouse_move(1, 1)
    assert app.hover is None


def test_many_people_render_without_overflow(app):
    for i in range(30):
        app.database.add(f"Person {i:02d}", GREEN)
    img = app.process_frame(make_frame())
    assert img.shape == (HEIGHT, WIDTH, 3)
    removes = [k for k in app.renderer.regions if k.startswith("remove:")]
    assert 0 < len(removes) < 30
    assert all(app.renderer.regions[k][3] < HEIGHT for k in removes)
