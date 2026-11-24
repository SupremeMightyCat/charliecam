"""Web app: the service logic and the HTTP API, with fake faces and no camera."""
import json

import cv2
import pytest
from fastapi.testclient import TestClient

from charliecam.database import FaceDatabase
from charliecam.events import EventRecorder
from charliecam.web.server import create_app
from charliecam.web.service import WebService
from conftest import BLUE, GREEN, RED, make_frame

H = {"X-CharlieCam": "1"}


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def service(pipeline, tmp_path, clock):
    rec = EventRecorder(tmp_path / "events.jsonl", tmp_path / "snaps", clock=clock)
    svc = WebService(pipeline.database, rec, tmp_path / "faces.npz", tmp_path / "faces",
                     settings_path=tmp_path / "web_settings.json", samples=3, clock=clock)
    svc.set_pipeline(pipeline)
    return svc


@pytest.fixture
def quits():
    return []


@pytest.fixture
def client(service, quits):
    return TestClient(create_app(service, on_quit=lambda: quits.append(True)))


def run_frames(service, frame, clock, n):
    for _ in range(n):
        clock.t += 1.0
        service.step(frame)


def png(frame):
    return cv2.imencode(".png", frame)[1].tobytes()


# ------------------------------------------------------------------ service
def test_step_publishes_annotated_jpeg(service):
    assert service.latest_jpeg() is None
    service.step(make_frame((20, 20, RED)))
    jpeg = service.latest_jpeg()
    assert jpeg[:2] == b"\xff\xd8" and service.frame_seq == 1


def test_camera_enrollment_saves_person_and_photos(service, clock, tmp_path):
    service.start_enrollment("  Maria   Lopez ")
    assert service.state()["enrolling"]["name"] == "Maria Lopez"
    run_frames(service, make_frame((20, 20, GREEN)), clock, 3)
    s = service.state()
    assert s["enrolling"] is None and s["last_result"]["ok"]
    assert service.database.people() == {"Maria Lopez": 3}
    assert FaceDatabase.load(tmp_path / "faces.npz", 0.9).people() == {"Maria Lopez": 3}
    assert service.people()[0] == {"name": "Maria Lopez", "samples": 3, "has_photo": True}


def test_enrollment_waits_for_exactly_one_face(service, clock):
    service.start_enrollment("Ann")
    run_frames(service, make_frame((10, 10, GREEN), (150, 10, BLUE)), clock, 3)
    assert service.state()["enrolling"]["message"] == "Only one person should be in view"
    assert len(service.database) == 0


def test_cancel_enrollment_saves_nothing(service, clock):
    service.start_enrollment("Ann")
    run_frames(service, make_frame((20, 20, GREEN)), clock, 2)
    assert service.cancel_enrollment()
    assert len(service.database) == 0 and service.state()["last_result"]["ok"] is False


def test_no_activity_logged_while_enrolling(service, clock):
    service.start_enrollment("Ann")
    run_frames(service, make_frame((20, 20, GREEN)), clock, 1)
    assert service.state()["activity"] == []


def test_activity_feed_and_since_filter(service, clock):
    service.database.add("Ann", (0, 255, 0))
    run_frames(service, make_frame((20, 20, GREEN), (150, 20, RED)), clock, 1)
    items = service.state()["activity"]
    assert {(a["label"], a["known"]) for a in items} == {("Ann", True), ("Unknown #1", False)}
    unknown = next(a for a in items if not a["known"])
    assert service.snapshot_path(unknown["snapshot"]) is not None
    assert service.state(since=max(a["id"] for a in items))["activity"] == []


def test_history_loaded_from_event_log(pipeline, tmp_path):
    log = tmp_path / "events.jsonl"
    log.write_text("\n".join([
        json.dumps({"timestamp": "2026-01-01T10:00:00+00:00", "label": "Ann", "known": True,
                    "similarity": 0.9, "snapshot": None}),
        "not json",
        json.dumps({"timestamp": "2026-01-01T10:01:00+00:00", "label": "Unknown #4", "known": False,
                    "similarity": 0.1, "snapshot": "data/snapshots/missing.jpg"}),
    ]))
    svc = WebService(pipeline.database, EventRecorder(log, tmp_path / "snaps"),
                     tmp_path / "faces.npz", tmp_path / "faces")
    items = svc.state()["activity"]
    assert [a["label"] for a in items] == ["Unknown #4", "Ann"]
    assert items[0]["snapshot"] is None  # file no longer exists


def test_upload_photo_enrolls_and_keeps_copy(service, tmp_path):
    service.enroll_photo("Ben", png(make_frame((20, 20, BLUE))))
    assert service.database.people() == {"Ben": 1}
    assert service.person_photo("Ben") is not None


@pytest.mark.parametrize("data, message", [
    (b"not an image", "isn't an image"),
    (None, "No face found"),
])
def test_upload_photo_errors(service, data, message):
    with pytest.raises(ValueError, match=message):
        service.enroll_photo("Ben", data if data else png(make_frame()))
    assert len(service.database) == 0


def test_remove_person_deletes_photos(service):
    service.enroll_photo("Ben", png(make_frame((20, 20, BLUE))))
    assert service.remove_person("Ben") == 1
    assert service.people() == [] and service.person_photo("Ben") is None
    assert service.remove_person("Ben") == 0


def test_settings_validate_and_persist(service, pipeline, tmp_path):
    assert service.update_settings(threshold=0.5, cooldown=30) == {"threshold": 0.5, "cooldown": 30.0}
    assert pipeline.database.threshold == 0.5 and pipeline.unknowns.threshold == 0.5
    with pytest.raises(ValueError):
        service.update_settings(threshold=0.95)
    reloaded = WebService(FaceDatabase(0.363), EventRecorder(tmp_path / "e.jsonl", tmp_path / "s"),
                          tmp_path / "f.npz", tmp_path / "faces",
                          settings_path=tmp_path / "web_settings.json")
    assert reloaded.settings() == {"threshold": 0.5, "cooldown": 30.0}


def test_enrollment_refused_before_models_load(tmp_path):
    svc = WebService(FaceDatabase(0.9), EventRecorder(tmp_path / "e.jsonl", tmp_path / "s"),
                     tmp_path / "f.npz", tmp_path / "faces")
    with pytest.raises(ValueError, match="still loading"):
        svc.start_enrollment("Ann")
    assert svc.state()["ready"] is False


def test_snapshot_path_rejects_traversal(service, tmp_path):
    (tmp_path / "secret.jpg").write_bytes(b"x")
    assert service.snapshot_path("../secret.jpg") is None
    assert service.snapshot_path("..\\secret.jpg") is None


def test_background_loop_processes_frames(service):
    frame = make_frame((20, 20, RED))
    service.start(lambda: frame, max_fps=200)
    try:
        seq, jpeg = service.wait_for_frame(0, timeout=2)
        assert seq >= 1 and jpeg
    finally:
        service.stop()
    assert service.stopped


# ------------------------------------------------------------------ HTTP
def test_page_and_static_files(client):
    page = client.get("/")
    assert page.status_code == 200 and "CharlieCam" in page.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


def test_frame_endpoint(client, service):
    assert client.get("/api/frame.jpg").status_code == 503
    service.step(make_frame((20, 20, RED)))
    r = client.get("/api/frame.jpg")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"


def test_enroll_over_http(client, service, clock):
    r = client.post("/api/enroll", json={"name": "Cara"}, headers=H)
    assert r.json() == {"name": "Cara"}
    assert client.post("/api/enroll", json={"name": "Dan"}, headers=H).status_code == 400
    run_frames(service, make_frame((20, 20, GREEN)), clock, 3)
    state = client.get("/api/state").json()
    assert state["people_count"] == 1 and state["last_result"]["ok"]
    people = client.get("/api/people").json()
    assert people == [{"name": "Cara", "samples": 3, "has_photo": True}]
    assert client.get("/api/people/Cara/photo").status_code == 200


def test_invalid_name_returns_message(client):
    r = client.post("/api/enroll", json={"name": "   "}, headers=H)
    assert r.status_code == 400 and r.json()["detail"] == "Please enter a name"
    r = client.post("/api/enroll", json={"name": "Unknown #2"}, headers=H)
    assert r.status_code == 400 and "reserved" in r.json()["detail"]


def test_cancel_over_http(client):
    client.post("/api/enroll", json={"name": "Cara"}, headers=H)
    assert client.post("/api/enroll/cancel", headers=H).json() == {"cancelled": True}
    assert client.post("/api/enroll/cancel", headers=H).json() == {"cancelled": False}


def test_upload_and_remove_over_http(client):
    r = client.post("/api/people/Eve%20Stone/photos", content=png(make_frame((20, 20, BLUE))), headers=H)
    assert r.status_code == 200 and r.json() == {"name": "Eve Stone", "samples": 1}
    bad = client.post("/api/people/Eve/photos", content=png(make_frame()), headers=H)
    assert bad.status_code == 400 and bad.json()["detail"] == "No face found in that photo"
    assert client.delete("/api/people/Eve%20Stone", headers=H).status_code == 200
    assert client.delete("/api/people/Eve%20Stone", headers=H).status_code == 404
    assert client.get("/api/people/Eve%20Stone/photo").status_code == 404


def test_snapshot_over_http(client, service, clock):
    run_frames(service, make_frame((20, 20, RED)), clock, 1)
    snap = client.get("/api/state").json()["activity"][0]["snapshot"]
    assert client.get(f"/api/snapshots/{snap}").status_code == 200
    assert client.get("/api/snapshots/nope.jpg").status_code == 404


def test_settings_over_http(client):
    assert client.put("/api/settings", json={"cooldown": 5}, headers=H).json()["cooldown"] == 5.0
    r = client.put("/api/settings", json={"threshold": 2}, headers=H)
    assert r.status_code == 400 and "between" in r.json()["detail"]
    assert client.get("/api/settings").json()["cooldown"] == 5.0


def test_changes_require_app_header(client, service):
    """Other websites open in the browser must not be able to change anything."""
    service.enroll_photo("Ann", png(make_frame((20, 20, GREEN))))
    assert client.delete("/api/people/Ann").status_code == 403
    assert client.post("/api/enroll", json={"name": "X"}).status_code == 403
    assert client.post("/api/quit").status_code == 403
    assert service.database.people() == {"Ann": 1}


def test_quit(client, quits):
    assert client.post("/api/quit", headers=H).json() == {"quitting": True}
    assert quits == [True]


def test_stream_yields_mjpeg_parts(service):
    """Drive the stream generator directly: TestClient can't end an endless response."""
    import asyncio

    from charliecam.web import server

    service.step(make_frame((20, 20, RED)))
    app = create_app(service)
    route = next(r for r in app.routes if getattr(r, "path", "") == "/api/stream")

    class FakeRequest:
        async def is_disconnected(self):
            return False

    async def first_chunk():
        resp = await route.endpoint(FakeRequest())
        assert resp.media_type.startswith("multipart/x-mixed-replace")
        chunk = await resp.body_iterator.__anext__()
        service.stop()
        return chunk

    chunk = asyncio.run(first_chunk())
    assert chunk.startswith(f"--{server.BOUNDARY}".encode()) and b"\xff\xd8" in chunk


def test_attendance_over_http(client, service, clock):
    service.database.add("Ann", (0, 255, 0))
    service.database.add("Ben", (255, 0, 0))
    run_frames(service, make_frame((20, 20, GREEN)), clock, 1)
    rep = client.get("/api/attendance").json()
    assert [a["name"] for a in rep["present"]] == ["Ann"] and rep["absent"] == ["Ben"]
    csv_resp = client.get(f"/api/attendance.csv?date={rep['date']}")
    assert csv_resp.status_code == 200 and "attachment" in csv_resp.headers["content-disposition"]
    assert "Ann,present" in csv_resp.text and "Ben,absent" in csv_resp.text
    assert client.get("/api/attendance?date=yesterday").status_code == 400
