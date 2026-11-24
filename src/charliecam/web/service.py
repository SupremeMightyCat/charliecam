"""Everything the web app does, independent of HTTP.

``WebService`` owns the camera loop, recognition, enrollment, the activity feed
and settings. The FastAPI layer in ``server.py`` only translates requests into
calls on this class, so the behaviour is tested here without a browser.

Threading: a background thread calls :meth:`step` in a loop; HTTP handlers run
on other threads. One lock guards the database, pipeline, enrollment session
and activity feed.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional

import cv2
import numpy as np

from ..attendance import report
from ..database import FaceDatabase
from ..enrollment import EnrollmentSession, delete_face_photos, face_folder, validate_name
from ..events import EventRecorder
from ..overlay import draw_faces
from ..pipeline import RecognitionPipeline

log = logging.getLogger("charliecam.web")

FrameSource = Callable[[], Optional[np.ndarray]]
ACTIVITY_SIZE = 50
THRESHOLD_RANGE = (0.2, 0.8)
COOLDOWN_RANGE = (0.0, 300.0)


class WebService:
    def __init__(
        self,
        database: FaceDatabase,
        recorder: EventRecorder,
        db_path: Path,
        faces_dir: Path,
        settings_path: Optional[Path] = None,
        samples: int = 5,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.database = database
        self.recorder = recorder
        self.db_path = Path(db_path)
        self.faces_dir = Path(faces_dir)
        self.settings_path = Path(settings_path) if settings_path else None
        self.samples = samples
        self.clock = clock
        self.pipeline: Optional[RecognitionPipeline] = None

        self.lock = threading.RLock()
        self.status, self.status_ok = "Starting…", False
        self.session: Optional[EnrollmentSession] = None
        self.last_result: Optional[Dict] = None  # outcome of the last enrollment
        self.activity: Deque[Dict] = deque(maxlen=ACTIVITY_SIZE)
        self._activity_id = 0
        self.fps = 0.0
        self._last_frame_t: Optional[float] = None

        self._jpeg: Optional[bytes] = None
        self.frame_seq = 0
        self._frame_ready = threading.Condition()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._load_settings()
        self._load_history()

    # ------------------------------------------------------------ lifecycle
    def set_pipeline(self, pipeline: RecognitionPipeline) -> None:
        with self.lock:
            pipeline.unknowns.threshold = self.database.threshold
            self.pipeline = pipeline
            self.set_status("Live", ok=True)

    def set_status(self, text: str, ok: bool = False) -> None:
        self.status, self.status_ok = text, ok

    def start(self, frame_source: FrameSource, error: Callable[[], Optional[str]] = lambda: None,
              max_fps: float = 30.0) -> None:
        """Run :meth:`step` on a background thread until :meth:`stop`."""

        def loop() -> None:
            period = 1.0 / max_fps
            while not self._stop.is_set():
                t0 = time.monotonic()
                msg = error()
                if msg:
                    self.set_status(msg)
                try:
                    self.step(frame_source())
                except Exception:  # keep serving the page even if one frame fails
                    log.exception("Frame processing failed")
                self._stop.wait(max(0.0, period - (time.monotonic() - t0)))

        self._thread = threading.Thread(target=loop, name="charliecam-web-loop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._frame_ready:
            self._frame_ready.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2)

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    # ---------------------------------------------------------------- frame
    def step(self, frame: Optional[np.ndarray]) -> None:
        """Process one camera frame: recognise, enroll or log, then publish a JPEG."""
        if frame is None:
            return
        with self.lock:
            faces = []
            if self.pipeline is not None:
                faces = self.pipeline.process(frame)
                if self.session is not None:
                    progress = self.session.feed(frame, faces)
                    if progress.done:
                        self._finish_enrollment()
                else:
                    for ev in self.recorder.record(frame, faces):
                        self._add_activity(ev)
                now = self.clock()
                if self._last_frame_t is not None:
                    dt = max(now - self._last_frame_t, 1e-6)
                    self.fps = 1 / dt if self.fps == 0 else 0.9 * self.fps + 0.1 / dt
                self._last_frame_t = now
        annotated = draw_faces(frame.copy(), faces)
        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self._frame_ready:
                self._jpeg = buf.tobytes()
                self.frame_seq += 1
                self._frame_ready.notify_all()

    def latest_jpeg(self) -> Optional[bytes]:
        return self._jpeg

    def wait_for_frame(self, after_seq: int, timeout: float = 1.0):
        """Block until a frame newer than ``after_seq`` exists; returns (seq, jpeg)."""
        with self._frame_ready:
            self._frame_ready.wait_for(lambda: self.frame_seq > after_seq or self.stopped, timeout)
            return self.frame_seq, self._jpeg

    # ------------------------------------------------------------- activity
    def _add_activity(self, ev: Dict) -> None:
        self._activity_id += 1
        snap = ev.get("snapshot")
        snap_name = Path(snap).name if snap else None
        if snap_name and not (self.recorder.snapshot_dir / snap_name).is_file():
            snap_name = None
        self.activity.appendleft({
            "id": self._activity_id,
            "timestamp": ev["timestamp"],
            "label": ev["label"],
            "known": bool(ev["known"]),
            "similarity": ev.get("similarity"),
            "snapshot": snap_name,
        })

    def _load_history(self) -> None:
        """Show the most recent logged sightings when the page first opens."""
        path = self.recorder.log_path
        if not path.is_file():
            return
        try:
            lines = path.read_text(encoding="utf-8").splitlines()[-ACTIVITY_SIZE:]
        except OSError:
            return
        for line in lines:
            try:
                ev = json.loads(line)
                datetime.fromisoformat(ev["timestamp"])
                self._add_activity(ev)
            except (ValueError, KeyError, TypeError):
                continue  # skip malformed lines rather than failing startup

    # ----------------------------------------------------------- enrollment
    def start_enrollment(self, name: str) -> str:
        """Begin capturing webcam samples. Raises ValueError with a user message."""
        with self.lock:
            if self.pipeline is None:
                raise ValueError("Face models are still loading")
            if self.session is not None:
                raise ValueError(f"Already capturing {self.session.name}")
            name = validate_name(name)
            self.session = EnrollmentSession(name, self.database, self.faces_dir,
                                             samples=self.samples, clock=self.clock)
            self.last_result = None
            return name

    def cancel_enrollment(self) -> bool:
        with self.lock:
            if self.session is None:
                return False
            self.session = None
            self.last_result = {"ok": False, "message": "Cancelled. Nothing was saved"}
            return True

    def _finish_enrollment(self) -> None:
        session = self.session
        self.session = None
        self.database.save(self.db_path)
        self.last_result = {"ok": True, "message": session.progress.message, "name": session.name}

    def enroll_photo(self, name: str, data: bytes) -> str:
        """Add one uploaded photo of ``name``. Raises ValueError with a user message."""
        name = validate_name(name)
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("That file isn't an image CharlieCam can read")
        with self.lock:
            if self.pipeline is None:
                raise ValueError("Face models are still loading")
            try:
                self.pipeline.enroll_from_image(name, image)
            except ValueError as exc:
                if "No face" in str(exc):
                    raise ValueError("No face found in that photo") from exc
                raise
            self.database.save(self.db_path)
        folder = face_folder(self.faces_dir, name)
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        cv2.imwrite(str(folder / f"upload_{stamp}.jpg"), image)
        return name

    # --------------------------------------------------------------- people
    def people(self) -> List[Dict]:
        with self.lock:
            counts = self.database.people()
        return [{"name": n, "samples": c, "has_photo": self.person_photo(n) is not None}
                for n, c in sorted(counts.items(), key=lambda kv: kv[0].lower())]

    def person_photo(self, name: str) -> Optional[Path]:
        folder = face_folder(self.faces_dir, name)
        if not folder.is_dir():
            return None
        photos = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        return photos[0] if photos else None

    def remove_person(self, name: str) -> int:
        with self.lock:
            removed = self.database.remove(name)
            if removed:
                self.database.save(self.db_path)
        if removed:
            delete_face_photos(self.faces_dir, name)
        return removed

    def snapshot_path(self, filename: str) -> Optional[Path]:
        """Resolve a snapshot file name, refusing anything outside the snapshot folder."""
        if not filename or Path(filename).name != filename:
            return None
        path = self.recorder.snapshot_dir / filename
        return path if path.is_file() else None

    # ----------------------------------------------------------- attendance
    def attendance(self, day: Optional[date] = None) -> Dict:
        """Who was seen on ``day`` (default today) and which enrolled people were not."""
        with self.lock:
            enrolled = list(self.database.people())
        return report(self.recorder.log_path, enrolled, day)

    # ------------------------------------------------------------- settings
    def settings(self) -> Dict:
        return {"threshold": round(self.database.threshold, 3),
                "cooldown": round(self.recorder.cooldown_s, 1)}

    def update_settings(self, threshold: Optional[float] = None,
                        cooldown: Optional[float] = None) -> Dict:
        with self.lock:
            if threshold is not None:
                lo, hi = THRESHOLD_RANGE
                if not lo <= threshold <= hi:
                    raise ValueError(f"Match threshold must be between {lo} and {hi}")
                self.database.threshold = threshold
                if self.pipeline is not None:
                    self.pipeline.unknowns.threshold = threshold
            if cooldown is not None:
                lo, hi = COOLDOWN_RANGE
                if not lo <= cooldown <= hi:
                    raise ValueError(f"Cooldown must be between {lo:g} and {hi:g} seconds")
                self.recorder.cooldown_s = cooldown
            self._save_settings()
            return self.settings()

    def _load_settings(self) -> None:
        if not (self.settings_path and self.settings_path.is_file()):
            return
        try:
            saved = json.loads(self.settings_path.read_text(encoding="utf-8"))
            self.update_settings(saved.get("threshold"), saved.get("cooldown"))
        except (ValueError, OSError, AttributeError):
            log.warning("Ignoring unreadable settings file %s", self.settings_path)

    def _save_settings(self) -> None:
        if not self.settings_path:
            return
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.settings_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.settings(), indent=2), encoding="utf-8")
        tmp.replace(self.settings_path)

    # ---------------------------------------------------------------- state
    def state(self, since: int = 0) -> Dict:
        """Everything the page needs on each poll. ``since`` trims the activity feed."""
        with self.lock:
            enrolling = None
            if self.session is not None:
                p = self.session.progress
                enrolling = {"name": self.session.name, "captured": p.captured,
                             "total": p.total, "message": p.message}
            return {
                "status": self.status,
                "status_ok": self.status_ok,
                "ready": self.pipeline is not None,
                "fps": round(self.fps, 1),
                "enrolling": enrolling,
                "last_result": self.last_result,
                "people_count": len(self.database.people()),
                "activity": [a for a in self.activity if a["id"] > since],
                "settings": self.settings(),
            }
