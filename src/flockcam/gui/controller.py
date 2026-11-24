"""GUI state and input handling, independent of any real window or camera."""
from __future__ import annotations

import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable, Deque, List, Optional

import numpy as np

from ..database import FaceDatabase
from ..enrollment import EnrollmentSession, delete_face_photos, validate_name
from ..events import EventRecorder
from ..pipeline import RecognitionPipeline
from ..types import RecognizedFace
from .view import ActivityItem, Renderer, ViewModel

KEY_BACKSPACE = {8, 127}
KEY_ENTER = {10, 13}
KEY_ESC = 27


class GuiController:
    def __init__(
        self,
        database: FaceDatabase,
        recorder: EventRecorder,
        db_path: Path,
        faces_dir: Path,
        samples: int = 5,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.database = database
        self.recorder = recorder
        self.db_path = Path(db_path)
        self.faces_dir = Path(faces_dir)
        self.samples = samples
        self.clock = clock
        self.pipeline: Optional[RecognitionPipeline] = None
        self.renderer = Renderer()

        self.status, self.status_ok = "Starting…", False
        self.name_text = ""
        self.name_error = ""
        self.session: Optional[EnrollmentSession] = None
        self.activity: Deque[ActivityItem] = deque(maxlen=8)
        self.hover: Optional[str] = None
        self.confirm_remove: Optional[str] = None
        self._confirm_until = 0.0
        self.toast = ""
        self.toast_ok = True
        self._toast_until = 0.0
        self.fps = 0.0
        self._last_frame_t: Optional[float] = None
        self.last_faces: List[RecognizedFace] = []

    # ------------------------------------------------------------ lifecycle
    def set_pipeline(self, pipeline: RecognitionPipeline) -> None:
        self.pipeline = pipeline
        self.set_status("Live", ok=True)

    def set_status(self, text: str, ok: bool = False) -> None:
        self.status, self.status_ok = text, ok

    def show_toast(self, text: str, ok: bool = True, seconds: float = 3.0) -> None:
        self.toast, self.toast_ok, self._toast_until = text, ok, self.clock() + seconds

    # ---------------------------------------------------------------- frame
    def process_frame(self, frame: Optional[np.ndarray]) -> np.ndarray:
        now = self.clock()
        faces: List[RecognizedFace] = []
        if frame is not None and self.pipeline is not None:
            faces = self.pipeline.process(frame)
            if self.session is not None:
                progress = self.session.feed(frame, faces)
                if progress.done:
                    self._finish_enrollment()
            else:
                for ev in self.recorder.record(frame, faces):
                    stamp = datetime.fromisoformat(ev["timestamp"]).astimezone()
                    self.activity.appendleft(ActivityItem(stamp.strftime("%H:%M:%S"), ev["label"], ev["known"]))
            if self._last_frame_t is not None:
                dt = max(now - self._last_frame_t, 1e-6)
                self.fps = 1 / dt if self.fps == 0 else 0.9 * self.fps + 0.1 / dt
            self._last_frame_t = now
        self.last_faces = faces
        if self.toast and now > self._toast_until:
            self.toast = ""
        if self.confirm_remove and now > self._confirm_until:
            self.confirm_remove = None
        return self.renderer.render(self.view_model(frame, faces))

    def view_model(self, frame, faces) -> ViewModel:
        return ViewModel(
            frame=frame, faces=faces, status=self.status, status_ok=self.status_ok, fps=self.fps,
            name_text=self.name_text, caret_visible=int(self.clock() * 2) % 2 == 0,
            name_error=self.name_error,
            enrolling_name=self.session.name if self.session else None,
            progress=self.session.progress if self.session else None,
            people=self.database.people(), confirm_remove=self.confirm_remove,
            activity=list(self.activity), toast=self.toast, toast_ok=self.toast_ok, hover=self.hover,
            can_capture=bool(self.name_text.strip()) and self.pipeline is not None,
        )

    # ---------------------------------------------------------------- input
    def on_key(self, code: int) -> None:
        if code < 0:
            return
        if code == KEY_ESC:
            if self.session:
                self.cancel_enrollment()
            else:
                self.name_text = ""
            return
        if self.session is not None:
            return  # typing is locked while capturing
        if code in KEY_ENTER:
            self.start_enrollment()
        elif code in KEY_BACKSPACE:
            self.name_text = self.name_text[:-1]
            self.name_error = ""
        elif 32 <= code <= 0xFFFF and chr(code).isprintable() and len(self.name_text) < 40:
            self.name_text += chr(code)
            self.name_error = ""

    def on_mouse_move(self, x: int, y: int) -> None:
        self.hover = self.renderer.hit(x, y)

    def on_click(self, x: int, y: int) -> None:
        action = self.renderer.hit(x, y)
        if action == "capture":
            self.start_enrollment()
        elif action == "cancel":
            self.cancel_enrollment()
        elif action and action.startswith("remove:"):
            self._remove_clicked(action.split(":", 1)[1])

    # -------------------------------------------------------------- actions
    def start_enrollment(self) -> None:
        if self.session is not None or self.pipeline is None:
            return
        try:
            name = validate_name(self.name_text)
        except ValueError as exc:
            self.name_error = str(exc)
            return
        self.name_error = ""
        self.session = EnrollmentSession(name, self.database, self.faces_dir,
                                         samples=self.samples, clock=self.clock)

    def cancel_enrollment(self) -> None:
        if self.session is not None:
            self.session = None
            self.show_toast("Cancelled. Nothing was saved", ok=False)

    def _finish_enrollment(self) -> None:
        session = self.session
        self.session = None
        self.database.save(self.db_path)
        self.name_text = ""
        self.show_toast(session.progress.message)

    def _remove_clicked(self, name: str) -> None:
        if self.confirm_remove == name:
            self.database.remove(name)
            self.database.save(self.db_path)
            delete_face_photos(self.faces_dir, name)
            self.confirm_remove = None
            self.show_toast(f"Removed {name}")
        else:
            self.confirm_remove = name
            self._confirm_until = self.clock() + 3.0
