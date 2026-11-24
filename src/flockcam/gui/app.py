"""Window, camera thread and main loop for the desktop app.

Run with `flockcam gui`, the `flockcam-gui` launcher, or by double-clicking
FlockCam.pyw in the project folder.
"""
from __future__ import annotations

import logging
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from ..config import Config
from ..database import FaceDatabase
from ..events import EventRecorder
from .controller import GuiController
from .theme import HEIGHT, WIDTH

WINDOW = "FlockCam"
log = logging.getLogger("flockcam.gui")


class CameraThread(threading.Thread):
    """Reads frames continuously so the UI never blocks on the camera."""

    def __init__(self, source: Union[int, str]) -> None:
        super().__init__(daemon=True)
        self.source = source
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._stop = threading.Event()
        self.error: Optional[str] = None

    def run(self) -> None:
        from ..camera import open_capture

        try:
            cap = open_capture(self.source)
        except RuntimeError as exc:
            self.error = "Camera not available. Is another app using it?"
            log.error("%s", exc)
            return
        # Video files would otherwise be read as fast as the disk allows;
        # play them back at their real frame rate instead.
        is_file = isinstance(self.source, str) and "://" not in self.source
        delay = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 30) if is_file else 0.0
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    self.error = "Video ended" if is_file else "Camera disconnected"
                    break
                if delay:
                    time.sleep(delay)
                with self._lock:
                    self._frame = frame
        finally:
            cap.release()

    def latest(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self) -> None:
        self._stop.set()


def _load_pipeline(cfg: Config, controller: GuiController) -> None:
    """Background thread: fetch models if missing, then build the pipeline."""
    try:
        if not (cfg.detector_model.exists() and cfg.recognizer_model.exists()):
            controller.set_status("Downloading face models (first run)…")
            from ..models import download_models

            download_models(cfg.models_dir)
        from ..engines import SFaceEmbedder, YuNetDetector
        from ..pipeline import RecognitionPipeline

        detector = YuNetDetector(cfg.detector_model, cfg.detection_confidence, cfg.min_face_size)
        controller.set_pipeline(RecognitionPipeline(detector, SFaceEmbedder(cfg.recognizer_model),
                                                    controller.database))
    except Exception as exc:  # shown in the window instead of a console
        log.exception("Model loading failed")
        controller.set_status(f"Could not load models: {exc}")


def run(cfg: Config, source: Union[int, str] = 0) -> None:
    db = FaceDatabase.load(cfg.database_path, cfg.match_threshold)
    recorder = EventRecorder(cfg.events_path, cfg.snapshots_dir, cooldown_s=cfg.event_cooldown_s)
    controller = GuiController(db, recorder, cfg.database_path, cfg.data_dir / "faces")
    controller.set_status("Loading face models…")
    threading.Thread(target=_load_pipeline, args=(cfg, controller), daemon=True).start()
    camera = CameraThread(source)
    camera.start()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(WINDOW, int(WIDTH * 0.9), int(HEIGHT * 0.9))

    def on_mouse(event, x, y, flags, param):  # OpenCV gives image coordinates
        if event == cv2.EVENT_MOUSEMOVE:
            controller.on_mouse_move(x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            controller.on_click(x, y)

    cv2.setMouseCallback(WINDOW, on_mouse)
    try:
        while True:
            if camera.error:
                controller.set_status(camera.error)
            canvas = controller.process_frame(camera.latest())
            cv2.imshow(WINDOW, canvas)
            controller.on_key(cv2.waitKeyEx(15))
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break  # user closed the window
    finally:
        camera.stop()
        cv2.destroyAllWindows()


def main(argv=None) -> int:
    import argparse

    from ..config import DEFAULT_MATCH_THRESHOLD, parse_source

    p = argparse.ArgumentParser(prog="flockcam-gui")
    p.add_argument("--source", default="0")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--models-dir", type=Path, default=Path("models"))
    p.add_argument("--threshold", type=float, default=DEFAULT_MATCH_THRESHOLD)
    args = p.parse_args(argv)
    cfg = Config(data_dir=args.data_dir, models_dir=args.models_dir, match_threshold=args.threshold)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=cfg.data_dir / "flockcam.log", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        run(cfg, parse_source(args.source))
    except Exception:
        log.error("Fatal error\n%s", traceback.format_exc())
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
