"""Window and main loop for the desktop app.

Run with `charliecam gui`, the `charliecam-gui` launcher, or by double-clicking
CharlieCam.pyw in the project folder.
"""
from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path
from typing import Union

import cv2

from ..camera import CameraThread
from ..config import Config
from ..database import FaceDatabase
from ..events import EventRecorder
from ..models import load_pipeline_in_background
from .controller import GuiController
from .theme import HEIGHT, WIDTH

WINDOW = "CharlieCam"
log = logging.getLogger("charliecam.gui")


def run(cfg: Config, source: Union[int, str] = 0) -> None:
    db = FaceDatabase.load(cfg.database_path, cfg.match_threshold)
    recorder = EventRecorder(cfg.events_path, cfg.snapshots_dir, cooldown_s=cfg.event_cooldown_s)
    controller = GuiController(db, recorder, cfg.database_path, cfg.data_dir / "faces")
    controller.set_status("Loading face models…")
    load_pipeline_in_background(cfg, controller)
    camera = CameraThread(source)
    camera.start()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(WINDOW, int(WIDTH * 0.9), int(HEIGHT * 0.9))

    def on_mouse(event, x, y, _flags, _param):  # OpenCV gives image coordinates
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

    p = argparse.ArgumentParser(prog="charliecam-gui")
    p.add_argument("--source", default="0")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--models-dir", type=Path, default=Path("models"))
    p.add_argument("--threshold", type=float, default=DEFAULT_MATCH_THRESHOLD)
    args = p.parse_args(argv)
    cfg = Config(data_dir=args.data_dir, models_dir=args.models_dir, match_threshold=args.threshold)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=cfg.data_dir / "charliecam.log", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        run(cfg, parse_source(args.source))
    except Exception:
        log.error("Fatal error\n%s", traceback.format_exc())
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
