"""Video capture plus the live recognition loop."""
from __future__ import annotations

import sys
import time
from typing import Optional, Union

import cv2

from .events import EventRecorder
from .overlay import draw_faces, draw_status
from .pipeline import RecognitionPipeline

WINDOW = "FlockCam - q: quit | e: enroll largest face"


def open_capture(source: Union[int, str]) -> cv2.VideoCapture:
    # DirectShow opens Windows webcams far faster than the default MSMF backend.
    if isinstance(source, int) and sys.platform.startswith("win"):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source {source!r}")
    return cap


def capture_frame(source: Union[int, str], warmup_frames: int = 10):
    """Grab one frame (lets the camera auto-expose first)."""
    cap = open_capture(source)
    try:
        frame = None
        for _ in range(warmup_frames):
            ok, frame = cap.read()
        if frame is None:
            raise RuntimeError("Camera returned no frames")
        return frame
    finally:
        cap.release()


def run_live(
    pipeline: RecognitionPipeline,
    recorder: EventRecorder,
    source: Union[int, str],
    process_every_n_frames: int = 1,
    show_window: bool = True,
    max_frames: Optional[int] = None,
    on_enroll=None,
) -> int:
    """Run until 'q'/ESC (or ``max_frames``). Returns frames processed."""
    cap = open_capture(source)
    frames, faces = 0, []
    fps, t_last = 0.0, time.perf_counter()
    try:
        while max_frames is None or frames < max_frames:
            ok, frame = cap.read()
            if not ok:
                print("Video source ended or disconnected.")
                break
            frames += 1
            if frames % process_every_n_frames == 0:
                faces = pipeline.process(frame)
                for ev in recorder.record(frame, faces):
                    tag = "KNOWN  " if ev["known"] else "UNKNOWN"
                    print(f"[{ev['timestamp']}] {tag} {ev['label']} sim={ev['similarity']}")
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_last, 1e-6))
            t_last = now
            if show_window:
                display = draw_faces(frame.copy(), faces)
                draw_status(display, f"{fps:4.1f} FPS | {len(pipeline.database.people())} people enrolled")
                cv2.imshow(WINDOW, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("e") and on_enroll is not None:
                    on_enroll(frame)
    finally:
        cap.release()
        if show_window:
            cv2.destroyAllWindows()
    return frames
