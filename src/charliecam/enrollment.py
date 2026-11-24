"""Guided enrollment: capture N good samples of one person, then commit atomically.

Pure logic (no UI, no camera) so it can be driven by the GUI, the CLI or tests.
Nothing is written until every sample is captured, so a cancelled or
interrupted enrollment never leaves a half-enrolled person behind.
"""
from __future__ import annotations

import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .database import FaceDatabase
from .types import UNKNOWN, RecognizedFace

MAX_NAME_LENGTH = 40


def validate_name(name: str) -> str:
    """Return the cleaned name or raise ValueError with a user-facing message."""
    name = " ".join(name.split())
    if not name:
        raise ValueError("Please enter a name")
    if len(name) > MAX_NAME_LENGTH:
        raise ValueError(f"Name must be at most {MAX_NAME_LENGTH} characters")
    if name.lower() == UNKNOWN.lower() or name.lower().startswith("unknown #"):
        raise ValueError(f"'{name}' is reserved")
    return name


def face_folder(faces_dir: Path, name: str) -> Path:
    """Filesystem-safe folder for a person's photos."""
    safe = re.sub(r"[^\w\- ]", "", name, flags=re.UNICODE).strip().replace(" ", "_")
    return Path(faces_dir) / (safe or "person")


def delete_face_photos(faces_dir: Path, name: str) -> None:
    folder = face_folder(faces_dir, name)
    if folder.is_dir():
        shutil.rmtree(folder)


def crop_face(frame: np.ndarray, face: RecognizedFace, pad_ratio: float = 0.3) -> np.ndarray:
    h, w = frame.shape[:2]
    b = face.detection.box
    pad = int(pad_ratio * max(b.w, b.h))
    return frame[max(0, b.y - pad):min(h, b.y + b.h + pad),
                 max(0, b.x - pad):min(w, b.x + b.w + pad)].copy()


@dataclass
class EnrollProgress:
    captured: int
    total: int
    message: str
    done: bool = False

    @property
    def fraction(self) -> float:
        return self.captured / self.total if self.total else 0.0


class EnrollmentSession:
    def __init__(
        self,
        name: str,
        database: FaceDatabase,
        faces_dir: Path,
        samples: int = 5,
        interval_s: float = 0.7,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if samples < 1:
            raise ValueError("samples must be >= 1")
        self.name = validate_name(name)
        self.database = database
        self.faces_dir = Path(faces_dir)
        self.samples = samples
        self.interval_s = interval_s
        self._clock = clock
        self._last_capture: Optional[float] = None
        self._captured: List[Tuple[np.ndarray, np.ndarray]] = []  # (embedding, crop)
        self.progress = EnrollProgress(0, samples, "Look at the camera")
        self.saved_files: List[Path] = []

    @property
    def done(self) -> bool:
        return self.progress.done

    def feed(self, frame: np.ndarray, faces: Sequence[RecognizedFace]) -> EnrollProgress:
        if self.done:
            return self.progress
        n = len(self._captured)
        if not faces:
            return self._update(n, "No face detected. Look at the camera")
        if len(faces) > 1:
            return self._update(n, "Only one person should be in view")
        now = self._clock()
        if self._last_capture is not None and now - self._last_capture < self.interval_s:
            return self._update(n, "Slowly turn your head a little")
        face = faces[0]
        self._captured.append((face.embedding.copy(), crop_face(frame, face)))
        self._last_capture = now
        n += 1
        if n < self.samples:
            return self._update(n, "Slowly turn your head a little")
        self._commit()
        self.progress = EnrollProgress(n, self.samples, f"Saved {n} photos of {self.name}", done=True)
        return self.progress

    def _update(self, captured: int, message: str) -> EnrollProgress:
        self.progress = EnrollProgress(captured, self.samples, message)
        return self.progress

    def _commit(self) -> None:
        folder = face_folder(self.faces_dir, self.name)
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, (embedding, crop) in enumerate(self._captured, start=1):
            self.database.add(self.name, embedding)
            if crop.size:
                path = folder / f"{stamp}_{i}.jpg"
                cv2.imwrite(str(path), crop)
                self.saved_files.append(path)
