"""Face detection and embedding engines.

The pipeline depends only on the two small protocols below, so the OpenCV
models can be swapped (or faked in tests) without touching anything else.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Protocol

import cv2
import numpy as np

from .types import BoundingBox, Detection


class FaceDetector(Protocol):
    def detect(self, frame: np.ndarray) -> List[Detection]: ...


class FaceEmbedder(Protocol):
    def embed(self, frame: np.ndarray, detection: Detection) -> np.ndarray: ...


def _require(path: Path) -> str:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Model not found: {path}. Run `python -m charliecam download-models` first."
        )
    return str(path)


class YuNetDetector:
    """OpenCV YuNet CNN face detector (fast on CPU, returns 5 landmarks)."""

    def __init__(self, model_path: Path, confidence: float = 0.8, min_face_size: int = 40):
        self._model = cv2.FaceDetectorYN.create(
            _require(model_path), "", (320, 320), confidence, 0.3, 5000
        )
        self._min_face_size = min_face_size
        self._input_size = (0, 0)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]
        if (w, h) != self._input_size:  # YuNet needs to know the frame size
            self._model.setInputSize((w, h))
            self._input_size = (w, h)
        _, faces = self._model.detect(frame)
        if faces is None:
            return []
        results = []
        for row in faces:
            box = BoundingBox(*(int(round(v)) for v in row[:4])).clip(w, h)
            if min(box.w, box.h) < self._min_face_size:
                continue
            results.append(Detection(box=box, confidence=float(row[14]), raw=row.copy()))
        return results


class SFaceEmbedder:
    """OpenCV SFace recogniser: aligns the face via landmarks, returns 128-d vector."""

    def __init__(self, model_path: Path):
        self._model = cv2.FaceRecognizerSF.create(_require(model_path), "")

    def embed(self, frame: np.ndarray, detection: Detection) -> np.ndarray:
        if detection.raw is None:
            raise ValueError("SFace needs YuNet landmarks (Detection.raw)")
        aligned = self._model.alignCrop(frame, detection.raw)
        return self._model.feature(aligned).reshape(-1).astype(np.float32)
