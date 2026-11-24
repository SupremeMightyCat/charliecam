"""Test doubles: "faces" are solid-colour squares and their colour is the embedding.

This lets every layer above the neural networks be tested deterministically,
with no model files and no camera.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pytest

from charliecam.database import FaceDatabase
from charliecam.pipeline import RecognitionPipeline
from charliecam.types import BoundingBox, Detection

# BGR colours that are orthogonal to each other => cosine similarity 0.
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
FACE = 60


def make_frame(*faces, size=(240, 320)) -> np.ndarray:
    """faces: (x, y, colour) tuples drawn as FACE x FACE squares on black."""
    frame = np.zeros((*size, 3), dtype=np.uint8)
    for x, y, colour in faces:
        frame[y:y + FACE, x:x + FACE] = colour
    return frame


class FakeDetector:
    """Finds every FACE-sized non-black square, scanning on a 10 px grid."""

    def detect(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]
        mask = frame.any(axis=2)
        seen = np.zeros_like(mask)
        out = []
        for y in range(0, h, 10):
            for x in range(0, w, 10):
                if mask[y, x] and not seen[y, x]:
                    ys, xs = np.where(mask[y:y + FACE * 2, x:x + FACE * 2])
                    bw, bh = xs.max() + 1, ys.max() + 1
                    seen[y:y + bh, x:x + bw] = True
                    out.append(Detection(BoundingBox(x, y, int(bw), int(bh)), 0.99))
        return out


class FakeEmbedder:
    def embed(self, frame: np.ndarray, det: Detection) -> np.ndarray:
        b = det.box
        return frame[b.y:b.y + b.h, b.x:b.x + b.w].reshape(-1, 3).mean(axis=0).astype(np.float32)


@pytest.fixture
def db() -> FaceDatabase:
    return FaceDatabase(threshold=0.9)


@pytest.fixture
def pipeline(db) -> RecognitionPipeline:
    return RecognitionPipeline(FakeDetector(), FakeEmbedder(), db)
