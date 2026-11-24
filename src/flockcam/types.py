"""Core value types shared across the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

UNKNOWN = "Unknown"


@dataclass(frozen=True)
class BoundingBox:
    x: int
    y: int
    w: int
    h: int

    def clip(self, width: int, height: int) -> "BoundingBox":
        x0 = max(0, min(self.x, width - 1))
        y0 = max(0, min(self.y, height - 1))
        x1 = max(x0 + 1, min(self.x + self.w, width))
        y1 = max(y0 + 1, min(self.y + self.h, height))
        return BoundingBox(x0, y0, x1 - x0, y1 - y0)

    @property
    def area(self) -> int:
        return self.w * self.h


@dataclass
class Detection:
    """A face found in a frame.

    ``raw`` holds the detector's native output row (YuNet's 15-value row with
    landmarks), which the SFace embedder needs to align the face.
    """

    box: BoundingBox
    confidence: float
    raw: Optional[np.ndarray] = None


@dataclass
class MatchResult:
    name: str  # person's name, or UNKNOWN
    similarity: float  # best cosine similarity found in the database (-1..1)

    @property
    def is_known(self) -> bool:
        return self.name != UNKNOWN


@dataclass
class RecognizedFace:
    detection: Detection
    match: MatchResult
    embedding: np.ndarray = field(repr=False)
    track_label: str = ""  # display label, e.g. "Alice" or "Unknown #3"
