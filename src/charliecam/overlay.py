"""Drawing boxes and labels on frames."""
from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from .types import RecognizedFace

GREEN = (0, 200, 0)  # BGR
RED = (0, 0, 230)
WHITE = (255, 255, 255)


def draw_faces(frame: np.ndarray, faces: Iterable[RecognizedFace]) -> np.ndarray:
    for face in faces:
        b = face.detection.box
        color = GREEN if face.match.is_known else RED
        cv2.rectangle(frame, (b.x, b.y), (b.x + b.w, b.y + b.h), color, 2)
        text = f"{face.track_label} ({face.match.similarity:.2f})"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        top = max(0, b.y - th - 8)
        cv2.rectangle(frame, (b.x, top), (b.x + tw + 6, top + th + 8), color, -1)
        cv2.putText(frame, text, (b.x + 3, top + th + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
    return frame


def draw_status(frame: np.ndarray, text: str) -> np.ndarray:
    cv2.putText(frame, text, (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    return frame
