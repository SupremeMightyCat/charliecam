"""Side effects for sightings: JSON-lines event log and snapshot images."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from .types import RecognizedFace


class EventRecorder:
    """Logs every *new* sighting and saves a snapshot of unknown faces.

    A per-identity cooldown stops a person who stays in frame from flooding the
    log: they are reported once, then again only after ``cooldown_s`` without
    being re-reported.
    """

    def __init__(
        self,
        log_path: Path,
        snapshot_dir: Path,
        cooldown_s: float = 10.0,
        snapshot_known: bool = False,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.log_path = Path(log_path)
        self.snapshot_dir = Path(snapshot_dir)
        self.cooldown_s = cooldown_s
        self.snapshot_known = snapshot_known
        self._clock = clock
        self._last_seen: Dict[str, float] = {}
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def _should_report(self, label: str) -> bool:
        now = self._clock()
        last = self._last_seen.get(label)
        if last is not None and now - last < self.cooldown_s:
            return False
        self._last_seen[label] = now
        return True

    def record(self, frame: np.ndarray, faces: List[RecognizedFace]) -> List[dict]:
        events = []
        for face in faces:
            if not self._should_report(face.track_label):
                continue
            stamp = datetime.now(timezone.utc)
            snapshot = None
            if not face.match.is_known or self.snapshot_known:
                snapshot = self._save_snapshot(frame, face, stamp)
            event = {
                "timestamp": stamp.isoformat(),
                "label": face.track_label,
                "known": face.match.is_known,
                "similarity": round(face.match.similarity, 4),
                "held": face.match.held,
                "confidence": round(face.detection.confidence, 4),
                "box": list(face.detection.box.__dict__.values()),
                "snapshot": str(snapshot) if snapshot else None,
            }
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event) + "\n")
            events.append(event)
        return events

    def _save_snapshot(
        self, frame: np.ndarray, face: RecognizedFace, stamp: datetime
    ) -> Optional[Path]:
        h, w = frame.shape[:2]
        b = face.detection.box
        pad = int(0.25 * max(b.w, b.h))  # include a bit of context around the face
        x0, y0 = max(0, b.x - pad), max(0, b.y - pad)
        x1, y1 = min(w, b.x + b.w + pad), min(h, b.y + b.h + pad)
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        safe = face.track_label.replace(" ", "_").replace("#", "")
        path = self.snapshot_dir / f"{stamp:%Y%m%dT%H%M%S_%f}_{safe}.jpg"
        cv2.imwrite(str(path), crop)
        return path
