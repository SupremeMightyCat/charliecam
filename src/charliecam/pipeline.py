"""Frame -> recognised faces. Pure logic: no camera, no UI, no disk."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import cv2
import numpy as np

from .database import FaceDatabase, _normalize
from .engines import FaceDetector, FaceEmbedder
from .types import BoundingBox, MatchResult, RecognizedFace

# Phone photos are 3000-4000 px tall. YuNet is trained on faces at camera-like
# sizes and misses a face that fills a frame that large, so photos are scaled
# down to this longest side before detection.
MAX_PHOTO_SIDE = 1280


def fit_photo(image: np.ndarray, max_side: int = MAX_PHOTO_SIDE) -> np.ndarray:
    """Scale ``image`` down so its longest side is at most ``max_side`` (never up)."""
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return image
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


@dataclass
class _UnknownTrack:
    ident: int
    embedding: np.ndarray


class UnknownRegistry:
    """Gives repeat sightings of the same stranger a stable id ("Unknown #2").

    Without this, one stranger standing in front of the camera would generate
    a new alert and snapshot every few seconds as a "new" unknown.
    """

    def __init__(self, threshold: float, max_tracks: int = 200) -> None:
        self.threshold = threshold
        self.max_tracks = max_tracks
        self._tracks: List[_UnknownTrack] = []
        self._next_id = 1

    def identify(self, embedding: np.ndarray) -> int:
        vec = _normalize(embedding)
        if self._tracks:
            sims = np.array([t.embedding @ vec for t in self._tracks])
            best = int(np.argmax(sims))
            if sims[best] >= self.threshold:
                return self._tracks[best].ident
        track = _UnknownTrack(self._next_id, vec)
        self._next_id += 1
        self._tracks.append(track)
        if len(self._tracks) > self.max_tracks:
            self._tracks.pop(0)
        return track.ident


def iou(a: BoundingBox, b: BoundingBox) -> float:
    """Intersection over union of two boxes (0 = apart, 1 = identical)."""
    x0, y0 = max(a.x, b.x), max(a.y, b.y)
    x1, y1 = min(a.x + a.w, b.x + b.w), min(a.y + a.h, b.y + b.h)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = a.area + b.area - inter
    return inter / union if union else 0.0


@dataclass
class _Track:
    name: str
    box: BoundingBox
    matched_at: float


class IdentityTracker:
    """Keeps a recognised name on a moving face through a few blurry frames.

    When someone walks past the camera, some frames are too blurred or turned
    away to match on their own. If such a face overlaps the box of a person who
    was confidently matched less than ``hold_s`` seconds ago, it keeps that
    person's name instead of flickering to "Unknown". The hold is measured from
    the last real match and is never extended by a held frame, so a stranger who
    steps into the same spot is labelled correctly after at most ``hold_s``.
    """

    def __init__(self, hold_s: float = 1.0, min_iou: float = 0.3,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.hold_s = hold_s
        self.min_iou = min_iou
        self._clock = clock
        self._tracks: List[_Track] = []  # at most one per person

    def held_name(self, box: BoundingBox) -> Optional[str]:
        now = self._clock()
        self._tracks = [t for t in self._tracks if now - t.matched_at <= self.hold_s]
        best = max(self._tracks, key=lambda t: iou(t.box, box), default=None)
        return best.name if best is not None and iou(best.box, box) >= self.min_iou else None

    def observe(self, name: str, box: BoundingBox, matched: bool) -> None:
        now = self._clock()
        for track in self._tracks:
            if track.name == name:
                track.box = box  # follow the face as it moves
                if matched:
                    track.matched_at = now
                return
        if matched:
            self._tracks.append(_Track(name, box, now))


class RecognitionPipeline:
    def __init__(
        self,
        detector: FaceDetector,
        embedder: FaceEmbedder,
        database: FaceDatabase,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.detector = detector
        self.embedder = embedder
        self.database = database
        self.unknowns = UnknownRegistry(database.threshold)
        self.tracker = IdentityTracker(clock=clock)

    def process(self, frame: np.ndarray) -> List[RecognizedFace]:
        faces = []
        for det in self.detector.detect(frame):
            emb = self.embedder.embed(frame, det)
            match = self.database.match(emb)
            matched = match.is_known
            if not matched:
                held = self.tracker.held_name(det.box)
                if held is not None:
                    match = MatchResult(held, match.similarity, held=True)
            if match.is_known:
                self.tracker.observe(match.name, det.box, matched)
                label = match.name
            else:
                label = f"Unknown #{self.unknowns.identify(emb)}"
            faces.append(RecognizedFace(det, match, emb, label))
        return faces

    def enroll_from_image(self, name: str, image: np.ndarray) -> np.ndarray:
        """Add the single largest face in ``image`` to the database."""
        image = fit_photo(image)
        dets = self.detector.detect(image)
        if not dets:
            raise ValueError("No face found in image")
        det = max(dets, key=lambda d: d.box.area)
        emb = self.embedder.embed(image, det)
        self.database.add(name, emb)
        return emb
