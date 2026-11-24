"""Frame -> recognised faces. Pure logic: no camera, no UI, no disk."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .database import FaceDatabase, _normalize
from .engines import FaceDetector, FaceEmbedder
from .types import RecognizedFace


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


class RecognitionPipeline:
    def __init__(
        self,
        detector: FaceDetector,
        embedder: FaceEmbedder,
        database: FaceDatabase,
    ) -> None:
        self.detector = detector
        self.embedder = embedder
        self.database = database
        self.unknowns = UnknownRegistry(database.threshold)

    def process(self, frame: np.ndarray) -> List[RecognizedFace]:
        faces = []
        for det in self.detector.detect(frame):
            emb = self.embedder.embed(frame, det)
            match = self.database.match(emb)
            if match.is_known:
                label = match.name
            else:
                label = f"Unknown #{self.unknowns.identify(emb)}"
            faces.append(RecognizedFace(det, match, emb, label))
        return faces

    def enroll_from_image(self, name: str, image: np.ndarray) -> np.ndarray:
        """Add the single largest face in ``image`` to the database."""
        dets = self.detector.detect(image)
        if not dets:
            raise ValueError("No face found in image")
        det = max(dets, key=lambda d: d.box.area)
        emb = self.embedder.embed(image, det)
        self.database.add(name, emb)
        return emb
