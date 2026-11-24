"""Persistent store of known face embeddings with vectorised matching."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from .types import UNKNOWN, MatchResult


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        raise ValueError("Cannot store a zero-length embedding")
    return vec / norm


class FaceDatabase:
    """Maps person names to one or more L2-normalised embeddings.

    Several embeddings per person (different angles/lighting) improve recall;
    a probe matches a person if it is close to *any* of their embeddings.
    """

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self._names: List[str] = []
        self._vectors = np.empty((0, 0), dtype=np.float32)

    # ----------------------------------------------------------- mutation
    def add(self, name: str, embedding: np.ndarray) -> None:
        name = name.strip()
        if not name:
            raise ValueError("Name must not be empty")
        if name == UNKNOWN:
            raise ValueError(f"'{UNKNOWN}' is reserved")
        vec = _normalize(embedding)
        if self._vectors.size == 0:
            self._vectors = vec[None, :]
        else:
            if vec.shape[0] != self._vectors.shape[1]:
                raise ValueError(
                    f"Embedding size {vec.shape[0]} != database size {self._vectors.shape[1]}"
                )
            self._vectors = np.vstack([self._vectors, vec])
        self._names.append(name)

    def remove(self, name: str) -> int:
        """Remove every embedding for ``name``; returns how many were removed."""
        keep = [i for i, n in enumerate(self._names) if n != name]
        removed = len(self._names) - len(keep)
        self._names = [self._names[i] for i in keep]
        self._vectors = self._vectors[keep] if keep else np.empty((0, 0), dtype=np.float32)
        return removed

    # -------------------------------------------------------------- query
    def match(self, embedding: np.ndarray) -> MatchResult:
        if not self._names:
            return MatchResult(UNKNOWN, -1.0)
        sims = self._vectors @ _normalize(embedding)
        best = int(np.argmax(sims))
        score = float(sims[best])
        name = self._names[best] if score >= self.threshold else UNKNOWN
        return MatchResult(name, score)

    def people(self) -> Dict[str, int]:
        """Person name -> number of stored samples."""
        counts: Dict[str, int] = {}
        for n in self._names:
            counts[n] = counts.get(n, 0) + 1
        return counts

    def __len__(self) -> int:
        return len(self._names)

    def __contains__(self, name: object) -> bool:
        return name in self._names

    # -------------------------------------------------------- persistence
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, vectors=self._vectors, names=np.array(json.dumps(self._names)))
        tmp.replace(path)  # atomic: never leaves a half-written database

    @classmethod
    def load(cls, path: Path, threshold: float) -> "FaceDatabase":
        db = cls(threshold)
        path = Path(path)
        if not path.exists():
            return db
        with np.load(path, allow_pickle=False) as data:
            db._names = json.loads(str(data["names"]))
            db._vectors = data["vectors"].astype(np.float32)
        if len(db._names) != (db._vectors.shape[0] if db._vectors.size else 0):
            raise ValueError(f"Corrupt face database: {path}")
        return db
