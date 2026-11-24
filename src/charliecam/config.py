"""Runtime configuration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

# OpenCV's published cosine threshold for SFace (higher = same person).
DEFAULT_MATCH_THRESHOLD = 0.363


@dataclass
class Config:
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    match_threshold: float = DEFAULT_MATCH_THRESHOLD
    detection_confidence: float = 0.8
    min_face_size: int = 40  # pixels; smaller faces are too blurry to trust
    event_cooldown_s: float = 10.0  # don't re-alert the same identity sooner

    @property
    def database_path(self) -> Path:
        return self.data_dir / "faces.npz"

    @property
    def events_path(self) -> Path:
        return self.data_dir / "events.jsonl"

    @property
    def snapshots_dir(self) -> Path:
        return self.data_dir / "snapshots"

    @property
    def detector_model(self) -> Path:
        return self.models_dir / "face_detection_yunet_2023mar.onnx"

    @property
    def recognizer_model(self) -> Path:
        return self.models_dir / "face_recognition_sface_2021dec.onnx"


def parse_source(value: str) -> Union[int, str]:
    """'0' -> 0 (webcam index); anything else stays a URL/path."""
    return int(value) if value.strip().isdigit() else value


__all__ = ["Config", "DEFAULT_MATCH_THRESHOLD", "parse_source"]
