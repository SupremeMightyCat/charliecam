"""Smoke tests against the real ONNX models. Skipped until they are downloaded."""
from pathlib import Path

import numpy as np
import pytest

from flockcam.config import Config

cfg = Config(models_dir=Path(__file__).resolve().parents[1] / "models")
pytestmark = pytest.mark.skipif(
    not (cfg.detector_model.exists() and cfg.recognizer_model.exists()),
    reason="models not downloaded (python -m flockcam download-models)",
)


def test_detector_finds_nothing_in_blank_frame():
    from flockcam.engines import YuNetDetector

    det = YuNetDetector(cfg.detector_model)
    assert det.detect(np.zeros((480, 640, 3), dtype=np.uint8)) == []


def test_detector_handles_changing_frame_sizes():
    from flockcam.engines import YuNetDetector

    det = YuNetDetector(cfg.detector_model)
    for shape in [(480, 640, 3), (720, 1280, 3), (240, 320, 3)]:
        assert det.detect(np.full(shape, 128, dtype=np.uint8)) == []


def test_embedder_loads():
    from flockcam.engines import SFaceEmbedder

    SFaceEmbedder(cfg.recognizer_model)
