"""Download the pretrained OpenCV Zoo models (Apache-2.0)."""
from __future__ import annotations

import logging
import threading
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .config import Config
    from .database import FaceDatabase
    from .pipeline import RecognitionPipeline

log = logging.getLogger("charliecam.models")

BASE = "https://github.com/opencv/opencv_zoo/raw/main/models"
MODELS = {
    "face_detection_yunet_2023mar.onnx":
        f"{BASE}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx":
        f"{BASE}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}


def download_models(models_dir: Path, force: bool = False) -> None:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in MODELS.items():
        target = models_dir / filename
        if target.exists() and not force:
            print(f"  ok      {target}")
            continue
        print(f"  fetch   {url}")
        tmp = target.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(target)
        print(f"  saved   {target} ({target.stat().st_size // 1024} KB)")


def build_pipeline(cfg: "Config", database: "FaceDatabase") -> "RecognitionPipeline":
    """YuNet detector + SFace embedder + ``database``. Models must be downloaded."""
    from .engines import SFaceEmbedder, YuNetDetector
    from .pipeline import RecognitionPipeline

    detector = YuNetDetector(cfg.detector_model, cfg.detection_confidence, cfg.min_face_size)
    return RecognitionPipeline(detector, SFaceEmbedder(cfg.recognizer_model), database)


class PipelineTarget(Protocol):
    """What the desktop and web apps expose so models can load in the background."""

    database: "FaceDatabase"

    def set_status(self, text: str, ok: bool = False) -> None: ...

    def set_pipeline(self, pipeline: "RecognitionPipeline") -> None: ...


def load_pipeline_in_background(cfg: "Config", target: PipelineTarget) -> threading.Thread:
    """Download models on first run, build the pipeline, then hand it to ``target``.

    Runs on a thread so the window or web page appears immediately; progress and
    errors are reported through ``target.set_status`` instead of a console.
    """

    def work() -> None:
        try:
            if not (cfg.detector_model.exists() and cfg.recognizer_model.exists()):
                target.set_status("Downloading face models (first run)…")
                download_models(cfg.models_dir)
            target.set_pipeline(build_pipeline(cfg, target.database))
        except Exception as exc:
            log.exception("Model loading failed")
            target.set_status(f"Could not load models: {exc}")

    thread = threading.Thread(target=work, name="charliecam-model-loader", daemon=True)
    thread.start()
    return thread
