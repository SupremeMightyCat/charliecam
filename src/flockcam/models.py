"""Download the pretrained OpenCV Zoo models (Apache-2.0)."""
from __future__ import annotations

import urllib.request
from pathlib import Path

BASE = "https://github.com/opencv/opencv_zoo/raw/main/models"
MODELS = {
    "face_detection_yunet_2023mar.onnx": f"{BASE}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": f"{BASE}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
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
