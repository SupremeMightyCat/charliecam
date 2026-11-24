"""Benchmark: does the IdentityTracker keep names on people walking past?

Enrolls ONE photo of a person (like the live-demo upload), then simulates them
walking across the frame: each other photo slides left to right over 40 frames,
and a random half of the frames get heavy horizontal motion blur. The same clips
run with the tracker off (hold 0 s) and on (hold 1 s).

    python tools/walk_benchmark.py data/faces/Charlie_Woodhead

Needs the models (`charliecam download-models`) and a folder of at least 2 photos
of one person. Results are synthetic: a guide for tuning, not a field accuracy figure.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from charliecam.config import Config
from charliecam.database import FaceDatabase
from charliecam.models import build_pipeline

FRAMES = 40
FPS = 30


def make_pipeline(cfg: Config, name: str, photo: Path, hold_s: float):
    pipeline = build_pipeline(cfg, FaceDatabase(cfg.match_threshold))
    pipeline.tracker.hold_s = hold_s
    pipeline.enroll_from_image(name, cv2.imread(str(photo)))
    return pipeline


def walk_past(pipeline, name: str, photo: Path, blur: int, rng: np.random.Generator):
    """Returns (frames labelled with the right name, frames with a face, false unknown ids)."""
    face = cv2.imread(str(photo))
    scale = 260 / face.shape[0]
    face = cv2.resize(face, None, fx=scale, fy=scale)
    now = [0.0]
    pipeline.tracker._clock = lambda: now[0]  # simulated time: exactly 30 fps
    named = seen = 0
    unknown_ids = set()
    for i in range(FRAMES):
        now[0] += 1 / FPS
        frame = np.full((480, 640, 3), 60, np.uint8)
        x = int(-face.shape[1] * 0.3 + i * (640 - face.shape[1] * 0.4) / (FRAMES - 1))
        x0, x1 = max(0, x), min(640, x + face.shape[1])
        frame[100:100 + face.shape[0], x0:x1] = face[:, x0 - x:x1 - x]
        if rng.random() < 0.5:  # uneven walking speed: some frames sharp, some smeared
            kernel = np.zeros((blur, blur), np.float32)
            kernel[blur // 2, :] = 1 / blur
            frame = cv2.filter2D(frame, -1, kernel)
        faces = pipeline.process(frame)
        if faces:
            seen += 1
            named += faces[0].track_label == name
            if not faces[0].match.is_known:
                unknown_ids.add(faces[0].track_label)
    return named, seen, len(unknown_ids)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("folder", type=Path, help="folder with 2+ photos of one person")
    p.add_argument("--models-dir", type=Path, default=Path("models"))
    p.add_argument("--blur", type=int, nargs="+", default=[31, 41, 51], help="blur lengths in px")
    args = p.parse_args(argv)

    photos = sorted(f for f in args.folder.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(photos) < 2:
        print("Need at least 2 photos: one to enroll, the rest to walk past.")
        return 2
    cfg = Config(models_dir=args.models_dir)
    name = args.folder.name.replace("_", " ")

    print(f"{name}: enrolled from 1 photo, {len(photos) - 1} walk-past clips per setting\n")
    print(f"{'blur':>6} {'tracker':>8} {'named frames':>14} {'false unknowns':>15}")
    totals = {}
    for blur in args.blur:
        for hold in (0.0, 1.0):
            rng = np.random.default_rng(0)  # same blurred frames for both settings
            named = seen = unknown = 0
            for photo in photos[1:]:
                n, s, u = walk_past(make_pipeline(cfg, name, photos[0], hold), name, photo, blur, rng)
                named, seen, unknown = named + n, seen + s, unknown + u
            t = totals.setdefault(hold, [0, 0, 0])
            t[0], t[1], t[2] = t[0] + named, t[1] + seen, t[2] + unknown
            label = "on" if hold else "off"
            print(f"{blur:>4}px {label:>8} {named:>6}/{seen:<7} {unknown:>15}")
    print()
    for hold, (named, seen, unknown) in totals.items():
        pct = 100 * named / seen if seen else 0
        print(f"tracker {'on ' if hold else 'off'}: {pct:.1f}% of frames named, {unknown} false unknown ids")
    return 0


if __name__ == "__main__":
    sys.exit(main())
