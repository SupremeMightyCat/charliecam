"""Command-line interface: `python -m flockcam <command>`."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import cv2

from .config import DEFAULT_MATCH_THRESHOLD, Config, parse_source
from .database import FaceDatabase

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="flockcam", description="Live facial recognition camera")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--models-dir", type=Path, default=Path("models"))
    p.add_argument("--threshold", type=float, default=DEFAULT_MATCH_THRESHOLD,
                   help="cosine similarity needed to call a match (default %(default)s)")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("download-models", help="fetch the YuNet + SFace ONNX models")

    e = sub.add_parser("enroll", help="add a person from image files/folders or the webcam")
    e.add_argument("name")
    e.add_argument("images", nargs="*", type=Path, help="image files or folders")
    e.add_argument("--camera", action="store_true", help="capture from webcam instead")
    e.add_argument("--samples", type=int, default=5, help="webcam samples to capture")
    e.add_argument("--source", default="0")

    sub.add_parser("list", help="list enrolled people")
    r = sub.add_parser("remove", help="remove a person")
    r.add_argument("name")

    run = sub.add_parser("run", help="start live recognition")
    run.add_argument("--source", default="0", help="webcam index, RTSP URL or video file")
    run.add_argument("--cooldown", type=float, default=10.0, help="seconds between repeat alerts")
    run.add_argument("--every", type=int, default=1, help="process every Nth frame")
    run.add_argument("--no-window", action="store_true", help="headless mode (log only)")
    return p


def _config(args) -> Config:
    return Config(data_dir=args.data_dir, models_dir=args.models_dir,
                  match_threshold=args.threshold)


def _expand_images(paths: List[Path]) -> List[Path]:
    out = []
    for p in paths:
        if p.is_dir():
            out.extend(sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS))
        else:
            out.append(p)
    return out


def _build_pipeline(cfg: Config, db: FaceDatabase):
    from .engines import SFaceEmbedder, YuNetDetector
    from .pipeline import RecognitionPipeline

    detector = YuNetDetector(cfg.detector_model, cfg.detection_confidence, cfg.min_face_size)
    return RecognitionPipeline(detector, SFaceEmbedder(cfg.recognizer_model), db)


def cmd_enroll(args, cfg: Config, pipeline=None) -> int:
    db = FaceDatabase.load(cfg.database_path, cfg.match_threshold)
    pipeline = pipeline or _build_pipeline(cfg, db)
    pipeline.database = db
    added = 0
    if args.camera:
        from .camera import capture_frame

        for i in range(args.samples):
            input(f"Sample {i + 1}/{args.samples}: face the camera (vary angle slightly), press Enter...")
            try:
                pipeline.enroll_from_image(args.name, capture_frame(parse_source(args.source)))
                added += 1
                print("  captured")
            except ValueError as exc:
                print(f"  skipped: {exc}")
    else:
        files = _expand_images(args.images)
        if not files:
            print("No images given. Pass image paths/folders, or use --camera.")
            return 2
        for path in files:
            img = cv2.imread(str(path))
            if img is None:
                print(f"  skipped {path}: not a readable image")
                continue
            try:
                pipeline.enroll_from_image(args.name, img)
                added += 1
                print(f"  enrolled {path}")
            except ValueError as exc:
                print(f"  skipped {path}: {exc}")
    if added:
        db.save(cfg.database_path)
    print(f"Added {added} sample(s) for '{args.name}'.")
    return 0 if added else 1


def cmd_run(args, cfg: Config) -> int:
    from .camera import run_live
    from .events import EventRecorder

    db = FaceDatabase.load(cfg.database_path, cfg.match_threshold)
    pipeline = _build_pipeline(cfg, db)
    recorder = EventRecorder(cfg.events_path, cfg.snapshots_dir, cooldown_s=args.cooldown)

    def enroll_live(frame):
        cv2.destroyAllWindows()
        name = input("Name for the largest face in view (blank to cancel): ").strip()
        if not name:
            return
        try:
            pipeline.enroll_from_image(name, frame)
            db.save(cfg.database_path)
            print(f"Enrolled '{name}'.")
        except ValueError as exc:
            print(f"Enroll failed: {exc}")

    print(f"Loaded {len(db.people())} people. Events -> {cfg.events_path}")
    run_live(pipeline, recorder, parse_source(args.source), args.every,
             show_window=not args.no_window, on_enroll=enroll_live)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _config(args)
    if args.command == "download-models":
        from .models import download_models

        download_models(cfg.models_dir)
        return 0
    if args.command == "enroll":
        return cmd_enroll(args, cfg)
    if args.command == "list":
        people = FaceDatabase.load(cfg.database_path, cfg.match_threshold).people()
        if not people:
            print("No one enrolled yet.")
        for name, n in sorted(people.items()):
            print(f"{name}: {n} sample(s)")
        return 0
    if args.command == "remove":
        db = FaceDatabase.load(cfg.database_path, cfg.match_threshold)
        n = db.remove(args.name)
        db.save(cfg.database_path)
        print(f"Removed {n} sample(s) for '{args.name}'.")
        return 0 if n else 1
    if args.command == "run":
        return cmd_run(args, cfg)
    return 2
