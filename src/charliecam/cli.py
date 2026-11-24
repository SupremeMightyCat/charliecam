"""Command-line interface: `python -m charliecam <command>`."""
from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional

import cv2

from .config import DEFAULT_MATCH_THRESHOLD, Config, parse_source
from .database import FaceDatabase

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="charliecam", description="Live facial recognition camera")
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

    a = sub.add_parser("attendance", help="who was seen on a day (default today)")
    a.add_argument("--date", type=date.fromisoformat, help="YYYY-MM-DD")
    a.add_argument("--csv", type=Path, help="also save the report as a CSV file")

    run = sub.add_parser("run", help="start live recognition")
    run.add_argument("--source", default="0", help="webcam index, RTSP URL or video file")
    run.add_argument("--cooldown", type=float, default=10.0, help="seconds between repeat alerts")
    run.add_argument("--every", type=int, default=1, help="process every Nth frame")
    run.add_argument("--no-window", action="store_true", help="headless mode (log only)")

    g = sub.add_parser("gui", help="open the desktop app (live view + point-and-click enrollment)")
    g.add_argument("--source", default="0")

    w = sub.add_parser("web", help="open CharlieCam in your browser (local web app)")
    w.add_argument("--source", default="0", help="webcam index, RTSP URL or video file")
    w.add_argument("--host", default="127.0.0.1",
                   help="use 0.0.0.0 to allow other devices on your network (default: this PC only)")
    w.add_argument("--port", type=int, default=8000)
    w.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
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
    from .models import build_pipeline

    return build_pipeline(cfg, db)


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


def cmd_attendance(args, cfg: Config) -> int:
    from .attendance import report, to_csv

    enrolled = FaceDatabase.load(cfg.database_path, cfg.match_threshold).people()
    rep = report(cfg.events_path, enrolled, args.date)
    print(f"Attendance for {rep['date']}: {len(rep['present'])} present, {len(rep['absent'])} absent")
    for a in rep["present"]:
        first = datetime.fromisoformat(a["first_seen"]).strftime("%H:%M")
        print(f"  present  {a['name']}  (first seen {first}, {a['sightings']} sighting(s))")
    for name in rep["absent"]:
        print(f"  absent   {name}")
    if args.csv:
        args.csv.write_text(to_csv(rep), encoding="utf-8", newline="")
        print(f"Saved {args.csv}")
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
        if n:
            from .enrollment import delete_face_photos

            delete_face_photos(cfg.data_dir / "faces", args.name)
        print(f"Removed {n} sample(s) for '{args.name}'.")
        return 0 if n else 1
    if args.command == "attendance":
        return cmd_attendance(args, cfg)
    if args.command == "run":
        return cmd_run(args, cfg)
    if args.command == "gui":
        from .gui.app import main as gui_main

        return gui_main(["--source", args.source, "--data-dir", str(cfg.data_dir),
                         "--models-dir", str(cfg.models_dir), "--threshold", str(cfg.match_threshold)])
    if args.command == "web":
        from .web.app import main as web_main

        web_args = ["--source", args.source, "--host", args.host, "--port", str(args.port),
                    "--data-dir", str(cfg.data_dir), "--models-dir", str(cfg.models_dir),
                    "--threshold", str(cfg.match_threshold)]
        return web_main(web_args + (["--no-browser"] if args.no_browser else []))
    return 2
