"""Start the CharlieCam web app: camera + recognition + a local web server.

Run with `charliecam web` or double-click `CharlieCam Web.pyw`. The page opens in
your browser at http://127.0.0.1:8000.
"""
from __future__ import annotations

import argparse
import logging
import socket
import sys
import threading
import traceback
import webbrowser
from pathlib import Path
from typing import List, Optional

log = logging.getLogger("charliecam.web")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def build_parser() -> argparse.ArgumentParser:
    from ..config import DEFAULT_MATCH_THRESHOLD

    p = argparse.ArgumentParser(prog="charliecam-web", description="CharlieCam in your browser")
    p.add_argument("--source", default="0", help="webcam index, RTSP URL or video file")
    p.add_argument("--host", default=DEFAULT_HOST,
                   help="address to listen on; use 0.0.0.0 to allow other devices on your network")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--models-dir", type=Path, default=Path("models"))
    p.add_argument("--threshold", type=float, default=DEFAULT_MATCH_THRESHOLD)
    return p


def _port_in_use(host: str, port: int) -> bool:
    probe = "127.0.0.1" if host in ("0.0.0.0", "") else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((probe, port)) == 0


def page_url(host: str, port: int) -> str:
    return f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}/"


def run(args: argparse.Namespace) -> int:
    import uvicorn

    from ..camera import CameraThread
    from ..config import Config, parse_source
    from ..database import FaceDatabase
    from ..events import EventRecorder
    from ..models import load_pipeline_in_background
    from .server import create_app
    from .service import WebService

    url = page_url(args.host, args.port)
    if _port_in_use(args.host, args.port):
        # Most likely CharlieCam is already running: just show it.
        log.info("Port %s already in use; opening the existing page", args.port)
        if not args.no_browser:
            webbrowser.open(url)
        print(f"Something is already running at {url} (CharlieCam is probably open already).")
        return 0

    cfg = Config(data_dir=args.data_dir, models_dir=args.models_dir, match_threshold=args.threshold)
    db = FaceDatabase.load(cfg.database_path, cfg.match_threshold)
    recorder = EventRecorder(cfg.events_path, cfg.snapshots_dir, cooldown_s=cfg.event_cooldown_s)
    service = WebService(db, recorder, cfg.database_path, cfg.data_dir / "faces",
                         settings_path=cfg.data_dir / "web_settings.json")
    service.set_status("Loading face models…")
    load_pipeline_in_background(cfg, service)

    camera = CameraThread(parse_source(args.source))
    camera.start()
    service.start(camera.latest, error=lambda: camera.error)

    server: Optional[uvicorn.Server] = None

    def quit_server() -> None:
        if server is not None:
            server.should_exit = True

    app = create_app(service, on_quit=quit_server)
    server = uvicorn.Server(uvicorn.Config(
        app, host=args.host, port=args.port, log_config=None, access_log=False,
        timeout_graceful_shutdown=2,
    ))
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        log.warning("Listening on %s: other devices on your network can open CharlieCam", args.host)
        print(f"Warning: CharlieCam is reachable from other devices on your network ({args.host}).")

    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    print(f"CharlieCam is running at {url}  (Ctrl+C or the Quit button to stop)")
    try:
        server.run()
    finally:
        service.stop()
        camera.stop()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=args.data_dir / "charliecam.log", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        return run(args)
    except Exception:
        log.error("Fatal error\n%s", traceback.format_exc())
        raise


if __name__ == "__main__":
    sys.exit(main())
