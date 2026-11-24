"""HTTP layer: a thin FastAPI wrapper around :class:`WebService`."""
from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path
from typing import Callable, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..attendance import to_csv
from .service import WebService

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
BOUNDARY = "charliecamframe"
NO_STORE = {"Cache-Control": "no-store"}
CSRF_HEADER = "X-CharlieCam"


class EnrollRequest(BaseModel):
    name: str


class SettingsRequest(BaseModel):
    threshold: Optional[float] = None
    cooldown: Optional[float] = None


def create_app(service: WebService, on_quit: Optional[Callable[[], None]] = None) -> FastAPI:
    app = FastAPI(title="CharlieCam", docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def require_app_header(request: Request, call_next):
        # Changes must come from CharlieCam's own page. A custom header can't be
        # sent cross-site without a CORS preflight, which this server never
        # approves, so other websites open in the browser can't trigger them.
        if request.method not in ("GET", "HEAD", "OPTIONS") and request.headers.get(CSRF_HEADER) != "1":
            return JSONResponse({"detail": "Missing CharlieCam request header"}, status_code=403)
        return await call_next(request)

    @app.exception_handler(ValueError)
    async def value_error(_request, exc: ValueError):  # user-facing validation messages
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html", headers=NO_STORE)

    # ------------------------------------------------------------- video
    @app.get("/api/stream")
    async def stream(request: Request):
        async def frames():
            seq = 0
            while not service.stopped:
                if await request.is_disconnected():
                    break
                new_seq, jpeg = await asyncio.to_thread(service.wait_for_frame, seq, 0.5)
                if jpeg is None or new_seq == seq:
                    continue
                seq = new_seq
                yield (f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                       f"Content-Length: {len(jpeg)}\r\n\r\n").encode() + jpeg + b"\r\n"

        return StreamingResponse(frames(), headers=NO_STORE,
                                 media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}")

    @app.get("/api/frame.jpg")
    def frame():
        jpeg = service.latest_jpeg()
        if jpeg is None:
            raise HTTPException(503, "No camera frame yet")
        return Response(jpeg, media_type="image/jpeg", headers=NO_STORE)

    # ------------------------------------------------------------- state
    @app.get("/api/state")
    def state(since: int = Query(0, ge=0)):
        return JSONResponse(service.state(since), headers=NO_STORE)

    # ------------------------------------------------------------ people
    @app.get("/api/people")
    def people():
        return JSONResponse(service.people(), headers=NO_STORE)

    @app.get("/api/people/{name}/photo")
    def person_photo(name: str):
        path = service.person_photo(name)
        if path is None:
            raise HTTPException(404, "No photo")
        return FileResponse(path, headers=NO_STORE)

    @app.delete("/api/people/{name}")
    def remove_person(name: str):
        if not service.remove_person(name):
            raise HTTPException(404, f"No one called {name!r} is enrolled")
        return {"removed": name}

    @app.post("/api/people/{name}/photos")
    async def upload_photo(name: str, request: Request):
        data = await request.body()
        if not data:
            raise HTTPException(400, "Empty upload")
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Photo is larger than 15 MB")
        enrolled = await asyncio.to_thread(service.enroll_photo, name, data)
        return {"name": enrolled, "samples": service.database.people().get(enrolled, 0)}

    # -------------------------------------------------------- enrollment
    @app.post("/api/enroll")
    def enroll(req: EnrollRequest):
        return {"name": service.start_enrollment(req.name)}

    @app.post("/api/enroll/cancel")
    def cancel():
        return {"cancelled": service.cancel_enrollment()}

    # ---------------------------------------------------------- activity
    @app.get("/api/snapshots/{filename}")
    def snapshot(filename: str):
        path = service.snapshot_path(filename)
        if path is None:
            raise HTTPException(404, "No such snapshot")
        return FileResponse(path)

    # -------------------------------------------------------- attendance
    def parse_day(value: Optional[str]) -> Optional[date]:
        try:
            return date.fromisoformat(value) if value else None
        except ValueError:
            raise HTTPException(400, "Dates look like 2026-11-23") from None

    @app.get("/api/attendance")
    def get_attendance(day: Optional[str] = Query(None, alias="date")):
        return JSONResponse(service.attendance(parse_day(day)), headers=NO_STORE)

    @app.get("/api/attendance.csv")
    def attendance_csv(day: Optional[str] = Query(None, alias="date")):
        rep = service.attendance(parse_day(day))
        return Response(to_csv(rep), media_type="text/csv", headers={
            **NO_STORE, "Content-Disposition": f'attachment; filename="attendance-{rep["date"]}.csv"'})

    # ---------------------------------------------------------- settings
    @app.get("/api/settings")
    def get_settings():
        return service.settings()

    @app.put("/api/settings")
    def put_settings(req: SettingsRequest):
        return service.update_settings(req.threshold, req.cooldown)

    # -------------------------------------------------------------- quit
    @app.post("/api/quit")
    def quit_app():
        if on_quit is None:
            raise HTTPException(404, "Quit is not available")
        on_quit()
        return {"quitting": True}

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
