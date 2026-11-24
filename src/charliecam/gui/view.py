"""Renders the whole window as one image from a plain ViewModel.

Rendering is a pure function: ViewModel in, (BGR image, clickable regions) out.
That keeps the drawing code free of state and lets tests check the layout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ..enrollment import EnrollProgress
from ..types import RecognizedFace
from . import theme as T

Rect = Tuple[int, int, int, int]  # x0, y0, x1, y1


@dataclass
class ActivityItem:
    time: str
    label: str
    known: bool


@dataclass
class ViewModel:
    frame: Optional[np.ndarray] = None  # BGR camera frame
    faces: List[RecognizedFace] = field(default_factory=list)
    status: str = "Starting…"
    status_ok: bool = False
    fps: float = 0.0
    name_text: str = ""
    caret_visible: bool = True
    name_error: str = ""
    enrolling_name: Optional[str] = None
    progress: Optional[EnrollProgress] = None
    people: Dict[str, int] = field(default_factory=dict)
    confirm_remove: Optional[str] = None
    activity: List[ActivityItem] = field(default_factory=list)
    toast: str = ""
    toast_ok: bool = True
    hover: Optional[str] = None
    can_capture: bool = False


def _text_w(draw: ImageDraw.ImageDraw, text: str, f) -> int:
    return int(draw.textlength(text, font=f))


def _ellipsize(draw, text: str, f, max_w: int) -> str:
    if _text_w(draw, text, f) <= max_w:
        return text
    while text and _text_w(draw, text + "…", f) > max_w:
        text = text[:-1]
    return text + "…"


class Renderer:
    def __init__(self) -> None:
        self.regions: Dict[str, Rect] = {}
        self._mask_cache: Dict[Tuple[int, int], Image.Image] = {}

    def _region(self, key: str, rect: Rect) -> Rect:
        self.regions[key] = rect
        return rect

    def hit(self, x: int, y: int) -> Optional[str]:
        for key, (x0, y0, x1, y1) in self.regions.items():
            if x0 <= x <= x1 and y0 <= y <= y1:
                return key
        return None

    # ------------------------------------------------------------------ main
    def render(self, vm: ViewModel) -> np.ndarray:
        self.regions = {}
        img = Image.new("RGB", (T.WIDTH, T.HEIGHT), T.BG)
        d = ImageDraw.Draw(img)
        video = (T.MARGIN, T.MARGIN, T.WIDTH - T.SIDEBAR_W - 2 * T.MARGIN, T.HEIGHT - T.MARGIN)
        self._video(img, d, vm, video)
        x0 = video[2] + T.MARGIN
        y = self._header(d, vm, x0, T.MARGIN)
        y = self._enroll_card(d, vm, x0, y + 18)
        y = self._people_card(d, vm, x0, y + 16, T.HEIGHT - T.MARGIN - 150)
        self._activity_card(d, vm, x0, y + 16, T.HEIGHT - T.MARGIN)
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    # ----------------------------------------------------------------- video
    def _video(self, img, d, vm: ViewModel, box: Rect) -> None:
        x0, y0, x1, y1 = box
        bw, bh = x1 - x0, y1 - y0
        if vm.frame is None:
            d.rounded_rectangle(box, 18, fill=T.CARD, outline=T.CARD_BORDER)
            msg = vm.status
            f = T.font(20)
            d.text(((x0 + x1) // 2, (y0 + y1) // 2), msg, font=f, fill=T.MUTED, anchor="mm")
            return
        fh, fw = vm.frame.shape[:2]
        scale = min(bw / fw, bh / fh)
        w, h = int(fw * scale), int(fh * scale)
        px, py = x0 + (bw - w) // 2, y0 + (bh - h) // 2
        rgb = cv2.cvtColor(cv2.resize(vm.frame, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        frame_img = Image.fromarray(rgb)
        fd = ImageDraw.Draw(frame_img)
        if vm.enrolling_name:
            self._enroll_overlay(frame_img, fd, vm, w, h)
        for face in vm.faces:
            self._face_box(fd, face, scale, enrolling=bool(vm.enrolling_name))
        img.paste(frame_img, (px, py), self._rounded_mask(w, h, 18))
        # status chip
        chip = f"●  {vm.status}" + (f"  ·  {vm.fps:.0f} FPS" if vm.status_ok else "")
        f = T.font(14, bold=True)
        cw = _text_w(d, chip, f) + 28
        d.rounded_rectangle((px + 16, py + 16, px + 16 + cw, py + 48), 16, fill=(0, 0, 0))
        d.text((px + 30, py + 32), "●", font=f, fill=T.KNOWN if vm.status_ok else T.WARN, anchor="lm")
        d.text((px + 30 + _text_w(d, "●  ", f), py + 32), chip[3:], font=f, fill=T.TEXT, anchor="lm")
        if vm.toast:
            f = T.font(16, bold=True)
            tw = _text_w(d, vm.toast, f) + 40
            cx = px + w // 2
            d.rounded_rectangle((cx - tw // 2, py + h - 64, cx + tw // 2, py + h - 24), 20,
                                fill=T.KNOWN if vm.toast_ok else (55, 60, 72))
            d.text((cx, py + h - 44), vm.toast, font=f, fill=(255, 255, 255), anchor="mm")

    def _rounded_mask(self, w: int, h: int, r: int) -> Image.Image:
        key = (w, h)
        if key not in self._mask_cache:
            m = Image.new("L", (w, h), 0)
            ImageDraw.Draw(m).rounded_rectangle((0, 0, w - 1, h - 1), r, fill=255)
            self._mask_cache = {key: m}
        return self._mask_cache[key]

    def _face_box(self, d, face: RecognizedFace, s: float, enrolling: bool) -> None:
        b = face.detection.box
        box = (int(b.x * s), int(b.y * s), int((b.x + b.w) * s), int((b.y + b.h) * s))
        colour = T.ACCENT if enrolling else (T.KNOWN if face.match.is_known else T.UNKNOWN)
        d.rounded_rectangle(box, 12, outline=colour, width=3)
        if enrolling:
            return
        f = T.font(15, bold=True)
        label = face.track_label
        lw = _text_w(d, label, f) + 24
        top = max(0, box[1] - 34)
        d.rounded_rectangle((box[0], top, box[0] + lw, top + 28), 14, fill=colour)
        d.text((box[0] + 12, top + 14), label, font=f, fill=(255, 255, 255), anchor="lm")

    def _enroll_overlay(self, frame_img, fd, vm: ViewModel, w: int, h: int) -> None:
        shade = Image.new("RGB", (w, 92), (0, 0, 0))
        frame_img.paste(Image.blend(frame_img.crop((0, h - 92, w, h)), shade, 0.6), (0, h - 92))
        p = vm.progress
        fd.text((24, h - 70), f"Adding {vm.enrolling_name}", font=T.font(20, bold=True), fill=T.TEXT)
        if p:
            fd.text((24, h - 40), p.message, font=T.font(15), fill=(210, 214, 222))
            fd.text((w - 24, h - 58), f"{p.captured} / {p.total}", font=T.font(22, bold=True),
                    fill=T.TEXT, anchor="rm")
            bar = (24, h - 12, w - 24, h - 6)
            fd.rounded_rectangle(bar, 3, fill=(60, 64, 74))
            fill_w = int((bar[2] - bar[0]) * p.fraction)
            if fill_w > 6:
                fd.rounded_rectangle((bar[0], bar[1], bar[0] + fill_w, bar[3]), 3, fill=T.ACCENT)

    # --------------------------------------------------------------- sidebar
    def _card(self, d, x: int, y0: int, y1: int) -> None:
        d.rounded_rectangle((x, y0, T.WIDTH - T.MARGIN, y1), 16, fill=T.CARD, outline=T.CARD_BORDER)

    def _header(self, d, vm: ViewModel, x: int, y: int) -> int:
        d.rounded_rectangle((x, y + 2, x + 36, y + 38), 10, fill=T.ACCENT)
        d.ellipse((x + 10, y + 12, x + 26, y + 28), outline=(255, 255, 255), width=3)
        d.text((x + 50, y + 20), "CharlieCam", font=T.font(24, bold=True), fill=T.TEXT, anchor="lm")
        return y + 40

    def _button(self, d, key: str, rect: Rect, text: str, enabled: bool, hover: bool,
                primary: bool = True) -> None:
        if primary:
            fill = (T.ACCENT_HOVER if hover else T.ACCENT) if enabled else T.DISABLED
            fg = (255, 255, 255) if enabled else T.FAINT
            d.rounded_rectangle(rect, 10, fill=fill)
        else:
            d.rounded_rectangle(rect, 10, fill=(44, 48, 58) if hover else T.CARD, outline=T.CARD_BORDER)
            fg = T.TEXT
        d.text(((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2), text,
               font=T.font(15, bold=True), fill=fg, anchor="mm")
        if enabled:
            self._region(key, rect)

    def _enroll_card(self, d, vm: ViewModel, x: int, y: int) -> int:
        x1 = T.WIDTH - T.MARGIN
        y1 = y + 186
        self._card(d, x, y, y1)
        pad = 20
        d.text((x + pad, y + 18), "Add a person", font=T.font(17, bold=True), fill=T.TEXT)
        d.text((x + pad, y + 42), "Type a name, then look at the camera.", font=T.font(13), fill=T.MUTED)
        field_rect = (x + pad, y + 70, x1 - pad, y + 114)
        enrolling = vm.enrolling_name is not None
        border = T.UNKNOWN if vm.name_error else (T.ACCENT if not enrolling else T.CARD_BORDER)
        d.rounded_rectangle(field_rect, 10, fill=T.FIELD, outline=border, width=2)
        self._region("field", field_rect)
        f = T.font(16)
        tx, ty = field_rect[0] + 14, (field_rect[1] + field_rect[3]) // 2
        if vm.name_text:
            shown = vm.name_text
            max_w = field_rect[2] - field_rect[0] - 30
            while _text_w(d, shown, f) > max_w:
                shown = shown[1:]
            d.text((tx, ty), shown, font=f, fill=T.TEXT, anchor="lm")
            caret_x = tx + _text_w(d, shown, f) + 2
        else:
            d.text((tx, ty), "Full name", font=f, fill=T.FAINT, anchor="lm")
            caret_x = tx
        if vm.caret_visible and not enrolling:
            d.line((caret_x, ty - 11, caret_x, ty + 11), fill=T.TEXT, width=2)
        btn = (x + pad, y + 126, x1 - pad, y + 168)
        if enrolling:
            self._button(d, "cancel", btn, "Cancel", True, vm.hover == "cancel", primary=False)
        else:
            self._button(d, "capture", btn, "Capture photos", vm.can_capture, vm.hover == "capture")
        if vm.name_error:
            d.text((x + pad, y1 + 4), vm.name_error, font=T.font(13), fill=T.UNKNOWN)
        return y1

    def _people_card(self, d, vm: ViewModel, x: int, y: int, y_max: int) -> int:
        x1 = T.WIDTH - T.MARGIN
        pad, row_h = 20, 44
        names = sorted(vm.people, key=str.lower)
        max_rows = max(1, (y_max - y - 60) // row_h)
        shown = names[:max_rows] if len(names) <= max_rows else names[:max_rows - 1]
        rows = max(1, len(shown) + (len(names) > len(shown)))
        y1 = min(y_max, y + 54 + rows * row_h + 6)
        self._card(d, x, y, y1)
        d.text((x + pad, y + 18), "People", font=T.font(17, bold=True), fill=T.TEXT)
        d.text((x1 - pad, y + 28), str(len(names)), font=T.font(14, bold=True), fill=T.MUTED, anchor="rm")
        ry = y + 54
        if not names:
            d.text((x + pad, ry + 12), "No one added yet", font=T.font(14), fill=T.FAINT)
        for name in shown:
            cy = ry + row_h // 2
            d.ellipse((x + pad, cy - 15, x + pad + 30, cy + 15), fill=T.avatar_colour(name))
            d.text((x + pad + 15, cy), name[0].upper(), font=T.font(14, bold=True),
                   fill=(255, 255, 255), anchor="mm")
            confirming = vm.confirm_remove == name
            key = f"remove:{name}"
            btn_w = 78 if confirming else 30
            btn = (x1 - pad - btn_w, cy - 14, x1 - pad, cy + 14)
            name_w = btn[0] - (x + pad + 44) - 8
            d.text((x + pad + 44, cy - 8), _ellipsize(d, name, T.font(15, bold=True), name_w),
                   font=T.font(15, bold=True), fill=T.TEXT, anchor="lm")
            n = vm.people[name]
            d.text((x + pad + 44, cy + 10), f"{n} photo{'s' if n != 1 else ''}", font=T.font(12),
                   fill=T.MUTED, anchor="lm")
            hovered = vm.hover == key
            if confirming:
                d.rounded_rectangle(btn, 8, fill=T.UNKNOWN)
                d.text(((btn[0] + btn[2]) // 2, cy), "Remove?", font=T.font(13, bold=True),
                       fill=(255, 255, 255), anchor="mm")
            else:
                if hovered:
                    d.rounded_rectangle(btn, 8, fill=(60, 30, 34))
                c = T.UNKNOWN if hovered else T.FAINT
                cx = (btn[0] + btn[2]) // 2
                d.line((cx - 5, cy - 5, cx + 5, cy + 5), fill=c, width=2)
                d.line((cx - 5, cy + 5, cx + 5, cy - 5), fill=c, width=2)
            self._region(key, btn)
            ry += row_h
        if len(names) > len(shown):
            d.text((x + pad, ry + 12), f"+ {len(names) - len(shown)} more", font=T.font(13), fill=T.MUTED)
        return y1

    def _activity_card(self, d, vm: ViewModel, x: int, y: int, y1: int) -> None:
        x1 = T.WIDTH - T.MARGIN
        pad, row_h = 20, 26
        if y1 - y < 70:
            return
        self._card(d, x, y, y1)
        d.text((x + pad, y + 16), "Recent activity", font=T.font(15, bold=True), fill=T.TEXT)
        ry = y + 46
        if not vm.activity:
            d.text((x + pad, ry), "Sightings will appear here", font=T.font(13), fill=T.FAINT)
        for item in vm.activity:
            if ry + row_h > y1 - 6:
                break
            cy = ry + 8
            d.ellipse((x + pad, cy - 4, x + pad + 8, cy + 4), fill=T.KNOWN if item.known else T.UNKNOWN)
            label = _ellipsize(d, item.label, T.font(13), 190)
            d.text((x + pad + 18, cy), label, font=T.font(13), fill=T.TEXT, anchor="lm")
            d.text((x1 - pad, cy), item.time, font=T.font(12), fill=T.MUTED, anchor="rm")
            ry += row_h
