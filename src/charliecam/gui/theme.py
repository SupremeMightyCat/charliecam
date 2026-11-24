"""Colours, sizes and fonts for the GUI."""
from __future__ import annotations

from functools import lru_cache

from PIL import ImageFont

WIDTH, HEIGHT = 1280, 720
MARGIN = 24
SIDEBAR_W = 340

BG = (13, 15, 20)
CARD = (22, 25, 32)
CARD_BORDER = (38, 42, 52)
FIELD = (13, 15, 20)
TEXT = (236, 239, 244)
MUTED = (139, 147, 162)
FAINT = (88, 95, 110)
ACCENT = (59, 130, 246)
ACCENT_HOVER = (37, 99, 235)
DISABLED = (48, 53, 64)
KNOWN = (34, 197, 94)
UNKNOWN = (239, 68, 68)
WARN = (245, 158, 11)

AVATAR_COLOURS = [
    (99, 102, 241), (14, 165, 233), (16, 185, 129), (245, 158, 11),
    (236, 72, 153), (168, 85, 247), (20, 184, 166), (249, 115, 22),
]

_REGULAR = ["segoeui.ttf", "Inter-Regular.otf", "DejaVuSans.ttf", "Arial.ttf"]
_BOLD = ["segoeuisb.ttf", "seguisb.ttf", "Inter-SemiBold.otf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf"]
_SEARCH_DIRS = ["", "C:/Windows/Fonts/", "/usr/share/fonts/opentype/inter/",
                "/usr/share/fonts/truetype/dejavu/", "/Library/Fonts/", "/System/Library/Fonts/"]


@lru_cache(maxsize=None)
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for name in (_BOLD if bold else _REGULAR):
        for folder in _SEARCH_DIRS:
            try:
                return ImageFont.truetype(folder + name, size)
            except OSError:
                continue
    return ImageFont.load_default(size)


def avatar_colour(name: str):
    return AVATAR_COLOURS[sum(map(ord, name)) % len(AVATAR_COLOURS)]
