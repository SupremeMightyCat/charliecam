"""Attendance: who was seen on each day, built from the event log.

The event log (``events.jsonl``) is the single source of truth. Attendance is a
read-only view of it, so there is no second store that could drift out of sync,
and past meetings can be re-read at any time.
"""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional


@dataclass
class Attendee:
    name: str
    first_seen: datetime
    last_seen: datetime
    sightings: int

    def to_dict(self) -> Dict:
        return {"name": self.name, "first_seen": self.first_seen.isoformat(),
                "last_seen": self.last_seen.isoformat(), "sightings": self.sightings}


def read_events(path: Path) -> Iterator[Dict]:
    """Yield well-formed events, skipping lines that are not valid JSON events."""
    path = Path(path)
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                ev = json.loads(line)
                ev["_local"] = datetime.fromisoformat(ev["timestamp"]).astimezone()
                yield ev
            except (ValueError, KeyError, TypeError):
                continue


def meeting_days(events: Iterable[Dict]) -> List[date]:
    """Days (local time) on which at least one known person was seen, newest first."""
    return sorted({ev["_local"].date() for ev in events if ev.get("known")}, reverse=True)


def attendance(events: Iterable[Dict], day: date) -> List[Attendee]:
    """Known people seen on ``day`` (local time), in order of arrival."""
    people: Dict[str, Attendee] = {}
    for ev in events:
        seen = ev["_local"]
        if not ev.get("known") or seen.date() != day:
            continue
        a = people.get(ev["label"])
        if a is None:
            people[ev["label"]] = Attendee(ev["label"], seen, seen, 1)
        else:
            a.first_seen, a.last_seen = min(a.first_seen, seen), max(a.last_seen, seen)
            a.sightings += 1
    return sorted(people.values(), key=lambda a: a.first_seen)


def report(events_path: Path, enrolled: Iterable[str], day: Optional[date] = None) -> Dict:
    """Attendance for one day plus who was absent and which days have records."""
    events = list(read_events(events_path))
    day = day or date.today()
    present = attendance(events, day)
    seen = {a.name for a in present}
    return {
        "date": day.isoformat(),
        "days": [d.isoformat() for d in meeting_days(events)],
        "present": [a.to_dict() for a in present],
        "absent": sorted((n for n in enrolled if n not in seen), key=str.lower),
    }


def to_csv(rep: Dict) -> str:
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["date", "name", "status", "first_seen", "last_seen", "sightings"])
    for a in rep["present"]:
        first = datetime.fromisoformat(a["first_seen"]).strftime("%H:%M:%S")
        last = datetime.fromisoformat(a["last_seen"]).strftime("%H:%M:%S")
        writer.writerow([rep["date"], a["name"], "present", first, last, a["sightings"]])
    for name in rep["absent"]:
        writer.writerow([rep["date"], name, "absent", "", "", 0])
    return out.getvalue()
