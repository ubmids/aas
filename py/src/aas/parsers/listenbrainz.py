"""ListenBrainz listen export.

ListenBrainz (listenbrainz.org/settings/export) has shipped two shapes:
  zip of JSON Lines   listens/<year>/<month>.jsonl, one listen per line (2024 onwards)
  one JSON array      listens.json or <user>_lb-<date>.json (older exports)
Each listen is {"listened_at": unix seconds, "track_metadata": {"artist_name", "track_name",
"release_name", "additional_info": {"artist_mbids": [...], ...}}}. Reads every listen with an artist and
a track into Play (start time, no length). Ignores feedback, pins, playlists and everything else.

Not registered in aas.parsers.PARSERS (detection by file name alone would be ambiguous with other
JSON exports); scripts/my_card.py calls it directly. A bare .json or .jsonl file works too.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from aas.parsers._bundle import Bundle, sort_key
from aas.person import Person, Play

SOURCE = "listenbrainz"
_FILES = re.compile(r".*\.jsonl?", re.I)


def _rows(text: str) -> list[dict]:
    text = text.strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        return [r for r in data if isinstance(r, dict)]
    if text[0] == "{" and "\n" not in text:
        r = json.loads(text)
        return r.get("payload", {}).get("listens", [r]) if isinstance(r, dict) else []
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def is_listen(r: dict) -> bool:
    return isinstance(r, dict) and "listened_at" in r and isinstance(r.get("track_metadata"), dict)


def _play(r: dict) -> Play | None:
    md = r["track_metadata"]
    artist, track = md.get("artist_name"), md.get("track_name")
    try:
        when = datetime.fromtimestamp(int(r["listened_at"]), UTC)
    except (TypeError, ValueError, OverflowError):
        return None
    if not artist or not track:
        return None
    info = md.get("additional_info") or {}
    mbids = info.get("artist_mbids") or []
    # one MusicBrainz artist id is a better key than the credit string (as the bench does)
    key = f"mbid:{mbids[0]}" if isinstance(mbids, list) and len(mbids) == 1 else artist
    return Play(when=when, artist=key, track=track, ms_played=None, source=SOURCE,
                album=md.get("release_name") or None)


def parse(path: str | Path) -> Person:
    path = Path(path).expanduser()
    p = Person(sources=[SOURCE])
    texts: list[str] = []
    if path.is_file() and path.suffix.lower() in (".json", ".jsonl"):
        texts.append(path.read_text(encoding="utf-8-sig", errors="replace"))
    else:
        with Bundle(path) as b:
            for f in b.find(_FILES):
                texts.append(b.read_text(f))
    bad = 0
    for t in texts:
        try:
            rows = _rows(t)
        except json.JSONDecodeError:
            continue
        for r in rows:
            if not is_listen(r):
                continue
            pl = _play(r)
            if pl is None:
                bad += 1
            else:
                p.plays.append(pl)
    p.plays.sort(key=sort_key)
    p.notes.append(f"listenbrainz: {len(p.plays):,} listens (start times, no play lengths or skips)")
    if bad:
        p.notes.append(f"listenbrainz: skipped {bad} listens without an artist, a track or a time")
    return p


def matches_path(path: str | Path) -> bool:
    """True when the path holds at least one ListenBrainz listen (reads up to the first listen file)."""
    try:
        q = parse(path)
    except (ValueError, FileNotFoundError, OSError):
        return False
    return bool(q.plays)
