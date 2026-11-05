"""Spotify listening history.

Two different exports exist (spotify.com/account/privacy):
  Extended streaming history  Streaming_History_Audio_*.json (older name: endsong_*.json)
      ts, ms_played, master_metadata_track_name, master_metadata_album_artist_name,
      reason_start, reason_end, skipped, shuffle. Whole account lifetime.
  Account data (basic)        StreamingHistory0.json / StreamingHistory_music_0.json
      endTime, artistName, trackName, msPlayed. Last ~12 months, no skips/reasons/shuffle.
Reads   music plays -> Play. If both are present, only the extended one is used.
Ignores podcast/audiobook rows (no track name), StreamingHistory_podcast_*, video
        history, playlists, search queries, library, inferences, everything else.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from aas.parsers._bundle import Bundle, parse_time, sort_key
from aas.person import Person, Play

SOURCE = "spotify"
_EXTENDED = re.compile(r"(Streaming_History_Audio_.*|endsong_\d+)\.json", re.I)
_BASIC = re.compile(r"StreamingHistory(_music_)?\d+\.json", re.I)


def matches(b: Bundle) -> bool:
    return bool(b.find(_EXTENDED) or b.find(_BASIC))


def _bool(v) -> bool | None:
    if v is None:
        return None
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1")
    return bool(v)


def parse(path: str | Path) -> Person:
    p = Person(sources=[SOURCE])
    non_music = 0
    with Bundle(path) as b:
        ext, basic = b.find(_EXTENDED), b.find(_BASIC)
        if ext:
            for f in ext:
                for r in json.loads(b.read_text(f)):
                    track = r.get("master_metadata_track_name")
                    when = parse_time(r.get("ts"))
                    if not track or when is None:       # podcast episode, audiobook, or junk
                        non_music += 1
                        continue
                    p.plays.append(Play(
                        when=when,
                        artist=r.get("master_metadata_album_artist_name") or "",
                        track=track,
                        ms_played=r.get("ms_played"),
                        skipped=_bool(r.get("skipped")),
                        reason_start=r.get("reason_start"),
                        reason_end=r.get("reason_end"),
                        shuffle=_bool(r.get("shuffle")),
                        country=r.get("conn_country"),
                    ))
            p.notes.append("spotify: extended streaming history (whole account lifetime, "
                           "with skips, start/end reasons and shuffle)")
            if basic:
                p.notes.append("spotify: basic StreamingHistory files also present; ignored "
                               "because the extended history covers them")
        elif basic:
            for f in basic:
                for r in json.loads(b.read_text(f)):
                    track, when = r.get("trackName"), parse_time(r.get("endTime"))
                    if not track or when is None:
                        non_music += 1
                        continue
                    p.plays.append(Play(when=when, artist=r.get("artistName") or "",
                                        track=track, ms_played=r.get("msPlayed")))
            p.notes.append("spotify: basic account-data history only (about the last 12 months; "
                           "no skips, start/end reasons or shuffle, so skip-based measures "
                           "can't be computed). Request the Extended streaming history for more.")
        else:
            p.notes.append("spotify: no streaming history files found")
    p.plays.sort(key=sort_key)
    if non_music:
        p.notes.append(f"spotify: skipped {non_music} podcast/audiobook/undated rows")
    return p
