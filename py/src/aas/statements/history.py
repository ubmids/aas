"""A listening history as columns, and a time window over it.

Every statement family reads a `Window`: the plays in [lo, hi), plus everything before lo as context
(so "an artist you first heard four weeks ago" can look back past the window's start). Nothing after
hi is ever read, which is what lets a statement made from the first half of someone's history be
re-checked on the second half without the first measurement having seen it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from ..person import Play

MIN_MS = 60_000
HOUR = 60 * MIN_MS
DAY = 24 * HOUR
WEEK = 7 * DAY
SEASON = 13 * WEEK
SESSION_GAP = 30 * MIN_MS


@dataclass
class History:
    """Plays sorted by start time. Artist, track and release are integer codes; release -1 = unknown."""
    t: np.ndarray        # start, ms since epoch, int64
    artist: np.ndarray   # int64 codes
    track: np.ndarray    # int64 codes (artist and title together)
    release: np.ndarray  # int64 codes, -1 where unknown
    n_artists: int
    first: np.ndarray = field(init=False)        # first start per artist code
    nth: np.ndarray = field(init=False)          # per play: how many earlier plays of this artist
    prev_artist_t: np.ndarray = field(init=False)  # per play: previous play of the same artist, -1 if none
    prev_track_i: np.ndarray = field(init=False)   # per play: index of the previous play of the same track, -1

    def __post_init__(self) -> None:
        n = len(self.t)
        self.first = np.full(self.n_artists, np.iinfo(np.int64).max, dtype=np.int64)
        np.minimum.at(self.first, self.artist, self.t)
        order = np.lexsort((np.arange(n), self.artist))
        a_sorted = self.artist[order]
        same = np.r_[False, a_sorted[1:] == a_sorted[:-1]]
        prev = np.full(n, -1, dtype=np.int64)
        prev[order[same]] = self.t[order[np.nonzero(same)[0] - 1]]
        self.prev_artist_t = prev
        # running count within artist: position minus the start index of the artist's block
        starts = np.nonzero(~same)[0]
        block = np.cumsum(~same) - 1
        nth = np.empty(n, dtype=np.int64)
        nth[order] = np.arange(n) - starts[block]
        self.nth = nth
        order = np.lexsort((np.arange(n), self.track))
        tr = self.track[order]
        same = np.r_[False, tr[1:] == tr[:-1]]
        pti = np.full(n, -1, dtype=np.int64)
        pti[order[same]] = order[np.nonzero(same)[0] - 1]
        self.prev_track_i = pti

    @property
    def t0(self) -> int:
        return int(self.t[0])

    @property
    def t1(self) -> int:
        return int(self.t[-1]) + 1

    def window(self, lo: int | None = None, hi: int | None = None) -> Window:
        lo = self.t0 if lo is None else int(lo)
        hi = self.t1 if hi is None else int(hi)
        return Window(self, lo, hi)

    def halves(self) -> tuple[Window, Window]:
        """Split at the midpoint in time (not in plays)."""
        mid = (self.t0 + self.t1) // 2
        return self.window(self.t0, mid), self.window(mid, self.t1)


def from_rows(rows: list[tuple[int, str, str, str | None]]) -> History:
    """(start_ms, artist_key, track_title, release_key or None) rows, any order."""
    rows = sorted(rows, key=lambda r: r[0])
    ac: dict[str, int] = {}
    tc: dict[tuple[int, str], int] = {}
    rc: dict[str, int] = {}
    art = np.empty(len(rows), dtype=np.int64)
    trk = np.empty(len(rows), dtype=np.int64)
    rel = np.empty(len(rows), dtype=np.int64)
    for i, (_, a, tr, r) in enumerate(rows):
        art[i] = ac.setdefault(a, len(ac))
        trk[i] = tc.setdefault((art[i], (tr or "").strip().lower()), len(tc))
        rel[i] = rc.setdefault(r, len(rc)) if r else -1
    t = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
    return History(t=t, artist=art, track=trk, release=rel, n_artists=len(ac))


def from_plays(plays: list[Play]) -> History:
    """From the engine's Play objects. Play.when is the end time when ms_played is known (Spotify)."""
    rows = []
    for p in plays:
        d = p.when if p.when.tzinfo else p.when.replace(tzinfo=UTC)
        end = int(round(d.timestamp() * 1000))
        start = end - int(p.ms_played) if p.ms_played is not None else end
        rows.append((start, p.artist.strip().lower(), p.track, None))
    return from_rows(rows)


class Window:
    """Plays in [lo, hi) of one history; earlier plays are context, later plays are invisible."""

    def __init__(self, h: History, lo: int, hi: int) -> None:
        self.h, self.lo, self.hi = h, lo, hi
        self.i0 = int(np.searchsorted(h.t, lo, "left"))
        self.i1 = int(np.searchsorted(h.t, hi, "left"))
        self._cache: dict = {}

    def __len__(self) -> int:
        return self.i1 - self.i0

    @property
    def days(self) -> float:
        return (self.hi - self.lo) / DAY

    @property
    def t(self) -> np.ndarray:
        return self.h.t[self.i0:self.i1]

    @property
    def artist(self) -> np.ndarray:
        return self.h.artist[self.i0:self.i1]

    @property
    def track(self) -> np.ndarray:
        return self.h.track[self.i0:self.i1]

    @property
    def release(self) -> np.ndarray:
        return self.h.release[self.i0:self.i1]

    def quarters(self) -> list[Window]:
        step = (self.hi - self.lo) / 4
        return [Window(self.h, int(self.lo + k * step), int(self.lo + (k + 1) * step)) for k in range(4)]

    # ------------------------------------------------------------------ derived, cached

    def centre_minute(self) -> float:
        """Circular mean minute of the day (UTC) of the window's plays."""
        if "centre" not in self._cache:
            m = np.mod(self.t // MIN_MS, 1440)
            ang = 2 * np.pi * m / 1440
            c = math.atan2(float(np.sin(ang).sum()), float(np.cos(ang).sum())) if len(m) else 0.0
            self._cache["centre"] = (c / (2 * np.pi) * 1440) % 1440
        return self._cache["centre"]

    def day(self) -> np.ndarray:
        """Person-day of each play: days start 12 hours from the window's centre of listening, so a
        night owl's 1 am play belongs to the evening before. Time-zone free."""
        if "day" not in self._cache:
            start = (self.centre_minute() + 720) % 1440
            self._cache["day"] = np.floor((self.t / MIN_MS - start) / 1440).astype(np.int64)
        return self._cache["day"]

    def day_range(self) -> tuple[int, int]:
        start = (self.centre_minute() + 720) % 1440
        a = math.floor((self.lo / MIN_MS - start) / 1440)
        b = math.floor(((self.hi - 1) / MIN_MS - start) / 1440)
        return a, b + 1

    def session(self) -> np.ndarray:
        """Session id per play: a new session after a gap of more than 30 minutes (inside the window)."""
        if "session" not in self._cache:
            t = self.t
            brk = np.r_[True, np.diff(t) > SESSION_GAP]
            self._cache["session"] = np.cumsum(brk) - 1
        return self._cache["session"]

    def seasons(self) -> list[tuple[int, int]]:
        """Index ranges (into the window) of the full 13-week seasons from lo."""
        out, s = [], self.lo
        while s + SEASON <= self.hi:
            a = int(np.searchsorted(self.t, s, "left"))
            b = int(np.searchsorted(self.t, s + SEASON, "left"))
            out.append((a, b))
            s += SEASON
        return out

    def weeks(self) -> np.ndarray:
        """Play counts per full week from lo (zeros included)."""
        nw = int((self.hi - self.lo) // WEEK)
        if nw <= 0:
            return np.zeros(0, dtype=np.int64)
        k = (self.t - self.lo) // WEEK
        k = k[k < nw]
        return np.bincount(k, minlength=nw)


def ms_to_date(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=UTC).strftime("%b %Y")
