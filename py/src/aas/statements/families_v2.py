"""Round 2 statement families: the 34 of round 1 plus 23 more (57 in all), same rules.

Every new family is a share, a rate or a typical length measured on a window, with a fixed grid of
round thresholds and two sentences. None names an artist, a track or a release, none is a raw play
count, and none says anything about health, religion, sexuality or politics. Like round 1, the clock
families are time-zone free and nothing reads plays after the window's end.

The new families were written on 5 November 2026 after the round 1 results, to raise coverage: more
measures give more people a pattern at the edge of the population. They were chosen for what they
describe (catalogue age, variety, pace inside a session, the shape of a listening day, how often a
song or artist comes back), and they are tried on the 304 listeners of round 1 only.
"""

from __future__ import annotations

import numpy as np

from .families import FAMILIES, NAN, PCT, Family, _per_season, _runs, _scored_from
from .history import DAY, HOUR, MIN_MS, SESSION_GAP, History, Window

YEAR = 365 * DAY


def _track_first(h: History) -> np.ndarray:
    """First start per track code, cached on the history."""
    tf = h.__dict__.get("_track_first")
    if tf is None:
        tf = np.full(int(h.track.max()) + 1 if len(h.track) else 0, np.iinfo(np.int64).max, dtype=np.int64)
        np.minimum.at(tf, h.track, h.t)
        h.__dict__["_track_first"] = tf
    return tf


def _prev_track_t(w: Window) -> np.ndarray:
    """Per play in the window: start of the previous play of the same track (any time before), -1 if none."""
    h = w.h
    idx = np.arange(w.i0, w.i1)
    p = h.prev_track_i[idx]
    return np.where(p >= 0, h.t[np.maximum(p, 0)], -1)


# ------------------------------------------------------------------------------------- catalogue

def new_track_share(w: Window) -> float:
    a = _scored_from(w)
    sel = w.t >= a
    if sel.sum() < 300:
        return NAN
    tf = _track_first(w.h)
    return float(((w.t[sel] - tf[w.track[sel]]) < 28 * DAY).mean())


def old_songs(w: Window) -> float:
    """Share of plays (once a year of history is behind) of songs first played over a year before."""
    sel = w.t >= w.h.t0 + YEAR
    if sel.sum() < 300:
        return NAN
    tf = _track_first(w.h)
    return float(((w.t[sel] - tf[w.track[sel]]) >= YEAR).mean())


def long_known(w: Window) -> float:
    """Share of plays (once two years of history are behind) of artists first played over two years before."""
    sel = w.t >= w.h.t0 + 2 * YEAR
    if sel.sum() < 300:
        return NAN
    return float(((w.t[sel] - w.h.first[w.artist[sel]]) >= 2 * YEAR).mean())


def song_comebacks(w: Window) -> float:
    """Share of plays of a song played before but not in the 180 days before."""
    sel = w.t >= w.h.t0 + 180 * DAY
    if sel.sum() < 300:
        return NAN
    pt = _prev_track_t(w)[sel]
    return float(((pt >= 0) & (w.t[sel] - pt >= 180 * DAY)).mean())


def first_day_binge(w: Window) -> float:
    """Share of new artists played five or more times on the day they were first played."""
    a = _scored_from(w)
    h = w.h
    sl = slice(w.i0, w.i1)
    t, art, nth = h.t[sl], h.artist[sl], h.nth[sl]
    new = (nth == 0) & (t >= a)
    if new.sum() < 30:
        return NAN
    # plays within 24 hours of the first play, counted per new artist
    ok = (t - h.first[art] < DAY)
    cnt = np.bincount(art[ok], minlength=h.n_artists)
    return float((cnt[art[new]] >= 5).mean())


# ------------------------------------------------------------------------------------- variety

def once_tracks(w: Window) -> float:
    def f(s: slice) -> float:
        _, c = np.unique(w.track[s], return_counts=True)
        return float((c == 1).mean())
    return _per_season(w, f)


def track_variety(w: Window) -> float:
    return _per_season(w, lambda s: len(np.unique(w.track[s])) / (s.stop - s.start))


def top10_tracks(w: Window) -> float:
    return _per_season(w, lambda s: np.sort(np.unique(w.track[s], return_counts=True)[1])[-10:].sum() / (s.stop - s.start))


def artist_tail(w: Window) -> float:
    def f(s: slice) -> float:
        _, inv, c = np.unique(w.artist[s], return_inverse=True, return_counts=True)
        return float((c[inv] <= 2).mean())
    return _per_season(w, f)


def artists_per_season(w: Window) -> float:
    return _per_season(w, lambda s: float(len(np.unique(w.artist[s]))))


# ------------------------------------------------------------------------------------- returns

def song_return_days(w: Window) -> float:
    """Median days between a play of a song and its previous play, over gaps of a day or more."""
    if len(w) < 1000:
        return NAN
    pt = _prev_track_t(w)
    g = (w.t - pt)[(pt >= 0)]
    g = g[g >= DAY]
    if len(g) < 100:
        return NAN
    return float(np.median(g) / DAY)


def artist_return_days(w: Window) -> float:
    """Median days between a play of an artist and its previous play, over gaps of a day or more."""
    if len(w) < 1000:
        return NAN
    prev = w.h.prev_artist_t[w.i0:w.i1]
    g = (w.t - prev)[prev >= 0]
    g = g[g >= DAY]
    if len(g) < 100:
        return NAN
    return float(np.median(g) / DAY)


# ------------------------------------------------------------------------------------- pace

def _session_parts(w: Window) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = w.session()
    n = np.bincount(s)
    first = np.r_[0, np.cumsum(n)[:-1]]
    return s, n, first


def track_gap(w: Window) -> float:
    """Median minutes between consecutive plays inside a session (about the length of a track)."""
    if len(w) < 1000:
        return NAN
    g = np.diff(w.t)
    g = g[(g > 0) & (g <= SESSION_GAP)]
    if len(g) < 500:
        return NAN
    return float(np.median(g) / MIN_MS)


def session_plays(w: Window) -> float:
    if len(w) < 1000:
        return NAN
    _, n, _ = _session_parts(w)
    return float(np.median(n))


def single_play_sessions(w: Window) -> float:
    if len(w) < 1000:
        return NAN
    _, n, _ = _session_parts(w)
    return float((n == 1).mean())


def artists_per_session(w: Window) -> float:
    """Median number of different artists in a session of ten or more plays."""
    if len(w) < 1000:
        return NAN
    s, n, first = _session_parts(w)
    big = np.nonzero(n >= 10)[0]
    if len(big) < 30:
        return NAN
    pairs = np.unique(s * (w.h.n_artists + 1) + w.artist)
    per = np.bincount(pairs // (w.h.n_artists + 1), minlength=len(n))
    return float(np.median(per[big]))


def session_gap_hours(w: Window) -> float:
    """Median hours between the end of one session and the start of the next (gaps under 7 days)."""
    if len(w) < 1000:
        return NAN
    g = np.diff(w.t)
    g = g[(g > SESSION_GAP) & (g < 7 * DAY)]
    if len(g) < 100:
        return NAN
    return float(np.median(g) / HOUR)


# ------------------------------------------------------------------------------------- the day

def _days(w: Window) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = w.day()
    ud, inv, n = np.unique(d, return_inverse=True, return_counts=True)
    return ud, inv, n


def day_span(w: Window) -> float:
    """Median hours from the first to the last play of a day, on days with five or more plays."""
    if w.days < 60 or len(w) < 500:
        return NAN
    _, inv, n = _days(w)
    lo = np.full(len(n), np.iinfo(np.int64).max)
    hi = np.full(len(n), np.iinfo(np.int64).min)
    np.minimum.at(lo, inv, w.t)
    np.maximum.at(hi, inv, w.t)
    ok = n >= 5
    if ok.sum() < 30:
        return NAN
    return float(np.median((hi - lo)[ok]) / HOUR)


def active_hours(w: Window) -> float:
    """Median number of different hours (counted from the window's own day start) a day has plays in."""
    if w.days < 60 or len(w) < 500:
        return NAN
    d = w.day()
    start = (w.centre_minute() + 720) % 1440
    hr = np.floor(np.mod(w.t / MIN_MS - start, 1440) / 60).astype(np.int64)
    pairs = np.unique((d - d.min()) * 24 + hr)
    per = np.bincount(pairs // 24)
    per = per[per > 0]
    if len(per) < 30:
        return NAN
    return float(np.median(per))


def busy_days(w: Window) -> float:
    """Share of listening days with at least three times the person's median day."""
    if w.days < 60 or len(w) < 500:
        return NAN
    _, _, n = _days(w)
    if len(n) < 30:
        return NAN
    return float((n >= 3 * np.median(n)).mean())


def longest_break(w: Window) -> float:
    """Per season: the longest gap with no plays, in days. Averaged."""
    def f(s: slice) -> float:
        t = w.t[s]
        return float(np.diff(t).max() / DAY) if len(t) > 1 else NAN
    return _per_season(w, f)


def same_busiest_day(w: Window) -> float:
    """Share of weeks (20+ plays) whose busiest day falls on the window's usual busiest day of the
    seven-day cycle (person-days, so no weekday names and no time zone)."""
    if w.days < 84 or len(w) < 500:
        return NAN
    d = w.day()
    wd = np.mod(d, 7)
    usual = int(np.argmax(np.bincount(wd, minlength=7)))
    wk = (d - d.min()) // 7
    hits = tot = 0
    for k in np.unique(wk):
        m = wk == k
        if m.sum() < 20:
            continue
        tot += 1
        hits += int(np.argmax(np.bincount(wd[m], minlength=7)) == usual)
    return hits / tot if tot >= 12 else NAN


# ------------------------------------------------------------------------------------- albums

def album_sittings(w: Window) -> float:
    """Share of plays in runs of eight or more from the same release inside one session."""
    r = w.release
    if len(w) < 1000 or (r >= 0).mean() < 0.5:
        return NAN
    brk = np.r_[True, np.diff(w.t) > SESSION_GAP] | (r < 0)
    runs = _runs(np.where(r >= 0, r, -1 - np.arange(len(r))), brk)
    return float(runs[runs >= 8].sum() / runs.sum())


NEW: list[Family] = [
    Family("new_track_share", "catalogue", new_track_share, tuple(PCT), "pct",
           "At least {th} of your plays are songs you first played in the four weeks before.",
           "No more than {th} of your plays are songs you first played in the four weeks before."),
    Family("old_songs", "catalogue", old_songs, tuple(PCT), "pct",
           "At least {th} of your plays are songs you first played more than a year before.",
           "No more than {th} of your plays are songs you first played more than a year before."),
    Family("long_known", "catalogue", long_known, tuple(PCT), "pct",
           "At least {th} of your plays go to artists you have been playing for more than two years.",
           "No more than {th} of your plays go to artists you have been playing for more than two years."),
    Family("song_comebacks", "catalogue", song_comebacks, tuple(PCT), "pct",
           "At least {th} of your plays are songs you come back to after leaving them alone for six months or more.",
           "No more than {th} of your plays are songs you come back to after six months or more away."),
    Family("first_day_binge", "discovery", first_day_binge, tuple(PCT), "pct",
           "When you find a new artist you dive in: at least {th} of them get five or more plays on the first day.",
           "You try new artists lightly: no more than {th} of them get five or more plays on the first day."),
    Family("once_tracks", "variety", once_tracks, tuple(PCT), "pct",
           "At least {th} of the songs you play in a season get played only once that season.",
           "No more than {th} of the songs you play in a season get played only once that season."),
    Family("track_variety", "variety", track_variety, tuple(PCT), "pct",
           "In a season, different songs make up at least {th} of your plays, so you rarely repeat.",
           "In a season, different songs make up no more than {th} of your plays, so you repeat a lot."),
    Family("top10_tracks", "variety", top10_tracks, tuple(PCT), "pct",
           "In a typical season your ten most played songs take at least {th} of your plays.",
           "In a typical season your ten most played songs take no more than {th} of your plays."),
    Family("artist_tail", "variety", artist_tail, tuple(PCT), "pct",
           "At least {th} of a season's plays go to artists you play only once or twice that season.",
           "No more than {th} of a season's plays go to artists you play only once or twice that season."),
    Family("artists_per_season", "variety", artists_per_season,
           (10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000), "num",
           "You play at least {th} different artists in a typical season.",
           "You play no more than {th} different artists in a typical season.",
           per="artists a season"),
    Family("song_return_days", "returns", song_return_days,
           (1.5, 2, 3, 4, 5, 7, 10, 14, 21, 30, 45, 60, 90, 120, 180, 270, 365), "days",
           "When you come back to a song, it is typically after {th} or more away.",
           "When you come back to a song, it is typically within {th}."),
    Family("artist_return_days", "returns", artist_return_days,
           (1.5, 2, 3, 4, 5, 7, 10, 14, 21, 30, 45, 60, 90, 120, 180), "days",
           "When you come back to an artist, it is typically after {th} or more away.",
           "When you come back to an artist, it is typically within {th}."),
    Family("track_gap", "pace", track_gap, (2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 5, 5.5, 6, 7, 8, 10, 12, 15), "x",
           "Your songs run long: inside a session a new play starts {th} minutes or more after the one before.",
           "Your songs run short: inside a session a new play starts within {th} minutes of the one before.",
           per="minutes between starts"),
    Family("session_plays", "pace", session_plays, (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100), "num",
           "A typical session of yours holds {th} or more plays.",
           "A typical session of yours holds {th} plays or fewer.",
           per="plays a session"),
    Family("single_play_sessions", "pace", single_play_sessions, tuple(PCT), "pct",
           "At least {th} of your sessions are a single song and nothing else.",
           "No more than {th} of your sessions are a single song and nothing else."),
    Family("artists_per_session", "pace", artists_per_session,
           (1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40), "num",
           "A long session of yours (ten plays or more) takes in {th} or more different artists.",
           "A long session of yours (ten plays or more) stays with {th} artists or fewer.",
           per="artists a session"),
    Family("session_gap_hours", "pace", session_gap_hours, (0.75, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 36, 48), "h",
           "Between one session and the next there are typically {th} or more.",
           "Between one session and the next there are typically no more than {th}."),
    Family("day_span", "day", day_span, (0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20), "h",
           "Your listening days are long: a typical day runs {th} or more from first play to last.",
           "Your listening days are short: a typical day runs {th} or less from first play to last."),
    Family("active_hours", "day", active_hours, (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20), "num",
           "On a typical listening day you play music in {th} or more different hours.",
           "On a typical listening day you play music in {th} different hours or fewer.",
           per="hours a day"),
    Family("busy_days", "day", busy_days, tuple(PCT), "pct",
           "At least {th} of your listening days are big ones, three times your usual day or more.",
           "No more than {th} of your listening days are big ones, three times your usual day or more."),
    Family("longest_break", "rhythm", longest_break, (1, 1.5, 2, 3, 4, 5, 7, 10, 14, 21, 30, 45, 60), "days",
           "In a typical season your longest break from music lasts {th} or more.",
           "In a typical season your longest break from music lasts {th} or less."),
    Family("same_busiest_day", "day", same_busiest_day, tuple(PCT), "pct",
           "Your week has a fixed high point: in at least {th} of weeks your busiest day is the same day of the week.",
           "Your busiest day moves around: in no more than {th} of weeks is it your usual busiest day of the week."),
    Family("album_sittings", "repetition", album_sittings, tuple(PCT), "pct",
           "At least {th} of your plays come in sittings of eight or more from one release.",
           "No more than {th} of your plays come in sittings of eight or more from one release."),
]

FAMILIES_V2: list[Family] = FAMILIES + NEW
BY_KEY_V2 = {f.key: f for f in FAMILIES_V2}


def measure_v2(w: Window, families: list[Family] = FAMILIES_V2) -> np.ndarray:
    out = np.full(len(families), NAN)
    if len(w) == 0:
        return out
    for k, f in enumerate(families):
        out[k] = f.compute(w)
    return out

