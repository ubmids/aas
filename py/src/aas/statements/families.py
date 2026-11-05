"""Statement families: behavioural measures of a listening history, each with a threshold grid and
the sentence it becomes.

A family is a number measured on a window (a share, a rate, a typical length), never a raw count of
plays and never an artist's name, so a statement is always a pattern with a threshold: "at least
60% of your plays go to ..." is true or false of anyone. Measures that grow with the length of the
window (the longest streak ever, the number of artists ever heard) are taken per 13-week season and
averaged, or per 30 days, so a person with ten years of history is compared fairly with one who has
one. Clock measures are time-zone free (ListenBrainz has no time zone): they use the shape of the
day and the week, never the hour on the clock or the name of the day.

Nothing here infers anything about health, religion, sexuality or politics; every family is about
when and how often music is played.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .history import DAY, HOUR, MIN_MS, SEASON, SESSION_GAP, WEEK, Window

NAN = math.nan
WARM = 12 * WEEK   # discovery needs 12 weeks of history to know who you already knew

PCT = [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
       0.7, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.97, 0.98, 0.99]


@dataclass(frozen=True)
class Family:
    key: str
    group: str
    compute: Callable[[Window], float]
    grid: tuple[float, ...]
    unit: str            # "pct", "num", "min", "h", "days", "x"
    hi: str              # sentence when value >= threshold; {th} is the threshold
    lo: str              # sentence when value <= threshold
    dirs: tuple[str, ...] = ("hi", "lo")
    per: str = ""        # what a bare number counts, for showing a person's own value ("sessions a day")

    def fmt(self, x: float) -> str:
        return fmt(x, self.unit)

    def show(self, x: float) -> str:
        """A person's own value for a card: always with its unit, rounded for reading."""
        return show(x, self.unit, self.per)

    def text(self, direction: str, th: float) -> str:
        return (self.hi if direction == "hi" else self.lo).format(th=self.fmt(th))


def fmt(x: float, unit: str) -> str:
    if not math.isfinite(x):
        return "n/a"
    if unit == "pct":
        v = 100 * x
        return f"{v:.1f}%" if abs(v - round(v)) > 1e-9 and v < 10 else f"{round(v)}%"
    if unit == "min":
        return f"{x:.0f} minutes" if x < 120 else f"{x / 60:.1f} hours".replace(".0 ", " ")
    if unit == "h":
        v = f"{x:g}" if x == round(x, 1) else f"{x:.1f}"
        return v + (" hour" if v == "1" else " hours")
    if unit == "days":
        v = f"{x:g}" if x == round(x, 1) else f"{x:.1f}"
        return v + (" day" if v == "1" else " days")
    if unit == "x":
        return f"{x:g}" if x == round(x, 2) else f"{x:.2f}"
    if x >= 10:
        return f"{round(x):,}"
    return f"{x:g}" if x == round(x, 2) else f"{x:.1f}"


def show(x: float, unit: str, per: str = "") -> str:
    """Like fmt, for a measured value rather than a round threshold: ratios get one decimal, and a
    bare number carries what it counts (per), so no card prints a figure without its unit."""
    if not math.isfinite(x):
        return "n/a"
    if unit == "pct":
        v = 100 * x
        return f"{v:.1f}%" if v < 10 and round(v, 1) != round(v) else f"{round(v)}%"
    if unit == "min":
        return f"{x:.0f} minutes" if x < 120 else f"{x / 60:.1f} hours"
    if unit in ("h", "days"):
        v = f"{x:.1f}" if x < (100 if unit == "h" else 10) else f"{round(x):,}"
        word = {"h": "hours", "days": "days"}[unit]
        return f"{v} {word[:-1] if v == '1' else word}"
    if unit == "x":
        v = f"{x:.1f}" if x < 100 else f"{round(x):,}"
    elif x >= 10:
        v = f"{round(x):,}"
    else:
        v = f"{x:.1f}" if round(x, 1) != round(x) else f"{round(x)}"
    return f"{v} {per}".strip()


# ------------------------------------------------------------------------------------- helpers

def _scored_from(w: Window, warm: int = WARM) -> int:
    return max(w.lo, w.h.t0 + warm)


def _new_artists(w: Window, until: int | None = None) -> np.ndarray:
    """Codes of artists first heard inside the window, after the warm-up, before `until`."""
    a = _scored_from(w)
    b = w.hi if until is None else until
    f = w.h.first
    return np.nonzero((f >= a) & (f < b))[0]


def _top(x: np.ndarray, k: int) -> np.ndarray:
    if not len(x):
        return x
    u, c = np.unique(x, return_counts=True)
    return u[np.argsort(-c, kind="stable")[:k]]


def _season_slices(w: Window, min_plays: int = 200) -> list[slice]:
    return [slice(a, b) for a, b in w.seasons() if b - a >= min_plays]


def _runs(x: np.ndarray, brk: np.ndarray) -> np.ndarray:
    """Run lengths of equal consecutive values, also broken where brk is True."""
    if not len(x):
        return np.zeros(0, dtype=np.int64)
    start = np.r_[True, (x[1:] != x[:-1]) | brk[1:]]
    idx = np.nonzero(start)[0]
    return np.diff(np.r_[idx, len(x)])


# ------------------------------------------------------------------------------------- discovery

def new_share(w: Window) -> float:
    a = _scored_from(w)
    sel = w.t >= a
    if sel.sum() < 300:
        return NAN
    art, t = w.artist[sel], w.t[sel]
    return float(((t - w.h.first[art]) < 28 * DAY).mean())


def new_per_month(w: Window) -> float:
    a = _scored_from(w)
    span = (w.hi - a) / DAY
    if span < 60 or len(w) < 300:
        return NAN
    return len(_new_artists(w)) / (span / 30)


def one_and_done(w: Window) -> float:
    new = _new_artists(w, until=w.hi - 30 * DAY)
    if len(new) < 30:
        return NAN
    cnt = np.bincount(w.artist, minlength=w.h.n_artists)[new]
    return float((cnt == 1).mean())


def stick_rate(w: Window) -> float:
    new = _new_artists(w, until=w.hi - 90 * DAY)
    if len(new) < 30:
        return NAN
    isnew = np.zeros(w.h.n_artists, dtype=bool)
    isnew[new] = True
    art, t = w.artist, w.t
    sel = isnew[art] & ((t - w.h.first[art]) < 90 * DAY)
    cnt = np.bincount(art[sel], minlength=w.h.n_artists)[new]
    return float((cnt >= 10).mean())


def discovery_bursts(w: Window) -> float:
    a = _scored_from(w)
    new = _new_artists(w)
    nw = int((w.hi - a) // WEEK)
    if len(new) < 30 or nw < 20:
        return NAN
    k = (w.h.first[new] - a) // WEEK
    k = k[k < nw]
    c = np.sort(np.bincount(k, minlength=nw))[::-1]
    top = max(1, math.ceil(0.1 * nw))
    return float(c[:top].sum() / max(c.sum(), 1))


def comebacks_per_month(w: Window) -> float:
    a = max(w.lo, w.h.t0 + 365 * DAY)
    span = (w.hi - a) / DAY
    if span < 90 or len(w) < 300:
        return NAN
    h = w.h
    sl = slice(w.i0, w.i1)
    t, prev, nth = h.t[sl], h.prev_artist_t[sl], h.nth[sl]
    ev = (t >= a) & (prev >= 0) & (t - prev >= 365 * DAY) & (nth >= 5)
    return float(ev.sum() / (span / 30))


# ------------------------------------------------------------------------------------- loyalty

def old_friends(w: Window) -> float:
    h, shares = w.h, []
    s = w.lo
    while s + 8 * WEEK <= w.hi:
        if s - 16 * WEEK >= h.t0:
            i, j = np.searchsorted(h.t, [s, s + 8 * WEEK])
            if j - i > 100:
                p2 = np.searchsorted(h.t, [s - 16 * WEEK, s - 8 * WEEK])
                p3 = np.searchsorted(h.t, [s - 24 * WEEK, s - 16 * WEEK])
                old = np.union1d(_top(h.artist[p2[0]:p2[1]], 50), _top(h.artist[p3[0]:p3[1]], 50))
                shares.append(float(np.isin(h.artist[i:j], old).mean()))
        s += 8 * WEEK
    return float(np.mean(shares)) if shares else NAN


def _season_tops(w: Window, k: int, back: int = 1) -> list[tuple[np.ndarray, np.ndarray]]:
    """(previous season's top k, this season's top k) for each full season in the window whose
    previous season (possibly before the window) is inside the history."""
    h, out = w.h, []
    s = w.lo
    while s + SEASON <= w.hi:
        if s - SEASON >= h.t0:
            i, j = np.searchsorted(h.t, [s, s + SEASON])
            p, q = np.searchsorted(h.t, [s - SEASON, s])
            out.append((_top(h.artist[p:q], k), _top(h.artist[i:j], k)))
        s += SEASON
    return out


def season_turnover(w: Window) -> float:
    pairs = [(a, b) for a, b in _season_tops(w, 10) if len(a) == 10 and len(b) == 10]
    if not pairs:
        return NAN
    return float(np.mean([1 - len(np.intersect1d(a, b)) / 10 for a, b in pairs]))


def same_top(w: Window) -> float:
    pairs = [(a, b) for a, b in _season_tops(w, 1) if len(a) and len(b)]
    if len(pairs) < 2:
        return NAN
    return float(np.mean([a[0] == b[0] for a, b in pairs]))


def dropped_cold(w: Window) -> float:
    """Share of each season's top 20 not played at all in the next season (both inside the window)."""
    h, shares = w.h, []
    s = w.lo
    while s + 2 * SEASON <= w.hi:
        i, j, k = np.searchsorted(h.t, [s, s + SEASON, s + 2 * SEASON])
        top = _top(h.artist[i:j], 20)
        if len(top) == 20:
            shares.append(1 - np.isin(top, h.artist[j:k]).mean())
        s += SEASON
    return float(np.mean(shares)) if shares else NAN


# ------------------------------------------------------------------------------------- concentration

def _per_season(w: Window, f: Callable[[slice], float], min_plays: int = 200) -> float:
    vals = [f(s) for s in _season_slices(w, min_plays)]
    vals = [v for v in vals if math.isfinite(v)]
    return float(np.mean(vals)) if vals else NAN


def top1_share(w: Window) -> float:
    return _per_season(w, lambda s: np.bincount(w.artist[s]).max() / (s.stop - s.start))


def top5_share(w: Window) -> float:
    return _per_season(w, lambda s: np.sort(np.bincount(w.artist[s]))[-5:].sum() / (s.stop - s.start))


def effective_artists(w: Window) -> float:
    def f(s: slice) -> float:
        c = np.bincount(w.artist[s])
        p = c[c > 0] / c.sum()
        return float(np.exp(-(p * np.log(p)).sum()))
    return _per_season(w, f)


def weekly_one_artist(w: Window) -> float:
    k = (w.t - w.lo) // WEEK
    vals = []
    for wk in np.unique(k):
        a = w.artist[k == wk]
        if len(a) >= 20:
            vals.append(np.bincount(a).max() / len(a))
    return float(np.median(vals)) if len(vals) >= 8 else NAN


def artist_days(w: Window) -> float:
    d = w.day()
    order = np.lexsort((w.artist, d))
    dd, aa = d[order], w.artist[order]
    ud, start = np.unique(dd, return_index=True)
    n = np.diff(np.r_[start, len(dd)])
    hits = 0
    big = 0
    for s, m in zip(start, n):
        if m < 10:
            continue
        big += 1
        run = _runs(aa[s:s + m], np.zeros(m, dtype=bool))
        if run.max() > m / 2:
            hits += 1
    return hits / big if big >= 20 else NAN


# ------------------------------------------------------------------------------------- repetition

def back_to_back(w: Window) -> float:
    if len(w) < 1000:
        return NAN
    t, tr = w.t, w.track
    same = (tr[1:] == tr[:-1]) & (np.diff(t) <= SESSION_GAP)
    return float(same.mean())


def day_repeat(w: Window) -> float:
    if len(w) < 1000:
        return NAN
    h = w.h
    idx = np.arange(w.i0, w.i1)
    p = h.prev_track_i[idx]
    ok = (p >= 0) & (p != idx - 1)
    pt = np.where(ok, h.t[np.maximum(p, 0)], 0)
    return float((ok & (h.t[idx] - pt < DAY)).mean())


def heavy_rotation(w: Window) -> float:
    def f(s: slice) -> float:
        tr = w.track[s]
        u, inv, c = np.unique(tr, return_inverse=True, return_counts=True)
        return float((c[inv] >= 10).mean())
    return _per_season(w, f)


def album_runs(w: Window) -> float:
    r = w.release
    if len(w) < 1000 or (r >= 0).mean() < 0.5:
        return NAN
    both = (r[1:] >= 0) & (r[:-1] >= 0) & (np.diff(w.t) <= SESSION_GAP)
    if both.sum() < 500:
        return NAN
    return float((r[1:] == r[:-1])[both].mean())


def _artist_runs(w: Window) -> np.ndarray:
    brk = np.r_[True, np.diff(w.t) > SESSION_GAP]
    return _runs(w.artist, brk)


def artist_run_len(w: Window) -> float:
    if len(w) < 1000:
        return NAN
    return float(_artist_runs(w).mean())


def long_runs(w: Window) -> float:
    if len(w) < 1000:
        return NAN
    r = _artist_runs(w)
    return float(r[r >= 10].sum() / r.sum())


def depth(w: Window) -> float:
    def f(s: slice) -> float:
        a, tr = w.artist[s], w.track[s]
        top = _top(a, 10)
        if len(top) < 10:
            return NAN
        return float(np.mean([len(np.unique(tr[a == x])) for x in top]))
    return _per_season(w, f)


def artist_day_streak(w: Window) -> float:
    """Per season: the longest run of consecutive days on which one artist was played. Averaged."""
    d = w.day()

    def f(s: slice) -> float:
        a, dd = w.artist[s], d[s]
        pairs = np.unique(a * 1_000_003 + (dd - dd.min()))
        aa, ds = pairs // 1_000_003, pairs % 1_000_003
        brk = np.r_[True, (aa[1:] != aa[:-1]) | (np.diff(ds) != 1)]
        idx = np.nonzero(brk)[0]
        return float(np.diff(np.r_[idx, len(ds)]).max())
    return _per_season(w, f)


# ------------------------------------------------------------------------------------- the clock

def peak4_share(w: Window) -> float:
    if len(w) < 500:
        return NAN
    m = np.mod(w.t // MIN_MS, 1440)
    hist = np.bincount(m // 60, minlength=24)
    return float(max(hist[np.arange(h, h + 4) % 24].sum() for h in range(24)) / len(m))


def centre_shift(w: Window) -> float:
    """Median hours between a day's centre of listening and the usual centre (days with 5+ plays)."""
    d = w.day()
    m = np.mod(w.t // MIN_MS, 1440)
    ang = 2 * np.pi * m / 1440
    ud, inv = np.unique(d, return_inverse=True)
    n = np.bincount(inv)
    x = np.bincount(inv, weights=np.cos(ang))
    y = np.bincount(inv, weights=np.sin(ang))
    ok = (n >= 5) & (np.hypot(x, y) > 1e-9 * n)
    if ok.sum() < 30:
        return NAN
    da = np.arctan2(y[ok], x[ok])
    c = 2 * np.pi * w.centre_minute() / 1440
    dev = np.abs(np.angle(np.exp(1j * (da - c)))) * 24 / (2 * np.pi)
    return float(np.median(dev))


def weekly_two_days(w: Window) -> float:
    if len(w) < 500 or w.days < 56:
        return NAN
    wd = np.mod(w.day(), 7)
    c = np.bincount(wd, minlength=7)
    return float(max(c[k] + c[(k + 1) % 7] for k in range(7)) / c.sum())


def weekly_cv(w: Window) -> float:
    c = w.weeks()
    if len(c) < 12 or c.sum() < 300:
        return NAN
    return float(c.std() / c.mean())


# ------------------------------------------------------------------------------------- habit

def _active(w: Window) -> tuple[np.ndarray, np.ndarray]:
    a, b = w.day_range()
    d = w.day()
    on = np.zeros(b - a, dtype=bool)
    on[np.clip(d - a, 0, b - a - 1)] = True
    return on, np.bincount(np.clip(d - a, 0, b - a - 1), minlength=b - a)


def active_days(w: Window) -> float:
    if w.days < 60 or len(w) < 300:
        return NAN
    on, _ = _active(w)
    return float(on.mean())


def plays_per_active_day(w: Window) -> float:
    if w.days < 60 or len(w) < 300:
        return NAN
    _, c = _active(w)
    return float(np.median(c[c > 0]))


def silent_weeks(w: Window) -> float:
    c = w.weeks()
    if len(c) < 12 or c.sum() < 300:
        return NAN
    return float((c == 0).mean())


def mean_streak(w: Window) -> float:
    if w.days < 60 or len(w) < 300:
        return NAN
    on, _ = _active(w)
    r = _runs(on.astype(np.int8), np.zeros(len(on), dtype=bool))
    vals = on[np.r_[0, np.cumsum(r)[:-1]]]
    runs = r[vals]
    return float(runs.mean()) if len(runs) else NAN


def _sessions(w: Window) -> tuple[np.ndarray, np.ndarray]:
    s = w.session()
    n = np.bincount(s)
    first = np.r_[0, np.cumsum(n)[:-1]]
    last = first + n - 1
    dur = (w.t[last] - w.t[first]) / MIN_MS + 3.5
    return n, dur


def session_minutes(w: Window) -> float:
    if len(w) < 1000:
        return NAN
    _, dur = _sessions(w)
    return float(np.median(dur))


def marathon_share(w: Window) -> float:
    if len(w) < 1000:
        return NAN
    n, dur = _sessions(w)
    return float(n[dur >= 240].sum() / n.sum())


def sessions_per_day(w: Window) -> float:
    if len(w) < 1000:
        return NAN
    on, _ = _active(w)
    return float(len(np.unique(w.session())) / max(on.sum(), 1))


# ------------------------------------------------------------------------------------- the list

FAMILIES: list[Family] = [
    # discovery
    Family("new_share", "discovery", new_share, tuple(PCT), "pct",
           "At least {th} of your plays go to artists you first heard in the four weeks before.",
           "No more than {th} of your plays go to artists you first heard in the four weeks before."),
    Family("new_per_month", "discovery", new_per_month,
           (1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 125, 150, 200, 250, 300, 400, 500), "num",
           "You meet at least {th} artists you have never played before every month.",
           "You meet no more than {th} artists you have never played before in a typical month.",
           per="new artists a month"),
    Family("one_and_done", "discovery", one_and_done, tuple(PCT), "pct",
           "At least {th} of the new artists you try get exactly one play and never another.",
           "No more than {th} of the new artists you try stop at a single play; you nearly always go back for a second."),
    Family("stick_rate", "discovery", stick_rate, tuple(PCT), "pct",
           "At least {th} of the new artists you try reach ten plays within three months.",
           "No more than {th} of the new artists you try reach ten plays within three months."),
    Family("discovery_bursts", "discovery", discovery_bursts, tuple(PCT), "pct",
           "Your discovery comes in bursts: the busiest tenth of your weeks bring at least {th} of your new artists.",
           "Your discovery is steady: the busiest tenth of your weeks bring no more than {th} of your new artists."),
    Family("comebacks_per_month", "discovery", comebacks_per_month,
           (0.1, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 7, 10, 15, 20), "num",
           "At least {th} times a month you go back to an artist you had played before and then left alone for over a year.",
           "Fewer than {th} times a month do you go back to an artist you had left alone for over a year.",
           per="times a month"),
    # loyalty
    Family("old_friends", "loyalty", old_friends, tuple(PCT), "pct",
           "At least {th} of your plays go to artists that were already in your top 50 four to six months before.",
           "No more than {th} of your plays go to artists that were in your top 50 four to six months before."),
    Family("season_turnover", "loyalty", season_turnover, tuple(PCT), "pct",
           "From one season to the next, at least {th} of your top 10 artists are replaced.",
           "From one season to the next, no more than {th} of your top 10 artists are replaced."),
    Family("same_top", "loyalty", same_top, tuple(PCT), "pct",
           "In at least {th} of seasons your most played artist is the same one as the season before.",
           "In no more than {th} of seasons is your most played artist the same one as the season before."),
    Family("dropped_cold", "loyalty", dropped_cold, tuple(PCT), "pct",
           "You drop favourites cold: at least {th} of a season's top 20 artists get no play at all the next season.",
           "You rarely drop a favourite: no more than {th} of a season's top 20 artists go unplayed the next season."),
    # concentration
    Family("top1_share", "concentration", top1_share, tuple(PCT), "pct",
           "In a typical season one artist takes at least {th} of your plays.",
           "In a typical season no single artist takes more than {th} of your plays."),
    Family("top5_share", "concentration", top5_share, tuple(PCT), "pct",
           "In a typical season your top five artists take at least {th} of your plays.",
           "In a typical season your top five artists take no more than {th} of your plays."),
    Family("effective_artists", "concentration", effective_artists,
           (2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 125, 150, 200, 250, 300, 400, 500, 600, 800, 1000), "num",
           "Your season's listening is spread as widely as {th} or more artists played equally.",
           "Your season's listening is as concentrated as {th} or fewer artists played equally.",
           per="artists played equally"),
    Family("weekly_one_artist", "concentration", weekly_one_artist, tuple(PCT), "pct",
           "In a typical week one artist takes at least {th} of your plays.",
           "In a typical week no single artist takes more than {th} of your plays."),
    Family("artist_days", "concentration", artist_days, tuple(PCT), "pct",
           "On at least {th} of your busy days a single unbroken run of one artist makes up more than half the day.",
           "On no more than {th} of your busy days does one unbroken run of one artist make up more than half the day."),
    # repetition
    Family("back_to_back", "repetition", back_to_back, tuple(PCT), "pct",
           "At least {th} of your plays are the same song again, straight after itself.",
           "No more than {th} of your plays are a song played again straight after itself."),
    Family("day_repeat", "repetition", day_repeat, tuple(PCT), "pct",
           "At least {th} of your plays are a song you had already played in the 24 hours before (not counting straight repeats).",
           "No more than {th} of your plays are a song you had already played in the 24 hours before."),
    Family("heavy_rotation", "repetition", heavy_rotation, tuple(PCT), "pct",
           "At least {th} of a season's plays go to songs you play ten or more times that season.",
           "No more than {th} of a season's plays go to songs you play ten or more times that season."),
    Family("album_runs", "repetition", album_runs, tuple(PCT), "pct",
           "At least {th} of your plays come from the same release as the play before, so you listen in albums.",
           "No more than {th} of your plays come from the same release as the play before."),
    Family("depth", "repetition", depth, (2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 75, 100), "num",
           "In a season you play at least {th} different songs by each of your top 10 artists.",
           "In a season you play no more than {th} different songs by each of your top 10 artists.",
           per="songs per top artist"),
    # runs
    Family("artist_run_len", "runs", artist_run_len, (1.1, 1.2, 1.3, 1.5, 1.75, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 15), "x",
           "Once you put an artist on, you stay: a run of one artist lasts {th} plays or more on average.",
           "You shuffle between artists: a run of one artist lasts {th} plays or fewer on average.",
           per="plays in a run"),
    Family("long_runs", "runs", long_runs, tuple(PCT), "pct",
           "At least {th} of your plays sit in runs of ten or more by the same artist.",
           "No more than {th} of your plays sit in runs of ten or more by the same artist."),
    Family("artist_day_streak", "runs", artist_day_streak,
           (2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 17, 21, 28, 35, 42, 56, 70, 91), "num",
           "Every season some artist gets played on {th} or more days in a row.",
           "In a typical season no artist gets played on more than {th} days in a row.",
           per="days in a row"),
    # the clock
    Family("peak4_share", "clock", peak4_share, tuple(PCT), "pct",
           "At least {th} of your plays fall inside the same four hours of the day.",
           "No four hours of the day hold more than {th} of your plays; you listen around the clock."),
    Family("centre_shift", "clock", centre_shift, (0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 6), "h",
           "Your listening day moves: a typical day's centre sits at least {th} from your usual time.",
           "Your listening keeps a fixed time: a typical day's centre sits within {th} of your usual time."),
    Family("weekly_two_days", "clock", weekly_two_days, tuple(PCT), "pct",
           "Two days in a row each week hold at least {th} of your plays.",
           "No two days in a row hold more than {th} of your weekly plays; your week is even."),
    Family("weekly_cv", "rhythm", weekly_cv,
           (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0), "x",
           "Your weeks swing: the spread of your weekly play counts is at least {th} times their average.",
           "Your weeks are steady: the spread of your weekly play counts is no more than {th} times their average.",
           per="times the average"),
    # habit
    Family("active_days", "rhythm", active_days, tuple(PCT), "pct",
           "You play music on at least {th} of all days.",
           "You play music on no more than {th} of days."),
    Family("plays_per_active_day", "rhythm", plays_per_active_day,
           (5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150, 200), "num",
           "On a day you listen at all, you typically play {th} or more tracks.",
           "On a day you listen at all, you typically play {th} tracks or fewer.",
           per="tracks a day"),
    Family("silent_weeks", "rhythm", silent_weeks, tuple(PCT), "pct",
           "At least {th} of your weeks have no music at all.",
           "No more than {th} of your weeks pass with no music at all."),
    Family("mean_streak", "rhythm", mean_streak, (1.5, 2, 2.5, 3, 4, 5, 6, 7, 10, 14, 21, 30, 45, 60, 90), "days",
           "Once you start listening you keep going: a run of listening days lasts {th} or more on average.",
           "You listen in short spells: a run of listening days lasts {th} or fewer on average."),
    # sessions
    Family("session_minutes", "sessions", session_minutes,
           (5, 10, 15, 20, 30, 45, 60, 75, 90, 120, 150, 180, 240, 300), "min",
           "Your typical session runs {th} or longer.",
           "Your typical session is {th} or shorter."),
    Family("marathon_share", "sessions", marathon_share, tuple(PCT), "pct",
           "At least {th} of your plays happen in sessions of four hours or more.",
           "No more than {th} of your plays happen in sessions of four hours or more."),
    Family("sessions_per_day", "sessions", sessions_per_day, (1, 1.1, 1.2, 1.3, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 5), "x",
           "On a listening day you come back for {th} or more separate sessions.",
           "On a listening day you have {th} sessions or fewer; you listen in one go.",
           per="sessions a day"),
]

BY_KEY = {f.key: f for f in FAMILIES}


def measure(w: Window) -> np.ndarray:
    """Every family's value on one window, in FAMILIES order (nan where the data can't say)."""
    out = np.full(len(FAMILIES), NAN)
    if len(w) == 0:
        return out
    for k, f in enumerate(FAMILIES):
        out[k] = f.compute(w)
    return out
