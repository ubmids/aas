"""music: how someone listens, measured from their play history, with receipts.

This reader does NOT guess personality. Music taste predicts the Big Five only weakly, so the
October rounds (TypeScript engine, engine/src/model/traits.ts) replaced inferred personality with
measured traits: numbers read straight off the history, each with the plays that produced it and a
percentile against other listeners. This is a faithful Python port of the traits that held up:

  trait               definition (start times; a "proper" play is one of 30 s or more)
  new_share           share of proper plays going to artists first heard in the 28 days before,
                      after a 12-week warm-up
  old_friends         in each full 8-week stretch, the share of proper plays going to artists in the
                      top 50 of the stretches 2 and 3 before (4 to 6 months earlier), averaged over
                      stretches with more than 100 plays
  season_turnover     mean share of the top 10 artists replaced between consecutive 13-week seasons
  clock_spread        1 - R, R = length of the minutes-weighted mean unit vector of each proper
                      play's time of day on the 24-hour dial (minutes capped at 10)
  clock_regularity    length of the mean of the daily mean directions, over days with 5+ proper
                      plays (needs 30 such days); each day starts 12 h from the person's centre
  finish_share        share of plays whose reason_end is "trackdone" (needs reason_end data)
  skip_seconds        median seconds played before a skip ended by fwdbtn or unknown (needs ms)

  dial_explore        explore (high) against stay (low): new share, old friends, turnover
  dial_regular        listening-time regularity: clock spread (-) and clock regularity (+)

Not ported: map spread (needs the co-listening taste map built from the 304-listener bench table,
which is not shipped with this package), clock hour and night share (need a time zone; Play has
none, see the module doc in docs/reader_music.md).

Percentiles and the dials use py/src/aas/data/music_norms.json and music_dials.json, built by
py/scripts/build_music_norms.py from 304 public ListenBrainz histories (aggregate quantiles only).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.stats import norm

from ..person import Estimate, Person, Play

NAME = "music"
DATA = Path(__file__).resolve().parents[1] / "data"
DAY = 86_400_000
WEEK = 7 * DAY
SHORT = 8 * WEEK     # old-friends stretch
BLOCK = 13 * WEEK    # season
SKIP_MS = 30_000     # Spotify's own threshold for a stream
UNKNOWN_MS = 180_000  # play length assumed when a source does not record it (as the TS bench did)

# Minimums for the whole reader. Below these nothing is said.
MIN_PLAYS = 2_000
MIN_DAYS = 90

# Split-half test-retest r on the 304 ListenBrainz listeners (MATH.md section 15); Spotify bench
# (13 people) for the two traits ListenBrainz cannot measure.
RETEST = {
    "new_share": 0.754, "old_friends": 0.811, "season_turnover": 0.802,
    "clock_spread": 0.720, "clock_regularity": 0.725, "finish_share": 0.86, "skip_seconds": 0.99,
}


# ----------------------------------------------------------------------------------- arrays

@dataclass
class Arrays:
    """A history as columns, sorted by start time. `artist` holds codes into `names`."""
    t: np.ndarray            # start, ms since epoch (int64)
    ms: np.ndarray           # ms played (int64); UNKNOWN_MS where the source has none
    ms_known: np.ndarray     # bool
    artist: np.ndarray       # int64 codes
    names: list[str]
    reason_end: list[str | None]


def _ms_epoch(d: datetime) -> int:
    if d.tzinfo is None:
        d = d.replace(tzinfo=UTC)
    return int(round(d.timestamp() * 1000))


def arrays_from_plays(plays: list[Play]) -> Arrays:
    """Columns from Play objects. Play.when is the end time, so start = end - ms played."""
    rows = []
    for p in plays:
        end = _ms_epoch(p.when)
        known = p.ms_played is not None
        ms = int(p.ms_played) if known else UNKNOWN_MS
        start = end - ms if known else end
        rows.append((start, ms, known, p.artist, p.reason_end or None))
    rows.sort(key=lambda r: r[0])  # stable
    return arrays_from_rows(rows)


def arrays_from_rows(rows: list[tuple[int, int, bool, str, str | None]]) -> Arrays:
    """(start_ms, ms, ms_known, artist, reason_end) rows, already in time order."""
    codes: dict[str, int] = {}
    art = np.fromiter((codes.setdefault(r[3], len(codes)) for r in rows), dtype=np.int64, count=len(rows))
    return Arrays(
        t=np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows)),
        ms=np.fromiter((r[1] for r in rows), dtype=np.int64, count=len(rows)),
        ms_known=np.fromiter((r[2] for r in rows), dtype=bool, count=len(rows)),
        artist=art,
        names=list(codes),
        reason_end=[r[4] for r in rows],
    )


# ----------------------------------------------------------------------------------- traits

@dataclass
class Trait:
    value: float                 # nan when the data can't say
    why_not: str = ""            # plain words when value is nan
    detail: dict | None = None   # receipts


def _top(art: np.ndarray, k: int) -> list[int]:
    """Top-k codes by count; ties keep first-appearance order (as the JS Map + stable sort did)."""
    if not len(art):
        return []
    u, first, cnt = np.unique(art, return_index=True, return_counts=True)
    order = np.lexsort((first, -cnt))[:k]
    return [int(x) for x in u[order]]


def new_share(a: Arrays, proper: np.ndarray, until: int | None = None) -> Trait:
    t0 = int(a.t[0])
    # first time each artist was heard (any play, skips included)
    first = np.full(len(a.names), np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(first, a.artist, a.t)
    warm = t0 + 12 * WEEK
    sel = proper & (a.t >= warm)
    if until is not None:
        sel &= a.t >= until
    idx = np.nonzero(sel)[0]
    if len(idx) < 300:
        return Trait(math.nan, "needs 300 full plays after a 12-week warm-up (about 3 months of history to learn who you already knew)")
    fresh = idx[(a.t[idx] - first[a.artist[idx]]) < 28 * DAY]
    v = len(fresh) / len(idx)
    # receipts: the fresh artists you played most, most recent first among equals
    tops = _top(a.artist[fresh][::-1], 3)
    return Trait(v, detail={"fresh": len(fresh), "scored": len(idx), "examples": [a.names[c] for c in tops],
                            "from": int(a.t[idx[0]]), "to": int(a.t[idx[-1]])})


def old_friends(a: Arrays, proper: np.ndarray) -> Trait:
    t0, t1 = int(a.t[0]), int(a.t[-1])
    pt, part = a.t[proper], a.artist[proper]
    stretches = []
    s = t0
    while s + SHORT <= t1 + 1:
        lo, hi = np.searchsorted(pt, s, "left"), np.searchsorted(pt, s + SHORT, "left")
        stretches.append(part[lo:hi])
        s += SHORT
    tops = [_top(x, 50) for x in stretches]
    shares: list[float] = []
    friends = np.zeros(len(a.names), dtype=np.int64)
    for i in range(2, len(stretches)):
        old = set(tops[i - 2]) | (set(tops[i - 3]) if i >= 3 else set())
        cur = stretches[i]
        hit = np.isin(cur, np.fromiter(old, dtype=np.int64, count=len(old)))
        np.add.at(friends, cur[hit], 1)
        if len(cur) > 100:
            shares.append(float(hit.sum()) / len(cur))
    if len(shares) < 2:
        return Trait(math.nan, "needs at least two 8-week stretches with 100+ plays that sit 4 to 6 months after earlier ones (about 8 months of history)")
    named = [a.names[c] for c in np.argsort(-friends, kind="stable")[:3] if friends[c] > 0]
    return Trait(float(np.mean(shares)), detail={"stretches": len(shares), "named": named})


def season_turnover(a: Arrays, proper: np.ndarray) -> Trait:
    t0, t1 = int(a.t[0]), int(a.t[-1])
    pt, part = a.t[proper], a.artist[proper]
    tops = []
    s = t0
    while s + BLOCK <= t1 + 1:
        lo, hi = np.searchsorted(pt, s, "left"), np.searchsorted(pt, s + BLOCK, "left")
        tops.append(_top(part[lo:hi], 10))
        s += BLOCK
    turns, last_change = [], None
    for i in range(len(tops) - 1):
        if len(tops[i]) < 10 or len(tops[i + 1]) < 10:
            continue
        nxt = set(tops[i + 1])
        gone = [x for x in tops[i] if x not in nxt]
        turns.append(len(gone) / 10)
        last_change = (gone, [x for x in tops[i + 1] if x not in set(tops[i])], t0 + (i + 1) * BLOCK)
    if len(turns) < 2:
        return Trait(math.nan, "needs three back-to-back 13-week seasons with at least 10 artists each (about 9 months of history)")
    det = {"changes": len(turns)}
    if last_change:
        det.update(out=[a.names[c] for c in last_change[0][:3]], into=[a.names[c] for c in last_change[1][:3]], at=last_change[2])
    return Trait(float(np.mean(turns)), detail=det)


def clock_shape(a: Arrays, proper: np.ndarray, offset_min: int = 0) -> tuple[Trait, Trait]:
    """Time-zone-free clock traits. A fixed offset rotates every angle equally and changes neither."""
    MIN = 1440
    t = a.t[proper]
    if not len(t):
        return Trait(math.nan, "no full plays"), Trait(math.nan, "no full plays")
    local = np.mod(np.floor_divide(t, 60_000) + offset_min, MIN)
    ang = 2 * np.pi * local / MIN
    w = np.minimum(a.ms[proper], 600_000) / 60_000
    sx, sy, tot = float((w * np.cos(ang)).sum()), float((w * np.sin(ang)).sum()), float(w.sum())
    R = math.hypot(sx, sy) / tot if tot else math.nan
    # receipt: share of minutes inside the busiest 8 hours of the day
    hist = np.bincount((local // 60).astype(np.int64), weights=w, minlength=24)
    best8 = max(hist[np.arange(h, h + 8) % 24].sum() for h in range(24)) / tot if tot else math.nan
    spread = Trait(1 - R, detail={"plays": len(t), "R": R, "busiest8": float(best8)}) if len(t) >= 500 else \
        Trait(math.nan, "needs 500 full plays")
    centre = ((math.atan2(sy, sx) / (2 * math.pi)) * MIN + MIN) % MIN
    day_start = (centre + MIN / 2) % MIN
    day = np.floor((t / 60_000 + offset_min - day_start) / MIN).astype(np.int64)
    uday, inv = np.unique(day, return_inverse=True)
    x = np.bincount(inv, weights=np.cos(ang), minlength=len(uday))
    y = np.bincount(inv, weights=np.sin(ang), minlength=len(uday))
    n = np.bincount(inv, minlength=len(uday))
    # A day whose plays cancel out on the dial (e.g. evenly 12 h apart) has no centre: its vector
    # length is rounding noise (~1e-16) and its direction arbitrary. The TS engine kept such days
    # unless the sum was exactly 0, which made one bench person's value depend on the last bit of
    # cos(); here they are left out.
    ok = (n >= 5) & (np.hypot(x, y) > 1e-9 * n)
    nd = int(ok.sum())
    if nd < 30:
        return spread, Trait(math.nan, "needs 30 days with 5 or more full plays")
    r = np.hypot(x[ok], y[ok])
    ux, uy = x[ok] / r, y[ok] / r
    reg = math.hypot(float(ux.sum()), float(uy.sum())) / nd
    # receipt: how far a typical day's centre sits from the usual one, in hours
    mean_ang = math.atan2(float(uy.sum()), float(ux.sum()))
    dev = np.abs(np.angle(np.exp(1j * (np.arctan2(uy, ux) - mean_ang)))) * 24 / (2 * np.pi)
    return spread, Trait(reg, detail={"days": nd, "median_dev_h": float(np.median(dev))})


def finish_share(a: Arrays) -> Trait:
    ended = [r for r in a.reason_end if r]
    if len(ended) < 300:
        return Trait(math.nan, "needs 300 plays that record how they ended (Spotify extended history has this)")
    done = sum(1 for r in ended if r == "trackdone")
    return Trait(done / len(ended), detail={"done": done, "ended": len(ended)})


def skip_seconds(a: Arrays) -> Trait:
    re = a.reason_end
    sel = [i for i in np.nonzero(a.ms_known & (a.ms > 0) & (a.ms < SKIP_MS))[0] if re[i] is None or re[i] == "fwdbtn"]
    if len(sel) < 100:
        return Trait(math.nan, "needs 100 skips with real play lengths (Spotify history has them)")
    s = np.sort(a.ms[sel] / 1000)
    return Trait(float(s[len(s) >> 1]), detail={"skips": len(sel)})  # the TS median: upper middle


def measure(a: Arrays) -> dict[str, Trait]:
    """All traits on one history. Mirrors measureTraits() in engine/src/model/traits.ts."""
    if len(a.t) < 500:
        return {}
    proper = ~(a.ms_known & (a.ms < SKIP_MS))
    cs, cr = clock_shape(a, proper)
    return {
        "new_share": new_share(a, proper),
        "old_friends": old_friends(a, proper),
        "season_turnover": season_turnover(a, proper),
        "clock_spread": cs,
        "clock_regularity": cr,
        "finish_share": finish_share(a),
        "skip_seconds": skip_seconds(a),
    }


# ----------------------------------------------------------------------------------- norms, dials

def _load(name: str) -> dict | None:
    f = DATA / name
    return json.loads(f.read_text()) if f.exists() else None


def percentile_of(value: float, q: dict) -> float | None:
    """Percentile (0-100) from a quantile table {"p": [...], "v": [...]}, linear between knots."""
    if q is None or not math.isfinite(value):
        return None
    p, v = np.asarray(q["p"], float), np.asarray(q["v"], float)
    if value <= v[0]:
        return float(p[0] / 2)        # below the 1st percentile knot: call it half of it
    if value >= v[-1]:
        return float(100 - (100 - p[-1]) / 2)
    # flat stretches (ties) take the middle of their percentile range
    lo, hi = np.searchsorted(v, value, "left"), np.searchsorted(v, value, "right")
    if hi > lo:
        return float((p[lo] + p[hi - 1]) / 2)
    return float(np.interp(value, v, p))


def dial_scores(values: dict[str, float], norms: dict, dials: dict) -> dict[str, float]:
    """Each dial = sum of weight x normal score, the normal score read off the 304-listener norms."""
    out = {}
    for d in dials["dials"]:
        z = []
        for k in d["traits"]:
            pc = percentile_of(values.get(k, math.nan), norms["traits"].get(k))
            if pc is None:
                break
            z.append(norm.ppf(min(max(pc / 100, dials["clip"]), 1 - dials["clip"])))
        else:
            out[d["key"]] = float(np.dot(d["weights"], z))
    return out


# ----------------------------------------------------------------------------------- the reader

def _pct(x: float) -> str:
    return f"{round(100 * x)}%"


def _date(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=UTC).strftime("%b %Y")


def _names(xs: list[str]) -> str:
    return ", ".join(xs[:-1]) + (" and " if len(xs) > 1 else "") + xs[-1] if xs else ""


def _ord(x: float) -> str:
    n = round(x)
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def _vs(pc: float | None, n: int, more: str = "more than") -> str:
    if pc is None:
        return ""
    return f" That's {more} {round(pc)}% of the {n} ListenBrainz listeners we measured."


class MusicReader:
    """Describes how someone listens (not who they are). See module docstring."""
    name = NAME

    def __init__(self) -> None:
        self.norms = _load("music_norms.json")
        self.dials = _load("music_dials.json")

    def can_read(self, person: Person) -> tuple[bool, str]:
        n = len(person.plays)
        if n < MIN_PLAYS:
            return False, f"only {n:,} plays; the music reader needs at least {MIN_PLAYS:,} to say anything steady"
        ts = [p.when for p in person.plays]
        days = (_ms_epoch(max(ts)) - _ms_epoch(min(ts))) / DAY
        if days < MIN_DAYS:
            return False, f"the history covers {days:.0f} days; the music reader needs about 3 months ({MIN_DAYS} days)"
        return True, f"{n:,} plays over {days / 30.4:.0f} months"

    def _est(self, trait: str, value: float, evidence: str, pc: float | None = None, usable: bool = True) -> Estimate:
        return Estimate(reader=NAME, trait=trait, kind="measured", value=value, percentile=pc,
                        evidence=evidence, usable=usable and math.isfinite(value),
                        interval_level=None, reliability=RETEST.get(trait))

    def read(self, person: Person) -> list[Estimate]:
        ok, why = self.can_read(person)
        keys = list(RETEST) + ["dial_explore", "dial_regular"]
        if not ok:
            return [self._est(k, math.nan, why, usable=False) for k in keys]
        a = arrays_from_plays(person.plays)
        return self.read_arrays(a)

    def read_arrays(self, a: Arrays) -> list[Estimate]:
        tr = measure(a)
        norms = self.norms or {"traits": {}, "n": 0}
        pc = {k: percentile_of(t.value, norms["traits"].get(k)) for k, t in tr.items()}
        ns = {k: v["n"] for k, v in norms["traits"].items()}
        out: list[Estimate] = []

        def nan(k: str) -> Estimate:
            return self._est(k, math.nan, f"Not enough data: {tr[k].why_not}.", usable=False)

        # new share
        t = tr["new_share"]
        if math.isfinite(t.value):
            d = t.detail
            ev = (f"{_pct(t.value)} of your full plays went to artists you'd been listening to for under 4 weeks "
                  f"({d['fresh']:,} of {d['scored']:,}, {_date(d['from'])} to {_date(d['to'])}), "
                  f"most of all {_names(d['examples'])}.")
            last = new_share(a, ~(a.ms_known & (a.ms < SKIP_MS)), until=int(a.t[-1]) - 365 * DAY)
            if math.isfinite(last.value) and len(a.t) and a.t[-1] - a.t[0] > 450 * DAY:
                ev += f" In the last 12 months alone it was {_pct(last.value)}."
            out.append(self._est("new_share", t.value, ev + _vs(pc["new_share"], ns["new_share"]), pc["new_share"]))
        else:
            out.append(nan("new_share"))

        t = tr["old_friends"]
        if math.isfinite(t.value):
            d = t.detail
            ev = (f"Over {d['stretches']} two-month stretches, {_pct(t.value)} of your plays went to artists who were "
                  f"already among your top 50 four to six months before, most of all {_names(d['named'])}.")
            out.append(self._est("old_friends", t.value, ev + _vs(pc["old_friends"], ns["old_friends"]), pc["old_friends"]))
        else:
            out.append(nan("old_friends"))

        t = tr["season_turnover"]
        if math.isfinite(t.value):
            d = t.detail
            ev = f"Over {d['changes']} changes of season (13 weeks each), {_pct(t.value)} of your top 10 artists was replaced each time."
            if d.get("out"):
                ev += f" At the change around {_date(d['at'])}, {_names(d['out'])} left your top 10 and {_names(d['into'])} came in."
            out.append(self._est("season_turnover", t.value, ev + _vs(pc["season_turnover"], ns["season_turnover"]), pc["season_turnover"]))
        else:
            out.append(nan("season_turnover"))

        t = tr["clock_spread"]
        if math.isfinite(t.value):
            d = t.detail
            ev = (f"Across {d['plays']:,} full plays, {_pct(d['busiest8'])} of your listening minutes fall in your busiest "
                  f"8 hours of the day (0 would mean always the same time, 1 evenly round the clock; you're at {t.value:.2f}).")
            out.append(self._est("clock_spread", t.value, ev + _vs(pc["clock_spread"], ns["clock_spread"]), pc["clock_spread"]))
        else:
            out.append(nan("clock_spread"))

        t = tr["clock_regularity"]
        if math.isfinite(t.value):
            d = t.detail
            ev = (f"On {d['days']:,} days with 5 or more plays, the middle of the day's listening sat within "
                  f"{d['median_dev_h']:.1f} hours of your usual time on a typical day (score {t.value:.2f}, 1 would be the same time every day).")
            out.append(self._est("clock_regularity", t.value, ev + _vs(pc["clock_regularity"], ns["clock_regularity"]), pc["clock_regularity"]))
        else:
            out.append(nan("clock_regularity"))

        # no ListenBrainz reference for these two: ListenBrainz records neither skips nor how plays end
        t = tr["finish_share"]
        if math.isfinite(t.value):
            d = t.detail
            out.append(self._est("finish_share", t.value,
                                 f"{d['done']:,} of {d['ended']:,} plays ran to the end of the track ({_pct(t.value)}). "
                                 "No percentile: the reference listeners' data doesn't record how plays end."))
        else:
            out.append(nan("finish_share"))
        t = tr["skip_seconds"]
        if math.isfinite(t.value):
            out.append(self._est("skip_seconds", t.value,
                                 f"Half of your {t.detail['skips']:,} skips came within {t.value:.1f} seconds. "
                                 "No percentile: the reference listeners' data has no skips."))
        else:
            out.append(nan("skip_seconds"))

        out += self._dials(tr)
        return out

    def _dials(self, tr: dict[str, Trait]) -> list[Estimate]:
        if not self.dials or not self.norms:
            return [self._est(k, math.nan, "dial parameters not installed", usable=False) for k in ("dial_explore", "dial_regular")]
        vals = {k: t.value for k, t in tr.items()}
        sc = dial_scores(vals, self.norms, self.dials)
        out = []
        for d in self.dials["dials"]:
            k = d["key"]
            if k not in sc:
                miss = [x for x in d["traits"] if not math.isfinite(vals.get(x, math.nan))]
                out.append(self._est(k, math.nan, f"Needs {', '.join(miss)} first, which this history can't measure yet.", usable=False))
                continue
            v = sc[k]
            p = percentile_of(v, self.norms["dials"][k])
            parts = []
            for x in d["traits"]:
                px = percentile_of(vals[x], self.norms["traits"][x])
                parts.append(f"{x.replace('_', ' ')} at the {_ord(px)} percentile")
            side = d["high"] if v >= 0 else d["low"]
            ev = (f"{d['label']}: {v:+.2f} (0 is the typical listener, about +/-1 covers two in three people), "
                  f"so you lean {side}. Built from {', '.join(parts)}.")
            ev += _vs(p, self.norms["dials"][k]["n"], "further toward " + d["high"] + " than")
            out.append(self._est(k, v, ev, p))
        return out
