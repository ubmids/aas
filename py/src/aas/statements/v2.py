"""Round 2 of the statements lane, frozen before the confirmation set was opened (py/docs/statements.md,
"Round 2"). Everything here was chosen on round 1's 304 listeners (303 after the single-artist
exclusion) and is applied unchanged to the fresh listeners.

What changed from round 1:

- 57 families (the 34 of round 1 and 23 more, `families_v2.py`).
- The rarity line is 20%: a statement qualifies when it is true of the person and of at most a fifth
  of the reference population (round 1: a tenth). Rarer is still better, through the surprisal.
- Score = confidence^3 x surprisal x reliability^2, where reliability is the family's split-half
  Spearman correlation on round 1's listeners (frozen below). Cubing confidence prefers thresholds
  the person clears comfortably over ones they sit just past; the reliability term prefers families
  whose values hold from one half of a history to the other.
- Accounts where one artist takes over 75% of listens, or with fewer than 50 artists, are excluded.

Unchanged: a statement is a family, a direction and a round threshold from the family's grid; one per
family; a top-k takes the highest scores with at most two per group; confidence and surprisal are
computed as in round 1 (`engine.py`); the lineup, the matcher and the Barnum control are round 1's.

The fingerprint matcher (secondary) places every family value on the percentile scale of the
reference first halves and picks the lineup member whose second half is nearest the target's first
half in mean absolute percentile difference over the families both have.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from .engine import BARNUM, Statement, Table

LINE = 0.20              # true of at most this share of the reference population
CONF_POWER = 3
REL_POWER = 2
MAX_TOP_ARTIST = 0.75    # exclusion: one artist over this share of all listens
MIN_ARTISTS = 50         # exclusion: fewer different artists than this

# Split-half Spearman r of each family on round 1's 303 listeners (first half against second half),
# frozen on 5 November 2026. Negative or missing values would count as 0; none are.
RELIABILITY: dict[str, float] = {
    "new_share": 0.72, "new_per_month": 0.79, "one_and_done": 0.61, "stick_rate": 0.82,
    "discovery_bursts": 0.69, "comebacks_per_month": 0.67, "old_friends": 0.80, "season_turnover": 0.80,
    "same_top": 0.42, "dropped_cold": 0.68, "top1_share": 0.76, "top5_share": 0.81,
    "effective_artists": 0.82, "weekly_one_artist": 0.80, "artist_days": 0.80, "back_to_back": 0.71,
    "day_repeat": 0.79, "heavy_rotation": 0.82, "album_runs": 0.89, "depth": 0.78,
    "artist_run_len": 0.88, "long_runs": 0.85, "artist_day_streak": 0.75, "peak4_share": 0.66,
    "centre_shift": 0.62, "weekly_two_days": 0.50, "weekly_cv": 0.73, "active_days": 0.79,
    "plays_per_active_day": 0.78, "silent_weeks": 0.64, "mean_streak": 0.77, "session_minutes": 0.73,
    "marathon_share": 0.75, "sessions_per_day": 0.77, "new_track_share": 0.73, "old_songs": 0.64,
    "long_known": 0.52, "song_comebacks": 0.78, "first_day_binge": 0.81, "once_tracks": 0.78,
    "track_variety": 0.83, "top10_tracks": 0.83, "artist_tail": 0.85, "artists_per_season": 0.80,
    "song_return_days": 0.76, "artist_return_days": 0.77, "track_gap": 0.77, "session_plays": 0.71,
    "single_play_sessions": 0.69, "artists_per_session": 0.85, "session_gap_hours": 0.71, "day_span": 0.79,
    "active_hours": 0.75, "busy_days": 0.69, "longest_break": 0.61, "same_busiest_day": 0.31,
    "album_sittings": 0.87,
}


def candidates(tab: Table, i: int, exclude, mode: str = "only", line: float = LINE) -> list[Statement]:
    """The best statement per family for person i against everyone except `exclude` (which holds i).
    mode "only": true of at most `line` of the reference, ranked by the v2 score; mode "barnum": true
    of 50% to 90% of the reference, ranked by confidence (round 1's control)."""
    ex = np.unique(np.asarray(exclude, dtype=np.int64))
    n = tab.P - len(ex)
    out = []
    for (k, d), T in tab.T.items():
        v = tab.V[i, k]
        if not math.isfinite(v) or not T[i].any():
            continue
        c = tab.tot[(k, d)] - T[ex].sum(0)
        p = c / n
        g = tab.grid[k]
        conf = norm.cdf((v - g) / tab._spread(i, k) * (1 if d == "hi" else -1))
        if mode == "only":
            ok = T[i] & (p <= line)
            rel = max(RELIABILITY.get(tab.families[k].key, 0.0), 0.0)
            sc = conf ** CONF_POWER * -np.log2((c + 1) / (n + 1)) * rel ** REL_POWER
        else:
            ok = T[i] & (p >= BARNUM[0]) & (p <= BARNUM[1])
            sc = conf
        if not ok.any():
            continue
        j = int(np.argmax(np.where(ok, sc, -np.inf)))
        out.append(Statement(tab.families[k], d, float(g[j]), float(v), float(conf[j]), int(c[j]), n, float(sc[j])))
    out.sort(key=lambda s: -s.score)
    seen, uniq = set(), []
    for s in out:
        if s.family.key not in seen:
            seen.add(s.family.key)
            uniq.append(s)
    return uniq


def percentiles(ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Mid-rank percentile of each value of x (people x families) within the finite values of ref."""
    out = np.full(x.shape, np.nan)
    for k in range(x.shape[1]):
        r = np.sort(ref[:, k][np.isfinite(ref[:, k])])
        ok = np.isfinite(x[:, k])
        if len(r):
            out[ok, k] = (np.searchsorted(r, x[ok, k], "left") + np.searchsorted(r, x[ok, k], "right")) / 2 / len(r)
    return out


def fingerprint_credit(p1: np.ndarray, p2: np.ndarray, members: np.ndarray) -> np.ndarray:
    """For lineups `members` (... x 5, target first): credit of the nearest second half to the
    target's first half in mean absolute percentile difference. Ties share the credit."""
    flat = members.reshape(-1, members.shape[-1])
    cred = np.empty(len(flat))
    for r, m in enumerate(flat):
        diff = np.abs(p2[m] - p1[m[0]])
        n = np.isfinite(diff).sum(1)
        dist = np.where(n > 0, np.nansum(diff, 1) / np.maximum(n, 1), np.inf)
        tied = dist == dist.min()
        cred[r] = tied[0] / tied.sum()
    return cred.reshape(members.shape[:-1])


def stable_statements(first: Table, second: Table, i: int, line: float = LINE) -> list[Statement]:
    """Person i's v2 statements from the first half (against the other first halves) that are still
    true on i's second half and still true of at most `line` of the others' second halves."""
    others = np.delete(np.arange(first.P), i)
    keep = []
    for s in candidates(first, i, [i], "only", line):
        if bool(second.truth(s, i)) and second.truth(s, others).mean() <= line:
            keep.append(s)
    return keep
