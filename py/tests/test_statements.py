"""Statement engine on synthetic listeners: windows never read the future, clock families are
time-zone free, the only-you filter excludes the person and the decoys from the population and only
uses grid thresholds, and the lineup test finds distinct people and sits at chance for identical ones."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from aas.person import Play
from aas.statements import FAMILIES, Table, from_plays, from_rows, measure, top_k
from aas.statements import lineup as lu
from aas.statements.engine import ONLY_YOU, quarter_sd
from aas.statements.families import BY_KEY, fmt
from aas.statements.history import DAY, HOUR

T0 = int(datetime(2023, 1, 1, tzinfo=UTC).timestamp() * 1000)


def listener(seed: int, days: int = 500, explore: float = 0.2, per_day: int = 30, hour: float = 20.0,
             loop: float = 0.0, skip_days: float = 0.1, pool: int = 40, shift_ms: int = 0):
    """A synthetic history. explore = chance a play goes to a brand-new artist; loop = chance a play
    repeats the previous track; hour = usual start hour (UTC)."""
    rng = np.random.default_rng(seed)
    rows, new = [], 0
    for d in range(days):
        if rng.random() < skip_days:
            continue
        t = T0 + d * DAY + int((hour + rng.normal(0, 1)) * HOUR) + shift_ms
        prev = None
        for _ in range(max(1, int(rng.poisson(per_day)))):
            t += 200_000 + int(rng.integers(0, 60_000))
            if prev and rng.random() < loop:
                a, tr = prev
            elif rng.random() < explore:
                new += 1
                a, tr = f"new{seed}-{new}", f"s{int(rng.integers(5))}"
            else:
                a, tr = f"fav{seed}-{int(rng.zipf(1.6)) % pool}", f"s{int(rng.integers(12))}"
            rows.append((t, a, a + "|" + tr, a + "/album"))
            prev = (a, tr)
    return from_rows(rows)


def test_families_unique_and_well_formed():
    keys = [f.key for f in FAMILIES]
    assert len(keys) == len(set(keys)) and 20 <= len(keys) <= 40
    for f in FAMILIES:
        assert list(f.grid) == sorted(f.grid)
        assert "{th}" in f.hi and "{th}" in f.lo
        # no sensitive inference words in any sentence
        for w in ("health", "religio", "sexual", "politic", "depress", "mood"):
            assert w not in (f.hi + f.lo).lower()


def test_window_never_reads_the_future():
    h = listener(1, days=400)
    a, _ = h.halves()
    before = measure(a)
    # append a wildly different future: lots of new artists, at another hour
    rows = [(int(t), f"x{int(x)}", f"t{int(tr)}", f"r{int(r)}") for t, x, tr, r in zip(h.t, h.artist, h.track, h.release)]
    late = [(int(h.t[-1]) + DAY + i * 600_000, f"future{i}", f"future{i}|t", None) for i in range(3000)]
    h2 = from_rows(rows + late)
    a2 = h2.window(a.lo, a.hi)
    after = measure(a2)
    for f, x, y in zip(FAMILIES, before, after):
        assert (math.isnan(x) and math.isnan(y)) or x == pytest.approx(y), f.key


def test_clock_families_ignore_time_zone():
    a = measure(listener(2, shift_ms=0).window())
    b = measure(listener(2, shift_ms=5 * HOUR).window())
    for key in ("peak4_share", "centre_shift", "weekly_two_days"):
        k = [f.key for f in FAMILIES].index(key)
        assert a[k] == pytest.approx(b[k], abs=1e-9), key


def test_explorer_and_loyal_listener_differ_the_right_way():
    ex = measure(listener(3, explore=0.6).window())
    lo = measure(listener(4, explore=0.01).window())
    k = {f.key: i for i, f in enumerate(FAMILIES)}
    assert ex[k["new_share"]] > lo[k["new_share"]] + 0.2
    assert lo[k["old_friends"]] > ex[k["old_friends"]] + 0.2
    assert ex[k["new_per_month"]] > 5 * lo[k["new_per_month"]]


def test_looper_shows_back_to_back_repeats():
    k = [f.key for f in FAMILIES].index("back_to_back")
    assert measure(listener(5, loop=0.5).window())[k] > 0.3
    assert measure(listener(6, loop=0.0).window())[k] < 0.15


def test_from_plays_uses_start_times():
    start = datetime(2024, 3, 1, 12, tzinfo=UTC)
    plays = [Play(when=start + timedelta(minutes=4 * i), artist="A", track="t", ms_played=180_000) for i in range(5)]
    h = from_plays(plays)
    assert h.t[0] == int(start.timestamp() * 1000) - 180_000
    assert h.n_artists == 1


def _population(n: int = 40, odd: bool = True) -> np.ndarray:
    rows = [measure(listener(100 + i, days=420, explore=0.15, loop=0.02).window()) for i in range(n)]
    if odd:
        rows[0] = measure(listener(7, days=420, explore=0.15, loop=0.6).window())
    return np.stack(rows)


def test_only_you_finds_the_odd_one_and_counts_others_only():
    V = _population()
    tab = Table(V)
    sts = tab.candidates(0, [0])
    assert any(s.family.key == "back_to_back" and s.direction == "hi" for s in sts)
    for s in sts:
        assert s.th in s.family.grid                      # thresholds come from the grid only
        assert s.n == V.shape[0] - 1                      # the person is not in the population
        assert s.c / s.n <= ONLY_YOU
        assert bool(tab.truth(s, 0))                      # and it is true of them
        others = tab.truth(s, np.arange(1, V.shape[0]))
        assert others.sum() == s.c
    # excluding decoys shrinks the reference
    s2 = tab.candidates(0, [0, 1, 2, 3, 4])
    assert all(s.n == V.shape[0] - 5 for s in s2)


def test_top_k_respects_group_limit():
    V = _population()
    sts = Table(V).candidates(0, [0])
    top = top_k(sts, 5)
    groups = [s.family.group for s in top]
    assert all(groups.count(g) <= 2 for g in groups)
    assert [s.score for s in top] == sorted([s.score for s in top], reverse=True)


def test_lineup_distinct_people_beat_chance_and_identical_people_do_not():
    # distinct: people differ in explore, loop, hour and volume; halves of the same person agree
    rng = np.random.default_rng(0)
    h1, h2 = [], []
    for i in range(30):
        h = listener(200 + i, days=600, explore=float(rng.uniform(0.02, 0.6)), loop=float(rng.uniform(0, 0.5)),
                     per_day=int(rng.integers(10, 80)), skip_days=float(rng.uniform(0, 0.6)))
        a, b = h.halves()
        h1.append(measure(a))
        h2.append(measure(b))
    first, second = Table(np.stack(h1)), Table(np.stack(h2))
    r = lu.run(first, second, 3, 5, seed=1)
    assert r.accuracy > 0.5
    null, p = lu.permutation_null(r, 200)
    assert abs(null.mean() - 0.2) < 0.08 and p < 0.05
    # identical: everyone's second half is the same person's -> no one can be told apart
    same = Table(np.repeat(np.stack(h2)[:1], 30, axis=0))
    r2 = lu.run(first, same, 3, 5, seed=1)
    assert r2.accuracy == pytest.approx(0.2)


def test_quarter_sd_needs_two_quarters():
    q = np.array([[[1.0, np.nan], [2.0, np.nan], [3.0, 5.0], [np.nan, np.nan]]])
    sd = quarter_sd(q)
    assert sd[0, 0] == pytest.approx(1.0) and math.isnan(sd[0, 1])


def test_format():
    assert fmt(0.075, "pct") == "7.5%" and fmt(0.6, "pct") == "60%"
    assert BY_KEY["session_minutes"].fmt(180) == "3 hours"
    assert BY_KEY["old_friends"].text("hi", 0.8).startswith("At least 80% of your plays")
