"""Round 2 of the statements lane: the 57 families keep round 1's rules (no future, no time zone, no
names, round grids), the frozen reliabilities cover every family, the v2 filter counts others only and
respects its 20% line, and the fingerprint matcher finds distinct people and not identical ones."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aas.statements import from_rows
from aas.statements import lineup as lu
from aas.statements import v2
from aas.statements.engine import Table
from aas.statements.families_v2 import BY_KEY_V2, FAMILIES_V2, NEW, measure_v2
from aas.statements.history import DAY, HOUR
from test_statements import listener


def test_v2_families_unique_and_well_formed():
    keys = [f.key for f in FAMILIES_V2]
    assert len(keys) == len(set(keys)) and 50 <= len(keys) <= 70
    for f in NEW:
        assert list(f.grid) == sorted(f.grid) and "{th}" in f.hi and "{th}" in f.lo
        for w in ("health", "religio", "sexual", "politic", "depress", "mood"):
            assert w not in (f.hi + f.lo).lower()
    assert set(v2.RELIABILITY) == set(keys)
    assert all(0 < r < 1 for r in v2.RELIABILITY.values())


def test_v2_windows_never_read_the_future():
    h = listener(1, days=800)
    a, _ = h.halves()
    before = measure_v2(a, NEW)
    rows = [(int(t), f"x{int(x)}", f"t{int(tr)}", f"r{int(r)}") for t, x, tr, r in zip(h.t, h.artist, h.track, h.release)]
    late = [(int(h.t[-1]) + DAY + i * 600_000, f"future{i % 7}", f"future{i % 7}|t", None) for i in range(3000)]
    a2 = from_rows(rows + late).window(a.lo, a.hi)
    after = measure_v2(a2, NEW)
    for f, x, y in zip(NEW, before, after):
        assert (math.isnan(x) and math.isnan(y)) or x == pytest.approx(y), f.key


def test_v2_day_families_ignore_time_zone():
    a = measure_v2(listener(2, shift_ms=0).window(), NEW)
    b = measure_v2(listener(2, shift_ms=5 * HOUR).window(), NEW)
    for key in ("day_span", "active_hours", "busy_days", "same_busiest_day", "track_gap", "session_plays"):
        k = [f.key for f in NEW].index(key)
        assert a[k] == pytest.approx(b[k], abs=1e-9), key


def test_track_gap_reads_track_length():
    k = [f.key for f in NEW].index("track_gap")
    v = measure_v2(listener(3).window(), NEW)[k]
    assert 3.3 < v < 4.4           # synthetic plays are 200 to 260 seconds apart
    assert BY_KEY_V2["track_gap"].text("hi", 4.5).endswith("4.5 minutes or more after the one before.")


def _pop(n: int = 40) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    h1, h2 = [], []
    for i in range(n):
        h = listener(300 + i, days=600, explore=float(rng.uniform(0.02, 0.6)), loop=float(rng.uniform(0, 0.5)),
                     per_day=int(rng.integers(10, 80)), skip_days=float(rng.uniform(0, 0.6)))
        a, b = h.halves()
        h1.append(measure_v2(a))
        h2.append(measure_v2(b))
    return np.stack(h1), np.stack(h2)


def test_v2_candidates_count_others_and_respect_the_line():
    h1, _ = _pop()
    tab = Table(h1, families=FAMILIES_V2)
    for i in (0, 5):
        sts = v2.candidates(tab, i, [i, 1, 2, 3, 4] if i == 0 else [i])
        assert sts
        for s in sts:
            assert s.th in s.family.grid and bool(tab.truth(s, i))
            assert s.c / s.n <= v2.LINE
            assert s.n == tab.P - (5 if i == 0 else 1)
        assert len({s.family.key for s in sts}) == len(sts)


def test_v2_lineup_and_fingerprint():
    h1, h2 = _pop()
    first, second = Table(h1, families=FAMILIES_V2), Table(h2, families=FAMILIES_V2)
    r = lu.run(first, second, 3, 5, seed=1, cands=lambda t, i, ex, m: v2.candidates(t, i, ex, m))
    assert r.accuracy > 0.5
    fc = v2.fingerprint_credit(v2.percentiles(h1, h1), v2.percentiles(h1, h2), r.members)
    assert fc.mean() > 0.6
    same = np.repeat(h2[:1], len(h2), axis=0)
    fc2 = v2.fingerprint_credit(v2.percentiles(h1, h1), v2.percentiles(h1, same), r.members)
    assert fc2.mean() == pytest.approx(0.2)


def test_percentiles_are_mid_ranks():
    ref = np.array([[1.0], [2.0], [3.0], [np.nan]])
    p = v2.percentiles(ref, np.array([[2.0], [0.0], [np.nan]]))
    assert p[0, 0] == pytest.approx(0.5) and p[1, 0] == 0 and math.isnan(p[2, 0])


def test_every_card_value_carries_a_unit():
    # a bare count or ratio must say what it counts; the rest carry %, minutes, hours or days
    for f in FAMILIES_V2:
        if f.unit in ("num", "x"):
            assert f.per, f.key
        shown = f.show(3.456)
        assert any(c.isalpha() or c == "%" for c in shown), (f.key, shown)
    assert BY_KEY_V2["track_gap"].show(6.62) == "6.6 minutes between starts"
    assert BY_KEY_V2["sessions_per_day"].show(7.01) == "7.0 sessions a day"
    assert BY_KEY_V2["artists_per_season"].show(3342) == "3,342 artists a season"
    assert BY_KEY_V2["day_span"].show(19.53) == "19.5 hours"
    assert BY_KEY_V2["active_days"].show(0.0142) == "1.4%"
    assert BY_KEY_V2["artist_day_streak"].show(85) == "85 days in a row"
