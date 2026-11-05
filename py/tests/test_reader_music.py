"""MusicReader on synthetic plays: an explorer and a loyal listener land on opposite ends of the
explore dial; a fixed-hour listener and an any-hour listener on opposite ends of the time dial;
thin histories are refused with a reason; receipts name the person's own artists."""

from __future__ import annotations

import json
import math
import random
from datetime import UTC, datetime, timedelta

import pytest

from aas.person import Person, Play
from aas.readers import music
from aas.readers.music import MusicReader

START = datetime(2024, 1, 1, tzinfo=UTC)


def history(days: int, per_day: int, artist_for, hour_for, seed: int = 0, spotify: bool = True) -> list[Play]:
    """per_day plays a day for `days` days. artist_for(day, i, rng) picks the artist, hour_for(day, rng) the hour."""
    rng = random.Random(seed)
    out = []
    for d in range(days):
        h = hour_for(d, rng)
        t = START + timedelta(days=d, hours=h)
        for i in range(per_day):
            ms = rng.choice([200_000, 210_000, 15_000]) if spotify else None
            t = t + timedelta(milliseconds=ms or 180_000)
            out.append(Play(when=t, artist=artist_for(d, i, rng), track=f"t{i}", ms_played=ms,
                            reason_end=("trackdone" if ms and ms > 30_000 else "fwdbtn") if spotify else None))
    return out


def explorer(d, i, rng):
    # mostly artists met this week, a few from a small fixed pool
    return f"New {d // 7}-{rng.randrange(25)}" if rng.random() < 0.9 else f"Pool {rng.randrange(2)}"


def loyal(d, i, rng):
    return f"Old {rng.randrange(15)}"


fixed_hour = lambda d, rng: 8 + rng.random() * 0.5
any_hour = lambda d, rng: rng.random() * 24


def by_trait(ests):
    return {e.trait: e for e in ests}


@pytest.fixture(scope="module")
def reader():
    r = MusicReader()
    if r.norms is None or r.dials is None:
        pytest.skip("music_norms.json / music_dials.json not built (scripts/build_music_norms.py)")
    return r


def test_explorer_vs_loyal_on_dial_one(reader):
    ex = by_trait(reader.read(Person(plays=history(540, 12, explorer, any_hour, 1))))
    lo = by_trait(reader.read(Person(plays=history(540, 12, loyal, any_hour, 2))))
    assert ex["dial_explore"].usable and lo["dial_explore"].usable
    assert ex["dial_explore"].value > 1 and lo["dial_explore"].value < -1
    assert ex["dial_explore"].percentile > 90 and lo["dial_explore"].percentile < 10
    # and the traits underneath point the same way
    assert ex["new_share"].value > 0.5 and lo["new_share"].value < 0.01
    assert lo["old_friends"].value > 0.95 and ex["old_friends"].value < 0.3
    assert ex["season_turnover"].value >= 0.7 and lo["season_turnover"].value < 0.6


def test_fixed_hour_vs_any_hour_on_dial_two(reader):
    fx = by_trait(reader.read(Person(plays=history(300, 10, loyal, fixed_hour, 3))))
    an = by_trait(reader.read(Person(plays=history(300, 10, loyal, any_hour, 4))))
    assert fx["dial_regular"].value > 1 and an["dial_regular"].value < -1
    assert fx["clock_regularity"].value > 0.95 and an["clock_regularity"].value < 0.3
    assert fx["clock_spread"].value < an["clock_spread"].value


def test_every_estimate_is_measured_with_evidence(reader):
    ests = reader.read(Person(plays=history(540, 12, explorer, any_hour, 5)))
    assert {e.trait for e in ests} == set(music.RETEST) | {"dial_explore", "dial_regular"}
    for e in ests:
        assert e.kind == "measured" and e.reader == "music" and e.evidence
        if e.usable:
            assert math.isfinite(e.value)
    t = by_trait(ests)
    # receipts use the person's own artists
    assert "New " in t["new_share"].evidence
    assert "listeners we measured" in t["new_share"].evidence
    assert 0 <= t["new_share"].percentile <= 100


def test_finish_and_skips_only_with_that_data(reader):
    sp = by_trait(reader.read(Person(plays=history(200, 12, loyal, any_hour, 6, spotify=True))))
    assert sp["finish_share"].usable and 0.55 < sp["finish_share"].value < 0.8   # 2 of 3 lengths finish
    assert sp["skip_seconds"].usable and sp["skip_seconds"].value == 15.0
    assert sp["finish_share"].percentile is None   # no reference population for these
    lb = by_trait(reader.read(Person(plays=history(200, 12, loyal, any_hour, 6, spotify=False))))
    assert not lb["finish_share"].usable and not lb["skip_seconds"].usable
    assert "Not enough data" in lb["skip_seconds"].evidence
    assert lb["clock_spread"].usable  # unknown lengths still count as full plays


def test_thin_histories_are_refused(reader):
    few = Person(plays=history(150, 5, loyal, any_hour, 7))      # 750 plays
    ok, why = reader.can_read(few)
    assert not ok and "2,000" in why
    short = Person(plays=history(60, 50, loyal, any_hour, 8))    # 3,000 plays in 2 months
    ok, why = reader.can_read(short)
    assert not ok and "3 months" in why
    ests = reader.read(short)
    assert ests and all(not e.usable and e.evidence == why for e in ests)


def test_trait_needing_more_time_says_so(reader):
    # 4 months: enough for the clock and new share, not for old friends (needs ~8) or turnover (~9)
    t = by_trait(reader.read(Person(plays=history(120, 25, explorer, any_hour, 9))))
    assert t["new_share"].usable and t["clock_spread"].usable
    assert not t["old_friends"].usable and "8 months" in t["old_friends"].evidence
    assert not t["season_turnover"].usable and not t["dial_explore"].usable


def test_new_share_by_hand():
    # 12 weeks (the warm-up) of one artist, then 5 weeks where a third of each day's plays go to an
    # artist first heard that day -> exactly 1/3 of the scored plays are "fresh"
    plays = []
    for d in range(7 * 13 + 28):
        for i in range(12):
            when = START + timedelta(days=d, minutes=10 * i)
            a = "Old" if d < 7 * 12 or i % 3 else f"Fresh {d}"
            plays.append(Play(when=when, artist=a, track="x", ms_played=200_000))
    tr = music.measure(music.arrays_from_plays(plays))
    assert tr["new_share"].value == pytest.approx(1 / 3, abs=1e-9)


def test_norms_hold_no_rows():
    norms = json.loads((music.DATA / "music_norms.json").read_text())
    for block in list(norms["traits"].values()) + list(norms["dials"].values()):
        assert set(block) == {"n", "p", "v"} and len(block["v"]) == 99
    assert norms["n"] >= 290
