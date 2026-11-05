"""End to end: the invented person through the v1 default readers, profile.build, fusion and dump()."""

from __future__ import annotations

import math

import pytest

from aas import example
from aas import profile as P
from aas.readers import text as R
from aas.schema import TRAITS

pytestmark = pytest.mark.skipif(
    not all((R.MODELS_DIR / d / "model.joblib").exists() for d in ("text_bigfive", "text_mbti")),
    reason="run scripts/train_text_readers.py first")


@pytest.fixture(scope="module")
def prof():
    return P.build(example.synthetic_person(), example.default_readers())


def test_every_default_reader_speaks(prof):
    assert prof.readers_run == ["text-bigfive", "text-mbti", "music", "fused"]
    assert "text-distilled" not in prof.readers_run


def test_fused_bigfive_for_every_trait_with_80_intervals(prof):
    for t in TRAITS:
        fused = [e for e in prof.bigfive[t] if e.reader == "fused"]
        assert len(fused) == 1
        e = fused[0]
        assert e.usable and e.low < e.value < e.high and e.interval_level == 0.80
        assert e.validity is None                       # untested on people with several sources
        assert 2.3 < e.high - e.low < 2.6                # near-chance sources: close to the +-1.28 population interval
        assert "implied, not tested" in e.evidence


def test_openness_refused_by_text_and_fused_from_mbti_only(prof):
    assert not [e for e in prof.bigfive["O"] if e.reader == "text-bigfive"]
    assert any(u.reader == "text-bigfive" and u.trait == "O" and "not shown to work" in u.reason for u in prof.unread)
    (fo,) = prof.bigfive["O"]
    assert "text-mbti:N/S" in fo.evidence and "text-bigfive" not in fo.evidence


def test_mbti_and_music(prof):
    assert [e.trait for e in prof.mbti] == ["I/E", "N/S", "T/F", "J/P"]
    assert all(0 < e.value < 1 and e.validity_kind == "auc" for e in prof.mbti)
    m = {e.trait: e for e in prof.measured}
    assert {"new_share", "old_friends", "season_turnover", "clock_spread", "clock_regularity",
            "dial_explore", "dial_regular"} <= set(m)
    assert all(e.interval_level is None and e.validity is None for e in prof.measured)
    assert all(math.isfinite(e.value) for e in prof.measured)


def test_dump_and_dict(prof):
    out = P.dump(prof)
    for s in ("readers run: text-bigfive, text-mbti, music, fused", "bigfive O (z)", "mbti", "measured", "unread"):
        assert s in out
    d = prof.to_dict()
    assert len(d["estimates"]) == len(prof.estimates) and d["unread"]


def test_thin_person_refuses_with_reasons():
    p = example.synthetic_person()
    p.texts = [t for t in p.texts if t.source == "x"][:5]
    p.plays = p.plays[:300]
    prof = P.build(p, example.default_readers())
    reasons = {u.reader: u.reason for u in prof.unread if u.trait is None}
    assert "at least 100" in reasons["text-mbti"] and "2,000" in reasons["music"]
    assert not prof.mbti and not prof.measured
