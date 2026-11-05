"""Evaluation core: build, compare, evaluate, dump. Fake readers and fake people only."""

import json
import re
from datetime import datetime, timezone

import pytest

import random

from aas import quiz
from aas import profile as P
from aas.schema import TRAIT_NAMES
from aas.person import Estimate, Person, Play, TextItem


def fake_person() -> Person:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return Person(
        texts=[TextItem(now, "reddit", "comment", "hello there")] * 3,
        plays=[Play(now, "Artist", "Song", 200_000)],
        sources=["reddit.zip", "spotify.zip"],
        notes=["2 files skipped (not text)"],
    )


class FakeReader:
    def __init__(self, name, ests=(), ok=True, why="", boom=False):
        self.name, self._ests, self._ok, self._why, self._boom = name, list(ests), ok, why, boom

    def can_read(self, person):
        return self._ok, self._why

    def read(self, person):
        if self._boom:
            raise RuntimeError("model file missing")
        return self._ests


def bf(reader, trait, z, lo, hi, r=0.30):
    return Estimate(reader, trait, "bigfive", z, lo, hi, None, r, "r", evidence="1,204 comments")


E_TEXT = bf("text-bigfive", "E", -0.5, -1.2, 0.2)
E_MUSIC = bf("music-bigfive", "E", 0.4, -0.3, 1.1, r=0.12)
O_TEXT = bf("text-bigfive", "O", 1.0, 0.6, 1.4)
TF = Estimate("text-mbti", "T/F", "mbti", 0.78, 0.70, 0.86, None, 0.83, "auc", evidence="1,204 comments")
IE = Estimate("text-mbti", "I/E", "mbti", 0.70, 0.6, 0.8, None, 0.73, "auc")
SKIPS = Estimate("music", "skip rate", "measured", 0.31, evidence="9,810 plays")
UNUSABLE = Estimate("text-mbti", "J/P", "mbti", 0.55, usable=False, evidence="only 40 words about plans")


def readers():
    return [
        FakeReader("text-bigfive", [E_TEXT, O_TEXT]),
        FakeReader("music-bigfive", [E_MUSIC]),
        FakeReader("text-mbti", [TF, IE, UNUSABLE]),
        FakeReader("music", [SKIPS]),
        FakeReader("watch-reader", ok=False, why="no YouTube history in these exports"),
        FakeReader("broken", boom=True),
    ]


def quiz_result(**z):
    return {t: {"trait": TRAIT_NAMES[t], "z": v, "percentile": 100 * P.phi(v), "form": "short"} for t, v in z.items()}


# ---------------------------------------------------------------- build
def test_build_groups_and_keeps_reasons():
    prof = P.build(fake_person(), readers())
    assert [e.reader for e in prof.bigfive["E"]] == ["text-bigfive", "music-bigfive"]  # both, not averaged
    assert prof.bigfive["O"] == [O_TEXT]
    assert [e.trait for e in prof.mbti] == ["I/E", "T/F"]  # sorted, J/P held back
    assert prof.measured == [SKIPS]
    reasons = {(u.reader, u.trait): u.reason for u in prof.unread}
    assert reasons[("watch-reader", None)] == "no YouTube history in these exports"
    assert reasons[("text-mbti", "J/P")] == "only 40 words about plans"
    assert "model file missing" in reasons[("broken", None)]
    assert prof.sources == ["reddit.zip", "spotify.zip"]












# ---------------------------------------------------------------- compare
def test_compare_inside_and_outside():
    prof = P.build(fake_person(), readers())
    comp = P.compare(prof, quiz_result(O=0.0, E=-0.9, A=0.3, C=0.0, N=0.0))
    direct = {(c.reader, c.trait): c for c in comp.checks if c.via == "bigfive"}
    assert direct[("text-bigfive", "E")].inside is True    # -0.9 in [-1.2, 0.2]
    assert direct[("music-bigfive", "E")].inside is False  # -0.9 not in [-0.3, 1.1]
    assert direct[("text-bigfive", "O")].inside is False   # 0.0 not in [0.6, 1.4]
    assert direct[("text-bigfive", "E")].same_side is True
    assert "1 of 3 Big Five guesses" in comp.overall
    loose = {c.via: c for c in comp.checks if c.via != "bigfive"}
    assert loose["I/E"].trait == "E" and loose["I/E"].same_side is True   # leans I, quiz E low
    assert loose["T/F"].trait == "A" and loose["T/F"].same_side is False  # leans T, quiz A high
    assert "Loose MBTI checks: 1 of 2" in comp.overall
    assert set(comp.missing) == {"C", "N"}


def test_compare_with_real_quiz_scoring():
    answers = {c: 3 for c in quiz.short_form()}
    result = quiz.score(answers)
    prof = P.build(fake_person(), readers())
    comp = P.compare(prof, result)
    assert comp.checks and all(isinstance(c.quiz_z, float) for c in comp.checks)
    assert "vs quiz" in P.dump(prof, comp)


def test_compare_without_intervals():
    prof = P.build(Person(), [FakeReader("x", [Estimate("x", "E", "bigfive", 0.2)])])
    comp = P.compare(prof, quiz_result(E=0.1))
    assert comp.checks[0].inside is None
    assert "nothing to score strictly" in comp.overall


def test_unusable_never_becomes_a_claim():
    prof = P.build(fake_person(), readers())
    assert all(e.usable for e in prof.estimates)
    assert "J/P" not in {e.trait for e in prof.estimates}
    assert not prof.to_dict()["estimates"] or all(e["usable"] for e in prof.to_dict()["estimates"])


def test_two_readers_same_trait_not_averaged():
    prof = P.build(fake_person(), readers())
    assert [e.value for e in prof.bigfive["E"]] == [-0.5, 0.4]
    out = P.dump(prof)
    assert "shown separately" in out and "-0.050" not in out


def test_nothing_usable():
    prof = P.build(Person(), [FakeReader("t", [UNUSABLE])])
    assert prof.estimates == [] and prof.unread[0].reason == "only 40 words about plans"


def test_dump_lists_reasons_and_checks():
    prof = P.build(fake_person(), readers())
    out = P.dump(prof, P.compare(prof, quiz_result(E=-0.9, O=0.0)))
    assert "watch-reader: no YouTube history" in out
    assert "OUTSIDE" in out and "inside" in out and "loose via I/E" in out


# ---------------------------------------------------------------- many people
def _simulate(n, noise, half_width, seed=0):
    """People whose true z ~ N(0,1); a reader guesses true + N(0, noise), interval guess +/- half_width."""
    rng = random.Random(seed)
    pairs = []
    for _ in range(n):
        truth = rng.gauss(0, 1)
        g = truth + rng.gauss(0, noise)
        r = FakeReader("sim", [bf("sim", "E", g, g - half_width, g + half_width)])
        pairs.append((P.build(Person(), [r, FakeReader("deaf", ok=False, why="no data")]), quiz_result(E=truth)))
    return pairs


def test_evaluate_coverage_is_calibrated_when_intervals_are_honest():
    # 80% interval for N(0, 0.5) error is +/- 1.2816 * 0.5
    ev = P.evaluate(_simulate(4000, 0.5, 1.2816 * 0.5), level=0.80)
    (s,) = ev.scores
    assert s.n == 4000 and abs(s.coverage - 0.80) < 0.03
    assert s.skill > 0.4 and s.r > 0.8           # a good reader beats guessing average
    assert ev.skipped == {"deaf": 4000}


def test_evaluate_catches_overconfident_intervals():
    ev = P.evaluate(_simulate(2000, 1.0, 0.3, seed=1))
    assert ev.scores[0].coverage < 0.4


def test_evaluate_useless_reader_has_no_skill():
    rng = random.Random(2)
    pairs = []
    for _ in range(2000):
        g = rng.gauss(0, 1)   # guess unrelated to the truth
        pairs.append((P.build(Person(), [FakeReader("noise", [bf("noise", "O", g, g - 1, g + 1)])]),
                      quiz_result(O=rng.gauss(0, 1))))
    s = P.evaluate(pairs).scores[0]
    assert s.skill < 0 and abs(s.r) < 0.1          # worse than predict-the-average
    assert "skill -" in s.line()
