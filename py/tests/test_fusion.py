"""Fusion: missing sources, single source, coverage, no source -> no claim, FusedReader in build()."""

from math import sqrt

import numpy as np
import pytest

from aas import profile as P
from aas.fusion import (CVPlus, Fuser, FusedReader, MBTI_LINKS, Reading, Stack, auc_to_r,
                        gaussian_posterior, make_method, prior_error_corr, r_to_auc,
                        readings_from_estimates)
from aas.person import Estimate, Person
from aas.schema import TRAITS


# ---------------------------------------------------------------- helpers

def bf(reader, trait, z, r=0.30):
    return Estimate(reader, trait, "bigfive", z, z - 1.2, z + 1.2, None, r, "r")


def mbti(reader, axis, p, auc=0.787):
    return Estimate(reader, axis, "mbti", p, p - 0.1, p + 0.1, None, auc, "auc")


class FakeReader:
    def __init__(self, name, ests=(), ok=True, why=""):
        self.name, self._ests, self._ok, self._why = name, list(ests), ok, why

    def can_read(self, person):
        return self._ok, self._why

    def read(self, person):
        return self._ests


def sim(n, rng, v=(0.30, 0.25, 0.12), err_r=0.4, p_have=(0.7, 0.6, 0.4)):
    """One trait: y ~ N(0,1), three sources; first two share text (correlated errors)."""
    y = rng.standard_normal(n)
    R = np.array([[1, err_r, 0.1], [err_r, 1, 0.1], [0.1, 0.1, 1]])
    E = rng.multivariate_normal(np.zeros(3), R, size=n)
    v = np.array(v)
    U = v * y[:, None] + np.sqrt(1 - v ** 2) * E
    have = rng.random((n, 3)) < np.array(p_have)
    U[~have] = np.nan
    V = np.where(have, v, np.nan)
    Vsd = np.where(have, 0.1, np.nan)
    return y, U, V, Vsd, ["text", "text", "music"]


# ---------------------------------------------------------------- maths

def test_auc_r_round_trip_and_known_value():
    assert auc_to_r(0.5) == pytest.approx(0.0, abs=1e-12)
    assert auc_to_r(1.0) == pytest.approx(1.0, abs=1e-9)
    assert auc_to_r(0.682) == pytest.approx(0.40, abs=0.005)    # text-bigfive O: card says r_implied 0.40
    for r in (0.05, 0.2, 0.5, 0.8):
        assert auc_to_r(r_to_auc(r)) == pytest.approx(r, abs=1e-9)


def test_correlated_errors_limit_the_gain():
    v = np.array([0.3, 0.3])
    _, var0, _ = gaussian_posterior(np.zeros((1, 2)), v, np.eye(2))
    _, var8, _ = gaussian_posterior(np.zeros((1, 2)), v, np.array([[1, 0.8], [0.8, 1]]))
    r0, r8 = sqrt(1 - var0[0]), sqrt(1 - var8[0])
    assert r0 > r8 > 0.30
    assert r0 == pytest.approx(0.406, abs=0.002)


# ---------------------------------------------------------------- readings

def test_mbti_reading_sign_and_validity():
    rs, _ = readings_from_estimates([mbti("text-mbti", "I/E", 0.9), mbti("text-mbti", "J/P", 0.9)])
    by = {r.trait: r for r in rs}
    assert by["E"].u < 0                    # likely Introvert -> lower extraversion
    assert by["C"].u > 0                    # likely Judging -> higher conscientiousness
    assert by["E"].validity == pytest.approx(auc_to_r(0.787) * MBTI_LINKS["I/E"][2])
    assert by["E"].group == by["C"].group == "text"


def test_bigfive_reading_undoes_reader_shrinkage():
    rs, _ = readings_from_estimates([bf("text-bigfive", "O", 0.12, r=0.40)])
    assert rs[0].u == pytest.approx(0.30)
    assert rs[0].validity == pytest.approx(0.40)


def test_unusable_and_untested_estimates_are_left_out():
    e_bad = Estimate("text-bigfive", "O", "bigfive", 0.5, usable=False)
    e_untested = Estimate("x-bigfive", "C", "bigfive", 0.5)
    rs, skipped = readings_from_estimates([e_bad, e_untested])
    assert rs == []
    assert any("no tested validity" in s for s in skipped)


# ---------------------------------------------------------------- person-level Fuser

def test_single_source_equals_that_source():
    e = bf("text-bigfive", "E", -0.20, r=0.25)
    rs, _ = readings_from_estimates([e])
    exact = Fuser("bayes", discount=1.0)
    out = exact.predict(rs)
    # with validity uncertainty integrated over, the mean is close to (not exactly) the reader's value
    assert out["E"].value == pytest.approx(e.value, abs=0.02)
    # without uncertainty it is exactly v * u = the reader's own value
    m = make_method("bayes", 0, discount=1.0)
    m.base.draws = 0
    m.fit(None, [], None, None, ["text"])
    mu, lo, hi = m.predict(np.array([[rs[0].u]]), np.array([[0.25]]), np.array([[0.1]]))
    assert mu[0] == pytest.approx(e.value, abs=1e-9)
    assert hi[0] - lo[0] == pytest.approx(2 * 1.2815515655446004 * sqrt(1 - 0.25 ** 2), abs=1e-6)
    assert set(out) == {"E"}


def test_implied_r_describes_the_sources_not_the_reading():
    """A strong reading must not lower the stated accuracy of the source set (it printed 'r about 0.00')."""
    rho = auc_to_r(0.62)
    want = 0.75 * rho * MBTI_LINKS["N/S"][2]          # one source: its discounted validity
    for p in (0.5, 0.7, 0.95):
        rs, _ = readings_from_estimates([mbti("text-mbti", "N/S", p, auc=0.62)])
        assert Fuser("bayes").predict(rs)["O"].implied_r == pytest.approx(want, abs=1e-6)


def test_missing_sources_only_traits_with_evidence():
    rs, _ = readings_from_estimates([mbti("text-mbti", "T/F", 0.2)])     # speaks to A only
    out = Fuser("bayes").predict(rs)
    assert set(out) == {"A"}
    f = out["A"]
    assert f.low < f.value < f.high and np.isfinite([f.value, f.low, f.high]).all()
    assert f.value > 0                         # likely Feeling -> above-average agreeableness


def test_no_source_no_claim():
    assert Fuser("bayes").predict([]) == {}
    fr = FusedReader([FakeReader("text-bigfive", ok=False, why="too little writing")])
    ok, why = fr.can_read(Person())
    assert not ok and "nothing to combine" in why
    fr2 = FusedReader([FakeReader("text-mbti", [Estimate("text-mbti", "I/E", "mbti", 0.6, usable=False)])])
    ests = fr2.read(Person())
    assert len(ests) == 5 and not any(e.usable for e in ests)


def test_more_sources_narrower_and_more_accurate_on_average():
    rng = np.random.default_rng(1)
    y, U, V, Vsd, g = sim(20000, rng)
    m = make_method("bayes", 0, discount=1.0).fit(None, [], None, None, g)
    mu, lo, hi = m.predict(U, V, Vsd)
    k = (~np.isnan(U)).sum(1)
    one, three = k == 1, k == 3
    assert np.corrcoef(mu[three], y[three])[0, 1] > np.corrcoef(mu[one], y[one])[0, 1]
    assert np.mean(hi[three] - lo[three]) < np.mean(hi[one] - lo[one])
    assert np.isnan(mu[k == 0]).all()          # nobody with no source gets a number


# ---------------------------------------------------------------- coverage

@pytest.mark.parametrize("name,n_train", [("bayes", 0), ("stack", 250), ("bayes_eb_cp", 250), ("precision", 250)])
def test_coverage_near_nominal(name, n_train):
    """Conformal coverage is guaranteed on average over training sets; one set of 250 people moves it by
    about +-0.025 (sd), so average over 6 training sets."""
    rng = np.random.default_rng(7)
    covs = []
    for _ in range(6 if n_train else 1):
        m = make_method(name, n_train)
        if n_train:
            y, U, V, Vsd, g = sim(n_train, rng)
            m.fit(U, y, V, Vsd, g)
        else:
            m.fit(None, [], None, None, ["text", "text", "music"])
        y, U, V, Vsd, _ = sim(4000, rng)
        mu, lo, hi = m.predict(U, V, Vsd)
        ok = np.isfinite(mu)
        covs.append(np.mean((lo[ok] <= y[ok]) & (y[ok] <= hi[ok])))
    assert 0.77 <= np.mean(covs) <= 0.84, covs


def test_cvplus_holds_coverage_even_when_the_model_is_wrong():
    """Stated validities far too high (a source that says r 0.6, truly 0.1): conformal still covers."""
    rng = np.random.default_rng(3)
    y, U, V, Vsd, g = sim(300, rng, v=(0.10, 0.05, 0.05))
    V = np.where(np.isnan(V), np.nan, 0.6)
    cp = CVPlus(lambda: Stack()).fit(U, y, V, Vsd, g)
    y2, U2, V2, Vsd2, _ = sim(6000, rng, v=(0.10, 0.05, 0.05))
    V2 = np.where(np.isnan(V2), np.nan, 0.6)
    mu, lo, hi = cp.predict(U2, V2, Vsd2)
    ok = np.isfinite(mu)
    assert 0.76 <= np.mean((lo[ok] <= y2[ok]) & (y2[ok] <= hi[ok])) <= 0.86


def test_fitted_fuser_person_api():
    rng = np.random.default_rng(5)
    people, quizzes = [], []
    for _ in range(200):
        z = rng.standard_normal(5)
        rs = []
        for ti, t in enumerate(TRAITS):
            rs.append(Reading("text-bigfive", t, 0.3 * z[ti] + sqrt(1 - 0.09) * rng.standard_normal(), 0.3, 0.1, "text"))
            if rng.random() < 0.5:
                rs.append(Reading("music", t, 0.1 * z[ti] + sqrt(0.99) * rng.standard_normal(), 0.1, 0.1, "music"))
        people.append(rs)
        quizzes.append(dict(zip(TRAITS, z)))
    f = Fuser("bayes_eb").fit(people, quizzes)
    out = f.predict([Reading("text-bigfive", "O", 2.0, 0.3, 0.1, "text"), Reading("sensors", "O", 1.0, 0.1, 0.1, "sensors")])
    assert out["O"].value > 0
    assert any("sensors" in s for s in out["O"].ignored)       # a source the model never saw is not guessed at


# ---------------------------------------------------------------- wiring into profile.build

def test_fused_reader_in_profile_build_and_evaluate():
    text_bf = FakeReader("text-bigfive", [bf("text-bigfive", t, 0.3) for t in TRAITS])
    text_mbti = FakeReader("text-mbti", [mbti("text-mbti", "I/E", 0.2)])
    prof = P.build(Person(), [text_bf, text_mbti, FusedReader([text_bf, text_mbti])])
    assert "fused" in prof.readers_run
    fused_E = [e for e in prof.bigfive["E"] if e.reader == "fused"][0]
    plain_E = [e for e in prof.bigfive["E"] if e.reader == "text-bigfive"][0]
    # P(I) = 0.2 says extravert: the fused E moves above the text-only estimate
    assert fused_E.value > plain_E.value
    assert fused_E.interval_level == 0.80 and fused_E.validity is None      # not tested yet: no claim
    assert "text-mbti:I/E" in fused_E.evidence
    quiz = {t: {"z": 0.5, "percentile": 69} for t in TRAITS}
    ev = P.evaluate([(prof, quiz)])
    assert any(s.reader == "fused" for s in ev.scores)
