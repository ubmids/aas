"""Identity in k sentences: the bits estimate, the closed-form expected credit, the tie-aware rank
credit, the hold-probability model and the joint selection on a small synthetic population."""

from __future__ import annotations

import itertools
import math

import numpy as np

from aas.statements import identity as idn
from aas.statements.families_v2 import FAMILIES_V2


def test_kt_bits():
    assert math.isclose(float(idn.kt_bits(0, 99)), math.log2(200))
    assert math.isclose(float(idn.kt_bits(99.5, 99)), 0.0, abs_tol=1e-12)
    assert idn.kt_bits(1, 100) < idn.kt_bits(0, 100)


def test_credit_closed_form_matches_enumeration():
    rng = np.random.default_rng(0)
    for D in (1, 4, 7):
        t = rng.dirichlet(np.ones(4))
        dd = rng.dirichlet(np.ones(4))
        dd[2] = 0
        dd /= dd.sum()
        brute = 0.0
        for tt in range(4):
            for decs in itertools.product(range(4), repeat=D):
                p = t[tt] * np.prod([dd[x] for x in decs])
                if p == 0 or max(decs) > tt:
                    continue
                brute += p / (1 + sum(x == tt for x in decs))
        assert math.isclose(float(idn.credit_given(t, dd, D)), brute, rel_tol=1e-9, abs_tol=1e-12)


def test_rank_credit_ties():
    s = np.array([3, 3, 1, 3, 0])
    members = np.arange(5)
    assert math.isclose(idn.rank_credit(s, 0, members), 1 / 3)
    assert idn.rank_credit(s, 0, members, top=3) == 1.0
    assert math.isclose(idn.rank_credit(s, 0, members, top=2), 2 / 3)
    assert idn.rank_credit(s, 2, members) == 0.0
    assert idn.rank_credit(-s, 4, members) == 1.0


def test_normal_scores_order_and_nan():
    ref = np.array([[1.0], [2.0], [3.0], [np.nan]])
    z = idn.normal_scores(ref, np.array([[0.0], [2.0], [9.0], [np.nan]]))
    assert z[0, 0] < z[1, 0] < z[2, 0] and math.isclose(z[1, 0], 0.0, abs_tol=1e-12) and np.isnan(z[3, 0])


def population(P=120, seed=1):
    rng = np.random.default_rng(seed)
    F = len(FAMILIES_V2)
    lo = np.array([min(f.grid) for f in FAMILIES_V2])
    hi = np.array([max(f.grid) for f in FAMILIES_V2])
    mu = rng.uniform(lo, hi, (P, F))
    noise = 0.05 * (hi - lo)
    draw = lambda: mu + rng.normal(0, noise, (P, F))  # noqa: E731
    return idn.Population(draw(), draw(), draw(), draw(), FAMILIES_V2), mu


def test_hold_probability_only_for_true_statements_and_in_range():
    pop, _ = population()
    assert ((pop.Q > 0) <= pop.T1).all()
    assert (pop.Qall >= 0).all() and (pop.Qall <= 1).all()
    assert (pop.r >= idn.R_FLOOR).all() and (pop.r <= idn.R_CEIL).all()


def test_joint_select_true_unique_families_and_finds_people():
    pop, _ = population()
    P = pop.P
    hits = 0
    for i in range(20):
        ref = np.delete(np.arange(P), i)
        ch = idn.joint_select(pop, i, ref, 3, P - 1)
        assert len(ch) == 3
        assert pop.T1[i, ch].all()
        assert len({int(pop.cs.fam[c]) for c in ch}) == 3
        groups = [pop.groups[c] for c in ch]
        assert max(groups.count(g) for g in groups) <= idn.PER_GROUP
        s = pop.T2[:, ch].sum(1)
        hits += idn.rank_credit(s, i, np.arange(P), top=5)
    assert hits / 20 > 0.5


def test_set_bits_shrink_with_more_statements():
    pop, _ = population()
    i, ref = 0, np.arange(1, pop.P)
    ch = idn.joint_select(pop, i, ref, 5, pop.P - 1)
    b = [idn.set_bits(pop, i, ref, ch, k)[0] for k in range(0, len(ch) + 1)]
    assert b[0] == 0.0 or math.isclose(b[0], float(idn.kt_bits(len(ref), len(ref))))
    assert all(x <= y + 1e-12 for x, y in zip(b, b[1:]))


def test_mahalanobis_identity_is_euclid():
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=(4, 3)), rng.normal(size=(5, 3))
    d = idn.mahalanobis_dist(a, b, np.eye(3))
    assert np.allclose(d, ((a[:, None] - b[None]) ** 2).sum(-1))
