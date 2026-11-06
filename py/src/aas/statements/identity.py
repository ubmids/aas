"""Identity in k sentences (py/docs/identity.md): how many bits of identity a statement and a set of
statements carry, joint selection of the set, 1-of-N identification and a learned fingerprint metric.

Notation. A population of people with family values on windows: H1 (first half of the training
weeks), H2 (second half), and H1a, H1b (the two halves of H1). A statement s = (family f, direction,
round threshold) is true of a window x when x_f >= th ("hi") or x_f <= th ("lo"); a value that cannot
be measured makes it false.

Bits. Against a reference of n people (the person held out), a set S of statements that c_S of them
fit carries

    b(S) = -log2((c_S + 1/2) / (n + 1))

bits (the Krichevsky-Trofimov estimate of the share, so c_S = 0 gives log2(2n + 2) instead of
infinity). Singling one person out of N takes log2 N bits.

Hold probability. Every value is put on a normal score z = Phi^-1(mid-rank percentile) within the
population's values for the same kind of window. Per family, r_f is the correlation of z(H1a) with
z(H1b) across people (how much of a window's score is the person, the rest being window noise). A
person's score is their level plus noise, z = mu + e, with Var(mu) = r_f and their own noise variance
shrunk from the family's 1 - r_f towards what their two quarters show,

    tau_if^2 = (LAMBDA (1 - r_f) + (z_a - z_b)^2 / 2) / (LAMBDA + 1).

Given z1 = z(H1), the level is mu ~ N(z1 r_f / (r_f + tau^2), r_f tau^2 / (r_f + tau^2)) and the next
window's score z' = mu + e' has mean m = z1 r_f / (r_f + tau^2) and variance
v = KAPPA (r_f tau^2 / (r_f + tau^2) + tau^2). The chance that a statement holds on the next window is

    q_is = Phi(sign * (m - z(th)) / sqrt(v)),   sign = +1 for "hi", -1 for "lo",

which pulls extreme people back towards the middle (regression to the mean). With tau^2 = 1 - r_f it
is the bivariate normal Phi(sign (r_f z1 - z(th)) / sqrt(1 - r_f^2)). Only H1, H1a and H1b enter:
nothing from the second half.

Expected credit. The lineup matcher counts how many of the target's statements are true on each
candidate's second half and picks the highest, an m-way tie scoring 1/m. With D decoys drawn from the
reference, write P_T(t) for the distribution of the target's count (independent Bernoulli(q_s)), and
G(t) = P(decoy count < t), E(t) = P(decoy count = t) for a decoy drawn from the reference (each
reference person j counts statement s with probability a_js: their own q when "soft", the 0/1 truth
on their first half when "hard"). Then

    credit(S) = sum_t P_T(t) * [(G(t) + E(t))^(D+1) - G(t)^(D+1)] / ((D + 1) E(t))

(the expected 1/m over ties; G(t)^D when E(t) = 0). Joint selection adds, one at a time, the true
statement that most raises credit(S), at most one per family and at most PER_GROUP per group.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import ndtri
from scipy.stats import norm

from .families import Family

LAMBDA = 2.0
KAPPA = 1.44
R_FLOOR, R_CEIL = 0.05, 0.99
PER_GROUP = 2
RIDGE = 1e-3        # ridge on the within-person covariance, as a share of its mean variance
SHRINK = 0.0        # within-person covariance shrinkage towards its diagonal (fingerprint)
POOL = 200          # statements kept for the greedy steps after the first


def kt_bits(c, n):
    """-log2 of the Krichevsky-Trofimov share (c + 1/2) / (n + 1)."""
    return -np.log2((np.asarray(c, float) + 0.5) / (n + 1.0))


def normal_scores(ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Phi^-1 of the mid-rank percentile of each x within the finite values of ref, column by column,
    clipped to [1/(2m), 1 - 1/(2m)] for m reference values. NaN stays NaN."""
    out = np.full(x.shape, np.nan)
    for k in range(x.shape[1]):
        r = np.sort(ref[:, k][np.isfinite(ref[:, k])])
        m = len(r)
        ok = np.isfinite(x[:, k])
        if m:
            p = (np.searchsorted(r, x[ok, k], "left") + np.searchsorted(r, x[ok, k], "right")) / 2 / m
            out[ok, k] = ndtri(np.clip(p, 0.5 / m, 1 - 0.5 / m))
    return out


@dataclass
class Cand:
    """Every grid statement: family index, direction (+1 hi, -1 lo), threshold."""
    fam: np.ndarray
    sign: np.ndarray
    th: np.ndarray
    gidx: np.ndarray

    def __len__(self) -> int:
        return len(self.fam)


def all_statements(families: list[Family]) -> Cand:
    fam, sign, th, gi = [], [], [], []
    for k, f in enumerate(families):
        for d in f.dirs:
            for j, g in enumerate(f.grid):
                fam.append(k)
                sign.append(1 if d == "hi" else -1)
                th.append(float(g))
                gi.append(j)
    return Cand(np.array(fam), np.array(sign), np.array(th), np.array(gi))


def truth(X: np.ndarray, cs: Cand) -> np.ndarray:
    """people x statements, True where the statement is true of the window."""
    v = X[:, cs.fam]
    with np.errstate(invalid="ignore"):
        t = np.where(cs.sign > 0, v >= cs.th, v <= cs.th)
    return t & np.isfinite(v)


class Population:
    """Values of one population on H1, H2, H1a, H1b and everything derived from them that uses no H2."""

    def __init__(self, H1, H2, H1a, H1b, families: list[Family], lam: float = LAMBDA, kappa: float = KAPPA):
        self.H1, self.H2, self.H1a, self.H1b = (np.asarray(a, float) for a in (H1, H2, H1a, H1b))
        self.P = self.H1.shape[0]
        self.families = families
        self.cs = all_statements(families)
        self.T1 = truth(self.H1, self.cs)          # people x statements, first halves
        self.T2 = truth(self.H2, self.cs)          # second halves (only for scoring)
        self.groups = np.array([families[k].group for k in self.cs.fam])
        self.set_noise(lam, kappa)

    def set_noise(self, lam: float, kappa: float) -> None:
        z1 = normal_scores(self.H1, self.H1)
        za = normal_scores(self.H1a, self.H1a)
        zb = normal_scores(self.H1b, self.H1b)
        F = z1.shape[1]
        r = np.zeros(F)
        for k in range(F):
            ok = np.isfinite(za[:, k]) & np.isfinite(zb[:, k])
            r[k] = np.corrcoef(za[ok, k], zb[ok, k])[0, 1] if ok.sum() > 2 else 0.0
        self.r = np.clip(np.nan_to_num(r), R_FLOOR, R_CEIL)
        dd = (za - zb) ** 2 / 2
        R = self.r[None, :]
        tau2 = np.where(np.isfinite(dd), (lam * (1 - R) + dd) / (lam + 1), 1 - R)
        mu = z1 * R / (R + tau2)
        var = kappa * (R * tau2 / (R + tau2) + tau2)
        # z of each threshold on the H1 scale: the normal score a value equal to the threshold would get
        zth = np.empty(len(self.cs))
        for k in range(len(self.families)):
            sel = self.cs.fam == k
            zth[sel] = normal_scores(self.H1[:, [k]], self.cs.th[sel][:, None].astype(float))[:, 0]
        self.zth = zth
        f = self.cs.fam
        q = norm.cdf(self.cs.sign * (mu[:, f] - zth) / np.sqrt(var[:, f]))
        # unmeasurable on H1: the statement is false there; its chance next window is the population's
        base = self.T1.mean(0)
        self.Qall = np.where(np.isfinite(z1[:, f]), q, base[None, :])
        self.Q = np.where(self.T1, self.Qall, 0.0)


def credit_given(t_dist: np.ndarray, d_dist: np.ndarray, D: int) -> np.ndarray:
    """Expected tie-shared credit. t_dist: (..., K) target count distribution; d_dist: (..., K) decoy
    count distribution (same leading shape); D decoys."""
    G = np.cumsum(d_dist, -1) - d_dist          # P(decoy < t)
    E = d_dist
    with np.errstate(divide="ignore", invalid="ignore"):
        num = (G + E) ** (D + 1) - G ** (D + 1)
        cr = np.where(E > 1e-12, num / ((D + 1) * np.maximum(E, 1e-300)), G ** D)
    return (t_dist * cr).sum(-1)


def _add(dist: np.ndarray, p: np.ndarray) -> np.ndarray:
    """dist (..., K) count distribution, p (..., C) Bernoulli -> (..., C, K+1)."""
    d = dist[..., None, :]
    pp = p[..., None]
    out = np.zeros(np.broadcast_shapes(d.shape[:-1], pp.shape[:-1]) + (d.shape[-1] + 1,))
    out[..., :-1] += d * (1 - pp)
    out[..., 1:] += d * pp
    return out


def joint_select(pop: Population, i: int, ref: np.ndarray, k: int, D: int, soft: bool = True,
                 line: float = 1.0, allowed: np.ndarray | None = None, per_group: int = PER_GROUP,
                 pool: int = POOL) -> list[int]:
    """Greedy joint selection of up to k statement indices for person i, against reference people
    `ref` (which must not contain i), for a matcher facing D decoys. `line`: only statements true of
    at most this share of the reference are eligible. `allowed`: optional mask over statements."""
    true_i = pop.T1[i].copy()
    if allowed is not None:
        true_i &= allowed
    share = pop.T1[ref].mean(0)
    elig = np.nonzero(true_i & (share <= line))[0]
    if not len(elig):
        return []
    qi = pop.Q[i, elig]
    A = pop.Qall[ref][:, elig] if soft else pop.T1[ref][:, elig].astype(float)
    if len(elig) > pool:
        # keep the `pool` statements that do best alone (the first greedy step over all of them)
        one = credit_given(_add(np.ones(1), qi), _add(np.ones((len(ref), 1)), A).mean(0), D)
        keep = np.sort(np.lexsort((-qi, A.mean(0), -np.round(one, 12)))[:pool])
        elig, qi, A = elig[keep], qi[keep], A[:, keep]
    fam = pop.cs.fam[elig]
    grp = pop.groups[elig]
    t_dist = np.ones(1)
    d_dist = np.ones((len(ref), 1))
    chosen: list[int] = []
    used_f: set[int] = set()
    used_g: dict[str, int] = {}
    for _ in range(k):
        ok = np.array([(f not in used_f) and used_g.get(g, 0) < per_group for f, g in zip(fam, grp)])
        if not ok.any():
            break
        idx = np.nonzero(ok)[0]
        tn = _add(t_dist, qi[idx])                      # C x K+1
        dn = _add(d_dist, A[:, idx])                    # R x C x K+1
        mix = dn.mean(0)                                # C x K+1
        val = credit_given(tn, mix, D)
        # prefer, among equal credit, the rarer and then the surer statement
        b = idx[np.lexsort((-qi[idx], A[:, idx].mean(0), -np.round(val, 12)))[0]]
        chosen.append(int(elig[b]))
        used_f.add(int(fam[b]))
        used_g[grp[b]] = used_g.get(grp[b], 0) + 1
        t_dist = _add(t_dist, qi[[b]])[0]
        d_dist = _add(d_dist, A[:, [b]])[:, 0]
    return chosen


def counts_on(T: np.ndarray, chosen: list[int], upto: int) -> np.ndarray:
    """How many of the first `upto` chosen statements are true of each row of T."""
    if upto == 0 or not chosen:
        return np.zeros(T.shape[0], dtype=np.int64)
    return T[:, chosen[:upto]].sum(1)


def rank_credit(score: np.ndarray, target: int, members: np.ndarray, top: int = 1, higher: bool = True) -> float:
    """Chance that `target` is within the first `top` of `members` ordered by score (higher first when
    `higher`), ties broken at random."""
    s = score[members] if higher else -score[members]
    st = s[members == target][0]
    a = int((s > st).sum())
    m = int((s == st).sum())
    return float(np.clip(top - a, 0, m) / m)


def set_bits(pop: Population, i: int, ref: np.ndarray, chosen: list[int], upto: int) -> tuple[float, float, bool]:
    """(bits on the reference's first halves, bits on the reference's second halves when the target's
    second half keeps all of them else 0, whether it kept them)."""
    S = chosen[:upto]
    n = len(ref)
    c1 = int(pop.T1[ref][:, S].all(1).sum()) if S else n
    kept = bool(pop.T2[i, S].all()) if S else True
    c2 = int(pop.T2[ref][:, S].all(1).sum()) if S else n
    return float(kt_bits(c1, n)), float(kt_bits(c2, n)) if kept else 0.0, kept


# ------------------------------------------------------------------------------------- fingerprint

def fingerprint_z(ref1: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Normal scores on the reference first halves, unmeasurable values set to 0 (the median)."""
    return np.nan_to_num(normal_scores(ref1, X), nan=0.0)


def within_cov(Za: np.ndarray, Zb: np.ndarray, shrink: float) -> np.ndarray:
    """Within-person covariance of window-to-window changes, (Za - Zb)^T (Za - Zb) / (2P), with an
    unmeasurable score (NaN) counted as the median, 0, as the fingerprint does. A measure that fewer
    than 10% of people can be scored on in both windows gets variance 1 (the between-person scale)
    and no covariance. Shrunk towards its diagonal by `shrink`; eigenvalues are floored at RIDGE
    times the mean variance so it stays invertible."""
    few = (np.isfinite(Za) & np.isfinite(Zb)).mean(0) < 0.10
    d = np.nan_to_num(Za, nan=0.0) - np.nan_to_num(Zb, nan=0.0)
    W = d.T @ d / len(d) / 2
    W[few, :] = 0.0
    W[:, few] = 0.0
    dg = np.diag(W).copy()
    dg[few] = 1.0
    np.fill_diagonal(W, dg)
    W = (1 - shrink) * W + shrink * np.diag(np.diag(W))
    w, V = np.linalg.eigh((W + W.T) / 2)
    w = np.maximum(w, RIDGE * float(np.mean(dg)))
    return (V * w) @ V.T


def mahalanobis_dist(Z1: np.ndarray, Z2: np.ndarray, Winv: np.ndarray) -> np.ndarray:
    """P1 x P2 matrix of (z1 - z2)^T Winv (z1 - z2)."""
    a = np.einsum("ij,jk,ik->i", Z1, Winv, Z1)
    b = np.einsum("ij,jk,ik->i", Z2, Winv, Z2)
    return a[:, None] + b[None, :] - 2 * Z1 @ Winv @ Z2.T


def gaussian_identity_bits(Z: np.ndarray, W: np.ndarray) -> float:
    """1/2 log2 det(Sigma_total) / det(W): bits a noisy reading of the whole profile carries about who
    it is under a Gaussian model (between-person spread over within-person noise)."""
    S = np.cov(Z.T)
    s1 = np.linalg.slogdet(S)[1]
    s2 = np.linalg.slogdet(W)[1]
    return float((s1 - s2) / 2 / math.log(2))
