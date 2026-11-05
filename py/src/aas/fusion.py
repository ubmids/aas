"""Fusion: one Big Five estimate per trait from several sources, with an honest interval.

Each reader (text-bigfive, text-mbti, later music or sensors) speaks about a trait on its own.
A person has some sources and not others. This module turns whatever a person has into one
z-score per trait, with an interval that says how often it should contain their quiz score.

The common currency is a Reading: one source's standardised score u for one Big Five trait,
plus that source's stated validity v (its correlation with the quiz on unseen people, from the
source's own dataset). The measurement model behind every method is

    quiz z      ~ N(0, 1)                          (prior: z is relative to population norms)
    u_j         = v_j * z + sqrt(1 - v_j^2) * e_j  (each source a noisy measurement)
    corr(e_j, e_k) = R_jk                          (errors can move together: same text, same phone)

Methods (all per trait; array API in the classes, person API in Fuser):

    single       best single source available for that person (baseline)
    precision    precision-weighted (GLS) combination with plug-in validities and error covariance
                 estimated on training people, then the N(0,1) prior; CV+ conformal interval
    stack        ridge on source scores + missingness indicators, ridge penalty by inner LOO
                 inside every outer fold (nested); CV+ conformal interval
    bayes        hierarchical Gaussian model from stated validities only (ZERO joint training
                 data): validity uncertainty in Fisher z is integrated over; posterior interval
    bayes_eb     the same model, with validities, error correlations and score scales updated
                 from training people (empirical Bayes, shrunk toward the stated values)
    bayes_eb_cp  bayes_eb as the base model of a CV+ conformal interval

MBTI letters are a soft measurement of the Big Five: P(first letter) -> latent preference score ->
Big Five trait through the published MBTI x NEO-PI correlations (MBTI_LINKS). See docs/fusion.md.

Nothing here touches the network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import asin, ceil, floor, pi, sin, sqrt

import numpy as np

from .person import Estimate, Person
from .schema import TRAITS

LEVEL = 0.80

# MBTI axis -> (Big Five trait, sign when the FIRST letter is true, |r| between the continuous
# MBTI scale and the NEO-PI trait). McCrae & Costa (1989), J Pers 57(1):17-40, 267 men / 201 women,
# self-reported NEO-PI vs MBTI continuous scores, men: E-I x E -.74, S-N x O +.72, T-F x A +.44,
# J-P x C -.49 (women reported as -.69, .69, .46, -.46). Furnham (1996, PAID 21:303) and Furnham,
# Moutafi & Crump (2003, SBP 31:577, n = 900) replicate the same four pairings. PANDORA Reddit
# authors (Gjurkovic et al. 2021, Table 3, self-reported letters vs Big Five percentiles):
# I x E -.748, N x O +.434, T x A -.554, P x C -.575 (also T x N -.341).
# We use the McCrae & Costa men's values as the link, a published lab figure; the Reddit figures
# say the N/S -> O link may be weaker online (.43) and T/F -> A stronger (.55).
MBTI_LINKS: dict[str, tuple[str, int, float]] = {
    "I/E": ("E", -1, 0.74),
    "N/S": ("O", +1, 0.72),
    "T/F": ("A", -1, 0.44),
    "J/P": ("C", +1, 0.49),
}

VALIDITY_SD = 0.10        # Fisher-z sd of a reader's validity on new people (domain shift), Big Five readers
MBTI_VALIDITY_SD = 0.15   # wider: a chain of two correlations, one from a different population
RHO_SAME = 0.30           # prior error correlation between two sources of the same modality (same text)
RHO_CROSS = 0.0           # prior error correlation across modalities
DISCOUNT = 0.75           # stated validities are in-domain; new people's data is out of domain (fusion.md)
MIN_VALIDITY = 0.02


# ---------------------------------------------------------------- small maths

def auc_to_r(auc: float) -> float:
    """Correlation r of a score with a continuous trait, given its AUC against the median split.

    For (score, trait) bivariate normal with correlation r, the AUC for trait > 0 vs < 0 is
    1/2 + (2/pi) asin(r / sqrt 2) (a 3-d orthant probability). Inverted here."""
    a = min(max(float(auc), 0.0), 1.0)
    return sqrt(2) * sin(pi * (a - 0.5) / 2)


def r_to_auc(r: float) -> float:
    return 0.5 + (2 / pi) * asin(max(-1.0, min(1.0, r)) / sqrt(2))


def _probit(p: float) -> float:
    from scipy.stats import norm

    return float(norm.ppf(min(max(p, 1e-6), 1 - 1e-6)))


def _zq(level: float) -> float:
    from scipy.stats import norm

    return float(norm.ppf(0.5 + level / 2))


def _phi(z: float) -> float:
    from scipy.stats import norm

    return float(norm.cdf(z))


def _psd_corr(R: np.ndarray, floor_eig: float = 1e-3) -> np.ndarray:
    """Nearest-ish correlation matrix: clip eigenvalues, rescale to unit diagonal."""
    R = (R + R.T) / 2
    w, Q = np.linalg.eigh(R)
    R = (Q * np.maximum(w, floor_eig)) @ Q.T
    d = np.sqrt(np.diag(R))
    return R / np.outer(d, d)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------- readings

@dataclass
class Reading:
    """One source's standardised score on one Big Five trait, for one person."""
    channel: str            # e.g. "text-bigfive", "text-mbti:I/E"
    trait: str              # Big Five letter
    u: float                # standardised score: mean 0, sd 1 in the population, corr v with the quiz
    validity: float         # stated r with the quiz on unseen people (sign already applied: >= 0 expected)
    validity_sd: float = VALIDITY_SD   # uncertainty of that validity on new people, Fisher z
    group: str = ""         # modality; same-group sources get correlated errors a priori
    note: str = ""


def readings_from_estimates(estimates: list[Estimate], mbti_links=MBTI_LINKS,
                            measured_links: dict | None = None) -> tuple[list[Reading], list[str]]:
    """Turn reader Estimates into Readings. Returns (readings, reasons for anything skipped).

    bigfive  value m is the reader's expected z, a regression-shrunk prediction whose spread is
             about its validity r, so u = m / r (text-bigfive builds m exactly that way).
    mbti     value P = P(first letter); with rho = r implied by the letter AUC, the latent
             preference score is sqrt(1 - rho^2) * probit(P) (spread rho), so u_L = that / rho.
             For the linked Big Five trait: u = sign * u_L, validity = rho * |link r|.
    measured ignored unless measured_links maps (reader, trait) -> (bigfive letter, sign, r): no
             measured trait has a tested link to the Big Five yet.
    """
    out, skipped = [], []
    measured_links = measured_links or {}
    for e in estimates:
        if not e.usable or e.value is None or not np.isfinite(e.value):
            continue
        group = (e.reader or "").split("-")[0].split(":")[0]
        if e.validity is None:
            skipped.append(f"{e.reader} {e.trait}: no tested validity, cannot be weighted")
            continue
        r = e.validity if e.validity_kind == "r" else auc_to_r(e.validity)
        if e.kind == "bigfive" and e.trait in TRAITS:
            if r < MIN_VALIDITY:
                skipped.append(f"{e.reader} {e.trait}: tested validity {r:.2f} is no better than guessing")
                continue
            out.append(Reading(e.reader, e.trait, float(e.value) / r, float(r), VALIDITY_SD, group,
                               f"z {e.value:+.2f}, r {r:.2f}"))
        elif e.kind == "mbti" and e.trait in mbti_links:
            t, sign, c = mbti_links[e.trait]
            if r < MIN_VALIDITY:
                skipped.append(f"{e.reader} {e.trait}: letter accuracy no better than a coin flip")
                continue
            r = min(r, 0.95)
            u_l = sqrt(1 - r * r) * _probit(float(e.value)) / r
            out.append(Reading(f"{e.reader}:{e.trait}", t, sign * u_l, r * c, MBTI_VALIDITY_SD, group,
                               f"P({e.trait[0]}) {e.value:.0%}, letter r {r:.2f} x link {c:.2f}"))
        elif e.kind == "measured" and (e.reader, e.trait) in measured_links:
            t, sign, rr = measured_links[(e.reader, e.trait)]
            mu, sd = 0.0, 1.0
            if e.percentile is not None:   # percentile -> normal score
                u = _probit(min(max(e.percentile / 100, 0.005), 0.995))
            else:
                u = (float(e.value) - mu) / sd
            out.append(Reading(f"{e.reader}:{e.trait}", t, sign * u, rr, VALIDITY_SD, group))
    return out, skipped


# ---------------------------------------------------------------- the Gaussian core

def gaussian_posterior(u: np.ndarray, v: np.ndarray, R: np.ndarray, vsd: np.ndarray | None = None,
                       draws: int = 0, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Posterior of z for people sharing one set of sources.

    u (m, k) standardised readings, v (k,) validities, R (k, k) error correlation.
    Without validity uncertainty: mean = u S^-1 v, var = 1 - v' S^-1 v, S = v v' + D R D,
    D = diag(sqrt(1 - v^2)). With vsd and draws > 0, v is drawn in Fisher z and the result is the
    mixture over draws (mean of means; mean of variances plus variance of means).
    Returns (mean (m,), var (m,), weights (k,) = average S^-1 v)."""
    u = np.atleast_2d(u)
    if draws and vsd is not None and np.any(vsd > 0):
        rng = np.random.default_rng(seed)
        z = np.arctanh(np.clip(v, -0.95, 0.95))[None, :] + vsd[None, :] * rng.standard_normal((draws, len(v)))
        V = np.clip(np.tanh(z), -0.95, 0.95)
    else:
        V = np.clip(v, -0.95, 0.95)[None, :]
    D = np.sqrt(1 - V ** 2)
    S = V[:, :, None] * V[:, None, :] + D[:, :, None] * R[None] * D[:, None, :]
    W = np.linalg.solve(S, V[:, :, None])[:, :, 0]          # (K, k)
    var_k = 1 - np.einsum("kj,kj->k", V, W)                  # (K,)
    means = u @ W.T                                          # (m, K)
    mean = means.mean(1)
    var = var_k.mean() + means.var(1)
    return mean, var, W.mean(0)


def prior_error_corr(groups: list[str], rho_same=RHO_SAME, rho_cross=RHO_CROSS) -> np.ndarray:
    k = len(groups)
    R = np.full((k, k), rho_cross, float)
    for i in range(k):
        for j in range(k):
            if i != j and groups[i] and groups[i] == groups[j]:
                R[i, j] = rho_same
    np.fill_diagonal(R, 1.0)
    return R


# ---------------------------------------------------------------- array-level methods
# Every method: fit(U, y, V, Vsd, groups) and predict(U, V, Vsd) -> (mu, sd), NaN where a person has
# no source. U, V, Vsd are (n, J) with NaN where the source is missing; groups is a list of J strings.

def _patterns(U: np.ndarray):
    """Group rows by which sources are present. Yields (mask (J,), row indices)."""
    present = ~np.isnan(U)
    keys, inv = np.unique(present, axis=0, return_inverse=True)
    for i, m in enumerate(keys):
        if m.any():
            yield m, np.nonzero(inv.ravel() == i)[0]


class Bayes:
    """Hierarchical Gaussian fusion. With eb=False it ignores training data (works at n = 0)."""

    def __init__(self, eb: bool = False, discount: float = DISCOUNT, rho_same=RHO_SAME, rho_cross=RHO_CROSS,
                 draws: int = 200, n0_corr: float = 20, n0_scale: float = 20, seed: int = 0):
        self.eb, self.discount, self.rho_same, self.rho_cross = eb, discount, rho_same, rho_cross
        self.draws, self.n0_corr, self.n0_scale, self.seed = draws, n0_corr, n0_scale, seed

    def fit(self, U, y, V, Vsd, groups):
        J = len(groups)
        self.groups = list(groups)
        self.R = prior_error_corr(self.groups, self.rho_same, self.rho_cross)
        self.loc, self.scale = np.zeros(J), np.ones(J)
        self.dz = np.zeros(J)                    # Fisher-z shift of each validity from its stated value
        self.sd_post = None
        if not self.eb or U is None or len(y) == 0:
            return self
        y = np.asarray(y, float)
        sd_post = np.full(J, np.nan)
        vpost = np.full(J, np.nan)
        for j in range(J):
            ok = ~np.isnan(U[:, j])
            n = int(ok.sum())
            v0 = np.nanmedian(V[:, j]) * self.discount if n else np.nan
            s0 = np.nanmedian(Vsd[:, j]) if n else np.nan
            if n < 5:
                vpost[j], sd_post[j] = v0, s0
                continue
            u = U[ok, j]
            w = n / (n + self.n0_scale)
            self.loc[j] = w * u.mean()
            self.scale[j] = float(np.exp(w * np.log(max(u.std(), 1e-6))))
            r = _corr((u - self.loc[j]) / self.scale[j], y[ok])
            z0, prec0 = np.arctanh(np.clip(v0, -0.95, 0.95)), 1 / max(s0, 1e-3) ** 2
            prec1 = max(n - 3, 0)
            zp = (prec0 * z0 + prec1 * np.arctanh(np.clip(r, -0.95, 0.95))) / (prec0 + prec1)
            vpost[j], sd_post[j] = np.tanh(zp), 1 / sqrt(prec0 + prec1)
            self.dz[j] = zp - z0
        self.sd_post = sd_post
        # error correlations, pairwise on people with both sources, shrunk toward the prior
        Us = (U - self.loc) / self.scale
        E = (Us - vpost * y[:, None]) / np.sqrt(1 - np.clip(vpost, -0.95, 0.95) ** 2)
        R = self.R.copy()
        for a in range(J):
            for b in range(a + 1, J):
                ok = ~np.isnan(E[:, a]) & ~np.isnan(E[:, b])
                n = int(ok.sum())
                if n < 5:
                    continue
                rh = _corr(E[ok, a], E[ok, b])
                if not np.isfinite(rh):
                    continue
                z = (self.n0_corr * np.arctanh(R[a, b]) + (n - 3) * np.arctanh(np.clip(rh, -0.95, 0.95))) / (self.n0_corr + n - 3)
                R[a, b] = R[b, a] = np.tanh(z)
        self.R = _psd_corr(R)
        return self

    def params(self, V, Vsd):
        """Per-person validities after the discount and any EB shift, and their sds."""
        z = np.arctanh(np.clip(V * self.discount, -0.95, 0.95)) + self.dz
        sd = Vsd if self.sd_post is None else np.where(np.isnan(self.sd_post), Vsd, self.sd_post)
        return np.tanh(z), sd

    def predict(self, U, V, Vsd):
        n = U.shape[0]
        mu, var = np.full(n, np.nan), np.full(n, np.nan)
        Us = (U - self.loc) / self.scale
        Vp, Sd = self.params(V, Vsd)
        for m, rows in _patterns(U):
            # people in one pattern can have different stated validities (text length): group again
            key = np.round(np.c_[Vp[rows][:, m], Sd[rows][:, m]], 4)
            ukeys, inv = np.unique(key, axis=0, return_inverse=True)
            for i, kk in enumerate(ukeys):
                rr = rows[inv.ravel() == i]
                k = m.sum()
                mean, v, _ = gaussian_posterior(Us[rr][:, m], kk[:k], self.R[np.ix_(m, m)], kk[k:],
                                                self.draws, self.seed)
                mu[rr], var[rr] = mean, v
        return mu, np.sqrt(var)


class Precision:
    """Precision-weighted (GLS) fusion with plug-in estimates from training people.

    Validities = sample correlations, error covariance = sample correlation of residuals (pairwise,
    lightly shrunk to 0, made positive definite). Each source debiased (u / v = z + error / v), combined
    with weights proportional to the inverse error covariance, then the N(0, 1) prior (that last step is
    what keeps a single weak source from doing worse than guessing average). raw=True skips the prior."""

    def __init__(self, raw: bool = False, n0_corr: float = 5):
        self.raw, self.n0_corr = raw, n0_corr

    def fit(self, U, y, V, Vsd, groups):
        J = U.shape[1]
        self.loc, self.scale, self.a = np.zeros(J), np.ones(J), np.full(J, MIN_VALIDITY)
        for j in range(J):
            ok = ~np.isnan(U[:, j])
            if ok.sum() >= 5:
                u = U[ok, j]
                self.loc[j], self.scale[j] = u.mean(), max(u.std(), 1e-6)
                r = _corr((u - self.loc[j]) / self.scale[j], y[ok])
                self.a[j] = np.clip(r if np.isfinite(r) else 0, MIN_VALIDITY, 0.95)
        Us = (U - self.loc) / self.scale
        E = (Us - self.a * y[:, None]) / np.sqrt(1 - self.a ** 2)
        R = np.eye(J)
        for i in range(J):
            for k in range(i + 1, J):
                ok = ~np.isnan(E[:, i]) & ~np.isnan(E[:, k])
                n = int(ok.sum())
                if n >= 5:
                    rh = _corr(E[ok, i], E[ok, k])
                    if np.isfinite(rh):
                        R[i, k] = R[k, i] = np.tanh((n - 3) * np.arctanh(np.clip(rh, -0.95, 0.95)) / (n - 3 + self.n0_corr))
        self.R = _psd_corr(R)
        return self

    def predict(self, U, V=None, Vsd=None):
        n = U.shape[0]
        mu, sd = np.full(n, np.nan), np.full(n, np.nan)
        M = (U - self.loc) / self.scale / self.a              # debiased: z + error / a
        s = np.sqrt(1 - self.a ** 2) / self.a                 # sd of that error
        for m, rows in _patterns(U):
            C = self.R[np.ix_(m, m)] * np.outer(s[m], s[m])
            Ci1 = np.linalg.solve(C, np.ones(m.sum()))
            var_m = 1 / Ci1.sum()
            est = M[rows][:, m] @ (Ci1 * var_m)
            if self.raw:
                mu[rows], sd[rows] = est, sqrt(var_m)
            else:
                shrink = 1 / (1 + var_m)
                mu[rows], sd[rows] = est * shrink, sqrt(var_m * shrink)
        return mu, sd


class Single:
    """Best single source each person has: highest stated validity (n = 0) or highest training r."""

    def __init__(self, discount: float = DISCOUNT):
        self.discount = discount
        self.fitted = False

    def fit(self, U, y, V, Vsd, groups):
        self.fitted = U is not None and len(y) > 0
        if not self.fitted:
            return self
        J = U.shape[1]
        self.r, self.b, self.c = np.full(J, -1.0), np.zeros(J), np.zeros(J)
        for j in range(J):
            ok = ~np.isnan(U[:, j])
            if ok.sum() >= 5:
                r = _corr(U[ok, j], y[ok])
                if np.isfinite(r):
                    self.r[j] = r
                    self.b[j], self.c[j] = np.polyfit(U[ok, j], y[ok], 1)
        return self

    def predict(self, U, V, Vsd):
        n = U.shape[0]
        mu, sd = np.full(n, np.nan), np.full(n, np.nan)
        for i in range(n):
            have = np.nonzero(~np.isnan(U[i]))[0]
            if not len(have):
                continue
            if self.fitted:
                j = have[np.argmax(self.r[have])]
                mu[i] = self.b[j] * U[i, j] + self.c[j]
                sd[i] = sqrt(max(1 - max(self.r[j], 0) ** 2, 1e-6))
            else:
                v = V[i, have] * self.discount
                j = have[np.argmax(v)]
                vv = min(V[i, j] * self.discount, 0.95)
                mu[i], sd[i] = vv * U[i, j], sqrt(1 - vv * vv)
        return mu, sd


class Stack:
    """Ridge on [source scores (0 when missing), missingness indicators]; penalty chosen by efficient
    leave-one-out inside whatever data it is fitted on, so under CV+ the choice is nested in each fold.
    The scale used to normalise conformal scores is the posterior sd of an EB Gaussian model fitted on
    the same data (any function fixed before scoring keeps the conformal guarantee)."""

    ALPHAS = np.logspace(-1, 4, 26)

    def fit(self, U, y, V, Vsd, groups):
        from sklearn.linear_model import RidgeCV

        self.model = RidgeCV(alphas=self.ALPHAS).fit(self._X(U), y)
        self.scaler = Bayes(eb=True, draws=0).fit(U, y, V, Vsd, groups)
        return self

    @staticmethod
    def _X(U):
        miss = np.isnan(U)
        return np.c_[np.where(miss, 0.0, U), miss.astype(float)]

    def predict(self, U, V, Vsd):
        mu = self.model.predict(self._X(U))
        _, sd = self.scaler.predict(U, V, Vsd)
        none = np.isnan(U).all(1)
        mu[none] = np.nan
        return mu, sd


class CVPlus:
    """CV+ conformal interval (Barber, Candes, Ramdas & Tibshirani 2021) around a base method.

    K-fold: every training person gets an out-of-fold prediction and a score R_i = |y_i - mu_i| / sd_i.
    For a new person x: low  = the floor(a(n+1))-th smallest of  mu_k(i)(x) - sd_k(i)(x) R_i,
                        high = the ceil((1-a)(n+1))-th smallest of mu_k(i)(x) + sd_k(i)(x) R_i.
    Guarantee: coverage >= 1 - 2a for any distribution if people are exchangeable; about 1 - a in
    practice. normalize=False uses plain |residual| (equal widths for everyone)."""

    def __init__(self, make, folds: int = 10, level: float = LEVEL, normalize: bool = True, seed: int = 0):
        self.make, self.folds, self.level, self.normalize, self.seed = make, folds, level, normalize, seed

    def fit(self, U, y, V, Vsd, groups):
        keep = ~np.isnan(U).all(1)
        U, y, V, Vsd = U[keep], np.asarray(y, float)[keep], V[keep], Vsd[keep]
        n = len(y)
        rng = np.random.default_rng(self.seed)
        fold = rng.permutation(np.arange(n) % min(self.folds, n))
        self.models, self.fold, self.R = [], fold, np.zeros(n)
        for k in range(fold.max() + 1):
            tr, te = fold != k, fold == k
            m = self.make().fit(U[tr], y[tr], V[tr], Vsd[tr], groups)
            mu, sd = m.predict(U[te], V[te], Vsd[te])
            self.R[te] = np.abs(y[te] - mu) / (sd if self.normalize else 1.0)
            self.models.append(m)
        self.full = self.make().fit(U, y, V, Vsd, groups)
        self.n = n
        return self

    def predict(self, U, V, Vsd):
        mu, _ = self.full.predict(U, V, Vsd)
        a = 1 - self.level
        kl, ku = floor(a * (self.n + 1)), ceil((1 - a) * (self.n + 1))
        per = [m.predict(U, V, Vsd) for m in self.models]
        Mk = np.stack([p[0] for p in per], 1)[:, self.fold]          # (m, n): mu_{k(i)}(x)
        Sk = np.stack([p[1] for p in per], 1)[:, self.fold] if self.normalize else 1.0
        lo_c = Mk - Sk * self.R
        hi_c = Mk + Sk * self.R
        lo = np.partition(lo_c, kl - 1, axis=1)[:, kl - 1] if kl >= 1 else np.full(len(mu), -np.inf)
        hi = np.partition(hi_c, ku - 1, axis=1)[:, ku - 1] if ku <= self.n else np.full(len(mu), np.inf)
        return mu, lo, hi


METHODS = ("single", "precision", "stack", "bayes", "bayes_eb", "bayes_eb_cp")


def make_method(name: str, n_train: int, level: float = LEVEL, discount: float = DISCOUNT, seed: int = 0):
    """An object with fit(U, y, V, Vsd, groups) and predict(U, V, Vsd) -> (mu, low, high)."""
    if name == "bayes" or (name in ("bayes_eb", "bayes_eb_cp", "single") and n_train == 0):
        if name == "single":
            return _Gauss(Single(discount=discount), level)
        return _Gauss(Bayes(eb=False, discount=discount, seed=seed), level)
    if n_train == 0:
        raise ValueError(f"{name} needs training people with every source and a quiz")
    if name == "single":
        return CVPlus(lambda: Single(discount=discount), level=level, seed=seed)
    if name == "precision":
        return CVPlus(lambda: Precision(), level=level, seed=seed)
    if name == "stack":
        return CVPlus(lambda: Stack(), level=level, seed=seed)
    if name == "bayes_eb":
        return _Gauss(Bayes(eb=True, discount=discount, seed=seed), level)
    if name == "bayes_eb_cp":
        return CVPlus(lambda: Bayes(eb=True, discount=discount, draws=0, seed=seed), level=level, seed=seed)
    raise ValueError(name)


class _Gauss:
    """Model-based (posterior) interval: mu +- z * sd."""

    def __init__(self, base, level: float = LEVEL):
        self.base, self.level = base, level

    def fit(self, U, y, V, Vsd, groups):
        self.base.fit(U, y, V, Vsd, groups)
        return self

    def predict(self, U, V, Vsd):
        mu, sd = self.base.predict(U, V, Vsd)
        z = _zq(self.level)
        return mu, mu - z * sd, mu + z * sd


# ---------------------------------------------------------------- person-level API

@dataclass
class Fused:
    trait: str
    value: float
    low: float
    high: float
    implied_r: float | None          # Bayes only: sqrt(1 - posterior variance), model-implied accuracy
    used: list[Reading] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)
    ignored: list[str] = field(default_factory=list)


class Fuser:
    """Fuse a person's Readings into one estimate per Big Five trait.

    fuser = Fuser("bayes")                         # no training needed (pre-study)
    fuser = Fuser("bayes_eb").fit(people, quizzes) # people: list[list[Reading]], quizzes: list[{trait: z}]
    fused = fuser.predict(readings)                # {trait: Fused}; traits without any reading are absent
    """

    def __init__(self, method: str = "bayes", level: float = LEVEL, discount: float = DISCOUNT, seed: int = 0):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        self.method, self.level, self.discount, self.seed = method, level, discount, seed
        self.fitted: dict[str, tuple] = {}     # trait -> (channels, groups, model)
        self.n_train = 0

    @staticmethod
    def _arrays(people: list[list[Reading]], trait: str, channels: list[str]):
        n, J = len(people), len(channels)
        U, V, Vsd = (np.full((n, J), np.nan) for _ in range(3))
        idx = {c: j for j, c in enumerate(channels)}
        for i, rs in enumerate(people):
            for r in rs:
                if r.trait == trait and r.channel in idx:
                    j = idx[r.channel]
                    U[i, j], V[i, j], Vsd[i, j] = r.u, r.validity, r.validity_sd
        return U, V, Vsd

    def fit(self, people: list[list[Reading]], quizzes: list[dict[str, float]]):
        self.n_train = len(people)
        for t in TRAITS:
            rows = [i for i, q in enumerate(quizzes) if t in q and q[t] is not None]
            ppl = [people[i] for i in rows]
            y = np.array([quizzes[i][t] for i in rows], float)
            groups_of = {}
            for rs in ppl:
                for r in rs:
                    if r.trait == t:
                        groups_of.setdefault(r.channel, r.group)
            channels = sorted(groups_of)
            if not channels or len(y) < 10:
                continue
            groups = [groups_of[c] for c in channels]
            U, V, Vsd = self._arrays(ppl, t, channels)
            model = make_method(self.method, len(y), self.level, self.discount, self.seed).fit(U, y, V, Vsd, groups)
            self.fitted[t] = (channels, groups, model)
        return self

    def predict(self, readings: list[Reading]) -> dict[str, Fused]:
        out = {}
        for t in TRAITS:
            rs = [r for r in readings if r.trait == t]
            if not rs:
                continue
            ignored = []
            if t in self.fitted:
                channels, groups, model = self.fitted[t]
                known = [r for r in rs if r.channel in channels]
                ignored = [f"{r.channel}: not in the training data for this fusion model" for r in rs if r.channel not in channels]
                if not known:
                    continue
                rs = known
            else:   # zero-data model (pre-study, or a trait the training set never covered)
                channels = [r.channel for r in rs]
                if len(set(channels)) != len(channels):   # same source twice: keep the first
                    seen, uniq = set(), []
                    for r in rs:
                        if r.channel not in seen:
                            seen.add(r.channel)
                            uniq.append(r)
                    rs, channels = uniq, [r.channel for r in uniq]
                groups = [r.group for r in rs]
                model = make_method("bayes", 0, self.level, self.discount, self.seed).fit(None, [], None, None, groups)
            U, V, Vsd = self._arrays([rs], t, channels)
            mu, lo, hi = model.predict(U, V, Vsd)
            implied, weights = None, {}
            base = getattr(model, "base", None)
            if isinstance(base, Bayes):
                m = ~np.isnan(U[0])
                Vp, Sd = base.params(V, Vsd)
                _, _, w = gaussian_posterior((U[0, m] - base.loc[m]) / base.scale[m], Vp[0, m],
                                             base.R[np.ix_(m, m)], Sd[0, m], base.draws, base.seed)
                # Accuracy for this SET of sources, sqrt(v' S^-1 v) at the (discounted) validities with no draws
                # (MATH.md 5.3). The person's mixture variance is the wrong quantity here: its spread of means
                # over validity draws grows with this person's |u| (a strong reading printed "r about 0.00"),
                # and averaging v^2 over draws without that spread overstates r (Jensen).
                _, var, _ = gaussian_posterior(np.zeros((1, int(m.sum()))), Vp[0, m], base.R[np.ix_(m, m)])
                implied = float(sqrt(max(1 - var[0], 0.0)))
                weights = {c: float(x) for c, x in zip(np.array(channels)[m], w)}
            out[t] = Fused(t, float(mu[0]), float(lo[0]), float(hi[0]), implied, rs, weights, ignored)
        return out


class FusedReader:
    """A Reader that runs other readers and fuses their Big Five-relevant outputs into one Estimate
    per trait (reader name "fused"). Usable with aas.profile.build alongside the readers it wraps:
    the fused estimates sit next to the single-source ones and evaluate() scores them the same way.

        build(person, [text_bf, text_mbti, FusedReader([text_bf, text_mbti])])
    """

    name = "fused"

    def __init__(self, readers: list, fuser: Fuser | None = None, measured_links: dict | None = None):
        self.readers = list(readers)
        self.fuser = fuser or Fuser("bayes")
        self.measured_links = measured_links

    def can_read(self, person: Person) -> tuple[bool, str]:
        ok_names, why = [], []
        for r in self.readers:
            try:
                ok, w = r.can_read(person)
            except Exception as ex:   # noqa: BLE001
                ok, w = False, f"{type(ex).__name__}"
            (ok_names if ok else why).append(getattr(r, "name", type(r).__name__) if ok else f"{getattr(r, 'name', '?')}: {w}")
        if not ok_names:
            return False, "no source could be read, so there is nothing to combine (" + "; ".join(why) + ")"
        return True, "combining " + ", ".join(ok_names)

    def read(self, person: Person) -> list[Estimate]:
        ests = []
        for r in self.readers:
            try:
                ok, _ = r.can_read(person)
                if ok:
                    ests += r.read(person)
            except Exception:   # noqa: BLE001  a broken source just isn't combined
                continue
        readings, skipped = readings_from_estimates(ests, measured_links=self.measured_links)
        fused = self.fuser.predict(readings)
        out = []
        trained = self.fuser.n_train
        for t in TRAITS:
            if t not in fused:
                out.append(Estimate(self.name, t, "bigfive", float("nan"), usable=False, interval_level=None,
                                    evidence="no source said anything about this trait"))
                continue
            f = fused[t]
            parts = [f"{r.channel} ({r.note})" if r.note else r.channel for r in f.used]
            if f.weights:
                parts = [p + f", weight {f.weights.get(r.channel, 0):.2f}" for p, r in zip(parts, f.used)]
            how = (f"fusion model '{self.fuser.method}' fitted on {trained} people with several sources"
                   if trained else "zero-data fusion: each source weighted by its own tested accuracy, "
                   "discounted for new data; not yet checked on people who have several sources")
            ev = f"combined from {'; '.join(parts)}. {how}."
            if f.implied_r is not None:
                ev += f" Model-implied accuracy for this set of sources: r about {f.implied_r:.2f} (implied, not tested)."
            left = f.ignored + [s for s in skipped if f" {t}:" in s]
            if left:
                ev += " Left out: " + "; ".join(left) + "."
            # validity stays None: Estimate.validity means TESTED accuracy, and no fused model has been
            # tested on people with several sources yet (docs/fusion.md).
            out.append(Estimate(self.name, t, "bigfive", f.value, f.low, f.high, round(100 * _phi(f.value), 1),
                                None, None, ev, True, self.fuser.level))
        return out
