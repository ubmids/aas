"""Sample size for the AAS whole-person study (docs/study/POWER.md).

    cd py && uv run python scripts/study_power.py            # full run, ~2-4 min
    cd py && uv run python scripts/study_power.py --quick    # fewer sims, ~30 s

Four questions, each answered for n = 60 / 250 / 600 completers:

(a) precision of one source's validity (Pearson r vs the quiz), and n for +-0.10
(b) power to detect an increment Delta r = 0.05 / 0.10 when a second source is
    added to the best single source. The second source is NOT a pre-trained
    reader: it is a 10-feature model fit on the study's own people, inside
    nested cross-validation, then stacked with the text reader's score. That is
    the honest situation for Google and Spotify, which have no Big Five reader yet.
(c) precision of interval coverage (does the 80% interval contain 80%?)
(d) side questions: what a 10-feature in-study model is expected to score under
    CV at each n (shrinkage), retest reliability precision, and MBTI-letter AUC
    precision against an open type questionnaire.

(e) added 2026-11-01, run LAST so (a)-(d) reproduce exactly: the same questions with the text
    reader's MEASURED validity on writing it never saw (models/*/card.json "validity_by_source":
    tweets / Reddit-style prose, openness refused), the zero-data fused r for a Reddit / X user,
    and the smallest validity a better (PANDORA-trained) text model must reach for each study
    question to be answerable at n = 250 and 600.

Everything is simulated from stated assumptions. Nothing is fitted to real people.
Writes docs/study/power_results.json.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import optimize, stats

OUT = Path(__file__).resolve().parents[1] / "docs" / "study" / "power_results.json"
NS = [60, 250, 600]


# ---------------------------------------------------------------- (a) precision of r

def r_halfwidth(r: float, n: int, level: float = 0.95) -> float:
    """Half-width of the Fisher-z CI for r, back on the r scale (average of the two sides)."""
    zc = stats.norm.ppf(0.5 + level / 2)
    z = np.arctanh(r)
    se = 1 / math.sqrt(n - 3)
    lo, hi = np.tanh(z - zc * se), np.tanh(z + zc * se)
    return float((hi - lo) / 2)


def n_for_r_halfwidth(r: float, target: float = 0.10) -> int:
    n = 10
    while r_halfwidth(r, n) > target:
        n += 1
    return n


def sim_r_halfwidth(r: float, n: int, sims: int, rng) -> float:
    """Check by simulation: the 95% spread of observed r around the true r, halved."""
    obs = np.empty(sims)
    for i in range(sims):
        q = rng.standard_normal(n)
        s = r * q + math.sqrt(1 - r * r) * rng.standard_normal(n)
        obs[i] = np.corrcoef(q, s)[0, 1]
    lo, hi = np.quantile(obs, [0.025, 0.975])
    return float((hi - lo) / 2)


# ---------------------------------------------------------------- (b) increment

P2 = 10          # features in the second source's in-study model
K_REL = 0.90     # how well the 10 features together capture source 2's latent signal


def pop_R(r1: float, r2: float, rho: float) -> float:
    """Population multiple R of the quiz on (S1, S2).

    S1 = r1*Q + sqrt(1-r1^2)*u1, S2 = r2*Q + sqrt(1-r2^2)*u2, corr(u1,u2) = rho.
    rho > 0 means the two sources share error (e.g. both pick up "how online you are").
    """
    c12 = r1 * r2 + rho * math.sqrt((1 - r1**2) * (1 - r2**2))
    S = np.array([[1, c12], [c12, 1]])
    a = np.array([r1, r2])
    return float(math.sqrt(a @ np.linalg.solve(S, a)))


def r2_for_increment(r1: float, delta: float, rho: float) -> float:
    """Validity the second source must have on its own for the pair to reach r1 + delta."""
    f = lambda r2: pop_R(r1, r2, rho) - (r1 + delta)
    return float(optimize.brentq(f, 1e-6, 0.95))


def _ridge_fit(X, y, alpha=10.0):
    mx, my = X.mean(0), y.mean()
    Xc = X - mx
    w = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]), Xc.T @ (y - my))
    return mx, my, w


def _ridge_pred(m, X):
    mx, my, w = m
    return (X - mx) @ w + my


def _folds(n, k, rng):
    idx = rng.permutation(n)
    return np.array_split(idx, k)


def _ols(cols, y):
    A = np.column_stack([np.ones(len(y))] + cols)
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    return b


def one_increment_trial(n, r1, r2, rho, rng, k=10, inner=5):
    """Simulate one study's people for one trait and run the pre-registered analysis.

    Returns (r_single_oof, r_comb_oof, one-sided p for 'combined beats single')."""
    q = rng.standard_normal(n)
    u1 = rng.standard_normal(n)
    u2 = rho * u1 + math.sqrt(1 - rho * rho) * rng.standard_normal(n)
    s1 = r1 * q + math.sqrt(1 - r1 * r1) * u1                 # pre-trained text reader score
    lat2 = r2 / K_REL * q + math.sqrt(1 - (r2 / K_REL) ** 2) * u2  # source 2's latent signal
    # p features, each a noisy copy of lat2; their mean correlates K_REL with lat2
    lam = math.sqrt(K_REL**2 / (P2 - K_REL**2 * (P2 - 1)))
    X2 = lam * lat2[:, None] + math.sqrt(1 - lam * lam) * rng.standard_normal((n, P2))

    single = np.empty(n)
    comb = np.empty(n)
    for te in _folds(n, k, rng):
        tr = np.setdiff1d(np.arange(n), te)
        # single source: text score, linearly calibrated on training people only
        b1 = _ols([s1[tr]], q[tr])
        single[te] = b1[0] + b1[1] * s1[te]
        # source-2 model: inner OOF predictions for the training people (for stacking weights)
        oof2 = np.empty(len(tr))
        for ite in _folds(len(tr), inner, rng):
            itr = np.setdiff1d(np.arange(len(tr)), ite)
            m = _ridge_fit(X2[tr][itr], q[tr][itr])
            oof2[ite] = _ridge_pred(m, X2[tr][ite])
        b = _ols([s1[tr], oof2], q[tr])
        m = _ridge_fit(X2[tr], q[tr])
        comb[te] = b[0] + b[1] * s1[te] + b[2] * _ridge_pred(m, X2[te])

    # paired per-person loss difference: positive = combined closer to the quiz
    d = (q - single) ** 2 - (q - comb) ** 2
    t = d.mean() / (d.std(ddof=1) / math.sqrt(n))
    p = 1 - stats.t.cdf(t, n - 1)
    return np.corrcoef(single, q)[0, 1], np.corrcoef(comb, q)[0, 1], p


def increment_power(n, r1, delta, rho, sims, rng):
    r2 = r2_for_increment(r1, delta, rho)
    rs, rc, ps = [], [], []
    for _ in range(sims):
        a, b, p = one_increment_trial(n, r1, r2, rho, rng)
        rs.append(a); rc.append(b); ps.append(p)
    rs, rc, ps = map(np.array, (rs, rc, ps))
    dr = rc - rs
    # 5 traits tested together, Holm: the first test is at .05/5 = .01. Report both.
    # Omnibus: Stouffer over 5 independent traits with the same effect.
    zs = stats.norm.isf(np.clip(ps, 1e-12, 1 - 1e-12))
    zs5 = zs[: (len(zs) // 5) * 5].reshape(-1, 5)
    p_omni = stats.norm.sf(zs5.sum(1) / math.sqrt(5))
    return {
        "n": n, "r1": r1, "delta_target": delta, "rho_shared_error": rho,
        "r2_needed_alone": round(r2, 3),
        "pop_R_combined": round(pop_R(r1, r2, rho), 3),
        "cv_delta_r_mean": round(float(dr.mean()), 3),
        "cv_delta_r_p05_p95": [round(float(x), 3) for x in np.quantile(dr, [0.05, 0.95])],
        "power_one_trait_alpha05": round(float((ps < 0.05).mean()), 3),
        "power_one_trait_alpha01_holm_first": round(float((ps < 0.01).mean()), 3),
        "power_omnibus_5_traits_alpha05": round(float((p_omni < 0.05).mean()), 3),
        "sims": sims,
    }


def incremental_n_analytic(r1, delta, rho, alpha=0.05, power=0.80):
    """Classic nested F-test (adding one predictor), in-sample, as a lower bound on n.

    The CV stacked analysis above needs more people than this because source 2's
    model has to be learned from the same people."""
    r2 = r2_for_increment(r1, delta, rho)
    R2 = pop_R(r1, r2, rho) ** 2
    f2 = (R2 - r1**2) / (1 - R2)
    for n in range(20, 20000):
        df2 = n - 3
        lam = f2 * n
        crit = stats.f.ppf(1 - alpha, 1, df2)
        if 1 - stats.ncf.cdf(crit, 1, df2, lam) >= power:
            return n
    return None


# ---------------------------------------------------------------- (c) coverage

def coverage_halfwidth(n, p=0.80):
    return 1.96 * math.sqrt(p * (1 - p) / n)


def n_for_coverage(target=0.05, p=0.80):
    return math.ceil((1.96 / target) ** 2 * p * (1 - p))


def sim_coverage_pooled(n, icc, sims, rng, p=0.8, traits=5):
    """Coverage pooled over 5 traits when inside/outside is correlated within a person.

    A person-level frailty (latent normal, correlation icc across their 5 traits)
    makes the same people tend to be missed on several traits. Returns the 95%
    half-width of pooled coverage."""
    thr = stats.norm.ppf(p)
    est = np.empty(sims)
    for i in range(sims):
        z = math.sqrt(icc) * rng.standard_normal((n, 1)) + math.sqrt(1 - icc) * rng.standard_normal((n, traits))
        est[i] = (z < thr).mean()
    lo, hi = np.quantile(est, [0.025, 0.975])
    return float((hi - lo) / 2)


# ---------------------------------------------------------------- (d) side questions

def cv_r_of_learned_model(n, R, p, sims, rng, k=10):
    """Expected out-of-fold r of a p-feature ridge model whose population R is R."""
    lam = math.sqrt(K_REL**2 / (p - K_REL**2 * (p - 1)))
    out = np.empty(sims)
    for i in range(sims):
        q = rng.standard_normal(n)
        lat = R / K_REL * q + math.sqrt(1 - (R / K_REL) ** 2) * rng.standard_normal(n)
        X = lam * lat[:, None] + math.sqrt(1 - lam * lam) * rng.standard_normal((n, p))
        pred = np.empty(n)
        for te in _folds(n, k, rng):
            tr = np.setdiff1d(np.arange(n), te)
            pred[te] = _ridge_pred(_ridge_fit(X[tr], q[tr]), X[te])
        out[i] = np.corrcoef(pred, q)[0, 1]
    return {"mean": round(float(out.mean()), 3), "p05": round(float(np.quantile(out, 0.05)), 3),
            "p95": round(float(np.quantile(out, 0.95)), 3)}


def skill_power(r, n, sims, rng):
    """Skill = 1 - MAE/MAE(guess 0) for a calibrated predictor (weights known), and the power of a
    one-sided paired test that it beats guessing the average, one trait, alpha .05."""
    sk, hits = [], 0
    for _ in range(sims):
        q = rng.standard_normal(n)
        s = r * q + math.sqrt(1 - r * r) * rng.standard_normal(n)
        p = r * s
        d = np.abs(q) - np.abs(q - p)
        t = d.mean() / (d.std(ddof=1) / math.sqrt(n))
        hits += (1 - stats.t.cdf(t, n - 1)) < 0.05
        sk.append(1 - np.abs(q - p).mean() / np.abs(q).mean())
    return {"skill_mean": round(float(np.mean(sk)), 3), "power05": round(hits / sims, 3)}


def auc_halfwidth(auc, n, prev=0.5):
    """Hanley-McNeil 95% half-width."""
    n1, n0 = n * prev, n * (1 - prev)
    q1, q2 = auc / (2 - auc), 2 * auc * auc / (1 + auc)
    se = math.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc**2) + (n0 - 1) * (q2 - auc**2)) / (n1 * n0))
    return 1.96 * se


# ---------------------------------------------------------------- (e) measured validities (2026-11-01)

MODELS = Path(__file__).resolve().parents[1] / "models"


def n_for_r_power(r: float, alpha: float = 0.05, power: float = 0.80, sides: int = 2) -> int:
    """People needed to show a correlation r is above 0 (Fisher z)."""
    if r <= 0:
        return 10**9
    za, zb = stats.norm.ppf(1 - alpha / sides), stats.norm.ppf(power)
    return int(math.ceil(((za + zb) / np.arctanh(r)) ** 2 + 3))


def power_r(r: float, n: int, alpha: float = 0.05, sides: int = 2) -> float:
    if r <= 0:
        return alpha / sides
    return float(stats.norm.cdf(np.arctanh(r) * math.sqrt(n - 3) - stats.norm.ppf(1 - alpha / sides)))


def min_r_detectable(n: int, alpha: float = 0.05, power: float = 0.80, sides: int = 2) -> float:
    za, zb = stats.norm.ppf(1 - alpha / sides), stats.norm.ppf(power)
    return float(np.tanh((za + zb) / math.sqrt(n - 3)))


def measured_validities() -> dict:
    """Text-bigfive r per trait on Reddit / X writing (None = refused), MBTI-derived r per Big Five trait,
    and the zero-data fused r for someone with both, from the cards' validity_by_source blocks."""
    from aas.fusion import MBTI_LINKS, auc_to_r, gaussian_posterior

    b5 = json.loads((MODELS / "text_bigfive" / "card.json").read_text())
    vb = b5["validity_by_source"]
    tw = vb["sources"]["tweets"]["traits"]
    pan = b5["held_out_domains"]["per_domain"]["pan15"]["traits"]
    mb = json.loads((MODELS / "text_mbti" / "card.json").read_text())["validity_by_source"]
    out = {}
    for t in "OCEAN":
        text = None if t in vb["not_shown"] else round(auc_to_r(tw[t]["auc"]), 3)
        mbti = None
        for ax, (bt, _s, link) in MBTI_LINKS.items():
            if bt == t:
                mbti = round(mb["sources"]["other_prose"]["traits"][ax]["letter_r"] * link, 3)
        v = [x for x in (text, mbti) if x]
        if len(v) == 2:   # same text -> error correlation 0.4, as in sim_fusion.py
            _, var, _ = gaussian_posterior(np.zeros((1, 2)), np.array(v), np.array([[1, 0.4], [0.4, 1]]))
            fused = float(math.sqrt(1 - var[0]))
        else:
            fused = v[0] if v else 0.0
        out[t] = {"text_r_from_auc": text, "text_pearson_r_pan15": round(pan[t]["r"], 3),
                  "text_auc_95ci_pan15": [pan[t]["auc_lo"], pan[t]["auc_hi"]],
                  "mbti_r": mbti, "fused_r_text_plus_mbti": round(fused, 3)}
    return out


def section_e(rng, S: int, S_small: int) -> dict:
    mv = measured_validities()
    e: dict = {"measured": mv, "note": "text r from the AUC on tweets (PAN 2015); Reddit uses the same numbers (conservative stand-in). O: refused."}
    # Q1: is the text reader's validity above zero? power per trait, and n for 80%
    q1 = {}
    for t, v in mv.items():
        r = v["text_r_from_auc"] or 0.0
        q1[t] = {"r": r, "n80_two_sided_05": n_for_r_power(r), "n80_holm_first_01": n_for_r_power(r, 0.01),
                 **{f"power_n{n}": round(power_r(r, n), 3) for n in NS},
                 **{f"ci95_n{n}": [round(float(np.tanh(np.arctanh(r) - 1.96 / math.sqrt(n - 3))), 3),
                                   round(float(np.tanh(np.arctanh(r) + 1.96 / math.sqrt(n - 3))), 3)] for n in NS}}
    e["q1_text_validity_above_zero"] = q1
    # fused (text + MBTI, zero-data) r > 0
    e["fused_above_zero"] = {t: {"r": v["fused_r_text_plus_mbti"], "n80": n_for_r_power(v["fused_r_text_plus_mbti"]),
                                 **{f"power_n{n}": round(power_r(v["fused_r_text_plus_mbti"], n), 3) for n in NS}}
                             for t, v in mv.items()}
    # Q2 increments when the best single source is the measured text reader (r1 about 0.10)
    e["q2_increment_r1_0.10"] = [increment_power(n, 0.10, d, 0.0, S, rng) | {"n80_insample_Ftest_lower_bound": incremental_n_analytic(0.10, d, 0.0)}
                                 for d in (0.05, 0.10) for n in NS]
    # Q4 skill at the measured validities
    e["q4_skill"] = {f"r{r}": {n: skill_power(r, n, S_small, rng) for n in NS} for r in (0.10, 0.16)}
    # what a better text model must deliver
    e["needed_from_pandora"] = {
        "min_r_detectable_two_sided_05": {n: round(min_r_detectable(n), 3) for n in NS},
        "min_r_detectable_holm_first_01": {n: round(min_r_detectable(n, 0.01), 3) for n in NS},
        "min_r_for_q1_ci_to_exclude_0_at_n250": round(float(np.tanh(1.96 / math.sqrt(247))), 3),
        "omnibus_r1_0.20_delta_0.10_n250": "0.56 (section b)",
    }
    return e


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    S = 200 if args.quick else 1000          # increment sims per cell (multiple of 5)
    S_small = 300 if args.quick else 2000

    res: dict = {"assumptions": {
        "quiz": "IPIP-50, alpha 0.80-0.90; validities below are already against the quiz (attenuation included)",
        "text_bigfive_r": "0.20 central (0.10 to 0.30): cross-domain, Reddit/X vs 1990s essays; in-domain O was 0.40",
        "second_source": f"{P2}-feature model fit inside the study (Google or Spotify have no Big Five reader yet)",
        "shared_error_rho": "0 (independent) and 0.3 (sources share non-trait variance)",
        "coverage_target": 0.80,
    }}

    # (a)
    a = []
    for r in (0.10, 0.20, 0.30):
        row = {"r": r, "n_for_pm_0.10": n_for_r_halfwidth(r)}
        for n in NS:
            row[f"halfwidth_n{n}"] = round(r_halfwidth(r, n), 3)
            row[f"sim_halfwidth_n{n}"] = round(sim_r_halfwidth(r, n, S_small, rng), 3)
        a.append(row)
        print("(a)", row)
    res["a_validity_precision"] = a

    # (b)
    b = []
    for r1 in (0.20, 0.30):
        for delta in (0.05, 0.10):
            for rho in (0.0, 0.3):
                lb = incremental_n_analytic(r1, delta, rho)
                for n in NS:
                    row = increment_power(n, r1, delta, rho, S, rng)
                    row["n80_insample_Ftest_lower_bound"] = lb
                    b.append(row)
                    print("(b)", row)
    res["b_increment"] = b

    # (b2) n for 80% power of the CV stacked test, one trait, alpha .05 and .01, coarse grid
    b2 = []
    grid = [100, 150, 250, 400, 600, 900, 1300]
    for r1, delta, rho in [(0.20, 0.10, 0.0), (0.20, 0.05, 0.0), (0.30, 0.10, 0.0), (0.20, 0.10, 0.3)]:
        found = {"r1": r1, "delta": delta, "rho": rho}
        for n in grid:
            row = increment_power(n, r1, delta, rho, max(100, S // 2), rng)
            found[f"power05_n{n}"] = row["power_one_trait_alpha05"]
            found[f"power01_n{n}"] = row["power_one_trait_alpha01_holm_first"]
            found[f"omni05_n{n}"] = row["power_omnibus_5_traits_alpha05"]
        b2.append(found)
        print("(b2)", found)
    res["b2_power_curve"] = b2

    # (c)
    c = {"n_for_pm_0.05_one_trait": n_for_coverage(),
         "halfwidth_one_trait": {n: round(coverage_halfwidth(n), 3) for n in NS},
         "halfwidth_pooled_5_traits": {}}
    for icc in (0.0, 0.2, 0.5):
        c["halfwidth_pooled_5_traits"][f"icc{icc}"] = {n: round(sim_coverage_pooled(n, icc, S_small, rng), 3) for n in NS}
    print("(c)", c)
    res["c_coverage"] = c

    # (d)
    d = {"cv_r_10_feature_model": {}, "retest_r_halfwidth": {}, "mbti_auc_halfwidth": {}}
    for R in (0.10, 0.15, 0.25):
        d["cv_r_10_feature_model"][f"R{R}"] = {n: cv_r_of_learned_model(n, R, P2, max(200, S_small // 4), rng) for n in NS}
    for n in (30, 60, 100):
        d["retest_r_halfwidth"][n] = round(r_halfwidth(0.85, n), 3)
    for auc in (0.60, 0.70):
        d["mbti_auc_halfwidth"][f"auc{auc}"] = {n: round(auc_halfwidth(auc, n), 3) for n in NS}
    # skill last, so adding it did not change the random draws of everything above
    d["skill_calibrated"] = {f"r{r}": {n: skill_power(r, n, S_small, rng) for n in NS} for r in (0.10, 0.20, 0.30)}
    print("(d)", d)
    res["d_side"] = d

    # (e) last, so (a)-(d) keep their random draws
    e = section_e(rng, S, S_small)
    print("(e)", json.dumps(e, indent=1, default=str))
    res["e_measured_2026_11_01"] = e

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
