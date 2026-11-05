"""A fake AAS study folder with known truth, to test scripts/study_analysis.py before any real data.

    cd py
    uv run python scripts/simulate_study.py --n 250 --out /tmp/sim250          # one fake study + truth.json
    uv run python scripts/simulate_study.py --dryrun 60,250,600 --out /tmp/dry  # simulate, analyse, recovery tables

The world (all numbers are assumptions taken from PROTOCOL.md, POWER.md and docs/fusion.md):

- Latent Big Five with realistic intercorrelations; IPIP-50 answers generated item by item from the
  norms' item means and variances (so quiz.score, partial forms and the quiz-quality rules really run).
- Text reader, MEASURED validities (POWER.md section 9, fusion.md 5b, 1 Nov 2026): against the quiz about
  r .10 C, .10 E, .09 A, .16 N on Reddit/X (latent .11 .11 .10 .17), mildly rising with words (x0.8 at 100
  words to x1.2 at 20,000), same on Reddit and X (Reddit uses the tweet numbers). OPENNESS IS REFUSED
  (usable=False, validity 0, no number) unless most of the person's words are essay-like, which we give
  8% of people; for them O works at latent r .22 but the reader states its essay validity (AUC .68, r .40).
  The reader states the measured AUCs (C .54, E .54, A .54, N .57) and builds z and its 80% interval from
  them, exactly as text-bigfive does.
- text-mbti: letter r 0.26 on new data (inferred; stated the same); errors correlated 0.4 with the Big
  Five text errors (same text). OEJTS letters from the same latent MBTI scales (reliability 0.8), done by 90%.
- Google (85% have it, 92% of those pass the floor): 8 behaviour features + 20 topic shares weakly
  tied to the traits (best linear R about 0.1 to 0.25). Spotify (55%, 88% pass): 9 listening features,
  weaker still (R about 0.05 to 0.15). Everyone has writing plus at least one of the two.
- Funnel: Session 2 return 50% (slightly higher for conscientious people: non-random dropout); 2%
  have under 100 words; misconduct and integrity failures at a few percent each; 10 pilot people; 1%
  withdraw; 4% skip an item or two, 1.5% skip three of one trait (that trait goes missing).
- Retest: 25% of Session 1 completers (min 30, cap 100), 85% complete, 92% inside days 12 to 20.

Generation stops when the n-th participant who should be usable has been recruited (the PREREG
stopping rule), so a correct analysis finds exactly n usable people.

truth.json holds the population value of every estimand, computed on 40,000 simulated genuine
participants passed through the same selection, with no cross-validation (so learned-model increments
are the ORACLE values; the CV estimates at n = 250 to 600 are expected to fall short of them, POWER.md).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy import stats

from aas import quiz
from aas.fusion import MBTI_LINKS, Fuser, auc_to_r, make_method, r_to_auc, readings_from_estimates
from aas.person import Estimate
from aas.schema import TRAITS

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study_analysis as SA  # noqa: E402

EXTRACTOR = ("0.1.0", "9f2c" + "0" * 60)
MBTI_AXES = SA.MBTI_AXES

CORR = np.array([
    [1.00, 0.10, 0.20, 0.10, -0.05],
    [0.10, 1.00, 0.15, 0.25, -0.30],
    [0.20, 0.15, 1.00, 0.20, -0.25],
    [0.10, 0.25, 0.20, 1.00, -0.25],
    [-0.05, -0.30, -0.25, -0.25, 1.00]])
TEXT_V = np.array([0.22, 0.11, 0.11, 0.10, 0.17])      # latent; O only for essay-like writers
STATED_AUC = np.array([0.682, 0.54, 0.54, 0.54, 0.57])   # what the reader states (O: essay validity)
ESSAY_LIKE_SHARE = 0.08
PLATFORM_FACTOR = {"reddit": 1.0, "x": 1.0, "both": 1.0}
RHO_LETTER, RHO_LETTER_STATED, RHO_ERR = 0.26, 0.26, 0.40
OEJTS_FIRST_RATE = {"I/E": 0.55, "N/S": 0.55, "T/F": 0.45, "J/P": 0.55}

# Google: 8 behaviour latents (rows) x traits O C E A N
B_G = np.array([
    [0.00, 0.12, 0.00, 0.00, -0.06],   # earlier mean hour
    [0.00, 0.00, 0.10, 0.00, 0.00],    # concentration of the daily rhythm
    [0.00, 0.08, 0.06, 0.00, 0.04],    # days active
    [0.18, 0.00, 0.00, 0.00, 0.00],    # query length
    [0.10, 0.00, 0.00, 0.00, 0.10],    # question share
    [0.14, 0.00, 0.05, 0.00, 0.00],    # channel entropy
    [-0.08, 0.06, 0.00, 0.00, 0.00],   # top-10 channel share
    [-0.06, 0.00, 0.00, 0.00, 0.08]])  # repeat watches
TOPICS = ["sport", "gaming", "cooking", "tech", "finance", "travel", "music", "film_tv", "science", "education",
          "cars", "fashion", "parenting", "pets", "diy", "news", "comedy", "books", "outdoors", "other"]
B_T = np.zeros((20, 5))
for name, (o, c, e, a, n) in {"science": (.4, 0, 0, 0, 0), "education": (.3, .1, 0, 0, 0), "travel": (.2, 0, .2, 0, 0),
                              "sport": (0, 0, .3, 0, 0), "comedy": (0, 0, .2, 0, .1), "finance": (0, .3, 0, 0, 0),
                              "diy": (0, .2, 0, 0, 0), "gaming": (-.1, -.2, -.1, 0, .1), "cooking": (0, 0, 0, .2, 0),
                              "parenting": (0, .1, 0, .2, 0), "pets": (0, 0, 0, .15, 0), "news": (.1, 0, 0, 0, .15),
                              "books": (.3, 0, -.1, 0, 0), "fashion": (0, 0, .15, 0, 0)}.items():
    B_T[TOPICS.index(name)] = (o, c, e, a, n)
B_S = np.array([
    [0.12, 0.00, 0.00, 0.00, 0.00],   # new_share
    [-0.06, 0.05, 0.00, 0.00, 0.00],  # old_friends
    [0.08, 0.00, 0.00, 0.00, 0.00],   # season_turnover
    [0.00, -0.06, 0.08, 0.00, 0.00],  # clock_spread
    [0.00, 0.10, 0.00, 0.00, 0.00],   # clock_regularity
    [0.00, 0.06, 0.00, 0.04, 0.00],   # finish_share
    [0.00, 0.00, 0.00, 0.00, 0.06],   # skip_seconds
    [0.14, 0.00, 0.00, 0.00, 0.00],   # dial_explore
    [0.00, 0.08, 0.00, 0.00, 0.00]])  # dial_regular
N_VOCAB = 150


def _norms():
    nm = quiz.load_norms()
    items = [c for t in TRAITS for c in nm["traits"][t]["items"]]
    mu = np.array([nm["traits"][quiz.KEY[c][0]]["item_means"][c] for c in items])
    var = np.array([np.diag(nm["traits"][quiz.KEY[c][0]]["item_cov"])[list(nm["traits"][quiz.KEY[c][0]]["items"]).index(c)] for c in items])
    sign = np.array([quiz.KEY[c][1] for c in items])
    tr = np.array([TRAITS.index(quiz.KEY[c][0]) for c in items])
    return items, mu, var, sign, tr


ITEMS, ITEM_MU, ITEM_VAR, ITEM_SIGN, ITEM_TRAIT = _norms()


def _sig(x):
    return 1 / (1 + np.exp(-x))


def answers_from(theta, rng):
    """Raw IPIP answers (N, 50) from latent traits: one factor per trait, item-level noise."""
    lam, se = np.sqrt(0.45 * ITEM_VAR), np.sqrt(0.55 * ITEM_VAR)
    k = ITEM_MU + lam * theta[:, ITEM_TRAIT] + se * rng.standard_normal((len(theta), 50))
    k = np.clip(np.rint(k), 1, 5)
    return np.where(ITEM_SIGN > 0, k, 6 - k).astype(int)


def quiz_z_full(ans):
    """Vectorised quiz.score for complete answers (full form)."""
    nm = quiz.load_norms()
    keyed = np.where(ITEM_SIGN > 0, ans, 6 - ans)
    out = np.zeros((len(ans), 5))
    for j, t in enumerate(TRAITS):
        m = keyed[:, ITEM_TRAIT == j].mean(1)
        out[:, j] = np.round((m - nm["traits"][t]["mean"]) / nm["traits"][t]["sd"], 3)
    return out


def _s_of_words(w):
    return np.clip((np.log(w) - np.log(100)) / (np.log(20000) - np.log(100)), 0, 1)


def _round_shares(p, dec=2):
    """Round shares to `dec` decimals keeping the sum at exactly 1 (largest remainder)."""
    scale = 10 ** dec
    x = p * scale
    f = np.floor(x)
    short = int(round(scale - f.sum()))
    if short > 0:
        f[np.argsort(-(x - f))[:short]] += 1
    return f / scale


def generate(N: int, rng) -> dict:
    """N genuine Session-1 completers, everything the funnel and payload need (vectorised)."""
    th = rng.multivariate_normal(np.zeros(5), CORR, size=N)
    g = {"theta": th, "answers": answers_from(th, rng)}
    g["log_ms"] = np.log(3500) + 0.25 * rng.standard_normal(N)
    th2 = np.sqrt(0.92) * th + np.sqrt(0.08) * rng.standard_normal((N, 5))
    g["answers_retest"] = answers_from(th2, rng)
    # sources
    has_g = rng.random(N) < 0.85
    has_s = rng.random(N) < 0.55
    none = ~(has_g | has_s)
    has_g[none] = rng.random(none.sum()) < 0.85 / (0.85 + 0.55)
    has_s[none & ~has_g] = True
    g["has_g"], g["has_s"] = has_g, has_s
    g["floor_g"] = has_g & (rng.random(N) < 0.92)
    g["floor_s"] = has_s & (rng.random(N) < 0.88)
    words = np.exp(np.log(1500) + 1.3 * rng.standard_normal(N))
    g["words"] = words
    g["floor_t"] = words >= 100
    g["returned"] = rng.random(N) < _sig(0.3 * th[:, 1])
    g["platform"] = rng.choice(["reddit", "x", "both"], size=N, p=[0.55, 0.35, 0.10])
    pf = np.array([PLATFORM_FACTOR[p] for p in g["platform"]])
    # text-bigfive
    v = np.clip(TEXT_V * (0.8 + 0.4 * _s_of_words(words))[:, None] * pf[:, None], 0, 0.9)
    e_text = rng.standard_normal((N, 5))
    u = v * th + np.sqrt(1 - v ** 2) * e_text
    r_st = np.tile([auc_to_r(a) for a in STATED_AUC], (N, 1))
    g["essay_like"] = rng.random(N) < ESSAY_LIKE_SHARE
    g["o_usable"] = g["essay_like"] & (words >= 100)
    g["v_true"], g["u"], g["r_stated"] = v, u, r_st
    g["m"] = np.clip(r_st * u, -3, 3)
    sd = np.sqrt(1 - r_st ** 2 + 0.05 ** 2)
    g["lo"], g["hi"] = g["m"] - 1.2816 * sd, g["m"] + 1.2816 * sd
    g["auc_stated"] = np.vectorize(lambda r: round(r_to_auc(r), 4))(r_st)
    g["head"] = 1.7 * u + 0.3
    # text-mbti and OEJTS
    P, oe = np.zeros((N, 4)), np.zeros((N, 4), bool)
    for a, ax in enumerate(MBTI_AXES):
        t, sign, link = MBTI_LINKS[ax]
        j = TRAITS.index(t)
        L = sign * link * th[:, j] + math.sqrt(1 - link ** 2) * rng.standard_normal(N)
        eps = RHO_ERR * sign * e_text[:, j] + math.sqrt(1 - RHO_ERR ** 2) * rng.standard_normal(N)
        s = RHO_LETTER * L + math.sqrt(1 - RHO_LETTER ** 2) * eps
        P[:, a] = np.clip(stats.norm.cdf(RHO_LETTER_STATED * s / math.sqrt(1 - RHO_LETTER_STATED ** 2)), 1e-4, 1 - 1e-4)
        O = math.sqrt(0.8) * L + math.sqrt(0.2) * rng.standard_normal(N)
        oe[:, a] = O > stats.norm.ppf(1 - OEJTS_FIRST_RATE[ax])
    g["mbti_P"], g["oejts"] = P, oe
    g["oejts_done"] = rng.random(N) < 0.9
    g["mbti_auc"] = round(r_to_auc(RHO_LETTER_STATED), 4)
    # Google
    def lat(B):
        expl = (B @ CORR * B).sum(1)
        return th @ B.T + np.sqrt(np.maximum(1 - expl, 0.05)) * rng.standard_normal((N, B.shape[0]))
    Lg = lat(B_G)
    mu_h = 21 - 2.5 * Lg[:, 0]
    kappa = 0.8 * np.exp(0.35 * Lg[:, 1])
    hours = np.arange(24)
    def hist(shift):
        z = np.exp(kappa[:, None] * np.cos(2 * np.pi * (hours[None] - mu_h[:, None] - shift) / 24))
        return z / z.sum(1, keepdims=True)
    g["search_h"], g["watch_h"] = hist(0.0), hist(1.0)
    g["has_watch"] = rng.random(N) < 0.85
    g["gfeat"] = np.c_[_sig(0.2 + 0.6 * Lg[:, 2]), np.exp(np.log(3.3) + 0.15 * Lg[:, 3]), _sig(-2.2 + 0.4 * Lg[:, 4]),
                       np.maximum(0, 4.5 + 0.9 * Lg[:, 5]), _sig(0.3 + 0.7 * Lg[:, 6]), _sig(-2.0 + 0.5 * Lg[:, 7])]
    logits = rng.normal(0, 0.3, 20) + 0.5 * th @ B_T.T + 0.6 * rng.standard_normal((N, 20))
    tp = np.exp(logits)
    g["topics"] = tp / tp.sum(1, keepdims=True)
    # Spotify
    Ls = lat(B_S)
    g["sfeat"] = np.c_[_sig(-1 + 0.5 * Ls[:, 0]), _sig(0.5 * Ls[:, 1]), _sig(-0.5 + 0.5 * Ls[:, 2]), 3 + 0.8 * Ls[:, 3],
                       _sig(0.5 * Ls[:, 4]), _sig(0.8 + 0.5 * Ls[:, 5]), np.exp(np.log(40) + 0.4 * Ls[:, 6]), Ls[:, 7], Ls[:, 8]]
    g["spct"] = np.round(100 * stats.norm.cdf(Ls), 1)
    g["basic"] = rng.random(N) < 0.3
    # common-word block (opt-in, >= 500 words)
    g["wr_opt"] = (words >= 500) & (rng.random(N) < 0.6)
    Bw = rng.normal(0, 0.15, (N_VOCAB, 5))
    g["wrates"] = np.exp(rng.normal(0, 1, N_VOCAB) + th @ Bw.T * 0.6 + 0.5 * rng.standard_normal((N, N_VOCAB)))
    return g


def usable_genuine(g) -> np.ndarray:
    return g["returned"] & g["floor_t"] & (g["floor_g"] | g["floor_s"])


# ---------------------------------------------------------------- the study folder

def _estimates(g, i):
    out = []
    for j, t in enumerate(TRAITS):
        if t == "O" and not g["o_usable"][i]:     # refused: not shown to work on this kind of writing
            out.append({"reader": "text-bigfive", "trait": t, "kind": "bigfive", "value": None, "low": None, "high": None,
                        "validity": 0.0, "validity_kind": "r", "usable": False, "interval_level": None})
            continue
        out.append({"reader": "text-bigfive", "trait": t, "kind": "bigfive", "value": round(float(g["m"][i, j]), 4),
                    "low": round(float(g["lo"][i, j]), 4), "high": round(float(g["hi"][i, j]), 4),
                    "validity": float(g["auc_stated"][i, j]), "validity_kind": "auc", "usable": True, "interval_level": 0.8})
    for a, ax in enumerate(MBTI_AXES):
        p = float(g["mbti_P"][i, a])
        out.append({"reader": "text-mbti", "trait": ax, "kind": "mbti", "value": round(p, 4),
                    "low": round(max(p - 0.1, 0), 4), "high": round(min(p + 0.1, 1), 4),
                    "validity": g["mbti_auc"], "validity_kind": "auc", "usable": True, "interval_level": 0.8})
    return out


def _payload(g, i):
    words = float(g["words"][i])
    plat = {"reddit": ["reddit"], "x": ["x"], "both": ["reddit", "x"]}[g["platform"][i]]
    p = {"sources": {"text": True, "google": bool(g["has_g"][i]), "spotify": bool(g["has_s"][i])},
         "floors": {"text": {"passed": bool(g["floor_t"][i]), "reason": None if g["floor_t"][i] else "too_few_words"},
                    "google": {"passed": bool(g["floor_g"][i]), "reason": None if g["floor_g"][i] else "below_floor"},
                    "spotify": {"passed": bool(g["floor_s"][i]), "reason": None if g["floor_s"][i] else "below_floor"}},
         "text": {"words": int(round(words, -2)), "platforms": plat, "estimates": _estimates(g, i),
                  "head_scores": {t: round(float(g["head"][i, j]), 4) for j, t in enumerate(TRAITS)},
                  "word_rates": ({f"w{k:03d}": float(f"{g['wrates'][i, k]:.3g}") for k in range(N_VOCAB)} if g["wr_opt"][i] else None)},
         "google": None, "spotify": None}
    if g["has_g"][i]:
        gf = g["gfeat"][i]
        hw = bool(g["has_watch"][i])
        p["google"] = {"search_hour_shares": _round_shares(g["search_h"][i]).tolist(),
                       "watch_hour_shares": _round_shares(g["watch_h"][i]).tolist() if hw else None,
                       "days_active_share": round(float(gf[0]), 3), "query_mean_words": round(float(gf[1]), 2),
                       "query_question_share": round(float(gf[2]), 3),
                       "channel_entropy": round(float(gf[3]), 3) if hw else None,
                       "top10_channel_share": round(float(gf[4]), 3) if hw else None,
                       "repeat_watch_share": round(float(gf[5]), 3) if hw else None,
                       "topic_shares": dict(zip(TOPICS, _round_shares(g["topics"][i]).tolist()))}
    if g["has_s"][i]:
        p["spotify"] = {"export": "basic" if g["basic"][i] else "extended",
                        "traits": {k: {"value": round(float(g["sfeat"][i, c]), 4), "percentile": float(g["spct"][i, c])}
                                   for c, k in enumerate(SA.SPOTIFY_FEATURES)}}
    return p


def write_study(n_usable: int, seed: int, out: Path) -> dict:
    """Recruit until n_usable people who should be usable, then write the folder. Returns bookkeeping."""
    rng = np.random.default_rng(seed)
    out = Path(out)
    for sub in ("session1", "payloads", "retest"):
        (out / sub).mkdir(parents=True, exist_ok=True)
        for f in (out / sub).glob("*.json"):
            f.unlink()
    rows = []          # (g, i, flags)
    batches = []
    n_ok = 0
    while n_ok < n_usable:
        g = generate(64, rng)
        batches.append(g)
        bad = {k: rng.random(64) < p for k, p in (("att2", 0.03), ("att1", 0.04), ("speed", 0.02), ("straight", 0.015),
                                                   ("identical", 0.005), ("consent_no", 0.02), ("unknown_ext", 0.01),
                                                   ("impossible", 0.01), ("duplicate", 0.01), ("withdrawn", 0.01),
                                                   ("miss_few", 0.04), ("miss_trait", 0.015))}
        # one misconduct type per person at most
        excl_q = bad["att2"] | bad["speed"] | bad["straight"] | bad["identical"]
        for k in ("speed", "straight", "identical"):
            bad[k] &= ~bad["att2"]
        bad["straight"] &= ~bad["speed"]
        bad["identical"] &= ~(bad["speed"] | bad["straight"])
        ok = usable_genuine(g) & ~excl_q & ~bad["consent_no"] & ~bad["unknown_ext"] & ~bad["impossible"] & ~bad["duplicate"] & ~bad["withdrawn"]
        for i in range(64):
            if n_ok >= n_usable:
                break
            rows.append((g, i, {k: bool(v[i]) for k, v in bad.items()}, bool(ok[i])))
            n_ok += int(ok[i])
    N1 = len(rows)
    t0 = datetime(2027, 3, 1, 9, 0, 0)
    n_ret = int(np.clip(round(0.25 * N1), 30, 100))
    retest_pick = set(rng.choice(N1, size=min(n_ret, N1), replace=False).tolist())
    withdrawn, returners = [], []
    expect = {"usable": 0, "rules": {}}
    pilot = [(generate(1, rng), 0) for _ in range(10)]
    allrows = [(g, i, {}, False, True) for g, i in pilot] + [(g, i, f, ok, False) for g, i, f, ok in rows]
    for k, (g, i, f, ok, is_pilot) in enumerate(allrows):
        sid = f"S{k:05d}"
        when = t0 + timedelta(hours=3 * k)
        ans = g["answers"][i].astype(object)
        ms = np.exp(g["log_ms"][i] + 0.4 * rng.standard_normal(50))
        if f.get("att2") or f.get("speed"):
            ans = rng.integers(1, 6, 50).astype(object)
        if f.get("speed"):
            ms = np.exp(np.log(600) + 0.3 * rng.standard_normal(50))
        page_pos = {c: quiz.PAGE_ORDER.index(c) for c in ITEMS}
        if f.get("straight"):
            start = int(rng.integers(0, 30))
            for c in ITEMS:
                if start <= page_pos[c] < start + 16:
                    ans[ITEMS.index(c)] = 4
        if f.get("identical"):
            ans[:] = 3
        if f.get("miss_few"):
            for c in rng.choice(50, size=int(rng.integers(1, 3)), replace=False):
                ans[c] = None
        if f.get("miss_trait"):
            j = int(rng.integers(0, 5))
            for c in rng.choice(np.nonzero(ITEM_TRAIT == j)[0], size=3, replace=False):
                ans[c] = None
        att_fail = 2 if f.get("att2") else (1 if f.get("att1") else 0)
        oejts_done = bool(g["oejts_done"][i])
        n_att = 3 if oejts_done else 2
        att = [{"id": f"IC{a + 1}", "expected": 4, "answer": 2 if a < att_fail else 4} for a in range(n_att)]
        s1 = {"study_id": sid, "session1_at": when.isoformat(), "pilot": is_pilot,
              "ipip": {"answers": {c: (None if a is None else int(a)) for c, a in zip(ITEMS, ans)},
                       "item_ms": {c: int(x) for c, x in zip(ITEMS, ms)}},
              "attention": att, "use_answers": "no" if f.get("consent_no") else "yes",
              "oejts": ({ax: (ax[0] if g["oejts"][i, a] else ax[2]) for a, ax in enumerate(MBTI_AXES)} if oejts_done else None),
              "retest_assigned": (k - 10) in retest_pick}
        (out / "session1" / f"{sid}.json").write_text(json.dumps(s1))
        if s1["retest_assigned"] and rng.random() < 0.85:
            days = rng.uniform(12, 20.9) if rng.random() < 0.92 else rng.choice([rng.uniform(8, 11.9), rng.uniform(21.5, 28)])
            ra = g["answers_retest"][i]
            rms = np.exp(g["log_ms"][i] + 0.4 * rng.standard_normal(50))
            (out / "retest" / f"{sid}.json").write_text(json.dumps(
                {"study_id": sid, "taken_at": (when + timedelta(days=float(days))).isoformat(),
                 "ipip": {"answers": {c: int(a) for c, a in zip(ITEMS, ra)}, "item_ms": {c: int(x) for c, x in zip(ITEMS, rms)}}}))
        if f.get("withdrawn"):
            withdrawn.append(sid)
        if not g["returned"][i]:
            continue
        payload = _payload(g, i)
        if f.get("impossible"):
            payload["text"]["estimates"][5]["value"] = 1.37      # P(I) above 1
        rec = {"study_id": sid, "extractor_version": EXTRACTOR[0], "extractor_hash": EXTRACTOR[1],
               "submitted_at": (when + timedelta(days=float(rng.uniform(2, 30)))).isoformat(), "payload": payload}
        if f.get("unknown_ext"):
            rec["extractor_version"] = "0.0.9-dev"
        returners.append((sid, rec, f))
    # duplicates: copy an earlier returner's payload, submitted later
    sids = [s for s, _, _ in returners]
    for k, (sid, rec, f) in enumerate(returners):
        if f.get("duplicate") and k > 0:
            src = returners[int(rng.integers(0, k))][1]
            rec["payload"] = json.loads(json.dumps(src["payload"]))
            rec["submitted_at"] = (datetime.fromisoformat(src["submitted_at"]) + timedelta(days=1)).isoformat()
        (out / "payloads" / f"{sid}.json").write_text(json.dumps(rec))
    (out / "withdrawn.json").write_text(json.dumps(withdrawn))
    (out / "study.json").write_text(json.dumps({"extractors": {EXTRACTOR[0]: EXTRACTOR[1]},
                                                 "pilot_ids": [f"S{k:05d}" for k in range(10)], "simulated": True, "seed": seed}))
    return {"n_session1": N1 + 10, "n_usable_expected": n_usable, "n_returned": len(sids)}


# ---------------------------------------------------------------- population truth

def _auc(pos, neg):
    x = np.r_[pos, neg]
    rk = stats.rankdata(x)
    return float((rk[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def population_truth(seed: int = 12345, N: int = 40000) -> dict:
    rng = np.random.default_rng(seed)
    g = generate(N, rng)
    q_all = quiz_z_full(g["answers"])
    sel = usable_genuine(g)
    q = q_all[sel]
    m, lo, hi = g["m"][sel], g["lo"][sel], g["hi"][sel]
    ou = g["o_usable"][sel]
    rows = [ou if j == 0 else np.ones(len(q), bool) for j in range(5)]      # O only where not refused
    T = {"text_O_usable_share": float(ou.mean()), "pooled_traits_text": ["C", "E", "A", "N"]}
    T["H1_r"] = {t: SA.r_(m[rows[j], j], q[rows[j], j]) for j, t in enumerate(TRAITS)}
    inside = (q >= lo) & (q <= hi)
    T["H3a_coverage"] = {t: float(inside[rows[j], j].mean()) for j, t in enumerate(TRAITS)}
    T["H3a_coverage_pooled"] = float(inside[:, 1:].mean())
    T["H3b_coverage_pooled"] = 0.80
    T["H4_skill_text_shipped"] = {t: float(1 - np.abs(q[rows[j], j] - m[rows[j], j]).mean() / np.abs(q[rows[j], j]).mean())
                                  for j, t in enumerate(TRAITS)}
    # oracle combined (population OLS, no CV)
    fg, fs = g["floor_g"][sel], g["floor_s"][sel]
    hc = np.zeros(sel.sum())
    hs = np.zeros(sel.sum())
    ang = 2 * np.pi * np.arange(24) / 24
    hist = np.where(g["has_watch"][sel][:, None], (g["search_h"][sel] + g["watch_h"][sel]) / 2, g["search_h"][sel])
    hc, hs = hist @ np.cos(ang), hist @ np.sin(ang)
    gf = g["gfeat"][sel].copy()
    gf[~g["has_watch"][sel], 3:] = np.nan
    tp = g["topics"][sel]
    tpc = (tp - tp[fg].mean(0)) @ np.linalg.svd(tp[fg] - tp[fg].mean(0), full_matrices=False)[2][:2].T
    G = np.c_[hc, hs, gf, tpc]
    G = np.where(np.isnan(G), np.nanmedian(G[fg], 0), G)
    G[~fg] = 0
    S = np.where(fs[:, None], g["sfeat"][sel], 0)
    T["H2_oracle_r_single"], T["H2_oracle_r_combined"], T["H2_oracle_delta_r"] = {}, {}, {}
    T["S1_oracle_R"] = {"google": {}, "spotify": {}}
    for j, t in enumerate(TRAITS):
        R = rows[j]
        X = np.c_[np.ones(len(q)), m[:, j], G, ~fg, S, ~fs][R]
        pred = X @ np.linalg.lstsq(X, q[R, j], rcond=None)[0]
        rs, rc = abs(SA.r_(m[R, j], q[R, j])), SA.r_(pred, q[R, j])
        T["H2_oracle_r_single"][t], T["H2_oracle_r_combined"][t], T["H2_oracle_delta_r"][t] = rs, rc, rc - rs
        for name, M, h in (("google", G, fg), ("spotify", S, fs)):
            X1 = np.c_[np.ones(h.sum()), M[h]]
            p1 = X1 @ np.linalg.lstsq(X1, q[h, j], rcond=None)[0]
            T["S1_oracle_R"][name][t] = SA.r_(p1, q[h, j])
    # S2
    P, oe, done = g["mbti_P"][sel], g["oejts"][sel], g["oejts_done"][sel]
    T["S2_auc"] = {ax: _auc(P[done & oe[:, a], a], P[done & ~oe[:, a], a]) for a, ax in enumerate(MBTI_AXES)}
    # S3
    lw = np.log(np.maximum(np.round(g["words"][sel], -2), 1))
    lw = lw - lw.mean()
    Xs, ys = [], []
    for j in range(5):
        fe = np.zeros((len(q), 5))
        fe[:, j] = 1
        Xs.append(np.c_[fe, m[:, j], lw, m[:, j] * lw])
        ys.append(q[:, j])
    T["S3_beta_product"] = float(np.linalg.lstsq(np.vstack(Xs), np.concatenate(ys), rcond=None)[0][-1])
    # S4 calibration refit gain (population MSE difference)
    T["S4_calibration_gain"] = {}     # O: refit from the raw score vs z = 0 (the reader gives no number), as the analysis
    for j, t in enumerate(TRAITS):
        b, a = np.polyfit(g["head"][sel][:, j], q[:, j], 1)
        ship = np.zeros(len(q)) if t == "O" else m[:, j]
        T["S4_calibration_gain"][t] = float(np.mean((q[:, j] - ship) ** 2 - (q[:, j] - a - b * g["head"][sel][:, j]) ** 2))
    # S5 on all Session-1 completers (retest is randomised at Session 1)
    q2 = quiz_z_full(g["answers_retest"])
    T["S5_retest_r"] = {t: SA.r_(q_all[:, j], q2[:, j]) for j, t in enumerate(TRAITS)}
    T["S5_icc21"] = {t: SA.icc21(q_all[:5000, j], q2[:5000, j]) for j, t in enumerate(TRAITS)}
    # S6
    pl = g["platform"][sel]
    T["S6_r_reddit"] = {t: SA.r_(m[(pl == "reddit") & rows[j], j], q[(pl == "reddit") & rows[j], j]) for j, t in enumerate(TRAITS)}
    T["S6_r_x"] = {t: SA.r_(m[(pl == "x") & rows[j], j], q[(pl == "x") & rows[j], j]) for j, t in enumerate(TRAITS)}
    T["S6_diff"] = {t: T["S6_r_reddit"][t] - T["S6_r_x"][t] for t in TRAITS}
    # S7 bayes (array path, same model FusedReader uses) on a subsample
    k = min(8000, len(q))
    idx = np.nonzero(sel)[0][:k]
    readings = [readings_from_estimates([Estimate(**e) for e in _estimates(g, i)])[0] for i in idx]
    qb = q_all[idx]
    mu, l2, h2 = (np.full((k, 5), np.nan) for _ in range(3))
    for j, t in enumerate(TRAITS):
        chans = sorted({r.channel for rs in readings for r in rs if r.trait == t})
        groups = [next(r.group for rs in readings for r in rs if r.channel == c) for c in chans]
        U, V, Vsd = Fuser._arrays(readings, t, chans)
        mm = make_method("bayes", 0).fit(None, [], None, None, groups)
        mu[:, j], l2[:, j], h2[:, j] = mm.predict(U, V, Vsd)
    T["S7_bayes_r"] = {t: SA.r_(mu[:, j], qb[:, j]) for j, t in enumerate(TRAITS)}
    ins = (qb >= l2) & (qb <= h2)
    T["S7_bayes_coverage"] = {t: float(ins[:, j].mean()) for j, t in enumerate(TRAITS)}
    T["S7_bayes_coverage_pooled"] = float(ins.mean())
    T["S7_bayes_skill"] = {t: float(1 - np.abs(qb[:, j] - mu[:, j]).mean() / np.abs(qb[:, j]).mean()) for j, t in enumerate(TRAITS)}
    T["S7_check_real_r"] = {}
    for j, t in enumerate(TRAITS):
        us = {}
        for c in sorted({r.channel for rs in readings for r in rs if r.trait == t}):
            u = np.array([next((r.u for r in rs if r.channel == c and r.trait == t), np.nan) for rs in readings])
            ok = np.isfinite(u)
            us[c] = (u, SA.r_(u[ok], qb[ok, j]))
            T["S7_check_real_r"][f"{t} {c}"] = us[c][1]
        if len(us) == 2:   # same definition as study_analysis.fusion_checks (people with both)
            (c1, (u1, v1)), (c2, (u2, v2)) = sorted(us.items())
            ok = np.isfinite(u1) & np.isfinite(u2)
            e1 = (u1[ok] - v1 * qb[ok, j]) / math.sqrt(1 - v1 ** 2)
            e2 = (u2[ok] - v2 * qb[ok, j]) / math.sqrt(1 - v2 ** 2)
            T["S7_check_real_r"][f"{t} error corr {c1} x {c2}"] = SA.r_(e1, e2)
    T["n_population_usable"] = int(sel.sum())
    T["true_latent_validity_text_base"] = dict(zip(TRAITS, TEXT_V.tolist()))
    return SA._clean(T)


# ---------------------------------------------------------------- recovery check

def recovery(res: dict, T: dict) -> list[dict]:
    """Estimate, CI and truth side by side for every estimand that has a population value."""
    rows = []

    def add(name, est, ci, truth, kind="ci"):
        if est is None or truth is None:
            return
        inside = None if ci is None or ci[0] is None or truth is None else bool(ci[0] <= truth <= ci[1])
        rows.append({"estimand": name, "estimate": est, "ci": ci, "truth": truth, "truth_in_ci": inside, "kind": kind})

    for r in res["H1"]["traits"]:
        add(f"H1 r text {r['trait']}", r.get("r"), r.get("ci95"), T["H1_r"].get(r.get("trait")))
    add("H3a coverage pooled", res["H3a"]["coverage"], res["H3a"]["ci90"], T["H3a_coverage_pooled"], "ci90")
    add("H3b coverage pooled", res["H3b"]["coverage"], res["H3b"]["ci90"], T["H3b_coverage_pooled"], "ci90 (nominal)")
    for r in res["H4"]["traits"]:
        add(f"H4 skill text {r['trait']}", (r.get("text_as_shipped") or {}).get("skill"), (r.get("text_as_shipped") or {}).get("ci95"), T["H4_skill_text_shipped"][r.get("trait")])
    for r in res["H2"]["traits"]:
        add(f"H2 delta r {r['trait']}", r.get("delta_r"), r.get("delta_r_ci95"), T["H2_oracle_delta_r"][r.get("trait")], "oracle (CV expected lower)")
    for name in ("google", "spotify"):
        for r in res["S1"][name]["traits"]:
            if "r" in r:
                add(f"S1 {name} r {r['trait']}", r.get("r"), r.get("ci95"), T["S1_oracle_R"][name][r.get("trait")], "oracle (CV expected lower)")
    for r in res["S2"]["axes"]:
        if "auc" in r:
            add(f"S2 AUC {r['axis']}", r.get("auc"), r.get("ci95"), T["S2_auc"][r.get("axis")])
    add("S3 product beta", res["S3"]["beta_product"], res["S3"]["ci95_product"], T["S3_beta_product"])
    for r in res["S4"]["calibration_refit_all"]["traits"]:
        if "mean" in r:
            add(f"S4 calibration gain {r['trait']}", r.get("mean"), r.get("ci95"), T["S4_calibration_gain"][r.get("trait")], "CV (in-sample oracle)")
    for r in res["S5"]["traits"]:
        if "r" in r:
            add(f"S5 retest r {r['trait']}", r.get("r"), r.get("r_ci95"), T["S5_retest_r"][r.get("trait")])
            add(f"S5 ICC(2,1) {r['trait']}", r.get("icc21"), r.get("icc21_ci95"), T["S5_icc21"][r.get("trait")])
    for r in res["S6"]["traits"]:
        if "diff" in r:
            add(f"S6 reddit - x {r['trait']}", r.get("diff"), r.get("ci95"), T["S6_diff"][r.get("trait")])
    for r in res["S7"]["bayes"]["traits"]:
        add(f"S7 bayes r {r['trait']}", r.get("r"), r.get("ci95"), T["S7_bayes_r"][r.get("trait")])
        add(f"S7 bayes skill {r['trait']}", r.get("skill"), r.get("skill_ci95"), T["S7_bayes_skill"][r.get("trait")])
    c = res["S7"]["bayes"]["coverage_pooled"]
    add("S7 bayes coverage pooled", c["coverage"], c["ci90"], T["S7_bayes_coverage_pooled"], "ci90")
    for r in res["S7"]["checks"]["rows"]:
        if "skipped" in r:
            continue
        key = f"{r['trait']} {r['channel']}"
        if key in T["S7_check_real_r"]:
            add(f"S7 real r {key}", r.get("real_r"), r.get("ci95"), T["S7_check_real_r"][key])
    return rows


def recovery_text(rows, n) -> str:
    out = [f"n = {n}: estimate [CI] vs population truth"]
    strict = [r for r in rows if r["kind"] in ("ci", "ci90")]
    hit = sum(bool(r["truth_in_ci"]) for r in strict)
    for r in rows:
        ci = "n/a" if not r["ci"] or r["ci"][0] is None else f"[{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]"
        mark = "" if r["truth_in_ci"] is None else ("in CI" if r["truth_in_ci"] else "OUTSIDE")
        est = "n/a" if r["estimate"] is None else f"{r['estimate']:+.3f}"
        out.append(f"  {r['estimand']:<44} {est:>7} {ci:<18} truth {r['truth']:+.3f}  {mark:<8} {r['kind'] if r['kind'] != 'ci' else ''}")
    out.append(f"  directly estimable quantities: truth inside the CI for {hit} of {len(strict)} ({hit / max(len(strict), 1):.0%}); "
               "expected about 95% (90% for the coverage rows)")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=250, help="usable participants to recruit")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dryrun", default=None, help="comma list of n: simulate, analyse and print recovery for each")
    ap.add_argument("--fast", action="store_true", help="analysis with --fast settings")
    a = ap.parse_args(argv)
    out = Path(a.out)
    truth = population_truth()
    if not a.dryrun:
        info = write_study(a.n, a.seed, out)
        (out / "truth.json").write_text(json.dumps(truth, indent=1, sort_keys=True))
        print(json.dumps(info))
        return 0
    texts = []
    for n in [int(x) for x in a.dryrun.split(",")]:
        d = out / f"n{n}"
        t0 = time.time()
        write_study(n, a.seed + n, d)          # independent studies (not nested samples)
        (d / "truth.json").write_text(json.dumps(truth, indent=1, sort_keys=True))
        cfg = SA.Config.fast() if a.fast else SA.Config()
        res = SA.analyse(d, cfg)
        (d / "results").mkdir(exist_ok=True)
        (d / "results" / "results.json").write_text(json.dumps(res, indent=1, sort_keys=True))
        (d / "results" / "report.txt").write_text(SA.report(res) + "\n")
        rows = recovery(res, truth)
        (d / "results" / "recovery.json").write_text(json.dumps(rows, indent=1))
        txt = recovery_text(rows, n) + f"\n  (usable found {res['sample']['n_usable']}, analysis {time.time() - t0:.0f} s)"
        (d / "results" / "recovery.txt").write_text(txt + "\n")
        print(txt, flush=True)
        texts.append(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
