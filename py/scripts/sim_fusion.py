"""Simulation for the fusion methods in aas.fusion (docs/fusion.md).

Synthetic people with true (quiz) Big Five z-scores, four sources with the validities we measured,
correlated source errors, realistic missingness, and training sets of n in {0, 60, 250, 600}.
Every method is scored on 3,000 fresh test people per seed.

    cd py && uv run python scripts/sim_fusion.py            # full run (~minutes)
    uv run python scripts/sim_fusion.py --seeds 3 --test 1000             # quick look

Scenarios: honest / optimistic / hostile use the validities ASSUMED before any cross-domain test (kept so
the 2026-11-01 results reproduce exactly). measured / measured_old_stated / measured_hostile (added
2026-11-01) use the validities MEASURED on writing the readers never saw (models/*/card.json
"validity_by_source"): see MEASURED_V below and docs/fusion.md "Measured validities, 2026-11-01".
v1 / v1_distilled (added 2026-11-05) keep only the sources v1 fuses (text-bigfive, text-mbti); v1_distilled is a
sensitivity row with the failed distilled reader in place of text-bigfive (MATH.md 5.6):
    uv run python scripts/sim_fusion.py --scenarios v1,v1_distilled --ns 0     # -> docs/fusion_sim_v1.csv

Writes docs/fusion_sim.csv (every seed x scenario x n x method x trait) and prints the tables.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from aas.fusion import DISCOUNT, MBTI_LINKS, auc_to_r, gaussian_posterior, make_method
from aas.schema import TRAITS

OUT = Path(__file__).resolve().parents[1] / "docs"
LEVEL = 0.80

# ---------------------------------------------------------------- the world

# True validities (corr of each source's standardised score with the quiz z), per trait O C E A N.
# text-bigfive: Essays held-out implied r .40/.16/.21/.19/.19; assumed a little lower on real
#   people's own writing, inside the 0.2 to 0.35 band we measured or expect.
# text-mbti: letter r on new data (0.40, between the Essays transfer floor ~0.19 and the in-domain
#   0.62 implied by AUC .79) x the McCrae & Costa link (|r| .72 O, .49 C, .74 E, .44 A; no N link).
# music: hypothetical Big Five mapping at about 0.1 (taste predicts personality weakly).
# sensors: harmonised activity features, E ~0.15, C ~0.12, others ~0 (sensors_pooled.md).
RHO_LETTER = 0.40
LINK = {MBTI_LINKS[a][0]: MBTI_LINKS[a][2] for a in MBTI_LINKS}       # trait -> |link r|
TRUE_V = {
    "text":    dict(O=0.32, C=0.20, E=0.25, A=0.20, N=0.22),
    "mbti":    {t: (RHO_LETTER * LINK[t] if t in LINK else np.nan) for t in TRAITS},
    "music":   dict(O=0.12, C=0.05, E=0.10, A=0.08, N=0.08),
    "sensors": dict(O=0.03, C=0.12, E=0.15, A=0.00, N=0.05),
}
SOURCES = list(TRUE_V)
GROUP = {"text": "text", "mbti": "text", "music": "music", "sensors": "sensors"}

# What the readers would STATE (their in-domain validity). "honest": stated = true.
# "optimistic": stated = what our readers actually report in their own datasets, which is higher
#   than what they will get on new people's data (domain shift): text x1.35, mbti from the in-domain
#   letter AUC (.787 -> r .62) x link, music x1.5, sensors from round-1 (E .26, C .20).
STATED = {
    "honest": TRUE_V,
    "optimistic": {
        "text": {t: min(v * 1.35, 0.9) for t, v in TRUE_V["text"].items()},
        "mbti": {t: (auc_to_r(0.787) * LINK[t] if t in LINK else np.nan) for t in TRAITS},
        "music": {t: v * 1.5 for t, v in TRUE_V["music"].items()},
        "sensors": dict(O=0.10, C=0.20, E=0.26, A=0.05, N=0.10),
    },
}
# "hostile": the readers state the optimistic numbers, but on new people's data text works at half its
#   true value above, the MBTI letters at r 0.2 (the Essays transfer floor) and music not at all.
HOSTILE_V = {
    "text": {t: v * 0.5 for t, v in TRUE_V["text"].items()},
    "mbti": {t: (0.20 * LINK[t] if t in LINK else np.nan) for t in TRAITS},
    "music": {t: 0.0 for t in TRAITS},
    "sensors": TRUE_V["sensors"],
}
STATED["hostile"] = STATED["optimistic"]
WORLD_V = {"honest": TRUE_V, "optimistic": TRUE_V, "hostile": HOSTILE_V}

# ---- Measured validities (2026-11-01), what a Reddit / X user's text actually supports.
# text-bigfive on tweets (PAN 2015, held-out domain) and the conservative Reddit stand-in give the same AUCs
# for C E A N: .544 .543 .542 .571 -> r .098 .096 .095 .159. Openness is REFUSED off-domain (CI includes 0.5
# on every held-out domain), so a Reddit / X user has no text-bigfive O at all (NaN = source absent).
# text-mbti is untested off the forum with MBTI labels; its off-forum letter r is inferred from the Essays
# cross-construct test: I/E .251, N/S .273, T/F .262, J/P .255, times the McCrae & Costa link.
# Music and sensors: unchanged assumptions (nothing new measured).
_CARD = Path(__file__).resolve().parents[1] / "models"


def _measured_from_cards():
    import json

    b5 = json.loads((_CARD / "text_bigfive" / "card.json").read_text())["validity_by_source"]
    mb = json.loads((_CARD / "text_mbti" / "card.json").read_text())["validity_by_source"]
    tw = b5["sources"]["tweets"]["traits"]
    text = {t: (np.nan if t in b5["not_shown"] else round(auc_to_r(tw[t]["auc"]), 4)) for t in TRAITS}
    letter = {ax: mb["sources"]["other_prose"]["traits"][ax]["letter_r"] for ax in MBTI_LINKS}
    mbti = {t: np.nan for t in TRAITS}
    for ax, (t, _sign, link) in MBTI_LINKS.items():
        mbti[t] = round(letter[ax] * link, 4)
    return text, mbti, letter


MEASURED_TEXT, MEASURED_MBTI, MEASURED_LETTER = _measured_from_cards()
MEASURED_V = {"text": MEASURED_TEXT, "mbti": MEASURED_MBTI, "music": TRUE_V["music"], "sensors": TRUE_V["sensors"]}
# What the readers stated BEFORE 2026-11-01 for the same people: text-bigfive its Essays curve at full length
# (r .40 .16 .21 .19 .19), text-mbti its forum AUC (letter r .62 .60 .79 .46) x link. On real users openness
# had r about 0 (it was reported anyway), so its truth is 0 there.
OLD_STATED = {
    "text": dict(O=0.40, C=0.16, E=0.21, A=0.19, N=0.19),
    "mbti": {t: np.nan for t in TRAITS},
    "music": STATED["optimistic"]["music"],
    "sensors": STATED["optimistic"]["sensors"],
}
for _ax, (_t, _s, _l) in MBTI_LINKS.items():
    OLD_STATED["mbti"][_t] = auc_to_r({"I/E": 0.787, "N/S": 0.778, "T/F": 0.878, "J/P": 0.709}[_ax]) * _l
MEASURED_OLD_TRUTH = {**MEASURED_V, "text": {**MEASURED_TEXT, "O": 0.0}}
# "measured_hostile": readers state the measured numbers, but the truth sits near the low end of their CIs:
# text at half the measured r, MBTI letters at r 0.13 (the weakest tweet spot check, N/S), music worthless.
MEASURED_HOSTILE = {
    "text": {t: v * 0.5 for t, v in MEASURED_TEXT.items()},
    "mbti": {t: (0.13 * MBTI_LINKS[a][2] if t == MBTI_LINKS[a][0] else np.nan) for a in MBTI_LINKS for t in [MBTI_LINKS[a][0]]}
            | {t: np.nan for t in TRAITS if t not in {v[0] for v in MBTI_LINKS.values()}},
    "music": {t: 0.0 for t in TRAITS},
    "sensors": TRUE_V["sensors"],
}
STATED |= {"measured": MEASURED_V, "measured_old_stated": OLD_STATED, "measured_hostile": MEASURED_V}
WORLD_V |= {"measured": MEASURED_V, "measured_old_stated": MEASURED_OLD_TRUTH, "measured_hostile": MEASURED_HOSTILE}

# ---- v1 (2026-11-05): only what the v1 default readers can fuse. FusedReader fuses text-bigfive and text-mbti;
# music has no tested Big Five link and v1 has no sensor reader, so neither is a source here (NaN = absent).
# The "measured" scenario above kept hypothetical music and sensor sources, which v1 does not have.
# v1_distilled is a SENSITIVITY row, not a verdict: text-distilled (failed its pre-registered rule, not the
# default) replaces text-bigfive for non-forum text, at the AUCs its card states for tweets (measured on all 294
# PAN 2015 people); traits at or below AUC 0.5 there (O, C) are refused by that reader, so they are absent.
_NONE = {t: np.nan for t in TRAITS}


def _distilled_from_card():
    import json

    k = json.loads((_CARD / "text_distilled" / "card.json").read_text())["validity_by_source"]["kinds"]["tweets"]
    return {t: (round(auc_to_r(k[t]["auc"]), 4) if k[t]["auc"] > 0.5 else np.nan) for t in TRAITS}


DISTILLED_TEXT = _distilled_from_card()
V1_V = {"text": MEASURED_TEXT, "mbti": MEASURED_MBTI, "music": _NONE, "sensors": _NONE}
V1_DISTILLED_V = {**V1_V, "text": DISTILLED_TEXT}
STATED |= {"v1": V1_V, "v1_distilled": V1_DISTILLED_V}
WORLD_V |= {"v1": V1_V, "v1_distilled": V1_DISTILLED_V}
SCENARIOS = ("honest", "optimistic", "hostile", "measured", "measured_old_stated", "measured_hostile",
             "v1", "v1_distilled")
# Readers can also be mis-scaled on new data (their u not exactly mean 0 sd 1): applied in "optimistic".
MISCAL = {"text": (0.10, 1.15), "mbti": (-0.10, 1.10), "music": (0.15, 1.25), "sensors": (0.0, 1.20)}

# Error correlations between sources (same text -> strongly correlated; behaviour sources mildly).
ERR_R = np.array([
    # text  mbti  music sensors
    [1.00, 0.40, 0.10, 0.05],
    [0.40, 1.00, 0.10, 0.05],
    [0.10, 0.10, 1.00, 0.15],
    [0.05, 0.05, 0.15, 1.00],
])

# Big Five intercorrelations of the truth (typical adult self-report, e.g. van der Linden et al. 2010).
TRUTH_R = np.array([
    #  O     C     E     A     N
    [1.00, 0.05, 0.20, 0.10, -0.10],
    [0.05, 1.00, 0.15, 0.25, -0.30],
    [0.20, 0.15, 1.00, 0.15, -0.25],
    [0.10, 0.25, 0.15, 1.00, -0.25],
    [-0.10, -0.30, -0.25, -0.25, 1.00],
])

# Missingness: text 65% (of whom 85% also have enough for the MBTI reader), music 45%, sensors 30%,
# independent. Gives ~13% with nothing, ~43% one source, ~33% two, ~11% three, ~1% four.
P_TEXT, P_MBTI_GIVEN_TEXT, P_MUSIC, P_SENSORS = 0.65, 0.85, 0.45, 0.30


def simulate(n: int, rng: np.random.Generator, scenario: str):
    """Returns y (n, 5) quiz z, and per trait U, V, Vsd (n, J) with NaN = missing, groups."""
    y = rng.multivariate_normal(np.zeros(5), TRUTH_R, size=n)
    have_text = rng.random(n) < P_TEXT
    have = {
        "text": have_text,
        "mbti": have_text & (rng.random(n) < P_MBTI_GIVEN_TEXT),
        "music": rng.random(n) < P_MUSIC,
        "sensors": rng.random(n) < P_SENSORS,
    }
    out = {}
    for ti, t in enumerate(TRAITS):
        # a source exists for a trait when this world has a validity for it (measured: no text O, refused)
        srcs = [s for s in SOURCES if np.isfinite(WORLD_V[scenario][s][t]) and np.isfinite(STATED[scenario][s][t])]
        idx = [SOURCES.index(s) for s in srcs]
        E = rng.multivariate_normal(np.zeros(len(srcs)), ERR_R[np.ix_(idx, idx)], size=n)
        U, V, Vsd = (np.full((n, len(srcs)), np.nan) for _ in range(3))
        for j, s in enumerate(srcs):
            v = WORLD_V[scenario][s][t]
            u = v * y[:, ti] + np.sqrt(1 - v * v) * E[:, j]
            if scenario in ("optimistic", "hostile", "measured_old_stated"):
                loc, sc = MISCAL[s]
                u = loc + sc * u
            m = have[s]
            U[m, j] = u[m]
            V[m, j] = STATED[scenario][s][t]
            Vsd[m, j] = 0.15 if s == "mbti" else 0.10
        out[t] = (U, V, Vsd, [GROUP[s] for s in srcs], srcs)
    return y, out


# ---------------------------------------------------------------- scoring

def score(y, mu, lo, hi):
    ok = np.isfinite(mu)
    y, mu, lo, hi = y[ok], mu[ok], lo[ok], hi[ok]
    mae, base = np.mean(np.abs(mu - y)), np.mean(np.abs(y))
    return dict(n=int(ok.sum()), r=float(np.corrcoef(mu, y)[0, 1]) if mu.std() > 0 else np.nan,
                mae=float(mae), skill=float(1 - mae / base), coverage=float(np.mean((lo <= y) & (y <= hi))),
                width=float(np.mean(hi - lo)))


METHOD_ROWS = [
    # (label, method name in aas.fusion, kwargs) ; "single" at n=0 uses stated validity
    ("single", "single", {}),
    ("precision", "precision", {}),
    ("stack", "stack", {}),
    ("bayes (no discount)", "bayes", {"discount": 1.0}),
    ("bayes", "bayes", {}),
    ("bayes_eb", "bayes_eb", {}),
    ("bayes_eb_cp", "bayes_eb_cp", {}),
]


def run_one(seed: int, scenario: str, n_train: int, n_test: int, extras: bool):
    rng = np.random.default_rng(1000 * seed + SCENARIOS.index(scenario))
    ytr, tr = simulate(n_train, rng, scenario) if n_train else (None, None)
    yte, te = simulate(n_test, rng, scenario)
    rows = []
    for ti, t in enumerate(TRAITS):
        U, V, Vsd, groups, srcs = te[t]
        nsrc = (~np.isnan(U)).sum(1)
        for label, name, kw in METHOD_ROWS:
            if n_train == 0 and name in ("precision", "stack", "bayes_eb", "bayes_eb_cp"):
                continue
            if n_train and label == "bayes (no discount)" and not extras:
                continue
            m = make_method(name, n_train, LEVEL, seed=seed, **kw)
            if n_train:
                Ut, Vt, Vsdt, _, _ = tr[t]
                m.fit(Ut, ytr[:, ti], Vt, Vsdt, groups)
            else:
                m.fit(None, [], None, None, groups)
            mu, lo, hi = m.predict(U, V, Vsd)
            base = dict(seed=seed, scenario=scenario, n_train=n_train, method=label, trait=t)
            rows.append({**base, "subset": "all", **score(yte[:, ti], mu, lo, hi)})
            for k in (1, 2, 3):
                sel = nsrc == k if k < 3 else nsrc >= 3
                if sel.sum() > 20:
                    rows.append({**base, "subset": f"{k}{'+' if k == 3 else ''} sources",
                                 **score(yte[sel, ti], mu[sel], lo[sel], hi[sel])})
        # "no source -> no claim": every method returns NaN for people with nothing
    return rows


def oracle_table(TV=None):
    """Population r of the best possible fusion (true validities, true error correlations), for
    nested source sets, and the same if errors were independent."""
    TV = TRUE_V if TV is None else TV
    sets = [["text"], ["text", "mbti"], ["text", "mbti", "music"], ["text", "mbti", "music", "sensors"],
            ["music"], ["sensors"], ["music", "sensors"]]
    rows = []
    for t in TRAITS:
        for ss in sets:
            ss2 = [s for s in ss if np.isfinite(TV[s][t])]
            if not ss2:
                continue
            idx = [SOURCES.index(s) for s in ss2]
            v = np.array([TV[s][t] for s in ss2])
            for label, R in (("as simulated", ERR_R[np.ix_(idx, idx)]), ("independent errors", np.eye(len(idx)))):
                _, var, _ = gaussian_posterior(np.zeros((1, len(v))), v, R)
                rows.append(dict(trait=t, sources=" + ".join(ss), errors=label, r=float(np.sqrt(1 - var[0]))))
    return pd.DataFrame(rows)


def two_source_table():
    """Why errors that move together limit the gain: two sources of r = 0.30 each."""
    rows = []
    for rho in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        v = np.array([0.30, 0.30])
        R = np.array([[1, rho], [rho, 1]]) if rho < 1 else np.array([[1, 0.9999], [0.9999, 1]])
        _, var, w = gaussian_posterior(np.zeros((1, 2)), v, R)
        rows.append(dict(error_corr=rho, fused_r=float(np.sqrt(1 - var[0])), r_single=0.30))
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--test", type=int, default=3000)
    ap.add_argument("--ns", type=str, default="0,60,250,600")
    ap.add_argument("--scenarios", type=str, default=",".join(SCENARIOS),
                    help="subset to run; the CSV is then written to fusion_sim_<first scenario>.csv instead")
    a = ap.parse_args(argv)
    scen = [x for x in a.scenarios.split(",") if x]
    ns = [int(x) for x in a.ns.split(",")]
    t0 = time.time()
    rows = []
    for scenario in scen:
        for n in ns:
            for s in range(a.seeds):
                rows += run_one(s, scenario, n, a.test, extras=True)
            print(f"  {scenario} n={n} done ({time.time() - t0:.0f}s)", file=sys.stderr)
    df = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / ("fusion_sim.csv" if list(scen) == list(SCENARIOS) else f"fusion_sim_{scen[0]}.csv"), index=False)

    # mean over traits within seed, then mean and sd over seeds
    pd.set_option("display.width", 200)
    allp = df[df.subset == "all"].groupby(["scenario", "n_train", "method", "seed"])[
        ["r", "skill", "coverage", "width"]].mean().reset_index()
    summ = allp.groupby(["scenario", "n_train", "method"]).agg(["mean", "std"]).drop(columns="seed")
    print("\n== all people with >= 1 source, mean over traits; mean (sd over seeds) ==")
    for sc in scen:
        print(f"\n-- scenario: {sc}")
        print(fmt(summ.loc[sc]))
    by = df[df.subset != "all"].groupby(["scenario", "n_train", "method", "subset", "seed"])[
        ["r", "skill", "coverage", "width"]].mean().reset_index()
    bsum = by.groupby(["scenario", "n_train", "method", "subset"])[["r", "skill", "coverage", "width"]].mean()
    print("\n== by number of sources (mean over traits and seeds) ==")
    for sc in scen:
        for n in ns:
            for meth in (["single", "bayes"] if n == 0 else ["stack", "bayes_eb", "bayes_eb_cp"]):
                try:
                    print(f"{sc:<10} n={n:<4} {meth:<12}", bsum.loc[(sc, n, meth)].round(3).to_dict("index"))
                except KeyError:
                    pass
    per_t = df[(df.subset == "all")].groupby(["scenario", "n_train", "method", "trait"])[["r", "coverage"]].mean()
    for sc in scen:
        print(f"\n== per trait r ({sc}) ==")
        print(per_t.loc[sc]["r"].unstack("trait")[TRAITS].round(3).to_string())
        print(f"\n== per trait coverage ({sc}) ==")
        print(per_t.loc[sc]["coverage"].unstack("trait")[TRAITS].round(3).to_string())
    per_t4 = df[(df.subset == "all")].groupby(["scenario", "n_train", "method", "trait"])[
        ["r", "skill", "coverage", "width", "n"]].mean()
    for sc in [s for s in scen if s.startswith("v1")]:
        print(f"\n== {sc}: per trait at n_train = 0, mean over seeds ==")
        print(per_t4.loc[(sc, 0)].unstack("trait").round(3).to_string())
    if any(s.startswith("v1") for s in scen):
        print("v1 text r", MEASURED_TEXT, "distilled text r", DISTILLED_TEXT, "mbti r", MEASURED_MBTI)
    print("\n== oracle fusion r (true parameters, assumed) ==")
    o = oracle_table()
    print(o.pivot_table(index=["sources"], columns=["errors", "trait"], values="r").round(3).to_string())
    print("\n== oracle fusion r (MEASURED validities, 2026-11-01) ==")
    o = oracle_table(MEASURED_V)
    print(o.pivot_table(index=["sources"], columns=["errors", "trait"], values="r").round(3).to_string())
    print("measured text", MEASURED_TEXT, "mbti", MEASURED_MBTI, "letter", MEASURED_LETTER)
    print("\n== two sources, r = 0.30 each ==")
    print(two_source_table().round(3).to_string(index=False))
    print(f"\n{time.time() - t0:.0f}s", file=sys.stderr)


def fmt(s: pd.DataFrame) -> str:
    out = []
    for (n, meth), row in s.iterrows():
        cells = [f"{row[(c, 'mean')]:+.3f} ({row[(c, 'std')]:.3f})" for c in ("r", "skill", "coverage", "width")]
        out.append(f"  n={n:<4} {meth:<20} r {cells[0]}  skill {cells[1]}  cov80 {cells[2]}  width {cells[3]}")
    return "\n".join(out)


if __name__ == "__main__":
    main()
