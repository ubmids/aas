"""Build and check the IPIP ground-truth layer. Writes:

  data/raw/ipip/ipip_clean.parquet   cleaned answers + timing summaries (gitignored)
  src/aas/data/ipip_norms.json       norms + short form (committed, small)
  docs/quiz_results.json             every number quoted in docs/quiz.md

Run:  uv run python scripts/build_ipip.py [--delete-raw] [--perms 50]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr
from sklearn.decomposition import FactorAnalysis

from aas import quiz
from aas.evaluate import predict
from aas.schema import TRAITS, zscore

DOCS = Path(__file__).resolve().parents[1] / "docs"


def r(a, b) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delete-raw", action="store_true")
    ap.add_argument("--perms", type=int, default=50)
    ap.add_argument("--sample", type=int, default=20000)
    ap.add_argument("--short-k", type=int, default=4)
    args = ap.parse_args()
    res: dict = {}
    t0 = time.time()

    # 1. clean -------------------------------------------------------------
    if quiz.CLEAN.exists():
        df = quiz.load_clean()
        log = json.loads((quiz.RAW_DIR / "clean_log.json").read_text())
    else:
        df, log = quiz.build_clean(delete_raw=args.delete_raw)
    res["cleaning"] = [{"rule": a, "dropped": b, "remaining": c} for a, b, c in log]
    print("cleaning:", *log, sep="\n  ")
    items = df[quiz.ITEM_COLS]
    k = quiz.keyed(items)
    S = quiz.scales(items)

    # 2. keying proof ------------------------------------------------------
    # raw (unflipped) item vs the keyed total of the OTHER items of its trait:
    # reverse items should come out negative raw and positive after flipping.
    keyrows = []
    for t in TRAITS:
        cols = quiz.TRAIT_ITEMS[t]
        tot = k[cols].sum(axis=1)
        for c in cols:
            rest = tot - k[c]
            keyrows.append(
                {"item": c, "trait": t, "sign": quiz.KEY[c][1], "r_raw": round(r(items[c], rest), 3), "r_keyed": round(r(k[c], rest), 3)}
            )
    kr = pd.DataFrame(keyrows)
    res["keying"] = keyrows
    ok = bool((kr.r_keyed > 0).all() and (np.sign(kr.r_raw) == kr.sign).all())
    res["keying_ok"] = ok
    print(f"keying: all 50 items positive after flip = {(kr.r_keyed > 0).all()}, raw sign matches key = {ok}")
    print(f"  weakest keyed item-rest r: {kr.nsmallest(3, 'r_keyed')[['item','r_keyed']].to_dict('records')}")

    # 3. reliability + descriptives -----------------------------------------
    res["alpha_full"] = {t: round(quiz.cronbach_alpha(k[quiz.TRAIT_ITEMS[t]]), 3) for t in TRAITS}
    res["scale_desc"] = {t: {"mean": round(S[t].mean(), 3), "sd": round(S[t].std(), 3)} for t in TRAITS}
    res["scale_intercorr"] = S.corr().round(3).to_dict()
    print("alpha:", res["alpha_full"])

    # 4. short form: choose on train half, judge on held-out half -----------
    train, test = df[df.half == 0], df[df.half == 1]
    kt = quiz.keyed(test[quiz.ITEM_COLS])
    full_t = {t: kt[quiz.TRAIT_ITEMS[t]].mean(axis=1) for t in TRAITS}
    sf = {}
    for n_items in (2, 3, 4):
        for bal in (False, True):
            short = quiz.choose_short_form(train[quiz.ITEM_COLS], n_items, balanced=bal)
            rows = {}
            for t in TRAITS:
                s = kt[short[t]].mean(axis=1)
                rest_cols = [c for c in quiz.TRAIT_ITEMS[t] if c not in short[t]]
                rows[t] = {
                    "items": short[t],
                    "r_full": round(r(s, full_t[t]), 3),
                    "r_rest": round(r(s, kt[rest_cols].mean(axis=1)), 3),
                    "alpha_short": round(quiz.cronbach_alpha(kt[short[t]]), 3),
                }
            sf[f"k{n_items}_{'balanced' if bal else 'top'}"] = rows
    # Also: random 4-item forms, to show what choosing the items buys.
    rng = np.random.default_rng(0)
    rand = {t: [] for t in TRAITS}
    for _ in range(50):
        for t in TRAITS:
            pick = list(rng.choice(quiz.TRAIT_ITEMS[t], args.short_k, replace=False))
            rand[t].append(r(kt[pick].mean(axis=1), full_t[t]))
    sf[f"k{args.short_k}_random_mean_of_50"] = {t: {"r_full": round(float(np.mean(v)), 3)} for t, v in rand.items()}
    res["short_form"] = sf
    chosen = quiz.choose_short_form(train[quiz.ITEM_COLS], args.short_k, balanced=True)
    res["short_form_chosen"] = chosen
    for name, rows in sf.items():
        print(f"  {name:28s}", {t: v["r_full"] for t, v in rows.items()})

    # 5. factor check ------------------------------------------------------
    samp = k.sample(50000, random_state=1)
    Z = (samp - samp.mean()) / samp.std()
    eig = np.sort(np.linalg.eigvalsh(np.corrcoef(Z.T.to_numpy())))[::-1]
    fa = FactorAnalysis(n_components=5, rotation="varimax", random_state=0).fit(Z.to_numpy())
    L = pd.DataFrame(fa.components_.T, index=quiz.ITEM_COLS)
    # match factors to traits: maximise the summed loadings of each trait's items
    M = np.array([[L.loc[quiz.TRAIT_ITEMS[t], f].sum() for f in range(5)] for t in TRAITS])
    ti, fi = linear_sum_assignment(-np.abs(M))
    fmap = {TRAITS[a]: int(b) for a, b in zip(ti, fi)}
    for t, f in fmap.items():
        if M[TRAITS.index(t), f] < 0:
            L[f] = -L[f]
    L = L[[fmap[t] for t in TRAITS]]
    L.columns = TRAITS
    own = pd.Series({c: L.loc[c, quiz.KEY[c][0]] for c in quiz.ITEM_COLS})
    top = L.abs().idxmax(axis=1)
    misses = [c for c in quiz.ITEM_COLS if top[c] != quiz.KEY[c][0]]
    res["factor"] = {
        "n_sample": len(samp),
        "eigenvalues_top8": np.round(eig[:8], 2).tolist(),
        "items_loading_highest_on_own_trait": 50 - len(misses),
        "misses": {c: {"own": round(float(own[c]), 3), "top": top[c], "top_loading": round(float(L.loc[c, top[c]]), 3)} for c in misses},
        "min_own_loading": {"item": own.idxmin(), "loading": round(float(own.min()), 3)},
        "loadings": L.round(3).to_dict(orient="index"),
    }
    print(f"factor: eigen {res['factor']['eigenvalues_top8']}, own-trait max for {50 - len(misses)}/50, misses {res['factor']['misses']}")

    # 6. norms -------------------------------------------------------------
    norms = quiz.make_norms(df, chosen, len(df))
    quiz.NORMS.parent.mkdir(parents=True, exist_ok=True)
    quiz.NORMS.write_text(json.dumps(norms, separators=(",", ":")))
    print(f"norms -> {quiz.NORMS} ({quiz.NORMS.stat().st_size / 1024:.1f} KB)")

    # 7. bonus: does response-time behaviour (not answers) predict traits? --
    tf_cols = [
        "log_median_item_ms",
        "log_item_ms_sd",
        "log_item_ms_iqr",
        "frac_items_under_1500ms",
        "log_first_item_ms",
        "log_test_s",
        "log_intro_s",
        "log_end_s",
    ]
    # whole-sample univariate correlations: the "huge N makes everything significant" lesson
    uni = {}
    for c in tf_cols:
        x = df[c].astype(float)
        ok = x.notna()
        uni[c] = {}
        for t in TRAITS:
            rr, p = pearsonr(x[ok], S[t][ok])
            uni[c][t] = {"r": round(float(rr), 4), "p": float(p)}
    res["timing_univariate_full_sample"] = {"n": len(df), "r": uni}
    smp = df.sample(args.sample, random_state=7)
    smp.index = smp.index.astype(str)
    labels = quiz.scales(smp[quiz.ITEM_COLS]).apply(zscore)
    feats = smp[tf_cols].astype(float)
    t1 = time.time()
    out = predict(feats, labels, perms=args.perms, boots=1000, seed=0)
    res["timing_predict"] = {"n": args.sample, "perms": args.perms, "boots": 1000, "features": tf_cols, "scores": out.round(4).to_dict(orient="records")}
    print(f"timing-only predict ({time.time() - t1:.0f}s):\n{out.round(3)}")
    feats2 = feats.join(smp[["screenw", "screenh", "hour_utc"]].astype(float))
    feats2[["screenw", "screenh"]] = feats2[["screenw", "screenh"]].where(feats2[["screenw", "screenh"]] > 0)
    feats2["log_screen_area"] = np.log(feats2.screenw * feats2.screenh)
    feats2["small_screen"] = (feats2.screenw < 800).astype(float).where(feats2.screenw.notna())
    out2 = predict(feats2, labels, perms=args.perms, boots=1000, seed=0)
    res["timing_device_predict"] = {"n": args.sample, "perms": args.perms, "features": list(feats2.columns), "scores": out2.round(4).to_dict(orient="records")}
    print(f"timing+device+hour predict:\n{out2.round(3)}")

    # Same timing-only test inside one country: is the signal just "reading in a
    # second language" (slower + lower openness items like 'rich vocabulary')?
    us = df[df.country == "US"].sample(args.sample, random_state=8)
    us.index = us.index.astype(str)
    out3 = predict(us[tf_cols].astype(float), quiz.scales(us[quiz.ITEM_COLS]).apply(zscore), perms=args.perms, boots=1000, seed=0)
    res["timing_predict_us_only"] = {"n": args.sample, "perms": args.perms, "scores": out3.round(4).to_dict(orient="records")}
    print(f"timing-only, US only:\n{out3.round(3)}")
    nonus = df[df.country != "US"].sample(args.sample, random_state=9)
    nonus.index = nonus.index.astype(str)
    out4 = predict(nonus[tf_cols].astype(float), quiz.scales(nonus[quiz.ITEM_COLS]).apply(zscore), perms=args.perms, boots=1000, seed=0)
    res["timing_predict_non_us"] = {"n": args.sample, "perms": args.perms, "scores": out4.round(4).to_dict(orient="records")}
    print(f"timing-only, non-US:\n{out4.round(3)}")

    res["seconds"] = round(time.time() - t0)
    (DOCS / "quiz_results.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"done in {res['seconds']}s -> {DOCS / 'quiz_results.json'}")


if __name__ == "__main__":
    main()
