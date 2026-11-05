"""Run the play (Steam) slice: build, predict Big Five on unseen people, and
show which single behaviours relate to which traits.

    cd py && uv run python scripts/run_play.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from aas import schema
from aas.evaluate import holm, predict
from aas.slices import play

warnings.filterwarnings("ignore")
pd.set_option("display.width", 160)
pd.set_option("display.max_rows", 200)


def show(name: str, feats: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    res = predict(feats, labels)
    print(f"\n=== {name}: {feats.shape[1]} features, N={len(feats)} ===")
    print(res.round(3).to_string(index=False))
    return res.assign(feature_set=name)


def univariate(feats: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f in feats.columns:
        for t in schema.TRAITS:
            y = labels[t].reindex(feats.index)
            ok = feats[f].notna() & y.notna()
            rho, p = spearmanr(feats.loc[ok, f], y[ok])
            rows.append({"feature": f, "trait": t, "n": int(ok.sum()), "rho": rho, "p": p})
    u = pd.DataFrame(rows)
    u["p_holm"] = holm(u["p"].to_numpy())
    return u


def main() -> None:
    log = play.build()
    print("build log:", log)
    labels, feats = schema.load("steam")
    _, c, _ = play.prepare()

    alt = play.make_features(c, "missing")
    raw7 = feats[[f"log_{k}" for k in ["level", "badges", "games", "screenshots", "reviews", "groups", "friends"]]]

    out = [
        show("A main (blank=0, 14 feats)", feats, labels),
        show("B blank=unknown + 6 blank flags (20 feats)", alt, labels),
        show("C raw log counts only (7 feats)", raw7, labels),
    ]
    allres = pd.concat(out)
    print("\nNote: Holm above is within each feature set (5 traits). Across all 15 rows:")
    allres["p_holm_all15"] = holm(allres["p_perm"].to_numpy())
    print(allres[["feature_set", "trait", "r", "p_perm", "p_holm_all15"]].round(3).to_string(index=False))

    u = univariate(feats, labels)
    print(f"\n=== Univariate Spearman, main features, {len(u)} tests, Holm across all ===")
    wide = u.pivot(index="feature", columns="trait", values="rho")[schema.TRAITS]
    star = u.pivot(index="feature", columns="trait", values="p")[schema.TRAITS]
    print("rho (* raw p<.05, ** Holm p<.05):")
    sig = u.pivot(index="feature", columns="trait", values="p_holm")[schema.TRAITS]
    txt = wide.round(2).astype(str) + np.where(sig < 0.05, "**", np.where(star < 0.05, "*", ""))
    print(txt.reindex(feats.columns).to_string())
    print("\nSmallest p values:")
    print(u.sort_values("p").head(10).round(4).to_string(index=False))
    print(f"\nraw p<.05: {(u.p < .05).sum()} of {len(u)} (expected by chance ~{0.05 * len(u):.1f}); Holm survivors: {(u.p_holm < .05).sum()}")

    # Context only (not a feature): how much do the shipped demographics predict?
    df = play.read_raw()
    df.index = "steam:" + df["No."].astype(int).astype(str)
    demo = pd.DataFrame({
        "male": (df["4、gender"].str.strip() == "male").astype(float),
        "age_over_25": (~df["5、age"].str.strip().isin(["18~25", "13~18"])).astype(float),
    }).reindex(feats.index)
    show("D context: demographics only (gender, age>25), NOT behaviour", demo, labels)


if __name__ == "__main__":
    main()
