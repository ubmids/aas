"""Run the phone (StudentLife) and body (LifeSnaps) slices: build both, predict
Big Five on unseen people with the shared scorer, and show single-feature
associations for context.

    cd py && uv run python scripts/run_phone_body.py

One feature set per dataset, fixed before any model was scored (see the
module docstrings for the two StudentLife features dropped on data-quality
grounds before scoring). Nothing here is tuned on these results.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from aas import schema
from aas.evaluate import holm, predict
from aas.slices import body, phone

warnings.filterwarnings("ignore")
pd.set_option("display.width", 170)
pd.set_option("display.max_rows", 200)


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


def report(dataset: str, log: dict) -> pd.DataFrame:
    print(f"\n######## {dataset} ########")
    print("build:", log)
    labels, feats = schema.load(dataset)
    res = predict(feats, labels)
    print(f"\n=== predict(): {feats.shape[1]} features, N={len(feats)}, 5-fold x5 ridge, 200 perms ===")
    print(res.round(3).to_string(index=False))

    u = univariate(feats, labels)
    wide = u.pivot(index="feature", columns="trait", values="rho")[schema.TRAITS]
    raw = u.pivot(index="feature", columns="trait", values="p")[schema.TRAITS]
    adj = u.pivot(index="feature", columns="trait", values="p_holm")[schema.TRAITS]
    txt = wide.round(2).astype(str) + np.where(adj < 0.05, "**", np.where(raw < 0.05, "*", ""))
    print(f"\n=== univariate Spearman rho, {len(u)} tests (* raw p<.05, ** Holm p<.05 across all) ===")
    print(txt.reindex(feats.columns).to_string())
    print(f"raw p<.05: {(u.p < .05).sum()} of {len(u)} (chance ~{0.05 * len(u):.1f}); Holm survivors: {(u.p_holm < .05).sum()}")
    print(u.sort_values("p").head(6).round(4).to_string(index=False))
    return res.assign(dataset=dataset)


def main() -> None:
    out = [report("studentlife", phone.build()), report("lifesnaps", body.build())]
    allres = pd.concat(out)
    allres["p_holm_all10"] = holm(allres["p_perm"].to_numpy())
    print("\n=== both datasets, Holm across all 10 tests ===")
    print(allres[["dataset", "trait", "n", "r", "ci_low", "ci_high", "p_perm", "p_holm_all10"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
