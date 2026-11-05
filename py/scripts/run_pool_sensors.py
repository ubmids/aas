"""Does the phone/wearable extraversion signal hold up across StudentLife and
LifeSnaps together? Pre-registration and write-up: docs/sensors_pooled.md.

    cd py && uv run python scripts/run_pool_sensors.py

Runs, in the pre-registered order:
  0  label-free harmonisation check (do the 8 features relate to each other alike?)
  a  pooled model (stacked, z within dataset, dataset fixed effect, stratified folds)
  b  transfer StudentLife -> LifeSnaps and back
  c  meta-analysis of the 9 pre-registered pairs, Holm across them
  d  power
then the clearly labelled EXPLORATORY extras. Writes CSVs to docs/sensors_pooled_*.csv.
About 5 minutes.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from aas import schema
from aas.evaluate import predict
from aas.slices import sensors_pooled as sp

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
DOCS = schema.ROOT / "py" / "docs"
PERMS = 1000
COLS = ["trait", "n", "r", "ci_low", "ci_high", "null_mean", "p_perm", "p_holm", "survives", "max_penalty_share"]


def main() -> None:
    F, L = sp.load_both()
    feats = list(sp.FEATURES)
    print("people:", F["dataset"].value_counts().to_dict())
    print("missing per feature:\n", F.groupby("dataset")[feats].apply(lambda g: g.isna().sum()).T)

    # 0. harmonisation check, no labels
    cm = {ds: F.loc[F["dataset"] == ds, feats].corr(method="spearman") for ds in sp.DATASETS}
    iu = np.triu_indices(len(feats), 1)
    a, b = cm["studentlife"].to_numpy()[iu], cm["lifesnaps"].to_numpy()[iu]
    print(f"\n=== 0. feature inter-correlation structure: SL vs LS, r = {np.corrcoef(a, b)[0, 1]:.2f} over {len(a)} pairs ===")
    print(pd.concat({k: v.round(2) for k, v in cm.items()}, axis=1).to_string())

    # a. pooled
    print("\n=== a. pooled model, n = %d, 8 features + dataset dummy, folds stratified by dataset ===" % len(F))
    print("test-fold composition (seed 0):", sp.fold_check(F))
    pool = sp.pooled(F, L, perms=PERMS)
    print(pool[COLS].round(3).to_string(index=False))
    pool.to_csv(DOCS / "sensors_pooled_a_pooled.csv", index=False)

    # a'. context: the same 8 features within each dataset alone (same scorer)
    within = []
    for ds in sp.DATASETS:
        s = F["dataset"] == ds
        w = predict(F.loc[s, feats], L.loc[s], perms=PERMS).assign(dataset=ds)
        print(f"\n--- context: harmonised features within {ds} alone ---")
        print(w[COLS].round(3).to_string(index=False))
        within.append(w)
    pd.concat(within).to_csv(DOCS / "sensors_pooled_a_within.csv", index=False)

    # b. transfer
    tr = pd.concat([sp.transfer(F, L, "studentlife", "lifesnaps"), sp.transfer(F, L, "lifesnaps", "studentlife")])
    print("\n=== b. cross-dataset transfer (10,000 test-label shuffles) ===")
    print(tr.round(3).to_string(index=False))
    tr.to_csv(DOCS / "sensors_pooled_b_transfer.csv", index=False)
    for t in ("E", "C"):
        print(f"\nridge weights for {t} when trained on each dataset alone (z units):")
        print(sp.transfer_weights(F, L, t).round(3).to_string())

    # c. meta-analysis, pre-registered pairs
    mt = sp.meta_table(F, L, sp.HYPOTHESES)
    print("\n=== c. meta-analysis of pre-registered pairs (Spearman -> Fisher z, Holm over 9) ===")
    print(mt.round(3).to_string(index=False))
    mt.to_csv(DOCS / "sensors_pooled_c_meta.csv", index=False)

    # d. power. A model r at or below its own null is not an effect, so it is reported, not powered.
    pe = pool.loc[pool.trait == "E"].iloc[0]
    print("\n=== d. power (Fisher z, two-sided) ===")
    print(f"pooled E model: r = {pe.r:.3f} vs null_mean {pe.null_mean:.3f} -> no positive effect to power")
    e_pairs = mt[mt.trait == "E"]
    best = e_pairs.loc[e_pairs["random"].abs().idxmax()]
    rows = []
    for label, r in [(f"strongest E pair ({best.feature}) pooled rho", float(best["random"])),
                     (f"  same, lower 95% CI bound [added after results]", float(best["ci_low"])),
                     ("round-1 E model r (each dataset)", 0.26), ("Stachl 2020 E r", 0.37)]:
        rows.append({"effect": label, "r": round(r, 3),
                     "n_80_a05": sp.n_for_power(r), "n_80_a01": sp.n_for_power(r, alpha=0.01),
                     "power_n92": round(sp.power_at(r, 92), 2), "power_GLOBEM_497": round(sp.power_at(r, 497), 2),
                     "power_GLOBEM_497_a01": round(sp.power_at(r, 497, 0.01), 2)})
    pw = pd.DataFrame(rows)
    print(pw.to_string(index=False))
    pw.to_csv(DOCS / "sensors_pooled_d_power.csv", index=False)

    # EXPLORATORY (not pre-registered): every feature x every trait, same meta method
    allp = {f"{t}x{f}": (t, f, 0) for t in schema.TRAITS for f in feats}
    ex = sp.meta_table(F, L, allp).drop(columns=["id", "expected", "matches_expected"])
    ex = ex.rename(columns={"p_holm": "p_holm_40"})
    print("\n=== EXPLORATORY: all 40 feature x trait meta-analyses (Holm over 40) ===")
    print(ex.sort_values("p").head(12).round(3).to_string(index=False))
    ex.to_csv(DOCS / "sensors_pooled_x_all_pairs.csv", index=False)

    # EXPLORATORY: chosen AFTER seeing (c). Can the one E pair that showed up predict on its own?
    from scipy import stats
    for ds in sp.DATASETS:
        s = F["dataset"] == ds
        pr = stats.pearsonr(F.loc[s, "activity_centroid_h"], L.loc[s, "E"])
        print(f"EXPLORATORY {ds}: Pearson E x activity_centroid_h r = {pr.statistic:.3f}, p = {pr.pvalue:.3f}")
    X = F[["activity_centroid_h"]].copy()
    X["is_lifesnaps"] = (F["dataset"] == "lifesnaps").astype(float)
    with sp.stratified_folds(F["dataset"].to_numpy()):
        one = predict(X, L, traits=["E"], perms=PERMS)
    print("EXPLORATORY pooled model, activity_centroid_h only -> E:")
    print(one[COLS].round(3).to_string(index=False))
    # stability of the pre-registered pooled model to the CV seed (nothing chosen from this)
    X8 = F[feats].copy()
    X8["is_lifesnaps"] = (F["dataset"] == "lifesnaps").astype(float)
    with sp.stratified_folds(F["dataset"].to_numpy()):
        for seed in (1, 2):
            r = predict(X8, L, traits=["E", "C"], perms=200, seed=seed)
            print(f"seed check {seed}:", r[["trait", "r", "null_mean"]].round(3).to_dict("records"))


if __name__ == "__main__":
    main()
