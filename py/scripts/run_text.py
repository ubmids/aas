"""Text slice runner: MBTI PersonalityCafe, with and without the type-word leak.

    cd py && uv run python scripts/run_text.py            # full run
    uv run python scripts/run_text.py --posts-per-user 5 --limit 800    # quick smoke test
    uv run python scripts/run_text.py --dataset pandora                 # once PANDORA files land

Writes data/processed/mbti_pc/{persons,features}.parquet (leak-stripped features)
and py/docs/text_results.csv.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from aas import schema
from aas.evaluate import classify
from aas.slices import text as T

DOCS = Path(__file__).resolve().parents[1] / "docs"
CACHE = T.MBTI_RAW / "cache"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def score_block(name: str, variant: str, X: pd.DataFrame, y: pd.DataFrame) -> list[dict]:
    rows = []
    for letter in y.columns:
        t = time.time()
        r = classify(X, y[letter])
        npos = int(round(r["base_rate"] * r["n"]))
        lo, hi = T.auc_ci(r["auc"], npos, r["n"] - npos)
        rows.append({"text": variant, "features": name, "letter": letter, "n": r["n"], "n_features": X.shape[1],
                     "auc": round(r["auc"], 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                     "base_rate": round(r["base_rate"], 4)})
        log(f"  {variant:8s} {name:22s} {letter}: AUC {r['auc']:.3f} [{lo:.3f}, {hi:.3f}]  ({time.time() - t:.0f}s)")
    return rows


def run_mbti(k: int, limit: int | None):
    people = T.load_mbti()
    if limit:
        people = people.sample(limit, random_state=0).reset_index(drop=True)
    log(f"loaded {len(people)} users, {sum(map(len, people.posts))} posts")
    people["clean"] = [[T.strip_leaks(p) for p in ps] for ps in people.posts]
    y = T.letter_targets(people.set_index("person_id")["mbti"])

    allp = [p for ps in people.posts for p in ps]
    changed = sum(T.strip_leaks(p) != p for p in allp)
    leak = pd.DataFrame([T.leak_counts(ps) for ps in people.posts], index=people.person_id)
    residual = pd.DataFrame([T.leak_counts(ps) for ps in people.clean], index=people.person_id)
    says_type = leak.filter(like="leak_says_").sum(axis=1) > 0
    own = [ps_says.get(f"leak_says_{t}", 0) > 0 for ps_says, t in zip(leak.to_dict("records"), people.mbti)]
    stats = {
        "posts": len(allp), "posts_changed_by_stripping": changed,
        "users_naming_any_type": int(says_type.sum()), "users_naming_their_own_type": int(sum(own)),
        "residual_leak_hits_after_stripping": float(residual.drop(columns=["leak_url"]).sum().sum()),
    }
    log("leak stats:", stats)

    log("interpretable features...")
    lang_raw = T.language_table(people, "posts")
    lang_clean = T.language_table(people, "clean")

    log(f"embeddings: {k} posts per user (sampled once, same posts for raw and stripped)...")
    CACHE.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    pick = [T.sample_posts(ps, k, rng) for ps in people.posts]
    raw_chosen = [[ps[i] for i in ix] for ps, ix in zip(people.posts, pick)]
    clean_chosen = [[ps[i] for i in ix] for ps, ix in zip(people.clean, pick)]
    tag = f"k{k}_n{len(people)}"
    cache = {v: CACHE / f"emb_{v}_{tag}.parquet" for v in ("raw", "stripped")}
    if all(f.exists() for f in cache.values()):
        log("  using cached person-level embeddings")
        emb_raw, emb_clean = (pd.read_parquet(cache[v]) for v in ("raw", "stripped"))
    else:
        n_unique = len({t for ts in raw_chosen + clean_chosen for t in ts if t.strip()})
        log(f"  {n_unique} unique post texts to embed")
        t0 = time.time()
        tabs = T.embedding_tables(list(people.person_id), {"raw": raw_chosen, "stripped": clean_chosen}, T.Embedder(), log=log)
        emb_raw, emb_clean = tabs["raw"], tabs["stripped"]
        for v, df in tabs.items():
            df.to_parquet(cache[v])
        log(f"  embedding took {time.time() - t0:.0f}s")

    # Save the honest (leak-stripped) features under the shared contract.
    persons = pd.DataFrame({"person_id": people.person_id, "mbti": people.mbti})
    schema.save("mbti_pc", persons, pd.concat([lang_clean, emb_clean], axis=1))
    log("saved data/processed/mbti_pc")

    rows = []
    rows += score_block("leak tokens only", "raw", leak, y)
    for variant, lang, e in [("raw", lang_raw, emb_raw), ("stripped", lang_clean, emb_clean)]:
        rows += score_block("interpretable", variant, lang, y)
        rows += score_block("embeddings", variant, e, y)
        rows += score_block("both", variant, pd.concat([lang, e], axis=1), y)
    log("reference: bag of words (TF-IDF + logistic), raw / stripped / strict...")
    docs = {"raw": [" ".join(ps) for ps in people.posts], "stripped": [" ".join(ps) for ps in people.clean],
            "strict": [" ".join(T.strip_leaks(p, strict=True) for p in ps) for ps in people.posts]}
    for variant, dd in docs.items():
        for letter in y.columns:
            auc = T.bow_auc(dd, y[letter])
            npos = int(y[letter].sum())
            lo, hi = T.auc_ci(auc, npos, len(y) - npos)
            rows.append({"text": variant, "features": "bag of words (ref)", "letter": letter, "n": len(y), "n_features": 30000,
                         "auc": round(auc, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4), "base_rate": round(y[letter].mean(), 4)})
            log(f"  {variant:8s} bag of words {letter}: AUC {auc:.3f}")
    res = pd.DataFrame(rows)

    # Descriptive only (whole sample, not a prediction score): which plain measures move with each letter.
    corr = pd.DataFrame({c: lang_clean.corrwith(y[c].astype(float)) for c in y.columns})
    return res, stats, corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mbti", choices=["mbti", "pandora"])
    ap.add_argument("--posts-per-user", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None, help="subsample users (smoke test)")
    a = ap.parse_args()
    if a.dataset == "pandora":
        from aas.slices import text_pandora

        sys.exit(text_pandora.main(a.posts_per_user))
    res, stats, corr = run_mbti(a.posts_per_user, a.limit)
    suffix = f"_smoke{a.limit}" if a.limit else ""
    res.to_csv(DOCS / f"text_results{suffix}.csv", index=False)
    corr.round(4).to_csv(DOCS / f"text_lang_correlations{suffix}.csv")
    pd.Series(stats).to_csv(DOCS / f"text_leak_stats{suffix}.csv", header=["value"])
    piv = res.pivot_table(index=["text", "features"], columns="letter", values="auc", sort=False)
    print("\nAUC (0.5 = chance)\n" + piv.round(3).to_string())


if __name__ == "__main__":
    main()
