"""Leave-one-domain-out test of Big Five text recipes. Writes docs/text_crossdomain_results.csv (+ .json).

    cd py && uv run python scripts/text_crossdomain.py

Every dataset with Big Five labels and text is one "domain". For each domain D and each recipe:
train on every other domain, score D's people, AUC against D's own median split (and r against D's
continuous scores where it has them). Nothing from D is used to fit anything (vocabulary, idf, scaling,
hyperparameters). The decision rule is in docs/text_crossdomain.md, written before running this.

Recipes
  a  essays-only: the CURRENT saved artifact (models/text_bigfive), unchanged
  b  pooled bag of words (uni+bi TF-IDF, 20k terms, logistic regression)
  c1 pooled bag of words restricted to domain-stable terms (similar document frequency in every training domain)
  c2 pooled style features (function-word and punctuation rates; no topic words)
  d  pooled MiniLM sentence-embedding means
  d+c2 embeddings and style features side by side
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from aas.readers import text as R
from aas.readers import text_train as TT
from aas.schema import RAW

DOCS = Path(__file__).resolve().parents[1] / "docs"
MODELS = Path(__file__).resolve().parents[1] / "models"
TRAITS = ["O", "C", "E", "A", "N"]
BOW_CFG = {"name": "uni+bi", "ngram_range": [1, 2], "max_features": 20000}
GRIDS = {"b": [0.1, 0.3, 1.0, 3.0], "c1": [0.1, 0.3, 1.0, 3.0], "c2": [0.003, 0.01, 0.03, 0.1], "d": [0.003, 0.01, 0.03, 0.1], "d+c2": [0.003, 0.01, 0.03, 0.1]}
RECIPES = ["a", "b", "c1", "c2", "d", "d+c2"]
SIMPLICITY = ["b", "c1", "c2", "d", "d+c2"]
EMB_CHUNK = 100
EMB_MAX_CHUNKS = 40
log = TT.log


# ---------------------------------------------------------------- domains

def load_domains(names: list[str]) -> dict[str, dict]:
    out = {}
    for n in names:
        d = TT.load_b5_domain(n)
        out[n] = d
        log(f"domain {n}: {len(d['docs'])} people, label {d['label_kind']}, median words {int(np.median([R.n_words(x) for x in d['docs']]))}")
    return out


def embed_domain(name: str, docs: list[str]) -> np.ndarray:
    cache = RAW / "crossdomain_cache" / f"emb_{name}_c{EMB_CHUNK}_m{EMB_MAX_CHUNKS}_n{len(docs)}.npy"
    if cache.exists():
        return np.load(cache)
    from aas.slices.text import Embedder

    emb = Embedder()
    chunks, owner = [], []
    for i, t in enumerate(docs):
        w = t.split()
        pieces = [" ".join(w[s : s + EMB_CHUNK]) for s in range(0, len(w), EMB_CHUNK)] or [""]
        for p in pieces[:EMB_MAX_CHUNKS]:
            chunks.append(p)
            owner.append(i)
    log(f"embedding {len(chunks)} chunks for {name}")
    V = emb.embed(chunks, batch_size=32)
    owner = np.array(owner)
    E = np.stack([V[owner == i].mean(0) for i in range(len(docs))]).astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, E)
    return E


# ---------------------------------------------------------------- one recipe on given train/test people

EQUAL_DOMAIN_WEIGHTS = True


def domain_weights(dom: np.ndarray) -> np.ndarray:
    if not EQUAL_DOMAIN_WEIGHTS:
        return np.ones(len(dom))
    w = np.zeros(len(dom))
    for d in np.unique(dom):
        m = dom == d
        w[m] = 1.0 / m.sum()
    return w * len(dom) / w.sum()


class Fitted:
    """Features fitted on training docs only, then a logistic head per trait."""

    def __init__(self, recipe: str, docs_tr, dom_tr, dense_tr):
        self.recipe = recipe
        if recipe in ("b", "c1"):
            if recipe == "c1":
                terms = TT.domain_stable_terms(docs_tr, dom_tr, BOW_CFG)
                vocab = {t: i for i, t in enumerate(terms)}
                from sklearn.feature_extraction.text import CountVectorizer

                self.vec = CountVectorizer(vocabulary=vocab, ngram_range=(1, 2), dtype=np.float64)
                Cn = self.vec.transform(docs_tr)
                df = np.bincount(Cn.indices, minlength=Cn.shape[1])
                self.idf = np.log((1 + Cn.shape[0]) / (1 + df)) + 1
            else:
                _, self.idf, self.vec = TT.fit_vocab(docs_tr, BOW_CFG, 5)
            self.Xtr = R.tfidf(self.vec.transform(docs_tr), self.idf)
            self.n_features = len(self.idf)
        else:
            self.sc = StandardScaler().fit(dense_tr)
            self.Xtr = self.sc.transform(dense_tr)
            self.n_features = dense_tr.shape[1]

    def transform(self, docs, dense):
        if self.recipe in ("b", "c1"):
            return R.tfidf(self.vec.transform(docs), self.idf)
        return self.sc.transform(dense)


def head_scores(Xtr, ytr, wtr, Xte, C: float) -> np.ndarray:
    m = LogisticRegression(C=C, max_iter=4000).fit(Xtr, ytr, sample_weight=wtr)
    return m.decision_function(Xte)


def run_recipe(recipe: str, train_doms: list[str], test_dom: str, D: dict, dense: dict) -> tuple[dict, dict]:
    """Scores per trait for test_dom's people, and the C chosen per trait (inner selection on train_doms only)."""
    grid = GRIDS[recipe]

    def stack(doms):
        docs = sum((D[d]["docs"] for d in doms), [])
        dom = np.concatenate([[d] * len(D[d]["docs"]) for d in doms])
        Y = pd.concat([D[d]["Yb"] for d in doms], ignore_index=True)
        den = np.vstack([dense[d] for d in doms]) if dense.get(train_doms[0]) is not None else None
        return docs, dom, Y, den

    # inner selection of C per trait: leave-one-training-domain-out if >= 2 train domains, else 5-fold CV
    chosen = {}
    inner = {t: {C: [] for C in grid} for t in TRAITS}
    if len(train_doms) >= 2:
        splits = [([x for x in train_doms if x != h], [h]) for h in train_doms]
        for tr_d, te_d in splits:
            docs, dom, Y, den = stack(tr_d)
            docs_te, _, Yte, den_te = stack(te_d)
            f = Fitted(recipe, docs, dom, den)
            Xte = f.transform(docs_te, den_te)
            w = domain_weights(dom)
            for t in TRAITS:
                for C in grid:
                    inner[t][C].append(roc_auc_score(Yte[t], head_scores(f.Xtr, Y[t].to_numpy(), w, Xte, C)))
    else:
        from sklearn.model_selection import StratifiedKFold

        docs, dom, Y, den = stack(train_doms)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(docs, Y["O"]):
            f = Fitted(recipe, [docs[i] for i in tr], dom[tr], None if den is None else den[tr])
            Xte = f.transform([docs[i] for i in te], None if den is None else den[te])
            for t in TRAITS:
                for C in grid:
                    inner[t][C].append(roc_auc_score(Y[t].to_numpy()[te], head_scores(f.Xtr, Y[t].to_numpy()[tr], None, Xte, C)))
    for t in TRAITS:
        chosen[t] = max(grid, key=lambda C: np.mean(inner[t][C]))
    docs, dom, Y, den = stack(train_doms)
    f = Fitted(recipe, docs, dom, den)
    Xte = f.transform(D[test_dom]["docs"], dense.get(test_dom))
    w = domain_weights(dom)
    out = {t: head_scores(f.Xtr, Y[t].to_numpy(), w, Xte, chosen[t]) for t in TRAITS}
    return out, {"C": chosen, "n_features": int(f.n_features)}


def current_artifact_scores(docs: list[str]) -> dict:
    """Recipe a: the saved Essays model, applied as the reader applies it (no chunking: one score per doc)."""
    import joblib

    m = joblib.load(MODELS / "text_bigfive" / "model.joblib")
    vec = R.build_vectorizer(m["vocab"], m["ngram_range"])
    X = R.tfidf(vec.transform(docs), m["idf"])
    return {t: m["heads"][t].raw(X) for t in TRAITS}


# ---------------------------------------------------------------- scoring with CI and shuffled null

def score_block(s: np.ndarray, yb: np.ndarray, yz: np.ndarray | None, groups: np.ndarray | None, rng, boots=1000, perms=1000) -> dict:
    auc = roc_auc_score(yb, s)
    n = len(yb)
    # bootstrap: resample people, or whole groups (e.g. TV characters) when one person has many samples
    bs_auc, bs_r = [], []
    if groups is not None:
        ug = np.unique(groups)
        members = [np.where(groups == g)[0] for g in ug]
    for _ in range(boots):
        if groups is not None:
            pick = rng.integers(0, len(ug), len(ug))
            i = np.concatenate([members[k] for k in pick])
        else:
            i = rng.integers(0, n, n)
        if 0 < yb[i].mean() < 1:
            bs_auc.append(roc_auc_score(yb[i], s[i]))
        if yz is not None:
            bs_r.append(np.corrcoef(yz[i], s[i])[0, 1])
    null_auc = np.array([roc_auc_score(rng.permutation(yb), s) for _ in range(perms)])
    out = {"n": n, "auc": auc, "auc_lo": np.percentile(bs_auc, 2.5), "auc_hi": np.percentile(bs_auc, 97.5),
           "null_auc_mean": null_auc.mean(), "null_auc_lo": np.percentile(null_auc, 2.5), "null_auc_hi": np.percentile(null_auc, 97.5),
           "p_auc": (1 + np.sum(null_auc >= auc)) / (1 + perms)}
    if yz is not None:
        r = np.corrcoef(yz, s)[0, 1]
        null_r = np.array([np.corrcoef(rng.permutation(yz), s)[0, 1] for _ in range(perms)])
        out |= {"r": r, "r_lo": np.nanpercentile(bs_r, 2.5), "r_hi": np.nanpercentile(bs_r, 97.5), "null_r_mean": null_r.mean(),
                "null_r_lo": np.percentile(null_r, 2.5), "null_r_hi": np.percentile(null_r, 97.5), "p_r": (1 + np.sum(null_r >= r)) / (1 + perms)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default=",".join(TT.B5_DOMAINS))
    ap.add_argument("--recipes", default=",".join(RECIPES))
    ap.add_argument("--out", default=str(DOCS / "text_crossdomain_results"))
    ap.add_argument("--explore", action="store_true",
                    help="POST-HOC diagnostics (not used for the decision): single-domain transfer matrix and pooling without domain weights")
    ap.add_argument("--card-only", action="store_true", help="only copy the saved results into models/text_bigfive/card.json")
    a = ap.parse_args()
    if a.explore:
        return explore(a)
    if a.card_only:
        return update_card(pd.read_csv(a.out + ".csv"))
    names = a.domains.split(",")
    recipes = a.recipes.split(",")
    D = load_domains(names)
    style = {n: np.vstack([TT.style_features(x) for x in D[n]["docs"]]) for n in names}
    emb = {n: embed_domain(n, D[n]["docs"]) for n in names} if any(r.startswith("d") for r in recipes) else {}
    dense_by = {"b": {}, "c1": {}, "c2": style, "d": emb, "d+c2": {n: np.hstack([emb[n], style[n]]) for n in emb}}
    rows, chosen = [], {}
    rng = np.random.default_rng(0)
    for held in names:
        train = [n for n in names if n != held]
        d = D[held]
        for rec in recipes:
            t0 = time.time()
            if rec == "a":
                if held == "essays":
                    continue  # the current model was trained on Essays: not a held-out domain for it
                sc, info = current_artifact_scores(d["docs"]), {"C": "as saved", "n_features": 20000}
            else:
                sc, info = run_recipe(rec, train, held, D, dense_by[rec])
            chosen[f"{held}|{rec}"] = info
            for t in TRAITS:
                yz = d["Yz"][t].to_numpy() if d["Yz"] is not None else None
                res = score_block(sc[t], d["Yb"][t].to_numpy(), yz, d.get("groups"), rng)
                rows.append({"held_out": held, "recipe": rec, "trait": t, "label_kind": d["label_kind"], **res})
            m = np.mean([r["auc"] for r in rows if r["held_out"] == held and r["recipe"] == rec])
            log(f"held out {held:>10} recipe {rec:>5}: mean AUC {m:.3f}  ({time.time() - t0:.0f}s) {info}")
    df = pd.DataFrame(rows)
    df.round(4).to_csv(a.out + ".csv", index=False)
    Path(a.out + ".json").write_text(json.dumps({"chosen": chosen}, indent=1, default=str))
    summ = df.groupby(["recipe", "held_out"]).auc.mean().unstack()
    summ["mean_all"] = summ.mean(axis=1)
    non_es = [c for c in summ.columns if c not in ("essays", "mean_all")]
    summ["mean_non_essays"] = summ[non_es].mean(axis=1)
    print(summ.round(3).to_string())
    if "a" in recipes and set(names) == set(TT.B5_DOMAINS):
        update_card(df)
    return 0


def update_card(df: pd.DataFrame) -> int:
    """Write the current artifact's held-out-domain scores (recipe a) and the recipe comparison into its card."""
    p = MODELS / "text_bigfive" / "card.json"
    card = json.loads(p.read_text())
    a = df[df.recipe == "a"]
    per = {}
    for dom, g in a.groupby("held_out"):
        per[dom] = {"n": int(g.n.iloc[0]), "label_kind": g.label_kind.iloc[0], "traits": {
            r.trait: {k: round(float(getattr(r, k)), 4) for k in ("auc", "auc_lo", "auc_hi", "null_auc_lo", "null_auc_hi", "p_auc", "r", "r_lo", "r_hi")
                      if k in r._fields and pd.notna(getattr(r, k))} for r in g.itertuples()}}
    summ = df.groupby(["recipe", "held_out"]).auc.mean().unstack()
    card["held_out_domains"] = {
        "what": ("The saved model scored on whole datasets it never saw (other kinds of writing). AUC against each dataset's own "
                 "median split; r against its continuous scores where released; 95% CI by bootstrap (Friends: resampling characters); "
                 "null = 95% range of AUC with shuffled labels. Observer-rated domains measure how others see a person, not self-report."),
        "per_domain": per,
        "recipe_comparison_mean_auc": {r: {d: round(float(v), 4) for d, v in row.items() if pd.notna(v)} for r, row in summ.iterrows()},
        "decision": ("Leave-one-domain-out over Essays, FriendsPersona, PAN 2015 Twitter and YouTube vlogs (docs/text_crossdomain.md). "
                     "Best pooled recipe (style features, mean held-out AUC 0.514) did not beat this Essays-only model on the non-Essays "
                     "domains (0.514 vs 0.539), so the model was kept unchanged."),
        "doc": "docs/text_crossdomain.md",
    }
    p.write_text(json.dumps(card, indent=1, default=float))
    log(f"card updated: {p}")
    return 0


def explore(a) -> int:
    """Post-hoc, written after seeing that pooling lost to Essays-only. Two questions:
    1. which single training domain transfers to which test domain (train on ONE domain, test on another)?
    2. does pooling with natural weights (each person counts once, so Essays dominates) behave differently?"""
    global EQUAL_DOMAIN_WEIGHTS
    names = a.domains.split(",")
    D = load_domains(names)
    style = {n: np.vstack([TT.style_features(x) for x in D[n]["docs"]]) for n in names}
    emb = {n: embed_domain(n, D[n]["docs"]) for n in names}
    dense_by = {"b": {}, "c2": style, "d": emb}
    rows = []
    for rec in ["b", "c2", "d"]:
        for tr in names:
            for te in names:
                if tr == te:
                    continue
                sc, _ = run_recipe(rec, [tr], te, D, dense_by[rec])
                aucs = [roc_auc_score(D[te]["Yb"][t], sc[t]) for t in TRAITS]
                rows.append({"kind": "single", "recipe": rec, "train": tr, "test": te, **dict(zip(TRAITS, aucs)), "mean": np.mean(aucs)})
                log(f"{rec} {tr} -> {te}: {np.mean(aucs):.3f}")
    EQUAL_DOMAIN_WEIGHTS = False
    for rec in ["b", "c2"]:
        for te in names:
            sc, _ = run_recipe(rec, [n for n in names if n != te], te, D, dense_by[rec])
            aucs = [roc_auc_score(D[te]["Yb"][t], sc[t]) for t in TRAITS]
            rows.append({"kind": "pooled_natural_weights", "recipe": rec, "train": "all others", "test": te, **dict(zip(TRAITS, aucs)), "mean": np.mean(aucs)})
            log(f"{rec} natural-weight pool -> {te}: {np.mean(aucs):.3f}")
    df = pd.DataFrame(rows)
    df.round(4).to_csv(a.out + "_explore.csv", index=False)
    print(df.round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
