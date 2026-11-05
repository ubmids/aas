"""Training for the text readers (aas.readers.text). Run via scripts/train_text_readers.py.

One recipe for every text model here:

  1. clean every item with readers.text.clean() (the same leak strip the reader applies)
  2. split people once: 80% "dev", 20% "test" (stratified, seed 0). The test people are not
     looked at until the pipeline is fixed.
  3. on dev only, 5-fold CV picks the pipeline: vectorizer (unigrams vs uni+bigrams, vocab
     capped at 20k) shared by all traits, and the penalty per trait. Sentence embeddings
     (MiniLM, round 1's other family) are scored on the same folds as a challenger.
  4. fit the chosen pipeline on dev; calibrate it on dev out-of-fold scores computed at many
     text lengths (so short texts get honest, less confident probabilities)
  5. score on test: AUC (or r) with bootstrap CI, calibration (ECE + reliability table), and
     the minimum-text curve (test people cut down to k words)
  6. refit everything on all people for the saved artifact. The card reports step 5 numbers,
     which come from a model that never saw the test people.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..schema import RAW, TRAIT_NAMES
from . import text as R

VERSION = "1.0.0"
ESSAYS_CSV = RAW / "essays" / "essays.csv"
ESSAYS_URL = "https://raw.githubusercontent.com/SenticNet/personality-detection/master/essays.csv"
ESSAYS_SHA256 = "3e0ec809f1ca98c6ecdd15b688af4006a28931e226537619fc414238c3f7bc7f"
ESSAY_COLS = {"O": "cOPN", "C": "cCON", "E": "cEXT", "A": "cAGR", "N": "cNEU"}

VEC_CONFIGS = [
    {"name": "uni", "ngram_range": [1, 1], "max_features": 20000},
    {"name": "uni+bi", "ngram_range": [1, 2], "max_features": 20000},
]
CS = [0.1, 0.3, 1.0, 3.0]
ALPHAS = [0.3, 1.0, 3.0, 10.0, 30.0]
EMB_CS = [0.01, 0.03, 0.1, 0.3]
MAX_DROP = 0.03     # recommended minimum: mean AUC within this of full-text AUC
FLOOR_SHARE = 0.5   # floor: keep at least half of the full-text lift above chance


def log(*m):
    print(f"[{time.strftime('%H:%M:%S')}]", *m, flush=True)


# ---------------------------------------------------------------- data

@dataclass
class Corpus:
    name: str
    ids: list[str]
    items: list[list[str]]          # per person, cleaned items (posts, or 20-word essay pieces)
    Y: pd.DataFrame                 # per person, one column per trait (0/1, or z for continuous)
    target: str                     # "binary" | "binary_median" | "continuous_z"
    min_df: int
    k_grid: list[int]
    meta: dict = field(default_factory=dict)

    @property
    def words(self) -> np.ndarray:
        return np.array([sum(R.n_words(x) for x in it) for it in self.items], float)


def load_mbti_corpus() -> Corpus:
    from ..slices.text import letter_targets, load_mbti

    d = load_mbti()
    items = [[c for c in (R.clean(p) for p in ps) if c] for ps in d.posts]
    Y = letter_targets(d["mbti"]).astype(int)
    return Corpus("mbti", list(d.person_id), items, Y, "binary", 20, [25, 50, 100, 150, 200, 300, 400, 600, 800, 1200], {
        "source": "MBTI PersonalityCafe forum (Kaggle datasnaek/mbti-type), via Hugging Face mirror jingjietan/kaggle-mbti",
        "url": "https://huggingface.co/datasets/jingjietan/kaggle-mbti",
        "licence": "mirror tagged Apache-2.0; original Kaggle dataset is CC0 public domain scraped forum text; research use",
        "short": "8,675 PersonalityCafe forum users",
        "unit": "forum posts (the last ~50 per user)",
    })


def load_essays_corpus() -> Corpus:
    import hashlib

    if not ESSAYS_CSV.exists():
        raise FileNotFoundError(f"Essays data missing. Download:\n  mkdir -p {ESSAYS_CSV.parent} && curl -L -o {ESSAYS_CSV} {ESSAYS_URL}")
    sha = hashlib.sha256(ESSAYS_CSV.read_bytes()).hexdigest()
    if sha != ESSAYS_SHA256:
        log(f"WARNING essays.csv sha256 {sha} differs from the file this model card was built on")
    d = pd.read_csv(ESSAYS_CSV, encoding="cp1252", encoding_errors="replace")
    assert len(d) == 2467 and d["#AUTHID"].is_unique, "unexpected essays.csv shape"
    items = [R.units([R.clean(t)], min_units=10**9) for t in d["TEXT"]]  # always 20-word pieces
    Y = pd.DataFrame({t: (d[c].str.strip().str.lower() == "y").astype(int) for t, c in ESSAY_COLS.items()})
    return Corpus("essays", ("essays:" + d["#AUTHID"].astype(str)).tolist(), items, Y, "binary_median", 5,
                  [25, 50, 100, 150, 200, 300, 400, 500], {
        "source": "Pennebaker & King (1999) stream-of-consciousness Essays, University of Texas students 1997-2004, "
                  "Big Five median-split labels (cEXT/cNEU/cAGR/cCON/cOPN = y/n), as distributed with Mairesse et al. (2007) "
                  "and mirrored in the SenticNet personality-detection repo (Majumder et al. 2017)",
        "url": ESSAYS_URL,
        "sha256": ESSAYS_SHA256,
        "licence": "research dataset; no explicit data licence (the mirror repo's MIT licence covers its code). "
                   "Cite Pennebaker & King (1999), J Pers Soc Psychol 77(6):1296. Not for commercial use without checking.",
        "short": "2,467 student essays (Pennebaker & King 1999)",
        "unit": "one ~650-word stream-of-consciousness essay per person, cut into 20-word pieces",
    })


def load_pandora_corpus() -> Corpus:
    """PANDORA Reddit comments with continuous Big Five (z within PANDORA). Needs the files from text_pandora."""
    from ..slices import text_pandora as P

    prof = P.load_profiles()
    comments = P.load_comments(set(prof.author), log=log)
    prof = prof[prof.author.isin(comments)].reset_index(drop=True)
    items = [[c for c in (R.clean(x) for x in comments[a]) if c] for a in prof.author]
    ids = ("pandora:" + pd.util.hash_pandas_object(prof.author, index=False).astype(str)).tolist()
    return Corpus("pandora", ids, items, prof[list(TRAIT_NAMES)].reset_index(drop=True), "continuous_z", 5,
                  [25, 50, 100, 200, 400, 800, 1600, 3200], {
        "source": "PANDORA (Gjurkovic et al. 2021) Reddit comments, personality subreddits dropped",
        "url": "https://psy.takelab.fer.hr/datasets/all/pandora/",
        "licence": "access by request, research use only",
        "short": "PANDORA Reddit authors",
        "unit": "Reddit comments (up to 1,000 per author)",
    })


# ---------------------------------------------------------------- features

def doc(items: list[str]) -> str:
    return "\n".join(items)


def fit_vocab(docs: list[str], cfg: dict, min_df: int) -> tuple[dict, np.ndarray, CountVectorizer]:
    """Vocabulary + idf learned from these documents only. Same idf formula as sklearn (smooth_idf)."""
    v = CountVectorizer(min_df=min_df, max_features=cfg["max_features"], ngram_range=tuple(cfg["ngram_range"]), dtype=np.float64)
    Cn = v.fit_transform(docs)
    df = np.bincount(Cn.indices, minlength=Cn.shape[1])
    idf = np.log((1 + Cn.shape[0]) / (1 + df)) + 1
    vocab = {k: int(i) for k, i in v.vocabulary_.items()}
    if hasattr(v, "stop_words_"):
        del v.stop_words_  # the pruned-term list; not needed and large
    return vocab, idf, R.build_vectorizer(vocab, cfg["ngram_range"])


class ItemBank:
    """All items of a set of people as one sparse count matrix, so any subsample of a person's
    items is a sparse row selection (no re-tokenising)."""

    def __init__(self, corpus: Corpus, people: np.ndarray, vec: CountVectorizer):
        flat, owner = [], []
        for j, p in enumerate(people):
            flat += corpus.items[p]
            owner += [j] * len(corpus.items[p])
        self.C = vec.transform(flat).tocsr()
        self.owner = np.array(owner)
        self.iw = np.array([R.n_words(x) for x in flat], float)
        self.n = len(people)
        self.by_person = np.split(np.arange(len(flat)), np.cumsum(np.bincount(self.owner, minlength=self.n))[:-1])

    def select(self, k: float | None, rng: np.random.Generator):
        """Each person cut down to about k words (random items until >= k). k=None: everything."""
        rows, cols, words = [], [], np.zeros(self.n)
        for j, idx in enumerate(self.by_person):
            if k is None or len(idx) == 0:
                take = idx
            else:
                perm = rng.permutation(idx)
                cw = np.cumsum(self.iw[perm])
                take = perm[: int(np.searchsorted(cw, k)) + 1]
            rows += [j] * len(take)
            cols += list(take)
            words[j] = self.iw[take].sum()
        S = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(self.n, len(self.iw)))
        return S @ self.C, words

    def person_words(self) -> np.ndarray:
        return np.bincount(self.owner, weights=self.iw, minlength=self.n)


def fit_head(X, y, target: str, pen: float):
    if target == "continuous_z":
        m = Ridge(alpha=pen).fit(X, y)
        return m.coef_.ravel(), float(m.intercept_)
    m = LogisticRegression(C=pen, max_iter=3000).fit(X, y)
    return m.coef_.ravel(), float(m.intercept_[0])


def score(y, s, target: str) -> float:
    if target == "continuous_z":
        return float(np.corrcoef(y, s)[0, 1]) if np.std(s) > 0 else 0.0
    return float(roc_auc_score(y, s))


def boot_ci(y, s, target: str, boots: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(boots):
        i = rng.integers(0, len(y), len(y))
        if target != "continuous_z" and not (0 < y[i].mean() < 1):
            continue
        out.append(score(y[i], s[i], target))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi)


def folds_for(y_strat: np.ndarray, target: str, k: int, seed: int):
    if target == "continuous_z":
        return list(KFold(k, shuffle=True, random_state=seed).split(y_strat))
    return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(y_strat, y_strat))


# ---------------------------------------------------------------- calibration

def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> tuple[float, list[dict]]:
    """Expected calibration error: in 10 equal-width bins of predicted probability, the gap between
    the average prediction and the share that were really positive, weighted by bin size."""
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    table, e = [], 0.0
    for i in range(bins):
        m = b == i
        if m.sum() == 0:
            continue
        gap = abs(p[m].mean() - y[m].mean())
        e += m.mean() * gap
        table.append({"bin": f"{edges[i]:.1f}-{edges[i + 1]:.1f}", "n": int(m.sum()), "mean_pred": round(float(p[m].mean()), 3), "observed": round(float(y[m].mean()), 3)})
    return float(e), table


def fit_recal(z: np.ndarray, words: np.ndarray, y: np.ndarray, ref: float, target: str) -> np.ndarray:
    """Length-aware recalibration (a, b, c, d): out = a*z + b*z*lw + c*lw + d, lw = log(words/ref).
    Binary: Platt scaling with a length interaction (logistic). Continuous: least squares."""
    lw = np.log(np.maximum(words, 1) / ref)
    F = np.column_stack([z, z * lw, lw])
    if target == "continuous_z":
        m = LinearRegression().fit(F, y)
        return np.array([*m.coef_, m.intercept_])
    m = LogisticRegression(C=1e4, max_iter=5000).fit(F, y)
    return np.array([*m.coef_.ravel(), m.intercept_[0]])


# ---------------------------------------------------------------- the pipeline

def select_pipeline(c: Corpus, dev: np.ndarray, emb: np.ndarray | None = None, folds: int = 5, seed: int = 0) -> dict:
    """Dev-only CV. Returns the per-candidate table and the choice. Decided before any final fit."""
    traits = list(c.Y.columns)
    Y = c.Y.to_numpy()[dev]
    pens = ALPHAS if c.target == "continuous_z" else CS
    strat = Y[:, 0] if c.target != "continuous_z" else Y[:, 0]
    fl = folds_for(strat.astype(int) if c.target != "continuous_z" else strat, c.target, folds, seed)
    docs = [doc(c.items[p]) for p in dev]
    oof = {(cfg["name"], pen, t): np.zeros(len(dev)) for cfg in VEC_CONFIGS for pen in pens for t in traits}
    for f, (tr, te) in enumerate(fl):
        for cfg in VEC_CONFIGS:
            vocab, idf, vec = fit_vocab([docs[i] for i in tr], cfg, c.min_df)
            Xtr = R.tfidf(vec.transform([docs[i] for i in tr]), idf)
            Xte = R.tfidf(vec.transform([docs[i] for i in te]), idf)
            for ti, t in enumerate(traits):
                for pen in pens:
                    coef, b = fit_head(Xtr, Y[tr, ti], c.target, pen)
                    oof[(cfg["name"], pen, t)][te] = Xte @ coef + b
        log(f"  select fold {f + 1}/{folds} done")
    rows = []
    for (cfg, pen, t), s in oof.items():
        y = Y[:, traits.index(t)]
        rows.append({"family": "bow", "config": cfg, "penalty": pen, "trait": t, "score": score(y, s, c.target), "oof": s})
    if emb is not None:
        E = emb[dev]
        for pen in EMB_CS:
            for ti, t in enumerate(traits):
                s = np.zeros(len(dev))
                for tr, te in fl:
                    if c.target == "continuous_z":
                        m = make_pipeline(StandardScaler(), Ridge(alpha=1 / pen)).fit(E[tr], Y[tr, ti])
                        s[te] = m.predict(E[te])
                    else:
                        m = make_pipeline(StandardScaler(), LogisticRegression(C=pen, max_iter=3000)).fit(E[tr], Y[tr, ti])
                        s[te] = m.decision_function(E[te])
                rows.append({"family": "emb", "config": "minilm-mean", "penalty": pen, "trait": t, "score": score(Y[:, ti], s, c.target), "oof": s})
    tab = pd.DataFrame(rows)
    bow = tab[tab.family == "bow"]
    best_per = bow.groupby(["config", "trait"]).score.max().groupby("config").mean()
    cfg_name = str(best_per.idxmax())
    choice = {"config": next(x for x in VEC_CONFIGS if x["name"] == cfg_name), "penalty": {}, "dev_cv": {}, "emb_best": {}}
    for ti, t in enumerate(traits):
        sub = bow[(bow.config == cfg_name) & (bow.trait == t)].sort_values("score", ascending=False).iloc[0]
        lo, hi = boot_ci(Y[:, ti], sub.oof, c.target)
        choice["penalty"][t] = float(sub.penalty)
        choice["dev_cv"][t] = {"score": round(float(sub.score), 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4)}
        if emb is not None:
            e = tab[(tab.family == "emb") & (tab.trait == t)].score.max()
            choice["emb_best"][t] = round(float(e), 4)
            if e > sub.score + 0.005:
                raise SystemExit(f"embeddings beat bag-of-words on {t} in dev CV ({e:.3f} vs {sub.score:.3f}); "
                                 "the reader only implements bag-of-words, so build the embedding path before shipping")
    choice["table"] = tab.drop(columns="oof").assign(score=lambda d: d.score.round(4)).to_dict("records")
    return choice


def fit_model(c: Corpus, people: np.ndarray, choice: dict, folds: int = 5, seed: int = 0) -> dict:
    """Vocab + heads on these people; recalibration from their own out-of-fold scores at many lengths."""
    traits = list(c.Y.columns)
    Y = c.Y.to_numpy()[people]
    cfg = choice["config"]
    docs = [doc(c.items[p]) for p in people]
    vocab, idf, vec = fit_vocab(docs, cfg, c.min_df)
    bank = ItemBank(c, people, vec)
    X = R.tfidf(bank.select(None, None)[0], idf)
    words_all = bank.person_words()
    ref = float(np.median(words_all))
    heads = {}
    for ti, t in enumerate(traits):
        coef, b = fit_head(X, Y[:, ti], c.target, choice["penalty"][t])
        heads[t] = R.Head(coef, b, np.array([1.0, 0.0, 0.0, 0.0]), float(Y[:, ti].mean()) if c.target != "continuous_z" else None)
    # Out-of-fold scores at several lengths -> recalibration data. Vocab is the (label-free) one above.
    rng = np.random.default_rng(seed + 1)
    lengths = [None] + list(c.k_grid)
    sel = [(k, *bank.select(k, rng)) for k in lengths]
    Xs = [(k, R.tfidf(Cm, idf), w) for k, Cm, w in sel]
    strat = Y[:, 0].astype(int) if c.target != "continuous_z" else Y[:, 0]
    fl = folds_for(strat, c.target, folds, seed)
    for ti, t in enumerate(traits):
        zs, ws, ys = [], [], []
        for tr, te in fl:
            coef, b = fit_head(X[tr], Y[tr, ti], c.target, choice["penalty"][t])
            for k, Xk, wk in Xs:
                keep = te if k is None else te[words_all[te] >= k]
                zs.append(Xk[keep] @ coef + b)
                ws.append(wk[keep])
                ys.append(Y[keep, ti])
        heads[t].cal = fit_recal(np.concatenate(zs), np.concatenate(ws), np.concatenate(ys), ref, c.target)
    return {
        "vocab": vocab, "idf": idf, "ngram_range": list(cfg["ngram_range"]), "heads": heads,
        "ref_words": ref, "cap_words": float(np.percentile(words_all, 90)), "vec": vec,
    }


def apply(m: dict, Cm, words: np.ndarray, t: str, raw: bool = False) -> np.ndarray:
    h = m["heads"][t]
    z = h.raw(R.tfidf(Cm, m["idf"]))
    return z if raw else h.recal(z, words, m["ref_words"])


def r_auc_table(seed: int = 0, n: int = 200_000) -> tuple[np.ndarray, np.ndarray]:
    """AUC that a score with correlation r to a normal trait reaches for 'above the median'.
    Used to turn a tested AUC into the r the Big Five interval needs (binormal assumption)."""
    rng = np.random.default_rng(seed)
    rs = np.linspace(0, 0.95, 96)
    a = rng.standard_normal(n)
    e = rng.standard_normal(n)
    aucs = []
    for r in rs:
        s = r * a + np.sqrt(1 - r * r) * e
        aucs.append(roc_auc_score(a > 0, s))
    return rs, np.array(aucs)


def evaluate_test(c: Corpus, m: dict, test: np.ndarray, seed: int = 0, reps: int = 10) -> dict:
    """Held-out scores, calibration and the minimum-text curve, using a model fit on dev only."""
    traits = list(c.Y.columns)
    Y = c.Y.to_numpy()[test]
    bank = ItemBank(c, test, m["vec"])
    wfull = bank.person_words()
    Cfull, _ = bank.select(None, None)
    out = {"n": int(len(test)), "full": {}, "calibration": {}, "curve": []}
    rs, aucs = r_auc_table()
    full_scores = {}
    for ti, t in enumerate(traits):
        y = Y[:, ti]
        s = apply(m, Cfull, wfull, t)
        full_scores[t] = s
        lo, hi = boot_ci(y, s, c.target)
        out["full"][t] = {"score": round(score(y, s, c.target), 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4)}
        if c.target != "continuous_z":
            p = R.sigmoid(s)
            p_raw = R.sigmoid(apply(m, Cfull, wfull, t, raw=True))
            e_cal, table = ece(y, p)
            e_raw, _ = ece(y, p_raw)
            brier = float(np.mean((p - y) ** 2))
            # What ECE a perfectly calibrated model would still show on this many people (sampling noise).
            nrng = np.random.default_rng(seed)
            floor = float(np.mean([ece(nrng.binomial(1, p).astype(float), p)[0] for _ in range(300)]))
            out["calibration"][t] = {"ece": round(e_cal, 4), "ece_uncalibrated": round(e_raw, 4), "ece_noise_floor": round(floor, 4), "brier": round(brier, 4),
                                     "brier_base_rate": round(float(np.mean((y.mean() - y) ** 2)), 4), "reliability": table}
    rng = np.random.default_rng(seed + 7)
    for k in c.k_grid + [None]:
        keep = np.ones(len(test), bool) if k is None else wfull >= k
        row = {"words": int(np.median(wfull)) if k is None else int(k), "n": int(keep.sum()), "auc": {}, "auc_full_same_people": {}}
        cont = c.target == "continuous_z"
        if cont:
            row = {**row, "r": {}, "r_full_same_people": {}, "resid_q10": {}, "resid_q90": {}}
        eces = {}
        for ti, t in enumerate(traits):
            y = Y[keep, ti]
            sc, ec, res = [], [], []
            for _ in range(1 if k is None else reps):
                Ck, wk = (Cfull, wfull) if k is None else bank.select(k, rng)
                s = apply(m, Ck, wk, t)[keep]
                sc.append(score(y, s, c.target))
                if cont:
                    res.append(y - s)
                else:
                    ec.append(ece(y, R.sigmoid(s))[0])
            full_same = score(y, full_scores[t][keep], c.target)
            if cont:
                row["r"][t] = round(float(np.mean(sc)), 4)
                row["r_full_same_people"][t] = round(full_same, 4)
                rr = np.concatenate(res)
                row["resid_q10"][t] = round(float(np.quantile(rr, 0.1)), 4)
                row["resid_q90"][t] = round(float(np.quantile(rr, 0.9)), 4)
                row["auc"][t] = None
            else:
                row["auc"][t] = round(float(np.mean(sc)), 4)
                row["auc_full_same_people"][t] = round(full_same, 4)
                eces[t] = round(float(np.mean(ec)), 4)
                if c.target == "binary_median":
                    row.setdefault("r_implied", {})[t] = round(float(np.interp(max(np.mean(sc), 0.5), aucs, rs)), 4)
        if eces:
            row["ece"] = eces
        out["curve"].append(row)
        log(f"  k={row['words']}: n={row['n']} " + " ".join(f"{t}={(row['r'] if c.target == 'continuous_z' else row['auc'])[t]:.3f}" for t in traits))
    out["min_text"] = pick_min_text(out["curve"], traits, c.target)
    return out


def pick_min_text(curve: list[dict], traits: list[str], target: str) -> dict:
    """recommended: smallest k (and every k above it) whose mean score is within MAX_DROP of the
    same people's full-text score. floor: smallest k keeping >= FLOOR_SHARE of the full lift above chance."""
    key, fkey, chance = ("r", "r_full_same_people", 0.0) if target == "continuous_z" else ("auc", "auc_full_same_people", 0.5)
    rows = curve[:-1]
    drops = [np.mean([r[fkey][t] - r[key][t] for t in traits]) for r in rows]
    lift = [np.mean([r[key][t] - chance for t in traits]) / np.mean([r[fkey][t] - chance for t in traits]) for r in rows]
    rec = curve[-1]["words"]
    for i in range(len(rows) - 1, -1, -1):
        if drops[i] <= MAX_DROP:
            rec = rows[i]["words"]
        else:
            break
    floor = next((rows[i]["words"] for i in range(len(rows)) if all(l >= FLOOR_SHARE for l in lift[i:])), rec)
    return {"recommended_words": int(rec), "floor_words": int(min(floor, rec)), "rule": (
        f"recommended = smallest length whose mean {key.upper()} is within {MAX_DROP} of the same test people's full-text {key.upper()} "
        f"(and stays so at every longer length); floor = smallest length that keeps at least {FLOOR_SHARE:.0%} of the full-text "
        f"lift above chance ({chance}). Below the floor the reader says nothing."),
        "mean_drop_by_k": {str(r["words"]): round(float(d), 4) for r, d in zip(rows, drops)},
        "share_of_lift_by_k": {str(r["words"]): round(float(x), 3) for r, x in zip(rows, lift)}}


def save(c: Corpus, m: dict, choice: dict, test_eval: dict, out_dir: Path, name: str, extra: dict) -> dict:
    import joblib

    out_dir.mkdir(parents=True, exist_ok=True)
    traits = list(c.Y.columns)
    for h in m["heads"].values():
        h.coef = h.coef.astype(np.float32)
    joblib.dump({"vocab": m["vocab"], "idf": m["idf"].astype(np.float64), "ngram_range": m["ngram_range"], "heads": m["heads"]},
                out_dir / "model.joblib", compress=3)
    card = {
        "reader": name,
        "version": VERSION,
        "trained": date.today().isoformat(),
        "target": c.target,
        "traits": traits,
        "trait_names": {t: TRAIT_NAMES.get(t, t) for t in traits},
        "output_prior": "balanced" if c.target == "binary" else ("median" if c.target == "binary_median" else None),
        "data": {**c.meta, "n_people": len(c.ids), "words_per_person_median": int(np.median(c.words)),
                 "base_rates": {t: round(float(c.Y[t].mean()), 4) for t in traits} if c.target != "continuous_z" else None},
        "features": {
            "recipe": (f"readers.text.clean() on every item (MBTI strict leak strip + Big Five self-report words + URLs), "
                       f"word counts ({'unigrams' if m['ngram_range'] == [1, 1] else 'unigrams + bigrams'}, top {choice['config']['max_features']:,} "
                       f"terms seen in >= {c.min_df} people), sublinear TF-IDF, L2-normalised per person; "
                       f"{'logistic regression' if c.target != 'continuous_z' else 'ridge regression'} per trait; "
                       "length-aware recalibration fitted on out-of-fold scores at many text lengths"),
            "vocab_size": len(m["vocab"]), "ngram_range": m["ngram_range"],
            "ref_words": round(m["ref_words"], 1), "long_text_cap_words": round(m["cap_words"], 1),
            "penalty": choice["penalty"],
            "excluded_kinds": sorted(R.EXCLUDED_KINDS),
        },
        "selection": {"method": "5-fold CV on the 80% dev split only, before any final fit; vectorizer shared, penalty per trait; "
                                "MiniLM sentence-embedding means scored on the same folds as a challenger",
                      "dev_cv": choice["dev_cv"], "emb_best_dev_cv": choice["emb_best"], "chosen_vectorizer": choice["config"]["name"]},
        "test": {k: v for k, v in test_eval.items() if k not in ("curve", "min_text")},
        "min_text": {**test_eval["min_text"], "curve": test_eval["curve"]},
        "interval": ("80% bootstrap range of P over the person's own items" if c.target == "binary" else
                     "80% predictive interval for the person's true z: m +- 1.28*sqrt(1 - r^2 + bootstrap variance), "
                     "r implied by the tested AUC at that length assuming trait and score are jointly normal"
                     if c.target == "binary_median" else "80% predictive interval from held-out residual quantiles + bootstrap"),
        **extra,
    }
    (out_dir / "card.json").write_text(json.dumps(card, indent=1, default=float))
    size = sum(f.stat().st_size for f in out_dir.iterdir())
    log(f"saved {out_dir} ({size / 1e6:.1f} MB)")
    return card


def train(c: Corpus, out_dir: Path, name: str, emb: np.ndarray | None = None, extra: dict | None = None, seed: int = 0):
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(c.ids))
    strat = (c.Y.astype(str).agg("".join, axis=1) if c.target != "continuous_z" else None)
    dev, test = train_test_split(idx, test_size=0.2, random_state=seed, stratify=strat)
    log(f"{c.name}: {len(dev)} dev, {len(test)} test people")
    choice = select_pipeline(c, np.sort(dev), emb, seed=seed)
    log(f"chosen: {choice['config']['name']} penalties {choice['penalty']} dev CV {choice['dev_cv']} (emb best {choice['emb_best']})")
    m_dev = fit_model(c, np.sort(dev), choice, seed=seed)
    log("dev model fit; scoring held-out test people")
    test_eval = evaluate_test(c, m_dev, np.sort(test), seed=seed)
    log(f"test: {test_eval['full']} min text {test_eval['min_text']['recommended_words']}/{test_eval['min_text']['floor_words']}")
    m_all = fit_model(c, idx, choice, seed=seed)
    card = save(c, m_all, choice, test_eval, out_dir, name, {"split": {"dev": int(len(dev)), "test": int(len(test)), "seed": seed}, **(extra or {})})
    return card, m_all


def cross_domain(m: dict, corpus: Corpus, pairs: dict[str, tuple[str, int]]) -> dict:
    """Run a model trained on one dataset on another dataset's people and score it against the loosely
    related label there (MBTI axis <-> Big Five trait, McCrae & Costa 1989). An honest domain-shift probe."""
    people = np.arange(len(corpus.ids))
    bank = ItemBank(corpus, people, m["vec"])
    Cm, w = bank.select(None, None)
    out = {}
    for t_model, (t_label, sign) in pairs.items():
        s = apply(m, Cm, w, t_model) * sign
        y = corpus.Y[t_label].to_numpy()
        lo, hi = boot_ci(y, s, "binary")
        out[f"{t_model} -> {t_label}"] = {"auc": round(score(y, s, "binary"), 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4), "n": int(len(y))}
    return out


# ---------------------------------------------------------------- cross-domain (leave-one-domain-out) helpers
#
# Used by scripts/text_crossdomain.py. Style features are deliberately topic-free: rates of function
# words (the LIWC idea: pronouns, articles, prepositions, auxiliaries, conjunctions, negations,
# quantifiers, hedges) and punctuation, which every genre of English uses, unlike topic words.

FUNCTION_WORDS = {
    "pron_i": "i me my mine myself i'm i've i'd i'll im ive",
    "pron_we": "we us our ours ourselves we're we've we'd we'll",
    "pron_you": "you your yours yourself yourselves you're you've you'd you'll u ur",
    "pron_shehe": "she her hers herself he him his himself she's he's",
    "pron_they": "they them their theirs themselves they're they've they'd they'll",
    "pron_imp": "it its itself it's this that these those something anything nothing everything someone anyone everyone",
    "article": "a an the",
    "prep": "to of in for on with at by from about into over after under between through during without before around against",
    "aux": "is am are was were be been being have has had do does did will would can could should shall may might must",
    "conj": "and but or so because if although though while whereas unless since",
    "negate": "no not never none nobody nothing nowhere neither nor cannot don't doesn't didn't isn't aren't wasn't weren't can't couldn't won't wouldn't shouldn't haven't hasn't",
    "quant": "all some any many much more most few several every each lot lots both half",
    "tentat": "maybe perhaps might possibly probably guess seems seem apparently somewhat unsure suppose kinda sorta",
    "certain": "always definitely certainly absolutely sure clearly obviously completely totally exactly really",
    "adverb": "very just so too also only even still then now here there again already",
    "interrog": "what why how when where who which whom whose",
    "past": "was were had did went got said thought felt",
    "future": "will gonna going tomorrow later soon",
    "social": "friend friends people family mom dad talk said tell told",
    "posemo": "love good great happy nice fun like glad best enjoy",
    "negemo": "hate bad sad angry mad worried afraid hurt upset annoying",
    "swear": "fuck fucking shit damn hell crap ass bitch",
}
_FW_SETS = {k: set(v.split()) for k, v in FUNCTION_WORDS.items()}
_ALL_FW = sorted(set().union(*_FW_SETS.values()))
_WORD_RE = None


def style_features(text: str) -> np.ndarray:
    """Per-document topic-free style vector: category rates, rates of each single function word,
    punctuation per word, word length, vocabulary variety. Same length for every input."""
    import re
    from collections import Counter

    global _WORD_RE
    if _WORD_RE is None:
        _WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
    t = text.lower().replace("’", "'")
    w = _WORD_RE.findall(t)
    n = max(len(w), 1)
    c = Counter(w)
    cats = [sum(c[x] for x in s) / n for s in _FW_SETS.values()]
    each = [c[x] / n for x in _ALL_FW]
    punct = [t.count(ch) / n for ch in ".,!?;:'\"()-"]
    chunks = [w[i : i + 50] for i in range(0, len(w) - 49, 50)]
    ttr = float(np.mean([len(set(x)) / 50 for x in chunks])) if chunks else len(set(w)) / n
    other = [sum(len(x) for x in w) / n, ttr, sum(1 for x in w if len(x) >= 7) / n, sum(1 for ch in text if ch.isupper()) / max(len(text), 1)]
    return np.array(cats + each + punct + other, float)


def style_names() -> list[str]:
    return ([f"cat_{k}" for k in FUNCTION_WORDS] + [f"fw_{x}" for x in _ALL_FW] + [f"punct_{ch}" for ch in ".,!?;:'\"()-"]
            + ["word_len", "ttr50", "long_words", "caps"])


def domain_stable_terms(docs: list[str], domains: np.ndarray, cfg: dict, min_share: float = 0.005, max_ratio: float = 4.0) -> list[str]:
    """Terms whose document frequency is similar in every training domain: present in at least
    min_share of documents in EACH domain, and no domain uses it more than max_ratio times as often
    (by share of documents) as another. Removes topic and genre words ('roommate', 'monica',
    'subscribe') and keeps the vocabulary every kind of writing shares."""
    v = CountVectorizer(ngram_range=tuple(cfg["ngram_range"]), min_df=2, dtype=np.float64, binary=True)
    B = v.fit_transform(docs).tocsc()
    names = np.array(v.get_feature_names_out())
    shares = []
    for d in np.unique(domains):
        m = domains == d
        shares.append(np.asarray(B[m].mean(0)).ravel())
    S = np.vstack(shares)
    lo, hi = S.min(0), S.max(0)
    keep = (lo >= min_share) & (hi <= max_ratio * np.maximum(lo, 1e-12))
    return names[keep].tolist()


# ---------------------------------------------------------------- Big Five domains for leave-one-domain-out
#
# Each loader returns {"docs": one cleaned document per person/sample, "Yb": 0/1 above the dataset's own
# median per trait, "Yz": z within dataset (or None if the dataset only released yes/no), "label_kind",
# "groups": cluster ids when one person contributes several samples (else None), "meta": provenance}.

FRIENDS_CSV = RAW / "friendspersona" / "friends-personality.csv"
FRIENDS_URL = "https://raw.githubusercontent.com/emorynlp/personality-detection/master/CSV/friends-personality.csv"
PAN15_DIR = RAW / "pan15"
PAN15_URL = "https://zenodo.org/records/3745945"
VLOGS_DIR = RAW / "youtube_vlogs"
VLOGS_URL = "https://huggingface.co/datasets/facells/youtube-vlog-personality-recognition-wcpr14"
B5_DOMAINS = ["essays", "friends", "pan15", "vlogs"]
# Domains whose terms allow research evaluation but not redistribution of derived models: used in the
# leave-one-domain-out study, never in a saved artifact. (vlogs: CC BY-NC-ND 4.0.)
EVAL_ONLY_DOMAINS = {"vlogs"}


def median_split(z: pd.DataFrame) -> pd.DataFrame:
    """1 if above the dataset's own median. Ties at the median go to 0 (as in the Essays release)."""
    return pd.DataFrame({t: (z[t] > z[t].median()).astype(int) for t in z.columns})


def _essays_domain() -> dict:
    c = load_essays_corpus()
    return {"docs": [doc(it) for it in c.items], "Yb": c.Y.reset_index(drop=True), "Yz": None, "groups": None,
            "label_kind": "self-report questionnaire, yes/no median split (released that way)", "meta": c.meta}


def _friends_domain() -> dict:
    """FriendsPersona (Jiang, Zhang & Choi, AAAI 2020): 711 scenes from Friends seasons 1-4; for each scene,
    3 annotators rated the main speaker's Big Five from the dialogue (-1/0/1), summed and median-split by the
    authors (only the yes/no labels are released). We keep ONLY the main speaker's own lines (their 'writing'),
    with stage directions removed. Labels are OBSERVER impressions of a fictional character in one scene,
    not self-report. Samples cluster by character, so CIs resample characters."""
    import re

    if not FRIENDS_CSV.exists():
        raise FileNotFoundError(f"mkdir -p {FRIENDS_CSV.parent} && curl -L -o {FRIENDS_CSV} {FRIENDS_URL}")
    d = pd.read_csv(FRIENDS_CSV)
    docs, groups = [], []
    for text, ch in zip(d.text, d.character):
        who = re.match(r"<b>s\S+\(\d+\) for ([^<]+)</b>", text).group(1).strip()
        utts = re.findall(r"<b>([^<]+)</b>: ?(.*?)(?=<br>|$)", text)
        own = " ".join(u for s, u in utts if s.strip() == who)
        own = re.sub(r"\([^)]*\)", " ", own)  # stage directions are the script writers', not the speaker's
        docs.append(R.clean(own))
        groups.append(ch.split()[0].lower())
    Yb = pd.DataFrame({t: d[f"c{k}"].astype(int) for t, k in {"O": "OPN", "C": "CON", "E": "EXT", "A": "AGR", "N": "NEU"}.items()})
    # cNEU is flipped: the annotation form asked "Emotionally Stable" (README), and with labels alone (no
    # model) cNEU correlates +0.17 with A and +0.15 with C, the pattern of stability, where Essays' cNEU is
    # -0.19 / -0.15. Decided before any model was scored on Friends.
    Yb["N"] = 1 - Yb["N"]
    return {"docs": docs, "Yb": Yb, "Yz": None, "groups": np.array(groups),
            "label_kind": "observer-rated (3 annotators, from the scene's dialogue), yes/no median split (released that way)",
            "meta": {"source": "FriendsPersona (Jiang, Zhang & Choi, AAAI 2020), Friends TV transcripts seasons 1-4", "url": FRIENDS_URL,
                     "short": "711 Friends scenes (FriendsPersona)", "licence": "see docs/text_datasets.md"}}


def _pan15_domain() -> dict:
    """PAN 2015 Author Profiling, English (Rangel et al. 2015): 294 Twitter users (152 training + 142 test
    release, pooled here), ~100 tweets each, self-reported BFI-10 scores in [-0.5, 0.5]. Retweets and
    quote-tweets are dropped (not the user's words); @mentions were already anonymised to @username and are removed."""
    import re
    import xml.etree.ElementTree as ET

    dirs = sorted(p for p in PAN15_DIR.glob("*/*english*") if p.is_dir())
    if len(dirs) != 2:
        raise FileNotFoundError(f"PAN15 English train+test folders missing under {PAN15_DIR}; download from {PAN15_URL}")
    rows, docs = [], []
    for dd in dirs:
        for line in (dd / "truth.txt").read_text().splitlines():
            uid, _g, _a, ext, stab, agr, con, opn = line.split(":::")
            tw = [x.text or "" for x in ET.parse(dd / f"{uid}.xml").getroot().iter("document")]
            # drop retweets and quote-tweets ("RT @x", "MT @x", '"@x: ...') : someone else's words
            tw = [re.sub(r"@username", " ", t) for t in tw if not re.search(r"^\s*(?:RT|MT)\b|@username:", t.strip())]
            docs.append("\n".join(c for c in (R.clean(t) for t in tw) if c))
            rows.append({"id": uid, "O": float(opn), "C": float(con), "E": float(ext), "A": float(agr), "N": -float(stab)})
    raw = pd.DataFrame(rows)
    assert raw.id.is_unique
    Yz = raw[list("OCEAN")].apply(lambda c: (c - c.mean()) / c.std(ddof=0))
    return {"docs": docs, "Yb": median_split(Yz), "Yz": Yz, "groups": None,
            "label_kind": "self-report BFI-10 (continuous; N = -stable)",
            "meta": {"source": "PAN 2015 Author Profiling shared task, English Twitter (Rangel et al. 2015)", "url": PAN15_URL,
                     "short": "294 Twitter users (PAN 2015)", "licence": "see docs/text_datasets.md"}}


def _vlogs_domain() -> dict:
    """YouTube vlogs (Biel & Gatica-Perez 2013), as released for WCPR 2014: 404 vloggers, transcript of one
    vlog each, Big Five IMPRESSIONS (1-7) from MTurk raters who WATCHED a minute of the video (so they also saw
    face, voice, setting). Observer-rated, continuous. N = -stability."""
    f = [VLOGS_DIR / f"vlogs-wcpr14-{s}.csv" for s in ("training", "test")]
    if not all(x.exists() for x in f):
        raise FileNotFoundError(f"vlog CSVs missing in {VLOGS_DIR}; download from {VLOGS_URL}")
    d = pd.concat([pd.read_csv(x) for x in f], ignore_index=True)
    assert d.vlogId.is_unique and len(d) == 404
    raw = pd.DataFrame({"O": d.openness, "C": d.conscientiousness, "E": d.extraversion, "A": d.agreableness, "N": -d.stability})
    Yz = raw.apply(lambda c: (c - c.mean()) / c.std(ddof=0))
    return {"docs": [R.clean(str(t)) for t in d.transcript], "Yb": median_split(Yz), "Yz": Yz, "groups": None,
            "label_kind": "observer-rated impressions from watching the vlog (MTurk, continuous 1-7; N = -stability)",
            "meta": {"source": "YouTube Personality vlogs (Biel & Gatica-Perez 2013), WCPR 2014 release", "url": VLOGS_URL,
                     "short": "404 YouTube vloggers", "licence": "CC BY-NC-ND 4.0 (Hugging Face re-host); evaluation only"}}


def load_b5_domain(name: str) -> dict:
    return {"essays": _essays_domain, "friends": _friends_domain, "pan15": _pan15_domain, "vlogs": _vlogs_domain}[name]()


def load_pooled_corpus(domains: list[str] | None = None) -> Corpus:
    """All shippable Big Five domains as one corpus for the saved artifact. Labels: each dataset's own median
    split (the one label every dataset has). Each person's text is cut into 20-word pieces, like Essays."""
    domains = domains or [d for d in B5_DOMAINS if d not in EVAL_ONLY_DOMAINS]
    ids, items, Ys, metas, dom = [], [], [], {}, []
    for n in domains:
        d = load_b5_domain(n)
        items += [R.units([x], min_units=10**9) for x in d["docs"]]
        ids += [f"{n}:{i}" for i in range(len(d["docs"]))]
        Ys.append(d["Yb"])
        dom += [n] * len(d["docs"])
        metas[n] = {**d["meta"], "label_kind": d["label_kind"], "n": len(d["docs"])}
    Y = pd.concat(Ys, ignore_index=True)
    short = " + ".join(f"{m['n']:,} {k}" for k, m in metas.items())
    c = Corpus("pooled", ids, items, Y, "binary_median", 5, [25, 50, 100, 150, 200, 300, 400, 500], {
        "source": "pooled: " + "; ".join(m["source"] for m in metas.values()),
        "short": f"{len(ids):,} people from {len(metas)} kinds of writing ({short})",
        "unit": "each person's own text cut into 20-word pieces",
        "domains": metas,
        "licence": "research use only; see docs/text_datasets.md (Essays and PAN 2015 have no explicit licence; FriendsPersona Apache-2.0 project, Friends script text)",
    })
    c.domain = np.array(dom)
    return c
