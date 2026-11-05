"""How every claim in AAS gets scored. One function, used by every slice.

predict(features, labels) fits a model on some people and scores it on people
it never saw (cross-validation), so a number here means "works on strangers".

The score is Pearson r between predicted and true trait. Context from the
field: r around 0.3 to 0.4 is what digital footprints usually reach for Big
Five; r = 0.1 is weak; with a few dozen people anything under ~0.3 can be luck.

Two guards against luck:
  permutation p   shuffle the labels, refit, see how often noise scores as well
  bootstrap CI    resample people to show how wide the honest range is
Holm correction is applied across the traits tested together.

Read r against null_mean, not against 0: with no real signal, cross-validation
predicts "the average of the others", which runs backwards for the held-out
person, so pure noise scores below zero (about -0.1 at n~150). A negative r
is therefore never a finding by itself.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .schema import TRAITS

ALPHAS = np.logspace(-2, 4, 25)


def ridge():
    # Imputer and scaler live inside the pipeline, so they only ever see training people.
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=ALPHAS))


def _oof(X: np.ndarray, y: np.ndarray, make_model, folds: int, repeats: int, seed: int, maxed: list | None = None) -> np.ndarray:
    """Out-of-fold predictions, one row per repeat of k-fold (repeats x people).

    Score each repeat on its own and average the scores. Averaging the predictions
    first and then scoring pushes a model that learned nothing towards r = -1,
    because each repeat's "average of the others" bias lines up.

    If `maxed` is given, appends True/False per fold: did ridge pick the
    strongest penalty on the grid? True means the fold's model is close to
    "predict the training average", i.e. it found nothing to learn."""
    preds = np.zeros((repeats, len(y)))
    for r in range(repeats):
        kf = KFold(folds, shuffle=True, random_state=seed + r)
        for tr, te in kf.split(X):
            m = make_model().fit(X[tr], y[tr])
            preds[r, te] = m.predict(X[te])
            if maxed is not None:
                alpha = getattr(m[-1], "alpha_", None) if hasattr(m, "__getitem__") else None
                maxed.append(alpha is not None and alpha >= ALPHAS[-1])
    return preds


def _r(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


@dataclass
class Score:
    trait: str
    n: int
    r: float
    ci_low: float
    ci_high: float
    p_perm: float
    null_mean: float  # what shuffled labels score on average; below 0 when there's no signal (CV bias)
    max_penalty_share: float  # share of folds where ridge gave up (strongest penalty); near 1 = no signal
    p_holm: float = np.nan
    survives: bool = False


def predict(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    traits: list[str] = TRAITS,
    make_model=ridge,
    folds: int = 5,
    repeats: int = 5,
    perms: int = 200,
    boots: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Score features -> each trait on unseen people. Returns one row per trait."""
    rng = np.random.default_rng(seed)
    floor = len(traits) / (1 + perms)  # smallest possible Holm-adjusted p
    if floor >= 0.05:
        warnings.warn(f"perms={perms} with {len(traits)} traits: smallest possible Holm p is {floor:.3f}, nothing can survive; use perms >= {int(20 * len(traits))}")
    rows = []
    for t in traits:
        y_all = labels[t].reindex(features.index)
        keep = y_all.notna()
        X = features[keep].to_numpy(float)
        y = y_all[keep].to_numpy(float)
        n = len(y)
        if n < 15:
            continue
        k = min(folds, n)
        maxed: list = []
        preds = _oof(X, y, make_model, k, repeats, seed, maxed)
        r = float(np.mean([_r(pr, y) for pr in preds]))
        # Permutation: refit on shuffled labels, fewer repeats to keep it affordable.
        null = np.array([_r(_oof(X, p, make_model, k, 1, seed)[0], p) for p in (rng.permutation(y) for _ in range(perms))])
        p_perm = (1 + np.sum(null >= r)) / (1 + perms)
        idx = rng.integers(0, n, (boots, n))
        bs = np.array([np.mean([_r(pr[i], y[i]) for pr in preds]) for i in idx])
        rows.append(Score(t, n, r, *np.percentile(bs, [2.5, 97.5]), p_perm, float(null.mean()), float(np.mean(maxed)) if maxed else np.nan))
    out = pd.DataFrame([asdict(s) for s in rows])
    if len(out):
        out["p_holm"] = holm(out["p_perm"].to_numpy())
        out["survives"] = out["p_holm"] < 0.05
    return out


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for i, j in enumerate(order):
        running = max(running, (m - i) * p[j])
        adj[j] = min(1.0, running)
    return adj


def classify(features: pd.DataFrame, y: pd.Series, folds: int = 5, seed: int = 0, boots: int = 1000) -> dict:
    """For binary labels (e.g. one MBTI letter): out-of-fold AUC vs 0.5."""
    from sklearn.metrics import roc_auc_score

    y = y.reindex(features.index).dropna()
    X = features.loc[y.index].to_numpy(float)
    yy = y.to_numpy().astype(int)
    pred = np.zeros(len(yy))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, yy):
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegressionCV(Cs=10, max_iter=2000))
        m.fit(X[tr], yy[tr])
        pred[te] = m.predict_proba(X[te])[:, 1]
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(boots):
        i = rng.integers(0, len(yy), len(yy))
        if 0 < yy[i].mean() < 1:
            bs.append(roc_auc_score(yy[i], pred[i]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"n": len(yy), "auc": float(roc_auc_score(yy, pred)), "ci_low": float(lo), "ci_high": float(hi), "base_rate": float(yy.mean())}
