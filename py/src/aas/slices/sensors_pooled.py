"""sensors_pooled: one harmonised feature set for StudentLife (phone) and
LifeSnaps (Fitbit), and the four ways of asking whether a trait signal holds up
across them: pooled model, cross-dataset transfer, meta-analysis, power.

Pre-registration, feature mappings and their compromises: docs/sensors_pooled.md.

Both datasets are first reduced to the same object, a person's hourly activity
series (local clock time, NaN = hour not observed):
  StudentLife  minutes moving per hour (share of accelerometer-classifier
               samples labelled walking/running, x 60; label 3 'unknown' dropped)
  LifeSnaps    steps per hour (authors' hourly CSV; NaN = watch off / no sync)
and every feature except sleep_h comes out of rhythm_features(), the same code
for both. sleep_h is device-native (phone darkness vs Fitbit sleep) and is the
weakest mapping.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyreadr
from scipy import stats
from sklearn.model_selection import StratifiedKFold

from .. import evaluate, schema
from . import body, phone

DATASETS = ("studentlife", "lifesnaps")

FEATURES = {
    "activity_level": "log(1 + mean daily activity total)",
    "activity_cv": "SD / mean of daily totals",
    "weekend_ratio": "log(weekend daily mean / weekday daily mean)",
    "interdaily_stability": "actigraphy IS on the hourly day x hour matrix, 0..1",
    "relative_amplitude": "(M10 - L5) / (M10 + L5) on the mean 24 h profile",
    "activity_centroid_h": "activity-weighted circular mean clock hour, on a 04..28 axis",
    "rest_midpoint_sd": "night-to-night SD (h) of the least-active-5-hours midpoint, noon-to-noon windows",
    "sleep_h": "device-native mean main-night sleep (phone darkness / Fitbit minutesAsleep)",
}

# Pre-registered pairs (docs/sensors_pooled.md section 1.2): id -> (trait, feature, expected sign)
HYPOTHESES = {
    "H1": ("E", "activity_level", +1),
    "H2": ("E", "weekend_ratio", +1),
    "H3": ("E", "activity_centroid_h", +1),
    "H4": ("C", "interdaily_stability", +1),
    "H5": ("C", "rest_midpoint_sd", -1),
    "H6": ("C", "activity_centroid_h", -1),
    "H7": ("C", "activity_cv", -1),
    "H8": ("C", "activity_level", +1),
    "H9": ("O", "rest_midpoint_sd", +1),
}

MIN_HOURS = 18  # a valid day / night window has at least this many observed hours
MIN_DAYS = 7


# ---------------------------------------------------------------- shared feature code

def _day_matrix(s: pd.Series, shift_h: int = 0) -> pd.DataFrame:
    """Rows = days (or noon-to-noon windows when shift_h=12), cols = 0..23 hour
    offsets, values = activity, NaN = unobserved. Keeps valid days only, fills
    their missing hours with the person's own mean for that column."""
    t = s.index - pd.Timedelta(hours=shift_h)
    m = pd.DataFrame({"d": t.normalize(), "h": t.hour, "v": s.to_numpy()})
    mat = m.pivot_table(index="d", columns="h", values="v", aggfunc="mean").reindex(columns=range(24))
    mat = mat[mat.notna().sum(1) >= MIN_HOURS]
    if len(mat) == 0:
        return mat
    return mat.fillna(mat.mean(0)).fillna(0.0)


def _is(x: np.ndarray) -> float:
    grand = x.mean()
    denom = ((x - grand) ** 2).mean()
    return float(((x.mean(0) - grand) ** 2).mean() / denom) if denom > 0 else np.nan


def _circ_window(p: np.ndarray, w: int) -> np.ndarray:
    """Mean of each w-hour window starting at hour 0..23, wrapping midnight."""
    pp = np.concatenate([p, p[: w - 1]])
    return np.convolve(pp, np.ones(w), "valid")[:24] / w


def _l5_mid(row: np.ndarray) -> float:
    """Midpoint (hours from window start) of the least-active 5 consecutive hours.
    Ties (a long run of zero hours) are common at night, so take the centre of
    the longest contiguous run of minimal windows instead of the first one."""
    w = np.convolve(row, np.ones(5), "valid")  # starts 0..19, no wrap
    lo = w.min()
    hit = np.isclose(w, lo, atol=1e-9 + 1e-6 * abs(lo))
    best, cur, best_s, cur_s = 0, 0, 0, 0
    for i, h in enumerate(hit):
        if h:
            if cur == 0:
                cur_s = i
            cur += 1
            if cur > best:
                best, best_s = cur, cur_s
        else:
            cur = 0
    centre_start = best_s + (best - 1) / 2
    return centre_start + 2.5


def rhythm_features(s: pd.Series) -> dict:
    """s: hourly activity, tz-naive local DatetimeIndex, NaN = unobserved hour."""
    out = {k: np.nan for k in FEATURES if k != "sleep_h"}
    s = s.sort_index()
    mat = _day_matrix(s)
    if len(mat) < MIN_DAYS:
        return out
    x = mat.to_numpy(float)
    tot = x.sum(1)
    out["activity_level"] = float(np.log1p(tot.mean()))
    out["activity_cv"] = float(tot.std(ddof=1) / tot.mean()) if tot.mean() > 0 else np.nan
    dow = pd.DatetimeIndex(mat.index).dayofweek
    we, wk = tot[dow >= 5], tot[dow < 5]
    if len(we) >= 2 and len(wk) >= 2:
        out["weekend_ratio"] = float(np.log((we.mean() + 1) / (wk.mean() + 1)))
    out["interdaily_stability"] = _is(x)
    p = x.mean(0)
    m10, l5 = _circ_window(p, 10).max(), _circ_window(p, 5).min()
    out["relative_amplitude"] = float((m10 - l5) / (m10 + l5)) if m10 + l5 > 0 else np.nan
    if p.sum() > 0:
        ang = 2 * np.pi * (np.arange(24) + 0.5) / 24
        a = np.arctan2((p * np.sin(ang)).sum(), (p * np.cos(ang)).sum())
        h = (a % (2 * np.pi)) * 24 / (2 * np.pi)
        out["activity_centroid_h"] = float(h + 24 if h < 4 else h)
    nights = _day_matrix(s, shift_h=12)
    if len(nights) >= MIN_DAYS:
        mids = np.array([12 + _l5_mid(r) for r in nights.to_numpy(float)])
        out["rest_midpoint_sd"] = float(mids.std(ddof=1))
    return out


# ---------------------------------------------------------------- per-dataset hourly series

def studentlife_hourly() -> dict[str, pd.Series]:
    a = phone._r("sensing/activity")
    a = a[a["activity_inference"].isin([0, 1, 2])]
    t = phone._local(a["timestamp"])
    keep = phone._in_term(t).to_numpy()
    a = a[keep]
    hour = t[keep].dt.floor("h").dt.tz_localize(None)
    df = pd.DataFrame({"uid": a["uid"].astype(str).to_numpy(), "hour": hour.to_numpy(),
                       "moving": a["activity_inference"].isin([1, 2]).to_numpy(float)})
    del a
    g = df.groupby(["uid", "hour"])["moving"].mean() * 60.0
    out = {}
    for uid, su in g.groupby(level=0):
        su = su.droplevel(0)
        full = pd.date_range(su.index.min(), su.index.max(), freq="h")
        out[uid] = su.reindex(full)
    return out


def studentlife_sleep() -> dict[str, float]:
    """Round 1's sleep_h: mean of the longest phone-dark episode starting 20:00-05:00, >= 7 nights."""
    d = phone._r("sensing/dark")
    d["t"] = phone._local(d["start_timestamp"])
    d = d[phone._in_term(d["t"])]
    d["h"] = (d["end_timestamp"] - d["start_timestamp"]) / 3600
    d["onset"] = (d["t"].dt.hour + d["t"].dt.minute / 60 - 18) % 24
    d = d[(d["onset"] >= 2) & (d["onset"] <= 11)]
    d["night"] = (d["t"] - pd.Timedelta(hours=12)).dt.date
    main = d.sort_values("h").groupby(["uid", "night"]).tail(1)
    return {uid: float(mu["h"].mean()) for uid, mu in main.groupby("uid", observed=True) if len(mu) >= MIN_DAYS}


def lifesnaps_hourly() -> dict[str, pd.Series]:
    h = pd.read_csv(body.RAW / "csv_rais_anonymized" / "hourly_fitbit_sema_df_unprocessed.csv",
                    usecols=["id", "date", "hour", "steps"], parse_dates=["date"])
    h = h.dropna(subset=["hour"])
    h["t"] = h["date"] + pd.to_timedelta(h["hour"].astype(int), unit="h")
    h = h.groupby(["id", "t"])["steps"].sum(min_count=1)  # a few duplicate hours (25 rows/day)
    out = {}
    for pid, su in h.groupby(level=0):
        su = su.droplevel(0)
        full = pd.date_range(su.index.min(), su.index.max(), freq="h")
        out[str(pid)] = su.reindex(full)
    return out


def lifesnaps_sleep() -> dict[str, float]:
    d = pd.read_csv(body.RAW / "csv_rais_anonymized" / "daily_fitbit_sema_df_unprocessed.csv",
                    usecols=["id", "minutesAsleep"])
    out = {}
    for pid, g in d.groupby("id"):
        sl = g["minutesAsleep"][g["minutesAsleep"] > 120]
        if len(sl) >= MIN_DAYS:
            out[str(pid)] = float(sl.mean() / 60)
    return out


def harmonised(dataset: str) -> pd.DataFrame:
    hourly, sleep = (studentlife_hourly(), studentlife_sleep()) if dataset == "studentlife" \
        else (lifesnaps_hourly(), lifesnaps_sleep())
    rows = {pid: {**rhythm_features(s), "sleep_h": sleep.get(pid, np.nan)} for pid, s in hourly.items()}
    for pid, v in sleep.items():
        rows.setdefault(pid, {k: np.nan for k in FEATURES})["sleep_h"] = v
    out = pd.DataFrame.from_dict(rows, orient="index")[list(FEATURES)].astype(float)
    out.index = dataset + ":" + out.index.astype(str)
    out.index.name = "person_id"
    return out


def load_both() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Harmonised features + traits for labelled people, both z-scored WITHIN dataset.
    Returns (features with a 'dataset' column, labels)."""
    F, L = [], []
    for ds in DATASETS:
        persons, _ = schema.load(ds)
        f = harmonised(ds)
        f = f[f.index.isin(persons.index) & f.notna().any(axis=1)]
        lab = persons.loc[f.index, schema.TRAITS].apply(schema.zscore)
        f = f.apply(schema.zscore)
        f["dataset"] = ds
        F.append(f)
        L.append(lab)
    return pd.concat(F), pd.concat(L)


# ---------------------------------------------------------------- (a) pooled model

@contextlib.contextmanager
def stratified_folds(strata: np.ndarray):
    """Make evaluate.predict() use folds stratified by `strata` (row-aligned to X),
    so every fold holds people from both datasets. Restores KFold afterwards."""
    orig = evaluate.KFold

    class _Strat:
        def __init__(self, n_splits, shuffle=True, random_state=None):
            self.skf = StratifiedKFold(n_splits, shuffle=True, random_state=random_state)

        def split(self, X):
            assert len(X) == len(strata), "strata not aligned to X"
            return self.skf.split(X, strata)

    evaluate.KFold = _Strat
    try:
        yield
    finally:
        evaluate.KFold = orig


def pooled(F: pd.DataFrame, L: pd.DataFrame, perms: int = 1000, seed: int = 0) -> pd.DataFrame:
    X = F[list(FEATURES)].copy()
    X["is_lifesnaps"] = (F["dataset"] == "lifesnaps").astype(float)  # dataset fixed effect
    assert L.notna().all().all(), "pooled() assumes every person has all five traits"
    strata = F["dataset"].to_numpy()
    with stratified_folds(strata):
        return evaluate.predict(X, L, perms=perms, seed=seed)


def fold_check(F: pd.DataFrame, folds: int = 5, seed: int = 0) -> list[dict]:
    strata = F["dataset"].to_numpy()
    out = []
    for _, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(np.zeros(len(F)), strata):
        out.append(pd.Series(strata[te]).value_counts().to_dict())
    return out


# ---------------------------------------------------------------- (b) transfer

def transfer(F: pd.DataFrame, L: pd.DataFrame, src: str, dst: str, traits=schema.TRAITS,
             perms: int = 10000, boots: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = list(FEATURES)
    s, d = F["dataset"] == src, F["dataset"] == dst
    Xs, Xd = F.loc[s, cols].to_numpy(float), F.loc[d, cols].to_numpy(float)
    rows = []
    for t in traits:
        ys, yd = L.loc[s, t].to_numpy(float), L.loc[d, t].to_numpy(float)
        m = evaluate.ridge().fit(Xs, ys)
        pred = m.predict(Xd)
        r = evaluate._r(pred, yd)
        null = np.array([evaluate._r(pred, rng.permutation(yd)) for _ in range(perms)])
        idx = rng.integers(0, len(yd), (boots, len(yd)))
        bs = np.array([evaluate._r(pred[i], yd[i]) for i in idx])
        rows.append({"train": src, "test": dst, "trait": t, "n_train": len(ys), "n_test": len(yd), "r": r,
                     "ci_low": np.percentile(bs, 2.5), "ci_high": np.percentile(bs, 97.5),
                     "p_perm": (1 + np.sum(null >= r)) / (1 + perms), "null_mean": null.mean(),
                     "alpha": float(m[-1].alpha_), "max_penalty": bool(m[-1].alpha_ >= evaluate.ALPHAS[-1])})
    return pd.DataFrame(rows)


def transfer_weights(F: pd.DataFrame, L: pd.DataFrame, trait: str) -> pd.DataFrame:
    """Standardised ridge coefficients when trained on each dataset alone (what each model leans on)."""
    out = {}
    for ds in DATASETS:
        s = F["dataset"] == ds
        m = evaluate.ridge().fit(F.loc[s, list(FEATURES)].to_numpy(float), L.loc[s, trait].to_numpy(float))
        out[f"coef_{ds}"] = m[-1].coef_
    return pd.DataFrame(out, index=list(FEATURES))


# ---------------------------------------------------------------- (c) meta-analysis

@dataclass
class Meta:
    rho: list
    n: list
    fixed: float
    fixed_lo: float
    fixed_hi: float
    random: float
    random_lo: float
    random_hi: float
    p: float
    Q: float
    I2: float
    tau2: float


def meta(rhos: list[float], ns: list[int]) -> Meta:
    """Fisher-z meta-analysis of Spearman rhos (Bonett-Wright SE), fixed + DerSimonian-Laird."""
    z = np.arctanh(np.asarray(rhos, float))
    n = np.asarray(ns, float)
    v = (1 + np.asarray(rhos, float) ** 2 / 2) / (n - 3)
    w = 1 / v
    zf = (w * z).sum() / w.sum()
    sef = np.sqrt(1 / w.sum())
    Q = float((w * (z - zf) ** 2).sum())
    k = len(z)
    c = w.sum() - (w**2).sum() / w.sum()
    tau2 = max(0.0, (Q - (k - 1)) / c) if c > 0 else 0.0
    wr = 1 / (v + tau2)
    zr = (wr * z).sum() / wr.sum()
    ser = np.sqrt(1 / wr.sum())
    p = 2 * stats.norm.sf(abs(zr / ser))
    I2 = max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0
    t = np.tanh
    return Meta(list(rhos), list(map(int, ns)), t(zf), t(zf - 1.96 * sef), t(zf + 1.96 * sef),
                t(zr), t(zr - 1.96 * ser), t(zr + 1.96 * ser), float(p), Q, I2, tau2)


def meta_table(F: pd.DataFrame, L: pd.DataFrame, pairs: dict) -> pd.DataFrame:
    rows = []
    for hid, (trait, feat, sign) in pairs.items():
        rhos, ns = [], []
        for ds in DATASETS:
            s = (F["dataset"] == ds) & F[feat].notna()
            rho = stats.spearmanr(F.loc[s, feat], L.loc[s, trait]).statistic
            rhos.append(float(rho))
            ns.append(int(s.sum()))
        m = meta(rhos, ns)
        rows.append({"id": hid, "trait": trait, "feature": feat, "expected": "+" if sign > 0 else "-",
                     "rho_SL": rhos[0], "n_SL": ns[0], "rho_LS": rhos[1], "n_LS": ns[1],
                     "fixed": m.fixed, "random": m.random, "ci_low": m.random_lo, "ci_high": m.random_hi,
                     "p": m.p, "I2": m.I2, "tau2": m.tau2,
                     "same_sign_both": bool(np.sign(rhos[0]) == np.sign(rhos[1])),
                     "matches_expected": bool(np.sign(m.random) == sign) if sign else None})
    out = pd.DataFrame(rows)
    out["p_holm"] = evaluate.holm(out["p"].to_numpy())
    return out


# ---------------------------------------------------------------- (d) power

def n_for_power(r: float, power: float = 0.8, alpha: float = 0.05) -> int:
    """People needed to detect correlation r (two-sided), Fisher z approximation."""
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return int(np.ceil(((za + zb) / np.arctanh(abs(r))) ** 2 + 3))


def power_at(r: float, n: int, alpha: float = 0.05) -> float:
    za = stats.norm.ppf(1 - alpha / 2)
    ncp = np.arctanh(abs(r)) * np.sqrt(n - 3)
    return float(stats.norm.sf(za - ncp) + stats.norm.cdf(-za - ncp))
