"""body: LifeSnaps Fitbit wearable data -> Big Five.

Source: Yfantidou et al. 2022, "LifeSnaps, a 4-month multi-modal dataset
capturing unobtrusive snapshots of our lives in the wild", Scientific Data.
71 participants (Greece, Cyprus, Italy, Sweden), Fitbit Sense, May-Sep 2021.
Zenodo record 6832186 (v2) is now access-restricted; we used the open v4,
record 7229547, rais_anonymized.zip (615 MB). We extracted only the CSV
exports (csv_rais_anonymized/, 23 MB), scored_surveys/ and the raw survey
answers mongo_rais_anonymized/surveys.bson (0.6 MB), then deleted the zip.
Not extracted: fitbit.bson (9.7 GB raw minute-level Fitbit, far over the disk
budget; the daily and hourly CSVs are the authors' own aggregates of it).

Labels: the 'bfpt' survey is Goldberg's 50-item IPIP Big-Five Factor Markers
(ipip.ori.org), answers 1..5, items ipip[SQ001]..ipip[SQ050] in the published
order. 50 of the 71 people completed it, so N = 50 here, not 71.
Key (+ forward, - reverse, scored 6 - x):
  E (Extraversion)        1 -6 11 -16 21 -26 31 -36 41 -46
  A (Agreeableness)      -2  7 -12 17 -22 27 -32 37 42 47
  C (Conscientiousness)   3 -8 13 -18 23 -28 33 -38 43 48
  N (Neuroticism)         4 -9 14 -19 24 29 34 39 44 49   (the IPIP scores
      "Emotional Stability", items 4,14,24,29,34,39,44,49 reverse; we flip the
      whole scale so high = neurotic, matching the other datasets)
  O (Intellect/Openness)  5 -10 15 -20 25 -30 35 40 45 50
Trait = mean of answered items, then z-scored within the dataset.
check_against_shipped() confirms our sums equal the authors' scored
personality.csv for every person.

Features: 15 behaviour/physiology measures, one row per person, from that
person's Fitbit rows only (see FEATURES). Self-reported SEMA mood and place
answers are NOT used as features: they are questionnaires, and mood in
particular is close to a neuroticism item, which would make N look predictable
for a reason that has nothing to do with sensing. We only count HOW OFTEN a
person answered (compliance), not WHAT they answered.
"""

from __future__ import annotations

import bson
import numpy as np
import pandas as pd

from .. import schema

DATASET = "lifesnaps"
RAW = schema.RAW / "lifesnaps" / "rais_anonymized"

IPIP = {
    "E": [1, -6, 11, -16, 21, -26, 31, -36, 41, -46],
    "A": [-2, 7, -12, 17, -22, 27, -32, 37, 42, 47],
    "C": [3, -8, 13, -18, 23, -28, 33, -38, 43, 48],
    "N": [4, -9, 14, -19, 24, 29, 34, 39, 44, 49],
    "O": [5, -10, 15, -20, 25, -30, 35, 40, 45, 50],
}
SHIPPED = {"E": "extraversion", "A": "agreeableness", "C": "conscientiousness", "N": "stability", "O": "intellect"}

FEATURES = {
    # activity level
    "log_steps": "E/C: mean daily steps on worn days; active, energetic people move more",
    "active_min": "E/C: mean fairly + very active minutes per day (exercise)",
    "vo2max": "C: Fitbit cardio-fitness estimate, the slow result of sustained exercise",
    # routine
    "steps_cv": "C-: day-to-day coefficient of variation of steps, an irregular week",
    "weekend_ratio": "E/C: weekend steps / weekday steps (going out vs. a weekday routine)",
    "interdaily_stability": "C: actigraphy IS on hourly steps, how alike each day's rhythm is (0..1)",
    "activity_centroid_h": "C-/O: step-weighted mean hour of the day, a chronotype proxy (later = evening type)",
    "night_step_share": "C-/N: share of steps taken 00:00-05:00",
    # sleep
    "sleep_min": "C/N-: mean minutes asleep per night",
    "sleep_sd": "C-/N: night-to-night SD of minutes asleep",
    "sleep_eff": "N-: mean sleep efficiency (asleep / in bed)",
    # autonomic
    "resting_hr": "N/E-: mean resting heart rate (higher with stress and lower fitness)",
    "rmssd": "N-: mean nightly heart-rate variability (RMSSD); low HRV is linked to anxiety",
    # engagement
    "log_wear_days": "C: log of days with the watch worn and syncing (keeping up a study device is a habit)",
    "sema_days": "C/A: days the person answered at least one in-the-moment survey (compliance only)",
}


# ---------------------------------------------------------------- labels

def items() -> pd.DataFrame:
    docs = bson.decode_all((RAW / "mongo_rais_anonymized" / "surveys.bson").read_bytes())
    rows = {}
    for d in docs:
        if d.get("type") != "bfpt":
            continue
        rows[str(d["user_id"])] = {i: d["data"].get(f"ipip[SQ{i:03d}]") for i in range(1, 51)}
    q = pd.DataFrame.from_dict(rows, orient="index").astype(float)
    q.index.name = "user_id"
    return q


def score(q: pd.DataFrame, how: str = "mean") -> pd.DataFrame:
    out = {}
    for t, key in IPIP.items():
        m = pd.concat([6 - q[abs(k)] if k < 0 else q[abs(k)] for k in key], axis=1)
        out[t] = m.sum(1, min_count=len(key)) if how == "sum" else m.mean(1).where(m.notna().sum(1) >= 8)
    return pd.DataFrame(out)


def check_against_shipped(q: pd.DataFrame) -> dict:
    ours = score(q, "sum")
    ours["N"] = 60 - ours["N"]  # back to Emotional Stability (each item flipped: 6-x, 10 items)
    s = pd.read_csv(RAW / "scored_surveys" / "personality.csv").drop_duplicates("user_id").set_index("user_id")
    both = ours.index.intersection(s.index)
    return {t: f"{int((ours.loc[both, t] == s.loc[both, c]).sum())}/{len(both)} match" for t, c in SHIPPED.items()}


# ---------------------------------------------------------------- features

def _interdaily_stability(x: np.ndarray) -> float:
    if len(x) < 7:
        return np.nan
    grand = x.mean()
    denom = ((x - grand) ** 2).mean()
    return float(((x.mean(0) - grand) ** 2).mean() / denom) if denom > 0 else np.nan


def features() -> pd.DataFrame:
    d = pd.read_csv(RAW / "csv_rais_anonymized" / "daily_fitbit_sema_df_unprocessed.csv", parse_dates=["date"])
    h = pd.read_csv(RAW / "csv_rais_anonymized" / "hourly_fitbit_sema_df_unprocessed.csv",
                    usecols=["id", "date", "hour", "steps"], parse_dates=["date"])
    sema_cols = ["ALERT", "HAPPY", "NEUTRAL", "RESTED/RELAXED", "SAD", "TENSE/ANXIOUS", "TIRED",
                 "ENTERTAINMENT", "GYM", "HOME", "HOME_OFFICE", "OTHER", "OUTDOORS", "TRANSIT", "WORK/SCHOOL"]
    rows = {}
    for pid, g in d.groupby("id"):
        worn = g[g["steps"] > 500]  # a day with < 500 steps is almost always a day the watch was off
        wk = worn[worn["date"].dt.dayofweek < 5]["steps"].mean()
        we = worn[worn["date"].dt.dayofweek >= 5]["steps"].mean()
        sl = g["minutesAsleep"][g["minutesAsleep"] > 120]  # drop naps / fragments
        hr = g["rmssd"][g["rmssd"] > 0]
        r = {
            "log_steps": np.log(worn["steps"].mean()) if len(worn) >= 7 else np.nan,
            "active_min": (worn["moderately_active_minutes"] + worn["very_active_minutes"]).mean() if len(worn) >= 7 else np.nan,
            "vo2max": g["filteredDemographicVO2Max"].mean(),
            "steps_cv": worn["steps"].std() / worn["steps"].mean() if len(worn) >= 7 else np.nan,
            "weekend_ratio": we / wk if len(worn) >= 7 and wk > 0 else np.nan,
            "sleep_min": sl.mean() if len(sl) >= 7 else np.nan,
            "sleep_sd": sl.std() if len(sl) >= 7 else np.nan,
            "sleep_eff": g["sleep_efficiency"][g["minutesAsleep"] > 120].mean() if len(sl) >= 7 else np.nan,
            "resting_hr": g["resting_hr"].mean(),
            "rmssd": hr.mean() if len(hr) >= 7 else np.nan,
            "log_wear_days": float(np.log1p(len(worn))),
            "sema_days": float(g[sema_cols].notna().any(axis=1).sum()),
        }
        # hourly rhythm on worn days only; missing hours on a worn day = no steps
        hg = h[(h["id"] == pid) & h["date"].isin(worn["date"])]
        if len(hg):
            prof = hg.pivot_table(index="date", columns="hour", values="steps", aggfunc="sum").reindex(columns=range(24)).fillna(0)
            prof = prof[prof.sum(1) > 0]
            tot = prof.to_numpy().sum()
            hours = prof.sum(0)
            r["interdaily_stability"] = _interdaily_stability(prof.to_numpy(float))
            r["activity_centroid_h"] = float((hours * np.arange(24)).sum() / tot) if tot > 0 else np.nan
            r["night_step_share"] = float(hours.loc[0:4].sum() / tot) if tot > 0 else np.nan
        rows[pid] = r
    out = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=list(FEATURES))
    out.index = DATASET + ":" + out.index.astype(str)
    out.index.name = "person_id"
    return out.astype(float)


def build() -> dict:
    q = items()
    traits = score(q)
    persons = traits.apply(schema.zscore).reset_index(names="uid")
    persons["person_id"] = DATASET + ":" + persons["uid"]
    persons = persons.drop(columns="uid")
    feats = features()
    feats = feats[feats.index.isin(persons["person_id"])]
    schema.save(DATASET, persons, feats)
    return {
        "labelled": len(persons),
        "with_features": len(feats),
        "features": feats.shape[1],
        "missing_cells": int(feats.isna().sum().sum()),
        "ipip_vs_shipped": check_against_shipped(q),
    }
