"""phone: StudentLife smartphone sensing -> Big Five.

Source: Wang et al. 2014, "StudentLife: assessing mental health, academic
performance and behavioral trends of college students using smartphones"
(UbiComp). 48 Dartmouth students (uids 0..59 with gaps), spring term 2013
(~10 weeks), Android phones. The official server does not answer from here, so
we use the Zenodo RData/Rds mirror made for the R package 'studentlife'
(Fry & ..., github.com/frycast/studentlife): Zenodo record 3529253,
dataset_rds.zip (230 MB). We extracted survey/, EMA/, other/ and eight sensing
files (87 MB) and deleted the zip. Not extracted: sensing/audio.Rds (266 MB,
raw audio-class inferences; conversation.Rds already summarises it) and
sensing/wifi.Rds (51 MB raw scans; wifi_location.Rds is the building-level
version).

Labels: BigFive.Rds holds the BFI-44 (John & Srivastava 1999) as text answers,
items Q1..Q44 in the published order, one row per (uid, pre|post). We use the
PRE survey (taken before most sensing, so the label cannot be shaped by the
term). Answers map Disagree strongly=1 .. Agree strongly=5. Key:
  E: 1, 6R, 11, 16, 21R, 26, 31R, 36
  A: 2R, 7, 12R, 17, 22, 27R, 32, 37R, 42
  C: 3, 8R, 13, 18R, 23R, 28, 33, 38, 43R
  N: 4, 9R, 14, 19, 24R, 29, 34R, 39
  O: 5, 10, 15, 20, 25, 30, 35R, 40, 41R, 44
Reverse-keyed items are scored 6 - x. Trait = mean of answered items (a person
needs >= 75% of a trait's items), then z-scored within the dataset.
check_bfi() verifies the key on this data: every reverse-keyed item must
correlate negatively with its trait's forward items.

Features: 14 behaviour measures, one row per student, from that student's
sensor streams only (see FEATURES for the hypothesis behind each).
Dropped on data-quality grounds BEFORE any model was scored:
  calls per day   other/call_log has zero calls for 29 of 49 students
                  (logging failed, not "never phoned"); SMS has no message
                  times at all, only upload times.
  app usage       other/app_usage is a fixed 20-minute poll of the recent-tasks
                  list, not foreground use: every student's night share came
                  out 0.25 (= 6/24 hours). Replaced by night_locked_frac.
phonelock and dark only record stretches of 1 hour or more, so unlock counts
are not recoverable from this mirror. Times are
converted to US/Eastern (Dartmouth) before any time-of-day logic. "Per day"
means divided by the number of days the always-on activity sensor produced
data for that student, so a phone that was off for a week is not read as
"stayed home and talked to nobody".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyreadr

from .. import schema

DATASET = "studentlife"
RAW = schema.RAW / "studentlife" / "dataset_rds"
TZ = "US/Eastern"
TERM = (pd.Timestamp("2013-03-24", tz=TZ), pd.Timestamp("2013-06-06", tz=TZ))

LIKERT = {
    "disagree strongly": 1,
    "disagree a little": 2,
    "neither agree nor disagree": 3,
    "agree a little": 4,
    "agree strongly": 5,
}
BFI = {
    "E": [1, -6, 11, 16, -21, 26, -31, 36],
    "A": [-2, 7, -12, 17, 22, -27, 32, -37, 42],
    "C": [3, -8, 13, -18, -23, 28, 33, 38, -43],
    "N": [4, -9, 14, 19, -24, 29, -34, 39],
    "O": [5, 10, 15, 20, 25, 30, -35, 40, -41, 44],
}

FEATURES = {
    # mobility
    "log_gyration_km": "E/O: daily radius of gyration (median day), how far you range; outgoing, novelty-seeking people roam",
    "places_per_day": "E/O: distinct campus buildings visited per day (WiFi), variety of settings",
    "home_frac_day": "E- : share of daytime (9-21h) WiFi fixes in your own dorm (the building you sleep in)",
    # sociability
    "conv_min_per_day": "E: minutes per day the microphone classifier heard you in conversation",
    "conv_n_per_day": "E: number of separate conversations per day",
    "log_bt_devices_per_day": "E: distinct Bluetooth devices nearby per day, a co-presence proxy (people around you)",
    # activity and rhythm
    "active_frac": "E/C: share of the day the accelerometer says walking or running",
    "interdaily_stability": "C: actigraphy IS score, how much each day's hourly activity pattern matches your average day (0..1)",
    "sleep_h": "C/N: hours of the main night-time phone-dark episode (a sleep proxy)",
    "sleep_onset_h": "C-: when that episode starts, hours after 18:00 (later = later bedtime)",
    "sleep_onset_sd": "C-/N: night-to-night spread of that start time, irregular bedtimes",
    # phone use
    "locked_h_per_day": "C/N: hours per day the phone sat locked for 1h+ stretches (less = more on the phone)",
    "night_locked_frac": "C/N-: share of 00:00-06:00 the phone sat in a 1h+ locked stretch (low = phone in use at night)",
    # engagement
    "data_days": "C: days the phone delivered sensor data (keeping the study app running is itself a habit)",
}


def _r(name: str) -> pd.DataFrame:
    df = next(iter(pyreadr.read_r(str(RAW / f"{name}.Rds")).values()))
    df["uid"] = df["uid"].astype(str)
    return df


def _local(ts: pd.Series, unit: str = "s") -> pd.Series:
    return pd.to_datetime(ts, unit=unit, utc=True).dt.tz_convert(TZ)


def _in_term(t: pd.Series) -> pd.Series:
    return (t >= TERM[0]) & (t < TERM[1])


# ---------------------------------------------------------------- labels

def items() -> pd.DataFrame:
    b = _r("survey/BigFive")
    b = b[b["type"].astype(str) == "pre"].set_index("uid")
    q = b[[f"Q{i}" for i in range(1, 45)]]
    return q.apply(lambda c: c.astype(str).str.strip().str.lower().map(LIKERT)).astype(float)


def score(q: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for t, key in BFI.items():
        cols = []
        for k in key:
            x = q[f"Q{abs(k)}"]
            cols.append(6 - x if k < 0 else x)
        m = pd.concat(cols, axis=1)
        enough = m.notna().sum(1) >= 0.75 * len(key)
        out[t] = m.mean(1).where(enough)
    return pd.DataFrame(out)


def check_bfi(q: pd.DataFrame) -> dict:
    """Corrected item-total correlation sign check + Cronbach alpha per trait."""
    report = {}
    for t, key in BFI.items():
        fwd = [f"Q{k}" for k in key if k > 0]
        rev = [f"Q{-k}" for k in key if k < 0]
        fwd_mean = q[fwd].mean(1)
        rev_r = {c: round(q[c].corr(fwd_mean), 2) for c in rev}
        keyed = pd.concat([q[f"Q{abs(k)}"] * (1 if k > 0 else -1) for k in key], axis=1).dropna()
        k = keyed.shape[1]
        alpha = k / (k - 1) * (1 - keyed.var(ddof=1).sum() / keyed.sum(1).var(ddof=1))
        report[t] = {"alpha": round(float(alpha), 2), "reverse_item_r": rev_r,
                     "all_reverse_negative": all(v < 0 for v in rev_r.values())}
    return report


# ---------------------------------------------------------------- features

def _interdaily_stability(hourly: pd.DataFrame) -> float:
    """hourly: rows = days, cols = 24 hours, values = activity level."""
    x = hourly.to_numpy(float)
    if len(x) < 7:
        return np.nan
    grand = x.mean()
    denom = ((x - grand) ** 2).mean()
    if denom == 0:
        return np.nan
    return float(((x.mean(0) - grand) ** 2).mean() / denom)


def _gyration_km(g: pd.DataFrame) -> float:
    lat, lon = np.radians(g["latitude"].to_numpy()), np.radians(g["longitude"].to_numpy())
    if len(lat) < 3:
        return np.nan
    clat, clon = lat.mean(), lon.mean()
    dx = (lon - clon) * np.cos(clat) * 6371.0
    dy = (lat - clat) * 6371.0
    return float(np.sqrt(np.mean(dx**2 + dy**2)))


def features() -> pd.DataFrame:
    f: dict[str, dict] = {}

    # activity: the always-on sensor; defines observed days
    a = _r("sensing/activity")
    a["t"] = _local(a["timestamp"])
    a = a[_in_term(a["t"]) & a["activity_inference"].isin([0, 1, 2])]
    a["day"] = a["t"].dt.date
    a["hour"] = a["t"].dt.hour
    a["moving"] = a["activity_inference"].isin([1, 2]).astype(float)
    days = a.groupby("uid")["day"].nunique()
    for uid, g in a.groupby("uid"):
        prof = g.groupby(["day", "hour"])["moving"].mean().unstack().reindex(columns=range(24))
        prof = prof[prof.notna().sum(1) >= 18].fillna(prof.mean())  # keep days with 18+ hours of data
        f.setdefault(uid, {}).update({
            "active_frac": g["moving"].mean(),
            "interdaily_stability": _interdaily_stability(prof),
            "data_days": float(days[uid]),
        })
    del a

    def per_day(uid: str, v: float) -> float:
        d = days.get(uid, np.nan)
        return v / d if d and d > 0 else np.nan

    # GPS: daily radius of gyration, median over days with 5+ fixes
    g = _r("sensing/gps")
    g["t"] = _local(g["timestamp"])
    g = g[_in_term(g["t"]) & (g["accuracy"] < 200)]
    g["day"] = g["t"].dt.date
    for uid, gu in g.groupby("uid"):
        daily = [_gyration_km(x) for _, x in gu.groupby("day") if len(x) >= 5]
        daily = [d for d in daily if np.isfinite(d)]
        if daily:
            f.setdefault(uid, {})["log_gyration_km"] = np.log1p(np.median(daily))

    # WiFi building fixes: places per day and time at home
    w = _r("sensing/wifi_location")
    w = w[w["location"].str.startswith("in[")]
    w["t"] = _local(w["timestamp"])
    w = w[_in_term(w["t"])]
    w["day"] = w["t"].dt.date
    w["hour"] = w["t"].dt.hour
    for uid, wu in w.groupby("uid"):
        night = wu[wu["hour"] < 6]
        home = night["location"].mode().iloc[0] if len(night) else None
        dayt = wu[(wu["hour"] >= 9) & (wu["hour"] < 21)]
        f.setdefault(uid, {}).update({
            "places_per_day": wu.groupby("day")["location"].nunique().mean(),
            "home_frac_day": (dayt["location"] == home).mean() if home is not None and len(dayt) else np.nan,
        })
    del w

    # conversation
    c = _r("sensing/conversation")
    c["t"] = _local(c["start_timestamp"])
    c = c[_in_term(c["t"])]
    c["min"] = (c["end_timestamp"] - c["start_timestamp"]) / 60
    for uid, cu in c.groupby("uid"):
        f.setdefault(uid, {}).update({
            "conv_min_per_day": per_day(uid, cu["min"].sum()),
            "conv_n_per_day": per_day(uid, len(cu)),
        })

    # bluetooth co-presence
    b = _r("sensing/bluetooth")
    b["t"] = _local(b["timestamp"])
    b = b[_in_term(b["t"])]
    b["day"] = b["t"].dt.date
    bt = b.groupby(["uid", "day"])["MAC"].nunique().groupby("uid").mean()
    for uid, v in bt.items():
        f.setdefault(uid, {})["log_bt_devices_per_day"] = np.log1p(v)

    # sleep proxy: longest dark (phone in darkness >= 1h) episode starting 20:00-05:00, per night
    d = _r("sensing/dark")
    d["t"] = _local(d["start_timestamp"])
    d = d[_in_term(d["t"])]
    d["h"] = (d["end_timestamp"] - d["start_timestamp"]) / 3600
    d["onset"] = (d["t"].dt.hour + d["t"].dt.minute / 60 - 18) % 24  # hours after 18:00
    d = d[(d["onset"] >= 2) & (d["onset"] <= 11)]  # 20:00..05:00
    d["night"] = (d["t"] - pd.Timedelta(hours=12)).dt.date
    main = d.sort_values("h").groupby(["uid", "night"]).tail(1)
    for uid, mu in main.groupby("uid"):
        if len(mu) >= 7:
            f.setdefault(uid, {}).update({
                "sleep_h": mu["h"].mean(),
                "sleep_onset_h": mu["onset"].mean(),
                "sleep_onset_sd": mu["onset"].std(),
            })

    # phone lock (>= 1h stretches)
    pl = _r("sensing/phonelock")
    pl["t"] = _local(pl["start_timestamp"])
    pl = pl[_in_term(pl["t"])]
    pl["h"] = (pl["end_timestamp"] - pl["start_timestamp"]) / 3600
    for uid, pu in pl.groupby("uid"):
        f.setdefault(uid, {})["locked_h_per_day"] = per_day(uid, pu["h"].sum())

    # night phone use: overlap of lock stretches with 00:00-06:00, per observed night
    for uid, pu in pl.groupby("uid"):
        tot = 0.0
        for st, en in zip(_local(pu["start_timestamp"]), _local(pu["end_timestamp"])):
            day0 = st.normalize()
            for base in (day0, day0 + pd.Timedelta(days=1)):
                lo, hi = max(st, base), min(en, base + pd.Timedelta(hours=6))
                if hi > lo:
                    tot += (hi - lo).total_seconds() / 3600
        f.setdefault(uid, {})["night_locked_frac"] = per_day(uid, tot) / 6

    out = pd.DataFrame.from_dict(f, orient="index")[list(FEATURES)]
    out.index = DATASET + ":" + out.index.astype(str)
    out.index.name = "person_id"
    return out.astype(float)


def build() -> dict:
    q = items()
    traits = score(q)
    persons = traits.apply(schema.zscore).reset_index(names="uid")
    persons["person_id"] = DATASET + ":" + persons["uid"].astype(str)
    persons = persons.drop(columns="uid")
    feats = features()
    feats = feats[feats.index.isin(persons["person_id"])]
    schema.save(DATASET, persons, feats)
    return {
        "labelled": len(persons),
        "with_features": len(feats),
        "features": feats.shape[1],
        "missing_cells": int(feats.isna().sum().sum()),
        "bfi_check": check_bfi(q),
    }
