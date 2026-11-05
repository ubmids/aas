"""play: Steam gaming-platform footprints -> Big Five.

Source: Hao Lin, Leixiao Li, "A dataset on big five personality traits and user
data on digital gaming social platforms", Mendeley Data v5,
doi:10.17632/yyz69kysy3.5, licence CC BY 4.0.
File used: Steam0706.csv (66,775 bytes, sha256 dfd0f83b...644d).
Not used: web2.rar (234 MB of profile-page screenshots, over our disk budget and
identifying), questionnaire.docx (read for the instrument only).

Columns, sorted by role
  identifiers   No., nickname/IP/homepage (all masked "***"), URL of avatar     -> dropped
  paradata      time stamp, time spent on questionnaire                        -> exclusion rule only
  demographics  gender, age band, country                                       -> not features
  questionnaire BFI-2(1..60) raw items 1-5; EXT AGR CON NEG OPE precomputed sums -> labels only
  behaviour     Platform level, Number of badges/games/screenshots/evaluations
                (= reviews)/groups/friends, Is it banned or not                 -> features

Labels: we re-score the BFI-2 ourselves from the 60 raw items (Soto & John 2017
key: item i belongs to domain (i-1) % 5 in E,A,C,N,O order; 30 reverse-keyed
items scored 6 - x). Our sums match the shipped EXT..OPE columns for 171/173
people; rows No. 90 and 91 disagree on E, A and N, so the shipped totals look
miscopied there and we trust the items. Scores are item means (one person has a
single missing item, prorated), then z-scored within the kept sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import schema

RAW = schema.RAW / "steam" / "Steam0706.csv"
DATASET = "steam"

DOMAINS = ["E", "A", "C", "N", "O"]  # BFI-2 item order cycles through these
REVERSED = {3, 4, 5, 8, 9, 11, 12, 16, 17, 22, 23, 24, 25, 26, 28, 29, 30, 31,
            36, 37, 42, 44, 45, 47, 48, 49, 50, 51, 55, 58}
SPEEDER_SECONDS = 120  # 60 items + demographics in under 2 s per item

COUNTS = {
    "Platform level": "level",
    "Number of badges": "badges",
    "Number of games": "games",
    "Number of screenshots": "screenshots",
    "Number of evaluations": "reviews",
    "Number of groups": "groups",
    "Number of friends": "friends",
}

# Plain-English hypothesis per feature (also printed in the write-up).
HYPOTHESES = {
    "log_level": "C/E: levelling up takes sustained, goal-directed effort (C) and is social display (E)",
    "log_badges": "C: badges reward completing sets and events, a completionist habit",
    "log_games": "O: a bigger library means more variety-seeking across genres",
    "log_screenshots": "O: capturing scenes suggests aesthetic interest",
    "log_reviews": "O/A: writing reviews is opinion-sharing; tone aside, it is helping others choose",
    "log_groups": "E: joining communities",
    "log_friends": "E: more friends on the platform",
    "banned": "A-, C-: a VAC/game ban means cheating or rule-breaking (only 3 people, so near-useless)",
    "screenshots_per_game": "O: aesthetic attention per game rather than per hour of life",
    "reviews_per_game": "A/O: share of what you play that you bother to review",
    "badges_per_level": "C: how much of your level comes from crafting badges (completionism)",
    "friends_per_level": "E: sociability relative to how invested you are in the platform",
    "groups_per_friend": "O/E-: community-of-interest joining vs one-to-one ties",
    "sections_shown": "O/E: how many profile showcase sections are non-empty (an open, filled-in profile)",
}


def _count(s: pd.Series) -> pd.Series:
    """Values arrive as '\\n\\t\\t 16 \\t\\t' or 'Nan'. Return float with NaN for 'Nan'."""
    s = s.astype(str).str.strip()
    return pd.to_numeric(s.where(~s.str.lower().isin(["nan", ""])), errors="coerce")


def _seconds(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    mins = s.str.extract(r"^(\d+)\s*分$")[0]  # one row is "5分" = 5 minutes
    out = pd.to_numeric(s, errors="coerce")
    return out.fillna(pd.to_numeric(mins, errors="coerce") * 60)


def read_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW, encoding="utf-8-sig")
    df.columns = [" ".join(c.split()) for c in df.columns]
    return df


def score_bfi2(df: pd.DataFrame) -> pd.DataFrame:
    items = df[[f"BFI-2({i})" for i in range(1, 61)]].astype(float)
    for i in REVERSED:
        items[f"BFI-2({i})"] = 6 - items[f"BFI-2({i})"]
    out = {}
    for k, d in enumerate(DOMAINS):
        cols = [f"BFI-2({i})" for i in range(k + 1, 61, 5)]
        out[d] = items[cols].mean(axis=1)  # mean prorates the one missing item
    return pd.DataFrame(out, index=df.index)


def counts(df: pd.DataFrame) -> pd.DataFrame:
    c = pd.DataFrame({v: _count(df[k]) for k, v in COUNTS.items()}, index=df.index)
    c["banned"] = pd.to_numeric(df["Is it banned or not"], errors="coerce")
    return c


def make_features(c: pd.DataFrame, mode: str = "zero") -> pd.DataFrame:
    """mode='zero': a blank section on a reachable profile counts as 0 (Steam does
    not render zero-count sections). mode='missing': blank stays unknown (imputed
    with the training median inside evaluate.predict) plus a was-blank flag."""
    raw = c.copy()
    if mode == "zero":
        c = c.fillna(0)
    lg = np.log1p
    f = pd.DataFrame(index=c.index)
    for k in ["level", "badges", "games", "screenshots", "reviews", "groups", "friends"]:
        f[f"log_{k}"] = lg(c[k])
    f["banned"] = c["banned"]
    # log((a+1)/(b+1)): a ratio that stays finite at zero and is symmetric on a log scale
    f["screenshots_per_game"] = lg(c["screenshots"]) - lg(c["games"])
    f["reviews_per_game"] = lg(c["reviews"]) - lg(c["games"])
    f["badges_per_level"] = lg(c["badges"]) - lg(c["level"])
    f["friends_per_level"] = lg(c["friends"]) - lg(c["level"])
    f["groups_per_friend"] = lg(c["groups"]) - lg(c["friends"])
    f["sections_shown"] = (raw[["badges", "games", "screenshots", "reviews", "groups", "friends"]].fillna(0) > 0).sum(axis=1).astype(float)
    if mode == "missing":
        for k in ["badges", "games", "screenshots", "reviews", "groups", "friends"]:
            f[f"blank_{k}"] = raw[k].isna().astype(float)
    return f


def prepare() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (persons, counts table for people with a profile, exclusion log)."""
    df = read_raw()
    log = {"rows": len(df)}
    df["pid"] = "steam:" + df["No."].astype(int).astype(str)
    secs = _seconds(df["Time spent on questionnaire completion"])
    fast = secs < SPEEDER_SECONDS
    log["speeders_excluded"] = df.loc[fast, "No."].tolist()
    df, secs = df[~fast], secs[~fast]

    traits = score_bfi2(df)
    shipped = {"E": "EXT", "A": "AGR", "C": "CON", "N": "NEG", "O": "OPE"}
    mism = set()
    for d, col in shipped.items():
        bad = (traits[d] * 12).round() != df[col]
        mism |= set(df.loc[bad, "No."])
    log["shipped_sum_mismatch"] = sorted(mism)
    log["missing_items"] = int(df[[f"BFI-2({i})" for i in range(1, 61)]].isna().sum().sum())

    persons = pd.DataFrame({"person_id": df["pid"].to_numpy()})
    for d in DOMAINS:
        persons[d] = schema.zscore(traits[d]).to_numpy()
    persons["mbti"] = None
    log["persons"] = len(persons)

    c = counts(df)
    c.index = df["pid"]
    no_profile = c["level"].isna()  # every behaviour blank: profile not reachable
    log["no_profile_excluded"] = int(no_profile.sum())
    c = c[~no_profile]
    log["with_features"] = len(c)
    return persons, c, log


def build() -> dict:
    persons, c, log = prepare()
    features = make_features(c, "zero")
    schema.save(DATASET, persons, features)
    return log


if __name__ == "__main__":
    print(build())
