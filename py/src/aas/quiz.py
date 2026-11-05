"""Ground truth: the Big Five questionnaire, its norms, and a short in-app quiz.

Built on the Open-Source Psychometrics IPIP-FFM dump (Goldberg's 50-item
"Big-Five Factor Markers", ~1M online respondents, 2016-2018):
https://openpsychometrics.org/_rawdata/IPIP-FFM-data-8Nov2018.zip

What lives here
  ITEMS / KEY        the 50 items and which trait each scores, with its direction
  build_clean()      raw zip -> cleaned compact parquet (exclusions documented)
  keyed(), scales()  answers -> keyed answers -> trait scores (mean of 1..5)
  make_norms()       means, SDs, percentile tables, short form -> ipip_norms.json
  score(answers)     {item: 1..5} -> {trait: {z, percentile, ...}} for one person

Traits use the project letters O C E A N. The dataset's EST scale is emotional
STABILITY, so N (neuroticism) is EST with every direction flipped.
"""

from __future__ import annotations

import json
import zipfile
from functools import lru_cache
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import RAW, TRAITS, TRAIT_NAMES

URL = "https://openpsychometrics.org/_rawdata/IPIP-FFM-data-8Nov2018.zip"
RAW_DIR = RAW / "ipip"
ZIP = RAW_DIR / "IPIP-FFM-data-8Nov2018.zip"
CLEAN = RAW_DIR / "ipip_clean.parquet"
NORMS = Path(__file__).parent / "data" / "ipip_norms.json"

# ---------------------------------------------------------------- the items
# Text from the dataset codebook. Direction from https://ipip.ori.org/newBigFive5broadKey.htm
# (+ = agreeing means MORE of the dataset's scale; - = agreeing means less).
ITEMS: dict[str, str] = {
    "EXT1": "I am the life of the party.",
    "EXT2": "I don't talk a lot.",
    "EXT3": "I feel comfortable around people.",
    "EXT4": "I keep in the background.",
    "EXT5": "I start conversations.",
    "EXT6": "I have little to say.",
    "EXT7": "I talk to a lot of different people at parties.",
    "EXT8": "I don't like to draw attention to myself.",
    "EXT9": "I don't mind being the center of attention.",
    "EXT10": "I am quiet around strangers.",
    "EST1": "I get stressed out easily.",
    "EST2": "I am relaxed most of the time.",
    "EST3": "I worry about things.",
    "EST4": "I seldom feel blue.",
    "EST5": "I am easily disturbed.",
    "EST6": "I get upset easily.",
    "EST7": "I change my mood a lot.",
    "EST8": "I have frequent mood swings.",
    "EST9": "I get irritated easily.",
    "EST10": "I often feel blue.",
    "AGR1": "I feel little concern for others.",
    "AGR2": "I am interested in people.",
    "AGR3": "I insult people.",
    "AGR4": "I sympathize with others' feelings.",
    "AGR5": "I am not interested in other people's problems.",
    "AGR6": "I have a soft heart.",
    "AGR7": "I am not really interested in others.",
    "AGR8": "I take time out for others.",
    "AGR9": "I feel others' emotions.",
    "AGR10": "I make people feel at ease.",
    "CSN1": "I am always prepared.",
    "CSN2": "I leave my belongings around.",
    "CSN3": "I pay attention to details.",
    "CSN4": "I make a mess of things.",
    "CSN5": "I get chores done right away.",
    "CSN6": "I often forget to put things back in their proper place.",
    "CSN7": "I like order.",
    "CSN8": "I shirk my duties.",
    "CSN9": "I follow a schedule.",
    "CSN10": "I am exacting in my work.",
    "OPN1": "I have a rich vocabulary.",
    "OPN2": "I have difficulty understanding abstract ideas.",
    "OPN3": "I have a vivid imagination.",
    "OPN4": "I am not interested in abstract ideas.",
    "OPN5": "I have excellent ideas.",
    "OPN6": "I do not have a good imagination.",
    "OPN7": "I am quick to understand things.",
    "OPN8": "I use difficult words.",
    "OPN9": "I spend time reflecting on things.",
    "OPN10": "I am full of ideas.",
}

# Reverse-keyed items on the dataset's own scales (EXT, EST, AGR, CSN, OPN).
_REVERSED_ON_SCALE = {
    "EXT": {2, 4, 6, 8, 10},
    "EST": {1, 3, 5, 6, 7, 8, 9, 10},  # stability: only "relaxed" and "seldom blue" are +
    "AGR": {1, 3, 5, 7},
    "CSN": {2, 4, 6, 8},
    "OPN": {2, 4, 6},
}
_PREFIX_TO_TRAIT = {"EXT": "E", "EST": "N", "AGR": "A", "CSN": "C", "OPN": "O"}


def _build_key() -> dict[str, tuple[str, int]]:
    key = {}
    for code in ITEMS:
        prefix, num = code[:3], int(code[3:])
        sign = -1 if num in _REVERSED_ON_SCALE[prefix] else 1
        if prefix == "EST":  # stability -> neuroticism
            sign = -sign
        key[code] = (_PREFIX_TO_TRAIT[prefix], sign)
    return key


KEY: dict[str, tuple[str, int]] = _build_key()  # item -> (trait letter, +1 / -1)
TRAIT_ITEMS: dict[str, list[str]] = {t: [c for c, (tt, _) in KEY.items() if tt == t] for t in TRAITS}
ITEM_COLS = list(ITEMS)
TIME_COLS = [c + "_E" for c in ITEM_COLS]
# Order the items were shown on the page: EXT1, AGR1, CSN1, EST1, OPN1, EXT2, ...
PAGE_ORDER = [f"{p}{i}" for i in range(1, 11) for p in ("EXT", "AGR", "CSN", "EST", "OPN")]


# ---------------------------------------------------------------- scoring
def keyed(items: pd.DataFrame) -> pd.DataFrame:
    """Flip reverse-keyed answers (x -> 6 - x) so higher always means more of the trait."""
    out = items.astype(float).copy()
    for c in out.columns:
        if KEY[c][1] < 0:
            out[c] = 6 - out[c]
    return out


def scales(items: pd.DataFrame, which: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """Trait score = mean of that trait's keyed items (1..5 scale)."""
    which = which or TRAIT_ITEMS
    k = keyed(items[[c for t in which for c in which[t]]])
    return pd.DataFrame({t: k[which[t]].mean(axis=1) for t in which})


def cronbach_alpha(x: pd.DataFrame) -> float:
    k = x.shape[1]
    return float(k / (k - 1) * (1 - x.var(ddof=1).sum() / x.sum(axis=1).var(ddof=1)))


# ---------------------------------------------------------------- data
def download() -> Path:
    import urllib.request

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not ZIP.exists():
        urllib.request.urlretrieve(URL, ZIP)
    return ZIP


def load_raw(path: Path = ZIP) -> pd.DataFrame:
    """Read the tab-separated CSV straight out of the zip (never unpacked to disk)."""
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith("data-final.csv"))
        return pd.read_csv(z.open(name), sep="\t", low_memory=False)


def _longest_run(a: np.ndarray) -> np.ndarray:
    """Longest run of identical consecutive answers per row (rows = people)."""
    n, m = a.shape
    best = np.ones(n, dtype=np.int16)
    cur = np.ones(n, dtype=np.int16)
    for j in range(1, m):
        same = a[:, j] == a[:, j - 1]
        cur = np.where(same, cur + 1, 1)
        best = np.maximum(best, cur)
    return best


# Exclusion thresholds (see docs/quiz.md for the reasoning).
MIN_MEDIAN_ITEM_MS = 1000   # under ~1 s per statement nobody is reading it
MAX_RUN = 15                # >= 15 identical answers in a row, in page order, = straight-lining


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, int, int]]]:
    """Apply exclusions in order. Returns kept rows and [(rule, dropped, remaining)]."""
    log = [("raw rows", 0, len(df))]

    def step(name, keep):
        nonlocal df
        before = len(df)
        df = df[keep]
        log.append((name, before - len(df), len(df)))

    a = df[ITEM_COLS]
    step("missing or 0 (skipped) answer", a.notna().all(axis=1) & a.isin([1, 2, 3, 4, 5]).all(axis=1))
    step("IPC != 1 (several submissions from same IP)", df["IPC"] == 1)
    t = df[TIME_COLS]
    step("timing missing or negative", t.notna().all(axis=1) & (t > 0).all(axis=1) & df["testelapse"].notna())
    med = df[TIME_COLS].median(axis=1)
    step(f"median time per item < {MIN_MEDIAN_ITEM_MS} ms (speeder)", med >= MIN_MEDIAN_ITEM_MS)
    raw = df[PAGE_ORDER].to_numpy()
    step("identical answer to all 50 items", raw.std(axis=1) > 0)
    raw = df[PAGE_ORDER].to_numpy()
    step(f"longest identical run >= {MAX_RUN} (straight-lining)", _longest_run(raw) < MAX_RUN)
    return df, log


def timing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Response-time behaviour only, never the answers. Times are in ms per item.

    EXT1_E (the first item) also counts page-load / reading-the-instructions time,
    so it is left out of the per-item summaries.
    """
    t = np.log(df[TIME_COLS[1:]].astype(float).clip(lower=100))
    return pd.DataFrame(
        {
            "log_median_item_ms": t.median(axis=1),
            "log_item_ms_sd": t.std(axis=1),
            "log_item_ms_iqr": t.quantile(0.75, axis=1) - t.quantile(0.25, axis=1),
            "frac_items_under_1500ms": (df[TIME_COLS[1:]] < 1500).mean(axis=1),
            "log_first_item_ms": np.log(df["EXT1_E"].astype(float).clip(lower=100)),
            "log_test_s": np.log1p(df["testelapse"].astype(float).clip(lower=0)),
            "log_intro_s": np.log1p(df["introelapse"].astype(float).clip(lower=0)),
            "log_end_s": np.log1p(df["endelapse"].astype(float).clip(lower=0)),
        },
        index=df.index,
    )


def build_clean(delete_raw: bool = False) -> tuple[pd.DataFrame, list]:
    """Raw zip -> data/raw/ipip/ipip_clean.parquet (answers int8 + timing summaries + a little metadata)."""
    df, log = clean(load_raw(download()))
    out = df[ITEM_COLS].astype("int8")
    out = out.join(timing_features(df).astype("float32"))
    out["screenw"] = df["screenw"].astype("float32")
    out["screenh"] = df["screenh"].astype("float32")
    out["country"] = df["country"].astype("category")
    out["hour_utc"] = pd.to_datetime(df["dateload"], errors="coerce").dt.hour.astype("float32")
    rng = np.random.default_rng(20181108)
    out["half"] = rng.integers(0, 2, len(out)).astype("int8")  # 0 = train (choose items), 1 = held out
    out = out.reset_index(drop=True)
    out.index.name = "row"
    out.to_parquet(CLEAN, compression="zstd")
    (RAW_DIR / "clean_log.json").write_text(json.dumps(log, indent=1))
    if delete_raw:
        ZIP.unlink(missing_ok=True)
    return out, log


def load_clean() -> pd.DataFrame:
    return pd.read_parquet(CLEAN)


# ---------------------------------------------------------------- norms
def _percentile_table(sums: pd.Series, lo: int, hi: int) -> list[float]:
    """Mid-rank percentile for every possible raw sum lo..hi: % below + half of % equal."""
    counts = sums.value_counts().reindex(range(lo, hi + 1), fill_value=0).to_numpy(float)
    n = counts.sum()
    below = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return [round(float(v), 3) for v in 100 * (below + counts / 2) / n]


def corrected_item_total(k: pd.DataFrame) -> pd.Series:
    """Each item vs the sum of the OTHER items of its scale (so it can't correlate with itself)."""
    tot = k.sum(axis=1)
    return pd.Series({c: np.corrcoef(k[c], tot - k[c])[0, 1] for c in k.columns})


def choose_short_form(train: pd.DataFrame, n_items: int = 4, balanced: bool = True) -> dict[str, list[str]]:
    """Best items per trait by corrected item-total r on the training half.

    balanced=True keeps at least one reverse-keyed and one forward-keyed item when
    the trait has both, so a person who agrees with everything doesn't score high
    on everything (acquiescence).
    """
    short = {}
    for t in TRAITS:
        r = corrected_item_total(keyed(train[TRAIT_ITEMS[t]])).sort_values(ascending=False)
        pick = list(r.index[:n_items])
        if balanced and n_items >= 2:
            for want in (+1, -1):
                if not any(KEY[c][1] == want for c in pick):
                    best = next(c for c in r.index if KEY[c][1] == want)
                    # replace the weakest item of the over-represented direction
                    drop = next(c for c in reversed(pick) if KEY[c][1] != want)
                    pick[pick.index(drop)] = best
        short[t] = sorted(pick, key=lambda c: -r[c])
    return short


def make_norms(df: pd.DataFrame, short: dict[str, list[str]], n_clean: int) -> dict:
    k = keyed(df[ITEM_COLS])
    norms = {
        "source": URL,
        "n": int(n_clean),
        "scale": "trait score = mean of keyed items, 1..5",
        "traits": {},
    }
    for t in TRAITS:
        items, sitems = TRAIT_ITEMS[t], short[t]
        full = k[items].mean(axis=1)
        ssum = k[sitems].sum(axis=1).round().astype(int)
        fsum = k[items].sum(axis=1).round().astype(int)
        norms["traits"][t] = {
            "name": TRAIT_NAMES[t],
            "items": {c: KEY[c][1] for c in items},
            "mean": round(float(full.mean()), 5),
            "sd": round(float(full.std(ddof=0)), 5),
            "pct_by_sum": {"lo": len(items), "table": _percentile_table(fsum, len(items), 5 * len(items))},
            "short": {
                "items": sitems,
                "mean": round(float(ssum.mean() / len(sitems)), 5),
                "sd": round(float((ssum / len(sitems)).std(ddof=0)), 5),
                "pct_by_sum": {"lo": len(sitems), "table": _percentile_table(ssum, len(sitems), 5 * len(sitems))},
            },
            "item_means": {c: round(float(k[c].mean()), 5) for c in items},
            "item_cov": np.round(k[items].cov(ddof=0).to_numpy(), 5).tolist(),
        }
    return norms


@lru_cache(maxsize=1)
def load_norms(path: str | None = None) -> dict:
    return json.loads(Path(path or NORMS).read_text())


def short_form(norms: dict | None = None) -> list[str]:
    """The short-form items in page order (interleaved across traits)."""
    norms = norms or load_norms()
    per = [norms["traits"][t]["short"]["items"] for t in TRAITS]
    return [c for i in range(max(map(len, per))) for p in per for c in p[i : i + 1]]


def _phi(z: float) -> float:
    return 0.5 * (1 + erf(z / sqrt(2)))


def score(answers: dict[str, int], norms: dict | None = None) -> dict[str, dict]:
    """One person's answers {item code: 1..5} -> per trait {z, percentile, n_items, form}.

    z is relative to the ~600k cleaned IPIP respondents (mostly US/UK/CA/AU adults
    who took a free online test). Uses the exact empirical percentile table when the
    answered items are the full 10 or exactly the short form; otherwise z comes from
    the stored item means/covariances and the percentile from the normal curve.
    Traits with no answered items are left out.
    """
    norms = norms or load_norms()
    for c, v in answers.items():
        if c not in KEY:
            raise KeyError(f"unknown item {c!r}")
        if v not in (1, 2, 3, 4, 5):
            raise ValueError(f"{c}: answer must be an integer 1..5, got {v!r}")
    out = {}
    for t in TRAITS:
        nt = norms["traits"][t]
        items = [c for c in nt["items"] if c in answers]
        if not items:
            continue
        kv = {c: answers[c] if nt["items"][c] > 0 else 6 - answers[c] for c in items}
        s = sum(kv.values())
        m = s / len(items)
        if set(items) == set(nt["items"]):
            form, mean, sd, tab = "full", nt["mean"], nt["sd"], nt["pct_by_sum"]
        elif set(items) == set(nt["short"]["items"]):
            sn = nt["short"]
            form, mean, sd, tab = "short", sn["mean"], sn["sd"], sn["pct_by_sum"]
        else:
            order = list(nt["items"])
            idx = [order.index(c) for c in items]
            cov = np.array(nt["item_cov"])[np.ix_(idx, idx)]
            form, tab = "partial", None
            mean = float(np.mean([nt["item_means"][c] for c in items]))
            sd = float(np.sqrt(cov.sum()) / len(items))
        z = (m - mean) / sd
        pct = tab["table"][s - tab["lo"]] if tab else 100 * _phi(z)
        out[t] = {
            "trait": nt["name"],
            "score": round(m, 3),
            "z": round(float(z), 3),
            "percentile": round(float(pct), 1),
            "n_items": len(items),
            "form": form,
        }
    return out
