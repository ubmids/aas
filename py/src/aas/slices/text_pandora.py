"""PANDORA (Gjurkovic et al. 2021, arXiv 2004.04460): Reddit comments + Big Five.

Access is by request (https://psy.takelab.fer.hr/datasets/all/pandora/). Once granted,
put the two files here, unzipped:

    data/raw/pandora/author_profiles.csv          one row per Reddit author; Big Five
                                                  (percentiles, ~1.6k authors have them),
                                                  MBTI, enneagram, age, gender, ...
    data/raw/pandora/all_comments_since_2015.csv  one row per comment: author, body,
                                                  subreddit, created_utc, ... (several GB)

then run:  cd py && uv run python scripts/run_text.py --dataset pandora

What it does:
  1. keeps authors with all five Big Five scores; z-scores them within PANDORA (schema rule)
  2. streams the comments file in chunks (it is too big for memory), keeping only those
     authors and at most MAX_COMMENTS per author
  3. drops comments in personality/typology subreddits and strips leak tokens: the Big
     Five labels came from people posting their own test results, so "I scored 92nd
     percentile on openness" in r/BigFive would be the answer written on the page
  4. same two feature families as the MBTI slice; scores with evaluate.predict() (Pearson r,
     permutation p, bootstrap CI, Holm)
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .. import schema
from ..evaluate import predict
from . import text as T

PANDORA = schema.RAW / "pandora"
PROFILES = PANDORA / "author_profiles.csv"
COMMENTS = PANDORA / "all_comments_since_2015.csv"
MAX_COMMENTS = 1000  # per author, first seen in file order

# Column names as published; aliases cover small spelling differences between releases.
BIG5_COLS = {
    "O": ["openness"],
    "C": ["conscientiousness"],
    "E": ["extraversion", "extroversion"],
    "A": ["agreeableness"],
    "N": ["neuroticism", "emotional_stability"],  # if only stability is present it is flipped below
}

PERSONALITY_SUBS = re.compile(
    r"mbti|[ie][ns][tf][jp]|enneagram|personality|typology|socionics|bigfive|big5|introvert|extrovert|"
    r"16personalities|psychology_of|myersbriggs|jung",
    re.I,
)
BIG5_LEAK = re.compile(
    r"\b(?:openness|conscientious\w*|extr[ao]version|extr[ao]vert\w*|agreeable\w*|neurotic\w*|"
    r"emotional stability|big ?five|big ?5|ocean|ipip|neo-?pi\w*|bfi|percentile\w*|"
    r"truity|understandmyself|\d{1,3}(?:st|nd|rd|th)\s+percentile)\b",
    re.I,
)


def _need_files():
    missing = [p for p in (PROFILES, COMMENTS) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "PANDORA is not on this machine yet. Missing:\n  "
            + "\n  ".join(str(p) for p in missing)
            + "\nRequest access at https://psy.takelab.fer.hr/datasets/all/pandora/ (Gjurkovic et al. 2021), "
            f"then unzip author_profiles.csv and all_comments_since_2015.csv into {PANDORA}/"
        )


def _pick(cols: list[str], names: list[str]) -> str | None:
    low = {c.lower().strip(): c for c in cols}
    for n in names:
        if n in low:
            return low[n]
    return None


def load_profiles() -> pd.DataFrame:
    _need_files()
    prof = pd.read_csv(PROFILES, low_memory=False)
    author = _pick(list(prof.columns), ["author", "username", "user"])
    if author is None:
        raise ValueError(f"no author column in {PROFILES.name}; columns are {list(prof.columns)}")
    out = pd.DataFrame({"author": prof[author].astype(str)})
    for t, names in BIG5_COLS.items():
        c = _pick(list(prof.columns), names)
        if c is None:
            raise ValueError(f"no column for {t} ({names}) in {PROFILES.name}; columns are {list(prof.columns)}")
        v = pd.to_numeric(prof[c], errors="coerce")
        out[t] = -v if c.lower() == "emotional_stability" else v
    mbti = _pick(list(prof.columns), ["mbti", "mbti_type"])
    out["mbti"] = prof[mbti].astype("string").str.upper().where(lambda s: s.str.fullmatch(r"[IE][NS][TF][JP]"), None) if mbti else None
    out = out.dropna(subset=list(BIG5_COLS)).drop_duplicates("author")
    if len(out) == 0:
        raise ValueError("no author has all five Big Five scores; check the file version")
    for t in BIG5_COLS:
        out[t] = schema.zscore(out[t])
    return out.reset_index(drop=True)


def load_comments(authors: set[str], chunksize: int = 500_000, log=print) -> dict[str, list[str]]:
    _need_files()
    head = pd.read_csv(COMMENTS, nrows=5)
    cols = list(head.columns)
    a, b, s = _pick(cols, ["author"]), _pick(cols, ["body", "text", "comment"]), _pick(cols, ["subreddit"])
    if a is None or b is None:
        raise ValueError(f"need author and body columns in {COMMENTS.name}; columns are {cols}")
    keep: dict[str, list[str]] = defaultdict(list)
    seen = dropped_sub = 0
    for i, ch in enumerate(pd.read_csv(COMMENTS, usecols=[c for c in (a, b, s) if c], chunksize=chunksize, dtype=str)):
        ch = ch[ch[a].isin(authors)]
        seen += len(ch)
        if s:
            bad = ch[s].fillna("").str.contains(PERSONALITY_SUBS)
            dropped_sub += int(bad.sum())
            ch = ch[~bad]
        for au, body in zip(ch[a], ch[b].fillna("")):
            if len(keep[au]) < MAX_COMMENTS:
                keep[au].append(body)
        log(f"  chunk {i}: {seen} comments by labelled authors so far, {dropped_sub} dropped (personality subreddits)")
    return keep


def strip_all(post: str) -> str:
    return BIG5_LEAK.sub(" ", T.strip_leaks(post)).strip()


def main(k: int = 20) -> int:
    log = lambda *m: print(f"[{time.strftime('%H:%M:%S')}]", *m, flush=True)  # noqa: E731
    try:
        prof = load_profiles()
    except (FileNotFoundError, ValueError) as e:
        print(f"PANDORA: {e}")
        return 1
    log(f"{len(prof)} authors with Big Five")
    comments = load_comments(set(prof.author), log=log)
    prof = prof[prof.author.isin(comments)].reset_index(drop=True)
    prof["person_id"] = "pandora:" + pd.util.hash_pandas_object(prof.author, index=False).astype(str)  # no handles
    people = pd.DataFrame({"person_id": prof.person_id, "posts": [[strip_all(c) for c in comments[x]] for x in prof.author]})
    log(f"{len(people)} authors with comments; median {int(np.median([len(p) for p in people.posts]))} comments each")

    lang = T.language_table(people, "posts")
    rng = np.random.default_rng(0)
    chosen = [[ps[i] for i in T.sample_posts(ps, k, rng)] for ps in people.posts]
    emb = T.embedding_tables(list(people.person_id), {"stripped": chosen}, T.Embedder(), log=log)["stripped"]

    persons = prof[["person_id", *BIG5_COLS, "mbti"]].copy()
    schema.save("pandora", persons, pd.concat([lang, emb], axis=1))
    labels = persons.set_index("person_id")
    rows = []
    for name, X in [("interpretable", lang), ("embeddings", emb), ("both", pd.concat([lang, emb], axis=1))]:
        r = predict(X, labels)
        r.insert(0, "features", name)
        rows.append(r)
        log(name + "\n" + r.round(3).to_string(index=False))
    out = Path(__file__).resolve().parents[3] / "docs" / "text_pandora_results.csv"
    pd.concat(rows).to_csv(out, index=False)
    log(f"wrote {out}")
    return 0
