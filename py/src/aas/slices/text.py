"""Text slice: what people write -> personality.

Dataset here: MBTI PersonalityCafe (Kaggle datasnaek/mbti-type, mirrored on
Hugging Face as jingjietan/kaggle-mbti). 8,675 forum users, their last ~50
posts, and the MBTI type they reported.

Two feature families, both computed per person from their own posts only:

  lang_*  interpretable language measures (counts and rates you can explain)
  emb_*   sentence embeddings: a small neural model turns each post into 384
          numbers that capture its meaning; we average a person's posts.

The leak: this is an MBTI forum, so people write their own type ("as an INTJ"),
other type names, cognitive-function jargon ("my Fi"), and links to type
sub-forums. A model can then "predict" type by reading the answer off the page.
strip_leaks() removes those tokens; results are reported with and without it.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from ..schema import RAW

MBTI_RAW = RAW / "mbti"
LEXICONS = RAW / "lexicons"
MODELS = RAW / "models"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Published class counts of the original mbti_1.csv, used to prove that the
# mirror's integer type codes decode to the right letters.
KAGGLE_COUNTS = {
    "INFP": 1832, "INFJ": 1470, "INTP": 1304, "INTJ": 1091, "ENTP": 685, "ENFP": 675,
    "ISTP": 337, "ISFP": 271, "ENTJ": 231, "ISTJ": 205, "ENFJ": 190, "ISFJ": 166,
    "ESTP": 89, "ESFP": 48, "ESFJ": 42, "ESTJ": 39,
}
# The four binary targets. Value 1 means the first letter of the pair.
LETTERS = {"I/E": ("I", 0), "N/S": ("N", 1), "T/F": ("T", 2), "J/P": ("J", 3)}


# ---------------------------------------------------------------- loading

def _decode(code: int) -> str:
    # Mirror encodes type as 4 bits: 8=N, 4=J, 2=E, 1=F (verified against counts below).
    return ("E" if code & 2 else "I") + ("N" if code & 8 else "S") + ("F" if code & 1 else "T") + ("J" if code & 4 else "P")


def load_mbti() -> pd.DataFrame:
    """One row per user: person_id, mbti, posts (list of str). Raw, unstripped."""
    files = [MBTI_RAW / f"{s}.parquet" for s in ("train", "validation", "test")]
    missing = [f for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(
            f"MBTI data missing: {missing}. Download with:\n"
            "  for s in train validation test; do curl -L -o data/raw/mbti/$s.parquet "
            "https://huggingface.co/datasets/jingjietan/kaggle-mbti/resolve/main/data/$s-00000-of-00001.parquet; done"
        )
    d = pd.concat([pd.read_parquet(f) for f in files]).sort_values("__index_level_0__")
    d["mbti"] = d["ptype"].astype(int).map(_decode)
    got = d["mbti"].value_counts().to_dict()
    assert got == KAGGLE_COUNTS, f"type decoding does not match the original dataset: {got}"
    assert d["__index_level_0__"].is_unique and len(d) == 8675
    return pd.DataFrame({
        "person_id": "mbti_pc:" + d["__index_level_0__"].astype(str),
        "mbti": d["mbti"].to_numpy(),
        "posts": [t.strip("'").split("|||") for t in d["text"]],
    }).reset_index(drop=True)


def letter_targets(mbti: pd.Series) -> pd.DataFrame:
    """I/E -> 1 if I, N/S -> 1 if N, T/F -> 1 if T, J/P -> 1 if J."""
    return pd.DataFrame({k: (mbti.str[i] == ch).astype(int) for k, (ch, i) in LETTERS.items()}, index=mbti.index)


# ---------------------------------------------------------------- the leak

URL = re.compile(r"(?:https?://|www\.)\S+", re.I)
# Full types, plurals/possessives, and wildcard partials (INxJ, xNTP, IN*J, ENFP-T).
TYPE = re.compile(r"\b[iex*_][nsx*_][tfx*_][jpx*_](?:s|'s|es)?\b", re.I)
# Temperaments / pairs people use as labels (case-sensitive so ordinary words survive).
TEMPERAMENT = re.compile(r"\b(?:NF|NT|SJ|SP|SF|ST|NJ|NP)(?:s|'s)?\b")
# Jung/MBTI cognitive functions: Ne, Fi, Ti-Ne, NeTi, Fe/Ni. Case-sensitive: "Ne" not "ne".
FUNCTION = re.compile(r"\b(?:[NSTF][eiI]){1,4}(?:-(?:dom|aux|tert|inf)\w*)?\b")
LABEL_WORDS = re.compile(
    r"\b(?:intro|extra|extro|ambi)vert\w*|\bintroversion\b|\bextr[ao]version\b"
    r"|\bintuitives?\b|\bsensors?\b|\bthinkers?\b|\bfeelers?\b|\bjudgers?\b|\bperceivers?\b"
    r"|\bmbti\b|\bmyers\W*briggs\b|\bsocionics\b|\bkeirsey\b|\btypology\b"
    r"|\benneagram\w*\b|\b[1-9]w[1-9]\b|\bpersonality\s*cafe\b|\bperc\b",
    re.I,
)
LEAK_PATTERNS = {"url": URL, "type": TYPE, "temperament": TEMPERAMENT, "function": FUNCTION, "label_word": LABEL_WORDS}
# Strict mode also removes typology jargon that hints at a letter without naming it
# (found by inspecting what a word-count model leaned on after the default strip).
THEORY_WORDS = re.compile(
    r"\b(?:intuition|sensing|judging|perceiving|inferior|dominant|auxiliary|tertiary|cognitive|"
    r"functions?|types?|typed|typing|doors?lam\w*|sx|so/sp|sp/so|sx/so|so/sx|sp/sx|sx/sp)\b",
    re.I,
)


def strip_leaks(post: str, strict: bool = False) -> str:
    for p in LEAK_PATTERNS.values():
        post = p.sub(" ", post)
    if strict:
        post = THEORY_WORDS.sub(" ", post)
    return re.sub(r"\s+", " ", post).strip()


def leak_counts(posts: list[str]) -> dict[str, float]:
    """How often each leak pattern fires in a person's raw posts (per post)."""
    n = max(len(posts), 1)
    out = {f"leak_{k}": sum(len(p.findall(x)) for x in posts) / n for k, p in LEAK_PATTERNS.items()}
    # Which of the 16 types a person names most is the purest leak: one column per type.
    c = Counter(m.upper().rstrip("'S").rstrip("S")[:4] for x in posts for m in TYPE.findall(x))
    for t in KAGGLE_COUNTS:
        out[f"leak_says_{t}"] = c.get(t, 0) / n
    return out


# ---------------------------------------------------------------- interpretable features

WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")
I_WORDS = {"i", "me", "my", "mine", "myself", "i'm", "i've", "i'd", "i'll", "im", "ive"}
WE_WORDS = {"we", "us", "our", "ours", "ourselves", "we're", "we've", "we'd", "we'll"}
YOU_WORDS = {"you", "your", "yours", "yourself", "yourselves", "you're", "you've", "you'd", "you'll", "u", "ur"}
NEGATION = {"no", "not", "never", "none", "nobody", "nothing", "nowhere", "neither", "nor", "cannot", "without"}
SWEAR = {
    "fuck", "fucking", "fucked", "fucker", "fuckin", "motherfucker", "shit", "shitty", "bullshit", "damn",
    "dammit", "bitch", "bitches", "ass", "asshole", "crap", "bastard", "dick", "piss", "pissed", "wtf", "stfu", "hell",
}
TENTATIVE = {
    "maybe", "perhaps", "might", "possibly", "probably", "guess", "seems", "seem", "apparently", "somewhat",
    "unsure", "uncertain", "suppose", "presumably", "sometimes", "kinda", "sorta", "hopefully", "doubt",
}
CERTAIN = {
    "always", "definitely", "certainly", "absolutely", "sure", "clearly", "obviously", "completely", "totally",
    "undoubtedly", "exactly", "must", "fact", "truly", "never",
}
EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust", "positive", "negative"]

FEATURE_MEANING = {
    "lang_log_words": "how much they write in total (log of word count)",
    "lang_words_per_post": "how long a typical post is",
    "lang_i": "share of words that are I/me/my: attention on oneself",
    "lang_we": "share of words that are we/us/our: group belonging",
    "lang_you": "share of words that are you/your: addressing others directly",
    "lang_negation": "share of words like not/never/no and n't contractions: disagreement, denial",
    "lang_swear": "share of words that are swear words: informality, low restraint",
    "lang_question": "question marks per post: asking, engaging others",
    "lang_exclaim": "exclamation marks per post: enthusiasm, energy",
    "lang_ttr": "type-token ratio on 100-word windows: vocabulary variety",
    "lang_word_len": "average letters per word: more formal or abstract vocabulary",
    "lang_tentative": "share of hedging words (maybe, perhaps, probably)",
    "lang_certain": "share of certainty words (always, definitely, clearly)",
    "lang_tent_minus_cert": "hedging minus certainty: net cautiousness of tone",
    **{f"lang_emo_{e}": f"share of words the NRC lexicon tags as {e}" for e in EMOTIONS},
}


def load_emolex(path: Path | None = None) -> dict[str, set[str]]:
    """NRC Word-Emotion Association Lexicon v0.92 (Mohammad & Turney). Non-commercial research use."""
    path = path or LEXICONS / "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
    if not path.exists():
        raise FileNotFoundError(f"EmoLex missing at {path}; get it from https://saifmohammad.com/WebDocs/Lexicons/NRC-Emotion-Lexicon.zip")
    lex: dict[str, set[str]] = {}
    for line in path.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) == 3 and parts[2] == "1":
            lex.setdefault(parts[0], set()).add(parts[1])
    return lex


def _mattr(words: list[str], w: int = 100) -> float:
    """Mean type-token ratio over consecutive 100-word windows (plain TTR falls as people write more)."""
    chunks = [words[i : i + w] for i in range(0, len(words) - w + 1, w)]
    if not chunks:
        return np.nan
    return float(np.mean([len(set(c)) / w for c in chunks]))


def language_features(posts: list[str], lex: dict[str, set[str]]) -> dict[str, float]:
    text = "\n".join(posts).lower().replace("’", "'")
    words = WORD.findall(text)
    n = len(words)
    nposts = max(len(posts), 1)
    f = {"lang_log_words": np.log1p(n), "lang_words_per_post": n / nposts,
         "lang_question": text.count("?") / nposts, "lang_exclaim": text.count("!") / nposts}
    if n == 0:
        return f
    c = Counter(words)
    rate = lambda vocab: sum(c[w] for w in vocab) / n  # noqa: E731
    f["lang_i"] = rate(I_WORDS)
    f["lang_we"] = rate(WE_WORDS)
    f["lang_you"] = rate(YOU_WORDS)
    f["lang_negation"] = (sum(c[w] for w in NEGATION) + sum(v for w, v in c.items() if w.endswith("n't"))) / n
    f["lang_swear"] = rate(SWEAR)
    f["lang_ttr"] = _mattr(words)
    f["lang_word_len"] = sum(len(w) * k for w, k in c.items()) / n
    f["lang_tentative"] = rate(TENTATIVE) + (text.count("sort of") + text.count("kind of")) / n
    f["lang_certain"] = rate(CERTAIN)
    f["lang_tent_minus_cert"] = f["lang_tentative"] - f["lang_certain"]
    emo = Counter()
    for w, k in c.items():
        for e in lex.get(w, ()):
            emo[e] += k
    for e in EMOTIONS:
        f[f"lang_emo_{e}"] = emo[e] / n
    return f


def language_table(people: pd.DataFrame, posts_col: str = "posts") -> pd.DataFrame:
    lex = load_emolex()
    rows = [language_features(p, lex) for p in people[posts_col]]
    return pd.DataFrame(rows, index=people["person_id"]).reindex(columns=list(FEATURE_MEANING))


# ---------------------------------------------------------------- embeddings

def sample_posts(posts: list[str], k: int, rng: np.random.Generator) -> list[int]:
    idx = [i for i, p in enumerate(posts) if p.strip()]
    if len(idx) <= k:
        return idx
    return sorted(rng.choice(idx, k, replace=False).tolist())


class Embedder:
    """all-MiniLM-L6-v2 via fastembed (ONNX runtime, CPU, no torch). 384-dim, L2-normalised per post.

    Nothing per-post is written to disk (that would be ~540 MB); only per-person means are kept.
    """

    def __init__(self):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(EMB_MODEL, cache_dir=str(MODELS))

    def embed(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), np.float32)
        return np.array(list(self.model.embed(texts, batch_size=batch_size)), dtype=np.float32)


def embedding_tables(
    ids: list[str], variants: dict[str, list[list[str]]], embedder: Embedder, chunk: int = 400, log=print
) -> dict[str, pd.DataFrame]:
    """Mean post embedding per person, for several text variants of the same posts.

    variants maps a name to, per person, the list of (sampled) post texts. Identical
    texts across variants are embedded once (e.g. a post the leak stripper didn't touch).
    Processed in chunks of people so memory stays small.
    """
    names = list(variants)
    out = {v: np.full((len(ids), 384), np.nan, np.float32) for v in names}
    for s in range(0, len(ids), chunk):
        e = min(s + chunk, len(ids))
        uniq = sorted({t for v in names for ts in variants[v][s:e] for t in ts if t.strip()})
        vecs = embedder.embed(uniq)
        lookup = dict(zip(uniq, vecs))
        for v in names:
            for i in range(s, e):
                ts = [t for t in variants[v][i] if t.strip()]
                if ts:
                    out[v][i] = np.mean([lookup[t] for t in ts], axis=0)
        del vecs, lookup
        log(f"  embedded people {e}/{len(ids)} ({len(uniq)} unique posts in chunk)")
    cols = [f"emb_{j:03d}" for j in range(384)]
    return {v: pd.DataFrame(out[v], index=pd.Index(ids, name="person_id"), columns=cols) for v in names}


# ---------------------------------------------------------------- reference: bag of words

def bow_auc(docs: list[str], y: pd.Series, folds: int = 5, seed: int = 0) -> float:
    """Out-of-fold AUC of TF-IDF word counts + logistic regression (the classic method on this
    dataset). Not one of our two families; a yardstick, and a probe for leaks. The vocabulary is
    learned inside each training fold, C is fixed (no tuning)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline

    X = np.array(docs, dtype=object)
    yy = y.to_numpy().astype(int)
    pred = np.zeros(len(yy))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, yy):
        m = make_pipeline(TfidfVectorizer(min_df=20, max_features=30000, sublinear_tf=True), LogisticRegression(C=1.0, max_iter=2000))
        pred[te] = m.fit(X[tr], yy[tr]).predict_proba(X[te])[:, 1]
    return float(roc_auc_score(yy, pred))


# ---------------------------------------------------------------- scoring helper

def auc_ci(auc: float, n_pos: int, n_neg: int) -> tuple[float, float]:
    """95% CI for an AUC, Hanley & McNeil (1982). classify() returns no CI, so we add one."""
    q1, q2 = auc / (2 - auc), 2 * auc**2 / (1 + auc)
    se = np.sqrt((auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2) + (n_neg - 1) * (q2 - auc**2)) / (n_pos * n_neg))
    return auc - 1.96 * se, auc + 1.96 * se
