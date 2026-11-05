"""The derived feature record: the ONLY thing about a person that may leave their computer.

    record = build(person, readers, word_rates_opt_in=False)   a JSON-serialisable dict
    validate(record)                                            raises ValueError unless it fits the schema
    print(preview(record))                                      plain-English list of what would be sent

Design (docs/study/PRIVACY.md section 2, docs/features.md field by field):

  - Every value is a number, a boolean, null, or a string from a closed list (an enum in
    data/feature_schema.json) or a tight pattern (a "YYYY-MM" month, a version, a hex hash).
    No field can hold free text, so no post, query, title, channel, artist or track can travel.
  - Nothing keeps order: histograms, shares, counts and model outputs only, so text cannot be
    rebuilt from the record.
  - Counts are bucketed, word totals rounded to 100, shares to 0.01, dates to months.
  - Reader Estimates are copied without their `evidence` strings (those quote artist names and
    give exact counts).
  - Topic shares come from a local keyword lexicon with ~20 non-sensitive categories; any item that
    touches a sensitive term is put in "other" and that routing is not counted anywhere.
  - The common-word block is opt-in, and only ever holds words from a fixed 2,000-word list.

Dependencies: the standard library only (numpy is not even needed here). Readers are passed in and
may use numpy/scipy/joblib themselves. See docs/features.md "Running in the browser" for Pyodide.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from .person import Estimate, Person, Reader

SCHEMA_VERSION = "1.0.0"
EXTRACTOR_VERSION = "0.1.0"
DATA = Path(__file__).resolve().parent / "data"
SCHEMA_FILE = DATA / "feature_schema.json"
HASHED_FILES = ("common_words.json", "topic_lexicon.json", "word_blocklist.json", "feature_schema.json")

# Sources as the record names them. TextItem.source values map 1:1; Watch(source="youtube") is
# "youtube_watch"; plays are "spotify". Anything else a future parser emits is ignored until the
# schema learns it (it is never passed through by name).
WRITING_SOURCES = ("reddit", "x")
SEARCH_SOURCES = ("google_search", "youtube_search")
ACTIVITY_SOURCES = ("reddit", "x", "google_search", "youtube_search", "youtube_watch", "spotify")
WRITING_KINDS = ("comment", "post", "tweet", "reply")

READER_TRAITS = {
    "text-bigfive": ("O", "C", "E", "A", "N"),
    "text-mbti": ("I/E", "N/S", "T/F", "J/P"),
    "music": ("new_share", "old_friends", "season_turnover", "clock_spread", "clock_regularity",
              "finish_share", "skip_seconds", "dial_explore", "dial_regular"),
}
READER_STATUS = ("ok", "insufficient_data", "error", "not_run")

BUCKETS = ((0, "0"), (1, "1-49"), (50, "50-199"), (200, "200-999"), (1000, "1000-4999"),
           (5000, "5000-19999"), (20000, "20000+"))
BUCKET_LABELS = tuple(b for _, b in BUCKETS)

MIN_TOPIC_ITEMS = 20      # fewer items than this: topic shares are null (too noisy to mean anything)
MIN_SHAPE_ITEMS = 20      # fewer dated items / queries / watches than this: histograms and shape stats are
                          # null, because with 1-3 items an hour + month can point at one public post
MIN_STYLE_ITEMS = 5       # fewer posts than this: no style numbers (one post's length is a fingerprint)
MIN_WORD_RATE_WORDS = 500  # PRIVACY.md: the word-rate block needs at least 500 words
TTR_WINDOW = 1000

FLOORS = {  # PROTOCOL.md section 4, checked locally
    "text": "at least 100 words of own writing (Reddit/X)",
    "google": "at least 300 searches or 200 YouTube watches, over at least 90 days",
    "spotify": "at least 2,000 plays over at least 90 days",
}
FLOOR_REASONS = ("ok", "no_data", "too_little", "span_too_short")

STYLE_KEYS = ("mean_words_per_item", "question_share", "exclamation_share", "url_share", "mention_share",
              "first_person_singular_per_1k", "first_person_plural_per_1k", "second_person_per_1k",
              "negation_per_1k", "profanity_per_1k", "emoji_per_1k", "caps_word_share", "long_word_share",
              "mean_word_length", "ttr_1000")
QUERY_KEYS = ("mean_words", "question_share", "near_me_share", "english_share", "repeat_share")
VIEW_KEYS = ("distinct_channel_ratio", "channel_entropy_bits", "top10_channel_share", "repeat_watch_share",
             "music_video_share", "no_channel_share")


# ---------------------------------------------------------------- fixed data

@lru_cache(maxsize=1)
def _words_file() -> dict:
    return json.loads((DATA / "common_words.json").read_text())


@lru_cache(maxsize=1)
def _lexicon() -> dict:
    return json.loads((DATA / "topic_lexicon.json").read_text())


@lru_cache(maxsize=1)
def _blocklist() -> dict:
    return json.loads((DATA / "word_blocklist.json").read_text())


def common_words() -> tuple[str, ...]:
    """The fixed opt-in word list, in frequency order."""
    return tuple(_words_file()["words"])


@lru_cache(maxsize=1)
def _english_ref() -> frozenset[str]:
    return frozenset(_words_file()["english_reference"])


def topic_categories() -> tuple[str, ...]:
    return tuple(_lexicon()["order"]) + ("other",)


@lru_cache(maxsize=1)
def _topic_index() -> tuple[dict[str, tuple[str, ...]], frozenset[str], int]:
    """keyword -> categories it votes for; the sensitive set (lexicon + LDNOOBW); longest n-gram."""
    lex = _lexicon()
    kw: dict[str, list[str]] = {}
    for cat in lex["order"]:
        for k in lex["categories"][cat]:
            k = " ".join(_tokens(k))
            if k and cat not in kw.setdefault(k, []):
                kw[k].append(cat)
    sens = {" ".join(_tokens(s)) for s in lex["sensitive"]} | {" ".join(_tokens(s)) for s in _blocklist()["ldnoobw"]}
    sens.discard("")
    longest = max(len(k.split()) for k in list(kw) + list(sens))
    return {k: tuple(v) for k, v in kw.items()}, frozenset(sens), min(longest, 5)


@lru_cache(maxsize=1)
def _profanity() -> frozenset[str]:
    return frozenset(w for w in _blocklist()["ldnoobw"] if " " not in w)


@lru_cache(maxsize=1)
def extractor_hash() -> str:
    """sha256 over this module and every data file the record depends on."""
    h = hashlib.sha256(Path(__file__).read_bytes())
    for name in HASHED_FILES:
        f = DATA / name
        h.update(name.encode())
        h.update(f.read_bytes() if f.exists() else b"")
    return h.hexdigest()


# ---------------------------------------------------------------- small helpers

def _r(x: Any, nd: int) -> float | None:
    """Round, mapping None / NaN / inf to None (JSON has no NaN)."""
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x):
        return None
    v = round(x, nd)
    return 0.0 if v == 0 else v


def _sig3(x: float) -> float:
    return 0.0 if x == 0 else float(f"{x:.3g}")


def bucket(n: int) -> str:
    out = BUCKETS[0][1]
    for lo, label in BUCKETS:
        if n >= lo:
            out = label
    return out


def _utc(d: datetime | None) -> datetime | None:
    if d is None:
        return None
    return d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d.astimezone(timezone.utc)


def _month(d: datetime) -> str:
    return f"{d.year:04d}-{d.month:02d}"


_URL = re.compile(r"(?:https?://|www\.)\S+", re.I)
_EMAIL = re.compile(r"\S+@\S+\.\S+")
_HANDLE = re.compile(r"(?<!\w)[@/]?u/[\w-]+|(?<!\w)@\w+", re.I)
_WORD = re.compile(r"[a-z]{2,}")         # the opt-in block's tokens: letters only, 2+ (as the readers)
_ANYWORD = re.compile(r"[A-Za-z']+")
_TOK = re.compile(r"[a-z0-9&]+")


def _tokens(s: str) -> list[str]:
    return _TOK.findall(s.lower())


def _scrub(text: str) -> str:
    """Remove URLs, emails and @handles / u/names before anything is counted."""
    return _HANDLE.sub(" ", _EMAIL.sub(" ", _URL.sub(" ", text)))


def _is_emoji(ch: str) -> bool:
    o = ord(ch)
    return 0x1F300 <= o <= 0x1FAFF or 0x2600 <= o <= 0x27BF or 0x1F000 <= o <= 0x1F2FF


# ---------------------------------------------------------------- blocks

def activity(whens: Iterable[datetime | None]) -> dict:
    """Count bucket, month range, span, share of days active, UTC hour and weekday shares."""
    whens = list(whens)
    dated = sorted(w for w in (_utc(x) for x in whens) if w is not None)
    out: dict[str, Any] = {"items": bucket(len(whens)), "first_month": None, "last_month": None,
                           "span_months": None, "active_day_share": None, "hours_utc": None, "weekdays": None}
    if not dated:
        return out
    a, b = dated[0], dated[-1]
    out.update(first_month=_month(a), last_month=_month(b),
               span_months=(b.year - a.year) * 12 + (b.month - a.month) + 1)
    if len(dated) < MIN_SHAPE_ITEMS:
        return out
    hours, days = [0] * 24, [0] * 7
    for w in dated:
        hours[w.hour] += 1
        days[w.weekday()] += 1
    n = len(dated)
    span_days = (b.date() - a.date()).days + 1
    out.update(
        active_day_share=_r(len({w.date() for w in dated}) / span_days, 2),
        hours_utc=[_r(h / n, 2) for h in hours],
        weekdays=[_r(d / n, 2) for d in days],
    )
    return out


_FPS = frozenset("i me my mine myself i'm im i've ive i'd id i'll".split())
_FPP = frozenset("we us our ours ourselves we're we've we'd we'll".split())
_SP = frozenset("you your yours yourself yourselves you're youre you've you'd you'll u ur".split())
_NEG = frozenset("no not never none nobody nothing neither nor nowhere cannot cant dont doesnt didnt isnt "
                 "arent wasnt werent wont wouldnt shouldnt couldnt havent hasnt hadnt".split())


def style(texts: list[str]) -> dict:
    """About 15 writing-style numbers over the person's own posts. URLs/handles removed first."""
    n_items = len(texts)
    if n_items < MIN_STYLE_ITEMS:
        return {k: None for k in STYLE_KEYS}
    words: list[str] = []
    q = ex = url = men = emo = 0
    for t in texts:
        url += bool(_URL.search(t))
        men += bool(_HANDLE.search(t))
        s = _scrub(t)
        q += "?" in s
        ex += "!" in s
        emo += sum(_is_emoji(c) for c in s)
        words += _ANYWORD.findall(s)
    nw = len(words)
    low = [w.lower().replace("’", "'") for w in words]
    flat = [w.replace("'", "") for w in low]
    per1k = (lambda c: _r(1000 * c / nw, 1)) if nw else (lambda c: None)
    neg = sum(w in _NEG or w.endswith("n't") for w in low)
    ttr = None
    if nw >= TTR_WINDOW:
        wins = [flat[i:i + TTR_WINDOW] for i in range(0, nw - TTR_WINDOW + 1, TTR_WINDOW)]
        ttr = _r(sum(len(set(w)) / TTR_WINDOW for w in wins) / len(wins), 3)
    letters = [len(w.replace("'", "")) for w in words]
    return {
        "mean_words_per_item": _r(nw / n_items, 1) if n_items else None,
        "question_share": _r(q / n_items, 2) if n_items else None,
        "exclamation_share": _r(ex / n_items, 2) if n_items else None,
        "url_share": _r(url / n_items, 2) if n_items else None,
        "mention_share": _r(men / n_items, 2) if n_items else None,
        "first_person_singular_per_1k": per1k(sum(w in _FPS for w in low)),
        "first_person_plural_per_1k": per1k(sum(w in _FPP for w in low)),
        "second_person_per_1k": per1k(sum(w in _SP for w in low)),
        "negation_per_1k": per1k(neg),
        "profanity_per_1k": per1k(sum(w in _profanity() for w in flat)),
        "emoji_per_1k": per1k(emo),
        "caps_word_share": _r(sum(len(w) >= 2 and w.isupper() for w in words) / nw, 3) if nw else None,
        "long_word_share": _r(sum(n >= 7 for n in letters) / nw, 3) if nw else None,
        "mean_word_length": _r(sum(letters) / nw, 2) if nw else None,
        "ttr_1000": ttr,
    }


_QWORDS = frozenset("how what why when where who which whose is are can does do did should will could would".split())


def query_form(queries: list[str]) -> dict | None:
    """Shape of search queries (the queries themselves are discarded)."""
    if len(queries) < MIN_SHAPE_ITEMS:
        return None
    ref = _english_ref()
    n = len(queries)
    lens, qs, near, eng = [], 0, 0, 0
    for s in queries:
        toks = _tokens(s)
        lens.append(len(s.split()))
        qs += bool(toks) and (toks[0] in _QWORDS or s.rstrip().endswith("?"))
        near += "near me" in " ".join(toks)
        alpha = [t for t in toks if t.isalpha()]
        eng += bool(alpha) and s.isascii() and sum(t in ref for t in alpha) >= len(alpha) / 2
    norm = Counter(" ".join(_tokens(s)) for s in queries)
    return {
        "mean_words": _r(sum(lens) / n, 1),
        "question_share": _r(qs / n, 2),
        "near_me_share": _r(near / n, 2),
        "english_share": _r(eng / n, 2),
        "repeat_share": _r((n - len(norm)) / n, 2),
    }


_MUSIC_VIDEO = re.compile(r"official\s+(music\s+)?video|official\s+audio|\blyrics?\b|\(audio\)|\bft\.|\bfeat\.|"
                          r"\bvisuali[sz]er\b|\bmv\b|\blive\s+(at|from|session)\b|\bacoustic\b|\bremix\b", re.I)


def view_form(watches: list) -> dict | None:
    """Shape of YouTube viewing. Channel names and titles are used here and then dropped."""
    n = len(watches)
    if n < MIN_SHAPE_ITEMS:
        return None
    chans = Counter(w.channel for w in watches if w.channel)
    with_ch = sum(chans.values())
    ent = 0.0
    for c in chans.values():
        p = c / with_ch
        ent -= p * math.log2(p)
    titles = Counter(w.title.strip().lower() for w in watches)
    return {
        "distinct_channel_ratio": _r(len(chans) / with_ch, 2) if with_ch else None,
        "channel_entropy_bits": _r(ent, 1) if with_ch else None,
        "top10_channel_share": _r(sum(c for _, c in chans.most_common(10)) / with_ch, 2) if with_ch else None,
        "repeat_watch_share": _r((n - len(titles)) / n, 2),
        "music_video_share": _r(sum(bool(_MUSIC_VIDEO.search(w.title or "")) for w in watches) / n, 2),
        "no_channel_share": _r((n - with_ch) / n, 2),
    }


def topic_of(text: str) -> dict[str, float]:
    """One item -> {category: weight}, weights summing to 1. Sensitive items -> {"other": 1}."""
    kw, sens, longest = _topic_index()
    toks = _tokens(text)
    grams = {" ".join(toks[i:i + k]) for k in range(1, longest + 1) for i in range(len(toks) - k + 1)}
    if grams & sens:
        return {"other": 1.0}
    votes: Counter = Counter()
    for g in grams:
        for cat in kw.get(g, ()):
            votes[cat] += 1
    if not votes:
        return {"other": 1.0}
    top = max(votes.values())
    best = [c for c, v in votes.items() if v == top]
    return {c: 1 / len(best) for c in best}


def topic_shares(texts: list[str]) -> dict | None:
    if len(texts) < MIN_TOPIC_ITEMS:
        return None
    tot = Counter({c: 0.0 for c in topic_categories()})
    for t in texts:
        tot.update(topic_of(t))
    return {c: _r(tot[c] / len(texts), 2) for c in topic_categories()}


def word_rates(texts: list[str]) -> tuple[dict | None, str]:
    """Per-1,000-word rates of the fixed list, 3 significant figures. URLs, emails, handles and
    numbers are removed first; the denominator is every letters-only token of 2+ letters."""
    counts: Counter = Counter()
    total = 0
    for t in texts:
        toks = _WORD.findall(_scrub(t).lower())
        total += len(toks)
        counts.update(toks)
    if total < MIN_WORD_RATE_WORDS:
        return None, "below_500_words"
    return {"tokens_rounded": int(round(total, -2)),
            "per_1k": {w: _sig3(1000 * counts[w] / total) for w in common_words()}}, "present"


# ---------------------------------------------------------------- readers

def _reader_version(r: Reader, name: str) -> str | None:
    card = getattr(r, "card", None)
    if isinstance(card, dict) and card.get("version"):
        return str(card["version"])
    v = getattr(r, "version", None)
    if v:
        return str(v)
    if name == "music":  # no model card: identify it by the norms it percentiles against
        h = hashlib.sha256()
        for f in ("music_norms.json", "music_dials.json"):
            p = DATA / f
            h.update(p.read_bytes() if p.exists() else b"")
        return "norms-" + h.hexdigest()[:12]
    return None


def _estimate(e: Estimate) -> dict:
    ok = bool(e.usable) and e.value is not None and math.isfinite(float(e.value))
    vk = e.validity_kind if e.validity_kind in ("r", "auc") else None
    return {
        "usable": ok,
        "reason": None if ok else "insufficient_data",
        "value": _r(e.value, 3) if ok else None,
        "low": _r(e.low, 3) if ok else None,
        "high": _r(e.high, 3) if ok else None,
        "interval_level": _r(e.interval_level, 2) if ok else None,
        "percentile": _r(e.percentile, 1) if ok else None,
        "validity": _r(e.validity, 3) if ok else None,
        "validity_kind": vk if ok else None,
        "reliability": _r(e.reliability, 3),
    }


def _raw_scores(r: Reader, person: Person) -> dict | None:
    """Text heads' raw scores before calibration (one number per trait), so the study can refit the
    calibration on ordinary people. Needs the reader's internals; None when they are not there."""
    m, vec = getattr(r, "m", None), getattr(r, "vec", None)
    if not isinstance(m, dict) or vec is None or "heads" not in m:
        return None
    from .readers.text import own_prose, tfidf   # reader-side code (scipy); imported only when used

    texts, _ = own_prose(person, getattr(r, "exclude_kinds", frozenset({"search"})))
    if not texts:
        return None
    X = tfidf(vec.transform(texts).sum(0), m["idf"])
    return {t: _r(float(h.raw(X)[0]), 4) for t, h in m["heads"].items()}


def run_reader(r: Reader, person: Person) -> tuple[str, dict]:
    name = getattr(r, "name", type(r).__name__)
    if name not in READER_TRAITS:
        raise ValueError(f"reader {name!r} is not in the feature schema; add it there (and bump "
                         f"SCHEMA_VERSION) before its output may leave the machine")
    out: dict[str, Any] = {"status": "ok", "version": _reader_version(r, name), "estimates": {}, "raw_scores": None}
    try:
        ok, _why = r.can_read(person)        # the reason text is NOT kept: it quotes counts
        if not ok:
            out["status"] = "insufficient_data"
            return name, out
        ests = r.read(person)
    except Exception:                        # exception messages can quote data: keep the code only
        out["status"] = "error"
        return name, out
    for e in ests:
        if e.trait not in READER_TRAITS[name]:
            raise ValueError(f"reader {name!r} returned trait {e.trait!r}, which the schema does not know")
        out["estimates"][e.trait] = _estimate(e)
    if name.startswith("text-"):
        try:
            out["raw_scores"] = _raw_scores(r, person)
        except Exception:
            out["raw_scores"] = None
    return name, out


# ---------------------------------------------------------------- build

def _floor(passed: bool, reason: str) -> dict:
    return {"passed": passed, "reason": "ok" if passed else reason}


def build(person: Person, readers: list[Reader] | tuple = (), *, word_rates_opt_in: bool = False) -> dict:
    """Person -> the feature record. Deterministic for a given person and readers."""
    writing = [t for t in person.texts if t.source in WRITING_SOURCES and t.kind != "search"]
    searches = {s: [t for t in person.texts if t.source == s] for s in SEARCH_SOURCES}
    yt = [w for w in person.watches if w.source == "youtube"]
    plays = person.plays

    whens = {
        "reddit": [t.when for t in writing if t.source == "reddit"],
        "x": [t.when for t in writing if t.source == "x"],
        "google_search": [t.when for t in searches["google_search"]],
        "youtube_search": [t.when for t in searches["youtube_search"]],
        "youtube_watch": [w.when for w in yt],
        "spotify": [p.when for p in plays],
    }
    act = {s: activity(whens[s]) for s in ACTIVITY_SOURCES}

    wtexts = [t.text or "" for t in writing]
    n_words = sum(len(_scrub(t).split()) for t in wtexts)
    kinds = Counter(t.kind if t.kind in WRITING_KINDS else "other" for t in writing)

    def span_days(ws):
        d = [x for x in (_utc(w) for w in ws) if x]
        return (max(d) - min(d)).days if d else 0

    g_n, g_w = len(searches["google_search"]), len(yt)
    g_span = span_days(whens["google_search"] + whens["youtube_watch"] + whens["youtube_search"])
    s_span = span_days(whens["spotify"])
    floors = {
        "text": _floor(n_words >= 100, "no_data" if not writing else "too_little"),
        "google": _floor((g_n >= 300 or g_w >= 200) and g_span >= 90,
                         "no_data" if not (g_n or g_w or searches["youtube_search"]) else
                         ("too_little" if not (g_n >= 300 or g_w >= 200) else "span_too_short")),
        "spotify": _floor(len(plays) >= 2000 and s_span >= 90,
                          "no_data" if not plays else ("too_little" if len(plays) < 2000 else "span_too_short")),
    }

    extended = any(p.reason_end is not None or p.shuffle is not None for p in plays)
    shuf = [p.shuffle for p in plays if p.shuffle is not None]

    reader_out = {name: {"status": "not_run", "version": None, "estimates": {}, "raw_scores": None}
                  for name in READER_TRAITS}
    for r in readers:
        name, res = run_reader(r, person)
        reader_out[name] = res

    rec: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "extractor": {
            "version": EXTRACTOR_VERSION,
            "hash": extractor_hash(),
            "word_list_version": _words_file()["version"],
            "topic_lexicon_version": _lexicon()["version"],
        },
        "models": [{"reader": n, "version": reader_out[n]["version"]}
                   for n in READER_TRAITS if reader_out[n]["status"] != "not_run" and reader_out[n]["version"]],
        "consent": {"word_rates": bool(word_rates_opt_in)},
        "sources": {s: bool(whens[s]) for s in ACTIVITY_SOURCES},
        "floors": floors,
        "readers": reader_out,
        "activity": act,
        "writing": None if not writing else {
            "words_rounded": int(round(n_words, -2)),
            "items_by_kind": {k: bucket(kinds.get(k, 0)) for k in WRITING_KINDS + ("other",)},
            "style": style(wtexts),
        },
        "search_form": {s: query_form([t.text or "" for t in searches[s]]) for s in SEARCH_SOURCES},
        "viewing_form": view_form(yt),
        "topics": {
            "google_search": topic_shares([t.text or "" for t in searches["google_search"]]),
            "youtube_search": topic_shares([t.text or "" for t in searches["youtube_search"]]),
            "youtube_watch": topic_shares([w.title or "" for w in yt]),
        },
        "spotify": None if not plays else {
            "export": "extended" if extended else "basic",
            "shuffle_share": _r(sum(shuf) / len(shuf), 2) if shuf else None,
        },
        "word_rates_status": "absent_no_consent",
    }
    if word_rates_opt_in:
        block, status = word_rates(wtexts)
        rec["word_rates_status"] = status
        if block is not None:
            rec["word_rates"] = block
    return rec


# ---------------------------------------------------------------- validation (a small JSON Schema subset)

@lru_cache(maxsize=1)
def schema() -> dict:
    return json.loads(SCHEMA_FILE.read_text())


_TYPES = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v),
    "null": lambda v: v is None,
}


def _check(v: Any, s: dict, path: str, root: dict) -> None:
    if "$ref" in s:
        ref = s["$ref"]
        assert ref.startswith("#/$defs/"), ref
        s = root["$defs"][ref[len("#/$defs/"):]]
    t = s.get("type")
    if t is not None:
        ts = t if isinstance(t, list) else [t]
        if not any(_TYPES[x](v) for x in ts):
            raise ValueError(f"{path}: expected {t}, got {type(v).__name__}")
    if "const" in s and v != s["const"]:
        raise ValueError(f"{path}: must be {s['const']!r}")
    if "enum" in s and v not in s["enum"]:
        raise ValueError(f"{path}: {v!r} is not an allowed value")
    if isinstance(v, str):
        if "maxLength" in s and len(v) > s["maxLength"]:
            raise ValueError(f"{path}: string too long")
        if "pattern" in s and not re.search(s["pattern"], v):
            raise ValueError(f"{path}: {v!r} does not match {s['pattern']}")
    if _TYPES["number"](v):
        if "minimum" in s and v < s["minimum"]:
            raise ValueError(f"{path}: {v} < {s['minimum']}")
        if "maximum" in s and v > s["maximum"]:
            raise ValueError(f"{path}: {v} > {s['maximum']}")
    if isinstance(v, list):
        if "minItems" in s and len(v) < s["minItems"]:
            raise ValueError(f"{path}: too few items")
        if "maxItems" in s and len(v) > s["maxItems"]:
            raise ValueError(f"{path}: too many items")
        if "items" in s:
            for i, x in enumerate(v):
                _check(x, s["items"], f"{path}[{i}]", root)
    if isinstance(v, dict):
        props = s.get("properties", {})
        for k in s.get("required", []):
            if k not in v:
                raise ValueError(f"{path}: missing {k!r}")
        names = s.get("propertyNames")
        for k, x in v.items():
            if names is not None:
                _check(k, names, f"{path}.<key>", root)
            if k in props:
                _check(x, props[k], f"{path}.{k}", root)
            else:
                ap = s.get("additionalProperties", True)
                if ap is False:
                    raise ValueError(f"{path}: field {k!r} is not in the schema")
                if isinstance(ap, dict):
                    _check(x, ap, f"{path}.{k}", root)


def validate(record: dict) -> None:
    """Raise ValueError if the record has anything the schema does not allow (extra fields, free
    text, wrong types, NaN, out-of-range numbers, words not on the list)."""
    s = schema()
    _check(record, s, "$", s)
    # belt and braces: the JSON must round-trip exactly (no NaN, no non-JSON types)
    json.loads(json.dumps(record, allow_nan=False))


# ---------------------------------------------------------------- preview

_NICE = {"reddit": "Reddit", "x": "X", "google_search": "Google searches", "youtube_search": "YouTube searches",
         "youtube_watch": "YouTube watches", "spotify": "Spotify plays"}


def _top_hours(hours: list[float]) -> str:
    best = max(range(24), key=lambda h: sum(hours[(h + i) % 24] for i in range(4)))
    return f"{best:02d}:00-{(best + 4) % 24:02d}:00 UTC"


def preview(rec: dict) -> str:
    """Plain-English description of everything in the record (what the participant reviews)."""
    L = ["WHAT WOULD BE SENT", "=" * 60,
         f"Extractor {rec['extractor']['version']} (hash {rec['extractor']['hash'][:12]}...), schema {rec['schema_version']}",
         "No posts, searches, titles, channel names, artists, tracks or usernames are in it.", ""]
    L.append("Sources and activity (counts are ranges; dates are months; times are UTC):")
    for s in ACTIVITY_SOURCES:
        a = rec["activity"][s]
        if not rec["sources"][s]:
            L.append(f"  {_NICE[s]:<17} not included")
            continue
        line = f"  {_NICE[s]:<17} {a['items']} items"
        if a["first_month"]:
            line += f", {a['first_month']} to {a['last_month']} ({a['span_months']} months)"
        if a["hours_utc"]:
            line += f", active on {round(100 * a['active_day_share'])}% of days, busiest {_top_hours(a['hours_utc'])}"
        L.append(line)
    L += ["", "Data floors:"] + [f"  {k:<8} {'passed' if v['passed'] else 'not met (' + v['reason'] + ')'}  [{FLOORS[k]}]"
                                 for k, v in rec["floors"].items()]
    w = rec["writing"]
    if w:
        L += ["", f"Writing: about {w['words_rounded']:,} words. Style numbers:"]
        L += [f"  {k:<30} {v}" for k, v in w["style"].items()]
    for s, q in rec["search_form"].items():
        if q:
            L += ["", f"{_NICE[s]} (shape only, queries discarded):"] + [f"  {k:<30} {v}" for k, v in q.items()]
    if rec["viewing_form"]:
        L += ["", "YouTube viewing (shape only, no titles or channel names):"]
        L += [f"  {k:<30} {v}" for k, v in rec["viewing_form"].items()]
    for s, t in rec["topics"].items():
        if t:
            parts = [f"{k} {round(100 * v)}%" for k, v in sorted(t.items(), key=lambda kv: -kv[1]) if v]
            L += ["", f"Broad topics of {_NICE[s]}: " + ", ".join(parts)]
    if rec["spotify"]:
        L += ["", f"Spotify export: {rec['spotify']['export']}; shuffle share {rec['spotify']['shuffle_share']}"]
    L += ["", "Model outputs (no explanations or examples are sent):"]
    for name, r in rec["readers"].items():
        if r["status"] != "ok":
            L.append(f"  {name}: {r['status'].replace('_', ' ')}")
            continue
        for t, e in r["estimates"].items():
            if e["usable"]:
                rng = f" [{e['low']}, {e['high']}]" if e["low"] is not None else ""
                pc = f", percentile {e['percentile']}" if e["percentile"] is not None else ""
                L.append(f"  {name:<13} {t:<17} {e['value']}{rng}{pc}")
            else:
                L.append(f"  {name:<13} {t:<17} not enough data")
        if r["raw_scores"]:
            L.append(f"  {name:<13} raw scores: " + ", ".join(f"{k} {v}" for k, v in r["raw_scores"].items()))
    L.append("")
    if "word_rates" in rec:
        top = sorted(rec["word_rates"]["per_1k"].items(), key=lambda kv: -kv[1])[:12]
        nz = sum(1 for v in rec["word_rates"]["per_1k"].values() if v)
        L.append(f"Common-word rates (you opted in): {len(rec['word_rates']['per_1k']):,} fixed common words, "
                 f"{nz:,} of which you used; e.g. " + ", ".join(f"'{k}' {v}" for k, v in top) + " per 1,000 words.")
    else:
        L.append(f"Common-word rates: not included ({rec['word_rates_status'].replace('_', ' ')}).")
    return "\n".join(L)
