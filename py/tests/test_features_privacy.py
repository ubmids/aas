"""Privacy tests for aas.features: the record must never carry content.

A synthetic person is filled with canaries (made-up tokens, an email, URLs, phone numbers, names,
handles, a whole sentence, canary artists/tracks/titles/channels, even canary parser notes and a
reader whose evidence and error messages quote canaries). Then:

  - no canary, nor any piece of one, appears anywhere in json.dumps(record), keys included
  - every string in the record is a known label (schema key/enum/word list) or a month/version/hash
  - no string is longer than 40 characters (bar the one extractor hash)
  - the schema itself cannot hold free text (every string node is an enum or a tight pattern)
  - dates are months; nothing looks like a day or a timestamp
  - the record validates, and validate() rejects extras, free text, NaN and off-list words
  - the word-rate block is absent unless opted in, and then only has words from the fixed list
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from aas import features as F
from aas.person import Estimate, Person, Play, TextItem, Watch

PY = Path(__file__).resolve().parents[1]
MODELS = PY / "models"
HAVE_TEXT_MODELS = all((MODELS / d / "model.joblib").exists() for d in ("text_bigfive", "text_mbti"))

# ---------------------------------------------------------------- canaries

SENTENCE = "The purple walrus negotiated a treaty with seventeen lighthouse keepers on Tuesday."
CANARIES = [
    "zqxjvorbel", "plimfrazzet", "snorvaquint",                       # made-up tokens
    "jane.canary.doe@examplecanary.org", "examplecanary",              # email
    "https://canary-tracker.example/u/zzq123?id=998877", "canary-tracker", "zzq123", "998877",
    "www.quobbleshop.example", "quobbleshop",                          # urls
    "+44 7700 900461", "7700 900461", "900461", "(555) 013-7788", "013-7788",   # phones
    "Bartholomew Quiggleton", "Bartholomew", "Quiggleton",             # a name
    "u/QuiggyCanary42", "QuiggyCanary42", "@quiggy_canary", "quiggy_canary",  # handles
    SENTENCE, "purple walrus", "lighthouse keepers",                   # a sentence and pieces of it
    "Zzyzx Quorbletones", "Quorbletones", "Flimmerwax Serenade", "Flimmerwax",   # artist, track
    "Glorbnax Fizzlewhip", "Fizzlewhip", "Vortmund",                  # video title
    "CanaryChannelXyz", "Snickerblat Studios", "Snickerblat",          # channels
    "how to cure zorpleitis", "zorpleitis",                            # a search
    "canary_source_name", "canary-note-file.zip",                      # parser metadata
    "evidencecanary", "exceptioncanary",                               # reader evidence / errors
]

FILLER = ("i think we should go out tonight with friends because the party sounds fun and honestly i love "
          "meeting new people but sometimes i worry about work and my plans for the week the book i read was "
          "about science and ideas and the universe which i found really interesting so maybe you should read "
          "it too did you see the game last night it was amazing").split()


def _t(start: datetime, i: int, step_h: float) -> datetime:
    return start + timedelta(hours=step_h * i)


def canary_person(seed: int = 0) -> Person:
    import random

    rnd = random.Random(seed)
    t0 = datetime(2023, 3, 14, 9, 26, 53, tzinfo=timezone.utc)
    texts: list[TextItem] = []
    secret_bits = [
        "my email is jane.canary.doe@examplecanary.org",
        "see https://canary-tracker.example/u/zzq123?id=998877 and www.quobbleshop.example",
        "call me on +44 7700 900461 or (555) 013-7788",
        "Bartholomew Quiggleton told u/QuiggyCanary42 and @quiggy_canary about it",
        SENTENCE,
        "zqxjvorbel plimfrazzet snorvaquint zqxjvorbel",
    ]
    for i in range(160):
        words = " ".join(rnd.choice(FILLER) for _ in range(rnd.randint(15, 45)))
        body = f"{words}. {secret_bits[i % len(secret_bits)]}!" if i % 3 == 0 else words + "?"
        texts.append(TextItem(_t(t0, i, 13.7), "reddit", "comment" if i % 4 else "post", body))
    for i in range(80):
        body = " ".join(rnd.choice(FILLER) for _ in range(20)) + " " + secret_bits[i % len(secret_bits)]
        texts.append(TextItem(_t(t0, i, 29.1), "x", "reply" if i % 2 else "tweet", body))
    queries = ["how to cure zorpleitis", "Bartholomew Quiggleton phone number", "weather tomorrow",
               "best pizza recipe", "depression symptoms", "nba highlights", "Zzyzx Quorbletones tour",
               "flights to Vortmund", "jane.canary.doe@examplecanary.org", "how to fix a leaking tap"]
    for i in range(420):
        texts.append(TextItem(_t(t0, i, 7.3), "google_search", "search", queries[i % len(queries)]))
    for i in range(60):
        texts.append(TextItem(_t(t0, i, 41.0), "youtube_search", "search", queries[(i * 3) % len(queries)]))
    texts.append(TextItem(None, "canary_source_name", "comment", "zqxjvorbel from an unknown source"))
    watches = []
    titles = ["Glorbnax Fizzlewhip Live at Vortmund", "Zzyzx Quorbletones - Flimmerwax Serenade (Official Video)",
              "funny cat compilation", "How to bake bread", "Snickerblat Studios trailer"]
    for i in range(300):
        watches.append(Watch(_t(t0, i, 11.0), "youtube", titles[i % len(titles)],
                             ["CanaryChannelXyz", "Snickerblat Studios", None][i % 3]))
    plays = []
    artists = ["Zzyzx Quorbletones"] + [f"Quorbletones Tribute {k}" for k in range(60)]
    t = t0
    for i in range(4000):
        t += timedelta(minutes=rnd.choice([3, 4, 5, 60, 400, 900]))
        a = artists[0] if i % 7 == 0 else rnd.choice(artists[: 10 + i // 100])
        ms = rnd.choice([200_000, 180_000, 12_000, 5_000])
        plays.append(Play(t, a, "Flimmerwax Serenade" if i % 5 == 0 else f"Flimmerwax {i % 40}", ms,
                          skipped=ms < 30_000, reason_start="clickrow",
                          reason_end="trackdone" if ms > 30_000 else "fwdbtn", shuffle=bool(i % 2),
                          country="GB"))
    return Person(texts=texts, plays=plays, watches=watches,
                  sources=["reddit", "x", "google", "spotify", "canary_source_name"],
                  notes=["parsed canary-note-file.zip", "evidencecanary"])


class CanaryReader:
    """A reader in the schema ('music') whose evidence quotes canaries."""
    name = "music"

    def can_read(self, person):
        return True, "evidencecanary Zzyzx Quorbletones"

    def read(self, person):
        return [Estimate(reader="music", trait="new_share", kind="measured", value=0.25, percentile=40.0,
                         evidence="you loved Zzyzx Quorbletones and Flimmerwax Serenade evidencecanary",
                         reliability=0.75, interval_level=None)]


class BoomReader:
    name = "text-mbti"

    def can_read(self, person):
        return True, "fine"

    def read(self, person):
        raise RuntimeError("exceptioncanary while reading Bartholomew Quiggleton")


def real_readers():
    from aas.readers.music import MusicReader

    rs = [MusicReader()]
    if HAVE_TEXT_MODELS:
        from aas.readers.text import TextBigFiveReader, TextMBTIReader

        rs = [TextBigFiveReader(boots=50), TextMBTIReader(boots=50)] + rs
    return rs


@pytest.fixture(scope="module")
def person():
    return canary_person()


@pytest.fixture(scope="module")
def record(person):
    return F.build(person, real_readers(), word_rates_opt_in=True)


@pytest.fixture(scope="module")
def record_no_optin(person):
    return F.build(person, real_readers(), word_rates_opt_in=False)


# ---------------------------------------------------------------- walking helpers

def walk(x, path="$"):
    """Yield (path, kind, value) for every key and every leaf."""
    if isinstance(x, dict):
        for k, v in x.items():
            yield path, "key", k
            yield from walk(v, f"{path}.{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from walk(v, f"{path}[{i}]")
    else:
        yield path, "leaf", x


def strings(rec) -> list[tuple[str, str]]:
    return [(p, v) for p, kind, v in walk(rec) if isinstance(v, str)]


def input_strings(p: Person) -> list[str]:
    out = [t.text for t in p.texts] + [w.title for w in p.watches] + [w.channel for w in p.watches if w.channel]
    out += [x.artist for x in p.plays] + [x.track for x in p.plays] + p.notes + p.sources
    return [s for s in out if s]


# ---------------------------------------------------------------- the person really exercises every block

def test_synthetic_person_fills_every_block(record):
    assert all(record["sources"][s] for s in F.ACTIVITY_SOURCES)
    assert record["writing"] and record["viewing_form"] and record["spotify"]
    assert all(record["search_form"][s] for s in F.SEARCH_SOURCES)
    assert all(record["topics"][s] for s in ("google_search", "youtube_search", "youtube_watch"))
    assert "word_rates" in record
    assert record["readers"]["music"]["status"] == "ok"
    assert any(e["usable"] for e in record["readers"]["music"]["estimates"].values())
    if HAVE_TEXT_MODELS:
        for r in ("text-bigfive", "text-mbti"):
            assert record["readers"][r]["status"] == "ok"
            assert record["readers"][r]["raw_scores"]
    assert all(v["passed"] for v in record["floors"].values()), record["floors"]


# ---------------------------------------------------------------- canaries

@pytest.mark.parametrize("opt_in", [True, False])
def test_no_canary_anywhere(person, opt_in):
    readers = real_readers()
    rec = F.build(person, readers, word_rates_opt_in=opt_in)
    dumped = json.dumps(rec).lower()
    found = [c for c in CANARIES if c.lower() in dumped]
    assert not found, f"canaries leaked: {found}"
    # also every key and value separately (catches anything json.dumps escaping might hide)
    for p, kind, v in walk(rec):
        if isinstance(v, str):
            low = v.lower()
            assert not any(c.lower() in low for c in CANARIES), (p, v)
    # digits from the phone numbers and url id never appear as numbers either
    nums = {str(v) for _, _, v in walk(rec) if isinstance(v, (int, float)) and not isinstance(v, bool)}
    assert not nums & {"900461", "998877", "7700900461", "5550137788", "0137788"}


def test_canary_reader_evidence_and_errors_dropped(person):
    rec = F.build(person, [CanaryReader(), BoomReader()])
    dumped = json.dumps(rec).lower()
    assert "evidencecanary" not in dumped and "exceptioncanary" not in dumped
    assert "quorbletones" not in dumped and "quiggleton" not in dumped
    assert rec["readers"]["music"]["estimates"]["new_share"]["value"] == 0.25
    assert rec["readers"]["text-mbti"]["status"] == "error"
    F.validate(rec)


def test_no_string_longer_than_40_and_no_input_window(record, person):
    h = F.extractor_hash()
    for p, s in strings(record):
        assert len(s) <= 40 or (s == h and p == "$.extractor.hash"), (p, s)
    # direct check: no 41-character window of any input string occurs in the record
    dumped = json.dumps(record)
    seen = set()
    for s in input_strings(person):
        for i in range(0, max(1, len(s) - 40), 7):
            w = s[i:i + 41]
            if len(w) == 41 and w not in seen:
                seen.add(w)
                assert w not in dumped, w


# ---------------------------------------------------------------- whitelist of strings

MONTH = re.compile(r"^(19|20)\d\d-(0[1-9]|1[0-2])$")
VERSION = re.compile(r"^(\d+\.\d+\.\d+|norms-[0-9a-f]{12})$")
HASH = re.compile(r"^[0-9a-f]{64}$")


def schema_labels(s) -> set[str]:
    """Every property name, enum value and const in the schema (the closed vocabulary)."""
    out: set[str] = set()
    if isinstance(s, dict):
        out |= set(s.get("properties", {}))
        for e in s.get("enum", []) or []:
            if isinstance(e, str):
                out.add(e)
        if isinstance(s.get("const"), str):
            out.add(s["const"])
        for v in s.values():
            out |= schema_labels(v)
    elif isinstance(s, list):
        for v in s:
            out |= schema_labels(v)
    return out


def test_every_string_is_a_known_label(record):
    labels = schema_labels(F.schema()) - {F.schema()["$schema"], F.schema()["$id"]}
    for p, s in strings(record):
        ok = s in labels or MONTH.match(s) or VERSION.match(s) or (HASH.match(s) and p == "$.extractor.hash")
        assert ok, f"{p}: {s!r} is not a closed-list label"


def test_schema_cannot_hold_free_text():
    """Every place a string may go is an enum/const or an anchored short pattern; every object is closed."""
    s = F.schema()

    def check(node, path):
        if isinstance(node, dict):
            t = node.get("type")
            ts = t if isinstance(t, list) else [t] if t else []
            if "string" in ts:
                assert "enum" in node or "const" in node or (
                    node.get("pattern", "").startswith("^") and node["pattern"].endswith("$")
                    and node.get("maxLength", 999) <= 64), path
            if "object" in ts:
                ap = node.get("additionalProperties")
                assert ap is False or (isinstance(ap, dict) and "propertyNames" in node
                                       and "enum" in node["propertyNames"] and ap.get("type") == "number"), path
            for k, v in node.items():
                if k not in ("enum", "required"):
                    check(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                check(v, f"{path}[{i}]")

    check(s, "$")


def test_no_free_text_fields_in_record(record):
    """No string value is long or sentence-like: free text would need spaces or be long."""
    for p, s in strings(record):
        assert " " not in s, (p, s)


# ---------------------------------------------------------------- dates

def test_dates_coarsened_to_months(record):
    dumped = json.dumps(record)
    assert not re.search(r"\d{4}-\d{2}-\d{2}", dumped)
    assert not re.search(r"\d{1,2}:\d{2}", dumped)
    assert not re.search(r"T\d\d", dumped)
    for s in F.ACTIVITY_SOURCES:
        a = record["activity"][s]
        assert MONTH.match(a["first_month"]) and MONTH.match(a["last_month"])
    # no number that could be an epoch timestamp (seconds or ms) or an exact count of anything big
    for p, kind, v in walk(record):
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            assert abs(v) < 1e7, (p, v)


def test_counts_bucketed_and_shares_rounded(record):
    for s in F.ACTIVITY_SOURCES:
        a = record["activity"][s]
        assert a["items"] in F.BUCKET_LABELS
        for v in a["hours_utc"] + a["weekdays"]:
            assert v == round(v, 2)
    assert record["writing"]["words_rounded"] % 100 == 0
    if "word_rates" in record:
        assert record["word_rates"]["tokens_rounded"] % 100 == 0


# ---------------------------------------------------------------- schema validation

def test_record_validates(record, record_no_optin):
    F.validate(record)
    F.validate(record_no_optin)


def _mut(rec, fn):
    r = json.loads(json.dumps(rec))
    fn(r)
    return r


@pytest.mark.parametrize("name,fn", [
    ("extra top-level field", lambda r: r.__setitem__("notes", ["parsed reddit.zip"])),
    ("extra nested field", lambda r: r["writing"]["style"].__setitem__("favourite_word", 1.0)),
    ("free text in an enum field", lambda r: r["activity"]["reddit"].__setitem__("items", "about 40 comments")),
    ("free text in a month field", lambda r: r["activity"]["reddit"].__setitem__("first_month", "2023-03-14T09:26")),
    ("text where a number goes", lambda r: r["writing"]["style"].__setitem__("ttr_1000", "high")),
    ("NaN", lambda r: r["writing"]["style"].__setitem__("ttr_1000", float("nan"))),
    ("share above 1", lambda r: r["activity"]["x"].__setitem__("active_day_share", 1.5)),
    ("25-hour histogram", lambda r: r["activity"]["x"]["hours_utc"].append(0.0)),
    ("evidence on an estimate", lambda r: r["readers"]["music"]["estimates"]["new_share"].__setitem__("evidence", "x")),
    ("unknown reader", lambda r: r["readers"].__setitem__("text-secret", {})),
    ("unknown trait", lambda r: r["readers"]["music"]["estimates"].__setitem__("artist_loyalty", {})),
    ("word not on the list", lambda r: r["word_rates"]["per_1k"].__setitem__("quiggleton", 1.0)),
    ("word missing from the list", lambda r: r["word_rates"]["per_1k"].pop("the")),
    ("wrong schema version", lambda r: r.__setitem__("schema_version", "0.9.0")),
    ("model version with text", lambda r: r["models"].append({"reader": "music", "version": "built from my files"})),
    ("unknown topic", lambda r: r["topics"]["youtube_watch"].__setitem__("health", 0.1)),
])
def test_validate_rejects(record, name, fn):
    with pytest.raises(ValueError):
        F.validate(_mut(record, fn))


def test_schema_file_is_current():
    spec = importlib.util.spec_from_file_location("build_feature_schema", PY / "scripts" / "build_feature_schema.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert F.SCHEMA_FILE.read_text() == mod.render(), "run scripts/build_feature_schema.py"


@pytest.mark.skipif(importlib.util.find_spec("jsonschema") is None, reason="jsonschema not installed")
def test_schema_agrees_with_jsonschema(record):  # pragma: no cover - optional cross-check
    import jsonschema

    jsonschema.validate(record, F.schema())


# ---------------------------------------------------------------- the word-rate block

def test_word_rates_absent_unless_opted_in(record_no_optin):
    assert "word_rates" not in record_no_optin
    assert record_no_optin["consent"]["word_rates"] is False
    assert record_no_optin["word_rates_status"] == "absent_no_consent"
    dumped = json.dumps(record_no_optin)
    assert "\"per_1k\"" not in dumped and "tokens_rounded" not in dumped


def test_word_rates_only_list_words(record):
    wr = record["word_rates"]["per_1k"]
    assert record["consent"]["word_rates"] is True and record["word_rates_status"] == "present"
    assert list(wr) == list(F.common_words())
    assert all(isinstance(v, float) and v >= 0 for v in wr.values())
    assert sum(1 for v in wr.values() if v) > 20     # filler words were really counted
    assert wr["think"] > 0                            # a filler word
    for v in wr.values():                             # 3 significant figures
        assert v == float(f"{v:.3g}")


def test_word_rates_need_500_words():
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    p = Person(texts=[TextItem(t0, "reddit", "comment", "short and sweet " * 10)] * 5)
    rec = F.build(p, [], word_rates_opt_in=True)
    assert "word_rates" not in rec and rec["word_rates_status"] == "below_500_words"
    F.validate(rec)


def test_word_list_is_clean():
    words = F.common_words()
    assert len(words) == len(set(words)) == 2000
    assert all(re.fullmatch(r"[a-z]{2,}", w) for w in words)
    bl = json.loads((F.DATA / "word_blocklist.json").read_text())
    blocked = {w for ws in bl["hand"].values() for w in ws} | set(bl["ldnoobw"])
    assert not set(words) & blocked
    for w in ("god", "gay", "cancer", "depression", "trump", "muslim", "london", "john", "fuck", "black", "debt",
              "husband", "prison", "google"):
        assert w not in words, w
    for w in ("the", "think", "really", "weekend", "i" if False else "you", "love", "never"):
        assert w in words, w
    names = Path("/usr/share/dict/propernames")
    if names.exists():
        spec = importlib.util.spec_from_file_location("bfw", PY / "scripts" / "build_feature_words.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        pn = {x.strip().lower() for x in names.read_text().splitlines()} - mod.NOT_NAMES
        assert not set(words) & pn


# ---------------------------------------------------------------- topics

def test_topics_exclude_sensitive_categories():
    cats = F.topic_categories()
    assert len(cats) == 21 and cats[-1] == "other"
    for bad in ("health", "religion", "sexuality", "politics", "ethnicity", "crime", "debt", "news"):
        assert not any(bad in c for c in cats)


@pytest.mark.parametrize("q,cat", [
    ("nba highlights lakers", "sport"), ("best pizza recipe", "food_drink"), ("minecraft lets play", "gaming"),
    ("weather tomorrow", "utilities"), ("puppy and kitten care", "pets_animals"),
    ("depression symptoms", "other"), ("church near me", "other"), ("election results", "other"),
    ("christian rock music", "other"), ("payday loan", "other"), ("zqxjvorbel", "other"),
])
def test_topic_assignment(q, cat):
    got = F.topic_of(q)
    assert got.get(cat, 0) == max(got.values()), got
    assert abs(sum(got.values()) - 1) < 1e-9


def test_sensitive_routing_is_not_counted(record):
    """Only the 21 shares exist; there is no 'sensitive' count that would itself be a signal."""
    for s, t in record["topics"].items():
        assert set(t) == set(F.topic_categories())
    assert "sensitive" not in json.dumps(record)


# ---------------------------------------------------------------- determinism, readers, misc

def test_deterministic(person):
    a = F.build(person, real_readers(), word_rates_opt_in=True)
    b = F.build(person, real_readers(), word_rates_opt_in=True)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_unknown_reader_refused(person):
    class Other:
        name = "text-secret-sauce"

        def can_read(self, p):
            return True, ""

        def read(self, p):
            return []

    with pytest.raises(ValueError):
        F.build(person, [Other()])


def test_unknown_trait_refused(person):
    class Odd(CanaryReader):
        def read(self, p):
            return [Estimate(reader="music", trait="favourite_artist", kind="measured", value=1.0)]

    with pytest.raises(ValueError):
        F.build(person, [Odd()])


def test_music_trait_list_matches_reader():
    from aas.readers.music import RETEST

    assert F.READER_TRAITS["music"] == tuple(RETEST) + ("dial_explore", "dial_regular")


def test_empty_person_validates():
    rec = F.build(Person(), [], word_rates_opt_in=True)
    F.validate(rec)
    assert not any(rec["sources"].values())
    assert rec["floors"]["text"] == {"passed": False, "reason": "no_data"}


def test_unusable_estimates_carry_no_numbers():
    class Thin(CanaryReader):
        def read(self, p):
            return [Estimate(reader="music", trait="old_friends", kind="measured", value=math.nan,
                             usable=False, evidence="Not enough data: evidencecanary")]

    rec = F.build(Person(), [Thin()])
    e = rec["readers"]["music"]["estimates"]["old_friends"]
    assert e["usable"] is False and e["reason"] == "insufficient_data" and e["value"] is None
    F.validate(rec)


def test_extractor_hash_covers_data():
    h = F.extractor_hash()
    assert HASH.match(h)
    assert set(F.HASHED_FILES) >= {"common_words.json", "topic_lexicon.json", "feature_schema.json"}


def test_preview_mentions_no_canaries(record):
    text = F.preview(record).lower()
    assert not [c for c in CANARIES if c.lower() in text]
    assert "what would be sent" in text


def test_fixture_exports_end_to_end():
    """Real parser output from the fixture exports: validates, and no input string of 4+ characters
    (title, channel, artist, track, query, post) shows up as a value or key in the record."""
    from aas import parsers

    fx = PY / "tests" / "fixtures" / "exports"
    p = parsers.load(*sorted(fx.iterdir()))
    rec = F.build(p, [], word_rates_opt_in=True)
    F.validate(rec)
    labels = schema_labels(F.schema()) | set(F.common_words())
    inputs = {s.strip().lower() for s in input_strings(p) if len(s.strip()) >= 4} - {x.lower() for x in labels}
    assert inputs
    for path, s in strings(rec):
        assert s.lower() not in inputs, (path, s)


def test_tiny_sources_give_no_shapes():
    """One or two posts: an exact hour + month could point at the public post, so no histogram."""
    t0 = datetime(2024, 5, 6, 21, 15, tzinfo=timezone.utc)
    p = Person(texts=[TextItem(t0, "reddit", "comment", "a lone comment " * 60),
                      TextItem(t0, "google_search", "search", "weather")],
               watches=[Watch(t0, "youtube", "one video", "one channel")])
    rec = F.build(p, [], word_rates_opt_in=True)
    F.validate(rec)
    for s in ("reddit", "google_search", "youtube_watch"):
        a = rec["activity"][s]
        assert a["hours_utc"] is None and a["weekdays"] is None and a["active_day_share"] is None
        assert a["first_month"] == "2024-05"
    assert all(v is None for v in rec["writing"]["style"].values())
    assert rec["search_form"]["google_search"] is None and rec["viewing_form"] is None


def test_features_module_needs_no_heavy_dependencies():
    """For the browser (Pyodide) build: aas.features itself imports only the standard library."""
    import subprocess
    import sys

    code = ("import sys, aas.features as F; from aas.person import Person; F.validate(F.build(Person(), [])); "
            "print(','.join(m for m in ('numpy','scipy','sklearn','pandas','joblib') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "", out
