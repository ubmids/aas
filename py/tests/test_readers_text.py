"""Text readers: saved artifacts load and read a synthetic person; too-little-text path; leak stripping;
searches left out; long histories; and the PANDORA (continuous) training path on synthetic data."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aas.person import Person, TextItem
from aas.readers import text as R

MBTI_DIR = R.MODELS_DIR / "text_mbti"
B5_DIR = R.MODELS_DIR / "text_bigfive"
need_mbti = pytest.mark.skipif(not (MBTI_DIR / "model.joblib").exists(), reason="run scripts/train_text_readers.py first")
need_b5 = pytest.mark.skipif(not (B5_DIR / "model.joblib").exists(), reason="run scripts/train_text_readers.py first")

WORDS = ("i think we should go out tonight with friends because the party sounds fun and honestly i love meeting "
         "new people but sometimes i worry about work and my plans for the week the book i read was about science "
         "and ideas and the universe which i found really interesting so maybe you should read it too").split()


def synthetic_person(n_items: int = 40, words_per: int = 25, seed: int = 0, kind: str = "comment", source: str = "reddit") -> Person:
    rng = np.random.default_rng(seed)
    items = [TextItem(None, source, kind, " ".join(rng.choice(WORDS, words_per))) for _ in range(n_items)]
    return Person(texts=items, sources=[source])


def relabel(p: Person, source: str, kind: str) -> Person:
    """Same words, different kind of writing."""
    return Person(texts=[TextItem(t.when, source, kind, t.text) for t in p.texts], sources=[source])


def _check_common(e, reader: str):
    assert e.reader == reader and e.usable
    assert e.low <= e.value <= e.high
    assert e.validity_kind == "auc" and 0.5 < e.validity < 1
    assert e.interval_level == 0.80
    assert "words" in e.evidence


# ---------------------------------------------------------------- artifacts

@pytest.mark.parametrize("d", [MBTI_DIR, B5_DIR])
def test_artifacts_small_and_carded(d: Path):
    if not (d / "model.joblib").exists():
        pytest.skip("not trained")
    size = sum(f.stat().st_size for f in d.iterdir())
    assert size < 50e6, f"{d} is {size / 1e6:.1f} MB"
    card = json.loads((d / "card.json").read_text())
    for k in ("version", "trained", "data", "features", "selection", "test", "min_text"):
        assert k in card
    assert card["min_text"]["floor_words"] <= card["min_text"]["recommended_words"]
    assert card["data"]["n_people"] > 1000


@need_mbti
def test_mbti_shapes():
    r = R.TextMBTIReader()
    p = synthetic_person()
    ok, why = r.can_read(p)
    assert ok, why
    est = r.read(p)
    assert [e.trait for e in est] == ["I/E", "N/S", "T/F", "J/P"]
    for e in est:
        _check_common(e, "text-mbti")
        assert e.kind == "mbti" and 0 <= e.low <= e.high <= 1


@need_b5
def test_bigfive_shapes():
    r = R.TextBigFiveReader()
    est = r.read(synthetic_person(kind="essay"))      # essay-like: in domain, every trait answered
    assert [e.trait for e in est] == ["O", "C", "E", "A", "N"]
    for e in est:
        _check_common(e, "text-bigfive")
        assert e.kind == "bigfive" and math.isfinite(e.value) and 0 <= e.percentile <= 100
        # An honest 80% interval for one person's true z from text alone is wide (the model is weak).
        assert e.high - e.low > 1.5


# ---------------------------------------------------------------- too little text, searches

@need_mbti
@need_b5
@pytest.mark.parametrize("cls", [R.TextMBTIReader, R.TextBigFiveReader])
def test_too_little_text(cls):
    r = cls()
    p = Person(texts=[TextItem(None, "x", "tweet", "hello there")])
    ok, why = r.can_read(p)
    assert not ok and "too little" in why
    est = r.read(p)
    assert est and all(not e.usable for e in est) and all("too little" in e.evidence for e in est)


@need_mbti
def test_searches_left_out():
    r = R.TextMBTIReader()
    searches = synthetic_person(n_items=400, words_per=4, kind="search")
    ok, why = r.can_read(searches)
    assert not ok and "search" in why
    both = synthetic_person().merge(searches)
    a, b = r.read(synthetic_person()), r.read(both)
    assert [e.value for e in a] == [e.value for e in b]          # searches change nothing
    assert "searches left out" in b[0].evidence
    # ...unless the caller opts in
    r2 = R.TextMBTIReader(exclude_kinds=())
    assert r2.can_read(searches)[0]


# ---------------------------------------------------------------- leak stripping

def test_clean_strips_leaks():
    s = R.clean("As an INTJ (INxJ really) my Fi-dom is strong, 4w5, introvert, https://x.com/a and I scored 90th percentile on openness")
    low = s.lower()
    for leak in ("intj", "inxj", "fi", "4w5", "introvert", "http", "percentile", "openness"):
        assert leak not in low.split() and leak not in low, (leak, s)
    assert "strong" in low


@need_mbti
@need_b5
@pytest.mark.parametrize("cls", [R.TextMBTIReader, R.TextBigFiveReader])
def test_reader_applies_leak_stripping(cls):
    r = cls()
    plain = synthetic_person()
    leaky = Person(texts=[TextItem(t.when, t.source, t.kind, f"{t.text} INTJ Ni Te introverts 95th percentile openness ENFP's https://reddit.com/r/intj")
                          for t in plain.texts])
    a, b = r.read(plain), r.read(leaky)
    key = lambda est: [(e.usable, round(e.value, 9) if e.usable else None) for e in est]
    assert key(a) == key(b)


@need_b5
def test_long_history_is_chunked():
    r = R.TextBigFiveReader()
    p = synthetic_person(n_items=1500, words_per=30, kind="essay")
    est = r.read(p)
    assert all(e.usable and "chunks" in e.evidence for e in est)
    assert all(e.low < e.value < e.high for e in est)


# ---------------------------------------------------------------- PANDORA-style continuous path

def test_continuous_training_path(tmp_path):
    """Synthetic Reddit-like corpus with continuous z labels: --bigfive-data pandora uses this path."""
    from aas.readers import text_train as TT

    rng = np.random.default_rng(1)
    n = 300
    vocab = [f"w{i}" for i in range(300)]
    Y = pd.DataFrame(rng.standard_normal((n, 5)), columns=list("OCEAN"))
    items = []
    for i in range(n):
        # word choice leans on O and E so there is real signal to find
        logits = np.zeros(len(vocab))
        logits[:20] += 1.2 * Y.O[i]
        logits[20:40] += 1.2 * Y.E[i]
        pr = np.exp(logits) / np.exp(logits).sum()
        items.append([" ".join(rng.choice(vocab, 15, p=pr)) for _ in range(40)])
    c = TT.Corpus("synthetic", [f"s:{i}" for i in range(n)], items, Y, "continuous_z", 5, [25, 50, 100, 200, 400],
                  {"short": "synthetic", "source": "synthetic", "url": "", "licence": ""})
    card, _ = TT.train(c, tmp_path / "text_bigfive", "text-bigfive")
    assert card["target"] == "continuous_z"
    assert card["test"]["full"]["O"]["score"] > 0.3 and card["test"]["full"]["E"]["score"] > 0.3
    r = R.TextBigFiveReader(model_dir=tmp_path / "text_bigfive")
    p = Person(texts=[TextItem(None, "reddit", "comment", t) for t in items[0]])
    est = r.read(p)
    assert len(est) == 5 and all(e.validity_kind == "r" and e.low < e.value < e.high for e in est)


# ---------------------------------------------------------------- cross-domain helpers

def test_style_features_topic_free_and_fixed_length():
    from aas.readers import text_train as TT

    a = TT.style_features("I think we should go, but maybe not! What do you say?")
    b = TT.style_features("Quantum chromodynamics lattice gauge renormalisation.")
    assert a.shape == b.shape == (len(TT.style_names()),)
    names = TT.style_names()
    assert a[names.index("cat_pron_i")] > 0 and b[names.index("cat_pron_i")] == 0
    assert TT.style_features("").shape == a.shape


def test_domain_stable_terms_drop_domain_words():
    from aas.readers import text_train as TT

    docs = ["the cat and i went to class roommate"] * 20 + ["the dog and i went to the vlog subscribe"] * 20
    dom = np.array(["essays"] * 20 + ["vlogs"] * 20)
    terms = set(TT.domain_stable_terms(docs, dom, {"ngram_range": [1, 1]}))
    assert {"the", "and", "went"} <= terms
    assert not ({"roommate", "subscribe", "class", "vlog"} & terms)


def test_author_tokens_drop_author_unique_and_mentions():
    from aas.readers import text_train as TT

    items = [
        ["i think the plan is good, zorblax88 signing off", "the plan works for me"],
        ["i think the plan is fine", "thanks @Kestrel_Moon for the plan"],
        ["the plan is good i think", "i think so too"],
    ]
    drop = TT.author_tokens(items)
    assert "zorblax88" in drop          # only one author ever writes it
    assert "kestrel_moon" in drop       # an @mention handle
    assert not ({"the", "plan", "think", "is"} & drop)


def test_author_tokens_drop_one_author_dominated_token():
    from aas.readers import text_train as TT

    items = [["mirelle says hi mirelle out mirelle again"], ["hi there mirelle"], ["hi all"]]
    drop = TT.author_tokens(items)
    assert "mirelle" in drop and "hi" not in drop


def test_fit_vocab_removes_dropped_tokens_and_their_ngrams():
    from aas.readers import text_train as TT

    docs = ["the plan zorblax88 works"] * 3 + ["the plan works well"] * 3
    vocab, idf, vec = TT.fit_vocab(docs, {"ngram_range": [1, 2], "max_features": 1000}, 1, {"zorblax88"})
    assert not any("zorblax88" in k for k in vocab)
    assert {"the", "plan", "works", "the plan"} <= set(vocab)
    assert sorted(vocab.values()) == list(range(len(vocab))) and len(idf) == len(vocab)
    assert vec.transform(["the plan zorblax88"]).sum() == 3


def test_median_split_within_dataset():
    from aas.readers import text_train as TT

    z = pd.DataFrame({"O": [-1.0, 0.0, 1.0, 2.0]})
    assert TT.median_split(z).O.tolist() == [0, 0, 1, 1]


@need_b5
def test_bigfive_card_reports_held_out_domains():
    card = json.loads((B5_DIR / "card.json").read_text())
    hd = card.get("held_out_domains")
    if hd is None:
        pytest.skip("run scripts/text_crossdomain.py")
    for dom, v in hd["per_domain"].items():
        assert dom != "essays" and set(v["traits"]) == set("OCEAN")


# ---------------------------------------------------------------- validity by kind of writing (2026-11-01)

def test_kind_of_writing():
    k = R.kind_of_writing
    assert k(TextItem(None, "x", "tweet", ""), 10) == "tweets"
    assert k(TextItem(None, "x", "reply", ""), 500) == "tweets"           # X stays tweets however long
    assert k(TextItem(None, "reddit", "comment", ""), 40) == "other_prose"
    assert k(TextItem(None, "reddit", "post", ""), 350) == "essay_like"   # one long item of prose
    assert k(TextItem(None, "notes", "journal", ""), 50) == "essay_like"


@need_b5
def test_bigfive_card_validity_by_source_matches_measurements():
    card = json.loads((B5_DIR / "card.json").read_text())
    v = card["validity_by_source"]
    hod = card["held_out_domains"]["per_domain"]
    # openness is the one trait whose CI includes 0.5 on every held-out domain
    assert v["not_shown"] == ["O"]
    for t in "OCEAN":
        assert v["sources"]["tweets"]["traits"][t]["auc"] == hod["pan15"]["traits"][t]["auc"]
        lowest = min(hod[d]["traits"][t]["auc"] for d in ("pan15", "vlogs"))
        assert v["sources"]["other_prose"]["traits"][t]["auc"] == lowest
    assert v["sources"]["essay_like"]["in_domain"]


@need_b5
def test_bigfive_validity_follows_the_kind_of_writing():
    r = R.TextBigFiveReader()
    card = r.card
    base = synthetic_person(n_items=60, words_per=25)                     # 1,500 words: full-length lift
    essay = {e.trait: e for e in r.read(relabel(base, "journal", "essay"))}
    tweets = {e.trait: e for e in r.read(relabel(base, "x", "tweet"))}
    reddit = {e.trait: e for e in r.read(base)}
    for t in "CEAN":
        assert essay[t].validity == pytest.approx(r._curve(t, "auc", 1500), abs=1e-3)
        assert tweets[t].validity == pytest.approx(card["validity_by_source"]["sources"]["tweets"]["traits"][t]["auc"], abs=1e-3)
        assert reddit[t].validity == pytest.approx(card["validity_by_source"]["sources"]["other_prose"]["traits"][t]["auc"], abs=1e-3)
        assert tweets[t].validity < essay[t].validity
        # same words, lower accuracy: the estimate moves toward 0 and the interval gets wider
        assert abs(tweets[t].value) <= abs(essay[t].value) + 1e-9
        assert tweets[t].high - tweets[t].low >= essay[t].high - essay[t].low - 1e-9
        assert "tweets" in tweets[t].evidence
    assert "not clearly better than chance" in tweets["C"].evidence      # PAN CI for C includes 0.5
    assert "Reddit-style prose has not been measured" in reddit["N"].evidence


@need_b5
def test_openness_refused_off_domain():
    r = R.TextBigFiveReader()
    base = synthetic_person(n_items=60, words_per=25)
    for p in (base, relabel(base, "x", "tweet")):
        assert r.can_read(p)[0]                                           # enough text; only O is refused
        o = r.read(p)[0]
        assert o.trait == "O" and not o.usable and math.isnan(o.value)
        assert o.validity == 0.0 and o.validity_kind == "r" and o.low is None
        assert "not shown to work on this kind of writing" in o.evidence
    assert r.read(relabel(base, "journal", "essay"))[0].usable           # essays are its home ground
    # the fuser never uses a refused estimate
    from aas.fusion import readings_from_estimates

    rs, skipped = readings_from_estimates(r.read(base))
    assert "O" not in {x.trait for x in rs} and {x.trait for x in rs} == set("CEAN")


@need_b5
def test_mixed_sources_weighted_by_words():
    r = R.TextBigFiveReader()
    essay = synthetic_person(n_items=3, words_per=400, kind="essay", source="journal", seed=1)    # 1,200 words
    few_tweets = synthetic_person(n_items=40, words_per=20, kind="tweet", source="x", seed=2)    # 800 words
    many_tweets = synthetic_person(n_items=120, words_per=20, kind="tweet", source="x", seed=3)  # 2,400 words
    p = essay.merge(few_tweets)
    st = R.own_prose(p)[1]
    assert st["words_by_writing"] == {"essay_like": 1200, "tweets": 800}
    words = st["words"]
    for e in r.read(p):
        V = r.validity_mix(e.trait, min(words, r.card["features"]["long_text_cap_words"]), st)
        tw = r.card["validity_by_source"]["sources"]["tweets"]["traits"][e.trait]
        expect_tw = 0.5 if e.trait == "O" else tw["auc"]
        assert V["auc"] == pytest.approx(0.6 * V["auc_in"] + 0.4 * expect_tw, abs=1e-6)
        assert e.usable and e.validity == pytest.approx(V["auc"], abs=1e-4)
    # mostly tweets: openness no longer has enough essay-like words behind it
    o = r.read(essay.merge(many_tweets))[0]
    assert not o.usable and "33% of your words" in o.evidence


@need_mbti
def test_mbti_untested_off_forum_is_discounted():
    r = R.TextMBTIReader()
    p = synthetic_person(n_items=60, words_per=25)
    card = r.card["validity_by_source"]
    assert card["tested_off_forum_with_mbti_labels"] is False
    old = R.TextMBTIReader()
    old.vbs = None                                                        # in-domain only, as before 2026-11-01
    for e, e0 in zip(r.read(p), old.read(p)):
        assert e.validity < e0.validity                                   # off-forum accuracy below the forum's
        assert e.validity == pytest.approx(card["sources"]["other_prose"]["traits"][e.trait]["auc"], abs=0.02)
        assert abs(e.value - 0.5) <= abs(e0.value - 0.5) + 1e-9          # probability pulled toward 50%
        assert "Untested off the forum" in e.evidence


def test_shrink_p():
    assert R.shrink_p(0.8, 0.6, 0.6) == pytest.approx(0.8)
    assert R.shrink_p(0.8, 0.6, 0.0) == pytest.approx(0.5)
    assert 0.5 < R.shrink_p(0.8, 0.6, 0.3) < 0.8
    assert R.shrink_p(0.2, 0.6, 0.3) == pytest.approx(1 - R.shrink_p(0.8, 0.6, 0.3))


@need_b5
def test_bigfive_r_is_closed_form_of_stated_auc():
    """The r behind the expected z and the interval is auc_to_r(stated AUC), the same conversion fusion uses
    to read the estimate back, on essays (ratio 1) and on tweets (ratio < 1). With one bootstrap draw the
    bootstrap variance is 0, so the half-width is exactly Z80 * sqrt(1 - r^2)."""
    from aas.fusion import auc_to_r as fusion_auc_to_r

    r = R.TextBigFiveReader(boots=1)
    base = synthetic_person(n_items=60, words_per=25)
    for p in (relabel(base, "journal", "essay"), relabel(base, "x", "tweet")):
        for e in r.read(p):
            if not e.usable:
                continue
            rr = fusion_auc_to_r(e.validity)
            assert R.auc_to_r(e.validity) == pytest.approx(rr, abs=1e-12)
            assert (e.high - e.low) / 2 == pytest.approx(R.Z80 * math.sqrt(1 - rr * rr), abs=1e-9)
            assert f"about r = {rr:.2f}" in e.evidence
