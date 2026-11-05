"""text-distilled reader: a tiny fake artifact (bow, emb and mix recipes) read end to end with a fake embedder;
the floor and refusal paths; validity by kind of writing; the batch cache records match LLMRater's; and the saved
artifact, when present, is small, carded and readable."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

from aas.person import Person, TextItem
from aas.readers import distilled as Dm
from aas.readers import llm_rater as L
from aas.readers import text as R

WORDS = ("i think we should go out tonight with friends because the party sounds fun and honestly i love meeting "
         "new people but sometimes i worry about work and my plans for the week").split()
TRAITS = ["O", "C", "E", "A", "N"]


class FakeEmbedder:
    def embed(self, texts, batch_size=16):
        out = []
        for t in texts:
            v = np.zeros(8, np.float32)
            for w in t.split():
                v[hash(w) % 8] += 1
            out.append(v / (np.linalg.norm(v) or 1))
        return np.array(out)


def person(n_items=40, words_per=25, seed=0, source="reddit", kind="comment"):
    rng = np.random.default_rng(seed)
    return Person(texts=[TextItem(None, source, kind, " ".join(rng.choice(WORDS, words_per))) for _ in range(n_items)])


def make_artifact(d: Path, recipe: str, passed: bool = False, refused=()):
    from sklearn.feature_extraction.text import CountVectorizer

    rng = np.random.default_rng(0)
    vec = CountVectorizer(ngram_range=(1, 2)).fit([" ".join(WORDS)])
    V = len(vec.vocabulary_)
    m = {"recipe": recipe, "traits": TRAITS, "budget_tokens": 1500, "emb_chunk": 100, "emb_max_chunks": 40,
         "ref": {"mean": np.full(5, 3.0) if recipe != "mix" else np.zeros(5), "sd": np.full(5, 0.3) if recipe != "mix" else np.ones(5)}}
    if recipe in ("bow", "mix"):
        m["bow"] = {"vocab": {k: int(v) for k, v in vec.vocabulary_.items()}, "ngram_range": (1, 2),
                    "idf": np.ones(V), "coef": rng.normal(0, 0.5, (5, V)), "intercept": np.full(5, 3.0)}
    if recipe in ("emb", "mix"):
        m["emb"] = {"model": "fake", "mean": np.zeros(8), "scale": np.ones(8), "coef": rng.normal(0, 0.2, (5, 8)),
                    "intercept": np.full(5, 3.0)}
    if recipe == "mix":
        m["mix_ref"] = {k: {"mean": np.full(5, 3.0), "sd": np.full(5, 0.3)} for k in ("bow", "emb")}
    kinds = {"essay_like": {t: {"auc": 0.62} for t in TRAITS}, "tweets": {t: {"auc": 0.55} for t in TRAITS},
             "other_prose": {t: {"auc": 0.58} for t in TRAITS}}
    kinds["tweets"]["O"] = {"auc": 0.49}
    card = {"reader": "text-distilled", "traits": TRAITS, "min_text": {"floor_words": 50},
            "verdict": {"pass": passed}, "validity_by_source": {"kinds": kinds, "refused_traits": list(refused)}}
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(m, d / "model.joblib")
    (d / "card.json").write_text(json.dumps(card))
    Dm._load.cache_clear()
    return d


@pytest.mark.parametrize("recipe", ["bow", "emb", "mix"])
def test_reads_end_to_end(tmp_path, recipe):
    r = Dm.TextDistilledReader(make_artifact(tmp_path / recipe, recipe), embedder=FakeEmbedder())
    p = person()
    assert r.can_read(p)[0]
    est = r.read(p)
    assert [e.trait for e in est] == TRAITS
    for e in est:
        assert e.usable and e.kind == "bigfive" and e.reader == "text-distilled"
        assert e.low < e.value < e.high and np.isfinite(e.value)
        assert e.validity == pytest.approx(0.58) and e.validity_kind == "auc"
        assert "not the default" in e.evidence      # artifact marked as failing


def test_interval_matches_stated_auc(tmp_path):
    r = Dm.TextDistilledReader(make_artifact(tmp_path / "a", "bow"))
    e = r.read(person())[0]
    rr = R.auc_to_r(0.58)
    assert (e.high - e.low) / 2 == pytest.approx(R.Z80 * np.sqrt(1 - rr * rr))
    assert abs(e.value) <= rr * 10            # shrunk by r


def test_too_little_text(tmp_path):
    r = Dm.TextDistilledReader(make_artifact(tmp_path / "a", "bow"))
    p = person(n_items=2, words_per=10)
    ok, why = r.can_read(p)
    assert not ok and "50" in why
    assert all(not e.usable and np.isnan(e.value) for e in r.read(p))


def test_tweets_use_tweet_validity_and_refuse_at_chance(tmp_path):
    r = Dm.TextDistilledReader(make_artifact(tmp_path / "a", "bow"))
    est = {e.trait: e for e in r.read(person(source="x", kind="tweet"))}
    assert not est["O"].usable and "Not shown to work" in est["O"].evidence
    assert est["C"].usable and est["C"].validity == pytest.approx(0.55)


def test_refused_trait_in_card(tmp_path):
    r = Dm.TextDistilledReader(make_artifact(tmp_path / "a", "bow", refused=["N"]))
    est = {e.trait: e for e in r.read(person())}
    assert not est["N"].usable and est["A"].usable


def test_leak_words_do_not_change_reading(tmp_path):
    r = Dm.TextDistilledReader(make_artifact(tmp_path / "a", "bow", passed=True))
    p = person()
    p2 = Person(texts=[TextItem(None, "reddit", "comment", t.text + " INTJ openness") for t in p.texts])
    a, b = r.read(p), r.read(p2)
    assert [x.value for x in a] == pytest.approx([x.value for x in b])
    assert "not the default" not in a[0].evidence


def test_long_history_reads_several_samples():
    texts = [" ".join(np.random.default_rng(i).choice(WORDS, 30)) for i in range(400)]   # ~12k words, ~8 budgets
    s = Dm.budget_samples(texts, 1500, 5)
    assert len(s) == 5 and len(set(s)) > 1
    assert Dm.budget_samples(texts[:5], 1500, 5) == ["\n".join(texts[:5])]


def test_batch_cache_record_is_read_by_llmrater(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import distill

    class P:
        name, model, settings, max_tokens = "anthropic", "claude-haiku-4-5-20251001", {}, 600

    r = L.LLMRater(P(), cache_dir=tmp_path)
    raw = "\n".join(" ".join(WORDS) for _ in range(5))
    s, u, info = r.prepare(raw)
    k = r.key(s, u)
    parsed = {t: {"score": 3.5, "reason": "x"} for t in TRAITS}
    distill._write(r, k, info, [{"via": "batch"}], "{}", None, parsed)
    got = r.cached(raw)
    assert got.ok and got.scores == {t: 3.5 for t in TRAITS}


# ---------------------------------------------------------------- the saved artifact

need_model = pytest.mark.skipif(not (Dm.MODEL_DIR / "model.joblib").exists(), reason="run scripts/distill.py --evaluate first")


@need_model
def test_saved_artifact_small_and_carded():
    size = sum(f.stat().st_size for f in Dm.MODEL_DIR.iterdir())
    assert size < 50e6
    card = json.loads((Dm.MODEL_DIR / "card.json").read_text())
    for k in ("teacher", "data", "features", "fidelity_pool_cv", "test", "verdict", "validity_by_source", "interval", "min_text"):
        assert k in card
    assert set(card["validity_by_source"]["kinds"]) == {"essay_like", "tweets", "other_prose"}
    assert card["default"] is card["verdict"]["pass"] or card["default"] is False


@need_model
def test_saved_artifact_reads_a_person():
    Dm._load.cache_clear()
    est = Dm.TextDistilledReader().read(person(n_items=60))
    assert len(est) == 5
    assert all((e.usable and e.low <= e.value <= e.high) or not e.usable for e in est)
