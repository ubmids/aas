"""scripts/study_analysis.py: exclusions, quiz rules, statistics helpers, recovery on a seeded simulated
study, Holm, determinism. The simulated study comes from scripts/simulate_study.py."""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import simulate_study as SS  # noqa: E402
import study_analysis as SA  # noqa: E402

from aas import quiz  # noqa: E402
from aas.schema import TRAITS  # noqa: E402


# ---------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def sim120(tmp_path_factory):
    d = tmp_path_factory.mktemp("sim120")
    SS.write_study(120, seed=7, out=d)
    return d


@pytest.fixture(scope="module")
def truth():
    return SS.population_truth(N=20000)


def _good_answers(rng):
    th = rng.standard_normal((1, 5))
    return dict(zip(SS.ITEMS, SS.answers_from(th, rng)[0].tolist()))


def _person(sid, rng, **kw):
    P = SA.Participant(sid=sid, session1_at=datetime(2027, 1, 1), answers=_good_answers(rng),
                       item_ms={c: 3000 for c in SS.ITEMS}, attention_n=3, has_payload=True,
                       extractor_version="0.1.0", extractor_hash="h", submitted_at=datetime(2027, 1, 5),
                       payload_key=sid, floors={"text": True, "google": True, "spotify": False})
    for k, v in kw.items():
        setattr(P, k, v)
    return P


# ---------------------------------------------------------------- exclusions

def test_each_exclusion_rule_hits_the_right_person():
    rng = np.random.default_rng(0)
    straight = _good_answers(rng)
    for c in quiz.PAGE_ORDER[5:21]:
        straight[c] = 2
    people = [
        _person("a_ok", rng),
        _person("b_pilot", rng, pilot=True),
        _person("c_att", rng, attention_failed=2),
        _person("c_att_one", rng, attention_failed=1),                     # one failure: kept
        _person("c_att_two_of_two", rng, attention_failed=2, attention_n=2),  # OEJTS skipped: still out
        _person("d_speed", rng, item_ms={c: 600 for c in SS.ITEMS}),
        _person("d_straight", rng, answers=straight),
        _person("d_same", rng, answers={c: 3 for c in SS.ITEMS}),
        _person("e_no", rng, use_answers=False),
        _person("f_nopayload", rng, has_payload=False),
        _person("g_words", rng, floors={"text": False, "google": True, "spotify": True}),
        _person("g_nosecond", rng, floors={"text": True, "google": False, "spotify": False}),
        _person("h_ext", rng, extractor_hash="other"),
        _person("i_bad", rng, problems=["P(I) 1.37 outside [0, 1]"]),
        _person("j_dup_first", rng, payload_key="same", submitted_at=datetime(2027, 1, 2)),
        _person("j_dup_later", rng, payload_key="same", submitted_at=datetime(2027, 1, 9)),
        _person("k_gone", rng, withdrawn=True),
    ]
    usable, flow, log = SA.apply_exclusions(people, {"extractors": {"0.1.0": "h"}})
    rule_of = {e["study_id"]: e["rule"] for e in log}
    assert rule_of == {"b_pilot": "1", "c_att": "2", "c_att_two_of_two": "2", "d_speed": "3", "d_straight": "3",
                       "d_same": "3", "e_no": "4", "f_nopayload": "5a", "g_words": "5b", "g_nosecond": "5b",
                       "h_ext": "6a", "i_bad": "6b", "j_dup_later": "6c", "k_gone": "7"}
    assert sorted(P.sid for P in usable) == ["a_ok", "c_att_one", "j_dup_first"]
    assert [r["rule"] for r in flow] == ["0", "1", "2", "3", "4", "5a", "5b", "6a", "6b", "6c", "7"]
    assert flow[-1]["remaining"] == 3 and sum(r["excluded"] for r in flow) == len(people) - 3


def test_duplicate_of_an_excluded_person_is_still_a_duplicate():
    rng = np.random.default_rng(1)
    people = [_person("x_orig", rng, payload_key="k", pilot=True, submitted_at=datetime(2027, 1, 1)),
              _person("y_copy", rng, payload_key="k", submitted_at=datetime(2027, 1, 3))]
    usable, _, log = SA.apply_exclusions(people, {"extractors": {"0.1.0": "h"}})
    assert usable == [] and {e["study_id"]: e["rule"] for e in log} == {"x_orig": "1", "y_copy": "6c"}


def test_simulated_study_gives_exactly_n_usable_and_logs_every_exclusion(sim120):
    people, meta = SA.load_study(sim120)
    usable, flow, log = SA.apply_exclusions(people, meta)
    assert len(usable) == 120
    assert len(log) == len(people) - 120 and all(e["reason"] for e in log)
    pilot = set(meta["pilot_ids"])
    assert {e["study_id"] for e in log if e["rule"] == "1"} == pilot
    assert {e["rule"] for e in log} <= {c for c, _ in SA.RULES}
    for code in ("2", "3", "5a"):
        assert any(e["rule"] == code for e in log), code


def test_impossible_values_are_found():
    p = {"text": {"words": 300, "estimates": [
        {"reader": "text-mbti", "trait": "I/E", "kind": "mbti", "value": 1.2, "validity": 0.7, "validity_kind": "auc"},
        {"reader": "text-bigfive", "trait": "O", "kind": "bigfive", "value": 0.5, "low": 0.6, "high": 1.0, "validity": 0.6, "validity_kind": "auc"}]},
         "google": {"days_active_share": 1.4, "topic_shares": {"a": 0.5, "b": 0.2}},
         "spotify": {"traits": {"new_share": {"value": 0.3, "percentile": 140}}}}
    bad = " | ".join(SA._payload_problems(p))
    for frag in ("outside [0, 1]", "outside its own interval", "days_active_share", "topic_shares", "percentile"):
        assert frag in bad


# ---------------------------------------------------------------- quiz

def test_quiz_quality_rules():
    rng = np.random.default_rng(2)
    a = _good_answers(rng)
    assert SA.quiz_quality(a, {c: 3000 for c in a}) is None
    assert "median" in SA.quiz_quality(a, {c: 900 for c in a})
    run = dict(a)
    for c in quiz.PAGE_ORDER[10:25]:
        run[c] = 5
    assert "in a row" in SA.quiz_quality(run, {c: 3000 for c in a})
    run14 = dict(a)
    for c in quiz.PAGE_ORDER[10:24]:
        run14[c] = 5
    run14[quiz.PAGE_ORDER[9]] = run14[quiz.PAGE_ORDER[24]] = 1
    assert SA.quiz_quality(run14, {c: 3000 for c in a}) is None


def test_trait_with_under_8_items_is_missing_only_for_that_trait():
    rng = np.random.default_rng(3)
    a = _good_answers(rng)
    o_items = [c for c in SS.ITEMS if quiz.KEY[c][0] == "O"]
    two, three = dict(a), dict(a)
    for c in o_items[:2]:
        two[c] = None
    for c in o_items[:3]:
        three[c] = None
    assert set(SA.quiz_z(two)) == set(TRAITS)
    assert set(SA.quiz_z(three)) == set(TRAITS) - {"O"}
    full = SA.quiz_z(a)
    assert full["C"] == quiz.score({c: v for c, v in a.items()})["C"]["z"]


# ---------------------------------------------------------------- statistics

def test_holm_values_and_family_flags():
    np.testing.assert_allclose(SA.holm(np.array([0.01, 0.04, 0.03, 0.5])), [0.04, 0.09, 0.09, 0.5])
    rows = [{"p": 0.01}, {"p": 0.04}, {"p": 0.03}, {"p": 0.5}, {"p": None}]
    SA.holm_into(rows)
    assert [r["supported"] for r in rows] == [True, False, False, False, False]
    assert rows[4]["p_holm"] != rows[4]["p_holm"]   # NaN: not part of the family


def test_paired_test_direction_and_disagreement_rule():
    rng = np.random.default_rng(4)
    d = rng.normal(0.3, 1, 200)
    out = SA.paired_test(d, 4000, np.random.default_rng(0))
    assert out["p_t"] < 0.01 and out["p_perm"] < 0.01 and out["p"] == out["p_t"]
    neg = SA.paired_test(-d, 4000, np.random.default_rng(0))
    assert neg["p"] > 0.9
    # constructed disagreement: mean just significant by t, not by a coarse permutation
    out2 = SA.paired_test(rng.normal(0.17, 1, 100), 50, np.random.default_rng(1))
    if (out2["p_t"] < 0.05) != (out2["p_perm"] < 0.05):
        assert out2["p"] == out2["p_perm"]


def test_ridge_matches_sklearn_and_picks_by_inner_cv():
    from sklearn.linear_model import Ridge
    rng = np.random.default_rng(5)
    X = rng.normal(size=(80, 6))
    X[rng.random(X.shape) < 0.05] = np.nan
    y = X[:, 0] * 0.5 + rng.normal(size=80)
    y = np.where(np.isnan(y), 0, y)
    Ztr, Zte = SA._prep(X[:60], X[60:])
    for a in (0.1, 10.0, 1000.0):
        mine = SA._ridge_path(Ztr, y[:60, None], Zte, np.array([a]))[0, :, 0]
        ref = Ridge(alpha=a).fit(Ztr, y[:60]).predict(Zte)
        np.testing.assert_allclose(mine, ref, atol=1e-8)
    pure_noise = SA.ridge_cv_predict(X[:60], rng.normal(size=60), X[60:], np.random.default_rng(0))
    assert np.std(pure_noise) < 0.3   # heavy penalty chosen when there is nothing to learn


def test_delong_icc_cluster_ols():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(6)
    y = rng.random(300) < 0.4
    s = rng.normal(size=300) + 0.8 * y
    auc, var = SA.delong_auc(s[y], s[~y])
    assert abs(auc - roc_auc_score(y, s)) < 1e-12 and 0 < var < 0.01
    a = rng.normal(size=50)
    assert SA.icc21(a, a) == pytest.approx(1.0)
    assert SA.icc21(a, a + 1.0) < 1.0           # absolute agreement penalises a shift
    X = np.c_[np.ones(100), rng.normal(size=100)]
    yy = X @ [1.0, 2.0] + rng.normal(size=100)
    beta, se, df = SA.cluster_ols(yy, X, np.arange(100) // 2)
    np.testing.assert_allclose(beta, np.linalg.lstsq(X, yy, rcond=None)[0])
    assert df == 49 and np.all(se > 0)


def test_pool_rule_names_dropped_traits_and_counts_partial_people():
    avail = np.ones((100, 5), bool)
    avail[:93, 0] = False          # O usable for 7 people
    avail[:3, 2] = False           # E missing for 3 (quiz items): still pooled, those 3 counted as partial
    pool = SA.pool_traits(avail, 100, 0.90)
    assert pool["traits"] == ["C", "E", "A", "N"] and list(pool["excluded"]) == ["O"]
    assert pool["n_by_trait"]["O"] == 7 and pool["people_with_some_pooled_traits"] == 3
    assert pool["people_with_all_pooled_traits"] == 97


def test_stratified_folds_spread_each_pattern():
    pat = np.r_[np.zeros(53), np.ones(31), 2 * np.ones(16)].astype(int)
    f = SA.stratified_folds(pat, 10, seed=0)
    for code in range(3):
        counts = np.bincount(f[pat == code], minlength=10)
        assert counts.max() - counts.min() <= 1
    assert np.array_equal(f, SA.stratified_folds(pat, 10, seed=0))


# ---------------------------------------------------------------- recovery and determinism

def test_recovers_known_truth_on_a_seeded_simulated_study(tmp_path, truth):
    """The analysis is correct if, on a world whose answers we know, its CIs contain the truth about as
    often as they claim. n = 250, fast settings (fewer repeats/resamples; same estimators)."""
    SS.write_study(250, seed=11, out=tmp_path)
    res = SA.analyse(tmp_path, SA.Config.fast())
    assert res["sample"]["n_usable"] == 250
    rows = SS.recovery(res, truth)
    strict = [r for r in rows if r["kind"] in ("ci", "ci90")]
    assert len(strict) >= 45
    hit = np.mean([r["truth_in_ci"] for r in strict])
    assert hit >= 0.85, SS.recovery_text(rows, 250)
    # the headline numbers individually
    h1 = {r["trait"]: r for r in res["H1"]["traits"]}
    assert abs(h1["N"]["r"] - truth["H1_r"]["N"]) < 0.15
    # openness: refused by the text reader for most people, so it is named, left out of every pooled test,
    # and still analysed per trait on the few who have it
    assert res["sample"]["text_reader_usable"]["O"] < 0.2 * 250
    for pooled in (res["H2"]["pool"]["traits"], res["H3a"]["pooled_traits"], res["H3b"]["pooled_traits"], res["H4"]["pooled_traits"]):
        assert pooled == ["C", "E", "A", "N"]
    assert "O" in res["H2"]["pool"]["excluded"] and h1["O"]["n"] == res["sample"]["text_reader_usable"]["O"]
    assert res["S7"]["bayes"]["pool"]["traits"] == TRAITS          # fused O exists (MBTI N/S)
    assert res["S7"]["bayes"]["pool_for_paired_error"] == ["C", "E", "A", "N"]
    s4o = next(r for r in res["S4"]["calibration_refit_all"]["traits"] if r["trait"] == "O")
    assert s4o["compared_with"].startswith("z = 0")
    assert abs(res["H3a"]["coverage"] - truth["H3a_coverage_pooled"]) < 0.04
    assert abs(res["H3b"]["coverage"] - 0.80) < 0.04          # conformal-style intervals hold their level
    assert res["S7"]["bayes"]["coverage_pooled"]["coverage"] == pytest.approx(truth["S7_bayes_coverage_pooled"], abs=0.04)
    # null sources are not "found": Spotify alone has R ~ 0.1, S1 should not be supported at n ~ 130
    assert sum(r.get("supported", False) for r in res["S1"]["spotify"]["traits"]) <= 1


def test_deterministic_given_seed(sim120, tmp_path):
    cfg = SA.Config.fast()
    a = json.dumps(SA.analyse(sim120, cfg), sort_keys=True)
    b = json.dumps(SA.analyse(sim120, cfg), sort_keys=True)
    assert a == b
    SS.write_study(120, seed=7, out=tmp_path)            # same seed, new folder: same data, same digest
    assert SA._digest(tmp_path) == SA._digest(sim120)
    c = json.loads(json.dumps(SA.analyse(sim120, SA.Config.fast(seed=1)), sort_keys=True))
    assert c["H1"]["traits"][1]["r"] == json.loads(a)["H1"]["traits"][1]["r"]      # r itself has no randomness (C)
    assert c["H1"]["traits"][1]["ci95"] != json.loads(a)["H1"]["traits"][1]["ci95"]  # bootstrap seed matters


def test_cli_writes_json_and_report(sim120, tmp_path):
    SA.main([str(sim120), "--fast", "--out", str(tmp_path), "--only", "H1,H3"])
    res = json.loads((tmp_path / "results.json").read_text())
    txt = (tmp_path / "report.txt").read_text()
    assert res["sample"]["n_usable"] == 120 and "H1" in res and "H3a" in res
    assert "EXCLUSIONS" in txt and "PREREG ISSUES" in txt and "P20" in txt


def test_loader_reads_an_aas_features_record():
    """The adapter accepts the real extractor output (aas.features.build) as well as the simulator's shape."""
    from datetime import timezone

    from aas import features
    from aas.person import Estimate, Person, TextItem, Watch

    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    texts = [TextItem(t0 + timedelta(days=i, hours=i % 24), "reddit", "comment",
                      "I really think we should go hiking today, it is great weather " * 2) for i in range(40)]
    texts += [TextItem(t0 + timedelta(days=i // 3, hours=i % 24), "google_search", "search", f"how to cook pasta {i}")
              for i in range(320)]
    watches = [Watch(t0 + timedelta(days=i // 2, hours=(i * 5) % 24), "youtube", f"video {i % 50}", f"chan{i % 13}")
               for i in range(250)]

    class R:
        def __init__(self, name, ests):
            self.name, self.ests = name, ests

        def can_read(self, p):
            return True, ""

        def read(self, p):
            return self.ests

    bf = R("text-bigfive", [Estimate("text-bigfive", t, "bigfive", 0.2, -1.0, 1.4, 58.0, 0.62, "auc", interval_level=0.8) for t in TRAITS])
    mb = R("text-mbti", [Estimate("text-mbti", a, "mbti", 0.6, 0.5, 0.7, None, 0.7, "auc", interval_level=0.8) for a in SA.MBTI_AXES])
    rec = features.build(Person(texts=texts, watches=watches, sources=["reddit", "google"]), [bf, mb])
    P = SA.Participant(sid="z")
    SA._parse_payload({"study_id": "z", "submitted_at": "2027-01-01T00:00:00", "payload": rec}, P)
    assert P.problems == []
    assert P.floors == {"text": True, "google": True, "spotify": False}
    assert P.extractor_version == features.EXTRACTOR_VERSION and P.extractor_hash == rec["extractor"]["hash"]
    assert P.platforms == ("reddit",) and P.words == rec["writing"]["words_rounded"]
    assert set(P.google) == set(SA.GOOGLE_DIRECT) and P.google["channel_entropy"] == rec["viewing_form"]["channel_entropy_bits"]
    assert len(P.estimates) == 9 and {e.reader for e in P.estimates} == {"text-bigfive", "text-mbti"}
    assert P.spotify is None and len(P.topics) >= 20
    rec["readers"]["text-mbti"]["estimates"]["I/E"]["value"] = 1.4      # impossible probability
    Q = SA.Participant(sid="q")
    SA._parse_payload({"study_id": "q", "payload": rec}, Q)
    assert Q.problems
