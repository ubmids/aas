import numpy as np
import pandas as pd
import pytest

from aas import quiz
from aas.schema import TRAITS


def test_fifty_items_ten_per_trait():
    assert len(quiz.ITEMS) == 50
    assert all(len(quiz.TRAIT_ITEMS[t]) == 10 for t in TRAITS)
    assert sorted(quiz.PAGE_ORDER) == sorted(quiz.ITEMS)


def test_reverse_keying_matches_ipip_key():
    # spot checks straight from https://ipip.ori.org/newBigFive5broadKey.htm
    assert quiz.KEY["EXT1"] == ("E", 1)  # life of the party
    assert quiz.KEY["EXT2"] == ("E", -1)  # don't talk a lot
    assert quiz.KEY["AGR1"] == ("A", -1)  # little concern for others
    assert quiz.KEY["CSN2"] == ("C", -1)  # leave belongings around
    assert quiz.KEY["OPN2"] == ("O", -1)  # difficulty with abstract ideas
    assert quiz.KEY["OPN8"] == ("O", 1)  # use difficult words
    # EST is stability; N flips it: "stressed out easily" is +N, "relaxed" is -N
    assert quiz.KEY["EST1"] == ("N", 1)
    assert quiz.KEY["EST2"] == ("N", -1)
    assert quiz.KEY["EST4"] == ("N", -1)
    reversed_count = {t: sum(quiz.KEY[c][1] < 0 for c in quiz.TRAIT_ITEMS[t]) for t in TRAITS}
    assert reversed_count == {"O": 3, "C": 4, "E": 5, "A": 4, "N": 2}


def test_keyed_flips_only_reversed_items():
    df = pd.DataFrame([{c: 1 for c in quiz.ITEMS}])
    k = quiz.keyed(df).iloc[0]
    for c, (_, sign) in quiz.KEY.items():
        assert k[c] == (1 if sign > 0 else 5)


def test_all_threes_scores_neutral_point():
    norms = quiz.load_norms()
    res = quiz.score({c: 3 for c in quiz.ITEMS})
    for t in TRAITS:
        nt = norms["traits"][t]
        assert res[t]["score"] == 3.0
        assert res[t]["form"] == "full"
        # all-3s sits exactly at the neutral point; z is its distance from the population mean
        assert res[t]["z"] == pytest.approx((3 - nt["mean"]) / nt["sd"], abs=1e-3)
        assert abs(res[t]["z"]) < 2
    # the population is ~neutral on E and N, so "all 3s" is about average there...
    assert abs(res["E"]["z"]) < 0.2 and abs(res["N"]["z"]) < 0.2
    # ...but people describe themselves as open and agreeable, so neutral = below average
    assert res["O"]["z"] < -1 and res["A"]["z"] < -0.5


def test_population_average_sheet_scores_near_zero():
    # a sheet whose trait score lands on the population mean gets z ~ 0 and percentile ~ 50
    norms = quiz.load_norms()
    for t in TRAITS:
        nt = norms["traits"][t]
        items = list(nt["items"])
        target = round(nt["mean"] * len(items))  # closest achievable keyed sum
        keyed = [target // len(items)] * len(items)
        for i in range(target - sum(keyed)):
            keyed[i] += 1
        ans = {c: v if nt["items"][c] > 0 else 6 - v for c, v in zip(items, keyed)}
        r = quiz.score(ans)[t]
        assert abs(r["z"]) < 0.1
        assert 40 < r["percentile"] < 60


def test_percentile_monotonic_in_answers():
    for t in TRAITS:
        items = quiz.TRAIT_ITEMS[t]
        prev_p, prev_z = -1.0, -np.inf
        for level in range(1, 6):  # keyed level 1..5 on every item
            ans = {c: level if quiz.KEY[c][1] > 0 else 6 - level for c in items}
            r = quiz.score(ans)[t]
            assert r["percentile"] > prev_p and r["z"] > prev_z
            prev_p, prev_z = r["percentile"], r["z"]


def test_percentile_tables_monotonic():
    norms = quiz.load_norms()
    for t in TRAITS:
        for tab in (norms["traits"][t]["pct_by_sum"], norms["traits"][t]["short"]["pct_by_sum"]):
            v = np.array(tab["table"])
            assert np.all(np.diff(v) >= 0) and 0 <= v[0] and v[-1] <= 100


def test_short_form_and_partial_paths():
    sf = quiz.short_form()
    assert len(sf) == 20 and len(set(sf)) == 20
    res = quiz.score({c: 5 for c in sf})
    assert {r["form"] for r in res.values()} == {"short"}
    res = quiz.score({"EXT1": 5, "EXT2": 1})  # not the short form: partial path, E only
    assert list(res) == ["E"] and res["E"]["form"] == "partial" and res["E"]["z"] > 1


def test_bad_answers_rejected():
    with pytest.raises(ValueError):
        quiz.score({"EXT1": 6})
    with pytest.raises(KeyError):
        quiz.score({"XYZ1": 3})
