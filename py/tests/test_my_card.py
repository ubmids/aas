"""Make my card and the recognition-test lineup, on synthetic exports only (the invented person in
aas/example.py and variants of it). No real person's data."""

from __future__ import annotations

import csv
import importlib.util
import json
import random
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from aas import example
from aas.parsers import listenbrainz as lb
from aas.person import Play
from aas.statements import card as C
from aas.statements import identity

SPEC = importlib.util.spec_from_file_location("my_card", Path(__file__).parents[1] / "scripts" / "my_card.py")
mc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mc)

START = datetime(2025, 11, 1, tzinfo=UTC)


def plays_for(seed: int, keep: float = 1.0, days: int = 365) -> list[Play]:
    ps = example._plays(START, days, seed)
    if keep < 1:
        rng = random.Random(seed)
        ps = [p for p in ps if rng.random() < keep]
    return ps


def spotify_rows(plays: list[Play]) -> list[dict]:
    return [{"ts": p.when.strftime("%Y-%m-%dT%H:%M:%SZ"), "ms_played": p.ms_played,
             "master_metadata_track_name": p.track, "master_metadata_album_artist_name": p.artist,
             "master_metadata_album_album_name": None, "reason_start": p.reason_start,
             "reason_end": p.reason_end, "skipped": p.skipped, "shuffle": p.shuffle, "conn_country": p.country}
            for p in plays]


def write_spotify_zip(path: Path, plays: list[Play]) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("Spotify Extended Streaming History/Streaming_History_Audio_2025-2026_0.json",
                   json.dumps(spotify_rows(plays)))
    return path


def lb_listens(plays: list[Play]) -> list[dict]:
    # ListenBrainz records start times; only listens of 30 s or more, like a scrobbler would
    return [{"listened_at": int((p.when - timedelta(milliseconds=p.ms_played)).timestamp()),
             "track_metadata": {"artist_name": p.artist, "track_name": p.track, "release_name": None,
                                "additional_info": {}}}
            for p in plays if p.ms_played >= 30_000]


def write_lb_zip(path: Path, plays: list[Play]) -> Path:
    rows = lb_listens(plays)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("listens/2025/11.jsonl", "\n".join(json.dumps(r) for r in rows[: len(rows) // 2]))
        z.writestr("listens/2026/1.jsonl", "\n".join(json.dumps(r) for r in rows[len(rows) // 2:]))
        z.writestr("feedback.jsonl", json.dumps({"recording_msid": "x", "score": 1}))
    return path


@pytest.fixture(scope="module")
def own_card():
    return C.make_card(example.synthetic_person())


# ------------------------------------------------------------------------------------- the card

def test_joint_rule_is_identity_rounds(own_card):
    assert C.joint_select is identity.joint_select
    assert len(own_card.lines) == 3


def test_card_lines_are_true_units_and_shares(own_card):
    for ln in own_card.lines:
        v = ln.value
        assert (v >= ln.th) if ln.direction == "hi" else (v <= ln.th)
        assert 0 <= ln.share <= 1 and 0 < ln.hold <= 1
    assert len({ln.family.key for ln in own_card.lines}) == 3
    groups = [ln.family.group for ln in own_card.lines]
    assert max(groups.count(g) for g in groups) <= 2
    text = "\n".join(C.render(own_card))
    assert text.count("True of ") == 3 and text.count("Yours: ") == 3
    assert "Explore or stay:" in text and "Old favourites:" in text
    # every own value carries a unit or the thing it counts
    for ln in own_card.lines:
        shown = ln.family.show(ln.value)
        assert not shown.replace(".", "").replace(",", "").isdigit()


def test_card_is_deterministic(own_card):
    again = C.make_card(example.synthetic_person())
    assert C.render(again) == C.render(own_card)


def test_reference_reproduces_real_shares():
    ref = C.reference()
    assert np.abs(ref.T1.mean(0) - ref.share).mean() < 0.01


def test_norms_are_aggregate_only():
    d = json.loads((C.DATA / "card_norms.json").read_text())
    assert set(d) == {"source", "n", "knots", "families", "corr"}
    for f in d["families"]:
        assert len(f["full"]) == 99 and len(f["half"]) == 99    # 1st..99th percentile, no extremes
        assert set(f) == {"key", "measured", "r", "full", "half", "grid", "pct_th", "true_hi", "true_lo"}


def test_barnum_lines_are_common_and_true(own_card):
    b = C.make_barnum(own_card)
    assert b.lines
    for ln in b.lines:
        assert 0.5 <= ln.share <= 0.9
        assert (ln.value >= ln.th) if ln.direction == "hi" else (ln.value <= ln.th)


def test_spotify_zip_gives_the_same_card(tmp_path, own_card):
    z = write_spotify_zip(tmp_path / "me.zip", example.synthetic_person().plays)
    c = C.make_card(mc.load_export(z))
    # Spotify timestamps are whole seconds, so values can move in the last digit; the sentences must not
    assert [ln.text for ln in c.lines] == [ln.text for ln in own_card.lines]
    assert np.allclose([ln.value for ln in c.lines], [ln.value for ln in own_card.lines], rtol=0.02)


def test_listenbrainz_exports(tmp_path):
    ps = plays_for(7)
    z = write_lb_zip(tmp_path / "lb.zip", ps)
    p = mc.load_export(z)
    assert len(p.plays) == sum(q.ms_played >= 30_000 for q in ps)
    assert all(q.ms_played is None for q in p.plays)
    one = tmp_path / "listens.json"
    one.write_text(json.dumps(lb_listens(ps)))
    assert len(mc.load_export(one).plays) == len(p.plays)
    c = C.make_card(p)
    assert len(c.lines) == 3


def test_listenbrainz_single_mbid_is_the_artist_key(tmp_path):
    rows = [{"listened_at": 1_700_000_000 + i, "track_metadata": {"artist_name": "Made Up", "track_name": "T",
             "additional_info": {"artist_mbids": ["abc"]}}} for i in range(3)]
    f = tmp_path / "x.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows))
    assert {p.artist for p in lb.parse(f).plays} == {"mbid:abc"}


def test_empty_export_is_refused(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit):
        mc.load_export(tmp_path / "empty")


# ------------------------------------------------------------------------------------- lineup

@pytest.mark.parametrize("n", [3, 4, 7, 10])
def test_assign_balanced(n):
    a = mc.assign(n, seed=3)
    assert [t[0] for t in a] == list(range(n))
    shown = np.zeros(n, int)
    for i, dec, order in a:
        assert i not in dec and len(set(dec)) == 2
        assert sorted(order) == sorted([i] + dec)
        for d in dec:
            shown[d] += 1
    assert (shown == 2).all()          # every card is a decoy exactly twice


def test_assign_shuffles_position():
    pos = [order.index(i) for s in range(60) for i, _, order in mc.assign(5, seed=s)]
    counts = np.bincount(pos, minlength=3)
    assert counts.min() > 60          # 300 placements, about 100 per letter
    assert mc.assign(6, 1) == mc.assign(6, 1) and mc.assign(6, 1) != mc.assign(6, 2)


def test_assign_needs_three():
    with pytest.raises(ValueError):
        mc.assign(2, 0)


@pytest.fixture(scope="module")
def lineup(tmp_path_factory):
    root = tmp_path_factory.mktemp("lineup")
    ex = root / "exports"
    ex.mkdir()
    people = {"ada": plays_for(11), "ben": plays_for(12, keep=0.5), "cy": plays_for(13, days=300),
              "dee": plays_for(14, keep=0.7)}
    write_spotify_zip(ex / "ada.zip", people["ada"])
    write_spotify_zip(ex / "ben.zip", people["ben"])
    write_lb_zip(ex / "cy.zip", people["cy"])
    (ex / "dee").mkdir()
    (ex / "dee" / "Streaming_History_Audio_2025_0.json").write_text(json.dumps(spotify_rows(people["dee"])))
    # eve keeps her export at home and sends only her card file
    home = tmp_path_factory.mktemp("eve_home")
    people["eve"] = plays_for(15, keep=0.6)
    write_spotify_zip(home / "eve_export.zip", people["eve"])
    mc.main([str(home / "eve_export.zip"), "--save-card", str(ex / "eve")])
    assert (ex / "eve.card.json").exists()
    before = {p for p in root.rglob("*")}
    out = root / "sheets"
    mc.run_lineup(ex, out, seed=4, barnum=True)
    return root, ex, out, people, before


def test_lineup_writes_only_sheets_and_key(lineup):
    root, ex, out, people, before = lineup
    after = {p for p in root.rglob("*")}
    new = after - before
    assert {p.name for p in new} == {"sheets", "answer_key.csv", "answers.csv", "sheet_ada.txt",
                                     "sheet_ben.txt", "sheet_cy.txt", "sheet_dee.txt", "sheet_eve.txt"}


def test_lineup_writes_no_raw_data(lineup):
    root, ex, out, people, _ = lineup
    names = {p.artist for ps in people.values() for p in ps} | {p.track for ps in people.values() for p in ps}
    blob = "\n".join(f.read_text() for f in out.iterdir()) + (ex / "eve.card.json").read_text()
    leaked = [n for n in names if n in blob]
    assert not leaked
    assert "Made from" not in blob and "plays over" not in blob    # no play counts or spans on sheets


def test_answer_key_matches_sheets(lineup):
    root, ex, out, people, _ = lineup
    key = list(csv.DictReader(open(out / "answer_key.csv")))
    assert len(key) == 5
    texts = {mc.label_of(p): mc.texts_for(p) for p in mc.exports_in(ex)}
    for r in key:
        assert r[r["own"]] == r["person"] and r["D"] == "barnum"
        assert len({r["A"], r["B"], r["C"]}) == 3
        sheet = (out / r["sheet"]).read_text()
        for L in "ABC":
            body = sheet.split(f"Card {L}\n\n", 1)[1]
            assert body.startswith("\n".join(texts[r[L]]["card"]))
        assert "Card D" in sheet
        d = sheet.split("Card D\n\n", 1)[1]
        assert d.startswith("\n".join(texts[r["person"]]["barnum"]))
    ans = list(csv.DictReader(open(out / "answers.csv")))
    assert [a["person"] for a in ans] == [r["person"] for r in key] and all(a["pick"] == "" for a in ans)


def test_score(tmp_path, lineup):
    _, _, out, _, _ = lineup
    key = list(csv.DictReader(open(out / "answer_key.csv")))
    with open(tmp_path / "answers.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["person", "pick", "rate_A", "rate_B", "rate_C", "rate_D"])
        for j, r in enumerate(key):
            pick = r["own"] if j < 3 else next(L for L in "ABC" if L != r["own"])
            rates = {L: (5 if L == r["own"] else 2) for L in "ABC"}
            w.writerow([r["person"], pick.lower(), rates["A"], rates["B"], rates["C"], 3])
    res = mc.score(tmp_path / "answers.csv", out / "answer_key.csv")
    assert res["people"] == 5 and res["picked_own"] == 3 and res["rate"] == 0.6
    assert res["meets_bar"] is False and 0 < res["binomial_p_vs_chance"] < 1
    assert res["mean_rating_own"] == 5 and res["mean_rating_others"] == 2 and res["mean_rating_barnum"] == 3


def test_card_file_holds_only_card_text(lineup):
    _, ex, _, _, _ = lineup
    d = json.loads((ex / "eve.card.json").read_text())
    assert set(d) == {"card", "barnum"} and all(isinstance(x, str) for x in d["card"] + d["barnum"])


def test_personal_card_text(own_card):
    t = mc.personal(own_card, show_all=True)
    assert t.startswith("Your listening card") and "Nothing left this computer" in t
    assert t.count("percentile") >= 50
