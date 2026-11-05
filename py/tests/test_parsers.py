"""Parsers against tiny synthetic exports in tests/fixtures/exports (all made up)."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

from aas.parsers import detect, google, load, reddit, spotify, x
from aas.person import Person

FX = Path(__file__).parent / "fixtures" / "exports"


def as_zip(folder: Path, tmp_path: Path) -> Path:
    return Path(shutil.make_archive(str(tmp_path / folder.name), "zip", folder))


@pytest.fixture(params=["folder", "zip"])
def form(request, tmp_path):
    """Run a test once on the unzipped folder and once on a zip of it."""
    return lambda name: FX / name if request.param == "folder" else as_zip(FX / name, tmp_path)


def texts(p: Person, kind: str | None = None) -> list[str]:
    return [t.text for t in p.texts if kind is None or t.kind == kind]


# ---------------------------------------------------------------- reddit
def test_reddit(form):
    p = reddit.parse(form("reddit"))
    assert p.sources == ["reddit"]
    assert len(texts(p, "comment")) == 2                    # [deleted] dropped
    assert "Second paragraph, with \"quotes\"." in texts(p, "comment")[0]  # multi-line CSV cell
    assert "Made bread for the first time\n\nIt came out flat but tasty." in texts(p, "post")
    assert "A foggy morning" in texts(p, "post")            # title-only post
    assert p.texts[0].when == datetime(2023, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    assert all(t.source == "reddit" for t in p.texts)
    assert any("deleted" in n for n in p.notes)


# ---------------------------------------------------------------- x
def test_x_drops_retweets_and_links(form):
    p = x.parse(form("x"))
    all_text = " ".join(texts(p))
    assert "somebody else's" not in all_text               # RT dropped
    assert "t.co" not in all_text                           # links stripped
    assert "notes on sourdough & patience" in all_text     # html entities decoded
    assert texts(p, "reply") == ["@friend agreed, the second half drags"]
    assert "new year, new houseplant" in texts(p, "tweet")  # tweets-part1.js read
    assert "not mine" not in all_text                       # like.js ignored
    assert any("dropped 1 retweets" in n for n in p.notes)
    assert p.texts[0].when == datetime(2024, 1, 7, 7, 7, 7, tzinfo=timezone.utc)


def test_x_note_tweet_replaces_truncated_copy(form):
    p = x.parse(form("x"))
    cello = [t for t in texts(p) if "cello" in t]
    assert len(cello) == 1 and cello[0].endswith("patient with yourself.")


def test_x_old_tweet_js_without_wrapper(tmp_path):
    d = tmp_path / "old" / "data"
    d.mkdir(parents=True)
    (d / "tweet.js").write_text(
        'window.YTD.tweet.part0 = [ {"created_at": "Mon Jan 01 00:00:00 +0000 2018", '
        '"full_text": "hello from 2018 https://t.co/abc", "retweeted": false},'
        '{"created_at": "Mon Jan 01 00:00:01 +0000 2018", "full_text": "boosted", "retweeted": true} ]')
    p = x.parse(tmp_path / "old")
    assert texts(p) == ["hello from 2018"]


# ---------------------------------------------------------------- google
def test_google_json(form):
    p = google.parse(form("google"))
    searches = [t.text for t in p.texts if t.source == "google_search"]
    assert searches == ["cello für anfänger", "how long to proof bread dough"]  # sorted; "Visited" dropped
    assert [t.text for t in p.texts if t.source == "youtube_search"] == ["cello bow hold"]
    assert all(t.kind == "search" for t in p.texts)
    titles = [(w.title, w.channel) for w in p.watches]
    assert ("Beginner cello lesson 1", "Made Up Strings Channel") in titles
    assert not any("advert" in t for t, _ in titles)        # ad dropped
    assert not any("removed" in t for t, _ in titles)       # removed video dropped
    assert len(p.watches) == 2


def test_google_old_html(form):
    p = google.parse(form("google_html"))
    assert texts(p) == ["rainy day playlist", "tom & jerry"]
    assert p.texts[0].when == datetime(2020, 3, 4, 21, 5, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------- spotify
def test_spotify_extended_skips_podcasts(form):
    p = spotify.parse(form("spotify_extended"))
    assert len(p.plays) == 2                                # podcast row skipped
    first, second = p.plays
    assert (first.artist, first.track, first.ms_played) == ("Imaginary Band", "Imaginary Song", 210000)
    assert second.skipped is True and second.shuffle is True and second.reason_end == "fwdbtn"
    assert first.when.tzinfo is not None
    assert any("extended" in n for n in p.notes)
    assert any("skipped 1 podcast" in n for n in p.notes)


def test_spotify_basic(form):
    p = spotify.parse(form("spotify_basic"))
    assert [pl.track for pl in p.plays] == ["Imaginary Song", "Second Fake Track"]  # podcast file ignored
    assert all(pl.skipped is None and pl.reason_end is None for pl in p.plays)
    assert p.plays[0].when == datetime(2024, 6, 1, 8, 0, tzinfo=timezone.utc)
    assert any("basic" in n for n in p.notes)


# ---------------------------------------------------------------- detect + load
@pytest.mark.parametrize("name,expected", [
    ("reddit", ["reddit"]), ("x", ["x"]), ("google", ["google"]), ("google_html", ["google"]),
    ("spotify_extended", ["spotify"]), ("spotify_basic", ["spotify"]),
])
def test_detect(form, name, expected):
    assert detect(form(name)) == expected


def test_detect_combined_folder(tmp_path):
    for name in ("reddit", "spotify_basic"):
        shutil.copytree(FX / name, tmp_path / "all" / name)
    assert detect(tmp_path / "all") == ["reddit", "spotify"]
    assert detect(as_zip(tmp_path / "all", tmp_path)) == ["reddit", "spotify"]


def test_load_merges(tmp_path):
    p = load(FX / "reddit", as_zip(FX / "x", tmp_path), FX / "google", FX / "spotify_extended")
    assert p.sources == ["reddit", "x", "google", "spotify"]
    assert {t.source for t in p.texts} == {"reddit", "x", "google_search", "youtube_search"}
    assert len(p.plays) == 2 and len(p.watches) == 2
    whens = [t.when for t in p.texts]
    assert whens == sorted(whens)                           # merged timeline is in order


def test_load_unknown(tmp_path):
    (tmp_path / "junk").mkdir()
    (tmp_path / "junk" / "readme.txt").write_text("nothing")
    p = load(tmp_path / "junk")
    assert p.texts == [] and any("no known export" in n for n in p.notes)


def test_macos_junk_ignored(tmp_path):
    d = tmp_path / "exp"
    shutil.copytree(FX / "reddit", d)
    (d / "._comments.csv").write_bytes(b"\x00\x05\x16\x07binary")
    assert len(reddit.parse(d).texts) == 4
