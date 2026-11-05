"""X / Twitter archive (Settings > Your account > Download an archive of your data).

Reads   data/tweets.js (or tweet.js in older archives, plus tweets-part1.js ...),
        each `window.YTD.tweets.part0 = [ {"tweet": {...}}, ... ]`
        data/note-tweet.js  long-form "note tweets" (full text of >280-char posts)
Keeps   the person's own tweets (kind "tweet") and replies (kind "reply").
Drops   retweets (not their words), t.co links, deleted-tweets.js, likes, DMs, everything else.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

from aas.parsers._bundle import Bundle, parse_time, sort_key
from aas.person import Person, TextItem

SOURCE = "x"
_TWEET_FILE = re.compile(r"tweets?(-part\d+)?\.js", re.I)
_NOTE_FILE = re.compile(r"note-tweet(-part\d+)?\.js", re.I)
_TCO = re.compile(r"https?://t\.co/\S+")


def matches(b: Bundle) -> bool:
    return bool(b.find(_TWEET_FILE))


def _load_js(text: str) -> list:
    """`window.YTD.x.part0 = [...]` -> the list."""
    start = text.find("=")
    if start < 0:
        return []
    body = text[start + 1:].strip().rstrip(";")
    data = json.loads(body)
    return data if isinstance(data, list) else []


def clean(text: str) -> str:
    text = html.unescape(text)          # archive text has &amp; &lt; &gt;
    text = _TCO.sub("", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def _is_retweet(t: dict) -> bool:
    txt = t.get("full_text") or t.get("text") or ""
    return txt.startswith("RT @") or str(t.get("retweeted", "")).lower() == "true" \
        or "retweeted_status" in t


def parse(path: str | Path) -> Person:
    p = Person(sources=[SOURCE])
    rts = 0
    with Bundle(path) as b:
        files = b.find(_TWEET_FILE)
        if not files:
            p.notes.append("x: no tweets.js / tweet.js found")
            return p
        for f in files:
            for entry in _load_js(b.read_text(f)):
                t = entry.get("tweet", entry) if isinstance(entry, dict) else {}
                if not t:
                    continue
                if _is_retweet(t):
                    rts += 1
                    continue
                text = clean(t.get("full_text") or t.get("text") or "")
                if not text:
                    continue
                reply = bool(t.get("in_reply_to_status_id_str") or t.get("in_reply_to_status_id"))
                p.texts.append(TextItem(parse_time(t.get("created_at")), SOURCE,
                                        "reply" if reply else "tweet", text))

        # Long posts appear in tweets.js truncated; note-tweet.js holds the full text.
        for f in b.find(_NOTE_FILE):
            for entry in _load_js(b.read_text(f)):
                n = entry.get("noteTweet", {}) if isinstance(entry, dict) else {}
                full = clean((n.get("core") or {}).get("text", ""))
                if not full:
                    continue
                for item in p.texts:
                    stem = item.text.rstrip("…").rstrip(".").strip()
                    if len(stem) >= 20 and full.startswith(stem):
                        item.text = full          # upgrade the truncated copy
                        break
                else:
                    p.texts.append(TextItem(parse_time(n.get("createdAt")), SOURCE, "tweet", full))
    p.texts.sort(key=sort_key)
    if rts:
        p.notes.append(f"x: dropped {rts} retweets (not the person's own words)")
    return p
