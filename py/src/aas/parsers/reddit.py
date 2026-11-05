"""Reddit GDPR data request export (reddit.com/settings/data-request).

Reads   comments.csv  (body, date, subreddit)  -> TextItem kind "comment"
        posts.csv     (title, body, date, subreddit) -> TextItem kind "post" (title + body)
Ignores every other CSV (votes, saved, messages, ip_logs, subscriptions, ...).
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

from aas.parsers._bundle import Bundle, parse_time, sort_key
from aas.person import Person, TextItem

SOURCE = "reddit"
_GONE = {"", "[deleted]", "[removed]"}


def matches(b: Bundle) -> bool:
    return bool(b.find("comments.csv") or b.find("posts.csv"))


def _rows(b: Bundle, name: str):
    for path in b.find(name):
        reader = csv.DictReader(io.StringIO(b.read_text(path), newline=""))
        for row in reader:
            # header names have been lower-case so far; normalise just in case
            yield {(k or "").strip().lower(): (v or "") for k, v in row.items()}


def parse(path: str | Path) -> Person:
    p = Person(sources=[SOURCE])
    gone = 0
    with Bundle(path) as b:
        if not matches(b):
            p.notes.append("reddit: no comments.csv or posts.csv found")
            return p
        for row in _rows(b, "comments.csv"):
            body = row.get("body", "").strip()
            if body in _GONE:
                gone += 1
                continue
            p.texts.append(TextItem(parse_time(row.get("date")), SOURCE, "comment", body))
        for row in _rows(b, "posts.csv"):
            title = row.get("title", "").strip()
            body = row.get("body", "").strip()
            if body in _GONE:
                body = ""
            text = f"{title}\n\n{body}" if title and body else (title or body)
            if not text:
                gone += 1
                continue
            p.texts.append(TextItem(parse_time(row.get("date")), SOURCE, "post", text))
    p.texts.sort(key=sort_key)
    if gone:
        p.notes.append(f"reddit: skipped {gone} deleted/removed/empty items")
    return p
