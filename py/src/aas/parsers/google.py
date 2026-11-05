"""Google Takeout (takeout.google.com).

Reads   My Activity/<Search>/MyActivity.json (or MyActivity.html in older exports,
        best-effort): Google searches -> TextItem source "google_search", kind "search"
        YouTube .../history/watch-history.json (.html best-effort) -> Watch(title, channel)
        YouTube .../history/search-history.json (.html best-effort) -> TextItem
        source "youtube_search", kind "search"
Ignores "Visited ..." clicks, ads shown on YouTube, removed/private videos, and every
        other Takeout product. Folder names are localised, so files are matched by name.
"""

from __future__ import annotations

import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qs, urlparse

from aas.parsers._bundle import Bundle, parse_time, sort_key
from aas.person import Person, TextItem, Watch

SOURCE = "google"


def matches(b: Bundle) -> bool:
    return bool(b.find("MyActivity.json") or b.find("MyActivity.html")
                or b.find("watch-history.*") or b.find("search-history.*"))


def _query_from_url(url: str | None, key: str) -> str | None:
    if not url:
        return None
    q = parse_qs(urlparse(url).query).get(key)
    return q[0].strip() if q else None


def _is_google_search_url(url: str | None) -> bool:
    if not url:
        return False
    u = urlparse(url)
    return "google." in u.netloc and u.path.startswith("/search") and "q=" in u.query


def _strip(prefix: str, title: str) -> str:
    return title[len(prefix):].strip() if title.startswith(prefix) else title.strip()


def _is_ad(e: dict) -> bool:
    return any("Ads" in (d.get("name") or "") for d in e.get("details") or [])


# ---------------------------------------------------------------- old HTML exports
_CELL = re.compile(
    r'<div class="content-cell[^"]*mdl-typography--body-1[^"]*">(.*?)</div>', re.S)
_DATE_TAIL = re.compile(r"([A-Z][a-z]{2} \d{1,2}, \d{4}, \d{1,2}:\d{2}:\d{2}\s?[AP]M)")


def _html_cells(text: str):
    """Yield (parts, hrefs, when) per activity cell; parts = the text pieces split on <br>."""
    for m in _CELL.finditer(text):
        raw = m.group(1)
        hrefs = re.findall(r'<a href="([^"]+)"', raw)
        parts = [html.unescape(re.sub(r"<[^>]+>", "", s)).replace("\xa0", " ").strip()
                 for s in re.split(r"<br\s*/?>", raw)]
        parts = [s for s in parts if s]
        when = None
        for s in reversed(parts):
            d = _DATE_TAIL.search(s)
            if d:
                try:  # the zone name after it (GMT, WAT, ...) is ignored: treated as UTC
                    stamp = re.sub(r"(\d)\s*([AP]M)", r"\1 \2", re.sub(r"\s+", " ", d.group(1)))
                    when = datetime.strptime(stamp, "%b %d, %Y, %I:%M:%S %p").replace(tzinfo=timezone.utc)
                except ValueError:
                    pass
                break
        yield parts, hrefs, when


# ---------------------------------------------------------------- parse
def parse(path: str | Path) -> Person:
    p = Person(sources=[SOURCE])
    skipped = {"ads": 0, "removed": 0}
    with Bundle(path) as b:
        # (a) Google Search activity. MyActivity.json exists once per product; YouTube's
        # duplicates watch/search-history, so only keep google.com/search entries.
        for f in b.find("MyActivity.json"):
            for e in json.loads(b.read_text(f)):
                if e.get("header") in ("YouTube", "YouTube Music"):
                    continue
                title, url = e.get("title", ""), e.get("titleUrl")
                q = None
                if title.startswith("Searched for "):
                    q = title[len("Searched for "):].strip()
                elif _is_google_search_url(url):          # localised title, e.g. "Gesucht nach"
                    q = _query_from_url(url, "q")
                if q:
                    p.texts.append(TextItem(parse_time(e.get("time")), "google_search", "search", q))
        if not p.texts:
            for f in b.find("MyActivity.html"):
                for parts, hrefs, when in _html_cells(b.read_text(f)):
                    if parts and parts[0].startswith("Searched for"):
                        q = _strip("Searched for", parts[0])
                        if q:
                            p.texts.append(TextItem(when, "google_search", "search", q))
            if p.texts:
                p.notes.append("google: search history read from old HTML export (best-effort, English only)")

        # (b) YouTube watch history
        for f in b.find("watch-history.json"):
            for e in json.loads(b.read_text(f)):
                if _is_ad(e):
                    skipped["ads"] += 1
                    continue
                if not e.get("titleUrl"):        # "Watched a video that has been removed"
                    skipped["removed"] += 1
                    continue
                subs = e.get("subtitles") or []
                p.watches.append(Watch(parse_time(e.get("time")), "youtube",
                                       _strip("Watched ", e.get("title", "")),
                                       subs[0].get("name") if subs else None))
        if not b.find("watch-history.json"):
            for f in b.find("watch-history.html"):
                for parts, hrefs, when in _html_cells(b.read_text(f)):
                    if not parts or not parts[0].startswith("Watched") or not hrefs:
                        continue
                    # cell = "Watched <a>Title</a><br><a>Channel</a><br>date"
                    title = _strip("Watched", parts[0])
                    chan = parts[1] if len(parts) >= 3 else None
                    p.watches.append(Watch(when, "youtube", title, chan))
                p.notes.append("google: YouTube watch history read from HTML (best-effort)")

        # (c) YouTube search history
        for f in b.find("search-history.json"):
            for e in json.loads(b.read_text(f)):
                if _is_ad(e):
                    continue
                title = e.get("title", "")
                q = _strip("Searched for ", title) if title.startswith("Searched for ") \
                    else _query_from_url(e.get("titleUrl"), "search_query")
                if q:
                    p.texts.append(TextItem(parse_time(e.get("time")), "youtube_search", "search", q))
        if not b.find("search-history.json"):
            for f in b.find("search-history.html"):
                for parts, hrefs, when in _html_cells(b.read_text(f)):
                    if parts and parts[0].startswith("Searched for"):
                        q = _strip("Searched for", parts[0])
                        if q:
                            p.texts.append(TextItem(when, "youtube_search", "search", q))

    p.texts.sort(key=sort_key)
    p.watches.sort(key=sort_key)
    if skipped["ads"]:
        p.notes.append(f"google: skipped {skipped['ads']} YouTube ads (not chosen by the person)")
    if skipped["removed"]:
        p.notes.append(f"google: skipped {skipped['removed']} removed/private videos (no title)")
    return p
