"""Sanity-check your own exports: what got read, from when to when, and a few examples.

    uv run python scripts/inspect_export.py ~/Downloads/twitter-archive.zip ~/Downloads/takeout.zip

Prints to the terminal only. Writes nothing, sends nothing.
"""

from __future__ import annotations

import random
import sys
from collections import Counter

from aas.parsers import detect, load


def _short(s: str, n: int = 100) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _span(items) -> str:
    ws = [i.when for i in items if i.when is not None]
    if not ws:
        return "no dates"
    undated = len(items) - len(ws)
    out = f"{min(ws):%Y-%m-%d} to {max(ws):%Y-%m-%d}"
    return out + (f" ({undated} undated)" if undated else "")


def main(paths: list[str]) -> None:
    if not paths:
        sys.exit(__doc__)
    for path in paths:
        print(f"{path}: {', '.join(detect(path)) or 'nothing recognised'}")
    person = load(*paths)
    rng = random.Random(0)

    groups: dict[str, list] = {}
    for t in person.texts:
        groups.setdefault(f"text  {t.source}/{t.kind}", []).append(t)
    for w in person.watches:
        groups.setdefault(f"watch {w.source}", []).append(w)
    for pl in person.plays:
        groups.setdefault(f"play  {pl.source}", []).append(pl)

    print()
    for name, items in groups.items():
        print(f"{name}: {len(items):,} items, {_span(items)}")
        for it in rng.sample(items, min(3, len(items))):
            when = f"{it.when:%Y-%m-%d}" if it.when else "????-??-??"
            if hasattr(it, "text"):
                body = it.text
            elif hasattr(it, "track"):
                body = f"{it.artist} - {it.track} ({(it.ms_played or 0) / 1000:.0f}s"
                body += ", skipped)" if it.skipped else ")"
            else:
                body = it.title + (f"  [{it.channel}]" if it.channel else "")
            print(f"    {when}  {_short(body)}")
    if person.plays:
        top = Counter(pl.artist for pl in person.plays).most_common(5)
        print("top artists:", ", ".join(f"{a} ({n})" for a, n in top))
    if person.notes:
        print("\nnotes:")
        for n in person.notes:
            print("  -", n)


if __name__ == "__main__":
    main(sys.argv[1:])
