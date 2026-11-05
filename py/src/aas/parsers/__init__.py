"""Parsers: personal data exports in, one Person out.

    detect(path)   which formats are inside a .zip or folder, e.g. ["x", "spotify"]
    load(*paths)   detect + parse every format found in every path, merged into one Person

Each source module has `parse(path) -> Person` and `matches(bundle) -> bool`.
Nothing here writes to disk or touches the network.
"""

from __future__ import annotations

from pathlib import Path

from aas.parsers import google, reddit, spotify, x
from aas.parsers._bundle import Bundle, sort_key
from aas.person import Person

PARSERS = {"reddit": reddit, "x": x, "google": google, "spotify": spotify}


def detect(path: str | Path) -> list[str]:
    with Bundle(path) as b:
        return [name for name, mod in PARSERS.items() if mod.matches(b)]


def load(*paths: str | Path) -> Person:
    person = Person()
    for path in paths:
        found = detect(path)
        if not found:
            person.notes.append(f"no known export format in {Path(path).name}")
        for name in found:
            person = person.merge(PARSERS[name].parse(path))
    person.texts.sort(key=sort_key)
    person.plays.sort(key=sort_key)
    person.watches.sort(key=sort_key)
    person.sources = list(dict.fromkeys(person.sources))   # one entry per source, in order
    return person


__all__ = ["PARSERS", "detect", "load", "google", "reddit", "spotify", "x"]
