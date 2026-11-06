"""One real person, read from their own data exports. The contract between
parsers (exports in), readers (models out) and the profile (what they see).

    parsers   aas.parsers.<source>.parse(path) -> Person      one per export format
    readers   a Reader turns a Person into Estimates           one per model
    profile   aas.profile.build(person, readers) -> Profile    what the person sees

Everything stays on the machine it runs on. Nothing in this package uploads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Protocol


@dataclass
class TextItem:
    """Something the person wrote themselves (not things they only read or watched)."""
    when: datetime | None
    source: str            # "reddit", "x", "google_search", ...
    kind: str              # "comment", "post", "tweet", "search", ...
    text: str


@dataclass
class Play:
    """One listen. Field names follow the Spotify extended history where they exist."""
    when: datetime          # UTC; end time for Spotify, start time for ListenBrainz (sub-track difference)
    artist: str
    track: str
    ms_played: int | None = None
    skipped: bool | None = None
    reason_start: str | None = None
    reason_end: str | None = None
    shuffle: bool | None = None
    source: str = "spotify"
    country: str | None = None   # where it was played (Spotify conn_country), for local time
    album: str | None = None     # release name where the export has one


@dataclass
class Watch:
    """Something the person watched or visited (title only, no page content fetched)."""
    when: datetime | None
    source: str             # "youtube", "tiktok", ...
    title: str
    channel: str | None = None


@dataclass
class Person:
    texts: list[TextItem] = field(default_factory=list)
    plays: list[Play] = field(default_factory=list)
    watches: list[Watch] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)   # which exports went in
    notes: list[str] = field(default_factory=list)     # parser warnings worth showing

    def merge(self, other: "Person") -> "Person":
        return Person(
            self.texts + other.texts,
            self.plays + other.plays,
            self.watches + other.watches,
            self.sources + other.sources,
            self.notes + other.notes,
        )


@dataclass
class Estimate:
    """One claim about the person, with how far to trust it."""
    reader: str                 # which model made it, e.g. "text-bigfive"
    trait: str                  # "O".."N", or "I/E" etc., or a measured trait name
    kind: Literal["bigfive", "mbti", "measured"]
    value: float                # bigfive: z-score; mbti: P(first letter); measured: the raw measure
    low: float | None = None    # interval on the same scale, from cross-validated error
    high: float | None = None
    percentile: float | None = None   # 0-100 where a population reference exists
    validity: float | None = None     # the reader's tested accuracy (r or AUC) on unseen people
    validity_kind: str | None = None  # "r" or "auc"
    evidence: str = ""          # plain-English "based on what": counts, examples
    usable: bool = True         # False when there was too little data to say anything
    interval_level: float | None = 0.80   # how often [low, high] claims to contain the truth
    reliability: float | None = None  # for measured traits: split-half r (is it stable?), not accuracy


class Reader(Protocol):
    name: str

    def can_read(self, person: Person) -> tuple[bool, str]:
        """Enough data? Returns (ok, reason in plain words)."""

    def read(self, person: Person) -> list[Estimate]: ...
