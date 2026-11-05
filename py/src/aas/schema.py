"""The one shape every data source is turned into.

Each dataset loader writes two tables under data/processed/<dataset>/:

persons.parquet   one row per person
    person_id   str     "<dataset>:<original id>", never a real name or handle
    dataset     str
    O C E A N   float   Big Five as z-scores within that dataset (NaN if not measured)
    mbti        str     four letters or None (only where that is the label)

features.parquet  one row per person, indexed by person_id
    numeric columns only, each one a behaviour or language measure,
    computed from the person's data alone (no labels, no other people).

A slice is a group of datasets that see the same kind of life:
text (what you write), phone (where you go, how you sleep, how you use it),
body (wearable), play (games), music (listening).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]  # repository root
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"

TRAITS = ["O", "C", "E", "A", "N"]
TRAIT_NAMES = {
    "O": "openness",
    "C": "conscientiousness",
    "E": "extraversion",
    "A": "agreeableness",
    "N": "neuroticism",
}


def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=0)


def save(dataset: str, persons: pd.DataFrame, features: pd.DataFrame) -> Path:
    """Check the contract and write both tables."""
    assert persons["person_id"].is_unique
    assert set(features.index) <= set(persons["person_id"]), "features for unknown people"
    for t in TRAITS:
        if t not in persons:
            persons[t] = np.nan
    if "mbti" not in persons:
        persons["mbti"] = None
    persons["dataset"] = dataset
    bad = [c for c in features.columns if not np.issubdtype(features[c].dtype, np.number)]
    assert not bad, f"non-numeric feature columns: {bad}"
    out = PROCESSED / dataset
    out.mkdir(parents=True, exist_ok=True)
    persons.to_parquet(out / "persons.parquet", index=False)
    features.to_parquet(out / "features.parquet")
    return out


def load(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = PROCESSED / dataset
    persons = pd.read_parquet(d / "persons.parquet").set_index("person_id")
    features = pd.read_parquet(d / "features.parquet")
    return persons, features
