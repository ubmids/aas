"""Write src/aas/data/feature_schema.json, the JSON Schema of the derived feature record.

    cd py && uv run python scripts/build_feature_schema.py

Generated (not hand-typed) so its enums always match aas.features and the shipped word list;
tests/test_features_privacy.py fails if the file on disk differs from what this script makes.
Rules: every object closes with additionalProperties: false; every string is an enum or a short,
anchored pattern; every number has a range. That is what makes "no free text" checkable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aas import features as F  # noqa: E402

OUT = F.SCHEMA_FILE


def obj(props: dict, required: list | None = None) -> dict:
    return {"type": "object", "properties": props, "required": list(props) if required is None else required,
            "additionalProperties": False}


def nullable(s: dict) -> dict:
    s = dict(s)
    t = s["type"]
    s["type"] = (t if isinstance(t, list) else [t]) + ["null"]
    if "enum" in s:
        s["enum"] = s["enum"] + [None]
    return s


def num(lo: float, hi: float) -> dict:
    return {"type": "number", "minimum": lo, "maximum": hi}


def enum(values) -> dict:
    return {"type": "string", "enum": list(values)}


SHARE = num(0, 1)
MONTH = {"type": "string", "pattern": r"^(19|20)\d\d-(0[1-9]|1[0-2])$", "maxLength": 7}
VERSION = {"type": "string", "pattern": r"^[A-Za-z0-9][A-Za-z0-9.+-]{0,39}$", "maxLength": 40}
HASH = {"type": "string", "pattern": r"^[0-9a-f]{64}$", "maxLength": 64}


def schema() -> dict:
    estimate = obj({
        "usable": {"type": "boolean"},
        "reason": nullable(enum(["insufficient_data"])),
        "value": nullable(num(-1000, 1000)),
        "low": nullable(num(-1000, 1000)),
        "high": nullable(num(-1000, 1000)),
        "interval_level": nullable(SHARE),
        "percentile": nullable(num(0, 100)),
        "validity": nullable(num(-1, 1)),
        "validity_kind": nullable(enum(["r", "auc"])),
        "reliability": nullable(num(-1, 1)),
    })
    readers = {}
    for name, traits in F.READER_TRAITS.items():
        readers[name] = obj({
            "status": enum(F.READER_STATUS),
            "version": nullable(VERSION),
            "estimates": obj({t: {"$ref": "#/$defs/estimate"} for t in traits}, required=[]),
            "raw_scores": nullable(obj({t: nullable(num(-1000, 1000)) for t in traits}, required=[]))
            if name.startswith("text-") else {"type": "null"},
        })
    hist = lambda n: {"type": ["array", "null"], "items": SHARE, "minItems": n, "maxItems": n}  # noqa: E731
    activity = obj({
        "items": enum(F.BUCKET_LABELS),
        "first_month": nullable(MONTH),
        "last_month": nullable(MONTH),
        "span_months": nullable({"type": "integer", "minimum": 1, "maximum": 1200}),
        "active_day_share": nullable(SHARE),
        "hours_utc": hist(24),
        "weekdays": hist(7),
    })
    style_ranges = {
        "mean_words_per_item": (0, 100000), "mean_word_length": (0, 100), "ttr_1000": (0, 1),
        "caps_word_share": (0, 1), "long_word_share": (0, 1),
    }
    style = obj({k: nullable(num(*style_ranges.get(k, (0, 1) if k.endswith("_share") else (0, 1000))))
                 for k in F.STYLE_KEYS})
    qform = obj({k: nullable(num(0, 1000) if k == "mean_words" else SHARE) for k in F.QUERY_KEYS})
    vform = obj({k: nullable(num(0, 64) if k == "channel_entropy_bits" else SHARE) for k in F.VIEW_KEYS})
    topics = obj({c: nullable(SHARE) for c in F.topic_categories()})
    words = list(F.common_words())
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"https://aas.local/feature_record/{F.SCHEMA_VERSION}",
        "title": "AAS derived feature record",
        "description": "The only data that leaves a participant's computer. See py/docs/features.md.",
        "$defs": {"estimate": estimate},
        **obj({
            "schema_version": {"type": "string", "const": F.SCHEMA_VERSION},
            "extractor": obj({"version": VERSION, "hash": HASH, "word_list_version": VERSION,
                              "topic_lexicon_version": VERSION}),
            "models": {"type": "array", "maxItems": len(F.READER_TRAITS),
                       "items": obj({"reader": enum(F.READER_TRAITS), "version": VERSION})},
            "consent": obj({"word_rates": {"type": "boolean"}}),
            "sources": obj({s: {"type": "boolean"} for s in F.ACTIVITY_SOURCES}),
            "floors": obj({k: obj({"passed": {"type": "boolean"}, "reason": enum(F.FLOOR_REASONS)}) for k in F.FLOORS}),
            "readers": obj(readers),
            "activity": obj({s: activity for s in F.ACTIVITY_SOURCES}),
            "writing": nullable(obj({
                "words_rounded": {"type": "integer", "minimum": 0, "maximum": 100_000_000},
                "items_by_kind": obj({k: enum(F.BUCKET_LABELS) for k in F.WRITING_KINDS + ("other",)}),
                "style": style,
            })),
            "search_form": obj({s: nullable(qform) for s in F.SEARCH_SOURCES}),
            "viewing_form": nullable(vform),
            "topics": obj({"google_search": nullable(topics), "youtube_search": nullable(topics),
                           "youtube_watch": nullable(topics)}),
            "spotify": nullable(obj({"export": enum(["extended", "basic"]), "shuffle_share": nullable(SHARE)})),
            "word_rates_status": enum(["absent_no_consent", "below_500_words", "present"]),
            "word_rates": obj({
                "tokens_rounded": {"type": "integer", "minimum": 500, "maximum": 100_000_000},
                "per_1k": {"type": "object", "propertyNames": enum(words), "required": words,
                           "additionalProperties": num(0, 1000)},
            }),
        }, required=["schema_version", "extractor", "models", "consent", "sources", "floors", "readers",
                     "activity", "writing", "search_form", "viewing_form", "topics", "spotify",
                     "word_rates_status"]),
    }


def render() -> str:
    return json.dumps(schema(), indent=1) + "\n"


if __name__ == "__main__":
    OUT.write_text(render())
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
