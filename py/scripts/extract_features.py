"""Turn your own data exports into the derived feature record, show it, then write it.

    cd py && uv run python scripts/extract_features.py ~/Downloads/reddit.zip \
        ~/Downloads/takeout.zip ~/Downloads/my_spotify_data.zip --out record.json [--word-rates] [--yes]

Everything runs on this machine; nothing is uploaded. Before anything is written you see a
plain-English list of exactly what is in the record (and, with --show-json, the raw JSON).
The record is validated against src/aas/data/feature_schema.json; a record that does not fit
is never written.

  --word-rates   also include the opt-in common-word block (rates of 2,000 fixed common words)
  --yes          write without asking (otherwise you are asked when running in a terminal)
  --show-json    print the full JSON as well as the summary
  --no-readers   skip the personality / music models (counts, rhythms and topics only)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aas import features as F
from aas import parsers


def readers(log) -> list:
    out = []
    try:
        from aas.readers.text import TextBigFiveReader, TextMBTIReader

        out += [TextBigFiveReader(), TextMBTIReader()]
    except FileNotFoundError as ex:
        log(f"note: text readers not available ({ex}); their block will say not_run")
    from aas.readers.music import MusicReader

    out.append(MusicReader())
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exports", nargs="+", type=Path, help="export .zip files or unzipped folders")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--word-rates", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--show-json", action="store_true")
    ap.add_argument("--no-readers", action="store_true")
    a = ap.parse_args(argv)
    log = lambda m: print(m, file=sys.stderr)  # noqa: E731

    for p in a.exports:
        if not p.expanduser().exists():
            ap.error(f"not found: {p}")
    log(f"reading {len(a.exports)} export(s) locally ...")
    person = parsers.load(*[p.expanduser() for p in a.exports])
    log(f"found: {', '.join(person.sources) or 'nothing recognisable'}")

    rec = F.build(person, [] if a.no_readers else readers(log), word_rates_opt_in=a.word_rates)
    F.validate(rec)
    text = json.dumps(rec, indent=1, allow_nan=False)

    print(F.preview(rec))
    print(f"\nThe record is {len(text.encode()):,} bytes of JSON and nothing else.")
    if a.show_json:
        print("\n" + text)
    sys.stdout.flush()

    if not a.yes:
        if not sys.stdin.isatty():
            log("not writing: run with --yes to write without a prompt")
            return 1
        if input(f"\nWrite this to {a.out}? [y/N] ").strip().lower() not in ("y", "yes"):
            log("nothing written")
            return 1
    a.out.write_text(text + "\n")
    log(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
