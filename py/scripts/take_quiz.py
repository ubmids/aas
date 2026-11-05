"""Take the AAS short-form Big Five quiz in the terminal.

    uv run python scripts/take_quiz.py          # short form (20 items, ~2 minutes)
    uv run python scripts/take_quiz.py --full   # all 50 items
    uv run python scripts/take_quiz.py --json   # also print the raw answers as JSON

Answer how well each statement describes you as you generally are now,
not as you wish to be: 1 = disagree, 3 = neutral, 5 = agree.
"""

from __future__ import annotations

import argparse
import json

from aas import quiz
from aas.schema import TRAITS


def bar(pct: float, width: int = 30) -> str:
    n = round(pct / 100 * width)
    return "#" * n + "." * (width - n)


def describe(z: float) -> str:
    if z <= -1.0:
        return "well below average"
    if z <= -0.4:
        return "below average"
    if z < 0.4:
        return "about average"
    if z < 1.0:
        return "above average"
    return "well above average"


def ask(code: str, i: int, n: int) -> int:
    while True:
        raw = input(f"[{i:2d}/{n}] {quiz.ITEMS[code]}  (1-5): ").strip()
        if raw in {"1", "2", "3", "4", "5"}:
            return int(raw)
        print("      please type a number from 1 (disagree) to 5 (agree)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="ask all 50 items")
    ap.add_argument("--json", action="store_true", help="print the answers as JSON at the end")
    args = ap.parse_args()

    items = quiz.PAGE_ORDER if args.full else quiz.short_form()
    print(__doc__.split("\n\n")[-1].strip(), "\n")
    try:
        answers = {c: ask(c, i, len(items)) for i, c in enumerate(items, 1)}
    except (KeyboardInterrupt, EOFError):
        print("\nstopped, nothing saved.")
        return

    res = quiz.score(answers)
    norms = quiz.load_norms()
    print(f"\nCompared with {norms['n']:,} people who took the full test online:\n")
    for t in TRAITS:
        r = res[t]
        print(f"  {r['trait']:<18} {bar(r['percentile'])}  {r['percentile']:5.1f} pct   z {r['z']:+.2f}  ({describe(r['z'])})")
    if not args.full:
        print(
            "\n  Short form: each trait is 4 statements. On people it was not tuned on, a 4-item"
            "\n  score correlates about 0.91-0.95 with the full 10-item score, so treat the"
            "\n  percentiles as +/- roughly 15 points (one standard error). --full gives the precise version."
        )
    if args.json:
        print(json.dumps(answers))


if __name__ == "__main__":
    main()
