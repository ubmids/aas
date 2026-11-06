"""Make my card: three sentences about how you listen, from your own export, on your own machine.

    cd py
    uv run python scripts/my_card.py ~/Downloads/my_spotify_data.zip        # Spotify extended history (zip or folder)
    uv run python scripts/my_card.py ~/Downloads/listenbrainz_export.zip     # ListenBrainz export (zip, folder, .json, .jsonl)
    uv run python scripts/my_card.py EXPORT --all                            # also every one of the 57 measures
    uv run python scripts/my_card.py --example                               # the invented person in aas/example.py

Recognition test (py/docs/recognition_test.md):

    uv run python scripts/my_card.py --lineup DIR --out SHEETS [--barnum] [--seed N]
    uv run python scripts/my_card.py --score SHEETS/answers.csv --key SHEETS/answer_key.csv

DIR holds one export per person (a zip, a folder or a .json/.jsonl file), named however you like;
the name is the person's label. To keep exports at home, each person runs

    uv run python scripts/my_card.py EXPORT --save-card NAME.card.json

and sends only NAME.card.json (the card's text and the Barnum card's text, nothing else); DIR can hold
those instead of, or as well as, exports. Lineup mode writes one sheet per person (their own card and two other
people's, shuffled and labelled A, B and C; with --barnum a fourth card, D, to rate only), the answer
key, and an empty answers sheet. Nothing else is written: no plays, artists, tracks or measures.
Nothing is uploaded; there is no network code here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

from aas.parsers import detect
from aas.parsers import listenbrainz as lb
from aas.parsers import spotify
from aas.person import Person
from aas.statements import card as C
from aas.statements.families_v2 import FAMILIES_V2

LETTERS = "ABC"


def load_export(path: str | Path) -> Person:
    """A Spotify export if the path holds one, else a ListenBrainz export."""
    path = Path(path).expanduser()
    if not path.exists():
        raise SystemExit(f"not found: {path}")
    if path.is_file() and path.suffix.lower() in (".json", ".jsonl"):
        p = lb.parse(path)
    else:
        try:
            found = detect(path)
        except ValueError as e:
            raise SystemExit(str(e)) from None
        p = spotify.parse(path) if "spotify" in found else lb.parse(path)
    if not p.plays:
        raise SystemExit(f"no listening history found in {path.name} (Spotify extended history or ListenBrainz listens)")
    return p


def personal(card: C.Card, show_all: bool = False) -> str:
    lines = ["Your listening card", "",
             f"Made from {card.plays:,} plays over {card.days / 365.25:.1f} years, against 636 ListenBrainz "
             "listeners (aggregate figures only). Nothing left this computer.", ""]
    lines += C.render(card)
    if card.notes:
        lines += ["", "Notes:"] + [f"  {n}" for n in card.notes]
    if show_all and card.values is not None:
        lines += ["", "All 57 measures (your value, and where it sits among the 636 listeners):"]
        for f, v, p in zip(FAMILIES_V2, card.values, card.percentiles):
            where = f"{round(100 * p)}th percentile" if math.isfinite(p) else "no percentile"
            lines.append(f"  {f.key:22s} {f.show(v):>28s}   {where}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------- lineup

def exports_in(folder: Path) -> list[Path]:
    out = []
    for p in sorted(folder.iterdir()):
        if p.name.startswith("."):
            continue
        if p.is_dir() or p.suffix.lower() in (".zip", ".json", ".jsonl"):   # includes NAME.card.json
            out.append(p)
    return out


CARD_SUFFIX = ".card.json"


def label_of(p: Path) -> str:
    if p.name.endswith(CARD_SUFFIX):
        return p.name[: -len(CARD_SUFFIX)]
    return p.name[: -len(p.suffix)] if p.is_file() and p.suffix else p.name


def card_texts(card: C.Card) -> dict:
    """What a card file holds: the rendered card and Barnum card, as text lines. Nothing else."""
    return {"card": C.render(card), "barnum": C.render(C.make_barnum(card))}


def texts_for(p: Path) -> dict | None:
    if p.name.endswith(CARD_SUFFIX):
        d = json.loads(p.read_text())
        if not isinstance(d.get("card"), list) or not isinstance(d.get("barnum"), list):
            raise SystemExit(f"{p.name} is not a card file")
        return {"card": [str(x) for x in d["card"]], "barnum": [str(x) for x in d["barnum"]]}
    c = C.make_card(load_export(p))
    return card_texts(c) if c.lines else None


def assign(n: int, seed: int) -> list[tuple[int, list[int], list[int]]]:
    """For each person: (person, the two other people whose cards they see, the order of the three
    cards on the sheet as person indices). People sit in a random cycle and see the next two along,
    so every card is shown as a decoy exactly twice; the order on each sheet is shuffled."""
    if n < 3:
        raise ValueError("a lineup needs at least 3 people")
    rng = np.random.default_rng(seed)
    cyc = [int(x) for x in rng.permutation(n)]
    out = []
    for j, i in enumerate(cyc):
        dec = [cyc[(j + 1) % n], cyc[(j + 2) % n]]
        order = [int(x) for x in rng.permutation([i] + dec)]
        out.append((i, dec, order))
    out.sort(key=lambda t: t[0])
    return out


def sheet_text(label: str, cards: list[list[str]], barnum: list[str] | None) -> str:
    lines = [f"Listening cards for {label}", "",
             "Each card below was made from someone's listening history by the same program. One of A, B and C",
             "was made from yours; the other two were made from other people in this test.", "",
             "1. Pick the one card, A, B or C, that you think is yours.",
             "2. For every card, say how much it sounds like you, from 1 (not me at all) to 5 (exactly me).", ""]
    for letter, c in zip(LETTERS, cards):
        lines += [f"Card {letter}", ""] + c + [""]
    if barnum is not None:
        lines += ["After you have picked: one more card to rate (not to pick). How much does it sound like you, 1 to 5?",
                  "", "Card D", ""] + barnum + [""]
    return "\n".join(lines)


def run_lineup(folder: Path, out: Path, seed: int, barnum: bool) -> None:
    paths = exports_in(folder)
    labels = [label_of(p) for p in paths]
    if len(set(labels)) != len(labels):
        raise SystemExit("two exports have the same name; rename one so every person has their own label")
    cards, keep = [], []
    for p, lab in zip(paths, labels):
        try:
            c = texts_for(p)
        except SystemExit as e:
            print(f"left out {lab}: {e}", file=sys.stderr)
            continue
        if c is None:
            print(f"left out {lab}: no sentence could be made", file=sys.stderr)
            continue
        cards.append(c)
        keep.append(lab)
    if len(keep) < 3:
        raise SystemExit(f"a lineup needs at least 3 people with a card; have {len(keep)}")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, dec, order in assign(len(keep), seed):
        own = LETTERS[order.index(i)]
        bc = cards[i]["barnum"] if barnum else None
        name = f"sheet_{keep[i]}.txt"
        (out / name).write_text(sheet_text(keep[i], [cards[j]["card"] for j in order], bc) + "\n")
        rows.append({"person": keep[i], "sheet": name, "own": own,
                     "A": keep[order[0]], "B": keep[order[1]], "C": keep[order[2]],
                     "D": "barnum" if barnum else ""})
    with open(out / "answer_key.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(out / "answers.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["person", "pick", "rate_A", "rate_B", "rate_C", "rate_D"])
        for r in rows:
            w.writerow([r["person"], "", "", "", "", ""])
    print(f"{len(rows)} sheets, answer_key.csv and answers.csv in {out}")
    print("Keep answer_key.csv away from the people taking the test until they have answered.")


# ------------------------------------------------------------------------------------- scoring

BAR = 2 / 3          # pre-registered: own card picked at least twice as often as chance (1/3)


def score(answers: Path, key: Path) -> dict:
    from scipy.stats import binomtest

    k = {r["person"]: r for r in csv.DictReader(open(key, newline=""))}
    hits = n = 0
    own_r, other_r, barnum_r = [], [], []
    for r in csv.DictReader(open(answers, newline="")):
        if r["person"] not in k or not r.get("pick", "").strip():
            continue
        kk = k[r["person"]]
        n += 1
        hits += int(r["pick"].strip().upper() == kk["own"])
        for L in LETTERS:
            v = r.get(f"rate_{L}", "").strip()
            if v:
                (own_r if L == kk["own"] else other_r).append(float(v))
        v = r.get("rate_D", "").strip()
        if v and kk.get("D") == "barnum":
            barnum_r.append(float(v))
    if n == 0:
        raise SystemExit("no answered rows")
    bt = binomtest(hits, n, 1 / 3, alternative="greater")
    p = bt.pvalue
    ci = binomtest(hits, n, 1 / 3).proportion_ci(0.95, method="exact")
    mean = lambda x: round(float(np.mean(x)), 2) if x else None  # noqa: E731
    return {"people": n, "picked_own": hits, "rate": hits / n, "chance": 1 / 3, "bar": BAR,
            "meets_bar": hits / n >= BAR, "binomial_p_vs_chance": float(p),
            "ci95": [round(float(ci.low), 3), round(float(ci.high), 3)],
            "mean_rating_own": mean(own_r), "mean_rating_others": mean(other_r), "mean_rating_barnum": mean(barnum_r)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("export", nargs="?", help="your Spotify extended history or ListenBrainz export")
    ap.add_argument("--all", action="store_true", help="also print every one of the 57 measures")
    ap.add_argument("--example", action="store_true", help="the invented person in aas/example.py")
    ap.add_argument("--save-card", type=Path, help="also write the card text (only) to NAME.card.json for a lineup")
    ap.add_argument("--lineup", type=Path, help="folder with one export per person")
    ap.add_argument("--out", type=Path, default=Path("card_sheets"), help="where lineup sheets go")
    ap.add_argument("--barnum", action="store_true", help="add a Barnum control card (D) to every sheet")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--score", type=Path, help="answers.csv to score")
    ap.add_argument("--key", type=Path, help="answer_key.csv for --score")
    a = ap.parse_args(argv)
    if a.score:
        if not a.key:
            raise SystemExit("--score needs --key")
        res = score(a.score, a.key)
        for k, v in res.items():
            print(f"{k}: {v}")
        return
    if a.lineup:
        return run_lineup(a.lineup, a.out, a.seed, a.barnum)
    if a.example:
        from aas import example
        person = example.synthetic_person()
    elif a.export:
        person = load_export(a.export)
    else:
        ap.error("give an export path, --example, --lineup or --score")
    card = C.make_card(person)
    print(personal(card, a.all))
    if a.save_card:
        dest = a.save_card if a.save_card.name.endswith(CARD_SUFFIX) else a.save_card.with_name(a.save_card.name + CARD_SUFFIX)
        dest.write_text(json.dumps(card_texts(card), indent=1) + "\n")
        print(f"\nCard text saved to {dest} (no plays, artists or tracks in it).")


if __name__ == "__main__":
    main()
