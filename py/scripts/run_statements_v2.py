"""Statements round 2 (py/docs/statements.md, "Round 2"): 57 families, the frozen v2 selection rule,
the lineup test, the fingerprint matcher, the stable-reading proxy and the Barnum control.

    cd py && uv run python scripts/run_statements_v2.py old     # frozen v2 on round 1's listeners
    cd py && uv run python scripts/run_statements_v2.py fresh   # the confirmation run (once)
    cd py && uv run python scripts/run_statements_v2.py cards   # redraw the cards from its cache

Reads the local, gitignored benches: ../bench/data/lbusers/plays (round 1's listeners) and
../bench/data/lbusers/plays_fresh (the confirmation set). Per-person values go to the system temp
dir, never to the repo. Aggregates go to docs/statements_v2_results.json (one key per set), cards
to docs/statements_cards.md.
"""

from __future__ import annotations

import gzip
import json
import multiprocessing as mp
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

from aas.statements import from_rows
from aas.statements import lineup as lu
from aas.statements import v2
from aas.statements.engine import Table, quarter_sd, top_k
from aas.statements.families_v2 import FAMILIES_V2, measure_v2

ROOT = Path(__file__).resolve().parents[2]
SETS = {"old": ROOT / "bench" / "data" / "lbusers" / "plays",
        "fresh": ROOT / "bench" / "data" / "lbusers" / "plays_fresh"}
DOCS = Path(__file__).resolve().parents[1] / "docs"
WEEK = 7 * 86_400_000
FAMS = FAMILIES_V2


def artist_key(p: dict) -> str:
    m = p["artistMbids"]
    return m[0] if m and len(m) == 1 else "n:" + p["artist"].lower()


def load(path: Path):
    """Round 1's person rule (2,000+ listens; last 8 weeks dropped with 1,000+ before and 200+ in
    them) plus round 2's single-artist exclusion (one artist over 75% of listens, or fewer than 50
    artists), on the cleaned listens. Returns (history, why-excluded)."""
    raw = json.loads(gzip.decompress(path.read_bytes()))
    if len(raw) < 2000:
        return None, "under 2,000 listens"
    c = Counter(artist_key(p) for p in raw)
    if max(c.values()) / len(raw) > v2.MAX_TOP_ARTIST or len(c) < v2.MIN_ARTISTS:
        return None, "one artist only"
    cut = raw[-1]["t"] - 8 * WEEK
    n = next(i for i, p in enumerate(raw) if p["t"] >= cut)
    if n < 1000 or len(raw) - n < 200:
        return None, "too little before or inside the last 8 weeks"
    rows = [(p["t"], artist_key(p), p["track"], p.get("album")) for p in raw[:n]]
    return from_rows(rows), None


def person_values(path: Path):
    h, why = load(path)
    if h is None:
        return path.name[:-8], None, why
    full = h.window()
    a, b = h.halves()
    out = {
        "full": measure_v2(full), "h1": measure_v2(a), "h2": measure_v2(b),
        "full_q": np.stack([measure_v2(q) for q in full.quarters()]),
        "h1_q": np.stack([measure_v2(q) for q in a.quarters()]),
        "plays": len(h.t), "days": full.days,
        "h1_plays": len(a), "h1_days": a.days,
    }
    return path.name[:-8], out, None


def values(which: str) -> dict:
    cache = Path(tempfile.gettempdir()) / f"aas_statements_v2_{which}.npz"
    keys = [f.key for f in FAMS]
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if list(z["families"]) == keys:
            return {k: z[k] for k in z.files}
    files = sorted(p for p in SETS[which].glob("*.json.gz") if p.stat().st_size > 0)
    t = time.time()
    with mp.Pool(max(1, mp.cpu_count() - 2)) as pool:
        res = list(pool.imap(person_values, files, chunksize=2))
    ok = [r for r in res if r[1] is not None]
    print(f"{which}: measured {len(ok)} of {len(res)} in {time.time() - t:.0f} s", file=sys.stderr)
    d = {k: np.stack([r[1][k] for r in ok]) for k in ("full", "h1", "h2", "full_q", "h1_q")}
    for k in ("plays", "days", "h1_plays", "h1_days"):
        d[k] = np.array([r[1][k] for r in ok])
    d["ids"] = np.array([r[0] for r in ok])
    d["families"] = np.array(keys)
    d["excluded"] = np.array([f"{r[0]}|{r[2]}" for r in res if r[1] is None])
    np.savez(cache, **d)
    return d


def tables(d: dict, fams=FAMS):
    idx = [list(d["families"]).index(f.key) for f in fams]
    full = Table(d["full"][:, idx], quarter_sd(d["full_q"][:, :, idx]), families=fams)
    first = Table(d["h1"][:, idx], quarter_sd(d["h1_q"][:, :, idx]), families=fams)
    second = Table(d["h2"][:, idx], families=fams)
    return full, first, second


L = 20          # lineups per target, as in round 1
SEED = 5


def cands(tab, i, ex, mode):
    return v2.candidates(tab, i, ex, mode)


def evaluate(which: str) -> tuple[dict, dict]:
    d = values(which)
    P = len(d["ids"])
    full, first, second = tables(d)
    excl = Counter(x.split("|", 1)[1] for x in d["excluded"])
    res: dict = {"set": which, "people": P, "families": len(FAMS), "excluded": dict(excl),
                 "lineups_per_target": L, "line": v2.LINE}
    print(f"{which}: {P} people, {len(FAMS)} families, excluded {dict(excl)}", file=sys.stderr)

    # coverage on whole histories, against the other P - 1
    counts, rare10, cards = [], [], {}
    for i in range(P):
        sts = v2.candidates(full, i, [i])
        counts.append(len(sts))
        rare10.append(sum(s.share <= 0.10 for s in sts))
        cards[i] = top_k(sts, 5)
    counts, rare10 = np.array(counts), np.array(rare10)
    res["coverage_full"] = {"ge1": float((counts >= 1).mean()), "ge3": float((counts >= 3).mean()),
                            "ge5": float((counts >= 5).mean()), "median": float(np.median(counts)),
                            "ge3_at_10pct": float((rare10 >= 3).mean())}
    top3 = [s for i in range(P) for s in cards[i][:3]]
    res["coverage_full"]["top3_share_at_or_under_10pct"] = float(np.mean([s.share <= 0.10 for s in top3]))

    # the reading people would recognise: 3+ statements that hold on the other half and stay rare
    stable = np.array([len(v2.stable_statements(first, second, i)) for i in range(P)])
    res["stable_reading"] = {"ge3": float((stable >= 3).mean()), "ge1": float((stable >= 1).mean()),
                             "median": float(np.median(stable))}

    # stability of every first-half statement, as in round 1
    made = held = 0
    for i in range(P):
        for s in v2.candidates(first, i, [i]):
            made += 1
            held += int(bool(second.truth(s, i)))
    res["stability"] = {"made": made, "held": held, "held_share": held / max(made, 1)}

    # lineups
    res["lineup"] = {}
    runs = {}
    for mode in ("only", "barnum"):
        for k in (3, 5):
            t = time.time()
            r = runs[(mode, k)] = lu.run(first, second, k, L, SEED + k, mode, cands=cands)
            lo, hi = lu.bootstrap_ci(r.per_target)
            null, p = lu.permutation_null(r, 1000)
            res["lineup"][f"{mode}_k{k}"] = {
                "accuracy": r.accuracy, "ci95": [lo, hi], "chance": 0.2,
                "perm_null_mean": float(null.mean()), "perm_p": p, "targets": int(len(r.per_target)),
                "mean_statements": float(r.n_statements.mean()),
                "share_lineups_short": float((r.n_statements < k).mean()),
                "target_alone_on_top": float((r.credit == 1).mean()),
            }
            print(f"{which} {mode} k={k}: {r.accuracy:.3f} [{lo:.3f}, {hi:.3f}] p={p:.4f} ({time.time() - t:.0f} s)",
                  file=sys.stderr)
    r3 = runs[("only", 3)]
    p1 = v2.percentiles(d["h1"], d["h1"])
    p2 = v2.percentiles(d["h1"], d["h2"])
    fc = v2.fingerprint_credit(p1, p2, r3.members)
    lo, hi = lu.bootstrap_ci(fc.mean(1))
    res["fingerprint"] = {"accuracy": float(fc.mean()), "ci95": [lo, hi], "lineups": "the only_k3 lineups"}
    o = res["lineup"]["only_k3"]
    res["verdict"] = {
        "primary": {"rule": "only-you top-3 accuracy >= 0.60, 95% CI lower bound > 0.50, permutation p < 0.01",
                    "passed": bool(o["accuracy"] >= 0.60 and o["ci95"][0] > 0.50 and o["perm_p"] < 0.01)},
        "fingerprint": {"rule": "fingerprint accuracy >= 0.75", "passed": bool(res["fingerprint"]["accuracy"] >= 0.75)},
        "barnum_k3_at_most_0.30": bool(res["lineup"]["barnum_k3"]["accuracy"] <= 0.30),
    }
    return res, {"d": d, "cards": cards, "P": P}


def write_cards(fresh: dict) -> None:
    d, cards, P = fresh["d"], fresh["cards"], fresh["P"]
    ids = d["ids"]

    def person(i: int) -> list[str]:
        out = [f"### {ids[i][:8]} ({d['plays'][i]:,} plays over {d['days'][i] / 365.25:.1f} years)", ""]
        if not cards[i]:
            out.append("No statement is true of this listener and of a fifth or fewer of the others.")
        for s in cards[i][:3]:
            out.append(f"- {s.card()} (confidence {s.conf:.2f})")
        return out + [""]

    rng = np.random.default_rng(SEED)
    pick = [int(i) for i in rng.choice(P, 3, replace=False)]
    # striking: the person whose top three are, together, the rarest (sum of surprisal), excluding the random three
    def rarity(i: int) -> float:
        return sum(-np.log2((s.c + 1) / (s.n + 1)) for s in cards[i][:3]) if len(cards[i]) >= 3 else -1.0
    striking = [i for i in sorted(range(P), key=lambda i: -rarity(i)) if i not in pick][:3]
    lines = ["# Statement cards (round 2, fresh listeners)", "",
             "Made by `scripts/run_statements_v2.py fresh` with the frozen round 2 rule, on each person's whole",
             f"history (training weeks), against the other {P - 1} fresh listeners. Each card shows the top three",
             "statements. Ids are the bench's salted hashes cut to eight characters; no artist, track, release or",
             "username appears.", "",
             "## Three listeners drawn at random", "",
             "Drawn with seed 5 from the fresh set, before anyone looked at their cards.", ""]
    for i in pick:
        lines += person(i)
    lines += ["## Three striking listeners (a selection)", "",
              "These are picked, not drawn: the three people outside the random draw whose top three statements are",
              "together the rarest. They show what the engine says at its best, and they are not typical.", ""]
    for i in striking:
        lines += person(i)
    (DOCS / "statements_cards.md").write_text("\n".join(lines))


def redraw_cards() -> None:
    """Rewrite docs/statements_cards.md from the cached fresh measures (display only).

    Reads the per-person values the confirmation run saved to the temp dir and redoes only what the
    cards need: each person's statements on the whole history and the same seed-5 draw. No lineup,
    fingerprint or verdict is touched, and docs/statements_v2_results.json is not written."""
    cache = Path(tempfile.gettempdir()) / "aas_statements_v2_fresh.npz"
    if not cache.exists():
        raise SystemExit("no cached fresh measures; the cards can only be redrawn from the saved run")
    z = np.load(cache, allow_pickle=False)
    if list(z["families"]) != [f.key for f in FAMS]:
        raise SystemExit("cached fresh measures were made with a different family list")
    d = {k: z[k] for k in z.files}
    P = len(d["ids"])
    full, _, _ = tables(d)
    cards = {i: top_k(v2.candidates(full, i, [i]), 5) for i in range(P)}
    write_cards({"d": d, "cards": cards, "P": P})


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "old"
    if which == "cards":
        return redraw_cards()
    out_file = DOCS / "statements_v2_results.json"
    allres = json.loads(out_file.read_text()) if out_file.exists() else {}
    if which == "fresh" and "fresh" in allres:
        raise SystemExit("the confirmation run has already been made; it is run once")
    res, extra = evaluate(which)
    allres[which] = res
    out_file.write_text(json.dumps(allres, indent=1) + "\n")
    if which == "fresh":
        write_cards(extra)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
