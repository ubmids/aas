"""The "only you" statement engine on the 304 ListenBrainz listeners: coverage, stability, the
pre-registered lineup test and the Barnum control (py/docs/statements.md).

    cd ~/Desktop/aas/py && uv run python scripts/run_statements.py values   # measure families (cached)
    cd ~/Desktop/aas/py && uv run python scripts/run_statements.py all      # everything, writes results

Reads the local, gitignored bench (../bench/data/lbusers/plays). People and their split are the
music reader's 304 (scripts/build_music_norms.py person_arrays: 2,000+ listens, last 8 weeks held
out with 1,000+ before and 200+ inside; the training weeks are used). Per-person values go to the
system temp dir, never to the repo. Writes aggregate results to docs/statements_results.json and
example cards (pseudonymous ids) to docs/statements_cards.md.
"""

from __future__ import annotations

import gzip
import json
import multiprocessing as mp
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from aas.statements import FAMILIES, Table, from_rows, measure, top_k
from aas.statements import lineup as lu
from aas.statements.engine import quarter_sd

ROOT = Path(__file__).resolve().parents[2]
PLAYS = ROOT / "bench" / "data" / "lbusers" / "plays"
DOCS = Path(__file__).resolve().parents[1] / "docs"
CACHE = Path(tempfile.gettempdir()) / "aas_statements_values.npz"
WEEK = 7 * 86_400_000
L = 20          # lineups per target
SEED = 5


def load(pid: str):
    raw = json.loads(gzip.decompress((PLAYS / f"{pid}.json.gz").read_bytes()))
    if len(raw) < 2000:
        return None
    cut = raw[-1]["t"] - 8 * WEEK
    n = next(i for i, p in enumerate(raw) if p["t"] >= cut)
    if n < 1000 or len(raw) - n < 200:
        return None
    rows = []
    for p in raw[:n]:
        m = p["artistMbids"]
        key = m[0] if m and len(m) == 1 else "n:" + p["artist"].lower()
        rows.append((p["t"], key, p["track"], p.get("album")))
    return from_rows(rows)


def person_values(pid: str):
    h = load(pid)
    if h is None:
        return pid, None
    full = h.window()
    a, b = h.halves()
    out = {
        "full": measure(full), "h1": measure(a), "h2": measure(b),
        "full_q": np.stack([measure(q) for q in full.quarters()]),
        "h1_q": np.stack([measure(q) for q in a.quarters()]),
        "plays": len(h.t), "days": full.days,
    }
    return pid, out


def values() -> dict:
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=False)
        keys = [f.key for f in FAMILIES]
        if list(z["families"]) == keys:
            return {k: z[k] for k in z.files}
    ids = sorted(p.name[:-8] for p in PLAYS.glob("*.json.gz"))
    t = time.time()
    with mp.Pool(max(1, mp.cpu_count() - 2)) as pool:
        res = [r for r in pool.imap(person_values, ids, chunksize=4) if r[1] is not None]
    print(f"measured {len(res)} people in {time.time() - t:.0f} s", file=sys.stderr)
    d = {k: np.stack([r[1][k] for r in res]) for k in ("full", "h1", "h2", "full_q", "h1_q")}
    d["plays"] = np.array([r[1]["plays"] for r in res])
    d["days"] = np.array([r[1]["days"] for r in res])
    d["ids"] = np.array([r[0] for r in res])
    d["families"] = np.array([f.key for f in FAMILIES])
    np.savez(CACHE, **d)
    return d


def describe(d: dict) -> list[dict]:
    rows = []
    for k, f in enumerate(FAMILIES):
        x = d["full"][:, k]
        ok = np.isfinite(x)
        h1, h2 = d["h1"][:, k], d["h2"][:, k]
        both = np.isfinite(h1) & np.isfinite(h2)
        r = float(np.corrcoef(h1[both], h2[both])[0, 1]) if both.sum() > 10 else None
        rows.append({"family": f.key, "group": f.group, "measured": int(ok.sum()),
                     "p10": float(np.nanpercentile(x, 10)), "median": float(np.nanmedian(x)),
                     "p90": float(np.nanpercentile(x, 90)), "split_half_r": r, "both_halves": int(both.sum())})
    return rows


def exploratory(d: dict, first: Table, second: Table, runs: dict) -> dict:
    """Not pre-registered; added after the pre-registered results were in, to explain them."""
    out: dict = {}
    r3 = runs[("only", 3)]
    full3 = r3.n_statements == 3
    out["only_k3_by_statements"] = {str(n): {"lineups": int((r3.n_statements == n).sum()),
                                             "accuracy": float(r3.credit[r3.n_statements == n].mean())}
                                    for n in range(4) if (r3.n_statements == n).any()}
    out["only_k3_targets_with_3_every_lineup"] = {"targets": int(full3.all(1).sum()),
                                                  "accuracy": float(r3.per_target[full3.all(1)].mean())}
    # how identifiable are the halves at all? nearest neighbour on all 34 values, z-scored on first halves
    mu, sd = np.nanmean(d["h1"], 0), np.nanstd(d["h1"], 0)
    sd[~(sd > 0)] = np.nan
    z1, z2 = (d["h1"] - mu) / sd, (d["h2"] - mu) / sd
    cred = np.zeros(r3.members.shape[:2])
    for a in range(r3.members.shape[0]):
        for l in range(r3.members.shape[1]):
            mem = r3.members[a, l]
            diff = np.abs(z2[mem] - z1[mem[0]])
            n = np.isfinite(diff).sum(1)
            dist = np.where(n > 0, np.nansum(diff, 1) / np.maximum(n, 1), np.inf)
            best = dist.min()
            tied = dist == best
            cred[a, l] = tied[0] / tied.sum()
    lo, hi = lu.bootstrap_ci(cred.mean(1))
    out["nearest_neighbour_all_families"] = {"accuracy": float(cred.mean()), "ci95": [lo, hi],
                                             "note": "same lineups as only_k3; mean absolute z difference over all 34 families"}
    # stability by confidence
    rows = []
    for i in range(first.P):
        for s in first.candidates(i, [i], "only"):
            rows.append((s.conf, bool(second.truth(s, i))))
    c = np.array(rows)
    bands = [(0.5, 0.7), (0.7, 0.9), (0.9, 0.99), (0.99, 1.01)]
    out["stability_by_confidence"] = {f"{a:.2f}-{min(b, 1):.2f}": {"made": int(((c[:, 0] >= a) & (c[:, 0] < b)).sum()),
                                                                   "held": float(c[(c[:, 0] >= a) & (c[:, 0] < b), 1].mean())}
                                      for a, b in bands if ((c[:, 0] >= a) & (c[:, 0] < b)).any()}
    return out


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    d = values()
    P = len(d["ids"])
    print(f"{P} people, {len(FAMILIES)} families", file=sys.stderr)
    if stage == "values":
        for r in describe(d):
            print(f"{r['family']:22s} n={r['measured']:3d} p10={r['p10']:8.3f} med={r['median']:8.3f} "
                  f"p90={r['p90']:8.3f} r={r['split_half_r'] if r['split_half_r'] is None else round(r['split_half_r'], 3)}")
        return

    full = Table(d["full"], quarter_sd(d["full_q"]))
    first = Table(d["h1"], quarter_sd(d["h1_q"]))
    second = Table(d["h2"])
    res: dict = {"people": P, "families": len(FAMILIES), "lineups_per_target": L, "describe": describe(d)}

    # 1. coverage on the whole history
    counts, cards = [], {}
    fam_hits: dict[str, int] = {}
    for i in range(P):
        sts = full.candidates(i, [i], "only")
        counts.append(len(sts))
        for s in sts:
            fam_hits[s.family.key] = fam_hits.get(s.family.key, 0) + 1
        cards[i] = top_k(sts, 5)
    counts = np.array(counts)
    res["coverage_full"] = {"ge1": float((counts >= 1).mean()), "ge3": float((counts >= 3).mean()),
                            "ge5": float((counts >= 5).mean()), "median": float(np.median(counts)),
                            "mean": float(counts.mean()), "per_family": fam_hits}
    c1 = np.array([len(first.candidates(i, [i], "only")) for i in range(P)])
    res["coverage_h1"] = {"ge3": float((c1 >= 3).mean()), "ge5": float((c1 >= 5).mean()), "median": float(np.median(c1))}

    # 2. stability
    st = lu.stability(first, second)
    res["stability"] = {"made": st.made, "held": st.held, "held_share": st.held / max(st.made, 1),
                        "uncheckable": st.uncheckable, "still_rare": st.still_rare,
                        "still_rare_share": st.still_rare / max(st.made, 1),
                        "top3_made": st.top3_made, "top3_held": st.top3_held,
                        "top3_held_share": st.top3_held / max(st.top3_made, 1),
                        "per_family": {k: {"made": m, "held": h} for k, (m, h) in sorted(st.per_family.items())}}

    # 3. lineup test, Barnum control
    res["lineup"] = {}
    runs = {}
    for mode in ("only", "barnum"):
        for k in (3, 5):
            t = time.time()
            r = runs[(mode, k)] = lu.run(first, second, k, L, SEED + k, mode)
            lo, hi = lu.bootstrap_ci(r.per_target)
            null, p = lu.permutation_null(r, 1000)
            res["lineup"][f"{mode}_k{k}"] = {
                "accuracy": r.accuracy, "ci95": [lo, hi], "chance": 0.2,
                "perm_null_mean": float(null.mean()), "perm_null_p95": float(np.quantile(null, 0.95)), "perm_p": p,
                "targets": int(len(r.per_target)), "lineups": int(r.credit.size),
                "mean_statements": float(r.n_statements.mean()),
                "targets_without_statements": int((r.n_statements == 0).all(1).sum()),
                "share_lineups_short": float((r.n_statements < k).mean()),
                "target_alone_on_top": float((r.credit == 1).mean()),
            }
            print(f"{mode} k={k}: {r.accuracy:.3f} [{lo:.3f}, {hi:.3f}] null {null.mean():.3f} p={p:.4f} "
                  f"({time.time() - t:.0f} s)", file=sys.stderr)
    o = res["lineup"]
    res["pass"] = {"rule": "only-you top-3 accuracy >= 0.60, 95% CI lower bound > 0.40, permutation p < 0.01",
                   "passed": bool(o["only_k3"]["accuracy"] >= 0.60 and o["only_k3"]["ci95"][0] > 0.40
                                  and o["only_k3"]["perm_p"] < 0.01)}

    res["exploratory"] = exploratory(d, first, second, runs)
    (DOCS / "statements_results.json").write_text(json.dumps(res, indent=1) + "\n")

    # example cards: three people drawn at random (pre-registered seed), plus the most striking statements
    rng = np.random.default_rng(SEED)
    pick = rng.choice(P, 3, replace=False)
    lines = ["# Statement cards, three listeners drawn at random", "",
             "Generated by `scripts/run_statements.py` on each person's whole history (training weeks), against the",
             f"other {P - 1} listeners. Pseudonymous ids are the bench's salted hashes, cut to eight characters.", ""]
    for i in pick:
        lines.append(f"## {d['ids'][i][:8]} ({d['plays'][i]:,} plays over {d['days'][i] / 365.25:.1f} years)")
        lines.append("")
        if not cards[int(i)]:
            lines.append("No statement is true of this listener and of 10% or fewer of the others.")
        for s in cards[int(i)]:
            lines.append(f"- {s.card()} (confidence {s.conf:.2f})")
        lines.append("")
    allst = [(s, i) for i in range(P) for s in cards[i][:1]]
    allst.sort(key=lambda x: (-(x[0].conf >= 0.95), x[0].c, -x[0].score))
    lines += ["## The most striking first statements", "",
              "Each listener's top statement, sorted by how few others it is true of (confidence 0.95 or more first).", ""]
    for s, i in allst[:10]:
        lines.append(f"- {d['ids'][i][:8]}: {s.card()} (confidence {s.conf:.2f})")
    (DOCS / "statements_cards.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "describe"}, indent=1))


if __name__ == "__main__":
    main()
