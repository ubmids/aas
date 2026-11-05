"""Build the music reader's norms and dial parameters from the 304-listener ListenBrainz bench, and
cross-check the Python traits against the TypeScript engine's saved values.

    cd py && uv run python scripts/build_music_norms.py [--limit N] [--out DIR]

Reads (local, gitignored, never written to the package):
  ../bench/data/lbusers/plays/<id>.json.gz    cleaned ListenBrainz listens (bench/lbusers/load.ts)
  ../bench/out/lb3/person/<id>.json           the TS engine's traits per person (bench/lb3/person.ts)
Writes:
  src/aas/data/music_norms.json   quantiles (1st..99th percentile) per trait and per dial, n; no rows
  src/aas/data/music_dials.json   which traits feed each dial, loadings, score weights, the fit summary
  <scratch>/music_bench_values.json  per-person values (pseudonyms), for the factor fit only; NOT in the repo

Each person is split exactly as round 3 did (bench/lb3/common.ts loadPerson): artists keyed by the
MusicBrainz id when a listen carries exactly one, else "n:" + lower-cased credit; at least 2,000
listens; the last 8 weeks held out, with 1,000+ listens before and 200+ inside. Traits are measured
on the training weeks, like the published stability and factor check.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm, rankdata

from aas.readers import music

ROOT = Path(__file__).resolve().parents[2]
PLAYS = ROOT / "bench" / "data" / "lbusers" / "plays"
TS_PERSON = ROOT / "bench" / "out" / "lb3" / "person"
DATA = Path(__file__).resolve().parents[1] / "src" / "aas" / "data"

TS_KEY = {"new_share": "newShare", "old_friends": "oldFriends", "season_turnover": "turnover",
          "clock_spread": "clockSpread", "clock_regularity": "clockRegular"}
LB_TRAITS = list(TS_KEY)
DIALS = [
    {"key": "dial_explore", "label": "Explore against stay", "traits": ["new_share", "old_friends", "season_turnover"],
     "high": "exploring", "low": "staying with old favourites", "sign_trait": "new_share"},
    {"key": "dial_regular", "label": "Listening-time regularity", "traits": ["clock_spread", "clock_regularity"],
     "high": "a regular listening time", "low": "listening at any hour", "sign_trait": "clock_regularity"},
]
P = list(range(1, 100))
CLIP = 1 / 302  # extreme normal score of the published check: rank 1 of n = 301 -> qnorm(1/302)


def person_arrays(pid: str) -> music.Arrays | None:
    raw = json.loads(gzip.decompress((PLAYS / f"{pid}.json.gz").read_bytes()))
    if len(raw) < 2000:
        return None
    cut = raw[-1]["t"] - 8 * music.WEEK
    n = next(i for i, p in enumerate(raw) if p["t"] >= cut)
    if n < 1000 or len(raw) - n < 200:
        return None
    rows = []
    for p in raw[:n]:
        m = p["artistMbids"]
        key = m[0] if m and len(m) == 1 else "n:" + p["artist"].lower()
        d = p["durMs"]
        rows.append((p["t"], d if d and d > 30_000 else 180_000, True, key, None))
    del raw
    return music.arrays_from_rows(rows)


# ---------------------------------------------------------------------------- factor maths (ports of engine/src/model/stats.ts)

def normal_scores(x: np.ndarray) -> np.ndarray:
    return norm.ppf(rankdata(x) / (len(x) + 1))


def varimax(L0: np.ndarray, iters: int = 100) -> np.ndarray:
    d, k = L0.shape
    if k < 2:
        return L0.copy()
    h = np.sqrt((L0 ** 2).sum(1))
    h[h == 0] = 1
    L = L0 / h[:, None]
    for _ in range(iters):
        moved = 0
        for p in range(k):
            for q in range(p + 1, k):
                u = L[:, p] ** 2 - L[:, q] ** 2
                v = 2 * L[:, p] * L[:, q]
                A, B, C, D = u.sum(), v.sum(), (u * u - v * v).sum(), (2 * u * v).sum()
                phi = math.atan2(D - 2 * A * B / d, C - (A * A - B * B) / d) / 4
                if abs(phi) < 1e-10:
                    continue
                moved += 1
                c, s = math.cos(phi), math.sin(phi)
                a, b = L[:, p].copy(), L[:, q].copy()
                L[:, p], L[:, q] = c * a + s * b, -s * a + c * b
        if not moved:
            break
    return L * h[:, None]


def pca_loadings(Z: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    R = np.corrcoef(Z, rowvar=False)
    val, vec = np.linalg.eigh(R)
    o = np.argsort(val)[::-1]
    val, vec = val[o], vec[:, o]
    L = vec[:, :k] * np.sqrt(np.maximum(val[:k], 0))
    L = varimax(L) if k >= 2 else L
    for c in range(k):
        if L[np.argmax(np.abs(L[:, c])), c] < 0:
            L[:, c] = -L[:, c]
    return L, val


def parallel_analysis(Z: np.ndarray, B: int = 1000, seed: int = 5) -> dict:
    obs = np.sort(np.linalg.eigvalsh(np.corrcoef(Z, rowvar=False)))[::-1]
    rng = np.random.default_rng(seed)
    sims = np.empty((B, Z.shape[1]))
    for b in range(B):
        Y = np.column_stack([rng.permutation(Z[:, j]) for j in range(Z.shape[1])])
        sims[b] = np.sort(np.linalg.eigvalsh(np.corrcoef(Y, rowvar=False)))[::-1]
    p95 = np.sort(sims, 0)[int(0.95 * (B - 1))]
    keep = 0
    while keep < len(obs) and obs[keep] > p95[keep]:
        keep += 1
    return {"observed": obs.tolist(), "p95": p95.tolist(), "keep": keep}


def compact(o: dict) -> str:
    """JSON with one line per leaf block, so the files read (and diff) as tables."""
    def enc(v, ind):
        if isinstance(v, dict) and any(isinstance(x, (dict, list)) and not _flat(x) for x in v.values()):
            pad = " " * (ind + 1)
            return "{\n" + ",\n".join(f"{pad}{json.dumps(k)}: {enc(x, ind + 1)}" for k, x in v.items()) + "\n" + " " * ind + "}"
        if isinstance(v, list) and v and not _flat(v):
            pad = " " * (ind + 1)
            return "[\n" + ",\n".join(pad + enc(x, ind + 1) for x in v) + "\n" + " " * ind + "]"
        return json.dumps(v)
    return enc(o, 0) + "\n"


def _flat(v) -> bool:
    xs = v.values() if isinstance(v, dict) else v
    return all(not isinstance(x, (dict, list)) or (isinstance(x, list) and all(not isinstance(y, (dict, list)) for y in x)) for x in xs)


def r5(x):
    return None if x is None or not math.isfinite(x) else round(float(x), 5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None, help="where per-person values go (default: the system temp dir, never the repo)")
    args = ap.parse_args()
    scratch = Path(args.out) if args.out else Path(tempfile.gettempdir()) / "aas_music"
    scratch.mkdir(parents=True, exist_ok=True)
    cache = scratch / "music_bench_values.json"

    ids = sorted(f.name[:-8] for f in PLAYS.glob("*.json.gz") if f.stat().st_size > 0)
    if args.limit:
        ids = ids[: args.limit]
    vals: dict[str, dict] = json.loads(cache.read_text()) if cache.exists() else {}
    t0 = time.time()
    for i, pid in enumerate(ids):
        if pid in vals:
            continue
        a = person_arrays(pid)
        if a is None:
            vals[pid] = {"skipped": True}
            continue
        tr = music.measure(a)
        vals[pid] = {k: r5(tr[k].value) for k in LB_TRAITS}
        del a
        gc.collect()
        if i % 20 == 0:
            print(f"{i}/{len(ids)} {time.time() - t0:.0f}s", file=sys.stderr)
            cache.write_text(json.dumps(vals))
    cache.write_text(json.dumps(vals))
    people = {k: v for k, v in vals.items() if not v.get("skipped")}
    print(f"people kept: {len(people)} of {len(ids)}")

    # ---- cross-check against the TS engine's saved values
    agree = {k: [0, 0, 0.0] for k in LB_TRAITS}  # matched within 1e-4, compared, max abs diff
    nan_mismatch = {k: 0 for k in LB_TRAITS}
    ts_vals = {}
    for pid, v in people.items():
        f = TS_PERSON / f"{pid}.json"
        if not f.exists():
            continue
        ts = json.loads(f.read_text())["traits"]["full"]
        ts_vals[pid] = ts
        for k in LB_TRAITS:
            a, b = v[k], ts[TS_KEY[k]]
            if (a is None) != (b is None):
                nan_mismatch[k] += 1
                continue
            if a is None:
                continue
            agree[k][1] += 1
            agree[k][0] += abs(a - b) <= 1e-4
            agree[k][2] = max(agree[k][2], abs(a - b))
    print("TS vs Python (training weeks):")
    for k in LB_TRAITS:
        m, n, mx = agree[k]
        print(f"  {k:18s} {m}/{n} within 1e-4, max |diff| {mx:.2e}, measurable-in-one-only {nan_mismatch[k]}")

    # ---- reproduce the published 6-trait loadings from the TS values (checks the factor port)
    keys6 = ["newShare", "spread", "oldFriends", "clockSpread", "clockRegular", "turnover"]
    rows6 = np.array([[t[k] for k in keys6] for t in ts_vals.values() if all(t[k] is not None for k in keys6)])
    Z6 = np.column_stack([normal_scores(rows6[:, j]) for j in range(6)])
    L6, ev6 = pca_loadings(Z6, 2)
    print(f"published check reproduced on n={len(rows6)}: eigenvalues {np.round(ev6[:3], 3)}")
    for k, row in zip(keys6, L6):
        print(f"  {k:13s} {row[0]:+.3f} {row[1]:+.3f}")

    # ---- norms: quantiles per trait over the Python values
    norms = {"source": "304 public ListenBrainz histories (AAS round 3 bench), training weeks; Python port of the TS traits",
             "n": len(people), "p": P, "traits": {}, "dials": {}}
    for k in LB_TRAITS:
        x = np.array([v[k] for v in people.values() if v[k] is not None])
        norms["traits"][k] = {"n": len(x), "p": P, "v": [r5(q) for q in np.percentile(x, P)]}

    # ---- the dials: factor check on the 5 traits a history without the taste map can give
    full = [v for v in people.values() if all(v[k] is not None for k in LB_TRAITS)]
    X = np.array([[v[k] for k in LB_TRAITS] for v in full])
    Z = np.column_stack([normal_scores(X[:, j]) for j in range(X.shape[1])])
    pa = parallel_analysis(Z, 1000, 5)
    L, ev = pca_loadings(Z, 2)
    Rm = np.corrcoef(Z, rowvar=False)
    W = np.linalg.solve(Rm, L)  # regression (Thurstone) score weights: scores = Z R^-1 L
    print(f"5-trait fit on n={len(full)}: eigenvalues {np.round(ev, 3)}, parallel keeps {pa['keep']} (p95 {np.round(pa['p95'], 3)})")
    dials_out = {"source": norms["source"], "n": len(full), "clip": CLIP, "traits": LB_TRAITS,
                 "standardise": "each trait -> percentile on music_norms.json -> normal score qnorm(p), clipped to [clip, 1-clip]",
                 "eigenvalues": [r5(x) for x in ev], "parallel": {k: ([r5(x) for x in v] if isinstance(v, list) else v) for k, v in pa.items()},
                 "corr": [[r5(x) for x in r] for r in Rm], "dials": []}
    for d in DIALS:
        # the component on which the dial's traits load most
        idx = [LB_TRAITS.index(k) for k in d["traits"]]
        c = int(np.argmax([np.abs(L[idx, c]).sum() for c in range(2)]))
        sign = 1 if L[LB_TRAITS.index(d["sign_trait"]), c] > 0 else -1
        w = sign * W[:, c]
        # weights on traits outside the dial are near zero after varimax; keep only the dial's own,
        # rescaled so the score has variance 1 on the bench
        wd = np.zeros(len(LB_TRAITS))
        wd[idx] = w[idx]
        s = Z @ wd
        wd /= s.std()
        sc = Z @ wd
        corr_full = float(np.corrcoef(sc, Z @ w)[0, 1])
        dials_out["dials"].append({
            "key": d["key"], "label": d["label"], "high": d["high"], "low": d["low"], "traits": d["traits"],
            "weights": [r5(wd[i]) for i in idx],
            "loadings": {k: r5(sign * L[LB_TRAITS.index(k), c]) for k in LB_TRAITS},
            "r_with_full_component_score": r5(corr_full),
        })
        print(f"{d['key']}: component {c + 1}, weights {dict(zip(d['traits'], np.round(wd[idx], 4)))}, r with full score {corr_full:.4f}")

    # dial quantiles via the same path the reader uses (quantile table -> normal score)
    DATA.mkdir(parents=True, exist_ok=True)
    scores = {d["key"]: [] for d in DIALS}
    for v in full:
        s = music.dial_scores(v, norms, dials_out)
        for k in scores:
            scores[k].append(s[k])
    for k, x in scores.items():
        norms["dials"][k] = {"n": len(x), "p": P, "v": [r5(q) for q in np.percentile(x, P)]}
        print(f"{k}: bench mean {np.mean(x):+.3f} sd {np.std(x):.3f}")
    (DATA / "music_norms.json").write_text(compact(norms))
    (DATA / "music_dials.json").write_text(compact(dials_out))
    print("wrote", DATA / "music_norms.json", DATA / "music_dials.json")


if __name__ == "__main__":
    main()
