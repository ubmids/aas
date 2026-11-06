"""Identity in k sentences (py/docs/identity.md): bits of identity per statement and per set, joint
selection of k statements, 1-of-N identification and a learned fingerprint metric.

    cd py && uv run python scripts/run_identity.py measure old|fresh|third
    cd py && uv run python scripts/run_identity.py explore    # old + fresh (636 people)
    cd py && uv run python scripts/run_identity.py confirm    # the third set, run once

Reads the gitignored benches under ../bench/data/lbusers (plays, plays_fresh, plays_third). Per-person
values go to the system temp dir, never to the repo; aggregates go to docs/identity_results.json and
the curves to docs/identity_*.csv.
"""

from __future__ import annotations

import gzip
import json
import multiprocessing as mp
import pickle
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_statements_v2 as r2  # noqa: E402  (load rule, families, tables)

from aas.statements import identity as idn  # noqa: E402
from aas.statements import lineup as lu  # noqa: E402
from aas.statements import v2  # noqa: E402
from aas.statements.engine import top_k  # noqa: E402
from aas.statements.families_v2 import FAMILIES_V2, measure_v2  # noqa: E402
from aas.statements.history import Window  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SETS = {"old": ROOT / "bench" / "data" / "lbusers" / "plays",
        "fresh": ROOT / "bench" / "data" / "lbusers" / "plays_fresh",
        "third": ROOT / "bench" / "data" / "lbusers" / "plays_third"}
DOCS = Path(__file__).resolve().parents[1] / "docs"
FAMS = FAMILIES_V2


MONTH = 30.44 * 86_400_000
DECAY_W = 26 * 7 * 86_400_000          # the two windows of the time-decay test are 26 weeks long
GAPS = (0, 1, 3, 6, 12)                # months between the end of window A and the start of window B


def artist_counts(h, w) -> dict[str, int]:
    """Plays per artist code in a window (codes are local to the history; mapped to keys by caller)."""
    a = w.artist
    u, c = np.unique(a, return_counts=True)
    return {int(x): int(n) for x, n in zip(u, c)}


def person_values(path: Path):
    """Round 2's measures (whole training window, halves, quarters) plus the two halves of the first
    half, h1a and h1b, used to estimate how a statement holds from one stretch to the next without
    looking at the second half."""
    h, why = r2.load(path)
    if h is None:
        return path.name[:-8], None, why
    # artist keys in code order (from_rows codes artists by first appearance in time order)
    raw = json.loads(gzip.decompress(path.read_bytes()))
    cut = raw[-1]["t"] - r2.WEEK * 8
    n = next(i for i, p in enumerate(raw) if p["t"] >= cut)
    names = list(dict.fromkeys(r2.artist_key(p) for p in sorted(raw[:n], key=lambda p: p["t"])))
    assert len(names) == h.n_artists
    full = h.window()
    a, b = h.halves()
    mid = (a.lo + a.hi) // 2
    aa, ab = Window(h, a.lo, mid), Window(h, mid, a.hi)
    # time decay: B is the last 26 training weeks, A a 26-week window ending `gap` months before B
    tE = full.hi
    B = Window(h, tE - DECAY_W, tE)
    dec = np.full((len(GAPS), 4, len(FAMS)), np.nan)   # gap x (A, Aa, Ab, B) x families
    dec_art = []
    for g, gap in enumerate(GAPS):
        hiA = int(tE - DECAY_W - gap * MONTH)
        loA = hiA - DECAY_W
        if loA < h.t0:
            dec_art.append(None)
            continue
        A = Window(h, loA, hiA)
        m = (loA + hiA) // 2
        dec[g] = [measure_v2(A), measure_v2(Window(h, loA, m)), measure_v2(Window(h, m, hiA)), measure_v2(B)]
        dec_art.append(artist_counts(h, A))
    out = {
        "full": measure_v2(full), "h1": measure_v2(a), "h2": measure_v2(b),
        "h1a": measure_v2(aa), "h1b": measure_v2(ab), "decay": dec,
        "art_h1": artist_counts(h, a), "art_h2": artist_counts(h, b), "art_B": artist_counts(h, B),
        "art_A": dec_art, "artist_names": names,
        "full_q": np.stack([measure_v2(q) for q in full.quarters()]),
        "h1_q": np.stack([measure_v2(q) for q in a.quarters()]),
        "plays": len(h.t), "days": full.days, "h1_plays": len(a), "h1_days": a.days,
    }
    return path.name[:-8], out, None


def values(which: str) -> dict:
    cache = Path(tempfile.gettempdir()) / f"aas_identity_{which}.npz"
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
    d = {k: np.stack([r[1][k] for r in ok]) for k in ("full", "h1", "h2", "h1a", "h1b", "full_q", "h1_q", "decay")}
    # artist play counts per window, as (person, artist key, plays) with keys shared across people;
    # kept in the temp dir only, like every per-person value
    arts = {}
    for w in ("h1", "h2", "B"):
        arts[w] = [{r[1]["artist_names"][a]: c for a, c in r[1]["art_" + w].items()} for r in ok]
    for g in range(len(GAPS)):
        arts[f"A{g}"] = [None if r[1]["art_A"][g] is None else
                         {r[1]["artist_names"][a]: c for a, c in r[1]["art_A"][g].items()} for r in ok]
    (Path(tempfile.gettempdir()) / f"aas_identity_{which}_artists.pkl").write_bytes(pickle.dumps(arts))
    for k in ("plays", "days", "h1_plays", "h1_days"):
        d[k] = np.array([r[1][k] for r in ok])
    d["ids"] = np.array([r[0] for r in ok])
    d["families"] = np.array(keys)
    d["excluded"] = np.array([f"{r[0]}|{r[2]}" for r in res if r[1] is None])
    np.savez(cache, **d)
    return d


def artists(which: str) -> dict:
    values(which)
    return pickle.loads((Path(tempfile.gettempdir()) / f"aas_identity_{which}_artists.pkl").read_bytes())


def pooled(which: list[str]) -> tuple[dict, dict]:
    ds = [values(w) for w in which]
    d = {k: np.concatenate([x[k] for x in ds]) for k in ("full", "h1", "h2", "h1a", "h1b", "full_q", "h1_q", "decay",
                                                          "plays", "days", "ids")}
    d["families"] = ds[0]["families"]
    d["excluded"] = np.concatenate([x["excluded"] for x in ds])
    arts = [artists(w) for w in which]
    a = {k: sum((x[k] for x in arts), []) for k in arts[0]}
    return d, a


# ------------------------------------------------------------------------------------- evaluation

L = 20            # lineups per target
SEED = 7          # lineup draw (round 2 used 5 + k; a new seed for a new round)
KMAX = 7
WORKERS = 7
G: dict = {}


def _init(H1, H2, H1a, H1b, first, M):
    G["pop"] = idn.Population(H1, H2, H1a, H1b, FAMS)
    G["first"], G["M"] = first, M


def stmt_index(pop: idn.Population, s) -> int:
    k = [f.key for f in pop.families].index(s.family.key)
    sg = 1 if s.direction == "hi" else -1
    return int(np.nonzero((pop.cs.fam == k) & (pop.cs.sign == sg) & (pop.cs.th == s.th))[0][0])


def _pad(ch: list[int]) -> np.ndarray:
    return np.array(ch + [-1] * (KMAX - len(ch)), dtype=np.int64)


def _target(i: int):
    """Joint and independent (round 2) selections for target i: per lineup (against everyone outside
    it, for 4 decoys) and against everyone else (for 1-of-N)."""
    pop, first, M = G["pop"], G["first"], G["M"]
    P = pop.P
    jl = np.zeros((L, KMAX), np.int64)
    vl = np.zeros((L, KMAX), np.int64)
    for l in range(L):
        mem = M[i, l]
        ref = np.setdiff1d(np.arange(P), mem)
        jl[l] = _pad(idn.joint_select(pop, i, ref, KMAX, len(mem) - 1))
        vl[l] = _pad([stmt_index(pop, s) for s in top_k(v2.candidates(first, i, mem), KMAX)])
    ref = np.delete(np.arange(P), i)
    jn = _pad(idn.joint_select(pop, i, ref, KMAX, P - 1))
    vn = _pad([stmt_index(pop, s) for s in top_k(v2.candidates(first, i, [i]), KMAX)])
    return jl, vl, jn, vn


def draw_lineups(P: int, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M = np.zeros((P, L, 5), np.int64)
    for i in range(P):
        others = np.delete(np.arange(P), i)
        for l in range(L):
            M[i, l] = np.r_[i, rng.choice(others, 4, replace=False)]
    return M


def count_scores(T2: np.ndarray, ch: np.ndarray, k: int) -> np.ndarray:
    """Counts of the first k chosen statements (-1 = none) true of every second half."""
    c = ch[:k][ch[:k] >= 0]
    return T2[:, c].sum(1) if len(c) else np.zeros(T2.shape[0], np.int64)


def lineup_acc(T2, chosen, M, k):
    """chosen: P x L x KMAX; returns per-target accuracy, and the P x L x P count array for k."""
    P = M.shape[0]
    cred = np.zeros((P, L))
    scores = np.zeros((P, L, T2.shape[0]), np.int16)
    for i in range(P):
        for l in range(L):
            sc = count_scores(T2, chosen[i, l], k)
            scores[i, l] = sc
            cred[i, l] = idn.rank_credit(sc, i, M[i, l])
    return cred, scores


def one_of_n(S: np.ndarray, higher: bool = True) -> dict:
    """S[i, j]: score of person j's second half for target i's first half."""
    P = S.shape[0]
    allp = np.arange(P)
    t1 = np.array([idn.rank_credit(S[i], i, allp, 1, higher) for i in range(P)])
    t5 = np.array([idn.rank_credit(S[i], i, allp, 5, higher) for i in range(P)])
    # mean rank with random tie-breaks, and the rank bits log2 N - E log2(rank)
    lb = []
    for i in range(P):
        s = S[i] if higher else -S[i]
        a = int((s > s[i]).sum())
        m = int((s == s[i]).sum())
        lb.append(np.mean(np.log2(np.arange(a + 1, a + m + 1))))
    lb = np.array(lb)
    lo1, hi1 = lu.bootstrap_ci(t1)
    return {"N": P, "top1": float(t1.mean()), "top1_ci95": [lo1, hi1], "top5": float(t5.mean()),
            "chance_top1": 1 / P, "rank_bits": float(np.log2(P) - lb.mean()), "_t1": t1}


def lineup_of(D: np.ndarray, M: np.ndarray, higher: bool = False) -> np.ndarray:
    """Tie-shared lineup credit from a score matrix D[target, candidate]."""
    return np.array([[idn.rank_credit(D[i], i, M[i, l], 1, higher) for l in range(M.shape[1])] for i in range(M.shape[0])])


def fingerprints(H1, H2, H1a, H1b) -> dict[str, tuple[np.ndarray, bool]]:
    """Score matrices (target first half x candidate second half) for every fingerprint matcher."""
    P = len(H1)
    p1, p2 = v2.percentiles(H1, H1), v2.percentiles(H1, H2)
    mad = np.full((P, P), np.inf)
    for i in range(P):
        diff = np.abs(p2 - p1[i])
        n = np.isfinite(diff).sum(1)
        mad[i] = np.where(n > 0, np.nansum(diff, 1) / np.maximum(n, 1), np.inf)
    Z1, Z2 = idn.fingerprint_z(H1, H1), idn.fingerprint_z(H1, H2)
    Za, Zb = idn.normal_scores(H1a, H1a), idn.normal_scores(H1b, H1b)
    euc = idn.mahalanobis_dist(Z1, Z2, np.eye(Z1.shape[1]))
    W = idn.within_cov(Za, Zb, idn.SHRINK)
    diag = idn.mahalanobis_dist(Z1, Z2, np.diag(1 / np.diag(W)))
    mah = idn.mahalanobis_dist(Z1, Z2, np.linalg.inv(W))
    return {"percentile_mad": (mad, False), "z_euclid": (euc, False), "z_diag_noise": (diag, False),
            "z_mahalanobis": (mah, False)}


def item_matrix(a1: list[dict], a2: list[dict]) -> np.ndarray:
    """Hellinger affinity of artist play shares: sum_a sqrt(p1_a p2_a) (target first half x candidate
    second half). Artists only; this is the item-based baseline, not something a card would show."""
    import scipy.sparse as sp
    keys = {a: j for j, a in enumerate(sorted({x for L_ in (a1, a2) for d in L_ if d for x in d}))}

    def mat(Ls):
        r, c, v = [], [], []
        for i, d in enumerate(Ls):
            tot = sum(d.values()) if d else 0
            for a, n in (d or {}).items():
                r.append(i), c.append(keys[a]), v.append(np.sqrt(n / tot))
        return sp.csr_matrix((v, (r, c)), shape=(len(Ls), len(keys)))
    return (mat(a1) @ mat(a2).T).toarray()


def bits_curve(pop: idn.Population, chosen: np.ndarray) -> list[dict]:
    """Per k: mean bits of the conjunction on the others' first halves (KT), the independence sum of
    single-statement bits, share of targets whose second half keeps all k, mean bits that survive."""
    P = pop.P
    out = []
    for k in range(1, KMAX + 1):
        b1, bsum, kept, b2, have = [], [], [], [], []
        for i in range(P):
            ch = chosen[i, :k]
            ch = list(ch[ch >= 0])
            have.append(len(ch) == k)
            ref = np.delete(np.arange(P), i)
            x1, x2, kp = idn.set_bits(pop, i, ref, ch, len(ch))
            b1.append(x1)
            b2.append(x2)
            kept.append(kp)
            bsum.append(float(sum(idn.kt_bits(pop.T1[ref, s].sum(), len(ref)) for s in ch)))
        out.append({"k": k, "bits_first_half": float(np.mean(b1)), "bits_independent_sum": float(np.mean(bsum)),
                    "kept_on_second_half": float(np.mean(kept)), "bits_surviving": float(np.mean(b2)),
                    "share_with_k": float(np.mean(have)), "bits_needed": float(np.log2(P))})
    return out


def evaluate(d: dict, arts: dict, tag: str) -> tuple[dict, dict]:
    P = len(d["ids"])
    t0 = time.time()
    full, first, second = r2.tables(d)
    M = draw_lineups(P)
    with mp.Pool(WORKERS, initializer=_init,
                 initargs=(d["h1"], d["h2"], d["h1a"], d["h1b"], first, M)) as pool:
        res = pool.map(_target, range(P), chunksize=2)
    pop = idn.Population(d["h1"], d["h2"], d["h1a"], d["h1b"], FAMS)
    JL = np.stack([r[0] for r in res]); VL = np.stack([r[1] for r in res])
    JN = np.stack([r[2] for r in res]); VN = np.stack([r[3] for r in res])
    print(f"{tag}: selections for {P} people in {time.time() - t0:.0f} s", file=sys.stderr)
    out: dict = {"set": tag, "people": P, "families": len(FAMS), "statements": len(pop.cs),
                 "lineups_per_target": L, "seed": SEED, "params": {
                     "LAMBDA": idn.LAMBDA, "KAPPA": idn.KAPPA, "PER_GROUP": idn.PER_GROUP, "POOL": idn.POOL,
                     "SHRINK": idn.SHRINK, "R_FLOOR": idn.R_FLOOR, "R_CEIL": idn.R_CEIL}}
    curves = {"lineup": [], "one_of_n": []}
    # lineups, k = 1..7, joint and independent
    out["lineup"] = {}
    for name, ch in (("joint", JL), ("independent", VL)):
        for k in range(1, KMAX + 1):
            cred, scores = lineup_acc(pop.T2, ch, M, k)
            lo, hi = lu.bootstrap_ci(cred.mean(1))
            row = {"accuracy": float(cred.mean()), "ci95": [lo, hi],
                   "share_lineups_short": float(((ch[:, :, :k] >= 0).sum(2) < k).mean()),
                   "target_alone_on_top": float((cred == 1).mean())}
            if k == 3:
                run = lu.LineupRun(k, name, cred, scores, M, (ch[:, :, :k] >= 0).sum(2), {})
                null, pv = lu.permutation_null(run, 1000)
                row.update({"perm_null_mean": float(null.mean()), "perm_p": pv, "per_target": cred.mean(1)})
                if name == "joint":
                    jt3 = cred.mean(1)
                else:
                    vt3 = cred.mean(1)
            out["lineup"][f"{name}_k{k}"] = {kk: vv for kk, vv in row.items() if kk != "per_target"}
            curves["lineup"].append({"selection": name, "k": k, "accuracy": row["accuracy"],
                                     "ci_lo": lo, "ci_hi": hi})
    diff = jt3 - vt3
    out["lineup"]["joint_minus_independent_k3"] = {"mean": float(diff.mean()), "ci95": list(lu.bootstrap_ci(diff))}
    # 1-of-N with statements
    out["one_of_n"] = {}
    t1_3: dict = {}
    for name, ch in (("joint", JN), ("independent", VN)):
        for k in range(1, KMAX + 1):
            S = np.stack([count_scores(pop.T2, ch[i], k) for i in range(P)]).astype(float)
            r = one_of_n(S)
            if k == 3:
                t1_3[name] = r["_t1"]
            out["one_of_n"][f"{name}_k{k}"] = {kk: vv for kk, vv in r.items() if not kk.startswith("_")}
            curves["one_of_n"].append({"matcher": f"{name} sentences", "k": k, "top1": r["top1"], "top5": r["top5"],
                                       "rank_bits": r["rank_bits"]})
    dd = t1_3["joint"] - t1_3["independent"]
    out["one_of_n"]["joint_minus_independent_k3_top1"] = {"mean": float(dd.mean()), "ci95": list(lu.bootstrap_ci(dd))}
    # fingerprints: 1-of-N and lineups
    out["fingerprint"] = {}
    fps = fingerprints(d["h1"], d["h2"], d["h1a"], d["h1b"])
    fps["items_hellinger"] = (item_matrix(arts["h1"], arts["h2"]), True)
    for name, (S, higher) in fps.items():
        r = one_of_n(S, higher)
        lc = lineup_of(S, M, higher)
        out["fingerprint"][name] = {**{kk: vv for kk, vv in r.items() if not kk.startswith("_")},
                                    "lineup": float(lc.mean()), "lineup_ci95": list(lu.bootstrap_ci(lc.mean(1)))}
        curves["one_of_n"].append({"matcher": name, "k": "all", "top1": r["top1"], "top5": r["top5"],
                                   "rank_bits": r["rank_bits"]})
    t_m = one_of_n(fps["z_mahalanobis"][0], False)["_t1"]
    t_p = one_of_n(fps["percentile_mad"][0], False)["_t1"]
    out["fingerprint"]["mahalanobis_minus_percentile_top1"] = {
        "mean": float((t_m - t_p).mean()), "ci95": list(lu.bootstrap_ci(t_m - t_p))}
    # bits
    curves["bits"] = [{"selection": n, **row} for n, ch in (("joint", JN), ("independent", VN)) for row in bits_curve(pop, ch)]
    single = []
    for i in range(P):
        ref = np.delete(np.arange(P), i)
        c = pop.T1[ref].sum(0)
        tr = np.nonzero(pop.T1[i])[0]
        if len(tr):
            single.append(float(idn.kt_bits(c[tr], len(ref)).max()))
    out["bits"] = {"needed": float(np.log2(P)), "best_single_statement_mean": float(np.mean(single)),
                   "curve": curves["bits"]}
    (Path(tempfile.gettempdir()) / f"aas_identity_{tag}_partial.json").write_text(json.dumps(out, indent=1))
    out["decay"] = decay(d, arts)
    curves["decay"] = out["decay"]["curve"]
    print(f"{tag}: done in {time.time() - t0:.0f} s", file=sys.stderr)
    return out, curves


def _init_decay(A_, B, Aa, Ab, M):
    G["pop"] = idn.Population(A_, B, Aa, Ab, FAMS)
    G["M"] = M


def _decay_target(i: int):
    pop, M = G["pop"], G["M"]
    n = pop.P
    jl = np.zeros((M.shape[1], KMAX), np.int64)
    for l in range(M.shape[1]):
        jl[l] = _pad(idn.joint_select(pop, i, np.setdiff1d(np.arange(n), M[i, l]), 3, 4))
    return jl, _pad(idn.joint_select(pop, i, np.delete(np.arange(n), i), 3, n - 1))


def decay(d: dict, arts: dict) -> dict:
    """Window A (26 weeks) against window B (the last 26 training weeks), A ending 0, 1, 3, 6 or 12
    months before B starts, on the people whose history reaches back far enough for every gap."""
    dec = d["decay"]                                      # P x gaps x 4 x F
    ok = np.isfinite(dec[:, -1, 0]).any(1) & np.array([a is not None for a in arts[f"A{len(GAPS) - 1}"]])
    idx = np.nonzero(ok)[0]
    n = len(idx)
    curve = []
    M = draw_lineups(n, SEED + 1)
    for g, gap in enumerate(GAPS):
        A_, Aa, Ab, B = (dec[idx, g, j] for j in range(4))
        pop = idn.Population(A_, B, Aa, Ab, FAMS)
        with mp.Pool(WORKERS, initializer=_init_decay, initargs=(A_, B, Aa, Ab, M)) as pool:
            res = pool.map(_decay_target, range(n), chunksize=2)
        jl = np.stack([r[0] for r in res])
        jn = np.stack([r[1] for r in res])
        cred, _ = lineup_acc(pop.T2, jl, M, 3)
        s3 = one_of_n(np.stack([count_scores(pop.T2, jn[i], 3) for i in range(n)]).astype(float))
        fp = fingerprints(A_, B, Aa, Ab)["z_mahalanobis"][0]
        f1 = one_of_n(fp, False)
        it = one_of_n(item_matrix([arts[f"A{g}"][i] for i in idx], [arts["B"][i] for i in idx]), True)
        curve.append({"gap_months": gap, "people": n, "sentences3_lineup": float(cred.mean()),
                      "sentences3_top1": s3["top1"], "fingerprint_top1": f1["top1"],
                      "fingerprint_lineup": float(lineup_of(fp, M).mean()), "items_top1": it["top1"]})
        print(f"decay gap {gap}: {curve[-1]}", file=sys.stderr)
    return {"window_weeks": 26, "gaps_months": list(GAPS), "people": n, "curve": curve}


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    lines = [",".join(keys)] + [",".join(f"{r.get(k, '')}" if not isinstance(r.get(k), float) else f"{r[k]:.4f}"
                                         for k in keys) for r in rows]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "explore"
    if cmd == "measure":
        d = values(sys.argv[2])
        print(len(d["ids"]), Counter(x.split("|", 1)[1] for x in d["excluded"]))
        return
    out_file = DOCS / "identity_results.json"
    allres = json.loads(out_file.read_text()) if out_file.exists() else {}
    if cmd == "explore":
        d, a = pooled(["old", "fresh"])
        tag = "exploration"
    elif cmd == "confirm":
        if "confirmation" in allres:
            raise SystemExit("the confirmation run has already been made; it is run once")
        d, a = pooled(["third"])
        tag = "confirmation"
    else:
        raise SystemExit(f"unknown command {cmd}")
    res, curves = evaluate(d, a, tag)
    if tag == "confirmation":
        res["verdict"] = verdict(res)
    allres[tag] = res
    out_file.write_text(json.dumps(allres, indent=1) + "\n")
    write_csv(DOCS / f"identity_{tag}_lineup.csv", curves["lineup"])
    write_csv(DOCS / f"identity_{tag}_one_of_n.csv", curves["one_of_n"])
    write_csv(DOCS / f"identity_{tag}_bits.csv", curves["bits"])
    write_csv(DOCS / f"identity_{tag}_decay.csv", curves["decay"])
    print(json.dumps({k: v for k, v in res.items() if k != "bits"}, indent=1))


def verdict(res: dict) -> dict:
    """The bars pre-registered in docs/identity.md before the third set was measured."""
    j3 = res["lineup"]["joint_k3"]
    fp = res["fingerprint"]
    on = res["one_of_n"]
    return {
        "primary": {"rule": "joint top-3 lineup accuracy >= 0.75, 95% CI lower bound > 0.675, permutation p < 0.01",
                    "accuracy": j3["accuracy"], "ci95": j3["ci95"], "perm_p": j3["perm_p"],
                    "passed": bool(j3["accuracy"] >= 0.75 and j3["ci95"][0] > 0.675 and j3["perm_p"] < 0.01)},
        "s1_fingerprint_top1": {"rule": "Mahalanobis fingerprint 1-of-N top-1 >= 0.40 and its paired gain over the "
                                        "percentile fingerprint has CI lower bound > 0",
                                "passed": bool(fp["z_mahalanobis"]["top1"] >= 0.40
                                               and fp["mahalanobis_minus_percentile_top1"]["ci95"][0] > 0)},
        "s2_sentences_top1": {"rule": "joint 3-sentence 1-of-N top-1 >= 0.10 and its paired gain over the round 2 "
                                      "selection has CI lower bound > 0",
                              "passed": bool(on["joint_k3"]["top1"] >= 0.10
                                             and on["joint_minus_independent_k3_top1"]["ci95"][0] > 0)},
        "s3_joint_beats_independent_lineup": {"rule": "paired joint minus round 2 top-3 lineup accuracy, CI lower "
                                                      "bound > 0",
                                              "passed": bool(res["lineup"]["joint_minus_independent_k3"]["ci95"][0] > 0)},
    }


if __name__ == "__main__":
    main()
