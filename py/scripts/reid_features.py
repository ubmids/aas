"""How identifying is a feature record? Two measurements on a real population of writers.

    cd py && uv run python scripts/reid_features.py [--n 8675] [--out docs/features_reid.json]

Population: the 8,675 PersonalityCafe forum users the MBTI reader was trained on (about 50 posts
each, data/raw/mbti), the only local corpus with many posts per real person. Their posts carry no
timestamps, so the activity histograms are SIMULATED (each person gets a habitual peak hour and a
weekday pattern; items are drawn from it). Writing-style and word-rate features are computed with
the real aas.features code on the real posts.

1. Singling out. Every field of the record, rounded exactly as the record rounds it, is treated
   as a label. In a population of N, how many fields (taken in a random order) does it take
   before a person's combination is unique? Reported per block.
2. Linkage. Each person's posts are split in two (alternate posts). Half A plays the record the
   study holds; half B plays the attacker's scrape of the same person's public posts, from which
   the attacker computes the same features. For each A, rank all N B's by distance; report how
   often the true person is the nearest (top-1) or in the nearest 10. Distances: standardised
   Euclidean on style; Burrows-style cosine on z-scored log rates for the 300 most common list words;
   L1 on hour+weekday shares; and the sum of the per-block z-scored distances.
"""

from __future__ import annotations

import argparse
import json
import warnings
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aas import features as F  # noqa: E402


def halves(posts: list[str]) -> tuple[list[str], list[str]]:
    posts = [p for p in posts if p.strip()]
    return posts[0::2], posts[1::2]


def sim_times(rng, n_people: int, n_items: int):
    """Hour and weekday shares (rounded like the record) for two independent samples per person from
    one habit. Habits are meant to look like a US/UK sample, not to be maximally distinct: a time zone
    (UK 40%, US Eastern 30%, Central 15%, Pacific 15%), a local peak hour around the evening
    (normal, mean 20:00, sd 3 h), a regularity (von Mises kappa 0.2 to 1.5) and a mild weekday tilt."""
    tz = rng.choice([0, -5, -6, -8], n_people, p=[0.4, 0.3, 0.15, 0.15])
    peak = (rng.normal(20, 3, n_people) - tz) % 24          # in UTC
    kappa = rng.uniform(0.2, 1.5, n_people)
    wk = rng.dirichlet(np.full(7, 20.0), n_people)
    out = []
    for _ in range(2):
        H = np.zeros((n_people, 24))
        W = np.zeros((n_people, 7))
        for i in range(n_people):
            ang = rng.vonmises(2 * np.pi * peak[i] / 24 - np.pi, kappa[i], n_items)
            hrs = (np.floor((ang + np.pi) / (2 * np.pi) * 24).astype(int)) % 24
            H[i] = np.bincount(hrs, minlength=24) / n_items
            W[i] = np.bincount(rng.choice(7, n_items, p=wk[i]), minlength=7) / n_items
        out.append(np.hstack([H, W]))
    return out


def as_record(T: np.ndarray) -> np.ndarray:
    """The record's own rounding: shares to 0.01."""
    return np.round(T, 2)


def coarse(T: np.ndarray) -> np.ndarray:
    """A candidate mitigation: six 4-hour blocks and weekday-vs-weekend, rounded to 0.05."""
    H, W = T[:, :24], T[:, 24:]
    blocks = H.reshape(len(H), 6, 4).sum(2)
    wkend = W[:, 5:].sum(1, keepdims=True)
    return np.round(np.hstack([blocks, wkend]) / 0.05) * 0.05


def feats(texts: list[str]) -> tuple[np.ndarray, np.ndarray | None]:
    st = F.style(texts)
    s = np.array([np.nan if st[k] is None else st[k] for k in F.STYLE_KEYS], float)
    wr, _ = F.word_rates(texts)
    w = None if wr is None else np.array(list(wr["per_1k"].values()), float)
    return s, w


def zscore(X: np.ndarray, ref: np.ndarray) -> np.ndarray:
    mu, sd = np.nanmean(ref, 0), np.nanstd(ref, 0)
    sd[sd == 0] = 1
    Z = (X - mu) / sd
    return np.nan_to_num(Z)


def ranks(D: np.ndarray) -> np.ndarray:
    """D[i, j] = distance from record i to scrape j; rank of the true j=i (0 = nearest)."""
    true = np.diag(D)[:, None]
    return (D < true).sum(1)


def link(D: np.ndarray) -> dict:
    r = ranks(D)
    return {"top1": round(float((r == 0).mean()), 3), "top10": round(float((r < 10).mean()), 3),
            "median_rank": float(np.median(r))}


def dist_euclid(A, B):
    sq = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
    return np.sqrt(np.maximum(sq, 0))


def dist_cos(A, B):
    An = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    Bn = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    return 1 - An @ Bn.T


def dist_l1(A, B, chunk: int = 200):
    return np.vstack([np.abs(A[i:i + chunk, None, :] - B[None, :, :]).sum(-1) for i in range(0, len(A), chunk)])


def fields_to_unique(M: np.ndarray, rng, reps: int = 5) -> dict:
    """Median number of fields (random order) until each row's combination is unique in M."""
    n, k = M.shape
    need = []
    for _ in range(reps):
        order = rng.permutation(k)
        got = np.full(n, k + 1)
        keys = [""] * n
        for j, c in enumerate(order, 1):
            keys = [a + "|" + repr(b) for a, b in zip(keys, M[:, c])]
            cnt: dict[str, int] = {}
            for x in keys:
                cnt[x] = cnt.get(x, 0) + 1
            uniq = np.array([cnt[x] == 1 for x in keys])
            got[(got == k + 1) & uniq] = j
            if (got <= k).all():
                break
        need.append(got)
    g = np.concatenate(need)
    ok = g[g <= k]
    return {"fields": k, "unique_share_all_fields": round(float((g <= k).mean()), 3),
            "median_fields_needed": float(np.median(ok)) if len(ok) else None,
            "p90_fields_needed": float(np.percentile(ok, 90)) if len(ok) else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8675)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "docs" / "features_reid.json")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    warnings.filterwarnings("ignore", category=RuntimeWarning)   # all-NaN columns (ttr_1000 needs 1,000 words)

    from aas.slices.text import load_mbti

    t = time.time()
    d = load_mbti()
    idx = rng.permutation(len(d))[: a.n]
    SA, SB, WA, WB, words = [], [], [], [], []
    for i in idx:
        A, B = halves(list(d["posts"].iloc[i]))
        sa, wa = feats(A)
        sb, wb = feats(B)
        if wa is None or wb is None:      # under 500 words in a half: no word-rate block
            continue
        SA.append(sa), SB.append(sb), WA.append(wa), WB.append(wb)
        words.append(sum(len(x.split()) for x in A))
    SA, SB, WA, WB = map(np.array, (SA, SB, WA, WB))
    n = len(SA)
    TA0, TB0 = sim_times(rng, n, 150)
    TA, TB = as_record(TA0), as_record(TB0)
    CA, CB = coarse(TA0), coarse(TB0)
    print(f"{n} people with >= 500 words in each half (median {int(np.median(words))} words per half); "
          f"features in {time.time() - t:.0f}s", file=sys.stderr)

    res: dict = {"population": "PersonalityCafe MBTI forum users (data/raw/mbti), real posts; "
                               "activity histograms simulated", "n_people": n,
                 "median_words_per_half": int(np.median(words)), "seed": a.seed}

    # 1. singling out, on the record's own rounding (half A)
    res["singling_out"] = {}
    for N in (300, 1000, n):
        sel = rng.permutation(n)[:N]
        res["singling_out"][str(N)] = {
            "style_15": fields_to_unique(SA[sel], rng),
            "hours_weekdays_31_simulated": fields_to_unique(TA[sel], rng),
            "coarse_rhythm_7_simulated": fields_to_unique(CA[sel], rng),
        }

    # 2. linkage, record half A vs attacker's half B
    top = 300
    LA, LB = np.log1p(WA[:, :top]), np.log1p(WB[:, :top])
    res["linkage"] = {}
    for N in (300, 1000, min(n, 3000)):
        sel = rng.permutation(n)[:N]
        ref = np.vstack([SA[sel], SB[sel]])
        Ds = dist_euclid(zscore(SA[sel], ref), zscore(SB[sel], ref))
        lref = np.vstack([LA[sel], LB[sel]])
        Dw = dist_cos(zscore(LA[sel], lref), zscore(LB[sel], lref))
        Dt = dist_l1(TA[sel], TB[sel])
        z = lambda D: (D - D.mean()) / D.std()  # noqa: E731
        res["linkage"][str(N)] = {
            "chance_top1": round(1 / N, 4),
            "style_only": link(Ds),
            "word_rates_only": link(Dw),
            "hours_weekdays_only_simulated": link(Dt),
            "coarse_rhythm_only_simulated": link(dist_l1(CA[sel], CB[sel])),
            "style_plus_hours": link(z(Ds) + z(Dt)),
            "everything": link(z(Ds) + z(Dw) + z(Dt)),
        }
        print(N, json.dumps(res["linkage"][str(N)]), file=sys.stderr)
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
