"""The lineup test and the stability check (pre-registered in py/docs/statements.md).

Lineup. For each target, statements are made from the FIRST half of their history against the
first halves of everyone except the target and that lineup's four decoys. A blind matcher then sees
the SECOND halves of the five people and picks the one on which the most statements are true; an
m-way tie that includes the target scores 1/m (the expected value of a random tie-break). Each
target gets L lineups with fresh decoys; their mean is the target's accuracy. Chance is 1/5.

Permutation null: the second halves are shuffled across people (so "the target's second half" is a
random other person's), statements and decoys unchanged, and accuracy recomputed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .engine import Table, top_k

SIZE = 5


@dataclass
class LineupRun:
    k: int
    mode: str
    credit: np.ndarray          # targets x L, the credit for each lineup
    scores: np.ndarray          # targets x L x people: statements true on each person's second half
    members: np.ndarray         # targets x L x 5 (target first)
    n_statements: np.ndarray    # targets x L: how many statements were available (<= k)
    statements: dict            # (target, lineup) -> list[Statement], for the first lineup only

    @property
    def per_target(self) -> np.ndarray:
        return self.credit.mean(1)

    @property
    def accuracy(self) -> float:
        return float(self.per_target.mean())


def credit_of(scores_row: np.ndarray, members: np.ndarray, perm: np.ndarray | None = None) -> float:
    s = scores_row[members if perm is None else perm[members]]
    top = s.max()
    tied = (s == top)
    return float(tied[0] / tied.sum())


def run(first: Table, second: Table, k: int, L: int, seed: int, mode: str = "only",
        people: np.ndarray | None = None, cands=None) -> LineupRun:
    """`cands(table, i, exclude, mode)` makes the statements; round 1's Table.candidates by default."""
    cands = cands or (lambda tab, i, ex, m: tab.candidates(i, ex, m))
    rng = np.random.default_rng(seed)
    P = first.P
    people = np.arange(P) if people is None else np.asarray(people)
    credit = np.zeros((len(people), L))
    scores = np.zeros((len(people), L, P), dtype=np.int16)
    members = np.zeros((len(people), L, SIZE), dtype=np.int64)
    nst = np.zeros((len(people), L), dtype=np.int64)
    kept: dict = {}
    for a, i in enumerate(people):
        others = people[people != i]
        for l in range(L):
            dec = rng.choice(others, SIZE - 1, replace=False)
            mem = np.r_[i, dec]
            sts = top_k(cands(first, int(i), mem, mode), k)
            members[a, l] = mem
            nst[a, l] = len(sts)
            if l == 0:
                kept[(int(i), l)] = sts
            sc = np.zeros(P, dtype=np.int16)
            for s in sts:
                sc += second.truth(s).astype(np.int16)
            scores[a, l] = sc
            credit[a, l] = credit_of(sc, mem)
    return LineupRun(k, mode, credit, scores, members, nst, kept)


def bootstrap_ci(per_target: np.ndarray, B: int = 4000, seed: int = 11, level: float = 0.95) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(per_target)
    means = per_target[rng.integers(0, n, (B, n))].mean(1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def permutation_null(r: LineupRun, B: int = 1000, seed: int = 13) -> tuple[np.ndarray, float]:
    """Accuracy with the second halves shuffled across people, B times; returns (null accuracies, p)."""
    rng = np.random.default_rng(seed)
    T, L, P = r.scores.shape
    obs = r.accuracy
    null = np.empty(B)
    rows = r.scores.reshape(T * L, P)
    mem = r.members.reshape(T * L, SIZE)
    for b in range(B):
        perm = rng.permutation(P)
        s = rows[np.arange(T * L)[:, None], perm[mem]]
        top = s.max(1, keepdims=True)
        tied = s == top
        null[b] = (tied[:, 0] / tied.sum(1)).mean()
    p = (1 + int((null >= obs).sum())) / (1 + B)
    return null, p


@dataclass
class Stability:
    made: int          # only-you statements made from first halves
    held: int          # still true on the second half
    uncheckable: int   # the family could not be measured on the second half
    still_rare: int    # true on the second half AND true of <= 10% of others' second halves
    per_family: dict   # family -> (made, held)
    top3_made: int
    top3_held: int


def stability(first: Table, second: Table, people: np.ndarray | None = None) -> Stability:
    P = first.P
    people = np.arange(P) if people is None else np.asarray(people)
    made = held = unc = rare = t3m = t3h = 0
    fam: dict[str, list[int]] = {}
    for i in people:
        sts = first.candidates(int(i), [int(i)], "only")
        top = {id(s) for s in top_k(sts, 3)}
        for s in sts:
            kk = s.family.key
            fi = first.families.index(s.family)
            v2 = second.V[i, fi]
            ok = bool(second.truth(s, int(i)))
            made += 1
            fam.setdefault(kk, [0, 0])
            fam[kk][0] += 1
            if not np.isfinite(v2):
                unc += 1
            if ok:
                held += 1
                fam[kk][1] += 1
                others = second.truth(s, people[people != i])
                if others.mean() <= 0.10:
                    rare += 1
            if id(s) in top:
                t3m += 1
                t3h += int(ok)
    return Stability(made, held, unc, rare, {k: tuple(v) for k, v in fam.items()}, t3m, t3h)
